# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel mHC seam collectives for DSv4 on ROCm TP8 (m-seq-parallel, variant C).

Plan (dossier 11.u1-plan-seq-parallel, variant C from iterations 26-28):
  S4a (this module): the collectives, not yet called by the model.
      reduce_scatter(): 8-rank INT4 reduce-scatter (AITER qr_reduce_scatter, the QuickReduce
          two-shot's first phase as its own launch), rank r gets rows [r*M/8, (r+1)*M/8) in bf16.
      gather_fp8(): fp8 e4m3 1x128-scaled all-gather of a row-sharded bf16 tensor through AITER's
          custom all-gather, dequantized to bf16 [M, H] on every rank.
  S4b: the ROCm DSv4 decoder layer runs its mHC seams on M/8 rows between these two calls at
      mixed/prefill steps (behind a switch, off by default).
  S5: switch on, accuracy gate and full ladder.

Numerics (iterations 27/28): the gathered layer input is 21.1 dB vs the exact fp32 reduce,
against 18.5 dB for the served INT4 all-reduce path. Not bitwise equal to the served path.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.config.attention import active_attention_config
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_WS = 8
_GROUP = 128
_GB = 8  # 128-element groups per program


@triton.jit
def _fp8_1x128_quant_pack(x_ptr, o_ptr, H: tl.constexpr, ROWB: tl.constexpr, GB: tl.constexpr):
    # packed row: [H fp8 e4m3 | H/128 fp32 scales]; bitwise equal to torch's
    # s = amax / 448 (scalar div = mul by fp32 reciprocal), q = (x / s).to(fp8)
    row = tl.program_id(0)
    gb = tl.program_id(1)
    g = gb * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    x = tl.load(x_ptr + row.to(tl.int64) * H + g[:, None] * 128 + c[None, :]).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12) * (1.0 / 448.0)
    q = tl.div_rn(x, s[:, None]).to(tl.float8e4nv)
    base = o_ptr + row.to(tl.int64) * ROWB
    tl.store(base.to(tl.pointer_type(tl.float8e4nv), bitcast=True) + g[:, None] * 128 + c[None, :], q)
    tl.store((base + H).to(tl.pointer_type(tl.float32), bitcast=True) + g, s)


@triton.jit
def _fp8_1x128_dequant(p_ptr, o_ptr, H: tl.constexpr, ROWB: tl.constexpr, GB: tl.constexpr):
    row = tl.program_id(0)
    gb = tl.program_id(1)
    g = gb * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    base = p_ptr + row.to(tl.int64) * ROWB
    q = tl.load(base.to(tl.pointer_type(tl.float8e4nv), bitcast=True) + g[:, None] * 128 + c[None, :]).to(tl.float32)
    s = tl.load((base + H).to(tl.pointer_type(tl.float32), bitcast=True) + g)
    tl.store(o_ptr + row.to(tl.int64) * H + g[:, None] * 128 + c[None, :], (q * s[:, None]).to(tl.bfloat16))


@triton.jit(do_not_specialize=["M"])
def _fp8_1x128_dequant_ts(p_ptr, o_ptr, st_ptr, M, H: tl.constexpr, ROWB: tl.constexpr,
                          GB: tl.constexpr):
    # _fp8_1x128_dequant plus the row's fp32 scales into st[g * M + row] for rows < M: the
    # column-major [H/128, M] layout AITER per_1x128 quant writes with transpose_scale=True.
    row = tl.program_id(0)
    gb = tl.program_id(1)
    g = gb * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    base = p_ptr + row.to(tl.int64) * ROWB
    q = tl.load(base.to(tl.pointer_type(tl.float8e4nv), bitcast=True) + g[:, None] * 128 + c[None, :]).to(tl.float32)
    s = tl.load((base + H).to(tl.pointer_type(tl.float32), bitcast=True) + g)
    tl.store(o_ptr + row.to(tl.int64) * H + g[:, None] * 128 + c[None, :], (q * s[:, None]).to(tl.bfloat16))
    tl.store(st_ptr + g.to(tl.int64) * M + row, s, mask=(g * 0 + row) < M)


def packed_row_bytes(hidden: int) -> int:
    return hidden + (hidden // _GROUP) * 4


def fp8_1x128_quant_pack(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 [m, H] -> uint8 [m, H + H/128*4] (fp8 e4m3 codes, then fp32 scales).

    `out` may be wider than packed_row_bytes(H): bytes past it are left for the caller (a per-row
    tail that travels in the same gather).
    """
    m, h = x.shape
    assert x.is_contiguous() and h % (_GROUP * _GB) == 0
    if out is None:
        out = torch.empty(m, packed_row_bytes(h), dtype=torch.uint8, device=x.device)
    rowb = out.shape[1]
    assert out.is_contiguous() and rowb >= packed_row_bytes(h)
    if m:
        _fp8_1x128_quant_pack[(m, h // _GROUP // _GB)](x, out, h, rowb, _GB, num_warps=4)
    return out


def fp8_1x128_dequant(p: torch.Tensor, hidden: int, out: torch.Tensor | None = None) -> torch.Tensor:
    m, rowb = p.shape
    assert rowb >= packed_row_bytes(hidden) and p.is_contiguous()
    if out is None:
        out = torch.empty(m, hidden, dtype=torch.bfloat16, device=p.device)
    if m:
        _fp8_1x128_dequant[(m, hidden // _GROUP // _GB)](p, out, hidden, rowb, _GB, num_warps=4)
    return out


# --- m-sp-dequant-skip S2: hand the gathered fp8 codes to the fp8 GEMM consumers ----------------
#
# Part of dsv4_seq_parallel (on whenever it is on). The SP layer-input gather carries fp8 e4m3
# codes with fp32 1x128 scales; the served path dequantizes them to bf16 and the attention's
# fused_wqa_wkv and the shared expert's gate_up then re-quantize per_1x128 (AITER hip quant,
# transpose_scale=True) for the bpreshuffle fp8 GEMM. That requant reproduces the gathered codes
# and scales bitwise (ledger m-sp-dequant-skip-requant-idempotent), so with the switch on those two
# GEMMs take the codes (a strided view of the gathered rows, no copy) and the scales (written
# transposed by the dequant) directly: exact, same GEMM. The bf16 rows stay for every other
# consumer (routed MoE per_1x32 quant, compressor, indexer, router).
# Debug-only first-step equivalence self-checks (the one-time comparisons against the unfused
# path). Off by default: a check that fails on one rank raises on that rank only, and the other
# ranks then wait in the next collective with no timeout. The same comparisons run as tests.
def sp_self_check_enabled() -> bool:
    return active_attention_config().dsv4_sp_self_check


SP_FP8_HANDOFF_LABEL = "sp_fp8_handoff"
_SP_FP8_HANDOFF = True  # on with dsv4_seq_parallel; no switch of its own


def _form_key(x: torch.Tensor) -> tuple:
    ver = None if x.is_inference() else x._version
    return (x.data_ptr(), tuple(x.shape), x.stride(), ver)


class _Fp8Form:
    """The fp8 form of the current sublayer's gathered bf16 input (one at a time).

    Matched by identity of the bf16 tensor (storage pointer, shape, strides; plus the version
    counter where autograd tracks one, so an in-place write invalidates it in eager evals). Under
    inference mode (the server) tensors carry no version counter: there the guard is the scope
    (registered by the sublayer's gather, cleared after the sublayer, and no in-place write to the
    gathered rows in between on the SP path) and the per-rank bitwise self-check of each consumer
    (sp_fp8_handoff_check). Holds references to the bf16 rows and the gathered buffer so neither
    storage is reused while registered.
    """

    x: torch.Tensor | None = None
    key: tuple | None = None
    codes: torch.Tensor | None = None
    scales_t: torch.Tensor | None = None
    checked: set = set()
    logged: bool = False
    used: bool = False

    @classmethod
    def set(cls, x: torch.Tensor, codes: torch.Tensor, scales_t: torch.Tensor) -> None:
        cls.x = x
        cls.key = _form_key(x)
        cls.codes = codes
        cls.scales_t = scales_t
        cls.used = False

    @classmethod
    def clear(cls) -> None:
        cls.x = cls.key = cls.codes = cls.scales_t = None


def sp_fp8_handoff_clear() -> None:
    if _SP_FP8_HANDOFF:
        _Fp8Form.clear()
        if _SP_MOE_CODES:
            _MoeForm.clear()


def sp_fp8_form(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(fp8 codes [M, H] row-strided, scales [M, H/128] in transposed memory) of x, or None.

    Returns the pair only for the registered gathered layer input itself; callers fall back to
    their own per_1x128 quant otherwise.
    """
    if not _SP_FP8_HANDOFF or _Fp8Form.key is None:
        return None
    if _form_key(x) != _Fp8Form.key:
        return None
    _Fp8Form.used = True
    return _Fp8Form.codes, _Fp8Form.scales_t


def sp_fp8_handoff_check(kind: str, x: torch.Tensor, out: torch.Tensor, served) -> None:
    """Once per consumer kind per rank: the handoff GEMM output must equal the served
    requant + GEMM bitwise. Logs the result (the switch's dispatch line); raises on a mismatch."""
    if not active_attention_config().dsv4_sp_self_check or kind in _Fp8Form.checked:
        return
    _Fp8Form.checked.add(kind)
    ref = served(x)
    same = bool(torch.equal(out, ref))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger.info(
        "%s: fp8 gather codes handed to %s on rank %d (%d rows, no requant); "
        "bitwise equal to the served requant path: %s", SP_FP8_HANDOFF_LABEL, kind, rank,
        x.shape[0], same)
    if not same:
        raise RuntimeError(f"{SP_FP8_HANDOFF_LABEL}: {kind} output differs from the served path")


class SeqParallelSeamComm:
    """Per-TP-group state: the AITER QuickReduce reduce-scatter and the fp8 gather.

    The reduce-scatter runs on an AITER QuickReduce communicator owned by this object (the TP
    all-reduce's QuickReduce is vLLM's own csrc copy; AITER's collectives need AITER's
    DeviceComms). Rows M must be a multiple of 16 at hidden 7168 (each rank's segment a whole
    number of 2048-element atoms); callers pad. Calls must be issued in the same order on every
    rank (stream-ordered); any sequence of QuickReduce collectives may follow each other.
    """

    def __init__(self, tp_group, max_rows: int, hidden: int = 7168):
        import aiter as aiter_ops
        from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles
        from aiter.ops.quick_reduce_scatter import (
            qr_reduce_scatter,
            qr_reduce_scatter_route,
        )

        self.tp = tp_group
        self.world = tp_group.world_size
        if self.world != _WS:
            raise ValueError(f"SeqParallelSeamComm needs TP{_WS}, got {self.world}")
        self.rank = tp_group.rank_in_group
        self.hidden = hidden
        self.max_rows = max_rows
        self.row_align = self.row_alignment(hidden)
        self._qr_rs = qr_reduce_scatter
        self._qr_rs_route = qr_reduce_scatter_route
        ca = getattr(tp_group.device_communicator, "aiter_ar_comm", None)
        if ca is None:
            raise RuntimeError("SeqParallelSeamComm needs AITER custom all-reduce (aiter_ar_comm)")
        self._ca = self._gather_ca(tp_group, ca, max_rows, hidden)
        # the QuickReduce communicator, sized for the largest reduce-scatter input (bf16 rows)
        self._aiter = aiter_ops
        self._qr = aiter_ops.init_custom_qr(self.rank, self.world, max_rows * hidden * 2)
        qr_exchange_handles(self._qr, self.world, tp_group.cpu_group)
        dist.barrier(group=tp_group.cpu_group)
        logger.info(
            "SeqParallelSeamComm rank %d: AITER QuickReduce INT4 reduce-scatter, max_rows %d, "
            "fp8 1x128 gather via %s", self.rank, max_rows, type(ca).__name__)
        logger.info(
            "%s (m-sp-padcopy) on rank %d: SP reduce-scatter takes ragged M "
            "(no padded copy)", SP_RAGGED_RS_LABEL, self.rank)

    # per-row byte tail a gather may carry beside the packed fp8 row (routing: 2 x K x 4 bytes)
    _GATHER_TAIL_BYTES = 64

    def _gather_ca(self, tp_group, ca, max_rows: int, hidden: int):
        """The AITER custom all-gather the fp8 layer-input gathers run on.

        The TP group's AITER CA is sized for all-reduce (input <= max_size / (2 x world) for a
        gather: 8 MiB at the default 128 MiB buffer, i.e. 1,134 packed rows per rank). A
        sequence-parallel step gathers max_rows / 8 packed rows per rank, so with
        --max-num-batched-tokens > ~9k the shared buffer refuses the gather. Then this comm
        owns a gather-only instance sized for max_rows (never graph-captured: SP steps run
        eager). The shared instance is not resized: its max_size also gates which all-reduces
        take the custom path. When the shared buffer fits, it is used as before.
        """
        need = (max_rows // _WS) * (packed_row_bytes(hidden) + self._GATHER_TAIL_BYTES)
        impl = getattr(ca, "aiter_ca", None) or getattr(ca, "_impl", None)
        shared_max = getattr(impl, "max_size", 0)
        if need <= shared_max / (self.world * 2):
            return ca
        from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
            AiterCustomAllreduce,
        )

        mib = 1 << 20
        max_size = (need * self.world * 2 + mib - 1) // mib * mib
        own = AiterCustomAllreduce(
            group=tp_group.cpu_group, device=tp_group.device, max_size=max_size,
            register_graph_buffers=False,
        )
        if own.disabled:
            raise RuntimeError("SeqParallelSeamComm: AITER custom all-gather instance disabled")
        logger.info(
            "SeqParallelSeamComm rank %d: own AITER custom all-gather buffer of %d MiB for "
            "%d-row steps (shared CA max %d MiB fits %d bytes per rank, gather needs %d)",
            self.rank, max_size // mib, max_rows, shared_max // mib,
            int(shared_max / (self.world * 2)), need)
        return own

    @staticmethod
    def row_alignment(hidden: int) -> int:
        # segment numel (M/8 * hidden) must be a multiple of 2048 elements
        import math
        per_row = 2048 // math.gcd(hidden, 2048)  # segment rows per atom-aligned unit
        return _WS * per_row

    def reduce_scatter(self, x: torch.Tensor, out: torch.Tensor | None = None,
                       mp: int | None = None,
                       route: tuple[torch.Tensor, int] | None = None) -> torch.Tensor:
        """x: this rank's partial sum, bf16 [M, H] -> reduced rows [Mp/8, H] bf16 (INT4 wire).

        mp: the padded row count (a multiple of row_align, >= M; None: M). Rows [M, Mp) read as
        zero inside the kernel (m-sp-padcopy), bitwise equal to reducing a zero-padded copy.
        route (m-moe-route-fold): (slots, topk), the routed MoE's OPUS stage-2 MXFP8 per-slot rows
        [M * topk, H + H / 8] uint8; they replace x as the routed term (x gives only the shape) and
        are summed on load as the OPUS slot reduce does, bitwise equal to reducing its output.
        """
        m, h = x.shape
        mp = m if mp is None else mp
        assert h == self.hidden and x.dtype == torch.bfloat16 and x.is_contiguous()
        assert 0 < m <= mp and mp % self.row_align == 0 and mp <= self.max_rows, (
            m, mp, self.row_align, self.max_rows)
        if out is None:
            out = torch.empty(mp // _WS, h, dtype=x.dtype, device=x.device)
        if route is not None:
            slots, topk = route
            assert (slots.dtype == torch.uint8 and slots.is_contiguous()
                    and slots.device == x.device
                    and tuple(slots.shape) == (m * topk, h + h // 8)), (
                tuple(slots.shape), slots.dtype, m, topk, h)
            self._qr_rs_route(self._qr, slots, out, m, int(topk), h)
        else:
            self._qr_rs(self._qr, x, out, True)
        return out

    def gather_fp8(self, x: torch.Tensor, out: torch.Tensor | None = None,
                   m: int | None = None) -> torch.Tensor:
        """x: this rank's rows [M/8, H] bf16 -> [M, H] bf16, fp8 e4m3 1x128 on the wire.

        m: the real rows (for the fp8 handoff's registration); None registers nothing."""
        packed = fp8_1x128_quant_pack(x)
        pk = packed.view(torch.bfloat16)
        if not self._ca.should_custom_ag(pk):
            raise RuntimeError(f"AITER custom all-gather refuses {tuple(pk.shape)}")
        gat = self._ca.custom_all_gather(pk)
        return self._dequant(gat.view(torch.uint8).view(-1, packed.shape[1]), out, m)

    def _dequant(self, gat: torch.Tensor, out: torch.Tensor | None = None,
                 m: int | None = None) -> torch.Tensor:
        """Gathered packed rows [Mp, rowb] -> bf16 [Mp, H]; with the fp8 handoff on, also registers
        the first m rows' (codes view, transposed scales) for the fp8 GEMM consumers."""
        h = self.hidden
        if not _SP_FP8_HANDOFF or m is None:
            return fp8_1x128_dequant(gat, h, out)
        mp, rowb = gat.shape
        assert gat.is_contiguous() and rowb >= packed_row_bytes(h) and 0 < m <= mp
        if out is None:
            out = torch.empty(mp, h, dtype=torch.bfloat16, device=gat.device)
        st = torch.empty(h // _GROUP, m, dtype=torch.float32, device=gat.device)
        _fp8_1x128_dequant_ts[(mp, h // _GROUP // _GB)](gat, out, st, m, h, rowb, _GB, num_warps=4)
        x = out[:m]
        _Fp8Form.set(x, gat[:m, :h].view(torch.float8_e4m3fn), st.view(m, h // _GROUP))
        return out

    def _codes_only(self, gat: torch.Tensor, m: int) -> torch.Tensor:
        """m-moe-input-fp8: gathered packed rows -> placeholder bf16 [Mp, H] (unwritten after the
        first call) + the registered fp8 1x128 handoff and MXFP8 MoE codes of the first m rows."""
        h = self.hidden
        mp, rowb = gat.shape
        assert gat.is_contiguous() and rowb >= packed_row_bytes(h) and 0 < m <= mp
        out = torch.empty(mp, h, dtype=torch.bfloat16, device=gat.device)
        st = torch.empty(h // _GROUP, m, dtype=torch.float32, device=gat.device)
        y = torch.empty(m, h, dtype=torch.float8_e4m3fn, device=gat.device)
        e = torch.empty(m, h // 32, dtype=torch.uint8, device=gat.device)
        _fp8_1x128_codes_mx[(m, h // _GROUP // _GB)](gat, st, y, e, m, h, rowb, _GB, num_warps=2)
        x = out[:m]
        if active_attention_config().dsv4_sp_self_check and not _MoeForm.checked:
            # once per rank: materialize the served bf16 rows (so this call's consumers and their
            # own self-checks see real rows) and check codes-only against the served pair.
            _MoeForm.checked = True
            from aiter import dtypes
            from aiter.ops.quant import per_1x32_mx_quant_hip
            st2 = torch.empty_like(st)
            _fp8_1x128_dequant_ts[(mp, h // _GROUP // _GB)](gat, out, st2, m, h, rowb, _GB,
                                                            num_warps=4)
            y2, e2 = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8,
                                           scale_type=dtypes.fp8_e8m0, shuffle=False)
            u8 = lambda t: t.contiguous().view(torch.uint8)  # noqa: E731
            # bit patterns, not values: a NaN in a dummy step must not read as a mismatch
            same = (torch.equal(u8(st), u8(st2)) and torch.equal(u8(y), u8(y2))
                    and torch.equal(e, u8(e2)))
            logger.info(
                "%s: codes-only MoE input on rank %d (%d rows): 1x128 scales, MXFP8 codes and "
                "e8m0 bitwise equal to the served dequant + per_1x32 quant: %s",
                SP_MOE_CODES_LABEL, self.rank, m, same)
            if not same:
                raise RuntimeError(f"{SP_MOE_CODES_LABEL}: codes-only MoE input differs")
        _Fp8Form.set(x, gat[:m, :h].view(torch.float8_e4m3fn), st.view(m, h // _GROUP))
        _MoeForm.set(x, y, e.view(torch.float8_e8m0fnu))
        return out

    def gather_fp8_with_tail(
        self, x: torch.Tensor, tail: torch.Tensor, m: int | None = None,
        codes_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """gather_fp8 plus a per-row byte tail in the same AITER CA gather (no extra collective).

        x: [S, H] bf16, tail: [S, E] uint8 (copied exactly). Returns ([Mp, H] bf16, [Mp, E] uint8).
        """
        s, e = tail.shape
        assert tail.dtype == torch.uint8 and x.shape[0] == s
        rowb = packed_row_bytes(self.hidden)
        packed = torch.empty(s, rowb + e, dtype=torch.uint8, device=x.device)
        fp8_1x128_quant_pack(x, packed)
        packed[:, rowb:].copy_(tail)
        pk = packed.view(torch.bfloat16)
        if not self._ca.should_custom_ag(pk):
            raise RuntimeError(f"AITER custom all-gather refuses {tuple(pk.shape)}")
        gat = self._ca.custom_all_gather(pk).view(torch.uint8).view(-1, rowb + e)
        if codes_only:
            return self._codes_only(gat, m), gat[:, rowb:]
        return self._dequant(gat, None, m), gat[:, rowb:]

    def close(self):
        if getattr(self, "_qr", None):
            self._aiter.qr_destroy(self._qr)
            self._qr = None


# --- S4b: per-step state for the decoder layers (m-seq-parallel variant C) ---------------------

SEQ_PARALLEL_LABEL = "dsv4_seq_parallel"
# Mixed/prefill steps of at least this many rows run sequence-parallel (iteration 30 census: every
# mixed step but 1 of 217 is >= 1024 rows; decode-only steps, <= max_num_seqs * 7 = 896 rows, never).
SEQ_PARALLEL_MIN_ROWS = 1024


def sp_config_disable_reasons(vllm_config) -> list[str]:
    """Deployment-level reasons sequence parallelism cannot engage (empty: it can). Shared by the
    model's SP decision and by the attention layers, which load the row-local full weights only
    when SP can actually run. Model-structure checks (mHC fusion, MegaMoE) stay in the model."""
    import vllm.envs as envs
    from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
    from vllm.platforms.rocm import on_gfx950

    spec = vllm_config.speculative_config
    pc = vllm_config.parallel_config
    reasons = []
    if get_tensor_model_parallel_world_size() != _WS:
        reasons.append(f"TP != {_WS}")
    if pc.pipeline_parallel_size != 1:
        reasons.append("pipeline parallel")
    if spec is not None and spec.method != "dspark":
        # _mtp_hidden_buffer is skipped on sequence-parallel steps; DSpark reads the aux hidden
        # states instead (dflash speculator).
        reasons.append(f"speculative method {spec.method}")
    if vllm_config.model_config.hf_config.hidden_size != 7168:
        reasons.append("hidden size")
    # the route-fold reduce-scatter and the fp8 seams are gfx950 kernels (v_cvt_pk_bf16_f32, OCP e4m3)
    if not on_gfx950():
        reasons.append("the device is not gfx950")
    # SP reduce-scatters on the INT4 quick-reduce wire with the bf16->fp16 cast: engage only when
    # the deployment chose exactly that for its all-reduces
    if envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION != "INT4":
        reasons.append("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION is not INT4")
    if not envs.VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:
        reasons.append("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0")
    if getattr(get_tp_group().device_communicator, "aiter_ar_comm", None) is None:
        reasons.append("no AITER custom all-reduce")
    if pc.enable_dbo:
        reasons.append("dual-batch overlap")
    if pc.decode_context_parallel_size != 1:
        reasons.append("decode context parallel")
    if pc.enable_expert_parallel:
        reasons.append("expert parallel")
    return reasons
# m-compressor-shard plan step 1: on sequence-parallel steps each rank runs the MoE router gate and
# top-k on its own Mp/8 rows of the FFN seam output (bf16, before the fp8 gather) and the routing
# (48 B per row) rides in the same fp8 layer-input gather, instead of every rank routing all M
# rows of the gathered input. Off by default.
SP_ROUTER_SHARD_LABEL = "sp_router_shard"

# m-moe-input-fp8 (FIX, lands off): codes-only routed-MoE input on router-sharded SP steps.
# With sp_router_shard=1 the router reads this rank's rows before the gather, so the
# gathered bf16 FFN rows have only two consumers: the shared expert's gate_up (which already takes
# the gather's fp8 1x128 codes + transposed scales through the SP fp8 handoff) and AITER fused_moe,
# whose split path at M > 2048/topk re-quantizes them with per_1x32_mx_quant_hip (MXFP8 group-32,
# e8m0) before mxfp4_moe_sort_hip. Here one Triton kernel reads the gathered packed rows and writes
# the transposed 1x128 scales plus the MXFP8 codes and e8m0 bytes directly (no bf16 rows), and
# fused_moe takes its fp8 passthrough branch (hidden fp8 + a1_scale -> mxfp4_moe_sort_fwd, the same
# sort). Bitwise equal to the served pair (eval_moe_input_codes_only, iter 18): -58.6 us per layer
# at 11,932 rows. The bf16 rows handed to the layer are an unwritten placeholder (identity key for
# both handoffs); the first call per rank materializes them and checks the codes, the scales and
# the routed-MoE output against the served path bitwise (raises on a mismatch), and every call
# asserts both consumers took the codes. Part of dsv4_seq_parallel. Exact.
SP_MOE_CODES_LABEL = "sp_moe_codes"
_SP_MOE_CODES = True  # on with dsv4_seq_parallel; no switch of its own
# AITER's quant+sort takes the split (per_1x32 + sort) path for stage 1 above 8*256/topk rows; the
# tuned stage-2 kernels are non-atomic from M 2048 up (deterministic, checkable bitwise).
SP_MOE_CODES_MIN_ROWS = 2048


@triton.jit(do_not_specialize=["M"])
def _fp8_1x128_codes_mx(p_ptr, st_ptr, q_ptr, e_ptr, M, H: tl.constexpr, ROWB: tl.constexpr,
                        GB: tl.constexpr):
    # Packed gathered row [H fp8 | H/128 fp32 | tail] -> st[g * M + row] (the 1x128 scales,
    # column-major as _fp8_1x128_dequant_ts writes them) + MXFP8: codes q[row, :] and e8m0 bytes
    # e[row, k] of each 1x32 group k, computed from the same bf16 value the served dequant stores
    # (AITER per_1x32 RoundUp: e = exponent of amax/448 rounded up; an all-zero group gets 85).
    row = tl.program_id(0)
    gb = tl.program_id(1)
    k = gb * GB * 4 + tl.arange(0, GB * 4)
    j = tl.arange(0, 32)
    base = p_ptr + row.to(tl.int64) * ROWB
    q = tl.load(base.to(tl.pointer_type(tl.float8e4nv), bitcast=True) + k[:, None] * 32 + j[None, :]).to(tl.float32)
    s = tl.load((base + H).to(tl.pointer_type(tl.float32), bitcast=True) + k // 4)
    x32 = (q * s[:, None]).to(tl.bfloat16).to(tl.float32)
    g = gb * GB + tl.arange(0, GB)
    s1 = tl.load((base + H).to(tl.pointer_type(tl.float32), bitcast=True) + g)
    tl.store(st_ptr + g.to(tl.int64) * M + row, s1)
    amax = tl.max(tl.abs(x32), axis=1)
    u = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    e = (u >> 23) & 0xFF
    e = tl.where((e < 0xFF) & ((u & 0x7FFFFF) != 0), e + 1, e)
    e = tl.where(amax == 0.0, 85, e)
    sc = (e << 23).to(tl.float32, bitcast=True)
    codes = tl.div_rn(x32, sc[:, None]).to(tl.float8e4nv)
    qrow = q_ptr + row.to(tl.int64) * H
    tl.store(qrow.to(tl.pointer_type(tl.float8e4nv), bitcast=True) + k[:, None] * 32 + j[None, :], codes)
    tl.store(e_ptr + row.to(tl.int64) * (H // 32) + k, e.to(tl.uint8))


class _MoeForm:
    """MXFP8 codes + e8m0 of the current FFN sublayer's gathered input (codes-only mode)."""

    key: tuple | None = None
    x: torch.Tensor | None = None
    codes: torch.Tensor | None = None
    e8m0: torch.Tensor | None = None
    used: bool = False
    checked: bool = False
    moe_checked: bool = False
    logged: bool = False

    @classmethod
    def set(cls, x, codes, e8m0) -> None:
        cls.x, cls.key, cls.codes, cls.e8m0, cls.used = x, _form_key(x), codes, e8m0, False

    @classmethod
    def clear(cls) -> None:
        cls.x = cls.key = cls.codes = cls.e8m0 = None
        cls.used = False


def sp_moe_codes_log_state(router_shard: bool) -> None:
    """Once per rank, on the first SP FFN seam: whether the switch's path is live (it needs the
    router shard and the fp8 handoff; without them the FFN input stays the served bf16 rows)."""
    if not _SP_MOE_CODES or _MoeForm.logged:
        return
    _MoeForm.logged = True
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    live = router_shard and _SP_FP8_HANDOFF
    logger.info(
        "%s on rank %d: codes-only routed-MoE input %s (router shard %s, fp8 handoff %s, "
        "SP steps >= %d rows)", SP_MOE_CODES_LABEL, rank, "LIVE" if live else "INACTIVE",
        router_shard, _SP_FP8_HANDOFF, SP_MOE_CODES_MIN_ROWS)


def sp_moe_codes_wanted(m: int) -> bool:
    return _SP_MOE_CODES and _SP_FP8_HANDOFF and m >= SP_MOE_CODES_MIN_ROWS


def sp_moe_codes_form(x: torch.Tensor):
    """(MXFP8 codes [M, H], e8m0 [M, H/32]) for the registered gathered FFN input x, else None.
    The second element tells whether this is the first (checked) call on this rank."""
    if not _SP_MOE_CODES or _MoeForm.key is None or _form_key(x) != _MoeForm.key:
        return None
    _MoeForm.used = True
    first = active_attention_config().dsv4_sp_self_check and not _MoeForm.moe_checked
    _MoeForm.moe_checked = True
    return _MoeForm.codes, _MoeForm.e8m0, first


def sp_moe_codes_check(out: torch.Tensor, ref: torch.Tensor, m: int) -> None:
    u8 = lambda t: t.contiguous().view(torch.uint8)  # noqa: E731
    same = bool(torch.equal(u8(out), u8(ref)))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger.info(
        "%s: routed-MoE input codes-only on rank %d (%d rows, AITER fused_moe fp8 passthrough); "
        "MoE output bitwise equal to the served bf16 path: %s", SP_MOE_CODES_LABEL, rank, m, same)
    if not same:
        raise RuntimeError(f"{SP_MOE_CODES_LABEL}: routed-MoE output differs from the served path")


def sp_moe_codes_done(shared_reads_rows: bool = True) -> None:
    """After the FFN sublayer of a codes-only step: both consumers must have taken the codes (the
    placeholder bf16 rows are unwritten). shared_reads_rows=False (m-shared-rowlocal): the shared
    expert ran on this rank's own rows and never read the gathered rows; only the routed MoE must."""
    if _MoeForm.key is None:
        return
    if not (_MoeForm.used and (_Fp8Form.used or not shared_reads_rows)):
        raise RuntimeError(
            f"{SP_MOE_CODES_LABEL}: a consumer read the placeholder rows (routed MoE took codes: "
            f"{_MoeForm.used}, shared expert took codes: {_Fp8Form.used})")


def _registered_rows(out: torch.Tensor, m: int) -> torch.Tensor:
    """out[:m], returning the very tensor the fp8 handoff registered (same object, same key)."""
    if _SP_FP8_HANDOFF and _Fp8Form.x is not None and _Fp8Form.x.data_ptr() == out.data_ptr() \
            and _Fp8Form.x.shape[0] == m:
        return _Fp8Form.x
    return out[:m]


# --- m-sp-padcopy: ragged INT4 reduce-scatter (C28 foundation plan step 2) --------------------
#
# Part of dsv4_seq_parallel (on whenever it is on). When M is not a multiple of row_align the
# [M, H] partial sum is reduce-scattered as it is: AITER's qr_reduce_scatter bounds the read of
# every rank's shard to the M real rows and the hardware returns zeros past it, so there is no
# zero-padded copy and the result equals a padded call bitwise. Exact.
SP_RAGGED_RS_LABEL = "sp_ragged_rs"
_RAGGED_RS_LOGGED = [False]

# --- m-moe-route-fold (RS host): the routed MoE's slot reduce folded into the INT4 RS ----------
#
# Part of dsv4_seq_parallel (on whenever it is on; the FFN seam uses it with the row-local shared
# expert). On sequence-parallel steps the FFN seam arms AITER's route-output capture around the
# routed MoE: an OPUS stage 2 that writes MXFP8 per-slot rows hands them over instead of running
# opus_moe_stage2_reduce_token_slot_route_output, and AITER's qr_reduce_scatter_route sums the topk
# slots on load (same fp32 order and bf16 rounding). The slot reduce's [M, 7168] bf16 write and
# the RS's read of it go away; the RS reads the slots instead. Exact. A stage 2 without MXFP8
# route rows (atomic or bf16 route_out) runs the ordinary reduce.
SP_RS_ROUTE_FOLD_LABEL = "sp_rs_route_fold"
_SP_RS_ROUTE_FOLD = True  # on with dsv4_seq_parallel; no switch of its own
_RS_ROUTE_FOLD_LOGGED = [False]


def sp_rs_route_fold_enabled() -> bool:
    return _SP_RS_ROUTE_FOLD


def _route_capture():
    from aiter.ops.opus.moe_stage2_a8w4 import route_output_capture
    return route_output_capture()


class sp_route_capture:
    """Arms AITER's route-output capture for the routed MoE of one SP FFN sublayer (no-op when the
    switch is off). On exit the capture is disarmed; `route` holds (slots, topk) or None."""

    def __init__(self, m: int, enabled: bool = True):
        self.m = m
        self.enabled = enabled and _SP_RS_ROUTE_FOLD
        self.route = None

    def __enter__(self):
        if self.enabled:
            cap = _route_capture()
            assert not cap.armed and cap.route_out is None
            cap.armed = True
        return self

    def __exit__(self, *exc):
        if self.enabled:
            cap = _route_capture()
            cap.armed = False
            r = cap.take()
            if r is not None and exc[0] is None:
                slots, topk, token_num = r
                if token_num != self.m:
                    raise RuntimeError(
                        f"{SP_RS_ROUTE_FOLD_LABEL}: captured {token_num} rows, step has {self.m}")
                self.route = (slots, topk)
        return False


class sp_route_capture_suspended:
    """Disarms the capture for a nested fused_moe that must reduce normally (a self-check)."""

    def __enter__(self):
        self.prev = False
        if _SP_RS_ROUTE_FOLD:
            cap = _route_capture()
            self.prev, cap.armed = cap.armed, False
        return self

    def __exit__(self, *exc):
        if _SP_RS_ROUTE_FOLD:
            _route_capture().armed = self.prev
        return False


def sp_route_capture_armed() -> bool:
    return _SP_RS_ROUTE_FOLD and _route_capture().armed


def _sp_route_reduce_ref(slots: torch.Tensor, topk: int, m: int, h: int) -> torch.Tensor:
    """The served slot reduce (AITER OPUS, MXFP8 route rows) -> bf16 [m, h]."""
    from aiter.ops.opus.moe_stage2_a8w4 import (
        opus_moe_stage2_reduce_token_slot_route_output_fwd,
    )
    return opus_moe_stage2_reduce_token_slot_route_output_fwd(slots, topk=topk)


def _sp_rs_route_fold_check(folded: torch.Tensor, ref: torch.Tensor, rank: int, m: int) -> None:
    eq = bool(torch.equal(folded.view(torch.int16), ref.view(torch.int16)))
    logger.info("%s (m-moe-route-fold) self-check on rank %d (%d rows): slot-folded RS bitwise %s "
                "the served slot reduce + RS", SP_RS_ROUTE_FOLD_LABEL, rank, m,
                "EQUAL to" if eq else "DIFFERENT from")
    if not eq:
        raise RuntimeError(f"{SP_RS_ROUTE_FOLD_LABEL}: slot-folded reduce-scatter differs on rank {rank}")


class SeqParallelStep:
    """One forward step's row split: M real rows padded to Mp = ceil(M / align) * align.

    Rank r owns rows [r * Mp/8, (r+1) * Mp/8). Rows >= M exist only in the last rank's segment; they
    are zero at entry, every seam op is row-independent (iteration 31: real rows bitwise unaffected by
    the tail's content), and the gathers drop them.
    """

    def __init__(self, comm: SeqParallelSeamComm, m: int):
        self.comm = comm
        self.m = m
        a = comm.row_align
        self.mp = (m + a - 1) // a * a
        self.s = self.mp // _WS
        self.start = comm.rank * self.s

    def rows(self, t: torch.Tensor) -> torch.Tensor:
        """Full-row tensor [M, ...] -> this rank's [Mp/8, ...] rows (zero tail past M)."""
        end = min(self.start + self.s, self.m)
        n = max(end - self.start, 0)
        if n == self.s:
            return t[self.start:end].contiguous()
        out = t.new_zeros((self.s,) + tuple(t.shape[1:]))
        if n:
            out[:n].copy_(t[self.start:end])
        return out

    def reduce_scatter(self, x: torch.Tensor,
                       route: tuple[torch.Tensor, int] | None = None) -> torch.Tensor:
        """This rank's partial sum [M, H] -> reduced rows [Mp/8, H] (INT4 wire, as the served AR).

        The zero tail [M, Mp) comes from the kernel's bounded input read (m-sp-padcopy), not from
        a padded [Mp, H] copy of x; bitwise equal.
        route (m-moe-route-fold): (slots, topk) of the routed MoE's MXFP8 per-slot rows; x is then
        unwritten (only its shape is used) and the kernel sums the slots on load."""
        if route is not None:
            return self._reduce_scatter_route(x, route)
        assert x.shape[0] == self.m, (x.shape, self.m)
        if not x.is_contiguous():
            x = x.contiguous()
        if self.mp != self.m and not _RAGGED_RS_LOGGED[0]:
            _RAGGED_RS_LOGGED[0] = True
            logger.info(
                "%s (m-sp-padcopy): ragged INT4 reduce-scatter dispatched on rank %d "
                "(%d rows read, padded to %d, zero tail from the bounded input read; "
                "no padded copy)", SP_RAGGED_RS_LABEL, self.comm.rank, self.m, self.mp)
        return self.comm.reduce_scatter(x, mp=self.mp)

    def _reduce_scatter_route(self, x: torch.Tensor,
                              route: tuple[torch.Tensor, int]) -> torch.Tensor:
        slots, topk = route
        h = x.shape[1]
        assert x.shape[0] == self.m, (x.shape, self.m)
        first = not _RS_ROUTE_FOLD_LOGGED[0]
        if first:
            _RS_ROUTE_FOLD_LOGGED[0] = True
            logger.info(
                "%s (m-moe-route-fold): routed MoE slot reduce folded into the INT4 "
                "reduce-scatter on rank %d (%d rows x %d MXFP8 slots read by "
                "qr_reduce_scatter_route; no opus slot reduce)",
                SP_RS_ROUTE_FOLD_LABEL, self.comm.rank, self.m, topk)
        out = self.comm.reduce_scatter(x, mp=self.mp, route=(slots, topk))
        if first and active_attention_config().dsv4_sp_self_check:
            ref = self.comm.reduce_scatter(_sp_route_reduce_ref(slots, topk, self.m, h),
                                           mp=self.mp)
            _sp_rs_route_fold_check(out, ref, self.comm.rank, self.m)
        return out

    def gather(self, xs: torch.Tensor) -> torch.Tensor:
        """Layer input rows [Mp/8, H] -> [M, H] on every rank (fp8 e4m3 1x128 wire)."""
        out = self.comm.gather_fp8(xs.contiguous(), m=self.m)
        return _registered_rows(out, self.m)

    def gather_with_routing(
        self, xs: torch.Tensor, w: torch.Tensor, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Layer input rows [Mp/8, H] plus their MoE routing (w fp32 [Mp/8, K], ids 4-byte
        [Mp/8, K]) -> [M, H] bf16 (fp8 wire), [M, K] weights and [M, K] ids (exact), in one gather.
        """
        s, k = w.shape
        assert ids.shape == (s, k) and w.element_size() == 4 and ids.element_size() == 4
        tail = torch.cat([w.contiguous().view(torch.int32), ids.contiguous().view(torch.int32)], 1)
        x, t = self.comm.gather_fp8_with_tail(xs.contiguous(), tail.view(torch.uint8), m=self.m,
                                              codes_only=sp_moe_codes_wanted(self.m))
        t = t[: self.m].contiguous().view(torch.int32)
        # the MoE kernels index topk weights/ids as dense [M, K]: materialize each half
        return (_registered_rows(x, self.m), t[:, :k].contiguous().view(w.dtype),
                t[:, k:].contiguous().view(ids.dtype))

    def gather_exact(self, xs: torch.Tensor) -> torch.Tensor:
        """[Mp/8, H] -> [M, H], exact bf16 (DSpark aux means and the head output)."""
        return self.comm.tp.all_gather(xs.contiguous(), dim=0)[: self.m]


# --- m-compressor-shard plan step 4: window-aligned compressor row shard on SP steps -----------
#
# Part of dsv4_seq_parallel (on whenever it is on). On a sequence-parallel step every rank
# holds the full layer input after sp.gather, and the served attention runs the compressor GEMM
# (x @ fused_wkv_wgate.T, fp32), save_partial_states and compress on all M rows on every rank. Here
# the compression windows of each prefill segment are split across the TP ranks (ledger
# m-compressor-shard-s2/-s3):
#   R    replicated rows, one GEMM with the same row set on every rank: every row of a request whose
#        start position the host cannot vouch for (decode rows with drafts, short chunks), each
#        prefill segment's tail (positions > last boundary - (coff-1)*ratio: the states a later
#        chunk or decode step reads), and for C4 the (coff-1)*ratio states before every cache-hit
#        boundary (multiples of the scheduler block): a request resuming there reads them.
#   S_r  one contiguous row slice covering the windows (with the C4 overlap halo) of the prefill
#        boundary tokens this rank owns (the segments' boundaries, split into 8 contiguous runs).
#   save S_r then R (R wins where they overlap); compress R's requests' tokens + owned boundaries.
#   The owned compressed rows (C4: main 584 B + indexer 132 B, C128: 584 B) then reach every rank's
#   paged caches through one AITER custom all-gather and a slot scatter (exact byte copies).
# Numerics: compressed rows differ from the served full-M GEMM by fp32 GEMM rounding (1-fp8-code
# flips on a few % of rows; iteration 42), so this is NUMERICS: yes.
SP_COMPRESSOR_SHARD_LABEL = "sp_compressor_shard"


@triton.jit(do_not_specialize=["nmax", "rank", "kvm_bs", "kvm_sw", "kvi_bs", "kvi_sw"])
def _comp_rows_xfer(buf_ptr, tok_tab_ptr, nmax, rank,
                    slot_m_ptr, kvm_ptr, kvm_bs, kvm_sw,
                    slot_i_ptr, kvi_ptr, kvi_bs, kvi_sw,
                    ROW_W: tl.constexpr, DM: tl.constexpr, SM: tl.constexpr,
                    DI: tl.constexpr, SI: tl.constexpr, HAS_I: tl.constexpr,
                    PACK: tl.constexpr, BLK: tl.constexpr):
    """Copy compressed cache rows (int32 words) between the paged caches and a [world*nmax, ROW_W]
    exchange buffer. PACK: this rank's owned rows -> buf[k]. Else: peers' rows buf[j*nmax+k] ->
    cache. Cache row of slot s: data at blk*sw + pos*D, scale at blk*sw + bs*D + pos*S (words).
    """
    pid = tl.program_id(0)
    if PACK:
        j = rank
        k = pid
        brow = pid
    else:
        j = pid // nmax
        k = pid % nmax
        brow = pid
        if j == rank:
            return
    tok = tl.load(tok_tab_ptr + j * nmax + k)
    if tok < 0:
        return
    o = tl.arange(0, BLK)
    b = buf_ptr + brow.to(tl.int64) * ROW_W
    slot = tl.load(slot_m_ptr + tok)
    if slot >= 0:
        blk = (slot // kvm_bs).to(tl.int64)
        pos = slot % kvm_bs
        cd = kvm_ptr + blk * kvm_sw + pos * DM + o
        cs = kvm_ptr + blk * kvm_sw + kvm_bs * DM + pos * SM + o
        if PACK:
            tl.store(b + o, tl.load(cd, mask=o < DM), mask=o < DM)
            tl.store(b + DM + o, tl.load(cs, mask=o < SM), mask=o < SM)
        else:
            tl.store(cd, tl.load(b + o, mask=o < DM), mask=o < DM)
            tl.store(cs, tl.load(b + DM + o, mask=o < SM), mask=o < SM)
    if HAS_I:
        si = tl.load(slot_i_ptr + tok)
        if si >= 0:
            bi = (si // kvi_bs).to(tl.int64)
            pi = si % kvi_bs
            ci = kvi_ptr + bi * kvi_sw + pi * DI + o
            cj = kvi_ptr + bi * kvi_sw + kvi_bs * DI + pi * SI + o
            if PACK:
                tl.store(b + DM + SM + o, tl.load(ci, mask=o < DI), mask=o < DI)
                tl.store(b + DM + SM + DI + o, tl.load(cj, mask=o < SI), mask=o < SI)
            else:
                tl.store(ci, tl.load(b + DM + SM + o, mask=o < DI), mask=o < DI)
                tl.store(cj, tl.load(b + DM + SM + DI + o, mask=o < SI), mask=o < SI)


class _CacheRows:
    """A compressed KV cache viewed as int32 words: (tensor, rows per block, block stride, D, S)."""

    def __init__(self, kv_cache: torch.Tensor, token_stride: int, scale_dim: int):
        assert kv_cache.dtype == torch.uint8 and kv_cache.stride(-1) == 1
        assert kv_cache.stride(0) % 4 == 0 and token_stride % 4 == 0 and scale_dim % 4 == 0
        self.base = kv_cache
        self.bs = kv_cache.shape[1]
        self.sw = kv_cache.stride(0) // 4
        self.d = token_stride // 4
        self.s = scale_dim // 4

    def words(self) -> torch.Tensor:
        # a flat int32 alias of the cache storage from the tensor's first element
        t = self.base
        n = (t.untyped_storage().nbytes() - t.storage_offset()) // 4 * 4
        return torch.as_strided(t, (n,), (1,)).view(torch.int32)


class CompressorShardPlan:
    """Host-built row split for one compress ratio on one SP step (identical on every rank)."""

    def __init__(self, sp: "SeqParallelStep", ratio: int, qsl, seq_ub, num_tokens: int,
                 max_decode_q: int, align: int, device, sp_aligned: bool = False,
                 need_head: int = 0):
        """sp_aligned (m-attn-rowlocal step 2b input, dsv4_rowlocal_input): rank q owns
        the prefill boundaries inside its SP slice [q*s, (q+1)*s) instead of the q-th of 8 equal
        runs, and the plan also lists the hidden rows each rank must receive from the others
        (replicated rows R, rows [0, need_head), and the window halo of each rank's slice) for a
        step whose layer input is not gathered."""
        import numpy as np

        world, rank = _WS, sp.comm.rank
        coff = 2 if ratio == 4 else 1
        m = num_tokens
        r_mask = np.zeros(m, dtype=bool)
        c_mask = np.zeros(m, dtype=bool)
        bnds, seg_t0 = [], []
        n_req = len(qsl) - 1
        for i in range(n_req):
            t0, t1 = int(qsl[i]), int(qsl[i + 1])
            t1 = min(t1, m)
            if t1 <= t0:
                continue
            ln = t1 - t0
            if ln <= max_decode_q:
                # decode rows (their CPU length is an upper bound with drafts) and short chunks:
                # replicated GEMM rows, every token offered to the compress kernel
                r_mask[t0:t1] = True
                c_mask[t0:t1] = True
                continue
            start = int(seq_ub[i]) - ln  # exact for a prefill chunk (no drafts)
            pos = np.arange(start, start + ln, dtype=np.int64)
            b = np.nonzero((pos + 1) % ratio == 0)[0]
            lb = int(pos[b[-1]]) if len(b) else start - 1
            r_mask[t0:t1] |= pos > lb - (coff - 1) * ratio
            if coff == 2:
                r_mask[t0:t1] |= (pos % align) >= align - (coff - 1) * ratio
            bnds.append(b + t0)
            seg_t0.append(np.full(len(b), t0, dtype=np.int64))
        bnd = np.concatenate(bnds) if bnds else np.zeros(0, dtype=np.int64)
        bt0 = np.concatenate(seg_t0) if seg_t0 else np.zeros(0, dtype=np.int64)
        nb = len(bnd)
        if sp_aligned:
            cuts = [int(np.searchsorted(bnd, q * sp.s)) for q in range(world)] + [nb]
        else:
            cuts = [(q * nb) // world for q in range(world + 1)]
        nmax = max(1, max(cuts[q + 1] - cuts[q] for q in range(world)))
        tok_tab = np.full((world, nmax), -1, dtype=np.int64)
        for q in range(world):
            tok_tab[q, : cuts[q + 1] - cuts[q]] = bnd[cuts[q]:cuts[q + 1]]
        own = bnd[cuts[rank]:cuts[rank + 1]]
        if len(own):
            w = coff * ratio - 1
            self.s_lo = int(max(bt0[cuts[rank]], own[0] - w))
            # Even start: the int64 per-token slices (positions, slot mapping) stay 16-byte
            # aligned, so the save kernel runs the pointer specialization warmup compiled.
            # The extra row (if any) saves its own token's state at its own slot: harmless.
            self.s_lo -= self.s_lo % 2
            self.s_hi = int(own[-1]) + 1
            c_mask[own] = True
        else:
            self.s_lo = self.s_hi = 0
        r_idx = np.nonzero(r_mask)[0].astype(np.int64)
        c_idx = np.nonzero(c_mask)[0].astype(np.int64)
        self.n_r, self.n_c, self.nmax, self.n_bnd, self.n_own = (
            len(r_idx), len(c_idx), nmax, nb, len(own))
        self.c_idx_np = c_idx
        extra = []
        self.sp_aligned = sp_aligned
        self.n_send = self.n_recv = self.kmax = 0
        if sp_aligned:
            # Rows some rank reads but does not own: R, the head rows, and each rank's slice
            # window outside its own SP rows (the halo before its slice start).
            w = coff * ratio - 1
            s = sp.s
            u = r_mask.copy()
            u[: min(need_head, m)] = True
            for q in range(world):
                o = bnd[cuts[q]:cuts[q + 1]]
                if not len(o):
                    continue
                lo = int(max(bt0[cuts[q]], o[0] - w))
                lo -= lo % 2
                hi = int(o[-1]) + 1
                u[lo:min(hi, q * s)] = True
                u[max(lo, (q + 1) * s):hi] = True
            sends = [np.nonzero(u[j * s:min((j + 1) * s, m)])[0].astype(np.int64) + j * s
                     if j * s < m else np.zeros(0, dtype=np.int64) for j in range(world)]
            kmax = max(1, max(len(x) for x in sends))
            send_idx = sends[rank] - rank * s
            src = [j * kmax + np.arange(len(sends[j]), dtype=np.int64)
                   for j in range(world) if j != rank]
            dst = [sends[j] for j in range(world) if j != rank]
            recv_src = np.concatenate(src) if src else np.zeros(0, dtype=np.int64)
            recv_dst = np.concatenate(dst) if dst else np.zeros(0, dtype=np.int64)
            self.n_send, self.n_recv, self.kmax = len(send_idx), len(recv_src), kmax
            self.n_need_rows = int(u.sum())
            extra = [send_idx, recv_src, recv_dst]
        host = torch.from_numpy(
            np.concatenate([r_idx, c_idx, tok_tab.reshape(-1)] + extra)).pin_memory()
        dev_all = host.to(device, non_blocking=True)
        self.r_idx = dev_all[: self.n_r]
        self.c_idx = dev_all[self.n_r: self.n_r + self.n_c]
        o = self.n_r + self.n_c + tok_tab.size
        self.tok_tab = dev_all[self.n_r + self.n_c: o]
        if sp_aligned:
            self.send_idx = dev_all[o: o + self.n_send]
            self.recv_src = dev_all[o + self.n_send: o + self.n_send + self.n_recv]
            self.recv_dst = dev_all[o + self.n_send + self.n_recv:]
        self._sub: dict = {}

    def gather_needed(self, comm: "SeqParallelSeamComm", packed: torch.Tensor,
                      x_full: torch.Tensor) -> None:
        """sp_aligned plans: send this rank's listed rows (fp8 1x128 packed, the SP gather's wire
        format) and write every other rank's listed rows, dequantized, into x_full at their step
        rows. One AITER custom all-gather of kmax packed rows per rank; the dequantized rows are
        bitwise the rows SeqParallelStep.gather would have produced."""
        assert self.sp_aligned
        rowb = packed.shape[1]
        buf = packed.new_zeros((self.kmax, rowb))
        if self.n_send:
            torch.index_select(packed, 0, self.send_idx, out=buf[: self.n_send])
        pk = buf.view(torch.bfloat16)
        if comm._ca.should_custom_ag(pk):
            gat = comm._ca.custom_all_gather(pk)
        else:
            gat = comm.tp.all_gather(pk, dim=0)
        if self.n_recv:
            rows = gat.view(torch.uint8).view(-1, rowb).index_select(0, self.recv_src)
            x_full.index_copy_(0, self.recv_dst, fp8_1x128_dequant(rows, comm.hidden))

    def sub(self, t: torch.Tensor, which: str) -> torch.Tensor:
        """t (per-token, [M, ...]) restricted to R ('r'), the compress list ('c') or S ('s')."""
        if which == "s":
            return t[self.s_lo:self.s_hi]
        key = (which, id(t))
        v = self._sub.get(key)
        if v is None:
            v = t.index_select(0, self.r_idx if which == "r" else self.c_idx)
            self._sub[key] = (v, t)  # keep t alive so its id stays unique this step
            return v
        return v[0]

    def exchange(self, comm: "SeqParallelSeamComm", main: tuple, idx: tuple | None) -> None:
        """Owned compressed rows -> every rank's caches. main/idx = (_CacheRows, kv slot mapping)."""
        if self.n_bnd == 0:
            return
        cm, sm = main
        has_i = idx is not None
        ci, si = idx if has_i else (cm, sm)
        row_w = cm.d + cm.s + ((ci.d + ci.s) if has_i else 0)
        row_w = (row_w + 3) // 4 * 4  # 16-byte rows
        blk = triton.next_power_of_2(max(cm.d, ci.d if has_i else 1))
        wm, wi = cm.words(), ci.words()
        buf = torch.empty(self.nmax, row_w, dtype=torch.int32, device=sm.device)
        args = (self.tok_tab, self.nmax, comm.rank, sm, wm, cm.bs, cm.sw, si, wi, ci.bs, ci.sw)
        kw = dict(ROW_W=row_w, DM=cm.d, SM=cm.s, DI=ci.d if has_i else 0,
                  SI=ci.s if has_i else 0, HAS_I=has_i, BLK=blk, num_warps=4)
        _comp_rows_xfer[(self.nmax,)](buf, *args, PACK=True, **kw)
        pk = buf.view(torch.bfloat16)
        if comm._ca.should_custom_ag(pk):
            gat = comm._ca.custom_all_gather(pk)
        else:
            gat = comm.tp.all_gather(pk, dim=0)
        gat = gat.view(torch.int32)
        _comp_rows_xfer[(_WS * self.nmax,)](gat, *args, PACK=False, **kw)


def sp_compressor_shard_warmup(worker) -> None:
    """Compile the compressor shard's kernels before the server reports healthy.

    The shard runs only on sequence-parallel steps (>= SEQ_PARALLEL_MIN_ROWS rows) with real
    attention metadata; the generic DSv4 mixed warmup uses 16 tokens and never reaches it, so the
    exchange kernel and the subset save/compress launches would otherwise JIT on the first served
    burst step. Two real mixed prefill+decode steps (the scheduler's chunk and an odd size) run
    through the served model: every launch variant of the shard (C4 main+indexer, C128; pack and
    scatter; slice and replicated saves) compiles here. No-op unless both switches are on.
    """
    runner = worker.model_runner
    if not getattr(getattr(runner, "vllm_config", None), "use_v2_model_runner", False):
        return
    if not any(getattr(m, "_sp_enabled", False) for m in worker.get_model().modules()):
        return
    from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

    mbt = worker.scheduler_config.max_num_batched_tokens
    sizes = sorted({mbt, max(SEQ_PARALLEL_MIN_ROWS + 3, mbt // 2 + 3)}, reverse=True)
    for n in sizes:
        if n < SEQ_PARALLEL_MIN_ROWS:
            continue
        ran = run_mixed_prefill_decode_warmup(
            runner, worker.execute_model, worker.sample_tokens, n,
            req_id_prefix=f"_dsv4_sp_comp_shard_warmup_{n}",
        )
        logger.info("%s: compressor shard warmup step of %d rows %s",
                    SP_COMPRESSOR_SHARD_LABEL, n, "ran" if ran else "skipped")
