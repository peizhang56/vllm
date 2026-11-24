# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel mHC seams for DeepSeek-V4 on ROCm TP8 (`dsv4_seq_parallel`).

On prefill and mixed steps of at least SEQ_PARALLEL_MIN_ROWS rows, each sublayer's
partial sum is reduce-scattered instead of all-reduced (AITER QuickReduce INT4, the
first phase of the two-shot all-reduce as its own launch), the memory-bound mHC seam
runs on each rank's 1/8 of the rows, and the next sublayer's input is all-gathered as
fp8 e4m3 with 1x128 fp32 scales. The gathered fp8 codes and scales are handed straight
to the fp8 block-scale GEMMs that read that input (attention fused_wqa_wkv and the
shared expert's gate_up), which would otherwise requantize the dequantized rows to the
same bytes.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_WS = 8
_GROUP = 128
_GB = 8  # 128-element groups per program

SEQ_PARALLEL_LABEL = "dsv4_seq_parallel"
# Mixed/prefill steps of at least this many rows run sequence-parallel; decode-only
# steps (<= max_num_seqs * (1 + num_speculative_tokens) rows) stay on the TP path.
SEQ_PARALLEL_MIN_ROWS = 1024

# Set from the model's VllmConfig when sequence parallelism is armed.
_SELF_CHECK = False


def set_sp_self_check(enabled: bool) -> None:
    global _SELF_CHECK
    _SELF_CHECK = enabled


@triton.jit
def _fp8_1x128_quant_pack(
    x_ptr, o_ptr, H: tl.constexpr, ROWB: tl.constexpr, GB: tl.constexpr
):
    # packed row: [H fp8 e4m3 | H/128 fp32 scales]; bitwise equal to
    # s = amax / 448 (scalar div = mul by fp32 reciprocal), q = (x / s).to(fp8)
    row = tl.program_id(0)
    gb = tl.program_id(1)
    g = gb * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    x = tl.load(x_ptr + row.to(tl.int64) * H + g[:, None] * 128 + c[None, :]).to(
        tl.float32
    )
    s = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12) * (1.0 / 448.0)
    q = tl.div_rn(x, s[:, None]).to(tl.float8e4nv)
    base = o_ptr + row.to(tl.int64) * ROWB
    tl.store(
        base.to(tl.pointer_type(tl.float8e4nv), bitcast=True)
        + g[:, None] * 128
        + c[None, :],
        q,
    )
    tl.store((base + H).to(tl.pointer_type(tl.float32), bitcast=True) + g, s)


@triton.jit
def _fp8_1x128_dequant(
    p_ptr, o_ptr, H: tl.constexpr, ROWB: tl.constexpr, GB: tl.constexpr
):
    row = tl.program_id(0)
    gb = tl.program_id(1)
    g = gb * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    base = p_ptr + row.to(tl.int64) * ROWB
    q = tl.load(
        base.to(tl.pointer_type(tl.float8e4nv), bitcast=True)
        + g[:, None] * 128
        + c[None, :]
    ).to(tl.float32)
    s = tl.load((base + H).to(tl.pointer_type(tl.float32), bitcast=True) + g)
    tl.store(
        o_ptr + row.to(tl.int64) * H + g[:, None] * 128 + c[None, :],
        (q * s[:, None]).to(tl.bfloat16),
    )


@triton.jit(do_not_specialize=["M"])
def _fp8_1x128_dequant_ts(
    p_ptr, o_ptr, st_ptr, M, H: tl.constexpr, ROWB: tl.constexpr, GB: tl.constexpr
):
    # _fp8_1x128_dequant plus the row's fp32 scales into st[g * M + row] for
    # rows < M: the column-major [H/128, M] layout AITER per_1x128 quant writes
    # with transpose_scale=True.
    row = tl.program_id(0)
    gb = tl.program_id(1)
    g = gb * GB + tl.arange(0, GB)
    c = tl.arange(0, 128)
    base = p_ptr + row.to(tl.int64) * ROWB
    q = tl.load(
        base.to(tl.pointer_type(tl.float8e4nv), bitcast=True)
        + g[:, None] * 128
        + c[None, :]
    ).to(tl.float32)
    s = tl.load((base + H).to(tl.pointer_type(tl.float32), bitcast=True) + g)
    tl.store(
        o_ptr + row.to(tl.int64) * H + g[:, None] * 128 + c[None, :],
        (q * s[:, None]).to(tl.bfloat16),
    )
    tl.store(st_ptr + g.to(tl.int64) * M + row, s, mask=(g * 0 + row) < M)


def packed_row_bytes(hidden: int) -> int:
    return hidden + (hidden // _GROUP) * 4


def fp8_1x128_quant_pack(x: torch.Tensor) -> torch.Tensor:
    """bf16 [m, H] -> uint8 [m, H + H/128*4] (fp8 e4m3 codes, then fp32 scales)."""
    m, h = x.shape
    if not x.is_contiguous() or h % (_GROUP * _GB) != 0:
        raise ValueError(f"fp8_1x128_quant_pack: bad input {tuple(x.shape)}")
    out = torch.empty(m, packed_row_bytes(h), dtype=torch.uint8, device=x.device)
    if m:
        _fp8_1x128_quant_pack[(m, h // _GROUP // _GB)](
            x, out, h, out.shape[1], _GB, num_warps=4
        )
    return out


def fp8_1x128_dequant(
    p: torch.Tensor, hidden: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    m, rowb = p.shape
    if rowb < packed_row_bytes(hidden) or not p.is_contiguous():
        raise ValueError(f"fp8_1x128_dequant: bad input {tuple(p.shape)}")
    if out is None:
        out = torch.empty(m, hidden, dtype=torch.bfloat16, device=p.device)
    if m:
        _fp8_1x128_dequant[(m, hidden // _GROUP // _GB)](
            p, out, hidden, rowb, _GB, num_warps=4
        )
    return out


# --- fp8 handoff: the gathered fp8 codes go straight to the fp8 GEMMs ----------
#
# The layer-input gather carries fp8 e4m3 codes with fp32 1x128 scales. The bf16
# rows are dequantized for every consumer, but fused_wqa_wkv and the shared
# expert's gate_up would requantize them per 1x128 (AITER, transpose_scale=True)
# to exactly the gathered codes and scales. Those two GEMMs take the codes (a
# strided view of the gathered buffer) and the scales (written transposed by the
# dequant) directly instead.

SP_FP8_HANDOFF_LABEL = "sp_fp8_handoff"


def _form_key(x: torch.Tensor) -> tuple:
    ver = None if x.is_inference() else x._version
    return (x.data_ptr(), tuple(x.shape), x.stride(), ver)


class _Fp8Form:
    """The fp8 form of the current sublayer's gathered bf16 input (one at a time).

    Matched by identity of the bf16 tensor (storage pointer, shape, strides, and
    the version counter where autograd tracks one). Under inference mode tensors
    carry no version counter: there the guard is the scope (registered by the
    sublayer's gather, cleared after the sublayer, with no in-place write to the
    gathered rows in between). Holds references to the bf16 rows and the
    gathered buffer so neither storage is reused while registered.
    """

    x: torch.Tensor | None = None
    key: tuple | None = None
    codes: torch.Tensor | None = None
    scales_t: torch.Tensor | None = None
    checked: set[str] = set()

    @classmethod
    def set(cls, x: torch.Tensor, codes: torch.Tensor, scales_t: torch.Tensor) -> None:
        cls.x = x
        cls.key = _form_key(x)
        cls.codes = codes
        cls.scales_t = scales_t

    @classmethod
    def clear(cls) -> None:
        cls.x = cls.key = cls.codes = cls.scales_t = None


def sp_fp8_handoff_clear() -> None:
    _Fp8Form.clear()


def sp_fp8_form(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(fp8 codes [M, H] row-strided, scales [M, H/128] in transposed memory)
    of x, or None.

    Returns the pair only for the registered gathered layer input itself;
    callers fall back to their own per-1x128 quant otherwise.
    """
    if _Fp8Form.key is None or _form_key(x) != _Fp8Form.key:
        return None
    assert _Fp8Form.codes is not None and _Fp8Form.scales_t is not None
    return _Fp8Form.codes, _Fp8Form.scales_t


def sp_fp8_handoff_check(kind: str, x: torch.Tensor, out: torch.Tensor, served) -> None:
    """With dsv4_sp_self_check, once per consumer kind per rank: the handoff GEMM
    output must equal the requant + GEMM path bitwise. Raises on a mismatch."""
    if not _SELF_CHECK or kind in _Fp8Form.checked:
        return
    _Fp8Form.checked.add(kind)
    same = bool(torch.equal(out, served(x)))
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger.info(
        "%s: fp8 gather codes handed to %s on rank %d (%d rows, no requant); "
        "bitwise equal to the requant path: %s",
        SP_FP8_HANDOFF_LABEL,
        kind,
        rank,
        x.shape[0],
        same,
    )
    if not same:
        raise RuntimeError(
            f"{SP_FP8_HANDOFF_LABEL}: {kind} output differs from the requant path"
        )


class SeqParallelSeamComm:
    """Per-TP-group state: the AITER QuickReduce reduce-scatter and the fp8 gather.

    The reduce-scatter runs on an AITER QuickReduce communicator owned by this
    object (vLLM's TP all-reduce uses its own QuickReduce copy, a different
    communicator type). Calls must be issued in the same order on every rank;
    any sequence of QuickReduce collectives may follow each other.
    """

    def __init__(self, tp_group, max_rows: int, hidden: int):
        import aiter as aiter_ops
        from aiter.dist.device_communicators.quick_all_reduce import (
            qr_exchange_handles,
        )
        from aiter.ops.quick_reduce_scatter import qr_reduce_scatter

        self.tp = tp_group
        self.world = tp_group.world_size
        if self.world != _WS:
            raise ValueError(f"SeqParallelSeamComm needs TP{_WS}, got {self.world}")
        self.rank = tp_group.rank_in_group
        self.hidden = hidden
        self.max_rows = max_rows
        self.row_align = self.row_alignment(hidden)
        self._qr_rs = qr_reduce_scatter
        ca = getattr(tp_group.device_communicator, "aiter_ar_comm", None)
        if ca is None or ca.disabled:
            raise RuntimeError(
                "SeqParallelSeamComm needs an enabled AITER custom all-reduce"
            )
        self._ca = self._gather_ca(tp_group, ca, max_rows, hidden)
        # The QuickReduce communicator, sized for the largest reduce-scatter
        # input (bf16 rows).
        self._aiter = aiter_ops
        self._qr = aiter_ops.init_custom_qr(
            self.rank, self.world, max_rows * hidden * 2
        )
        qr_exchange_handles(self._qr, self.world, tp_group.cpu_group)
        dist.barrier(group=tp_group.cpu_group)
        logger.info(
            "SeqParallelSeamComm rank %d: AITER QuickReduce INT4 reduce-scatter, "
            "max_rows %d, fp8 1x128 gather via %s",
            self.rank,
            max_rows,
            type(self._ca).__name__,
        )

    def _gather_ca(self, tp_group, ca, max_rows: int, hidden: int):
        """The AITER custom all-gather the fp8 layer-input gathers run on.

        The TP group's AITER custom all-reduce is sized for all-reduce. A
        sequence-parallel step gathers max_rows / 8 packed rows per rank, so
        with a large --max-num-batched-tokens the shared buffer refuses the
        gather; then this object owns a gather-only instance sized for
        max_rows (never graph-captured: SP steps run eager).
        """
        need = (max_rows // _WS) * packed_row_bytes(hidden)
        # Ask AITER itself: its cutoff (AITER_CUSTOM_AG_MAX_SIZE, default
        # max_size / (2 * world_size)) and peer connectivity decide. The answer
        # is the same on every rank.
        probe = torch.empty(need // 2, dtype=torch.bfloat16, device=tp_group.device)
        if ca.should_custom_ag(probe):
            return ca
        from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
            AiterCustomAllreduce,
        )

        mib = 1 << 20
        max_size = (need * self.world * 2 + mib - 1) // mib * mib
        own = AiterCustomAllreduce(
            group=tp_group.cpu_group,
            device=tp_group.device,
            max_size=max_size,
            register_graph_buffers=False,
        )
        if own.disabled or not own.should_custom_ag(probe):
            raise RuntimeError(
                "SeqParallelSeamComm: AITER custom all-gather cannot take "
                f"{need} bytes per rank (instance disabled, or capped by "
                "AITER_CUSTOM_AG_MAX_SIZE)"
            )
        logger.info(
            "SeqParallelSeamComm rank %d: own AITER custom all-gather buffer of "
            "%d MiB for %d-row steps",
            self.rank,
            max_size // mib,
            max_rows,
        )
        return own

    @staticmethod
    def row_alignment(hidden: int) -> int:
        # Each rank's segment numel (Mp/8 * hidden) must be a multiple of 2048.
        return _WS * (2048 // math.gcd(hidden, 2048))

    def reduce_scatter(self, x: torch.Tensor, mp: int) -> torch.Tensor:
        """x: this rank's partial sum, bf16 [M, H] -> reduced rows [Mp/8, H].

        mp: the padded row count (a multiple of row_align, >= M). Rows [M, Mp)
        read as zero inside the kernel, so no padded copy of x is made.
        """
        m, h = x.shape
        if h != self.hidden or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError(
                f"reduce_scatter: bad input {tuple(x.shape)} {x.dtype} "
                f"contiguous={x.is_contiguous()}"
            )
        if not (0 < m <= mp and mp % self.row_align == 0 and mp <= self.max_rows):
            raise ValueError(
                f"reduce_scatter: rows {m} -> {mp} (align {self.row_align}, "
                f"max {self.max_rows})"
            )
        out = torch.empty(mp // _WS, h, dtype=x.dtype, device=x.device)
        self._qr_rs(self._qr, x, out, True)
        return out

    def gather_fp8(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """x: this rank's rows [Mp/8, H] bf16 -> [Mp, H] bf16, fp8 e4m3 1x128 on
        the wire. Registers the first m rows' fp8 form for the fp8 GEMMs."""
        h = self.hidden
        packed = fp8_1x128_quant_pack(x)
        pk = packed.view(torch.bfloat16)
        if not self._ca.should_custom_ag(pk):
            raise RuntimeError(f"AITER custom all-gather refuses {tuple(pk.shape)}")
        gat = self._ca.custom_all_gather(pk).view(torch.uint8).view(-1, packed.shape[1])
        mp, rowb = gat.shape
        assert 0 < m <= mp
        out = torch.empty(mp, h, dtype=torch.bfloat16, device=gat.device)
        st = torch.empty(h // _GROUP, m, dtype=torch.float32, device=gat.device)
        _fp8_1x128_dequant_ts[(mp, h // _GROUP // _GB)](
            gat, out, st, m, h, rowb, _GB, num_warps=4
        )
        rows = out[:m]
        _Fp8Form.set(
            rows, gat[:m, :h].view(torch.float8_e4m3fn), st.view(m, h // _GROUP)
        )
        return rows

    def close(self):
        if getattr(self, "_qr", None):
            self._aiter.qr_destroy(self._qr)
            self._qr = None


def sp_config_disable_reasons(vllm_config) -> list[str]:
    """Deployment-level reasons sequence parallelism cannot engage (empty: it
    can). Model-structure checks (mHC fusion, MegaMoE) stay in the model."""
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
        # The MTP hidden buffer is not written on sequence-parallel steps;
        # DSpark reads the aux hidden states instead.
        reasons.append(f"speculative method {spec.method}")
    # The only shape the AITER reduce-scatter and the fp8 seams were tuned on.
    if vllm_config.model_config.hf_config.hidden_size != 7168:
        reasons.append("hidden size")
    # The fp8 seams use OCP e4m3.
    if not on_gfx950():
        reasons.append("the device is not gfx950")
    # SP reduce-scatters on the INT4 quick-reduce wire with the bf16->fp16
    # cast: engage only when the deployment chose exactly that for its
    # all-reduces.
    if envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION != "INT4":
        reasons.append("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION is not INT4")
    if not envs.VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:
        reasons.append("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0")
    ca = getattr(get_tp_group().device_communicator, "aiter_ar_comm", None)
    if ca is None or ca.disabled:
        reasons.append("no enabled AITER custom all-reduce")
    if pc.enable_dbo:
        reasons.append("dual-batch overlap")
    if pc.decode_context_parallel_size != 1:
        reasons.append("decode context parallel")
    if pc.enable_expert_parallel:
        reasons.append("expert parallel")
    return reasons


class SeqParallelStep:
    """One forward step's row split: M real rows padded to
    Mp = ceil(M / align) * align.

    Rank r owns rows [r * Mp/8, (r+1) * Mp/8). Rows >= M exist only in the last
    rank's segment; they are zero at entry, every seam op is row-independent,
    and the gathers drop them.
    """

    def __init__(self, comm: SeqParallelSeamComm, m: int):
        self.comm = comm
        self.m = m
        a = comm.row_align
        self.mp = (m + a - 1) // a * a
        self.s = self.mp // _WS
        self.start = comm.rank * self.s

    def rows(self, t: torch.Tensor) -> torch.Tensor:
        """Full-row tensor [M, ...] -> this rank's [Mp/8, ...] rows (zero tail)."""
        end = min(self.start + self.s, self.m)
        n = max(end - self.start, 0)
        if n == self.s:
            return t[self.start : end].contiguous()
        out = t.new_zeros((self.s,) + tuple(t.shape[1:]))
        if n:
            out[:n].copy_(t[self.start : end])
        return out

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's partial sum [M, H] -> reduced rows [Mp/8, H] (INT4 wire,
        as the TP all-reduce)."""
        assert x.shape[0] == self.m, (x.shape, self.m)
        return self.comm.reduce_scatter(x.contiguous(), self.mp)

    def gather(self, xs: torch.Tensor) -> torch.Tensor:
        """Layer input rows [Mp/8, H] -> [M, H] on every rank (fp8 1x128 wire)."""
        return self.comm.gather_fp8(xs.contiguous(), self.m)

    def gather_exact(self, xs: torch.Tensor) -> torch.Tensor:
        """[Mp/8, H] -> [M, H], exact bf16 (DSpark aux means and the head output)."""
        return self.comm.tp.all_gather(xs.contiguous(), dim=0)[: self.m]
