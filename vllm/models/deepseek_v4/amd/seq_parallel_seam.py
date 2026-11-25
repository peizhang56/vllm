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

With `dsv4_sp_moe` the MoE of these steps also works on the rank's own rows: each
rank routes its rows and the routing rides in the FFN input gather, the routed
experts take the gathered fp8 rows as MXFP8 codes, a full copy of the shared
expert runs on the rank's own rows, and the routed experts' per-slot output is
summed inside the reduce-scatter (AITER qr_reduce_scatter_route).
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
# Whether the first route-folded reduce-scatter was logged (and self-checked).
_ROUTE_FOLD_LOGGED = False


def set_sp_self_check(enabled: bool) -> None:
    global _SELF_CHECK
    _SELF_CHECK = enabled


def sp_self_check_enabled() -> bool:
    return _SELF_CHECK


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


def fp8_1x128_quant_pack(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """bf16 [m, H] -> uint8 [m, H + H/128*4] (fp8 e4m3 codes, then fp32 scales).

    `out` may be wider than packed_row_bytes(H); the bytes past it are left to
    the caller (a per-row tail that travels in the same gather).
    """
    m, h = x.shape
    if not x.is_contiguous() or h % (_GROUP * _GB) != 0:
        raise ValueError(f"fp8_1x128_quant_pack: bad input {tuple(x.shape)}")
    if out is None:
        out = torch.empty(m, packed_row_bytes(h), dtype=torch.uint8, device=x.device)
    elif (
        out.dtype != torch.uint8
        or not out.is_contiguous()
        or out.shape[0] != m
        or out.shape[1] < packed_row_bytes(h)
    ):
        raise ValueError(f"fp8_1x128_quant_pack: bad output {tuple(out.shape)}")
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
    # Whether a consumer took the codes since set().
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
    _Fp8Form.clear()
    _MoeForm.clear()


def sp_fp8_form(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(fp8 codes [M, H] row-strided, scales [M, H/128] in transposed memory)
    of x, or None.

    Returns the pair only for the registered gathered layer input itself;
    callers fall back to their own per-1x128 quant otherwise.
    """
    if _Fp8Form.key is None or _form_key(x) != _Fp8Form.key:
        return None
    assert _Fp8Form.codes is not None and _Fp8Form.scales_t is not None
    _Fp8Form.used = True
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


# --- dsv4_sp_moe: the MoE of sequence-parallel steps -----------------------------
#
# Router shard: each rank routes its own rows of the FFN input before the gather,
# and the routing (fp32 weights and int32 ids, 8 * topk bytes per row) rides as a
# byte tail in the same fp8 gather.
#
# MoE codes: with the router shard, the gathered bf16 FFN rows are read only by
# the shared expert's gate_up (which takes the fp8 1x128 handoff) and by AITER
# fused_moe, which from 256 rows quantizes them to MXFP8 (1x32 e8m0). On steps of
# at least SP_MOE_CODES_MIN_ROWS rows one kernel writes the 1x128 scales and the
# MXFP8 codes and e8m0 straight from the gathered packed rows, fused_moe takes its
# fp8 passthrough, and the bf16 rows are an unwritten placeholder (the identity
# key of both handoffs). Bitwise equal to the dequant + quant they replace.
#
# Route fold: the routed experts' OPUS stage 2 hands its per-slot MXFP8 rows to
# AITER's route-output capture instead of reducing them, and
# qr_reduce_scatter_route sums the topk slots on load (bitwise equal to the slot
# reduce followed by the reduce-scatter). Only with the row-local shared expert,
# which keeps the shared term out of the routed output.

SP_MOE_LABEL = "dsv4_sp_moe"
SP_ROUTER_SHARD_LABEL = "sp_router_shard"
SP_MOE_CODES_LABEL = "sp_moe_codes"
SP_ROUTE_FOLD_LABEL = "sp_route_fold"
SP_SHARED_ROWLOCAL_LABEL = "sp_shared_rowlocal"

# AITER fused_moe quantizes its input to MXFP8 from 256 rows (8 * 256 / topk
# with the split quant + sort), and its tuned stage 2 kernels are non-atomic
# from 2048 rows on, so the codes path (and its self-check) is deterministic.
SP_MOE_CODES_MIN_ROWS = 2048

# Room per row for a byte tail in the layer-input gather buffer (the routing).
GATHER_TAIL_BYTES = 64


def routing_tail_bytes(topk: int) -> int:
    """Gather tail bytes for the routing of one row: fp32 weights and int32
    ids, padded to 16 bytes so the packed rows stay 16-byte aligned."""
    return (8 * topk + 15) // 16 * 16


@triton.jit(do_not_specialize=["M"])
def _fp8_1x128_codes_mx(
    p_ptr,
    st_ptr,
    q_ptr,
    e_ptr,
    M,
    H: tl.constexpr,
    ROWB: tl.constexpr,
    GB: tl.constexpr,
):
    # Packed gathered row [H fp8 | H/128 fp32 | tail] -> st[g * M + row] (the
    # 1x128 scales, column-major as _fp8_1x128_dequant_ts writes them), plus
    # MXFP8 codes q[row, :] and e8m0 bytes e[row, k] of each 1x32 group k,
    # computed from the bf16 value the dequant stores (AITER per_1x32 RoundUp:
    # e = exponent of amax / 448 rounded up; an all-zero group gets 85).
    row = tl.program_id(0)
    gb = tl.program_id(1)
    k = gb * GB * 4 + tl.arange(0, GB * 4)
    j = tl.arange(0, 32)
    base = p_ptr + row.to(tl.int64) * ROWB
    q = tl.load(
        base.to(tl.pointer_type(tl.float8e4nv), bitcast=True)
        + k[:, None] * 32
        + j[None, :]
    ).to(tl.float32)
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
    tl.store(
        qrow.to(tl.pointer_type(tl.float8e4nv), bitcast=True)
        + k[:, None] * 32
        + j[None, :],
        codes,
    )
    tl.store(e_ptr + row.to(tl.int64) * (H // 32) + k, e.to(tl.uint8))


class _MoeForm:
    """MXFP8 codes and e8m0 of the current FFN sublayer's gathered input (the
    codes-only path), matched like _Fp8Form."""

    key: tuple | None = None
    x: torch.Tensor | None = None
    codes: torch.Tensor | None = None
    e8m0: torch.Tensor | None = None
    used: bool = False
    # Self-check state: the codes, and the routed-MoE output, once per rank.
    checked: bool = False
    moe_checked: bool = False
    logged: bool = False

    @classmethod
    def set(cls, x: torch.Tensor, codes: torch.Tensor, e8m0: torch.Tensor) -> None:
        cls.x, cls.key, cls.codes, cls.e8m0 = x, _form_key(x), codes, e8m0
        cls.used = False

    @classmethod
    def clear(cls) -> None:
        cls.x = cls.key = cls.codes = cls.e8m0 = None
        cls.used = False


def sp_moe_codes_wanted(m: int) -> bool:
    return m >= SP_MOE_CODES_MIN_ROWS


def sp_moe_codes_form(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool] | None:
    """(MXFP8 codes [M, H], e8m0 [M, H/32], first) for the registered gathered
    FFN input x, else None. `first`: the self-checked call on this rank."""
    if _MoeForm.key is None or _form_key(x) != _MoeForm.key:
        return None
    assert _MoeForm.codes is not None and _MoeForm.e8m0 is not None
    _MoeForm.used = True
    first = _SELF_CHECK and not _MoeForm.moe_checked
    _MoeForm.moe_checked = True
    return _MoeForm.codes, _MoeForm.e8m0, first


def _bytes(t: torch.Tensor) -> torch.Tensor:
    # Bit patterns, not values: a NaN in a dummy step must not read as a
    # mismatch.
    return t.contiguous().view(torch.uint8)


def sp_moe_codes_check(out: torch.Tensor, ref: torch.Tensor, m: int) -> None:
    same = bool(torch.equal(_bytes(out), _bytes(ref)))
    logger.info(
        "%s: routed-MoE input codes-only on rank %d (%d rows, AITER fused_moe "
        "fp8 passthrough); MoE output bitwise equal to the bf16 path: %s",
        SP_MOE_CODES_LABEL,
        dist.get_rank() if dist.is_initialized() else 0,
        m,
        same,
    )
    if not same:
        raise RuntimeError(
            f"{SP_MOE_CODES_LABEL}: routed-MoE output differs from the bf16 path"
        )


def sp_moe_codes_done(shared_reads_rows: bool) -> None:
    """After the FFN sublayer of a codes-only step: every consumer of the
    placeholder rows must have taken the codes instead. With the row-local
    shared expert (shared_reads_rows=False) only the routed MoE reads them."""
    if _MoeForm.key is None:
        return
    if not (_MoeForm.used and (_Fp8Form.used or not shared_reads_rows)):
        raise RuntimeError(
            f"{SP_MOE_CODES_LABEL}: a consumer read the placeholder rows (routed "
            f"MoE took codes: {_MoeForm.used}, shared expert took codes: "
            f"{_Fp8Form.used})"
        )


def sp_fused_moe_input_hook(hidden_states: torch.Tensor, run) -> torch.Tensor | None:
    """rocm_aiter_fused_experts input hook (set_rocm_aiter_fused_moe_input_hook):
    runs fused_moe on the MXFP8 codes of a registered codes-only FFN input."""
    form = sp_moe_codes_form(hidden_states)
    if form is None:
        return None
    codes, e8m0, first = form
    if not first:
        return run(codes, e8m0)
    # The self-check compares reduced outputs: run both with the slot reduce
    # (the route-output capture suspended), then once more to capture if armed.
    with sp_route_capture_suspended():
        out = run(codes, e8m0)
        sp_moe_codes_check(out, run(hidden_states, None), hidden_states.shape[0])
    if sp_route_capture_armed():
        out = run(codes, e8m0)
    return out


def _route_capture():
    from aiter.ops.opus.moe_stage2_a8w4 import route_output_capture

    return route_output_capture()


class sp_route_capture:
    """Arms AITER's route-output capture around the routed MoE of one SP FFN
    sublayer (no-op unless enabled). On exit `route` holds (slots, topk) of
    the captured OPUS stage 2 per-slot rows, or None (that stage 2 reduced as
    usual: atomic or bf16 route output)."""

    def __init__(self, m: int, enabled: bool):
        self.m = m
        self.enabled = enabled
        self.route: tuple[torch.Tensor, int] | None = None

    def __enter__(self):
        if self.enabled:
            cap = _route_capture()
            if cap.armed or cap.route_out is not None:
                raise RuntimeError(f"{SP_ROUTE_FOLD_LABEL}: capture already armed")
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
                        f"{SP_ROUTE_FOLD_LABEL}: captured {token_num} rows, the "
                        f"step has {self.m}"
                    )
                self.route = (slots, topk)
        return False


class sp_route_capture_suspended:
    """Disarms the capture for a nested fused_moe that must reduce (a
    self-check)."""

    def __enter__(self):
        self.prev = sp_route_capture_armed()
        if self.prev:
            _route_capture().armed = False
        return self

    def __exit__(self, *exc):
        if self.prev:
            _route_capture().armed = True
        return False


def sp_route_capture_armed() -> bool:
    try:
        return _route_capture().armed
    except ImportError:
        return False


def _sp_route_fold_check(folded: torch.Tensor, ref: torch.Tensor, m: int) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    same = bool(torch.equal(_bytes(folded), _bytes(ref)))
    logger.info(
        "%s self-check on rank %d (%d rows): slot-folded reduce-scatter bitwise "
        "equal to the slot reduce + reduce-scatter: %s",
        SP_ROUTE_FOLD_LABEL,
        rank,
        m,
        same,
    )
    if not same:
        raise RuntimeError(
            f"{SP_ROUTE_FOLD_LABEL}: slot-folded reduce-scatter differs on rank {rank}"
        )


class SeqParallelSeamComm:
    """Per-TP-group state: the AITER QuickReduce reduce-scatter and the fp8 gather.

    The reduce-scatter runs on an AITER QuickReduce communicator owned by this
    object (vLLM's TP all-reduce uses its own QuickReduce copy, a different
    communicator type). Calls must be issued in the same order on every rank;
    any sequence of QuickReduce collectives may follow each other.
    """

    def __init__(
        self,
        tp_group,
        max_rows: int,
        hidden: int,
        gather_tail_bytes: int = 0,
        route_fold: bool = False,
    ):
        """gather_tail_bytes: room per row for a byte tail in the layer-input
        gather (the dsv4_sp_moe routing). route_fold: use AITER
        qr_reduce_scatter_route when this AITER provides it."""
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
        self.gather_tail_bytes = gather_tail_bytes
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
        self._qr_rs_route = self._route_fold_kernel() if route_fold else None

    def _route_fold_kernel(self):
        """AITER qr_reduce_scatter_route if this AITER has it and the route-output
        capture, else None (the routed MoE then reduces its slots itself). Decided
        once, the same way on every rank."""
        reason = None
        try:
            from aiter.ops.opus.moe_stage2_a8w4 import route_output_capture
            from aiter.ops.quick_reduce_scatter import qr_reduce_scatter_route

            cap = route_output_capture()
            if not all(hasattr(cap, a) for a in ("armed", "route_out", "take")):
                raise ImportError("route_output_capture lacks armed/route_out/take")
            # A zero-row call loads (or JIT-builds) the module and resolves the
            # entry point, so a prebuilt module without it shows up here, not
            # mid-serve. It launches nothing.
            h = self.hidden
            dev = self.tp.device
            qr_reduce_scatter_route(
                self._qr,
                torch.empty(0, h + h // 8, dtype=torch.uint8, device=dev),
                torch.empty(0, dtype=torch.bfloat16, device=dev),
                0,
                1,
                h,
            )
        except Exception as e:  # any failure: fall back to the slot reduce
            reason = f"{type(e).__name__}: {e}"
        ok = torch.tensor([reason is None], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=self.tp.cpu_group)
        if ok.item():
            logger.info(
                "%s on rank %d: routed MoE slot reduce folded into the "
                "reduce-scatter (AITER qr_reduce_scatter_route)",
                SP_ROUTE_FOLD_LABEL,
                self.rank,
            )
            return qr_reduce_scatter_route
        logger.warning(
            "%s off on rank %d: %s; the routed MoE reduces its slots itself",
            SP_ROUTE_FOLD_LABEL,
            self.rank,
            reason or "unavailable on another rank",
        )
        return None

    @property
    def route_fold(self) -> bool:
        return self._qr_rs_route is not None

    def _gather_ca(self, tp_group, ca, max_rows: int, hidden: int):
        """The AITER custom all-gather the fp8 layer-input gathers run on.

        The TP group's AITER custom all-reduce is sized for all-reduce. A
        sequence-parallel step gathers max_rows / 8 packed rows per rank, so
        with a large --max-num-batched-tokens the shared buffer refuses the
        gather; then this object owns a gather-only instance sized for
        max_rows (never graph-captured: SP steps run eager).
        """
        need = (max_rows // _WS) * (packed_row_bytes(hidden) + self.gather_tail_bytes)
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

    def reduce_scatter(
        self,
        x: torch.Tensor,
        mp: int,
        route: tuple[torch.Tensor, int] | None = None,
    ) -> torch.Tensor:
        """x: this rank's partial sum, bf16 [M, H] -> reduced rows [Mp/8, H].

        mp: the padded row count (a multiple of row_align, >= M). Rows [M, Mp)
        read as zero inside the kernel, so no padded copy of x is made.
        route: (slots, topk), the routed MoE's OPUS stage 2 per-slot MXFP8 rows
        [M * topk, H + H/8] uint8. They replace x (unwritten; only its shape is
        used) and are summed on load as the OPUS slot reduce does, bitwise.
        """
        m, h = x.shape
        dense_ok = route is not None or x.is_contiguous()
        if h != self.hidden or x.dtype != torch.bfloat16 or not dense_ok:
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
        if route is None:
            self._qr_rs(self._qr, x, out, True)
            return out
        slots, topk = route
        if self._qr_rs_route is None:
            raise RuntimeError("reduce_scatter: no route-fold kernel")
        if (
            slots.dtype != torch.uint8
            or not slots.is_contiguous()
            or slots.device != x.device
            or tuple(slots.shape) != (m * topk, h + h // 8)
        ):
            raise ValueError(
                f"reduce_scatter: bad route slots {tuple(slots.shape)} "
                f"{slots.dtype} for {m} rows, topk {topk}"
            )
        self._qr_rs_route(self._qr, slots, out, m, int(topk), h)
        return out

    def _all_gather_packed(self, packed: torch.Tensor) -> torch.Tensor:
        pk = packed.view(torch.bfloat16)
        if not self._ca.should_custom_ag(pk):
            raise RuntimeError(f"AITER custom all-gather refuses {tuple(pk.shape)}")
        gat = self._ca.custom_all_gather(pk)
        return gat.view(torch.uint8).view(-1, packed.shape[1])

    def _dequant(self, gat: torch.Tensor, m: int) -> torch.Tensor:
        """Gathered packed rows [Mp, rowb] -> bf16 rows [m, H]; registers their
        fp8 form for the fp8 GEMMs."""
        h = self.hidden
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

    def _codes_only(self, gat: torch.Tensor, m: int) -> torch.Tensor:
        """Gathered packed rows -> a placeholder bf16 [m, H] (left unwritten),
        registered with the fp8 1x128 handoff and the MXFP8 MoE codes of the
        same rows."""
        h = self.hidden
        mp, rowb = gat.shape
        assert 0 < m <= mp
        rows = torch.empty(m, h, dtype=torch.bfloat16, device=gat.device)
        st = torch.empty(h // _GROUP, m, dtype=torch.float32, device=gat.device)
        y = torch.empty(m, h, dtype=torch.float8_e4m3fn, device=gat.device)
        e = torch.empty(m, h // 32, dtype=torch.uint8, device=gat.device)
        _fp8_1x128_codes_mx[(m, h // _GROUP // _GB)](
            gat, st, y, e, m, h, rowb, _GB, num_warps=2
        )
        if not _MoeForm.logged:
            _MoeForm.logged = True
            logger.info(
                "%s: routed experts take the gathered fp8 rows as MXFP8 codes on "
                "rank %d (%d rows, no bf16 rows written)",
                SP_MOE_CODES_LABEL,
                self.rank,
                m,
            )
        if _SELF_CHECK and not _MoeForm.checked:
            # Once per rank: write the bf16 rows (so this sublayer's consumers
            # and their own self-checks read real rows) and check the codes
            # against the dequant + AITER per_1x32 quant they replace.
            _MoeForm.checked = True
            from aiter import dtypes
            from aiter.ops.quant import per_1x32_mx_quant_hip

            st2 = torch.empty_like(st)
            _fp8_1x128_dequant_ts[(m, h // _GROUP // _GB)](
                gat, rows, st2, m, h, rowb, _GB, num_warps=4
            )
            y2, e2 = per_1x32_mx_quant_hip(
                rows, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
            )
            same = (
                torch.equal(_bytes(st), _bytes(st2))
                and torch.equal(_bytes(y), _bytes(y2))
                and torch.equal(e, _bytes(e2))
            )
            logger.info(
                "%s: codes-only MoE input on rank %d (%d rows): 1x128 scales, "
                "MXFP8 codes and e8m0 bitwise equal to the dequant + per_1x32 "
                "quant: %s",
                SP_MOE_CODES_LABEL,
                self.rank,
                m,
                same,
            )
            if not same:
                raise RuntimeError(f"{SP_MOE_CODES_LABEL}: codes-only input differs")
        _Fp8Form.set(
            rows, gat[:m, :h].view(torch.float8_e4m3fn), st.view(m, h // _GROUP)
        )
        _MoeForm.set(rows, y, e.view(torch.float8_e8m0fnu))
        return rows

    def gather_fp8(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """x: this rank's rows [Mp/8, H] bf16 -> [m, H] bf16, fp8 e4m3 1x128 on
        the wire. Registers the rows' fp8 form for the fp8 GEMMs."""
        return self._dequant(self._all_gather_packed(fp8_1x128_quant_pack(x)), m)

    def gather_fp8_with_tail(
        self, x: torch.Tensor, tail: torch.Tensor, m: int, codes_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """gather_fp8 plus a per-row byte tail in the same gather.

        x: [S, H] bf16, tail: [S, E] uint8 (copied exactly). Returns the [m, H]
        bf16 rows (a placeholder with codes_only) and the gathered [Mp, E] tail.
        """
        s, e = tail.shape
        if tail.dtype != torch.uint8 or x.shape[0] != s or e > self.gather_tail_bytes:
            raise ValueError(
                f"gather_fp8_with_tail: bad tail {tuple(tail.shape)} {tail.dtype} "
                f"(room {self.gather_tail_bytes} bytes)"
            )
        rowb = packed_row_bytes(self.hidden)
        packed = torch.empty(s, rowb + e, dtype=torch.uint8, device=x.device)
        fp8_1x128_quant_pack(x, packed)
        packed[:, rowb:].copy_(tail)
        gat = self._all_gather_packed(packed)
        rows = self._codes_only(gat, m) if codes_only else self._dequant(gat, m)
        return rows, gat[:, rowb:]

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

    def reduce_scatter(
        self, x: torch.Tensor, route: tuple[torch.Tensor, int] | None = None
    ) -> torch.Tensor:
        """This rank's partial sum [M, H] -> reduced rows [Mp/8, H] (INT4 wire,
        as the TP all-reduce).

        route: (slots, topk) of the routed MoE's captured per-slot rows; x is
        then unwritten (only its shape is used) and the slots are summed on
        load."""
        assert x.shape[0] == self.m, (x.shape, self.m)
        if route is None:
            return self.comm.reduce_scatter(x.contiguous(), self.mp)
        global _ROUTE_FOLD_LOGGED
        first = not _ROUTE_FOLD_LOGGED
        if first:
            _ROUTE_FOLD_LOGGED = True
            logger.info(
                "%s: routed MoE slot reduce folded into the INT4 reduce-scatter "
                "on rank %d (%d rows x %d MXFP8 slots)",
                SP_ROUTE_FOLD_LABEL,
                self.comm.rank,
                self.m,
                route[1],
            )
        out = self.comm.reduce_scatter(x, self.mp, route=route)
        if first and _SELF_CHECK:
            from aiter.ops.opus.moe_stage2_a8w4 import (
                opus_moe_stage2_reduce_token_slot_route_output_fwd,
            )

            slots, topk = route
            ref = opus_moe_stage2_reduce_token_slot_route_output_fwd(slots, topk=topk)
            _sp_route_fold_check(out, self.comm.reduce_scatter(ref, self.mp), self.m)
        return out

    def gather(self, xs: torch.Tensor) -> torch.Tensor:
        """Layer input rows [Mp/8, H] -> [M, H] on every rank (fp8 1x128 wire)."""
        return self.comm.gather_fp8(xs.contiguous(), self.m)

    def gather_with_routing(
        self,
        xs: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        codes_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Layer input rows [Mp/8, H] and their MoE routing ([Mp/8, K] weights
        and ids) -> [M, H] bf16 (fp8 wire; a placeholder with codes_only), and
        [M, K] weights and ids (exact), in one gather."""
        s, k = weights.shape
        if ids.shape != (s, k) or xs.shape[0] != s:
            raise ValueError(
                f"gather_with_routing: rows {tuple(xs.shape)}, weights "
                f"{tuple(weights.shape)}, ids {tuple(ids.shape)}"
            )
        e = routing_tail_bytes(k)
        tail = torch.zeros(s, e // 4, dtype=torch.int32, device=xs.device)
        tail[:, :k] = weights.to(torch.float32).view(torch.int32)
        tail[:, k : 2 * k] = ids.to(torch.int32)
        x, t = self.comm.gather_fp8_with_tail(
            xs.contiguous(), tail.view(torch.uint8), self.m, codes_only
        )
        t = t[: self.m].contiguous().view(torch.int32)
        # The MoE kernels read the routing as dense [M, K] tensors.
        w = t[:, :k].contiguous().view(torch.float32).to(weights.dtype)
        i = t[:, k : 2 * k].contiguous().to(ids.dtype)
        return x, w, i

    def gather_exact(self, xs: torch.Tensor) -> torch.Tensor:
        """[Mp/8, H] -> [M, H], exact bf16 (DSpark aux means and the head output)."""
        return self.comm.tp.all_gather(xs.contiguous(), dim=0)[: self.m]
