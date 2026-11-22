# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.attention import active_attention_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rocm_paged_mxfp4_indexer import RocmSparseAttnIndexer
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.models.deepseek_v4.amd.seq_parallel_seam import (
    SP_COMPRESSOR_SHARD_LABEL,
    CompressorShardPlan,
    SeqParallelStep,
    _CacheRows,
    fp8_1x128_dequant,
    fp8_1x128_quant_pack,
    sp_config_disable_reasons,
    sp_fp8_handoff_clear,
    sp_self_check_enabled,
)
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _pick_compress_num_splits,
    compress_norm_rope_store_triton,
    compress_norm_rope_store_two_stage_triton,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    _SAVE_PARTIAL_STATES_KERNEL,
)
from vllm.models.deepseek_v4.compressor import CompressorMetadata, DeepseekCompressor
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import _ON_GFX950, on_gfx1250
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import execute_in_parallel
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4
from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
    DeepseekV4RocmMxfp4IndexerBackend,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_prefill_topk_ragged_indices,
    build_ragged_indices_from_dense,
    rocm_inv_rope_einsum,
    rocm_inverse_rope_rows_,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    rocm_mxfp4_indexer_k_store,
    rocm_mxfp4_indexer_q_quant,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def _wo_a_block_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor | None:
    """Normalize checkpoint WO_A scales to raw OCP MX E8M0 bytes.

    E8M0 is an unsigned exponent-only scale format with bias 127. A finite
    encoded byte ``b`` in ``[0, 254]`` represents ``2 ** (b - 127)``;
    ``0xFF`` is reserved for NaN. This is the vendor-neutral OCP encoding used
    by AMD AITER/OPUS, not an NVIDIA-specific convention. Reference: OCP
    Microscaling Formats (MX) Specification, section 5.4.1:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

    Loaders may preserve the encoded byte as ``float8_e8m0fnu``/``uint8``,
    or decode it to a floating-point power of two. Floating-point inputs are
    accepted only when the original E8M0 byte can be recovered losslessly.
    This function does not quantize or round arbitrary scales.
    """
    if scale.dtype == torch.float8_e8m0fnu:
        # Reinterpret the native E8M0 storage. ``to(uint8)`` would perform a
        # numeric conversion instead of preserving the encoded exponent byte.
        return scale.view(torch.uint8).contiguous()
    if scale.dtype == torch.uint8:
        # The checkpoint loader already exposed the E8M0 wire representation.
        return scale.contiguous()
    if not scale.dtype.is_floating_point:
        return None

    scale_f32 = scale.detach().float()
    if not bool(torch.isfinite(scale_f32).all()) or bool((scale_f32 <= 0).any()):
        return None

    # With no mantissa, E8M0 can represent only exact powers of two. Rebuild
    # the value before encoding so this adapter never silently quantizes a
    # general floating-point checkpoint scale.
    exponent = torch.round(torch.log2(scale_f32))
    if not torch.equal(torch.exp2(exponent), scale_f32):
        return None

    encoded = exponent.to(torch.int32) + 127
    # 0xFF is NaN in OCP E8M0, not a finite exponent.
    if int(encoded.min()) < 0 or int(encoded.max()) > 254:
        return None
    return encoded.to(torch.uint8).contiguous()


def _trust_dsv4_extra_cache_nan_free(
    kv_cache_dtype: str,
    has_kv_transfer: bool,
    has_extra_cache: bool,
) -> bool:
    return (
        _ON_GFX950
        and kv_cache_dtype == "fp8_ds_mla"
        and not has_kv_transfer
        and has_extra_cache
    )


def _aiter_sparse_mla_enabled(has_kv_transfer: bool) -> bool:
    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.is_triton_sparse_mla_enabled():
        return False
    # The aiter kernel does not scrub NaNs, so like the trust check above it
    # needs a cache only the canonical writer has touched.
    if has_kv_transfer:
        logger.warning_once(
            "VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA is ignored with a KV "
            "connector; DeepSeek V4 keeps the Triton sparse attention."
        )
        return False
    return True


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


def apply_pre_quantized_block_scaled_mm(
    linear: torch.nn.Module,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled fp8 GEMM on pre-quantized activations.

    The fused q/kv norm kernel writes fp8 qr + per-1x128 scales; this
    drives the linear's block-scaled GEMM directly with them, bypassing
    apply_weights which would re-quantize the fp8 input. Only valid for
    the wq_b-style column/replicated linears: their output is the local
    TP shard, so no all-reduce is needed.
    """
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        FP8BlockParams,
    )

    params = FP8BlockParams.from_layer(linear)
    weight_scale = (
        params.weight_scale
        if params.weight_scale_inv is None
        else params.weight_scale_inv
    )
    kernel = linear.quant_method.fp8_linear
    out = kernel.apply_block_scaled_mm(
        A=x_fp8, B=params.weight, As=x_scale, Bs=weight_scale
    )
    return out.to(dtype=kernel.config.out_dtype)


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


# m-attn-gemm-rate. Behind dsv4_attn_e8m0=1 (default off, gfx950), the
# preshuffled attention fp8 GEMMs (fused_wqa_wkv, wo_b, row-local wo_b_full) keep the
# checkpoint's ue8m0 128x128 weight scales instead of upcasting them to fp32, quantize
# their activations with e8m0 (power-of-two) 1x128 scales, and so dispatch AITER's
# DSv4-tuned FlyDSL mxscale bpreshuffle rows (bmm / mxpsh / mxfp8 8-wave) instead of
# the fp32-scale CK / asm ones. The CUDA path's DeepGEMM ue8m0 recipe; the wo_a bmm
# already reads e8m0 activation scales here. Not bit-exact: the power-of-two act
# scale rounds (~0.5 dB GEMM SNR). The SP fp8 hand-off rows (fp32 scales from the
# gather) keep the fp32 route. Every tuned kernel the shapes can reach is compiled
# at load (FlyDSL JIT), before the server reports healthy.
ATTN_E8M0_LABEL = "dsv4_attn_e8m0"
_ATTN_E8M0_LOGGED = False
_ATTN_E8M0_PREP_LOGGED = False
_ATTN_E8M0_WARMED: set[tuple[int, int]] = set()


def _attn_e8m0_warm(weight: torch.Tensor, scale_e8: torch.Tensor, max_m: int) -> int:
    """Run one GEMM per distinct tuned kernel AITER can pick for this (N, K) at
    M in [1, max_m], so FlyDSL compiles (or loads) each before serving. Returns
    the number of kernels warmed (0 when this shape was warmed already)."""
    from aiter import dtypes
    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale_bpreshuffle,
        get_mxscale_bpreshuffle_config,
    )
    from aiter.ops.gemm_op_common import mxscale_w_scale_block
    from aiter.ops.quant import per_group_quant_hip

    n, k = int(weight.shape[0]), int(weight.shape[1])
    if (n, k) in _ATTN_E8M0_WARMED:
        return 0
    _ATTN_E8M0_WARMED.add((n, k))
    block = mxscale_w_scale_block(tuple(scale_e8.shape), n, k)
    reps: dict[Any, int] = {}
    for m in range(1, max_m + 1):
        row = get_mxscale_bpreshuffle_config(m, n, k, block)
        key = None if row is None else (str(row["kernelId"]), str(row["kernelName"]))
        reps.setdefault(key, m)
    x = torch.zeros(max(reps.values()), k, dtype=torch.bfloat16, device=weight.device)
    for m in reps.values():
        xq, xs = per_group_quant_hip(
            x[:m], quant_dtype=dtypes.fp8, group_size=128,
            transpose_scale=True, scale_type=dtypes.fp8_e8m0,
        )
        gemm_a8w8_blockscale_bpreshuffle(xq, weight, xs, scale_e8, torch.bfloat16)
    torch.cuda.synchronize()
    del x
    return len(reps)


def _default_first_token_threshold() -> int:
    """Max tokens at which the CSA/HCA fork enqueues the default chain first.

    At small batches starting the long default chain first shortens the step
    tail; larger batches saturate the device and aux-first is faster. Only
    tuned on gfx950 (24 tokens = conc 8 with MTP=2); 0 disables elsewhere.
    """
    return 24 if _ON_GFX950 else 0


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    left_visible_ptr,
    right_visible_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SWA_WIDTH: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PADDED_SWA_WIDTH: tl.constexpr,
    HAS_IMAGE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        if HAS_IMAGE:
            left = tl.load(left_visible_ptr + token_idx)
            right = tl.load(right_visible_ptr + token_idx)
        else:
            left = 0
            right = 0
        left_add = tl.maximum(left - (WINDOW_SIZE - 1), 0)
        # Prefix caching can resume inside an image span. Do not generate
        # indices outside the SWA rows present in the gathered workspace.
        swa_start = tl.maximum(
            tl.maximum(pos - (WINDOW_SIZE - 1) - left_add, 0), gather_start
        )
        swa_end = tl.minimum(pos + right + 1, seq_len)
        swa_len = tl.maximum(swa_end - swa_start, 0)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, PADDED_SWA_WIDTH)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + swa_start - gather_start,
            mask=(swa_offset < swa_len) & (swa_offset < SWA_WIDTH),
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    max_image_tokens: int = 0,
    left_visible: torch.Tensor | None = None,
    right_visible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (left_visible is None) != (right_visible is None):
        raise ValueError("left_visible and right_visible must be provided together")
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    has_image = left_visible is not None
    # Keep the row shape fixed for a vision model even when a particular batch
    # has no image.
    swa_width = window_size + max_image_tokens
    combined_topk = (
        (topk + swa_width + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        left_visible if left_visible is not None else topk_indices,
        right_visible if right_visible is not None else topk_indices,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        SWA_WIDTH=swa_width,
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
        PADDED_SWA_WIDTH=triton.next_power_of_2(swa_width),
        HAS_IMAGE=has_image,
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


# The strides and the state block size depend on the KV cache layout; keeping
# them out of the compile key lets one warmup variant serve every deployment.
@triton.jit(
    do_not_specialize=[
        "state_cache_stride0",
        "state_cache_stride1",
        "block_table_stride",
        "block_size",
        "pooled_k_stride",
    ]
)
def _compress_pool_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    kv_slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    pooled_k_ptr,
    pooled_k_stride,
    HEAD_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    """The softmax-gated pooling of `_fused_kv_compress_norm_rope_insert_*`,
    stopping before the norm: one bf16 row per boundary token."""
    token_idx = tl.program_id(0)
    # Pool exactly the rows aiter's writer stores; it skips the same tokens.
    if tl.load(kv_slot_mapping_ptr + token_idx) < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + pos // block_size,
        mask=mask_pos,
        other=0,
    )
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + (pos % block_size) * state_cache_stride1
        + head_offset
    )

    block = tl.arange(0, HEAD_SIZE)
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=mask_pos[:, None],
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(row_base[:, None] + block[None, :], mask=mask_pos[:, None], other=0.0)
    compressed_kv = tl.sum(kv * score, axis=0)
    tl.store(
        pooled_k_ptr + token_idx.to(tl.int64) * pooled_k_stride + block,
        compressed_kv.to(tl.bfloat16),
    )


def _compress_pool_warmup_inputs(
    *, head_dim: int, state_width: int, compress_ratio: int, overlap: bool
) -> list[dict[str, object]]:
    # ROCm specializes a pointer on whether its storage is at most 2 GB (see
    # TritonWarmupTensor.ptr_range_bytes). The state cache is a view of the KV
    # pool, whose size depends on the deployment and is usually far above
    # 2 GB, so warm up both variants; the other inputs are small buffers.
    return [
        dict(
            state_cache=TritonWarmupTensor(
                torch.float32,
                shape=(1, 1, 2 * state_width),
                ptr_range_bytes=pool_bytes,
            ),
            token_to_req_indices=TritonWarmupTensor(torch.int32),
            positions=TritonWarmupTensor(torch.int64),
            kv_slot_mapping=TritonWarmupTensor(torch.int64),
            block_table=TritonWarmupTensor(torch.int32, shape=(1, 1)),
            block_size=compress_ratio,
            pooled_k=TritonWarmupTensor(torch.bfloat16, shape=(1, head_dim)),
            state_width=state_width,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        for pool_bytes in (0, 1 << 32)
    ]


@triton_kernel_dispatcher_with_warmup(
    kernel=_compress_pool_kernel,
    warmup_inputs=_compress_pool_warmup_inputs,
)
def compress_pool(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    pooled_k: torch.Tensor,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
) -> DispatchSpec:
    """Pool each boundary token's compressor states into a bf16 row of
    ``pooled_k``, one program per token. Tensor inputs bind to the kernel's
    ``<name>_ptr`` arguments; the same launch spec drives the JIT warmup."""
    return (pooled_k.shape[0],), dict(
        state_cache_stride0=state_cache.stride(0),
        state_cache_stride1=state_cache.stride(1),
        block_table_stride=block_table.stride(0),
        pooled_k_stride=pooled_k.stride(0),
        HEAD_SIZE=pooled_k.shape[1],
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        num_warps=1,
    )


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


def _copy_ragged_to_graph_buffers(
    ragged_indices: torch.Tensor,
    ragged_indptr: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy dynamic ragged metadata into persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses. Keep the returned
    tensors backed by stable storage, while indptr continues to bound reads.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)

    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    source_entries = ragged_indices.numel()
    if source_entries > 0:
        ragged_out[:source_entries].copy_(ragged_indices, non_blocking=True)
    if _ON_GFX950:
        # Preserve the graph-stable base pointer while exposing source capacity
        # to the sync-free split selector; indptr still carries the true NNZ.
        ragged_out = ragged_out[: max(source_entries, 1)]
    return ragged_out, indptr_out


def _wo_a_rope_cos_sin(rotary_emb) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous bf16 cos and sin halves of a rotary embedding's cache, for
    the fp8 wo_a inverse-rope quant.

    One pair per rotary embedding, not per layer: get_rope shares the module
    across layers with the same parameters (main and compress here), and at
    max_position_embeddings 1,048,576 x rope dim 64 each pair is 128 MiB of
    bf16, so a copy per layer held 61 x 128 MiB = 7.6 GiB per rank. The pair
    is keyed on the source cache tensor so a replaced cache is not reused.
    """
    src = getattr(rotary_emb, "cos_sin_cache_bf16", None)
    if src is None:
        src = rotary_emb.cos_sin_cache
    key = (src.data_ptr(), src.dtype, tuple(src.shape), src.device)
    cached = getattr(rotary_emb, "_dsv4_wo_a_cos_sin", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    cache = src if src.dtype == torch.bfloat16 else src.to(dtype=torch.bfloat16)
    cos_cache, sin_cache = cache.chunk(2, dim=-1)
    pair = (cos_cache.contiguous(), sin_cache.contiguous())
    # A plain attribute, not a buffer: not part of the module's state.
    object.__setattr__(rotary_emb, "_dsv4_wo_a_cos_sin", (key, *pair))
    return pair


@dataclass
class DeepseekV4ROCMAiterMLASparseMetadata(DeepseekV4FlashMLAMetadata):
    """ROCm-specific DeepSeek V4 metadata carrying ragged decode topk."""

    c128a_decode_topk_ragged_indices: torch.Tensor | None = None
    c128a_decode_topk_ragged_indptr: torch.Tensor | None = None
    for_cudagraph_capture: bool = False


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None
    # Built only for the aiter prefill, which reads the SWA cache in place.
    prefill_swa_ragged_indices: torch.Tensor | None = None
    prefill_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4ROCMAiterMLASparseMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.enable_adaptive_verification:
            # All per-token metadata is built from device query boundaries into
            # persistent buffers, so adaptive verification can replay varlen
            # FULL decode graphs after reallocating drafts across requests.
            return AttentionCGSupport.ALWAYS
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c128a_decode_topk_ragged_indices_buffer: torch.Tensor | None = None
        self.c128a_decode_topk_ragged_indptr_buffer: torch.Tensor | None = None
        if self.compress_ratio == 128:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        dense_decode = base.c128a_global_decode_topk_indices
        decode_lens = base.c128a_decode_topk_lens
        if dense_decode is not None and decode_lens is not None:
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                dense_decode.reshape(dense_decode.shape[0], -1),
                decode_lens,
            )
            assert self.c128a_decode_topk_ragged_indices_buffer is not None
            assert self.c128a_decode_topk_ragged_indptr_buffer is not None
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.c128a_decode_topk_ragged_indices_buffer,
                self.c128a_decode_topk_ragged_indptr_buffer,
                dense_decode.shape[0],
                self.c128a_max_compressed,
            )

        return DeepseekV4ROCMAiterMLASparseMetadata(
            **vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata,
            super().build_for_cudagraph_capture(common_attn_metadata),
        )
        metadata.for_cudagraph_capture = _ON_GFX950
        return metadata


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.enable_adaptive_verification:
            # SWA indices, lengths, and token-to-request mappings are built from
            # device boundaries into persistent buffers, so adaptive verification
            # can replay varlen FULL decode graphs safely.
            return AttentionCGSupport.ALWAYS
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    # Keep fused multi-step decode disabled until update_draft_decode_metadata()
    # also refreshes the ROCm-specific ragged SWA indices and indptrs.
    supports_draft_decode_metadata_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # The non-causal (DSpark draft) path widens each token's SWA index list
        # to ``noncausal_index_width`` (>= window_size), so size the persistent
        # ragged buffer to the wider bound to cover both causal and non-causal.
        swa_index_width = max(self.window_size, self.noncausal_index_width)
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self._use_aiter_sparse_mla = _aiter_sparse_mla_enabled(
            self.vllm_config.kv_transfer_config is not None
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
            replay_start=replay_start,
        )

        prefill_ragged_indices = None
        prefill_ragged_indptr = None
        if (
            self._use_aiter_sparse_mla
            and base.num_prefill_tokens > 0
            and base.prefill_swa_indices is not None
            and base.prefill_swa_lens is not None
        ):
            # Paged-coordinate rows (image visibility included), so every
            # layer's prefill shares one build.
            prefill_ragged_indices, prefill_ragged_indptr = (
                build_ragged_indices_from_dense(
                    base.prefill_swa_indices.reshape(base.num_prefill_tokens, -1),
                    base.prefill_swa_lens,
                )
            )

        ragged_indices = None
        ragged_indptr = None
        if (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(
                    base.num_decode_tokens, base.decode_swa_width
                ),
                base.decode_swa_lens,
            )
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                base.num_decode_tokens,
                base.decode_swa_width,
            )

        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
            prefill_swa_ragged_indices=prefill_ragged_indices,
            prefill_swa_ragged_indptr=prefill_ragged_indptr,
        )


class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterMLASparseMetadataBuilder"]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder


# One dispatch line per rank for sp_compressor_shard.
_SP_COMP_SHARD_LOGGED = [False]

# m-attn-rowlocal (H128), landing step 1 of 3: behind dsv4_rowlocal_attention=1 (default off)
# every backbone attention layer also holds the full (all-head) wq_b / wo_a / wo_b and attn_sink,
# which row-local attention (each rank runs all 128 heads on its own Mp/8 rows of a
# sequence-parallel step) consumes. The TP shards of wq_b and wo_a become row views of the full
# weights (same bytes, same padded stride, checked bytewise at load), so the TP paths, the mono
# decode kernel included, read byte-identical tensors. wo_b's TP shard is a column slice that the
# mono kernel needs contiguous, so the full wo_b is an extra copy (checked equal before shuffling).
ATTN_ROWLOCAL_LABEL = "dsv4_rowlocal_attention"
# One dispatch line per rank for the row-local attention path (step 2).
_ATTN_ROWLOCAL_LOGGED = [False]
# Compress ratios whose row-local sparse-MLA variants are compiled (_rowlocal_warmup).
_ROWLOCAL_WARMED: set[int] = set()

# m-itl-stack X1 (c6b-r6): on a row-local step each rank reads only its prefill rows [r0, r1) of
# the top-k ragged stream, yet the stream was built for every prefill row of the step on every
# layer, and for C128 identically on all of its layers (its local top-k is step metadata). Build it
# for [r0, r1) only, and for C128 once per step (cached on the step's metadata object, which the
# builder makes fresh per step and shares across the attention group). The sparse-MLA launch reads
# the same per-row index lists in the same order: bitwise. Always on (the full build,
# sliced, is no longer selectable).
_ROWLOCAL_RAGGED_WINDOW_LOGGED: set[int] = set()

# m-attn-rowlocal (H128) step 2b, input side. Behind dsv4_rowlocal_input=1 (default off;
# acts only with dsv4_rowlocal_attention=1 and the compressor shard, and turns on the
# row-local indexer half, dsv4_rowlocal_input, with it): on a row-local sequence-parallel step the decoder layer does not gather the
# attention input. Each rank packs its own Mp/8 rows (fp8 1x128, the SP gather's wire format) and
# one AITER custom all-gather carries only the rows other ranks read: the compressor plan's
# replicated rows, the decode rows (C4: the indexer query of decode rows is replicated), and the
# window halo of each rank's slice (the compressor plan owns the boundaries inside each SP slice).
# fused_wqa_wkv runs on the rank's own rows (C4: plus the decode rows, in the indexer's compact
# layout), the normed kv of the own rows is all-gathered (bf16) so every rank inserts every row's
# SWA KV as before, and indexer weights_proj runs on the same rows. Hidden rows are bitwise the
# gathered rows; fused_wqa_wkv at a different M can differ by 1 bf16 ulp: NUMERICS: yes.
ROWLOCAL_INPUT_LABEL = "dsv4_rowlocal_input"
_ROWLOCAL_DECODE_ONLY_LOGGED = [False]
_ROWLOCAL_INPUT_CHECKED: set[int] = set()


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, stride, dtype and bytes."""
    return (
        a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
        and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    )


def _sp_save_states(c, cmd, part, positions, plan, which) -> None:
    """save_partial_states of the rows `which` ('s' slice or 'r' replicated) of a compressor."""
    kv, score = part.split([c.coff * c.head_dim, c.coff * c.head_dim], dim=-1)
    state_cache = c.state_cache.kv_cache
    _SAVE_PARTIAL_STATES_KERNEL(
        kv=kv,
        score=score,
        ape=c.ape,
        positions=plan.sub(positions, which),
        state_cache=state_cache,
        slot_mapping=plan.sub(cmd.slot_mapping, which),
        block_size=cmd.block_size,
        state_width=state_cache.shape[-1] // 2,
        compress_ratio=c.compress_ratio,
        pdl_kwargs={},
    )


def _sp_compress_rows(c, cmd, positions, rot, attn_metadata, plan) -> None:
    """The served compress -> norm -> RoPE -> quant -> store of a compressor, on the plan's
    compress list (replicated requests' tokens + this rank's owned boundary tokens)."""
    state_cache = c.state_cache.kv_cache
    kmd = cast(Any, attn_metadata[c.k_cache_prefix])
    kw: dict[str, Any] = dict(
        state_cache=state_cache,
        num_actual=plan.n_c,
        token_to_req_indices=plan.sub(cmd.token_to_req_indices, "c"),
        positions=plan.sub(positions, "c"),
        slot_mapping=plan.sub(cmd.slot_mapping, "c"),
        block_table=cmd.block_table,
        block_size=cmd.block_size,
        state_width=state_cache.shape[-1] // 2,
        cos_sin_cache=rot.cos_sin_cache,
        kv_cache=c._static_forward_context[c.k_cache_prefix].kv_cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=plan.sub(kmd.slot_mapping, "c")),
        pdl_kwargs={},
        head_dim=c.head_dim,
        rope_head_dim=c.rope_head_dim,
        compress_ratio=c.compress_ratio,
        overlap=c.overlap,
        use_fp4_cache=c.use_fp4_cache,
        rms_norm_weight=c.norm.weight,
        rms_norm_eps=c.rms_norm_eps,
        quant_block=c._quant_block,
        token_stride=c._token_stride,
        scale_dim=c._scale_dim,
    )
    if c._use_two_stage_fused_compressor:
        # Mirror the served launch: the same per-token decode/prefill kernel choice and the
        # stage-1 split the full step would pick, so a token's row does not depend on the subset.
        n_served = cmd.slot_mapping.shape[0]
        nd = min(max(cmd.num_decode_tokens or 0, 0), n_served)
        compress_norm_rope_store_two_stage_triton(
            **kw,
            num_decode_tokens=int(np.searchsorted(plan.c_idx_np, nd)),
            compress_scratch=c._compress_scratch,
            num_splits=_pick_compress_num_splits(n_served - nd, c.compress_ratio, c.head_dim),
        )
    else:
        compress_norm_rope_store_triton(**kw)


def sp_compressor_shard_run(x, positions, plan, weight, split, mods, attn_metadata, comm) -> None:
    """m-compressor-shard: the per-layer work of the window-aligned compressor shard.

    x: the gathered layer input [M, H]; weight: the compressor GEMM weight (C4: main + indexer
    fused, split by `split`); mods: [(compressor, its CompressorMetadata, rotary)]. Runs the
    slice and replicated GEMMs, the state saves, the compress of the plan's token list and the
    exchange of the owned compressed rows.
    """
    fs = (
        torch.mm(x[plan.s_lo:plan.s_hi], weight.T, out_dtype=torch.float32)
        if plan.s_hi > plan.s_lo else None
    )
    fr = (
        torch.mm(x.index_select(0, plan.r_idx), weight.T, out_dtype=torch.float32)
        if plan.n_r else None
    )
    parts_s = fs.split(split, dim=-1) if fs is not None and split else (fs,)
    parts_r = fr.split(split, dim=-1) if fr is not None and split else (fr,)
    for k, (c, cmd, rot) in enumerate(mods):
        # sharded slice first, replicated rows last (they win where both write a slot)
        if fs is not None:
            _sp_save_states(c, cmd, parts_s[k], positions, plan, "s")
        if fr is not None:
            _sp_save_states(c, cmd, parts_r[k], positions, plan, "r")
        if plan.n_c:
            _sp_compress_rows(c, cmd, positions, rot, attn_metadata, plan)
    rows = []
    for c, _, _ in mods:
        kmd = cast(Any, attn_metadata[c.k_cache_prefix])
        kv = c._static_forward_context[c.k_cache_prefix].kv_cache
        rows.append((_CacheRows(kv, c._token_stride, c._scale_dim), kmd.slot_mapping))
    plan.exchange(comm, rows[0], rows[1] if len(rows) > 1 else None)


class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend
    _use_aiter_sparse_mla = False
    # Set by the decoder layer around a sequence-parallel step: return wo_b's TP partial sum.
    _sp_defer_reduce = False
    # Set by the decoder layer around a sequence-parallel step when
    # sp_compressor_shard=1 (m-compressor-shard): the step's SeqParallelStep.
    _sp_step = None
    # Set by the decoder layer around a sequence-parallel step when dsv4_rowlocal_attention=1
    # and rowlocal_ready() (m-attn-rowlocal step 2): the step's SeqParallelStep; forward then
    # returns this rank's Mp/8 rows of the full (all-head) attention output, no reduce needed.
    _sp_rowlocal = None
    # Set with _sp_rowlocal when dsv4_rowlocal_input=1 and rowlocal_input_head() allowed
    # it: the indexer's decode-row count (C128: 0); forward then receives this rank's own rows.
    _sp_rowlocal_head = None

    def _get_indexer_cls(self, vllm_config: VllmConfig) -> type[DeepseekV4Indexer]:
        if dsa_indexer_uses_fp4(vllm_config):
            # aiter's paged MXFP4 kernels write and score the index K cache.
            return DeepseekV4RocmMxfp4Indexer
        return super()._get_indexer_cls(vllm_config)

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs["vllm_config"]
        super().__init__(*args, **kwargs)
        self._has_kv_transfer = vllm_config.kv_transfer_config is not None
        # Block scale for the preshuffled weight; None = not preshuffled.
        self._wqa_wkv_scale: torch.Tensor | None = None
        self._wo_b_scale: torch.Tensor | None = None
        # m-attn-gemm-rate: the checkpoint's e8m0 scales beside the fp32 ones (switch on).
        self._wqa_wkv_scale_e8: torch.Tensor | None = None
        self._wo_b_scale_e8: torch.Tensor | None = None
        self._wo_b_full_scale_e8: torch.Tensor | None = None
        self._wo_a_fp8_weight: torch.Tensor | None = None
        self._wo_a_e8m0_scale: torch.Tensor | None = None
        self._wo_a_cos_cache: torch.Tensor | None = None
        self._wo_a_sin_cache: torch.Tensor | None = None
        self._fused_compressor_weight: torch.Tensor | None
        self.register_buffer("_fused_compressor_weight", None, persistent=False)
        self._fused_compressor_split_sizes: tuple[int, int] | None = None

        if self.indexer is None:
            # Dense layers have no compressor work to overlap; HCA layers
            # (compressor, no indexer) keep the streams for the dual-stream
            # fork below.
            if self.compressor is None:
                self.aux_stream_list = None
        else:
            # Disable indexer inner overlap.
            self.indexer.aux_stream = None
        self._use_aiter_sparse_mla = _aiter_sparse_mla_enabled(self._has_kv_transfer)
        self._sp_vllm_config = vllm_config
        self._sp_comp_shard: bool | None = None
        # m-attn-rowlocal step 1: full-head attention weights (None unless the switch is on).
        self._rowlocal_full = False
        self._wo_b_full_scale: torch.Tensor | None = None
        self._wo_a_full_fp8_weight: torch.Tensor | None = None
        self._wo_a_full_e8m0_scale: torch.Tensor | None = None
        self._rowlocal_ok: bool | None = None
        if active_attention_config().dsv4_rowlocal_attention:
            self._init_rowlocal_full(vllm_config)

    def _init_rowlocal_full(self, vllm_config: VllmConfig) -> None:
        """Register the full wq_b / wo_a / wo_b / attn_sink of a backbone layer (step 1 of the
        m-attn-rowlocal landing). Draft and MTP layers (compress ratio 1) are left alone."""
        from vllm.model_executor.layers.linear import ReplicatedLinear
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm.model_executor.models.utils import extract_layer_index

        config = vllm_config.model_config.hf_config
        if (
            get_tensor_model_parallel_world_size() == 1
            or self.compress_ratio <= 1
            or extract_layer_index(self.prefix) >= config.num_hidden_layers
        ):
            return
        # the full weights cost ~15 GiB/rank of KV cache: load them only if SP can engage and
        # rowlocal_ready() can hold (AITER sparse MLA and the fp8 wo_a BMM are known here)
        if not vllm_config.attention_config.dsv4_seq_parallel or sp_config_disable_reasons(vllm_config):
            return
        if not (self._use_aiter_sparse_mla and _ON_GFX950 and envs.VLLM_ROCM_USE_AITER_FP8BMM):
            return
        quant_config = vllm_config.quant_config
        self.wq_b_full = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{self.prefix}.wq_b_full",
        )
        self.wo_a_full = ReplicatedLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{self.prefix}.wo_a_full",
        )
        self.wo_a_full.is_bmm = True
        self.wo_a_full.bmm_batch_size = self.n_groups
        self.wo_b_full = ReplicatedLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{self.prefix}.wo_b_full",
        )
        sink = torch.nn.Parameter(
            torch.full((self.n_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        sink.weight_loader = default_weight_loader
        self.attn_sink_full = sink
        self._rowlocal_full = True

    def _prepare_rowlocal_full(self) -> int:
        """Turn the TP wq_b / wo_a shards into row views of the full weights after a bytewise
        check, and check wo_b's shard against the full wo_b's columns (before either is
        shuffled). Returns the bytes the full weights add. Raises if a shard differs."""
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            get_fp8_block_weight_scale,
        )

        rank = get_tensor_model_parallel_rank()
        tp = get_tensor_model_parallel_world_size()

        def scale_attr(linear) -> str:
            return "weight_scale_inv" if hasattr(linear, "weight_scale_inv") else "weight_scale"

        added = 0
        for name in ("wq_b", "wo_a"):
            part, full = getattr(self, name), getattr(self, f"{name}_full")
            pw, fw = part.weight, full.weight
            ps, fs = get_fp8_block_weight_scale(part), get_fp8_block_weight_scale(full)
            assert ps is not None and fs is not None
            n, ns = pw.shape[0], ps.shape[0]
            assert fw.shape[0] == n * tp and fs.shape[0] == ns * tp
            vw, vs = fw.data[rank * n : (rank + 1) * n], fs.data[rank * ns : (rank + 1) * ns]
            if not (_bytes_equal(pw.data, vw) and _bytes_equal(ps.data, vs)):
                raise RuntimeError(
                    f"{ATTN_ROWLOCAL_LABEL}: {self.prefix}.{name} TP shard differs from the "
                    f"full weight's rows on rank {rank}"
                )
            # Rebind (not replace_parameter, which would copy into the shard's storage).
            pw.data = vw
            getattr(part, scale_attr(part)).data = vs
            added += (fw.numel() - vw.numel()) * fw.element_size()
            added += (fs.numel() - vs.numel()) * fs.element_size()
        pw, fw = self.wo_b.weight, self.wo_b_full.weight
        ps = get_fp8_block_weight_scale(self.wo_b)
        fs = get_fp8_block_weight_scale(self.wo_b_full)
        assert ps is not None and fs is not None
        k, ks = pw.shape[1], ps.shape[1]
        if not (
            torch.equal(pw.data.view(torch.uint8), fw.data[:, rank * k : (rank + 1) * k].view(torch.uint8))
            and torch.equal(ps.data, fs.data[:, rank * ks : (rank + 1) * ks])
        ):
            raise RuntimeError(
                f"{ATTN_ROWLOCAL_LABEL}: {self.prefix}.wo_b TP shard differs from the full "
                f"weight's columns on rank {rank}"
            )
        added += fw.numel() * fw.element_size() + fs.numel() * fs.element_size()
        return added

    def _run_sequential_pipeline(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        o_padded: torch.Tensor,
    ) -> None:
        """Disable ROCm streams when the current execution region cannot overlap."""
        aux_streams = self.aux_stream_list
        self.aux_stream_list = None
        try:
            sp = self._sp_step
            if sp is not None and self._sp_compressor_shard(hidden_states, positions, sp):
                # The compressors' work for this step is done; their scores are not needed.
                qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
                kv_score = indexer_kv_score = indexer_weights = None
                if self.indexer is not None:
                    indexer_weights, _ = self.indexer.weights_proj(hidden_states)
            else:
                qr_kv, kv_score, indexer_kv_score, indexer_weights = (
                    self._run_parallel_input_projections(hidden_states)
                )
            qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
            self._prepare_and_attn_fn(
                hidden_states,
                qr,
                kv,
                qr_scale,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                positions,
                o_padded,
            )
        finally:
            self.aux_stream_list = aux_streams

    def _sp_comp_shard_supported(self) -> bool:
        """m-compressor-shard: whether this layer's compressors can be row-sharded (cached)."""
        ok = self._sp_comp_shard
        if ok is not None:
            return ok
        comp, idx = self.compressor, self.indexer
        ok = comp is not None and type(comp) is DeepseekCompressor and comp.head_dim == 512
        if ok and idx is None:
            # HCA layer: C128 main compressor, ROCm two-stage compress
            ok = comp.compress_ratio == 128 and comp._use_two_stage_fused_compressor
        elif ok:
            # CSA layer: C4 main + C4 indexer compressor through the fused GEMM
            ic = idx.compressor
            ok = (
                comp.compress_ratio == 4
                and type(ic) is DeepseekCompressor
                and ic.compress_ratio == 4
                and ic.head_dim == 128
                and not ic.use_fp4_cache
                and self._fused_compressor_weight is not None
                and self._fused_compressor_split_sizes is not None
            )
        if ok:
            # Correctness preconditions of CompressorShardPlan: prefix-cache hits must land on
            # multiples of the compress ratio (hits are block-aligned, so the block size must
            # be), and only this engine's own scheduler may place cached rows (no KV connector
            # or offload, which can resume at other offsets).
            cfg = self._sp_vllm_config
            ok = (int(cfg.cache_config.block_size) % comp.compress_ratio == 0
                  and cfg.kv_transfer_config is None)
        if ok:
            kv = comp._static_forward_context[comp.k_cache_prefix].kv_cache
            ok = kv.dim() == 3 and kv.dtype == torch.uint8  # fp8_ds_mla paged rows
            if ok and idx is not None:
                ikv = idx.compressor._static_forward_context[
                    idx.compressor.k_cache_prefix].kv_cache
                ok = ikv.dim() == 3 and ikv.dtype == torch.uint8
        self._sp_comp_shard = ok
        return ok

    def _sp_compressor_shard(
        self, x: torch.Tensor, positions: torch.Tensor, sp: SeqParallelStep
    ) -> bool:
        """m-compressor-shard step 4: this layer's compressors on a sequence-parallel step, split
        by compression window across the TP ranks (CompressorShardPlan), then the owned compressed
        rows exchanged so every rank's caches hold every row. Returns False (served path) when the
        layer or the step is not eligible; the decision is the same on every rank.
        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or not self._sp_comp_shard_supported():
            return False
        comp = self.compressor
        indexer = self.indexer
        smd = cast(CompressorMetadata, attn_metadata[comp.state_cache.prefix])
        if smd.query_start_loc_cpu is None or smd.seq_lens_cpu_upper_bound is None:
            return False
        ratio = comp.compress_ratio
        plans = sp.__dict__.setdefault("_comp_plans", {})
        plan = plans.get(ratio)
        if plan is None:
            cfg = self._sp_vllm_config
            spec = cfg.speculative_config
            max_dec_q = 1 + ((spec.num_speculative_tokens or 0) if spec is not None else 0)
            align = int(cfg.cache_config.block_size)
            qsl = smd.query_start_loc_cpu.numpy()
            ub = smd.seq_lens_cpu_upper_bound.numpy()[: len(qsl) - 1]
            plan = CompressorShardPlan(
                sp, ratio, qsl, ub, x.shape[0], max_dec_q, align, x.device
            )
            plans[ratio] = plan
            if not _SP_COMP_SHARD_LOGGED[0]:
                _SP_COMP_SHARD_LOGGED[0] = True
                logger.info(
                    "%s: compressor row-sharded on rank %d (%s, C%d: %d rows, %d replicated, "
                    "slice [%d, %d), %d of %d prefill boundaries owned, hit alignment %d)",
                    SP_COMPRESSOR_SHARD_LABEL, sp.comm.rank, comp.prefix, ratio, x.shape[0],
                    plan.n_r, plan.s_lo, plan.s_hi, plan.n_own, plan.n_bnd, align,
                )
        if indexer is not None:
            weight = self._fused_compressor_weight
            split = self._fused_compressor_split_sizes
            mods = [
                (comp, smd, self.rotary_emb),
                (indexer.compressor,
                 cast(CompressorMetadata, attn_metadata[indexer.compressor.state_cache.prefix]),
                 self.indexer_rotary_emb),
            ]
        else:
            weight = comp.fused_wkv_wgate.weight
            split = None
            mods = [(comp, smd, self.rotary_emb)]
        sp_compressor_shard_run(x, positions, plan, weight, split, mods, attn_metadata, sp.comm)
        return True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sp_rowlocal = self._sp_rowlocal
        if sp_rowlocal is not None:
            if self._sp_rowlocal_head is not None:
                return self._forward_rowlocal_input(
                    positions, hidden_states, sp_rowlocal, self._sp_rowlocal_head)
            return self._forward_rowlocal(positions, hidden_states, sp_rowlocal)
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if current_platform.enable_multi_stream_overlap(
            self.aux_stream_list, get_forward_context().attn_metadata
        ):
            # The ROCm override consumes these sentinels inside the capture
            # boundary, moving the stream fan-out ahead of the projections.
            self._prepare_and_attn_fn(
                hidden_states,
                None,
                None,
                None,
                None,
                None,
                None,
                positions,
                o_padded,
            )
        else:
            self._run_sequential_pipeline(hidden_states, positions, o_padded)

        o = o_padded[:, : self.n_local_heads, :]

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)

    # m-attn-rowlocal (H128), landing step 2a: on a sequence-parallel step each rank runs the
    # attention core for all 128 heads on its own Mp/8 rows. The input side is the served one
    # (layer input gathered to M rows, fused_wqa_wkv, SWA KV insert, compressors and indexer on
    # all M rows, so every rank's caches and top-k lists hold every row); from wq_b on, only the
    # rank's rows run: full wq_b, q norm + RoPE, sparse MLA in two 64-head launches over the
    # rank's slice of the ragged SWA / top-k lists, full wo_a (16 groups) and full-K wo_b. The
    # output is this rank's rows of the attention output, so the attention reduce-scatter (and its
    # INT4 rounding) is skipped.
    def rowlocal_ready(self) -> bool:
        ok = self._rowlocal_ok
        if ok is not None:
            return ok
        ok = (
            self._rowlocal_full
            and self.compressor is not None
            and self._use_aiter_sparse_mla
            and self._wo_a_fp8_weight is not None
            and self._wo_a_full_fp8_weight is not None
            and self._wo_a_full_e8m0_scale is not None
            and self._wo_b_full_scale is not None
            and self._wq_b_uses_aiter_block_scaled
            and self.n_heads % 64 == 0
        )
        if not ok:
            self._rowlocal_ok = False
            return False
        # The SWA cache is bound after the memory-profile run: check it on every call until it is
        # there, and cache only a settled answer (an fp8_ds_mla paged cache, or another layout).
        cache = self.swa_cache_layer.kv_cache
        if not torch.is_tensor(cache) or cache.numel() == 0:
            return False
        self._rowlocal_ok = cache.dtype == torch.uint8
        return self._rowlocal_ok

    def _forward_rowlocal(self, positions, hidden_states, sp) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        m = hidden_states.shape[0]
        a = min(sp.start, m)
        b = min(sp.start + sp.s, m)
        n = b - a
        aux_streams = self.aux_stream_list
        self.aux_stream_list = None
        # Step 2b (dsv4_rowlocal_input=1): the indexer scores this rank's rows only.
        from vllm.model_executor.layers.sparse_attn_indexer import set_rowlocal_rows

        set_rowlocal_rows((sp.start, sp.s))
        try:
            comp_sp = self._sp_step
            if comp_sp is not None and self._sp_compressor_shard(hidden_states, positions, comp_sp):
                qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
                kv_score = indexer_kv_score = indexer_weights = None
                if self.indexer is not None:
                    indexer_weights, _ = self.indexer.weights_proj(hidden_states)
            else:
                qr_kv, kv_score, indexer_kv_score, indexer_weights = (
                    self._run_parallel_input_projections(hidden_states)
                )
            qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
            assert qr_scale is not None, "row-local attention needs the fused fp8 qr"
            q = None
            if n > 0:
                q = apply_pre_quantized_block_scaled_mm(
                    self.wq_b_full, qr[a:b], qr_scale[a:b]
                ).view(n, self.n_heads, self.head_dim)
            q = self._rowlocal_qnorm_rope_kv_insert(q, kv, positions, a, b, attn_metadata)
            if self.indexer is not None:
                index_q, index_q_scale, index_weights = self.indexer(
                    hidden_states,
                    qr,
                    indexer_kv_score,
                    indexer_weights,
                    positions,
                    self.indexer_rotary_emb,
                    qr_scale,
                )
            assert self.compressor is not None
            self.compressor(kv_score, positions, self.rotary_emb)
            if self.indexer is not None and index_q is not None:
                assert index_weights is not None
                q_quant = (
                    (index_q, index_q_scale) if index_q_scale is not None else index_q
                )
                self.indexer.indexer_op(hidden_states, q_quant, None, index_weights)
        finally:
            self.aux_stream_list = aux_streams
            set_rowlocal_rows(None)
        out = hidden_states.new_zeros((sp.s, self.hidden_size)) if n < sp.s else None
        if n > 0:
            y = self._rowlocal_attn_out(q, a, b, attn_metadata, positions)
            if out is None:
                out = y
            else:
                out[:n].copy_(y)
        if not _ATTN_ROWLOCAL_LOGGED[0]:
            _ATTN_ROWLOCAL_LOGGED[0] = True
            logger.info(
                "%s: row-local attention dispatched on rank %d (%s: rows [%d, %d) of %d, "
                "%d heads, full wq_b/wo_a/wo_b, no attention reduce-scatter)",
                ATTN_ROWLOCAL_LABEL, get_tensor_model_parallel_rank(), self.prefix, a, b, m,
                self.n_heads,
            )
        return out

    def rowlocal_input_head(self) -> int | None:
        """Step 2b input (dsv4_rowlocal_input): None when this layer's input must be
        gathered on this step (the served row-local path), else the number of leading decode rows
        whose fused_wqa_wkv rows this rank also computes (C4: the indexer's decode rows; C128: 0).
        Host-side metadata only, so the answer is the same on every rank; called before the
        gather, with rowlocal_ready() already true."""

        if not (active_attention_config().dsv4_rowlocal_input and self._sp_comp_shard_supported()):
            return None
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None
        smd = attn_metadata.get(self.compressor.state_cache.prefix)
        if smd is None or smd.query_start_loc_cpu is None or smd.seq_lens_cpu_upper_bound is None:
            return None
        if self.indexer is None:
            return 0
        if self.indexer._compact_q_row_shard(attn_metadata, None) is None:
            return None
        meta = attn_metadata[self.indexer.k_cache.prefix]
        return int(meta.num_decode_tokens)

    def _rowlocal_input_plan(self, sp, positions, head: int, attn_metadata):
        comp = self.compressor
        smd = cast(CompressorMetadata, attn_metadata[comp.state_cache.prefix])
        ratio = comp.compress_ratio
        plans = sp.__dict__.setdefault("_comp_plans", {})
        key = ("rowlocal_input", ratio, head)
        plan = plans.get(key)
        if plan is None:
            cfg = self._sp_vllm_config
            spec = cfg.speculative_config
            max_dec_q = 1 + ((spec.num_speculative_tokens or 0) if spec is not None else 0)
            align = int(cfg.cache_config.block_size)
            qsl = smd.query_start_loc_cpu.numpy()
            ub = smd.seq_lens_cpu_upper_bound.numpy()[: len(qsl) - 1]
            plan = CompressorShardPlan(
                sp, ratio, qsl, ub, sp.m, max_dec_q, align, positions.device,
                sp_aligned=True, need_head=head,
            )
            plans[key] = plan
        return plan

    def _forward_rowlocal_input(self, positions, xs, sp, head: int) -> torch.Tensor:
        """Step 2b input: xs is this rank's own rows [Mp/8, H] of the (normed) layer input."""
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        comm = sp.comm
        m = sp.m
        hdim = xs.shape[1]
        a = min(sp.start, m)
        b = min(sp.start + sp.s, m)
        n = b - a
        assert n > 0, (sp.start, sp.s, m)
        # Rows this rank's fused_wqa_wkv computes: [0, head) U [lo, b) (the indexer's compact
        # query layout); this rank's own rows [a, b) sit at [oc, oc + n) of it.
        lo = min(max(a, head), m)
        oc = a if a < head else head
        plan = self._rowlocal_input_plan(sp, positions, head, attn_metadata)
        ratio = self.compress_ratio
        check = sp_self_check_enabled() and ratio not in _ROWLOCAL_INPUT_CHECKED
        xs = xs.contiguous()
        packed = fp8_1x128_quant_pack(xs)
        if check:
            x_full = xs.new_full((m, hdim), float("nan"))
        else:
            x_full = xs.new_empty((m, hdim))
        fp8_1x128_dequant(packed[:n], hdim, out=x_full[a:b])
        plan.gather_needed(comm, packed, x_full)
        if head == 0:
            x_l = x_full[a:b]
        elif lo == head:
            # a < head: [0, head) U [head, b). If this rank's rows end inside the decode rows
            # (b <= head), the compact layout is still all head decode rows (the indexer row
            # shard gives this rank no prefill rows); the plan gathers [0, head) to every rank.
            x_l = x_full[:max(b, head)]
            if b < head and not _ROWLOCAL_DECODE_ONLY_LOGGED[0]:
                _ROWLOCAL_DECODE_ONLY_LOGGED[0] = True
                logger.info(
                    "%s: row-local input on rank %d with own rows [%d, %d) inside the %d "
                    "decode rows (%s)", ROWLOCAL_INPUT_LABEL, sp.comm.rank, a, b, head, self.prefix)
        else:
            x_l = torch.cat((x_full[:head], x_full[lo:b]), dim=0)
        aux_streams = self.aux_stream_list
        self.aux_stream_list = None
        from vllm.model_executor.layers.sparse_attn_indexer import set_rowlocal_rows

        set_rowlocal_rows((sp.start, sp.s))
        try:
            qr_kv = self._fused_wqa_wkv_gemm(x_l)
            qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
            assert qr_scale is not None, "row-local attention needs the fused fp8 qr"
            kv_own = xs.new_zeros((sp.s, kv.shape[1]))
            kv_own[:n].copy_(kv[oc:oc + n])
            if comm._ca.should_custom_ag(kv_own):
                kv_all = comm._ca.custom_all_gather(kv_own)
            else:
                kv_all = comm.tp.all_gather(kv_own, dim=0)
            kv_all = kv_all[:m]
            indexer_weights = None
            if self.indexer is not None:
                indexer_weights, _ = self.indexer.weights_proj(x_l)
            if self.indexer is not None:
                weight = self._fused_compressor_weight
                split = self._fused_compressor_split_sizes
                mods = [
                    (self.compressor,
                     cast(CompressorMetadata,
                          attn_metadata[self.compressor.state_cache.prefix]),
                     self.rotary_emb),
                    (self.indexer.compressor,
                     cast(CompressorMetadata,
                          attn_metadata[self.indexer.compressor.state_cache.prefix]),
                     self.indexer_rotary_emb),
                ]
            else:
                weight = self.compressor.fused_wkv_wgate.weight
                split = None
                mods = [(self.compressor,
                         cast(CompressorMetadata,
                              attn_metadata[self.compressor.state_cache.prefix]),
                         self.rotary_emb)]
            if check:
                self._rowlocal_input_check(sp, xs, x_full, x_l, kv_all, plan, a, b, head)
            sp_compressor_shard_run(x_full, positions, plan, weight, split, mods,
                                    attn_metadata, comm)
            q = apply_pre_quantized_block_scaled_mm(
                self.wq_b_full, qr[oc:oc + n], qr_scale[oc:oc + n]
            ).view(n, self.n_heads, self.head_dim)
            q = self._rowlocal_qnorm_rope_kv_insert(q, kv_all, positions, a, b, attn_metadata)
            index_q = index_q_scale = index_weights = None
            if self.indexer is not None:
                self.indexer._qr_compact = True
                try:
                    index_q, index_q_scale, index_weights = self.indexer(
                        x_full,
                        qr,
                        None,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                        qr_scale,
                    )
                finally:
                    self.indexer._qr_compact = False
            self.compressor(None, positions, self.rotary_emb)
            if self.indexer is not None and index_q is not None:
                assert index_weights is not None
                q_quant = (
                    (index_q, index_q_scale) if index_q_scale is not None else index_q
                )
                self.indexer.indexer_op(x_full, q_quant, None, index_weights)
        finally:
            self.aux_stream_list = aux_streams
            set_rowlocal_rows(None)
        out = xs.new_zeros((sp.s, self.hidden_size)) if n < sp.s else None
        y = self._rowlocal_attn_out(q, a, b, attn_metadata, positions)
        if out is None:
            out = y
        else:
            out[:n].copy_(y)
        return out

    def _rowlocal_input_check(self, sp, xs, x_full, x_l, kv_all, plan, a, b, head) -> None:
        """Once per compress ratio per rank (the same steps on every rank): every hidden row this
        rank reads (own, received, compressor slice and replicated rows, fused_wqa_wkv rows) is
        present and bitwise equal to the served gathered layer input; the gathered kv is compared
        with the served full-M fused_wqa_wkv + norm. Raises on a mismatch; logs the dispatch."""
        ratio = self.compress_ratio
        _ROWLOCAL_INPUT_CHECKED.add(ratio)
        ref = sp.gather(xs).clone()
        sp_fp8_handoff_clear()
        m = sp.m
        rows = [torch.arange(a, b, device=xs.device), plan.recv_dst, plan.r_idx,
                torch.arange(plan.s_lo, plan.s_hi, device=xs.device)]
        idx = torch.unique(torch.cat(rows))
        present = bool(torch.isfinite(x_full.index_select(0, idx)).all().item())
        present &= bool(torch.isfinite(x_l).all().item())
        equal = bool(torch.equal(x_full.index_select(0, idx), ref.index_select(0, idx)))
        _, _, kv_ref = self._split_qkv_and_norm(self._fused_wqa_wkv_gemm(ref))
        kv_eq = float((kv_all == kv_ref).float().mean().item())
        kv_err = float((kv_all.float() - kv_ref.float()).abs().max().item())
        kv_mag = float(kv_ref.float().abs().max().item())
        logger.info(
            "%s: row-local layer input dispatched on rank %d (%s, C%d: own rows [%d, %d) of "
            "%d, fused_wqa_wkv on %d rows (head %d), sent %d / received %d hidden rows "
            "(kmax %d, %d needed by some rank), %d of %d prefill boundaries owned; no layer-input "
            "gather). Self-check vs the gathered input: rows present %s, bitwise %s over %d rows; "
            "kv vs served full-M GEMM bitwise fraction %.4f, max abs %.3g (max |kv| %.3g)",
            ROWLOCAL_INPUT_LABEL, sp.comm.rank, self.prefix, ratio, a, b, m, x_l.shape[0], head,
            plan.n_send, plan.n_recv, plan.kmax, plan.n_need_rows, plan.n_own, plan.n_bnd,
            present, equal, idx.numel(), kv_eq, kv_err, kv_mag)
        # fused_wqa_wkv at another M may pick another kernel: allow bf16 rounding (2^-6 of the
        # largest |kv|), refuse anything larger.
        if not (present and equal) or kv_err > kv_mag * 2.0 ** -6:
            raise RuntimeError(
                f"{ROWLOCAL_INPUT_LABEL}: row-local layer input differs from the gathered input "
                f"(present {present}, bitwise {equal}, kv max abs {kv_err})")

    def _rowlocal_qnorm_rope_kv_insert(self, q, kv, positions, a, b, attn_metadata):
        """KV RoPE + quant + SWA insert for all M rows (every rank's cache holds every row);
        q norm + RoPE only for rows [a, b). Rows outside go through the same kernel with a
        zero-head q, so the cache bytes are those of the served single call."""
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None
        cache = self.swa_cache_layer.kv_cache
        cache_2d = cache.view(cache.shape[0], -1)
        slot = swa_metadata.slot_mapping
        cos_sin = self.rotary_emb.cos_sin_cache
        bs = swa_metadata.block_size
        op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
        m = kv.shape[0]
        qo = None
        if q is not None:
            qo = op(q, kv[a:b], cache_2d, slot[a:b], positions[a:b], cos_sin,
                    self.n_heads, self.eps, bs)
        for lo, hi in ((0, a), (b, m)):
            if hi > lo:
                dummy = kv.new_empty((hi - lo, 0, self.head_dim))
                op(dummy, kv[lo:hi], cache_2d, slot[lo:hi], positions[lo:hi], cos_sin, 0,
                   self.eps, bs)
        return qo

    def _rowlocal_sparse(self, q, o, swa_idx, swa_ptr, k_cache, tk_idx, tk_ptr) -> None:
        """64-head sparse-MLA launches for one decode or prefill row range (row-local)."""
        from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

        swa_cache = self.swa_cache_layer.kv_cache
        # The row slices of the step's indptr start at any int32 offset; Triton specializes
        # pointer arguments on 16-byte alignment, so give each launch its own (aligned) copy and
        # keep the kernel variants to the aligned set the warmup compiles.
        swa_ptr = swa_ptr.clone()
        tk_ptr = tk_ptr.clone()
        if self.compress_ratio not in _ROWLOCAL_WARMED:
            _ROWLOCAL_WARMED.add(self.compress_ratio)
            self._rowlocal_warmup(swa_cache, k_cache)
        # Two 64-head launches: 64-head programs (BLOCK_M 64) at M/8 rows; one 128-head launch
        # falls back to 16/32-head programs (m-sparse-mla-wide-s1-ref-c3). One split: each launch
        # reads its rows' absolute slice of the whole step's index stream, so the launcher's
        # length average (stream size / rows) would overstate the split-K need; the served
        # heuristic gives 1 (rarely 2) split at >= ~128 rows anyway, and a fixed split keeps the
        # kernel variants to the set _rowlocal_warmup compiles before the server is healthy.
        for h0 in range(0, self.n_heads, 64):
            sparse_mla_fwd(
                q[:, h0:h0 + 64],
                swa_cache,
                swa_ptr,
                swa_idx,
                self.scale,
                kv_splits=1,
                has_invalid=True,
                out=o[:, h0:h0 + 64],
                attn_sink=self.attn_sink_full[h0:h0 + 64],
                extra_kv=k_cache,
                extra_indptr=tk_ptr,
                extra_indices=tk_idx,
            )

    def _rowlocal_warmup(self, swa_cache, k_cache) -> None:
        """Compile the 64-head, one-split sparse-MLA variants row-local launches can reach for
        this cache class (block sizes are compile-time): every row count up to a rank's largest
        share of a step (each crosses the launcher's BLOCK_M / prefill / occupancy thresholds)
        at three per-row index lengths. Runs inside the first row-local step, which is the
        engine's warmup run, so nothing compiles after the server reports healthy."""
        from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

        dev = swa_cache.device
        max_rows = (self.max_num_batched_tokens + 7) // 8 + 128
        q = torch.zeros((max_rows, 64, self.head_dim), dtype=torch.bfloat16, device=dev)
        o = torch.empty_like(q)
        sink = self.attn_sink_full[:64]
        for per_row in (32, 192, 1024):
            idx = torch.zeros(max_rows * per_row, dtype=torch.int32, device=dev)
            ptr = torch.arange(0, max_rows + 1, dtype=torch.int32, device=dev) * per_row
            sidx = torch.zeros(max_rows * min(per_row, 128), dtype=torch.int32, device=dev)
            sptr = torch.arange(0, max_rows + 1, dtype=torch.int32, device=dev) * min(per_row, 128)
            for n in range(1, max_rows + 1):
                sparse_mla_fwd(
                    q[:n], swa_cache, sptr[: n + 1], sidx[: n * min(per_row, 128)], self.scale,
                    kv_splits=1, has_invalid=True, out=o[:n], attn_sink=sink,
                    extra_kv=k_cache, extra_indptr=ptr[: n + 1], extra_indices=idx[: n * per_row],
                )
        torch.cuda.synchronize(dev)
        logger.info(
            "%s: row-local sparse-MLA variants warmed on rank %d (C%d caches, 1..%d rows)",
            ATTN_ROWLOCAL_LABEL, get_tensor_model_parallel_rank(), self.compress_ratio, max_rows,
        )

    def _rowlocal_prefill_topk_window(self, rocm_metadata, swa_metadata, k_cache, d, r0, r1):
        """m-itl-stack X1: the prefill top-k ragged stream for this rank's prefill rows [r0, r1)
        only (indptr from 0). C128's comes from step metadata, so it is built once per step and
        reused by every C128 layer of the step."""
        num_rows = k_cache.shape[0] * k_cache.shape[1]
        block_size = rocm_metadata.block_size // self.compress_ratio
        num_compressed = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        key = None
        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[d + r0:d + r1]
        else:
            topk_indices = rocm_metadata.c128a_prefill_topk_indices[r0:r1]
            key = (self.compress_ratio, d, r0, r1, num_rows, block_size, num_compressed)
            hit = rocm_metadata.__dict__.get("_rowlocal_topk_window")
            if hit is not None and hit[0] == key:
                return hit[1], hit[2]
        tk_idx, tk_ptr = build_prefill_topk_ragged_indices(
            topk_indices,
            swa_metadata.token_to_req_indices,
            swa_metadata.query_start_loc,
            swa_metadata.seq_lens,
            swa_metadata.is_valid_token,
            rocm_metadata.block_table,
            block_size=block_size,
            compress_ratio=self.compress_ratio,
            num_compressed=num_compressed,
            token_offset=d + r0,
            num_rows=num_rows,
        )
        if key is not None:
            object.__setattr__(rocm_metadata, "_rowlocal_topk_window", (key, tk_idx, tk_ptr))
        if self.compress_ratio not in _ROWLOCAL_RAGGED_WINDOW_LOGGED:
            _ROWLOCAL_RAGGED_WINDOW_LOGGED.add(self.compress_ratio)
            logger.info(
                "row-local prefill top-k ragged stream built for the rank's rows only "
                "(C%d%s) on rank %d",
                self.compress_ratio,
                ", once per step" if key is not None else "", get_tensor_model_parallel_rank(),
            )
        return tk_idx, tk_ptr

    def _rowlocal_mqa(self, q, o, a, b, attn_metadata) -> None:
        """Sparse MLA for rows [a, b): the ragged SWA / top-k lists are built for all decode or
        prefill rows as the served path builds them; each launch reads its rows' absolute
        indptr slice of the full index stream."""
        rocm_metadata = attn_metadata.get(self.prefix)
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None and rocm_metadata is not None
        k_cache = self.kv_cache
        d = swa_metadata.num_decode_tokens
        p = swa_metadata.num_prefill_tokens
        covered_lo, covered_hi = a, a
        dlo, dhi = a, min(b, d)
        if dhi > dlo:
            num_decodes = swa_metadata.num_decodes
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                tk_idx, tk_ptr, _ = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:d],
                    swa_metadata.token_to_req_indices,
                    rocm_metadata.block_table[:num_decodes],
                    rocm_metadata.block_size // self.compress_ratio,
                    swa_metadata.is_valid_token[:d],
                )
            else:
                tk_idx = rocm_metadata.c128a_decode_topk_ragged_indices
                tk_ptr = rocm_metadata.c128a_decode_topk_ragged_indptr
            self._rowlocal_sparse(
                q[dlo - a:dhi - a],
                o[dlo - a:dhi - a],
                swa_metadata.decode_swa_ragged_indices,
                swa_metadata.decode_swa_ragged_indptr[dlo:dhi + 1],
                k_cache,
                tk_idx,
                tk_ptr[dlo:dhi + 1],
            )
            covered_hi = dhi
        plo, phi = max(a, d), min(b, d + p)
        if phi > plo:
            r0, r1 = plo - d, phi - d
            tk_idx, tk_ptr = self._rowlocal_prefill_topk_window(
                rocm_metadata, swa_metadata, k_cache, d, r0, r1
            )
            self._rowlocal_sparse(
                q[plo - a:phi - a],
                o[plo - a:phi - a],
                swa_metadata.prefill_swa_ragged_indices,
                swa_metadata.prefill_swa_ragged_indptr[r0:r1 + 1],
                k_cache,
                tk_idx,
                tk_ptr,
            )
            covered_lo = a if dhi > dlo else plo
            covered_hi = phi
        # Rows with no attention (past the scheduled tokens) stay zero, not uninitialized.
        for lo, hi in ((0, covered_lo - a), (covered_hi - a, b - a)):
            if hi > lo:
                o[lo:hi].zero_()

    def _rowlocal_attn_out(self, q, a, b, attn_metadata, positions) -> torch.Tensor:
        """Sparse MLA on rows [a, b) and the output projection (row-local)."""
        o = torch.empty_like(q)
        self._rowlocal_mqa(q, o, a, b, attn_metadata)
        return self._rowlocal_o_proj(o, positions[a:b])

    def _rowlocal_o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale
        from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant

        o_fp8, o_scale = inverse_rope_group_quant(
            o,
            positions.to(torch.int64),
            self._wo_a_cos_cache,
            self._wo_a_sin_cache,
            num_groups=self.n_groups,
            quant_group_size=128,
        )
        zf = batched_gemm_a8w8_mxscale(
            o_fp8,
            self._wo_a_full_fp8_weight,
            o_scale,
            self._wo_a_full_e8m0_scale,
            dtype=o.dtype,
        ).flatten(1)
        return self._bpre_attn_gemm(
            self.wo_b_full.weight, self._wo_b_full_scale, zf, False, self._wo_b_full_scale_e8
        )


    def _prepare_and_attn(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor | None,
        kv: torch.Tensor | None,
        qr_scale: torch.Tensor | None,
        kv_score: torch.Tensor | None,
        indexer_kv_score: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        positions: torch.Tensor,
        o_padded: torch.Tensor,
    ) -> None:
        """Run the ROCm fork/join (HCA or CSA) inside the capture boundary."""
        aux_streams = self.aux_stream_list
        # The sequential pipeline disables aux_stream_list before calling
        # back with real projection inputs; aux_streams is None ends that
        # recursion here.
        if aux_streams is None:
            saved_streams = self.aux_stream_list
            self.aux_stream_list = None
            try:
                super()._prepare_and_attn(
                    hidden_states,
                    cast(torch.Tensor, qr),
                    cast(torch.Tensor, kv),
                    qr_scale,
                    cast(torch.Tensor, kv_score),
                    cast(torch.Tensor, indexer_kv_score),
                    cast(torch.Tensor, indexer_weights),
                    positions,
                    o_padded,
                )
            finally:
                self.aux_stream_list = saved_streams
            return

        # Re-check: forward's gate ran inside a captured segment that
        # _prepare_and_attn_eager (MRV1) then broke, making this region eager.
        if not current_platform.enable_multi_stream_overlap(
            self.aux_stream_list, get_forward_context().attn_metadata
        ):
            self._run_sequential_pipeline(hidden_states, positions, o_padded)
            return

        indexer = self.indexer
        compressor = self.compressor
        assert compressor is not None
        # Static per captured graph: each capture size bakes its own order.
        default_first = hidden_states.shape[0] <= _default_first_token_threshold()

        def default_chain():
            qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
            qr_out, qr_scale_out, kv_out = self._split_qkv_and_norm(qr_kv)
            q = self._wq_b_proj(qr_out, qr_scale_out).view(
                -1, self.n_local_heads, self.head_dim
            )
            attn_metadata = get_forward_context().attn_metadata
            q = self._fused_qnorm_rope_kv_insert(q, kv_out, positions, attn_metadata)
            return q, qr_out, qr_scale_out, kv_out

        def main_compressor_chain() -> None:
            score = torch.mm(
                hidden_states,
                compressor.fused_wkv_wgate.weight.T,
                out_dtype=torch.float32,
            )
            compressor(score, positions, self.rotary_emb)

        if indexer is None:
            # HCA dual-stream: the main compressor runs on aux stream 0 while
            # the default stream produces q and inserts KV into the SWA cache.
            # Both branches only read hidden_states, so the join merely has to
            # precede the sparse attention that consumes the compressed KV.
            (q, _qr_out, _qr_scale_out, kv_out), _ = execute_in_parallel(
                default_chain,
                [main_compressor_chain],
                self.ln_events[0],
                [self.ln_events[1]],
                aux_streams[:1],
                enable=True,
                default_first=default_first,
            )
            self._sparse_indexer_and_attn(
                hidden_states, None, None, None, q, kv_out, positions, o_padded
            )
            return

        def indexer_compressor_chain() -> None:
            score = torch.mm(
                hidden_states,
                indexer.compressor.fused_wkv_wgate.weight.T,
                out_dtype=torch.float32,
            )
            indexer.compressor(score, positions, self.indexer_rotary_emb)

        # CSA three-stream: the main and indexer compressors run on aux
        # streams 0 and 1 while the default stream produces q and inserts KV
        # into the SWA cache. Every branch only reads hidden_states plus its
        # own state, so the join merely has to precede the indexer op and the
        # sparse attention, which consume the compressed KV caches.
        (q, qr_out, qr_scale_out, kv_out), _ = execute_in_parallel(
            default_chain,
            [main_compressor_chain, indexer_compressor_chain],
            self.ln_events[0],
            self.ln_events[1:3],
            aux_streams[:2],
            enable=True,
            default_first=default_first,
        )

        indexer_weights_out, _ = indexer.weights_proj(hidden_states)
        # The indexer compressor already ran on aux stream 1; build queries only.
        index_q, index_q_scale, weights = indexer(
            hidden_states,
            qr_out,
            None,
            indexer_weights_out,
            positions,
            self.indexer_rotary_emb,
            qr_scale_out,
            skip_compressor=True,
        )
        self._sparse_indexer_and_attn(
            hidden_states,
            index_q,
            index_q_scale,
            weights,
            q,
            kv_out,
            positions,
            o_padded,
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def prepare_attn_preshuffle(self) -> None:
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return
        # aiter's gemm_a8w8_blockscale_bpreshuffle currently disabled on gfx1250.
        # Leaving the block scales as None -> "not preshuffled"
        # _fused_wqa_wkv_gemm and _o_proj both fall back to the standard linear path
        if on_gfx1250():
            return
        rowlocal_bytes = self._prepare_rowlocal_full() if self._rowlocal_full else 0
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
            get_fp8_block_weight_scale,
        )
        from vllm.model_executor.utils import replace_parameter

        e8_scales: dict[int, torch.Tensor] = {}

        def _prep(linear) -> torch.Tensor | None:
            w = getattr(linear, "weight", None)
            if w is None or w.dim() != 2:
                return None
            # K % 128 (group-128 quant) and N % 16 (shuffle_weight) must hold.
            if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
                return None
            ws = get_fp8_block_weight_scale(linear)
            if ws is None:
                return None
            if (
                active_attention_config().dsv4_attn_e8m0
                and _ON_GFX950
                and ws.dim() == 2
                and tuple(ws.shape) == (w.shape[0] // 128, w.shape[1] // 128)
            ):
                # m-attn-gemm-rate: keep the checkpoint's ue8m0 128x128 scales. The AITER
                # linear kernel's process_weights_after_loading has usually upcast them to
                # fp32 already (scaled_mm/aiter.py); recover the exponent bytes losslessly
                # (None when a scale is not an exact power of two: the fp32 route stays).
                e8 = _wo_a_block_scale_to_e8m0(ws)
                if e8 is not None:
                    e8_scales[id(linear)] = e8.view(torch.float8_e8m0fnu).contiguous()
            if ws.dtype == torch.float8_e8m0fnu:
                ws = _upcast_e8m0_to_fp32(ws).contiguous()
            # Shuffle the weight in place (single weight, no unshuffled copy),
            # unless the linear kernel already B-preshuffled it at load time.
            if not getattr(linear, "aiter_bpreshuffled", False):
                replace_parameter(
                    linear,
                    "weight",
                    rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)),
                )
            return ws

        self._wqa_wkv_scale = _prep(self.fused_wqa_wkv)
        self._wo_b_scale = _prep(self.wo_b)
        if self._rowlocal_full:
            self._wo_b_full_scale = _prep(self.wo_b_full)
            assert self._wo_b_full_scale is not None
        routed: list[str] = []
        if e8_scales:
            # Only where the fp32 route is preshuffled (same weight, same layout).
            for lin, scale_attr, e8_attr in (
                (self.fused_wqa_wkv, "_wqa_wkv_scale", "_wqa_wkv_scale_e8"),
                (self.wo_b, "_wo_b_scale", "_wo_b_scale_e8"),
                (getattr(self, "wo_b_full", None), "_wo_b_full_scale", "_wo_b_full_scale_e8"),
            ):
                e8 = e8_scales.get(id(lin)) if lin is not None else None
                if e8 is None or getattr(self, scale_attr, None) is None:
                    continue
                setattr(self, e8_attr, e8)
                routed.append(e8_attr)
                warmed = _attn_e8m0_warm(lin.weight, e8, self.max_num_batched_tokens)
                if warmed:
                    logger.info(
                        "%s (m-attn-gemm-rate): warmed %d FlyDSL e8m0 GEMM kernels "
                        "for N=%d K=%d (M<=%d) on rank %d",
                        ATTN_E8M0_LABEL, warmed, lin.weight.shape[0], lin.weight.shape[1],
                        self.max_num_batched_tokens, get_tensor_model_parallel_rank(),
                    )
        if active_attention_config().dsv4_attn_e8m0:
            global _ATTN_E8M0_PREP_LOGGED
            if not _ATTN_E8M0_PREP_LOGGED:
                _ATTN_E8M0_PREP_LOGGED = True
                logger.info(
                    "%s (m-attn-gemm-rate): e8m0 attention GEMM route %s on rank %d "
                    "(layer %s: %s)",
                    ATTN_E8M0_LABEL, "armed" if routed else "NOT reachable",
                    get_tensor_model_parallel_rank(), self.prefix,
                    ",".join(routed) if routed else "no ue8m0-recoverable preshuffled scale",
                )
        if _ON_GFX950 and envs.VLLM_ROCM_USE_AITER_FP8BMM:
            self._prepare_fp8_wo_a()
            if self._rowlocal_full and self._wo_a_fp8_weight is not None:
                self._prepare_rowlocal_fp8_wo_a()
        self._rowlocal_added_bytes = rowlocal_bytes

    def _prepare_rowlocal_fp8_wo_a(self) -> None:
        """m-attn-rowlocal step 2: the full wo_a as n_groups fp8 groups (same layout and e8m0
        scale encoding as the TP shard's _prepare_fp8_wo_a, whose groups are row views of it)."""
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            get_fp8_block_weight_scale,
        )

        weight = self.wo_a_full.weight
        scale = get_fp8_block_weight_scale(self.wo_a_full)
        if scale is None:
            scale = getattr(self.wo_a_full, "weight_scale", None)
        if scale is None:
            return
        e8m0 = _wo_a_block_scale_to_e8m0(scale)
        if e8m0 is None:
            return
        out_features, in_features = weight.shape
        groups, per = self.n_groups, self.o_lora_rank
        if out_features != groups * per or scale.shape != (out_features // 128, in_features // 128):
            return
        self._wo_a_full_fp8_weight = weight.view(groups, per, in_features)
        self._wo_a_full_e8m0_scale = e8m0.view(groups, per // 128, in_features // 128)

    def _prepare_fp8_wo_a(self) -> None:
        try:
            from aiter.ops.batched_gemm_op_a8w8 import (
                batched_gemm_a8w8_mxscale as mxscale_op,
            )
            from aiter.ops.inverse_rope_group_quant import (
                inverse_rope_group_quant as inverse_quant_op,
            )
        except ImportError:
            logger.warning_once(
                "The DeepSeek V4 FP8 WO_A path requires AITER >= 0.1.20; "
                "falling back to BF16 WO_A."
            )
            return
        del mxscale_op, inverse_quant_op

        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            get_fp8_block_weight_scale,
            is_fp8,
        )

        weight = getattr(self.wo_a, "weight", None)
        scale = get_fp8_block_weight_scale(self.wo_a)
        if scale is None:
            # ModelOpt MXFP8 stores the multiplicative E8M0 scale without the
            # historical ``_inv`` suffix.
            scale = getattr(self.wo_a, "weight_scale", None)
        if weight is None or scale is None:
            logger.warning_once(
                "DeepSeek V4 FP8 WO_A needs a block-scaled FP8 wo_a weight; "
                "the layer exposes no weight/weight scale. Falling back to "
                "BF16 WO_A."
            )
            return
        if weight.dim() != 2 or scale.dim() != 2 or not is_fp8(weight.dtype):
            logger.warning_once(
                "DeepSeek V4 FP8 WO_A needs a 2-D FP8 wo_a weight with a 2-D "
                "block scale, got weight %s%s and scale %s. Falling back to "
                "BF16 WO_A.",
                weight.dtype,
                tuple(weight.shape),
                tuple(scale.shape),
            )
            return

        groups = self.n_local_groups
        out_per_group = self.o_lora_rank
        out_features, in_features = weight.shape
        if (
            out_features != groups * out_per_group
            or out_per_group % 128 != 0
            or in_features % 128 != 0
            or scale.shape != (out_features // 128, in_features // 128)
        ):
            logger.warning_once(
                "DeepSeek V4 FP8 WO_A needs group-128 blocks for %d groups of "
                "%d outputs, got weight %s and scale %s. Falling back to BF16 "
                "WO_A.",
                groups,
                out_per_group,
                tuple(weight.shape),
                tuple(scale.shape),
            )
            return

        e8m0_scale = _wo_a_block_scale_to_e8m0(scale)
        if e8m0_scale is None:
            logger.warning_once(
                "DeepSeek V4 FP8 WO_A could not losslessly encode the %s wo_a "
                "block scale as OCP E8M0. Falling back to BF16 WO_A.",
                scale.dtype,
            )
            return

        self._wo_a_fp8_weight = weight.view(groups, out_per_group, in_features)
        self._wo_a_e8m0_scale = e8m0_scale.view(
            groups, out_per_group // 128, in_features // 128
        )
        self._wo_a_cos_cache, self._wo_a_sin_cache = _wo_a_rope_cos_sin(
            self.rotary_emb
        )

    def prepare_compressor_gemm_fusion(self) -> bool:
        if self._fused_compressor_weight is not None:
            return False

        from vllm.model_executor.offloader import NoopOffloader, get_offloader

        if not isinstance(get_offloader(), NoopOffloader):
            logger.warning_once(
                "DeepSeek V4 compressor GEMM fusion is incompatible with "
                "weight offloading and will remain disabled."
            )
            return False

        compressor = self.compressor
        indexer = self.indexer
        if compressor is None or indexer is None:
            return False

        main_weight = compressor.fused_wkv_wgate.weight
        indexer_weight = indexer.compressor.fused_wkv_wgate.weight
        if main_weight.ndim != 2 or indexer_weight.ndim != 2:
            raise ValueError("DeepSeek V4 compressor weights must be matrices")
        if main_weight.shape[1] != indexer_weight.shape[1]:
            raise ValueError("DeepSeek V4 compressor weights must share K")
        if main_weight.dtype != indexer_weight.dtype:
            raise ValueError("DeepSeek V4 compressor weights must share dtype")
        if main_weight.device != indexer_weight.device:
            raise ValueError("DeepSeek V4 compressor weights must share device")

        main_size = main_weight.shape[0]
        indexer_size = indexer_weight.shape[0]
        fused_weight = torch.cat((main_weight, indexer_weight), dim=0)
        with torch.no_grad():
            main_weight.set_(fused_weight[:main_size])
            indexer_weight.set_(fused_weight[main_size:])

        self._fused_compressor_weight = fused_weight
        self._fused_compressor_split_sizes = (main_size, indexer_size)
        return True

    def _bpre_attn_gemm(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        x: torch.Tensor,
        reduce_tp: bool,
        scale_e8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm._aiter_ops import rocm_aiter_ops

        if scale_e8 is not None and x.dtype == torch.bfloat16:
            # m-attn-gemm-rate: e8m0 1x128 act scales + the raw ue8m0 weight scales
            # -> AITER's FlyDSL mxscale bpreshuffle rows.
            global _ATTN_E8M0_LOGGED
            from aiter import dtypes
            from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
            from aiter.ops.quant import per_group_quant_hip

            x_fp8, x_scale = per_group_quant_hip(
                x.contiguous(), quant_dtype=dtypes.fp8, group_size=128,
                transpose_scale=True, scale_type=dtypes.fp8_e8m0,
            )
            out = gemm_a8w8_blockscale_bpreshuffle(
                x_fp8, weight, x_scale, scale_e8, x.dtype
            )
            if not _ATTN_E8M0_LOGGED:
                _ATTN_E8M0_LOGGED = True
                logger.info(
                    "%s (m-attn-gemm-rate): e8m0 attention GEMM (FlyDSL mxscale "
                    "bpreshuffle) dispatched on rank %d (M=%d N=%d K=%d, capturing=%s)",
                    ATTN_E8M0_LABEL, get_tensor_model_parallel_rank(), x.shape[0],
                    weight.shape[0], weight.shape[1],
                    torch.cuda.is_current_stream_capturing(),
                )
        else:
            x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
            out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
                x_fp8, weight, x_scale, scale, output_dtype=x.dtype
            )
        if reduce_tp and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._wqa_wkv_scale is not None and hidden_states.dim() == 2:
            from .seq_parallel_seam import sp_fp8_form, sp_fp8_handoff_check

            form = sp_fp8_form(hidden_states)
            if form is not None:
                # m-sp-dequant-skip: the SP gather's fp8 codes + scales, bitwise equal to the
                # per_1x128 requant _bpre_attn_gemm would compute from the dequantized rows.
                from vllm._aiter_ops import rocm_aiter_ops

                out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
                    form[0], self.fused_wqa_wkv.weight, form[1], self._wqa_wkv_scale,
                    output_dtype=hidden_states.dtype,
                )
                sp_fp8_handoff_check(
                    "fused_wqa_wkv", hidden_states, out,
                    lambda x: self._bpre_attn_gemm(
                        self.fused_wqa_wkv.weight, self._wqa_wkv_scale, x, False),
                )
                return out
            return self._bpre_attn_gemm(
                self.fused_wqa_wkv.weight, self._wqa_wkv_scale, hidden_states, False,
                self._wqa_wkv_scale_e8,
            )
        return super()._fused_wqa_wkv_gemm(hidden_states)

    def _run_parallel_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        fused_weight = self._fused_compressor_weight
        split_sizes = self._fused_compressor_split_sizes
        if fused_weight is None or split_sizes is None:
            return super()._run_parallel_input_projections(hidden_states)

        indexer = self.indexer
        if indexer is None:
            raise RuntimeError("Fused compressor weight requires a C4 indexer")

        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        fused_scores = torch.mm(
            hidden_states,
            fused_weight.T,
            out_dtype=torch.float32,
        )
        kv_score, indexer_kv_score = fused_scores.split(split_sizes, dim=-1)
        indexer_weights, _ = indexer.weights_proj(hidden_states)
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    @functools.cached_property
    def _wq_b_uses_aiter_block_scaled(self) -> bool:
        """True when both wq_b GEMMs run the aiter block-scaled fp8 kernel.

        Cached: the linear kernels and the aiter env gates are fixed once
        the model is built, so this is evaluated at the first forward
        only.

        The fused norm+quant path is only valid if the quant and GEMM it
        replaces are exactly the aiter ones; otherwise fall back to the
        shared path.
        """
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.kernels.linear.scaled_mm import (
            Fp8BlockScaledMMLinearKernel,
        )

        if not rocm_aiter_ops.is_linear_fp8_enabled():
            return False

        linears = [self.wq_b]
        if self.indexer is not None:
            linears.append(self.indexer.wq_b)
        for linear in linears:
            kernel = getattr(getattr(linear, "quant_method", None), "fp8_linear", None)
            if not isinstance(kernel, Fp8BlockScaledMMLinearKernel):
                return False
        return True

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Fuse q/kv RMSNorm + per-1x128 fp8 q quant into one aiter kernel.

        The shared path norms q and kv in one triton kernel and the wq_b
        linears then re-read the bf16 qr to quantize it. The aiter kernel
        computes both RMSNorms (fp32 accumulate) and the fp8 group quant
        in a single pass, writing fp8 qr + group scales directly; both
        wq_b GEMMs (attention and indexer) then consume that pair and
        skip their own input quant. kv stays bf16: the fused insert
        kernel RoPE/quantizes it itself. Falls back to the shared path
        when the aiter linear path is not active.
        """
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        if not (
            qr.dim() == 2
            and qr.shape[0] > 0
            and self.q_lora_rank % 128 == 0
            and self._wq_b_uses_aiter_block_scaled
        ):
            return super()._split_qkv_and_norm(qr_kv)

        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.fused_qk_rmsnorm_group_quant(
            q=qr,
            q_weight=self.q_norm.weight.data,
            q_epsilon=self.eps,
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            kv_epsilon=self.eps,
            group_size=128,
            transpose_scale=False,
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self._wo_a_fp8_weight is not None:
            from aiter.ops.batched_gemm_op_a8w8 import (
                batched_gemm_a8w8_mxscale,
            )
            from aiter.ops.inverse_rope_group_quant import (
                inverse_rope_group_quant,
            )

            assert self._wo_a_cos_cache is not None
            assert self._wo_a_sin_cache is not None
            o_fp8, o_scale = inverse_rope_group_quant(
                o.view(o.shape[0], self.n_local_heads, self.head_dim),
                positions.to(torch.int64),
                self._wo_a_cos_cache,
                self._wo_a_sin_cache,
                num_groups=self.n_local_groups,
                quant_group_size=128,
            )
            assert self._wo_a_e8m0_scale is not None
            zf = batched_gemm_a8w8_mxscale(
                o_fp8,
                self._wo_a_fp8_weight,
                o_scale,
                self._wo_a_e8m0_scale,
                dtype=o.dtype,
            ).flatten(1)
        else:
            # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.
            z = rocm_inv_rope_einsum(
                self.rotary_emb,
                o,
                positions,
                self.rope_head_dim,
                self.n_local_groups,
                self.o_lora_rank,
                self.wo_a,
                inverse_rope=False,
            )
            zf = z.flatten(1)
        # Sequence-parallel steps (m-seq-parallel) reduce-scatter the partial sum themselves.
        reduce_tp = not self._sp_defer_reduce
        if self._wo_b_scale is not None and zf.dim() == 2:
            return self._bpre_attn_gemm(
                self.wo_b.weight, self._wo_b_scale, zf, reduce_tp, self._wo_b_scale_e8
            )
        if not reduce_tp:
            raise RuntimeError("sequence-parallel attention needs the preshuffled wo_b")
        return self.wo_b(zf)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step. The aiter prefill
            # reads the caches in place and never asks for it.
            if not self._use_aiter_sparse_mla:
                swa_only = self.compress_ratio <= 1
                N = (
                    0
                    if swa_only
                    else (self.max_model_len + self.compress_ratio - 1)
                    // self.compress_ratio
                )
                M = N + self.window_size + self.max_num_batched_tokens
                current_workspace_manager().get_simultaneous(
                    ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
        # The fp8 wo_a path rotates inside inverse_rope_group_quant, so folding
        # the rotation into the decode reduce would apply it twice. Only the
        # BF16 einsum path hands its rotation off to the decode.
        fuse_inv_rope = self._wo_a_fp8_weight is None
        rotated = 0
        if num_decodes > 0:
            fuse_decode_inv_rope = fuse_inv_rope and not on_gfx1250()
            rotated = self._forward_decode(
                q=q[:num_decode_tokens],
                positions=(
                    positions[:num_decode_tokens] if fuse_decode_inv_rope else None
                ),
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
                adaptive_splits=(
                    _ON_GFX950
                    and not swa_only
                    and self.compress_ratio == 128
                    and rocm_metadata is not None
                    and rocm_metadata.for_cudagraph_capture
                ),
            )
        if fuse_inv_rope:
            # Only the decode reduce rotates its own rows, and only the leading
            # `rotated` of them; prefill rows and any decode path that did not
            # fuse still owe the standalone pass. Settle that here rather than
            # in _o_proj: the split is batch-dependent and _o_proj runs
            # compiled, where such a value freezes at its trace-time value.
            rocm_inverse_rope_rows_(
                output[rotated:, : self.n_local_heads, :],
                positions[rotated:],
                self.rotary_emb.cos_sin_cache,
                self.rope_head_dim,
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        positions: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
        adaptive_splits: bool,
    ) -> int:
        """Returns how many leading rows the decode epilogue inverse-RoPE'd."""
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                (
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    topk_lens,
                ) = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr

        if self._use_aiter_sparse_mla:
            assert swa_metadata.decode_swa_ragged_indices is not None
            assert swa_metadata.decode_swa_ragged_indptr is not None
            self._aiter_sparse_mla(
                q,
                output,
                self.swa_cache_layer.kv_cache,
                swa_metadata.decode_swa_ragged_indices,
                swa_metadata.decode_swa_ragged_indptr,
                kv_cache,
                topk_ragged_indices,
                topk_ragged_indptr,
            )
            # The aiter kernel leaves the inverse RoPE to the caller.
            return 0

        return rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
            adaptive_splits=adaptive_splits,
            inv_rope_positions=positions,
            inv_rope_cos_sin_cache=self.rotary_emb.cos_sin_cache,
            extra_cache_nan_free=_trust_dsv4_extra_cache_nan_free(
                self.kv_cache_dtype,
                self._has_kv_transfer,
                not swa_only and kv_cache is not None,
            ),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        if self._use_aiter_sparse_mla:
            self._forward_prefill_aiter(
                q=q,
                compressed_k_cache=compressed_k_cache,
                swa_k_cache=swa_k_cache,
                output=output,
                attn_metadata=attn_metadata,
                swa_metadata=swa_metadata,
            )
            return

        swa_only = attn_metadata is None

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]
        left_visible = swa_metadata.prefill_left_visible
        right_visible = swa_metadata.prefill_right_visible
        if left_visible is not None:
            left_visible = left_visible[num_decode_tokens:]
            assert right_visible is not None
            right_visible = right_visible[num_decode_tokens:]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0

        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
            )[0]
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                # compressed_k_cache is OCP on every platform (Triton encoder).
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
                max_image_tokens=self.max_image_tokens,
                left_visible=(
                    left_visible[query_start:query_end]
                    if left_visible is not None
                    else None
                ),
                right_visible=(
                    right_visible[query_start:query_end]
                    if right_visible is not None
                    else None
                ),
            )
            rocm_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )

    def _forward_prefill_aiter(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        """Prefill straight off the paged caches, with no bf16 gather."""
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        assert swa_metadata.prefill_swa_ragged_indices is not None
        assert swa_metadata.prefill_swa_ragged_indptr is not None

        topk_ragged_indices = None
        topk_ragged_indptr = None
        if attn_metadata is None:
            compressed_k_cache = None
        else:
            assert compressed_k_cache is not None
            assert swa_metadata.token_to_req_indices is not None
            assert swa_metadata.query_start_loc is not None
            assert swa_metadata.seq_lens is not None
            assert swa_metadata.is_valid_token is not None
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ]
            else:
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            topk_ragged_indices, topk_ragged_indptr = build_prefill_topk_ragged_indices(
                topk_indices,
                swa_metadata.token_to_req_indices,
                swa_metadata.query_start_loc,
                swa_metadata.seq_lens,
                swa_metadata.is_valid_token,
                attn_metadata.block_table,
                block_size=attn_metadata.block_size // self.compress_ratio,
                compress_ratio=self.compress_ratio,
                num_compressed=(self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio,
                token_offset=num_decode_tokens,
                num_rows=compressed_k_cache.shape[0] * compressed_k_cache.shape[1],
            )

        self._aiter_sparse_mla(
            q[:num_prefill_tokens],
            output[:num_prefill_tokens],
            swa_k_cache,
            swa_metadata.prefill_swa_ragged_indices,
            swa_metadata.prefill_swa_ragged_indptr,
            compressed_k_cache,
            topk_ragged_indices,
            topk_ragged_indptr,
        )

    def _aiter_sparse_mla(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_indptr: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
        topk_indptr: torch.Tensor | None,
    ) -> None:
        from vllm._aiter_ops import rocm_aiter_ops

        # Both fp8 caches are read in place: the SWA window as the main segment,
        # the top-k compressed tokens as the extra one.
        rocm_aiter_ops.triton_sparse_mla_fwd(
            q,
            swa_k_cache,
            output,
            self.scale,
            swa_indptr,
            swa_indices,
            attn_sink=self.attn_sink[: q.shape[1]],
            extra_kv_buffer=compressed_k_cache,
            extra_kv_indptr=topk_indptr,
            extra_kv_indices=topk_indices,
        )


class DeepseekV4RocmMxfp4Compressor(DeepseekCompressor):
    """The indexer's compressor writing aiter's preshuffled MXFP4 pages: a
    pooling kernel, then aiter's norm + RoPE + MXFP4 cache op, which owns the
    page layout its MQA-logits kernel reads."""

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any) -> None:
        super().__init__(vllm_config, **kwargs)
        assert self.head_dim == 128 and self.use_fp4_cache
        self.num_index_heads = vllm_config.model_config.hf_config.index_n_heads
        if vllm_config.kernel_config.enable_jit_warmup:
            # Compile at startup instead of on the first boundary token.
            compress_pool.register_warmup(
                head_dim=self.head_dim,
                state_width=self.coff * self.head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
            )

    def _pooled_k(self, num_tokens: int) -> torch.Tensor:
        # A persistent buffer rather than the shared scratch: in the CSA fork
        # this runs on an aux stream next to other branches. The indexer
        # layers run one after another, so they can all share it.
        scratch = current_workspace_manager().get_persistent(
            "dsv4_rocm_mxfp4_pooled_k",
            (self.max_num_batched_tokens, self.head_dim),
            torch.bfloat16,
        )
        assert num_tokens <= scratch.shape[0]
        return scratch[:num_tokens]

    def forward(
        self, kv_score: torch.Tensor, positions: torch.Tensor, rotary_emb
    ) -> None:
        # Claim the scratch on the profiling run, before the workspace locks.
        self._pooled_k(0)
        super().forward(kv_score, positions, rotary_emb)

    def _compress_norm_rope_store(
        self,
        state_metadata: CompressorMetadata,
        state_cache: torch.Tensor,
        state_width: int,
        positions: torch.Tensor,
        rotary_emb,
        attn_metadata: dict[str, Any],
        pdl_kwargs: dict,
    ) -> None:
        num_tokens = state_metadata.slot_mapping.shape[0]
        if num_tokens == 0:
            return
        k_cache_metadata = attn_metadata[self.k_cache_prefix]
        kv_slot_mapping = k_cache_metadata.slot_mapping[:num_tokens]
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache
        pooled_k = self._pooled_k(num_tokens)
        compress_pool(
            state_cache,
            state_metadata.token_to_req_indices,
            positions,
            kv_slot_mapping,
            state_metadata.block_table,
            state_metadata.block_size,
            pooled_k,
            state_width,
            self.compress_ratio,
            self.overlap,
        )
        rocm_mxfp4_indexer_k_store(
            pooled_k,
            positions[:num_tokens],
            rotary_emb.cos_sin_cache,
            self.norm.weight,
            self.rms_norm_eps,
            kv_cache,
            kv_slot_mapping,
            self.compress_ratio,
            self.use_fp4_cache,
            num_heads=self.num_index_heads,
        )


class DeepseekV4RocmMxfp4IndexerCache(DeepseekV4IndexerCache):
    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4RocmMxfp4IndexerBackend


class DeepseekV4RocmMxfp4Indexer(DeepseekV4Indexer):
    """`DeepseekV4Indexer` on aiter's paged MXFP4 cache, scoring the whole
    context in place (V4 has no candidate pool)."""

    def _get_cache_cls(self) -> type[DeepseekV4IndexerCache]:
        return DeepseekV4RocmMxfp4IndexerCache

    def _get_compressor_cls(self) -> type[DeepseekCompressor]:
        return DeepseekV4RocmMxfp4Compressor

    def _get_attn_cls(self) -> type[SparseAttnIndexer]:
        return RocmSparseAttnIndexer

    def _register_q_quant_warmup(self) -> None:
        # The shared MXFP4 Q kernel packs FP4 with CUDA PTX; aiter's op runs here.
        pass

    def _q_rope_quant(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        rotary_emb: torch.nn.Module,
        indexer_weights: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # aiter's MQA-logits kernel takes fp32 weights with both scales folded.
        return rocm_mxfp4_indexer_q_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            self.softmax_scale,
            self.n_head**-0.5,
        )
