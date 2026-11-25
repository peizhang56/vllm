# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Any, Literal

from pydantic import Field, field_validator

from vllm.config.utils import config
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

IndexerKVDType = Literal["auto", "bf16", "fp8", "mxfp4", "nvfp4"]
MiniMaxM3MSADecodeBackend = Literal["triton", "cutlass"]


@config
class HiSparseConfig:
    """Configuration for HiSparse sparse-MLA KV offloading."""

    device_buffer_size: int | None = Field(default=None, gt=0)
    """Total per-request GPU hot-buffer rows, including the newest-token slot.

    Defaults to one top-k per decode query plus one top-k of LRU slack. The
    physical allocation is rounded up to the GPU cache block size selected
    from the active backends.
    """

    eager_host_mirror: bool = True
    """Mirror decode-written KV rows to the host pool during the forward so
    page spills complete without moving data. When disabled, decode rows stay
    resident-only and evicted pages are copied to host at spill time. Prefill
    rows are always mirrored during the forward."""


@config
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. Use "auto" or None for automatic selection."""

    minimax_m3_msa_decode_backend: MiniMaxM3MSADecodeBackend = "triton"
    """Sparse decode kernel used by the MiniMax M3 MSA backend."""

    backend_per_kind: dict[str, AttentionBackendEnum] = field(default_factory=dict)
    """Per-KV-cache-group attention backend overrides, keyed by
    `KVCacheSpecKind` (e.g. `{"mla_attention": "FLASHINFER_MLA",
    "sliding_window_mla": "TRITON_MLA"}`). This lets a model that splits its
    layers across multiple KV-cache groups (e.g. interleaved full and
    sliding-window attention) use a different backend per group.

    An entry overrides `backend` for layers of the matching kind; kinds not
    listed fall back to `backend` (or automatic selection). A selected backend
    that is invalid for that kind raises at startup."""

    flash_attn_version: Literal[2, 3, 4] | None = None
    """Force vllm to use a specific flash-attention version (2, 3, or 4).
    Only valid when using the flash-attention backend."""

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    tokenspeed_mla_min_split_kv: int = Field(default=1, ge=1, le=256)
    """Minimum KV splits for TokenSpeed MLA decode. The default of 1 preserves
    automatic scheduling. Larger values request a split floor and may increase
    workspace memory. Only applies to the TokenSpeed MLA backend."""

    tq_max_kv_splits_for_cuda_graph: int = 32
    """TurboQuant max NUM_KV_SPLITS for cuda graph decode.
    Fixes the split count so grid dimensions are constant across captures,
    and buffers can be pre-allocated to avoid inflating the memory estimate."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""

    mla_prefill_backend: MLAPrefillBackendEnum | None = None
    """MLA prefill backend to use. If None, will be selected automatically.
    Valid options: FLASH_ATTN (FA3/FA4), FLASHINFER, TRTLLM_RAGGED."""

    use_prefill_query_quantization: bool = False
    """If set, quantize query for attention in prefill."""

    trtllm_ragged_backend: Literal["trtllm-gen", "cute-dsl"] = "trtllm-gen"
    """Kernel backend for cached-context TRTLLM_RAGGED MLA prefill.
    CuTe DSL requires FP8 queries for head dimension 192 and does not support
    head dimension 256. New-token prefill always uses TRTLLM."""

    indexer_kv_dtype: IndexerKVDType | str = "auto"
    """Data type for the sparse-attention indexer K cache. "auto" picks the
    model's default (bf16 for MiniMax M3, fp8 for the DeepSeek sparse
    indexer). Quantized formats (fp8, mxfp4, nvfp4) require indexer kernel
    support in the backend."""

    indexer_sparse_logits: bool = False
    """DeepSeek V4.1 two-level indexer: score only the candidate blocks with
    DeepGEMM's sparse MQA-logits kernels instead of computing dense logits over
    the whole context and masking them. Requires `indexer_kv_dtype="mxfp4"`,
    an SM100-class GPU, DeepGEMM >= 2.8 and the DeepSelect top-k extension
    (the top-k runs on the kernels' bf16 logits). The sparse path costs
    O(candidate blocks) per query regardless of context length, so it pays off
    for long contexts (roughly 32K tokens and beyond) and is slower below.
    On ROCm gfx950 it runs aiter's paged MXFP4 MQA-logits kernel over the
    candidate pool instead, with fp32 logits, and keeps the masked dense walk
    for steps whose contexts are too short for the pool to pay."""

    indexer_tp_shard_min_rows: int = Field(default=65536, ge=1)
    """Minimum total prefill rows for the sparse-attention indexer to split
    them across TP ranks instead of scoring every row on each rank (CUDA)."""

    sparse_mla_shard_q: bool = True
    """Under tensor parallelism, shard the sparse-attention indexer prefill by
    token: each TP rank scores its zigzag slice of the prefill tokens and the
    ranks all-gather the int32 top-k back into token order. On ROCm,
    DeepSeek-V4 and V4.1 shard only their indexer prefill this way; their
    attention stays sharded by head. On DeepSeek-V4 steps that run
    sequence-parallel with row-local attention (`dsv4_rowlocal_input`), each
    rank scores its own sequence-parallel rows instead and nothing is
    exchanged. Off under DCP or PCP, and when the mixed CUDA-graph mode is
    FULL."""

    sparse_mla_shard_q_min_tokens: int = Field(default=1024, ge=1)
    """Minimum number of tokens after a step's decodes (requests of at most
    1 + num_speculative_tokens tokens) for `sparse_mla_shard_q` to apply. Below
    it the exchange can cost more than it saves."""

    minimax_m3_indexer_cp: bool = False
    """MiniMax M3 on ROCm with tensor_parallel_size > 1: score the sparse
    indexer context-parallel at decode. Each rank reads 1/TP of the index
    cache, scores every index head on it, and per-head top-k candidates are
    exchanged and merged, so the selected blocks match the default path.
    `indexer_kv_dtype` picks the kernels that do it: a bf16 index cache ("auto"
    or "bf16") runs the Triton CP indexer, an fp8 e4m3 one the AITER MSA CP
    indexer on gfx950. This shards indexer *work* and not the KV cache, so it
    is not `decode_context_parallel_size` and requires that to be 1."""

    hisparse_config: HiSparseConfig | None = None
    """HiSparse host-resident KV configuration. Setting this enables experimental
    Model Runner V2-only HiSparse sparse-MLA decode hot-buffering. It is inferred
    with defaults when HiSparseConnector is configured (directly or via
    MultiConnector); set it explicitly only to tune its fields."""

    dsv4_seq_parallel: bool = False
    """DeepSeek-V4 on ROCm gfx950 (TP8): sequence-parallel mHC seams on steps of at
    least 1024 rows (INT4 reduce-scatter, fp8 gather, and the router, compressor
    and reduce folds). Ignored, with the reasons logged, when the deployment cannot
    run it. No effect on other models."""

    dsv4_rowlocal_attention: bool = False
    """DeepSeek-V4 on ROCm: row-local attention on sequence-parallel steps. Each rank
    loads the full wq_b, wo_a and wo_b (about 15 GiB of KV cache per GPU). Needs
    dsv4_seq_parallel."""

    dsv4_rowlocal_input: bool = False
    """DeepSeek-V4 on ROCm: on a row-local sequence-parallel step, skip the attention
    layer-input gather and score the indexer prefill on the rank's own rows
    instead of its `sparse_mla_shard_q` split. Needs dsv4_rowlocal_attention and
    sparse_mla_shard_q."""

    dsv4_attn_e8m0: bool = False
    """DeepSeek-V4 on ROCm: e8m0 scale route for the attention GEMMs (FlyDSL mxscale
    bpreshuffle, gfx950)."""

    dsv4_sp_self_check: bool = False
    """Debug only: one-time bitwise self-checks of the sequence-parallel path against
    the unfused path. A check that fails on one rank raises there while the other
    ranks wait in the next collective."""

    kv_zero_coalesce: bool = False
    """Zero each newly allocated KV block's bytes once instead of once per overlapping
    per-layer page. Exact: the same kernel writes the same bytes."""

    use_non_causal: bool = False
    """Whether to use non-causal (bidirectional) attention."""

    sparse_mla_force_mqa: bool = False
    """Force sparse MLA to use forward_mqa for all requests, including prefill.
    When False (default), pure prefill batches use forward_mha when implemented.
    Set to True to always use the MQA path."""

    sparse_mla_prefill_workspace_factor: int | None = Field(default=None, ge=2)
    """Sparse MLA prefill workspace size, in multiples of `max_model_len`.
    FlashMLA sparse gathers a prefill step's context KV into this workspace and
    runs attention in more chunks when the step holds more. The sparse indexer's
    prefill K buffer is sized to fit in the same memory. Lower values return
    memory to the KV cache at long `max_model_len`; the minimum of 2 leaves room
    for a full-length request's context-parallel shard. If None, 5 when
    `max_model_len` is at most 256K tokens, else 2."""

    flex_attn_block_m: int | None = None
    """Triton kernel BLOCK_M tile size for flex attention.
    Must be a power of 2 >= 16. If None and VLLM_BATCH_INVARIANT=1,
    defaults to 16."""

    flex_attn_block_n: int | None = None
    """Triton kernel BLOCK_N tile size for flex attention.
    Must be a power of 2 >= 16. If None and VLLM_BATCH_INVARIANT=1,
    defaults to 16."""

    flex_attn_q_block_size: int | None = None
    """Logical Q block size for the flex attention block mask.
    Must be a power of 2 and divisible by flex_attn_block_m.
    If None, uses 16 for paged KV attention on PyTorch >= 2.9, and 128
    for encoder-only attention or older PyTorch versions."""

    flex_attn_kv_block_size: int | None = None
    """Logical KV block size for the flex attention block mask.
    Must be a power of 2 and divisible by flex_attn_block_n.
    If None, uses the KV cache block size for paged KV attention on
    PyTorch >= 2.9, and 128 for encoder-only attention or older PyTorch
    versions."""

    def __post_init__(self) -> None:
        msa_aliases: dict[AttentionBackendEnum, MiniMaxM3MSADecodeBackend] = {
            AttentionBackendEnum.CUTLASS_MSA: "cutlass",
            AttentionBackendEnum.TRITON_MSA: "triton",
        }
        if self.backend in msa_aliases:
            self.minimax_m3_msa_decode_backend = msa_aliases[self.backend]
            # The alias selects only MiniMax's sparse decode kernel. Dense
            # layers still use the platform's normal automatic backend.
            self.backend = None
        # Checked here rather than at indexer selection, which the two CP
        # backends reach separately: bf16 goes to the Triton indexer and fp8
        # to the AITER MSA one, so neither ever sees the other's dtype and
        # neither can speak for the dtypes that have no CP indexer at all.
        if self.minimax_m3_indexer_cp and self.indexer_kv_dtype not in (
            "auto",
            "bf16",
            "fp8",
            "fp8_e4m3",
        ):
            raise ValueError(
                "minimax_m3_indexer_cp requires a bf16 or fp8 e4m3 index cache "
                "(indexer_kv_dtype 'auto', 'bf16', 'fp8' or 'fp8_e4m3'), got "
                f"{self.indexer_kv_dtype!r}"
            )

    def resolve_indexer_kv_dtype(
        self, default: IndexerKVDType | str
    ) -> IndexerKVDType | str:
        """Resolve `indexer_kv_dtype`, substituting `default` for "auto"."""
        if self.indexer_kv_dtype == "auto":
            return default
        return self.indexer_kv_dtype

    def compute_hash(self) -> str:
        """Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, set())
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string.

        The special value "auto" is treated as None, which triggers
        automatic backend selection.
        """
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value

    @field_validator("mla_prefill_backend", mode="before")
    @classmethod
    def validate_mla_prefill_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `mla_prefill_backend` enum type from string."""
        if isinstance(value, str):
            return MLAPrefillBackendEnum[value.upper()]
        return value

    @field_validator("backend_per_kind", mode="before")
    @classmethod
    def validate_backend_per_kind_before(cls, value: Any) -> Any:
        """Parse the `backend_per_kind` map from strings.

        Keys must be valid `KVCacheSpecKind` values; values are parsed like
        `backend` (enum name, case-insensitive).
        """
        from vllm.v1.kv_cache_interface import KVCacheSpecKind

        if not isinstance(value, dict):
            return value
        valid_kinds = {kind.value for kind in KVCacheSpecKind}
        parsed: dict[str, AttentionBackendEnum] = {}
        for kind, backend in value.items():
            if kind not in valid_kinds:
                raise ValueError(
                    f"Unknown KV cache group kind '{kind}' in "
                    f"backend_per_kind. Valid kinds are: "
                    f"{', '.join(sorted(valid_kinds))}."
                )
            if isinstance(backend, str):
                backend = AttentionBackendEnum[backend.upper()]
            parsed[kind] = backend
        return parsed


# The DeepSeek-V4 switches above are read by the model code through the attention config of
# the worker process (set in Worker.init_device before the model is built). They are not
# environment variables.
_active_attention_config: AttentionConfig | None = None


def set_active_attention_config(cfg: AttentionConfig) -> None:
    global _active_attention_config
    _active_attention_config = cfg


def active_attention_config() -> AttentionConfig:
    """The attention config of this process (defaults if none was set)."""
    if _active_attention_config is not None:
        return _active_attention_config
    return AttentionConfig()
