# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import typing
import weakref
from collections.abc import Callable, Iterable
from dataclasses import replace
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.aiter_flydsl import (
    AiterFlyDSLMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
    RocmDotScaledMxfp8LinearKernel,
)
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    FusedMoEQuantConfig,
    GateLinear,
    MoERunner,
    RoutedExperts,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
    derive_aiter_quant_type,
    mxfp4_w4a16_extra_kwargs,
    rocm_aiter_fused_experts,
    set_rocm_aiter_fused_moe_input_hook,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.utils import (
    resolve_layer_fused_shared_expert,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import (
    HAS_AITER_MHC,
    HAS_TILELANG_MHC,
    HCHeadOp,
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    get_fused_shared_expert_layers,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    refill_derived,
    replace_parameter,
)
from vllm.models.deepseek_v4.amd.mega_moe import (
    DeepseekV4AiterMegaMoEExperts,
    finalize_mega_moe_weights,
)
from vllm.models.deepseek_v4.amd.mono_decode import MonoDecodeLayer
from vllm.models.deepseek_v4.amd.rocm import (
    ATTN_ROWLOCAL_LABEL,
    DeepseekV4ROCMAiterMLAAttention,
)
from vllm.models.deepseek_v4.amd.seq_parallel_seam import (
    GATHER_TAIL_BYTES,
    SEQ_PARALLEL_LABEL,
    SEQ_PARALLEL_MIN_ROWS,
    SP_MOE_LABEL,
    SP_ROUTE_FOLD_LABEL,
    SP_ROUTER_SHARD_LABEL,
    SP_SHARED_ROWLOCAL_LABEL,
    SeqParallelSeamComm,
    SeqParallelStep,
    routing_tail_bytes,
    set_sp_self_check,
    sp_config_disable_reasons,
    sp_fp8_form,
    sp_fp8_handoff_check,
    sp_fp8_handoff_clear,
    sp_fused_moe_input_hook,
    sp_moe_codes_done,
    sp_moe_codes_wanted,
    sp_route_capture,
    sp_self_check_enabled,
)
from vllm.models.deepseek_v4.amd.silu_and_mul_mxfp8 import silu_and_mul_mxfp8
from vllm.models.deepseek_v4.common.spec_decode import needs_mtp_hidden_buffer
from vllm.models.deepseek_v4.compressor import dsv4_swa_replay_enabled
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950, on_gfx1250
from vllm.sequence import IntermediateTensors
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream
from vllm.v1.worker.ubatching import dbo_current_ubatch_token_offset

from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

logger = init_logger(__name__)

_MXFP8_INPUT_ATTR = "_rocm_mxfp8_input"


def attach_mxfp8_input(x: torch.Tensor, qa: QuantizedActivation) -> None:
    """Carry the MXFP8 copy of a sublayer input on the bf16 tensor itself.

    Any reshape, slice or gather makes a new tensor without it, so a consumer
    only ever sees the quantized copy of the exact rows it projects.
    """
    setattr(x, _MXFP8_INPUT_ATTR, qa)


def mxfp8_input(x: torch.Tensor) -> QuantizedActivation | None:
    return getattr(x, _MXFP8_INPUT_ATTR, None)


def takes_rocm_mxfp8_input(linear: nn.Module) -> bool:
    """Whether `linear` reads a pre-quantized MXFP8 activation with row-major
    E8M0 scales as is (the quant key does not encode the scale layout)."""
    kernel = getattr(getattr(linear, "quant_method", None), "kernel", None)
    return (
        get_input_quant_key(linear) == kMxfp8Dynamic
        and isinstance(
            kernel, (RocmDotScaledMxfp8LinearKernel, AiterFlyDSLMxfp8LinearKernel)
        )
        and kernel.supports_quantized_input(linear)
    )


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

        # gate_up_proj B-preshuffle (ColumnParallel -> no all-reduce); set at load.
        self._gateup = rocm_aiter_ops.is_enabled()
        # Block scale for the preshuffled gate_up weight; None = not preshuffled.
        self._gateup_scale: torch.Tensor | None = None
        # Whether down_proj reads row-major MXFP8 as is; resolved on first use,
        # after the weights are processed.
        self._down_takes_mxfp8: bool | None = None
        # dsv4_sp_moe: the full (non-TP) copy of the shared expert also
        # preshuffles down_proj (forward_rowlocal); the TP copy skips the calls
        # whose shared term the full copy computes on the rank's own rows.
        self._rowlocal_full = False
        self._down_scale: torch.Tensor | None = None
        self._sp_skip = False

    def prepare_gateup_preshuffle(self) -> None:
        # B-preshuffle the gate_up_proj weight in place (single weight).
        if not self._gateup:
            return
        # aiter's gemm_a8w8_blockscale_bpreshuffle currently disabled on gfx1250.
        # Leaving _gateup_scale as None -> "not preshuffled"
        # forward() falls back to the standard gate_up_proj linear
        if on_gfx1250():
            return
        ws = self._preshuffle_linear(self.gate_up_proj)
        if ws is None:
            return
        # A plain tensor (the scale Parameter would register a second time on
        # this module), refilled in place on reloads for captured CUDA graphs.
        refill_derived(self, "_gateup_scale", ws.detach())
        if self._rowlocal_full:
            ws = self._preshuffle_linear(self.down_proj)
            if ws is not None:
                refill_derived(self, "_down_scale", ws.detach())

    @staticmethod
    def _preshuffle_linear(linear: nn.Module) -> torch.Tensor | None:
        """B-preshuffle a block-fp8 linear's weight in place and return its fp32
        block scale; None if the weight does not qualify."""
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
            get_fp8_block_weight_scale,
        )

        w = getattr(linear, "weight", None)
        ws = get_fp8_block_weight_scale(linear)
        if w is None or ws is None or w.dim() != 2:
            return None
        # K % 128 (group-128 quant) and N % 16 (shuffle_weight) must hold.
        if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
            return None
        if ws.dtype == torch.float8_e8m0fnu:
            ws = _upcast_e8m0_to_fp32(ws).contiguous()
        # Skip when the linear kernel already B-preshuffled this weight.
        if not (
            getattr(linear, "aiter_bpreshuffled", False) or is_weights_pre_processed()
        ):
            replace_parameter(
                linear,
                "weight",
                rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)),
            )
        return ws

    def forward_rowlocal(self, x: torch.Tensor) -> torch.Tensor:
        """The full shared expert (no TP) on this rank's own rows, both GEMMs
        on the AITER fp8 block-scale B-preshuffle kernel gate_up uses."""
        if self._gateup_scale is None or self._down_scale is None:
            return self.forward(x)
        gate_up = self._gateup_requant(x)
        h = self.act_fn(gate_up)
        h_fp8, h_scale = rocm_aiter_ops.group_fp8_quant(h, transpose_scale=True)
        return rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            h_fp8,
            self.down_proj.weight,
            h_scale,
            self._down_scale,
            output_dtype=x.dtype,
        )

    def _gateup_requant(self, x: torch.Tensor) -> torch.Tensor:
        # gate_up via fp8 group-quant (col-major) + B-preshuffle GEMM.
        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
        return rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_fp8,
            self.gate_up_proj.weight,
            x_scale,
            self._gateup_scale,
            output_dtype=x.dtype,
        )

    def forward(self, x):
        if self._sp_skip:
            # dsv4_sp_moe: the full copy computes this call's shared term on
            # the rank's own rows; the MoE runner hands this placeholder back
            # unsummed (defer_shared_add) and the caller drops it.
            return torch.empty_like(x)
        if self._gateup_scale is not None and x.dim() == 2:
            form = sp_fp8_form(x)
            if form is not None:
                # dsv4_seq_parallel: the layer-input gather's fp8 codes and
                # scales, bitwise the requant _gateup_requant does.
                gate_up = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
                    form[0],
                    self.gate_up_proj.weight,
                    form[1],
                    self._gateup_scale,
                    output_dtype=x.dtype,
                )
                sp_fp8_handoff_check("shared_gate_up", x, gate_up, self._gateup_requant)
            else:
                gate_up = self._gateup_requant(x)
        else:
            # The fused mHC seam may hand over x already quantized to MXFP8.
            qa = mxfp8_input(x)
            gate_up, _ = self.gate_up_proj(x if qa is None else qa)
        if self._down_takes_mxfp8 is None:
            self._down_takes_mxfp8 = (
                self.down_proj.input_is_parallel
                and takes_rocm_mxfp8_input(self.down_proj)
            )
        if self._down_takes_mxfp8 and gate_up.dim() == 2 and gate_up.is_contiguous():
            # One launch for the activation and down_proj's MXFP8 input quant.
            values, scales = silu_and_mul_mxfp8(
                gate_up, getattr(self.act_fn, "swiglu_limit", None)
            )
            x = QuantizedActivation(
                values, scales, gate_up.dtype, values.shape, kMxfp8Dynamic
            )
        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _use_heterogeneous_fhmoe(num_tokens: int) -> bool:
    return num_tokens > 0 and (
        rocm_aiter_ops.fused_moe_supports_heterogeneous_shared_expert(num_tokens)
        is True
    )


def _validate_heterogeneous_routes(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_per_token: int,
) -> None:
    if (
        topk_weights.ndim != 2
        or topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError(
            "Heterogeneous fused MoE routes must have equal two-dimensional shapes."
        )
    expected_columns = experts_per_token + 1
    if topk_weights.shape[1] != expected_columns:
        raise ValueError(
            "Heterogeneous fused MoE routes must have exactly "
            f"{expected_columns} columns, got {topk_weights.shape[1]}."
        )


def _heterogeneous_shared_expert_enabled(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    parallel_config = vllm_config.parallel_config
    reasons: list[str] = []

    if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
        return False
    if not on_gfx950():
        reasons.append("the device is not gfx950")
    if parallel_config.enable_expert_parallel:
        reasons.append("expert parallelism is enabled")
    if parallel_config.enable_eplb:
        reasons.append("EPLB is enabled")
    if parallel_config.tensor_parallel_size != 8:
        reasons.append("tensor parallelism is not 8")
    if parallel_config.data_parallel_size != 1:
        reasons.append("data parallelism is enabled")
    if parallel_config.prefill_context_parallel_size != 1:
        reasons.append("prefill context parallelism is enabled")
    if vllm_config.kernel_config.moe_backend != "aiter":
        reasons.append("the MoE backend is not AITER")
    if vllm_config.model_config.dtype != torch.bfloat16:
        reasons.append("the model dtype is not BF16")
    if not rocm_aiter_ops.fused_moe_supports_heterogeneous_shared_expert(1):
        reasons.append(
            "AITER lacks CSV-backed DSV4 I384 heterogeneous shared-expert support"
        )

    expected_model = {
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "hidden_size": 7168,
        "moe_intermediate_size": 3072,
        "hidden_act": "silu",
        "expert_dtype": "fp4",
        "topk_method": "noaux_tc",
    }
    for name, expected in expected_model.items():
        if getattr(config, name, None) != expected:
            reasons.append(f"{name} is not {expected!r}")

    if quant_config is None or quant_config.get_name() != "deepseek_v4_fp8":
        reasons.append("the DeepSeek V4 FP8 quantization config is not active")
    else:
        if getattr(quant_config, "moe_quant_algo", "").upper() == "NVFP4":
            reasons.append("routed experts use NVFP4")
        if getattr(quant_config, "weight_block_size", None) != [128, 128]:
            reasons.append("shared experts are not block-128 FP8")
        if not getattr(quant_config, "is_checkpoint_fp8_serialized", False):
            reasons.append("shared experts are not serialized as FP8")
        if not getattr(quant_config, "is_scale_e8m0", False):
            reasons.append("shared-expert scales are not E8M0")
        if getattr(quant_config, "ignored_layers", None):
            reasons.append("the quantization config has ignored layers")

    offload_config = getattr(vllm_config, "offload_config", None)
    if offload_config is not None and (
        getattr(getattr(offload_config, "uva", None), "cpu_offload_gb", 0) > 0
        or getattr(
            getattr(offload_config, "prefetch", None),
            "offload_group_size",
            0,
        )
        > 0
    ):
        reasons.append("weight offloading is enabled")

    if reasons:
        logger.debug_once(
            "DeepSeek V4 heterogeneous shared-expert fusion is unavailable: %s. "
            "Continuing with the existing shared-expert path.",
            "; ".join(reasons),
        )
        return False

    logger.info_once(
        "Using AITER DeepSeek V4 heterogeneous MoE for CSV-covered MoE input "
        "rows (M) per invocation (native intermediate width 384)."
    )
    return True


@torch.no_grad()
def _prepare_native_fp8_shared_expert(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand native block-128 E8M0 scales to FHMoE's 1x32 layout."""
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise ValueError("Heterogeneous shared-expert weights must be FP8 E4M3.")
    hidden_size = w13.shape[1]
    if w13.shape != (2 * intermediate_size, hidden_size):
        raise ValueError(f"Unexpected shared W13 shape {tuple(w13.shape)}.")
    if w2.shape != (hidden_size, intermediate_size):
        raise ValueError(f"Unexpected shared W2 shape {tuple(w2.shape)}.")

    def scale_bytes(scale: torch.Tensor) -> torch.Tensor:
        if scale.dtype == torch.float8_e8m0fnu:
            return scale.contiguous().view(torch.uint8)
        if scale.dtype == torch.uint8:
            return scale.contiguous()
        if scale.dtype == torch.float32:
            bits = scale.contiguous().view(torch.int32)
            exponent = (bits >> 23).to(torch.uint8)
            if not torch.equal(bits, exponent.to(torch.int32) << 23):
                raise ValueError("FP32 shared-expert scales are not exact E8M0.")
            return exponent
        raise ValueError(f"Unsupported shared-expert scale dtype {scale.dtype}.")

    w13_scale = scale_bytes(w13_scale)
    w2_scale = scale_bytes(w2_scale)
    expected_w13_scale = (2 * intermediate_size // 128, hidden_size // 128)
    expected_w2_scale = (hidden_size // 128, intermediate_size // 128)
    if tuple(w13_scale.shape) != expected_w13_scale:
        raise ValueError(
            f"Expected shared W13 scale shape {expected_w13_scale}, got "
            f"{tuple(w13_scale.shape)}."
        )
    if tuple(w2_scale.shape) != expected_w2_scale:
        raise ValueError(
            f"Expected shared W2 scale shape {expected_w2_scale}, got "
            f"{tuple(w2_scale.shape)}."
        )

    w13_scale = w13_scale.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    w2_scale = w2_scale.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)

    scale_cols = ((w2_scale.shape[1] + 7) // 8) * 8
    if scale_cols != w2_scale.shape[1]:
        padded_w2_scale = torch.full(
            (w2_scale.shape[0], scale_cols),
            0x7F,
            dtype=torch.uint8,
            device=w2_scale.device,
        )
        padded_w2_scale[:, : w2_scale.shape[1]].copy_(w2_scale)
        w2_scale = padded_w2_scale

    return (
        w13.unsqueeze(0),
        w2.unsqueeze(0),
        w13_scale.view(torch.float8_e8m0fnu),
        w2_scale.view(torch.float8_e8m0fnu),
    )


class DeepseekV4HeterogeneousMxfp4MoEMethod(Mxfp4MoEMethod):
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        super().process_weights_after_loading(layer)
        assert isinstance(layer, DeepseekV4HeterogeneousSharedRoutedExperts)
        layer.prepare_heterogeneous_shared_expert()


class DeepseekV4HeterogeneousSharedRoutedExperts(RoutedExperts):
    def __init__(
        self,
        *args,
        shared_expert: DeepseekV4MLP,
        shared_expert_id: int,
        **kwargs,
    ):
        self._shared_expert_ref = weakref.ref(shared_expert)
        self.shared_expert_id = shared_expert_id
        self._routed_quant_config: FusedMoEQuantConfig | None = None
        super().__init__(*args, **kwargs)
        if self.expert_map_manager.num_fused_shared_experts != 1:
            raise ValueError("Heterogeneous fusion requires one appended expert.")
        if self.shared_expert_id != self.global_num_experts:
            raise ValueError("The shared expert ID must name the appended slot.")
        # derived from the shared expert's weights after loading, refilled in
        # place on reloads (captured CUDA graphs read them)
        self.shared_w1: torch.Tensor | None = None
        self.shared_w2: torch.Tensor | None = None
        self.shared_w1_scale: torch.Tensor | None = None
        self.shared_w2_scale: torch.Tensor | None = None

    def _get_quant_method(self, prefix, quant_config, moe_config):
        quant_method = super()._get_quant_method(prefix, quant_config, moe_config)
        if not isinstance(quant_method, Mxfp4MoEMethod):
            raise ValueError("Heterogeneous fusion requires MXFP4 routed experts.")
        return DeepseekV4HeterogeneousMxfp4MoEMethod(moe_config)

    @torch.no_grad()
    def prepare_heterogeneous_shared_expert(self) -> None:
        shared_expert = self._shared_expert_ref()
        if shared_expert is None:
            raise RuntimeError("The native shared-expert module was released.")

        shared = _prepare_native_fp8_shared_expert(
            shared_expert.gate_up_proj.weight,
            shared_expert.down_proj.weight,
            shared_expert.gate_up_proj.weight_scale_inv,
            shared_expert.down_proj.weight_scale_inv,
            self.moe_config.intermediate_size_per_partition,
        )
        refill_derived(
            self, "shared_w1", rocm_aiter_ops.shuffle_weight_a16w4(shared[0], 16, True)
        )
        refill_derived(
            self, "shared_w2", rocm_aiter_ops.shuffle_weight_a16w4(shared[1], 16, False)
        )
        refill_derived(
            self,
            "shared_w1_scale",
            rocm_aiter_ops.shuffle_scale_a16w4(shared[2], 1, True),
        )
        refill_derived(self, "shared_w2_scale", rocm_aiter_ops.shuffle_scale(shared[3]))

        self._ensure_moe_quant_config_init()
        quant_config = self.quant_method.moe_quant_config
        if quant_config is None:
            raise RuntimeError("The routed MXFP4 quantization config is unavailable.")
        if quant_config.w1_scale is None or quant_config.w2_scale is None:
            raise RuntimeError("The routed MXFP4 scales are unavailable.")

        def routed_scales(name: str, scale: torch.Tensor) -> torch.Tensor:
            scale = scale.reshape(self.local_num_experts, -1, scale.shape[-1])[
                : self.shared_expert_id
            ]
            return refill_derived(
                self, name, scale.reshape(-1, scale.shape[-1]).contiguous()
            )

        self._routed_quant_config = replace(
            quant_config,
            _w1=replace(
                quant_config._w1,
                scale=routed_scales("_routed_w1_scale", quant_config.w1_scale),
            ),
            _w2=replace(
                quant_config._w2,
                scale=routed_scales("_routed_w2_scale", quant_config.w2_scale),
            ),
        )

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if shared_experts is not None or shared_experts_input is not None:
            raise ValueError("The heterogeneous expert owns the shared path.")
        _validate_heterogeneous_routes(
            topk_weights,
            topk_ids,
            self.moe_config.experts_per_token,
        )
        if any(
            tensor is None
            for tensor in (
                self.shared_w1,
                self.shared_w2,
                self.shared_w1_scale,
                self.shared_w2_scale,
            )
        ):
            raise RuntimeError("Heterogeneous shared weights were not prepared.")

        if _use_heterogeneous_fhmoe(x.shape[0]):
            self._ensure_moe_quant_config_init()
            quant_config = self.quant_method.moe_quant_config
            if quant_config is None:
                raise RuntimeError("The routed MXFP4 config is unavailable.")
            return rocm_aiter_fused_experts(
                hidden_states=x,
                w1=self.w13_weight,
                w2=self.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=self.moe_config,
                activation=self.activation,
                expert_mask=self.expert_map,
                quant_config=quant_config,
                quant_type=derive_aiter_quant_type(quant_config),
                output_dtype=x.dtype,
                moe_sorting_dispatch_policy=(rocm_aiter_ops.get_moe_dispatch_policy()),
                shared_w1=self.shared_w1,
                shared_w2=self.shared_w2,
                shared_w1_scale=self.shared_w1_scale,
                shared_w2_scale=self.shared_w2_scale,
                shared_expert_id=self.shared_expert_id,
                **mxfp4_w4a16_extra_kwargs(quant_config, self.moe_config),
            )

        shared_expert = self._shared_expert_ref()
        if shared_expert is None or self._routed_quant_config is None:
            raise RuntimeError("The separate shared-expert fallback is unavailable.")
        shared_out = shared_expert(x)
        routed_w1 = self.w13_weight[: self.shared_expert_id]
        routed_w2 = self.w2_weight[: self.shared_expert_id]
        routed_w1.is_shuffled = True
        routed_w2.is_shuffled = True
        routed = rocm_aiter_fused_experts(
            hidden_states=x,
            w1=routed_w1,
            w2=routed_w2,
            topk_weights=topk_weights[:, :-1].contiguous(),
            topk_ids=topk_ids[:, :-1].contiguous(),
            moe_config=self.moe_config,
            activation=self.activation,
            expert_mask=self.expert_map,
            quant_config=self._routed_quant_config,
            quant_type=derive_aiter_quant_type(self._routed_quant_config),
            output_dtype=x.dtype,
            moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
            **mxfp4_w4a16_extra_kwargs(self._routed_quant_config, self.moe_config),
        )
        return routed + shared_out


def _validate_aiter_mega_moe_config(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    if not rocm_aiter_ops.is_fused_moe_enabled():
        raise ValueError(
            "--moe-backend aiter_mega_moe requires VLLM_ROCM_USE_AITER=1 "
            " and VLLM_ROCM_USE_AITER_MOE=1."
        )
    if not on_gfx950():
        raise NotImplementedError("AITER MegaMoE requires gfx950.")
    if (
        not parallel_config.enable_expert_parallel
        or parallel_config.tensor_parallel_size != 1
        or parallel_config.pipeline_parallel_size != 1
    ):
        raise NotImplementedError(
            "AITER MegaMoE requires data parallel attention with expert "
            "parallel MoE: --data-parallel-size N --enable-expert-parallel "
            "with TP=1 and PP=1."
        )
    if parallel_config.all2all_backend not in (
        "mori_high_throughput",
        "mori_low_latency",
        "mori",
    ) or (parallel_config.all2all_backend == "mori" and parallel_config.mori_ep_v2):
        raise ValueError(
            "AITER MegaMoE runs on MoRI EPv1 shmem; add --all2all-backend mori "
            "(without VLLM_MORI_EP_V2=1)."
        )
    if parallel_config.enable_eplb:
        raise NotImplementedError("AITER MegaMoE does not support EPLB.")
    if getattr(vllm_config.model_config.hf_config, "expert_dtype", "fp4") != "fp4":
        raise NotImplementedError("AITER MegaMoE only supports fp4 experts.")


class DeepseekV4MoE(nn.Module):
    # None: use the padding vLLM added.
    experts_intermediate_pad: int | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        fuse_heterogeneous_shared_expert: bool = False,
        sp_moe: bool = False,
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        self.gate.bias_vl = None
        self.image_sentinel_lo = (
            IMAGE_SENTINEL_BASE_ID if getattr(config, "vision_n_layers", 0) > 0 else 0
        )
        num_hash_layers = getattr(config, "num_hash_layers", None)  # Set by V4.1
        if num_hash_layers is None:
            num_hash_layers = config.mlp_layer_types.count("hash_moe")
        is_hash_moe = extract_layer_index(prefix) < num_hash_layers
        self.hash_indices_dtype = torch.int32
        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        if getattr(config, "topk_method", None) == "noaux_tc" and (
            not is_hash_moe or getattr(config, "vision_n_layers", 0) > 0
        ):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if getattr(config, "vision_n_layers", 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        self.n_shared_experts = config.n_shared_experts

        self.fuse_heterogeneous_shared_expert = fuse_heterogeneous_shared_expert
        self.is_fused_shared_expert_enabled = False
        if (
            getattr(config, "n_shared_experts", None)
            and not self.fuse_heterogeneous_shared_expert
        ):
            # Fused shared experts are not supported under data parallelism:
            # the fused path cannot load the FP8 shared expert into the MXFP4
            # routed slot. Fail fast with an actionable message.
            if (
                not vllm_config.parallel_config.enable_expert_parallel
                and vllm_config.parallel_config.data_parallel_size > 1
                and rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
            ):
                raise ValueError(
                    "DeepSeek-V4 fused shared experts "
                    "(VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1) are not "
                    "supported with data parallelism "
                    "(data_parallel_size > 1). Set "
                    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 to run the "
                    "shared expert as an unfused FP8 MLP."
                )
            self.is_fused_shared_expert_enabled = resolve_layer_fused_shared_expert(
                quant_config, prefix, disable_under_ep=True
            )

        self.use_mega_moe = vllm_config.kernel_config.moe_backend == "aiter_mega_moe"
        fuse_mega_moe_shared_expert = (
            self.use_mega_moe
            and config.n_shared_experts == 1
            and resolve_layer_fused_shared_expert(
                quant_config, prefix, moe_backend="aiter_mega_moe"
            )
        )
        shared_quant_config = quant_config
        if fuse_mega_moe_shared_expert:
            # MegaMoE requantizes the checkpoint weights itself.
            assert quant_config is not None
            shared_quant_config = copy.copy(quant_config)
            shared_quant_config.online_quantization_config = None

        if config.n_shared_experts is None or self.is_fused_shared_expert_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=shared_quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # dsv4_sp_moe: a full (non-TP) copy of the shared expert, run on each
        # rank's own rows on sequence-parallel steps. Set below.
        self.shared_experts_full: DeepseekV4MLP | None = None
        self._sp_router_shard: bool | None = None
        self._sp_moe_codes: bool | None = None

        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0

        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        if self.use_mega_moe:
            _validate_aiter_mega_moe_config(vllm_config)
            self.experts = DeepseekV4AiterMegaMoEExperts(
                vllm_config,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                swiglu_limit=self.swiglu_limit,
                fuse_shared_expert=fuse_mega_moe_shared_expert,
                prefix=f"{prefix}.experts",
            )
            self.shared_experts_stream = (
                aux_stream()
                if self.shared_experts is not None
                and not self.experts.fuse_shared_expert
                and not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM
                else None
            )
            self.shared_experts_events = (
                (torch.cuda.Event(), torch.cuda.Event())
                if self.shared_experts_stream is not None
                else None
            )
            return

        fuse_shared_into_routed = (
            self.is_fused_shared_expert_enabled or self.fuse_heterogeneous_shared_expert
        )
        if sp_moe and self.shared_experts is not None and not fuse_shared_into_routed:
            self.shared_experts_full = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=shared_quant_config,
                reduce_results=False,
                is_sequence_parallel=True,
                prefix=f"{prefix}.shared_experts_full",
            )
            self.shared_experts_full._rowlocal_full = True
        self.experts = FusedMoEFactory(
            shared_experts=(None if fuse_shared_into_routed else self.shared_experts),
            n_shared_experts=(
                config.n_shared_experts if fuse_shared_into_routed else None
            ),
            fuse_shared_experts=fuse_shared_into_routed,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            intermediate_pad=self.experts_intermediate_pad,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=self.image_sentinel_lo,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            routed_experts_cls=(
                DeepseekV4HeterogeneousSharedRoutedExperts
                if self.fuse_heterogeneous_shared_expert
                else None
            ),
            routed_experts_args=(
                {
                    "shared_expert": self.shared_experts,
                    "shared_expert_id": config.n_routed_experts,
                }
                if self.fuse_heterogeneous_shared_expert
                else None
            ),
        )

    def _check_input_ids(self, input_ids: torch.Tensor | None) -> None:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
        if self.gate.bias_vl is not None and input_ids is None:
            raise ValueError("DeepSeek V4 vision MoE routing requires input_ids.")

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._check_input_ids(input_ids)

        org_shape = hidden_states.shape
        if self.use_mega_moe:
            return self._forward_mega_moe(hidden_states, input_ids).view(org_shape)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            input_ids=input_ids,
        )

        return final_hidden_states.view(org_shape)

    def forward_partial(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        routing: tuple[torch.Tensor, torch.Tensor] | None = None,
        defer_shared_add: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """forward() without the final TP all-reduce: this rank's partial sum
        (routed + shared), which sequence-parallel steps reduce-scatter.

        routing: (topk_weights, topk_ids) of every row (route_rows), used
        instead of the runner's gate and top-k. defer_shared_add: return
        (shared, routed) unsummed.
        """
        if self.use_mega_moe:
            raise RuntimeError("sequence-parallel MoE does not support aiter_mega_moe")
        cfg = self.experts.moe_config
        prev = cfg.skip_final_all_reduce
        cfg.skip_final_all_reduce = True
        try:
            if routing is None and not defer_shared_add:
                return self.forward(hidden_states, input_ids)
            if routing is None:
                self._check_input_ids(input_ids)
            out = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
                topk_weights=None if routing is None else routing[0],
                topk_ids=None if routing is None else routing[1],
                defer_shared_add=defer_shared_add,
            )
        finally:
            cfg.skip_final_all_reduce = prev
        if isinstance(out, tuple):
            return tuple(t.view(hidden_states.shape) for t in out)
        return out.view(hidden_states.shape)

    def sp_router_shard_ok(self) -> bool:
        """Whether this layer may route its own rows on sequence-parallel steps
        (dsv4_sp_moe): a learned router (no hash routing, no vision bias) on a
        modular MoE kernel, the runner holding the plain gate, and nothing
        that must see every row's routing (EPLB, routed-expert capture)."""
        if self._sp_router_shard is None:
            runner = self.experts
            router = getattr(runner, "router", None)
            self._sp_router_shard = (
                not self.use_mega_moe
                and self.gate.tid2eid is None
                and self.gate.bias_vl is None
                and isinstance(runner, MoERunner)
                and runner.gate is self.gate
                and not runner._fse_fuse_gate
                and not runner.routed_experts.quant_method.is_monolithic
                and router is not None
                and router.eplb_state is None
                and getattr(router, "capture_fn", None) is None
                and getattr(router, "_routing_replay_out", None) is None
            )
        return self._sp_router_shard

    def sp_moe_codes_ok(self) -> bool:
        """Whether the routed experts can take MXFP8 codes for their input
        (dsv4_sp_moe): the runner hands the input to AITER fused_moe as is (no
        input transform, no padding, no quantization before the kernel)."""
        if self._sp_moe_codes is None:
            runner = self.experts
            kernel = getattr(runner._quant_method, "moe_kernel", None)
            experts = getattr(kernel, "fused_experts", None)
            self._sp_moe_codes = (
                isinstance(experts, AiterMxfp4Experts)
                and experts.expects_unquantized_inputs
                and experts.quant_config.a1_scale is None
                and runner.routed_input_transform is None
                and runner.moe_config.hidden_dim == self.hidden_size
            )
        return self._sp_moe_codes

    def route_rows(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The served routing (the runner's gate and router) of the rows x:
        (topk_weights, topk_ids)."""
        runner = self.experts
        logits, _ = self.gate(x)
        return runner.router.select_experts(
            hidden_states=x,
            router_logits=logits,
            topk_indices_dtype=runner._quant_method.topk_indices_dtype,
        )

    def _forward_mega_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> torch.Tensor:
        shared_experts = self.shared_experts
        if shared_experts is None:
            return self._forward_mega_moe_routed(hidden_states, input_ids)
        events = self.shared_experts_events
        if (
            events is None
            or hidden_states.shape[0] > envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        ):
            return self._forward_mega_moe_routed(
                hidden_states, input_ids
            ) + shared_experts(hidden_states)
        # The unfused shared expert overlaps router + MegaMoE in small steps.
        routed, shared = maybe_execute_in_parallel(
            fn0=lambda: self._forward_mega_moe_routed(hidden_states, input_ids),
            fn1=lambda: shared_experts(hidden_states),
            event0=events[0],
            event1=events[1],
            aux_stream=self.shared_experts_stream,
        )
        return routed + shared

    def _forward_mega_moe_routed(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=self.image_sentinel_lo,
        )
        return self.experts(hidden_states, topk_weights, topk_ids)


# dsv4_sp_moe: one dispatch line per rank for the router shard and the
# row-local shared expert.
_SP_ROUTER_SHARD_LOGGED = False
_SP_SHARED_ROWLOCAL_LOGGED = False


def _shared_rowlocal_log(
    layer: "DeepseekV4DecoderLayer",
    x: torch.Tensor,
    xs: torch.Tensor,
    shr: torch.Tensor,
    sp: SeqParallelStep,
) -> None:
    """At the first row-local shared expert call on this rank: a dispatch line,
    and with dsv4_sp_self_check a comparison against the TP shared expert's
    slices on all rows, summed exactly (fp32 all-reduce). The fp8 quant groups
    are the same on both sides, so they differ only by GEMM accumulation order
    and bf16 rounding. Raises on non-finite rows or a relative L2 above 5%."""
    rank = get_tensor_model_parallel_rank()
    if not sp_self_check_enabled():
        logger.info(
            "%s: full shared expert on rank %d's own %d of %d rows; the "
            "reduce-scatter carries the routed term only",
            SP_SHARED_ROWLOCAL_LABEL,
            rank,
            xs.shape[0],
            sp.m,
        )
        return
    # The same layer on every rank, so this extra all-reduce is matched.
    ref = layer.ffn.shared_experts(x).float()
    torch.distributed.all_reduce(ref, group=get_tp_group().device_group)
    n = max(min(sp.start + sp.s, sp.m) - sp.start, 0)
    a, b = shr[:n].float(), ref[sp.start : sp.start + n]
    bn = float(b.norm()) if n else 0.0
    rel = float((a - b).norm()) / max(bn, 1e-30) if n else 0.0
    finite = bool(torch.isfinite(a).all())
    logger.info(
        "%s: full shared expert on rank %d's own %d of %d rows; self-check "
        "against the TP slices summed exactly: rel L2 %.4g (ref norm %.4g), "
        "finite %s",
        SP_SHARED_ROWLOCAL_LABEL,
        rank,
        n,
        sp.m,
        rel,
        bn,
        finite,
    )
    if not finite or (rel > 0.05 and bn > 1e-2):
        raise RuntimeError(
            f"{SP_SHARED_ROWLOCAL_LABEL}: row-local shared expert differs (rel {rel})"
        )


# Hidden sizes supported by AITER mhc_pre_big_fuse_rmsnorm.
_AITER_MHC_FUSED_RMSNORM_SIZES = frozenset({1280, 2560, 4096, 7168})


def _mhc_fusion(hidden_size: int, hc_mult: int) -> tuple[bool, bool]:
    """(fused mHC post+pre, attn/ffn RMSNorm folded into the mHC seam)."""
    # AITER mhc kernels (pre/post/fused) require hc_mult == 4.
    use_aiter_mhc = HAS_AITER_MHC and hidden_size % 256 == 0 and hc_mult == 4
    # Prefer AITER fused post+pre when eligible; otherwise TileLang.
    use_fused_mhc = use_aiter_mhc or HAS_TILELANG_MHC
    # Fold attn/ffn RMSNorm into MHC only when the active backend's
    # fused-rmsnorm path supports this hidden size.
    if use_aiter_mhc:
        fuse_rmsnorm = hidden_size in _AITER_MHC_FUSED_RMSNORM_SIZES
    else:
        fuse_rmsnorm = HAS_TILELANG_MHC and use_fused_mhc
    return use_fused_mhc, fuse_rmsnorm


def _sp_moe_requested(vllm_config: VllmConfig) -> bool:
    """Whether dsv4_sp_moe can engage (decided before the layers are built:
    the full shared expert copy costs about 3.8 GiB per GPU)."""
    attn = vllm_config.attention_config
    if not attn.dsv4_sp_moe:
        return False
    if not attn.dsv4_seq_parallel:
        logger.warning("%s ignored: dsv4_seq_parallel is off", SP_MOE_LABEL)
        return False
    config = vllm_config.model_config.hf_config
    reasons = sp_config_disable_reasons(vllm_config)
    if vllm_config.kernel_config.moe_backend == "aiter_mega_moe":
        reasons.append("aiter_mega_moe")
    use_fused_mhc, fuse_rmsnorm = _mhc_fusion(config.hidden_size, config.hc_mult)
    if not (use_fused_mhc and fuse_rmsnorm):
        reasons.append("the mHC seam does not fold the RMSNorm")
    if routing_tail_bytes(config.num_experts_per_tok) > GATHER_TAIL_BYTES:
        reasons.append(f"top-k {config.num_experts_per_tok}")
    if reasons:
        # dsv4_seq_parallel logs the deployment reasons itself.
        logger.warning("%s ignored: %s", SP_MOE_LABEL, ", ".join(reasons))
        return False
    return True


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        fuse_heterogeneous_shared_expert: bool = False,
        sp_moe: bool = False,
    ):
        super().__init__()

        # Lazy import to avoid top-level tilelang dependency.
        # Registers both torch.ops.vllm.mhc_pre and mhc_post
        import vllm.model_executor.layers.mhc  # noqa: F401

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4ROCMAiterMLAAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV4MoE(
            vllm_config,
            prefix=f"{prefix}.ffn",
            fuse_heterogeneous_shared_expert=fuse_heterogeneous_shared_expert,
            sp_moe=sp_moe,
        )
        # dsv4_sp_moe armed for this layer (set by the model).
        self.sp_moe = False

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()
        self.mhc_fused_post_pre = MHCFusedPostPreOp()
        self.use_fused_mhc, self.fuse_mhc_rmsnorm = _mhc_fusion(
            self.hidden_size, self.hc_mult
        )
        # Decode steps of an eligible layer may run as the mono decode layer.
        self.mono = MonoDecodeLayer.create(self, vllm_config)

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        """Reduce HC residual streams to the next sub-layer input.

        When `norm_weight` is set, RMSNorm is fused into the pre kernel.
        """
        post_mix, res_mix, layer_input = self.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post(x, residual, post, comb)

    def _forward_fused_post_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_norm_weight = self.attn_norm.weight if self.fuse_mhc_rmsnorm else None
        attn_norm_eps = (
            self.attn_norm.variance_epsilon if self.fuse_mhc_rmsnorm else 0.0
        )
        if residual is None:
            # Run standalone hc_pre on first layer
            residual = x
            x, post_mix, res_mix = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
        else:
            residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        if not self.fuse_mhc_rmsnorm:
            x = self.attn_norm(x)
        x = self.attn(positions, x, None)

        ffn_norm_weight = self.ffn_norm.weight if self.fuse_mhc_rmsnorm else None
        ffn_norm_eps = self.ffn_norm.variance_epsilon if self.fuse_mhc_rmsnorm else 0.0
        residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )
        if not self.fuse_mhc_rmsnorm:
            x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix

    def forward_seq_parallel(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        sp: SeqParallelStep,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """_forward_fused_post_pre with the mHC seams on this rank's Mp/8 rows.

        Each sublayer's partial sum (attention wo_b, MoE routed + shared) is
        reduce-scattered, the fused post+pre seam runs on Mp/8 rows, and the
        next sublayer's input is all-gathered in fp8 1x128. x, residual,
        post_mix and res_mix stay on Mp/8 rows between layers; the first
        layer's standalone pre seam runs on all rows and its outputs are
        sliced. Returns the reduced FFN output rows.
        """
        fold = self.fuse_mhc_rmsnorm
        attn_norm_weight = self.attn_norm.weight if fold else None
        attn_norm_eps = self.attn_norm.variance_epsilon if fold else 0.0
        if residual is None:
            full_residual = x
            x, post_mix, res_mix = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
            residual = sp.rows(full_residual)
            post_mix = sp.rows(post_mix)
            res_mix = sp.rows(res_mix)
        else:
            residual, post_mix, res_mix, xs = self.mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
            x = sp.gather(xs)
        if not fold:
            x = self.attn_norm(x)
        # dsv4_rowlocal_attention: the attention returns this rank's rows of
        # the full output, so there is nothing to reduce.
        rowlocal = self.attn.rowlocal_ready()
        self.attn._sp_defer_reduce = True
        if rowlocal:
            self.attn._sp_rowlocal = sp
        try:
            a = self.attn(positions, x, None)
        finally:
            self.attn._sp_defer_reduce = False
            self.attn._sp_rowlocal = None
            sp_fp8_handoff_clear()
        x = a if rowlocal else sp.reduce_scatter(a)

        ffn_norm_weight = self.ffn_norm.weight if fold else None
        ffn_norm_eps = self.ffn_norm.variance_epsilon if fold else 0.0
        residual, post_mix, res_mix, xs = self.mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )
        if self.sp_moe and fold:
            return self._sp_moe_ffn(xs, input_ids, sp), residual, post_mix, res_mix
        x = sp.gather(xs)
        if not fold:
            x = self.ffn_norm(x)
        try:
            f = self.ffn.forward_partial(x, input_ids)
        finally:
            sp_fp8_handoff_clear()
        x = sp.reduce_scatter(f)
        return x, residual, post_mix, res_mix

    def _sp_moe_ffn(
        self, xs: torch.Tensor, input_ids: torch.Tensor | None, sp: SeqParallelStep
    ) -> torch.Tensor:
        """The FFN sublayer of a sequence-parallel step with dsv4_sp_moe, from
        this rank's normalized FFN input rows xs [Mp/8, H] to its reduced output
        rows [Mp/8, H].

        Routes xs here and gathers the routing with the rows; gives the routed
        experts the gathered fp8 codes; runs the full shared expert on xs and
        adds it after the reduce-scatter, which then carries the routed term
        only (summed from the per-slot rows when the route fold is on).
        """
        ffn = self.ffn
        full = ffn.shared_experts_full
        shared_rowlocal = full is not None and full._down_scale is not None
        router_shard = ffn.sp_router_shard_ok()
        routing = None
        if router_shard:
            # Codes only when every reader of the gathered rows takes the codes:
            # the routed experts and, unless it is row-local, the TP shared
            # expert's gate_up.
            codes_only = (
                sp_moe_codes_wanted(sp.m)
                and ffn.sp_moe_codes_ok()
                and (
                    shared_rowlocal
                    or ffn.shared_experts is None
                    or ffn.shared_experts._gateup_scale is not None
                )
            )
            weights, ids = ffn.route_rows(xs)
            x, gw, gi = sp.gather_with_routing(xs, weights, ids, codes_only)
            routing = (gw, gi)
            global _SP_ROUTER_SHARD_LOGGED
            if not _SP_ROUTER_SHARD_LOGGED:
                _SP_ROUTER_SHARD_LOGGED = True
                logger.info(
                    "%s: MoE routing row-sharded on rank %d (%s: %d of %d rows "
                    "routed locally, routing carried in the fp8 layer-input "
                    "gather)",
                    SP_ROUTER_SHARD_LABEL,
                    get_tensor_model_parallel_rank(),
                    ffn.prefix,
                    xs.shape[0],
                    x.shape[0],
                )
        else:
            x = sp.gather(xs)

        shr = None
        tp_shared = ffn.shared_experts
        if shared_rowlocal:
            assert full is not None and tp_shared is not None
            shr = full.forward_rowlocal(xs)
            global _SP_SHARED_ROWLOCAL_LOGGED
            if not _SP_SHARED_ROWLOCAL_LOGGED:
                _SP_SHARED_ROWLOCAL_LOGGED = True
                _shared_rowlocal_log(self, x, xs, shr, sp)
            tp_shared._sp_skip = True
        # The slot fold needs the routed term on its own: only with the
        # row-local shared expert, and only if the runner leaves the routed
        # output as stage 2 writes it.
        runner = ffn.experts
        route_fold = (
            shr is not None
            and sp.comm.route_fold
            and getattr(runner, "routed_scaling_factor", 1.0) == 1.0
            and getattr(runner, "routed_output_transform", None) is None
        )
        cap = sp_route_capture(sp.m, enabled=route_fold)
        try:
            with cap:
                f = ffn.forward_partial(
                    x, input_ids, routing, defer_shared_add=shr is not None
                )
            if routing is not None:
                sp_moe_codes_done(shared_reads_rows=shr is None)
        finally:
            sp_fp8_handoff_clear()
            if shr is not None:
                tp_shared._sp_skip = False
        if shr is None:
            return sp.reduce_scatter(f)
        if not isinstance(f, tuple):
            raise RuntimeError(
                f"{SP_SHARED_ROWLOCAL_LABEL}: the MoE did not defer the shared add"
            )
        # The TP shared expert's placeholder f[0] is dropped.
        out = sp.reduce_scatter(f[1], route=cap.route)
        return out.add_(shr)

    def _forward_unfused_post_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x, None, None, None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        if self.mono is not None:
            out = self.mono(self, x, positions, input_ids, post_mix, res_mix, residual)
            if out is not None:
                return out
        if not self.use_fused_mhc:
            return self._forward_unfused_post_pre(
                x, positions, input_ids, post_mix, res_mix, residual
            )
        return self._forward_fused_post_pre(
            x, positions, input_ids, post_mix, res_mix, residual
        )


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        self.device = current_platform.device_type
        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.fuse_heterogeneous_shared_expert = _heterogeneous_shared_expert_enabled(
            vllm_config
        )
        self.use_mega_moe = vllm_config.kernel_config.moe_backend == "aiter_mega_moe"
        self._sp_moe_requested = _sp_moe_requested(vllm_config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                fuse_heterogeneous_shared_expert=(
                    self.fuse_heterogeneous_shared_expert
                ),
                sp_moe=self._sp_moe_requested,
            ),
            prefix=f"{prefix}.layers",
        )
        self.fused_shared_expert_layers = get_fused_shared_expert_layers(
            self,
            DeepseekV4MoE,
            enable_eplb=vllm_config.parallel_config.enable_eplb,
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(
                self.hc_mult,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_op = HCHeadOp()
        spec_config = vllm_config.speculative_config
        if get_pp_group().is_last_rank and needs_mtp_hidden_buffer(spec_config):
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        else:
            self._mtp_hidden_buffer = None
        self._init_seq_parallel(vllm_config)

    def _init_seq_parallel(self, vllm_config: VllmConfig) -> None:
        """Arm dsv4_seq_parallel if this deployment and model can run it."""
        self._sp_enabled = vllm_config.attention_config.dsv4_seq_parallel
        self._sp_comm: SeqParallelSeamComm | None = None
        self._sp_logged = False
        self._sp_moe = False
        if not self._sp_enabled:
            return
        reasons = sp_config_disable_reasons(vllm_config)
        if self.use_mega_moe:
            reasons.append("aiter_mega_moe")
        if any(
            not layer.use_fused_mhc
            for layer in islice(self.layers, self.start_layer, self.end_layer)
        ):
            reasons.append("unfused mHC")
        if reasons:
            logger.warning("%s ignored: %s", SEQ_PARALLEL_LABEL, ", ".join(reasons))
            self._sp_enabled = False
            return
        set_sp_self_check(vllm_config.attention_config.dsv4_sp_self_check)
        align = SeqParallelSeamComm.row_alignment(self.config.hidden_size)
        mbt = vllm_config.scheduler_config.max_num_batched_tokens
        self._sp_max_rows = (mbt + align - 1) // align * align
        logger.info(
            "%s: sequence-parallel mHC seams armed on rank %d (steps of %d..%d "
            "rows; INT4 reduce-scatter, seams on M/8 rows, fp8 1x128 gather)",
            SEQ_PARALLEL_LABEL,
            get_tensor_model_parallel_rank(),
            SEQ_PARALLEL_MIN_ROWS,
            mbt,
        )
        self._sp_moe = self._sp_moe_requested
        if not self._sp_moe:
            return
        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        for layer in layers:
            layer.sp_moe = True
        set_rocm_aiter_fused_moe_input_hook(sp_fused_moe_input_hook)
        logger.info(
            "%s: MoE on sequence-parallel steps armed on rank %d (router shard, "
            "MXFP8 codes for the routed experts, full shared expert on own rows "
            "in %d of %d layers; %s is decided at the first such step)",
            SP_MOE_LABEL,
            get_tensor_model_parallel_rank(),
            sum(layer.ffn.shared_experts_full is not None for layer in layers),
            len(layers),
            SP_ROUTE_FOLD_LABEL,
        )

    def _seq_parallel_step(self, num_tokens: int) -> SeqParallelStep | None:
        if not self._sp_enabled or num_tokens < SEQ_PARALLEL_MIN_ROWS:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        if not (get_pp_group().is_first_rank and get_pp_group().is_last_rank):
            return None
        if self._sp_comm is None:
            self._sp_comm = SeqParallelSeamComm(
                get_tp_group(),
                self._sp_max_rows,
                self.config.hidden_size,
                gather_tail_bytes=(
                    routing_tail_bytes(self.config.num_experts_per_tok)
                    if self._sp_moe
                    else 0
                ),
                # The slot fold runs only beside the row-local shared expert.
                route_fold=self._sp_moe
                and any(
                    layer.ffn.shared_experts_full is not None
                    for layer in islice(self.layers, self.start_layer, self.end_layer)
                ),
            )
        sp = SeqParallelStep(self._sp_comm, num_tokens)
        if sp.mp > self._sp_max_rows:
            return None
        if not self._sp_logged:
            self._sp_logged = True
            logger.info(
                "%s: sequence-parallel step dispatched on rank %d (%d rows -> %d "
                "per rank)",
                SEQ_PARALLEL_LABEL,
                sp.comm.rank,
                num_tokens,
                sp.s,
            )
        return sp

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until hc_head() collapses it.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual, post_mix, res_mix = None, None, None
        # EAGLE3 / DSpark / DFlash aux hidden states: reconstructed (post-mhc)
        # hidden state at the configured target layers, averaged over the
        # hc_mult streams to [T, hidden_size]. Empty unless a draft model set
        # aux_hidden_state_layers.
        aux_hidden_states: list[torch.Tensor] = []
        # On the fused path the final layer's hc_post output is reused below
        # (avoids computing hc_post twice when the last layer is also an aux
        # layer).
        final_aux_recon: torch.Tensor | None = None
        sp = self._seq_parallel_step(hidden_states.shape[0])
        if sp is not None:
            return self._forward_seq_parallel(hidden_states, positions, input_ids, sp)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
            if (idx + 1) in self.aux_hidden_state_layers:
                # On the unfused path the layer already applied hc_post,
                # so hidden_states is the reconstructed stream; on the fused
                # path reconstruct it via hc_post before averaging.
                if layer.use_fused_mhc:
                    aux_recon = layer.hc_post(
                        hidden_states, residual, post_mix, res_mix
                    )
                    final_aux_recon = aux_recon
                else:
                    aux_recon = hidden_states
                aux_hidden_states.append(aux_recon.mean(dim=1))
        if layer is not None and layer.use_fused_mhc:
            # Reuse the last layer's hc_post output if it was already computed
            # for the aux hidden state above; otherwise compute it now.
            if (
                final_aux_recon is not None
                and self.end_layer in self.aux_hidden_state_layers
            ):
                hidden_states = final_aux_recon
            else:
                hidden_states = layer.hc_post(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            # Each DBO ubatch owns a different slice of the batch. Writing
            # from row 0 in both threads would let one ubatch clobber the
            # other's rows in the buffer the MTP drafter reads after the step.
            offset = dbo_current_ubatch_token_offset()
            self._mtp_hidden_buffer[offset : offset + num_tokens].copy_(
                hidden_states.flatten(1)
            )

        hidden_states = self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _forward_seq_parallel(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        sp: SeqParallelStep,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        residual, post_mix, res_mix = None, None, None
        aux_hidden_states: list[torch.Tensor] = []
        layer = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer.forward_seq_parallel(
                hidden_states, positions, input_ids, post_mix, res_mix, residual, sp
            )
            if (idx + 1) in self.aux_hidden_state_layers:
                aux = layer.hc_post(hidden_states, residual, post_mix, res_mix)
                aux_hidden_states.append(sp.gather_exact(aux.mean(dim=1)))
        assert layer is not None
        hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
        # _mtp_hidden_buffer is not written: sp_config_disable_reasons allows
        # only DSpark, which reads the aux hidden states.
        hidden_states = self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = sp.gather_exact(self.norm(hidden_states))
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def _resolve_param_name(name: str) -> str:
            # Fp8LinearMethod registers block scales as ``weight_scale_inv``.
            # Quark checkpoints / QuarkW8A8Fp8PerBlock use ``weight_scale``.
            # Alias only when the registered param is the ``_inv`` name.
            inv_name = f"{name}_inv"
            if name not in params_dict and inv_name in params_dict:
                return inv_name
            return name

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Use each MoE's own per-layer fusion decision (computed with its prefix
        # at init) as the single source of truth, so the redirect below cannot
        # diverge from how the module was built if per-layer quantization ever
        # mixes fused and non-fused layers.
        fuse_by_layer = self.fused_shared_expert_layers
        n_routed = self.config.n_routed_experts
        # The redirect below maps the single shared-expert tensor group to one
        # appended slot; multiple shared experts would need per-expert slicing
        # (see deepseek_v2.py). DeepSeek-V4 has n_shared_experts == 1.
        if fuse_by_layer and self.config.n_shared_experts != 1:
            raise NotImplementedError(
                "deepseek-v4 fused shared-expert loading supports only "
                f"n_shared_experts == 1, got {self.config.n_shared_experts}"
            )

        for name, loaded_weight in weights:
            # Shared-expert fusion: redirect ``.ffn.shared_experts.w{1,2,3}``
            # into appended routed-expert slot ``.ffn.experts.{n_routed}``
            # so the MXFP4-quantized shared expert loads through the routed
            # expert loader (grouped GEMM). Single shared expert only.
            name = _redirect_shared_expert_name(name, fuse_by_layer, n_routed)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                param_name = _resolve_param_name(name)
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(param_name)
                full_name = _shared_full_name(param_name)
                if full_name is not None and full_name in params_dict:
                    full_param = params_dict[full_name]
                    full_param.weight_loader(full_param, loaded_weight, shard_id)
                    loaded_params.add(full_name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param_name = _resolve_param_name(name)
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)
                    # dsv4_rowlocal_attention / dsv4_sp_moe: the full (non-TP)
                    # copies, present in params_dict only where they exist.
                    for full_name in (
                        _rowlocal_full_name(param_name),
                        _shared_full_name(param_name),
                    ):
                        if full_name is None or full_name not in params_dict:
                            continue
                        full_param = params_dict[full_name]
                        getattr(full_param, "weight_loader", default_weight_loader)(
                            full_param, loaded_weight
                        )
                        loaded_params.add(full_name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # When fusing shared experts, include the appended slots
        # (ids n_routed_experts .. n_routed_experts + n_shared - 1) so the
        # redirected shared-expert weights route through the expert loader.
        n_shared = getattr(self.config, "n_shared_experts", 0) or 0
        num_experts = self.config.n_routed_experts + (
            n_shared if self.fused_shared_expert_layers else 0
        )
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=num_experts,
            routed_experts_prefix="" if self.use_mega_moe else "routed_experts",
        )


# dsv4_rowlocal_attention: backbone attention checkpoint tensors that also load
# into the layer's full (all-head) copy.
_ROWLOCAL_FULL_RE = re.compile(
    r"(\.attn\.)(?:(wq_b|wo_a|wo_b)\.(weight|weight_scale_inv|weight_scale)"
    r"|(attn_sink))$"
)


def _rowlocal_full_name(param_name: str) -> str | None:
    """The full-copy parameter a backbone attention tensor also loads into."""
    m = _ROWLOCAL_FULL_RE.search(param_name)
    if m is None:
        return None
    if m.group(4) is not None:
        return param_name[: m.start()] + m.group(1) + "attn_sink_full"
    return param_name[: m.start()] + f"{m.group(1)}{m.group(2)}_full.{m.group(3)}"


def _shared_full_name(param_name: str) -> str | None:
    """The full shared expert copy's parameter (dsv4_sp_moe) that a shared
    expert checkpoint tensor also loads into, or None. params_dict has it only
    where the copy exists."""
    if ".ffn.shared_experts." not in param_name:
        return None
    return param_name.replace(".ffn.shared_experts.", ".ffn.shared_experts_full.")


def _redirect_shared_expert_name(
    name: str, fuse_by_layer: frozenset[int], n_routed_experts: int
) -> str:
    """Route a DSV4 shared-expert checkpoint name to its final parameter name.

    Fused layers go to the appended routed-expert slot. When any layer is fused
    the weight mapper leaves the checkpoint `.w2` names untouched, so
    non-fused layers still need the native `down_proj` parameter name.
    """
    if ".ffn.shared_experts.w" not in name:
        return name
    if extract_layer_index(name) in fuse_by_layer:
        return name.replace(
            ".ffn.shared_experts.w", f".ffn.experts.{n_routed_experts}.w"
        )
    return name.replace(".ffn.shared_experts.w2", ".ffn.shared_experts.down_proj")


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, fuse_shared_experts: bool = False
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). Native DeepSeek FP8
        # linear / shared experts use Fp8LinearMethod's ``weight_scale_inv``.
        # Quark linear block-FP8 registers ``weight_scale`` and is left
        # as-is; load_weights aliases ``.weight_scale`` -> ``.weight_scale_inv``
        # only when the latter is the registered parameter.
        #
        #  - DeepSeek native ``.scale``: expert scales -> ``.weight_scale``,
        #    everything else -> ``.weight_scale_inv``.
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    # When shared experts are fused into the routed MXFP4 grouped GEMM, the
    # shared_experts tensors are redirected to routed expert slots ; leave
    # their names untouched here.
    orig_to_new_substr: dict[str, str | None] = (
        {}
        if fuse_shared_experts
        else {".shared_experts.w2": ".shared_experts.down_proj"}
    )
    orig_to_new_substr["mtp."] = None
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
            ".input_scale": ".input_scale_2",
        },
        orig_to_new_substr=orig_to_new_substr,
    )


def _swa_replay_model_state_cls():
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

    return DeepseekV41ModelState


class DeepseekV4ForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")
    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if expert_dtype != "fp4" or self.model.fused_shared_expert_layers:
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
                expert_dtype,
                fuse_shared_experts=bool(self.model.fused_shared_expert_layers),
            )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )
        # SWA replay (UMBP on ROCm) needs the replay-aware model state: it
        # passes replay_start to the SWA metadata and pads the replayed slots
        # of the prefix-cached groups so their restored KV is not rewritten.
        if dsv4_swa_replay_enabled(vllm_config):
            self.get_model_state_cls = _swa_replay_model_state_cls

    # No engram lookback; read by the replay model state.
    token_lookback_depth = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        # Must precede the gate_up preshuffle of the fused shared experts.
        finalize_mega_moe_weights(self)
        # After per-layer quant finalize, so we preshuffle the final fp8 weights.
        fused_compressor_layers = 0
        rowlocal_layers = 0
        rowlocal_bytes = 0
        for module in self.modules():
            if isinstance(module, DeepseekV4ROCMAiterMLAAttention):
                fused_compressor_layers += module.prepare_compressor_gemm_fusion()
                module.prepare_attn_preshuffle()
                if module._rowlocal_full:
                    rowlocal_layers += 1
                    rowlocal_bytes += module._rowlocal_added_bytes
            elif isinstance(module, DeepseekV4MLP):
                module.prepare_gateup_preshuffle()
        if rowlocal_layers:
            # The rebound wq_b / wo_a TP shards' storage is free now; return it
            # before the KV cache is sized.
            torch.accelerator.empty_cache()
            logger.info(
                "%s: rank %d holds full wq_b/wo_a/wo_b/attn_sink in %d backbone "
                "attention layers (+%.2f GiB); wq_b/wo_a TP shards are "
                "byte-identical row views, wo_b shard checked equal to the full "
                "columns",
                ATTN_ROWLOCAL_LABEL,
                get_tensor_model_parallel_rank(),
                rowlocal_layers,
                rowlocal_bytes / 2**30,
            )
        for module in self.modules():
            if isinstance(module, DeepseekV4DecoderLayer) and module.mono is not None:
                module.mono.prepare(module)
        if fused_compressor_layers:
            logger.info(
                "Fused the C4 compressor GEMMs in %d DeepSeek V4 layers",
                fused_compressor_layers,
            )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
