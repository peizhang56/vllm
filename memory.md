# DeepSeek-V2 fused MLA decode (ROCm / aiter) — work-in-progress notes

## Goal

Match ATOM's 3-kernel DeepSeek-V2 MLA **decode** kernel sequence in vLLM-native
by collapsing four kernels into a single launch of aiter's
`fuse_qk_rope_concat_and_cache_mla_per_head_kernel`.

Off by default; opt-in via env var `VLLM_ROCM_AITER_FUSED_MLA_DECODE=1`.

### Kernel sequence (decode-token slice)

| | Before (vLLM-native) | After (this branch) |
|---|---|---|
| 1 | `triton_poi_fused_..._add_clone_copy_..._stack_unsqueeze_view_1`  *(model-side RoPE)* | `_batched_gemm_a16wfp4_kernel_BLOCK_SIZE`  *(BF16 out, no in-BMM quant)* |
| 2 | `vllm::concat_and_cache_mla_kernel`  *(KV write)* | `aiter::fuse_qk_rope_concat_and_cache_mla_per_head_kernel`  *(RoPE + KV write + cat + Q FP8 quant)* |
| 3 | `_batched_gemm_a16wfp4_kernel_BLOCK_SIZE`  *(GEMM, FP8 out)* | `aiter::mla_a8w8_qh16_qseqlen1_gqaratio16_ps`  *(MLA decode)* |
| 4 | `triton_poi_fused__to_copy_cat_clamp_mul_reciprocal_view_0`  *(`_DecodeConcatQuantFP8`)* | — |
| 5 | `vectorized_elementwise_kernel<FillFunctor<bf16>>`  *(zeros init for `o`)* | — |
| 6 | `aiter::mla_a8w8_qh16_qseqlen1_gqaratio16_ps`  *(MLA decode)* | — |

Source of comparison: `/home/ubuntu/pzhang12/kimi2.5/diff.log`,
`/home/ubuntu/pzhang12/kimi2.5/ATOM/atom/model_ops/attention_mla.py`
(`forward_impl_server_mode` decode branch).

## Repo layout / branch

- Repo: `/home/ubuntu/pzhang12/kimi2.5/vllm-19`
- Base branch: `feat/rocm-fused-ar-rmsnorm` @ `3cc299667` (`[ROCm] Fuse TP all-reduce with residual-add + RMSNorm via aiter`)
- Working branch: **`feat/rocm-aiter-fused-mla-decode`** (uncommitted, 5 files modified)

ATOM reference: `/home/ubuntu/pzhang12/kimi2.5/ATOM/`
Aiter sources (vendored): `/home/ubuntu/pzhang12/kimi2.5/aiter-vllm-v19/`

## User-confirmed design choices

| Question | Choice |
|---|---|
| Scope | `aiter_only` (only AITER MLA backend) |
| Gating | `envvar` (`VLLM_ROCM_AITER_FUSED_MLA_DECODE`, default off) |
| FillFunctor removal | `yes_gated` (only when fused-decode flag is on) |
| Prefill handling | Mixed-batch is now supported in ATOM-parity style: `forward_impl` runs the prefill slice through a single fused CUDA/HIP kernel (`vllm._custom_ops.concat_and_cache_mla_rope_fused`, RoPE + KV-cache write together). Fallback for builds without that op: one aiter HIP kernel (`rope_cached_positions_2c_fwd_inplace`) + standalone `concat_and_cache_mla`. **Never** falls back to vLLM's `DeepseekScalingRotaryEmbedding.forward_hip` — that path silently dispatches `forward_native`, which is the eager-PyTorch swarm of `elementwise_kernel_manual_unroll` launches that caused the original prefill regression. |

## Files changed (6)

```
 vllm/_aiter_ops.py                                       | 174 ++++++++++
 vllm/envs.py                                             |  12 +
 vllm/model_executor/layers/attention/mla_attention.py    | 228 +++++++++++--
 vllm/model_executor/layers/mla.py                        |  ~30 ±
 vllm/v1/attention/backends/mla/rocm_aiter_mla.py         |  48 ++-
 vllm/v1/worker/gpu_model_runner.py                       |  ~20 +
 6 files changed
```

### `vllm/envs.py`

- Declares `VLLM_ROCM_AITER_FUSED_MLA_DECODE: bool = False`.
- Registers env-var lookup, defaults to `False`.

### `vllm/_aiter_ops.py`

- Adds module constant `_FUSED_MLA_DECODE_ENABLED = envs.VLLM_ROCM_AITER_FUSED_MLA_DECODE`.
- Adds `@classmethod is_fused_mla_decode_enabled(cls) -> bool` that requires `_AITER_ENABLED && _MLA_ENABLED && _FUSED_MLA_DECODE_ENABLED`.
- Adds `@staticmethod fused_qk_rope_concat_and_cache_mla_decode(...)` thin wrapper around `aiter.ops.cache.fused_qk_rope_concat_and_cache_mla` (signature in `aiter-vllm-v19/aiter/ops/cache.py`).

### `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` (`AiterMLAImpl`)

- `__init__` derives:
  ```python
  self.fuses_rope_in_decode = (
      rocm_aiter_ops.is_fused_mla_decode_enabled()
      and kv_cache_dtype.startswith("fp8")
      and self.q_lora_rank is not None
      and (num_heads in (4, 8, 16))
  )
  ```
- `forward_mqa` allocates the output tensor `o` with `torch.empty` (not `torch.zeros`) under that flag — eliminates the `vectorized_elementwise_kernel<FillFunctor<bf16>>` launch (the MLA decode kernel overwrites every lane, so the zero-init is dead under this path).

### `vllm/model_executor/layers/mla.py` (`MultiHeadLatentAttentionWrapper`)

- `__init__` caches `self._fuses_rope_in_decode = bool(getattr(self.mla_attn.impl, "fuses_rope_in_decode", False))` once (keeps the per-token forward path `torch.compile`-friendly).
- `__init__` now also passes `rotary_emb=self.rotary_emb` into `MLAAttention(...)` so the fused kernel can reach `cos_sin_cache` and `is_neox_style`.
- `forward`: when `_fuses_rope_in_decode` is True, **skips** the model-side `q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(positions, q[..., self.qk_nope_head_dim:], k_pe)` call entirely. No stashing happens inside the compiled forward — the model runner stashes `positions` into `forward_context.additional_kwargs["mla_positions"]` before entering the compiled model (see `gpu_model_runner.py` below). Otherwise the original rotary call runs unchanged.

### `vllm/v1/worker/gpu_model_runner.py`

- Adds `get_forward_context` to the `vllm.forward_context` import.
- Inside both `execute_model` and `_dummy_run`, immediately after entering the `with set_forward_context(...)` block (and *before* invoking `self.model(...)`), writes:
  ```python
  get_forward_context().additional_kwargs["mla_positions"] = positions
  ```
  This is unconditional (cheap, dict insert; only `MLAAttention` reads the key). Doing the stash *outside* the compiled forward is what makes the fused-decode path work under fullgraph AOT compile (`--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'`): the prior approach stashed inside the wrapper via a `@torch._dynamo.disable`-wrapped helper, which fullgraph capture refuses to skip-inline (gb0099).

### `vllm/model_executor/layers/attention/mla_attention.py` (`MLAAttention`)

- `__init__` accepts new kwarg `rotary_emb: torch.nn.Module | None = None`. Stored as a **plain attribute** via `object.__setattr__(self, "rotary_emb", rotary_emb)` so `nn.Module.__setattr__` does **not** register it as a submodule (would otherwise duplicate the rotary buffers under `MLAAttention`'s state_dict path; the wrapper is the single owner).
- After `self.impl = impl_cls(...)`, when `getattr(self.impl, "fuses_rope_in_decode", False) and self.rotary_emb is not None and hasattr(self.rotary_emb, "cos_sin_cache")`:
  - splits `cos_sin_cache: [max_pos, rot_dim]` into two **contiguous** halves.
  - registers them as non-persistent buffers `_fused_rope_cos_cache` and `_fused_rope_sin_cache`. (Aiter's kernel needs `[max_pos, rot_dim//2]` per buffer; the chunk views from a single concatenated cache are non-contiguous and cannot be passed directly.)
  - else sets both to `None`.
- `forward` (both `use_direct_call` and the `unified_mla_*` branches): wraps the standalone `concat_and_cache_mla` call with `if not getattr(self.impl, "fuses_rope_in_decode", False): ...`. The aiter fused kernel writes the decode-token KV slots itself; for the `unified_*` branch we hand a `torch.empty(0, ...)` dummy to the downstream attn op so its data-dep arg is still satisfied. **Important**: this skips the cache write for the *entire batch*, including prefill — `forward_impl` is responsible for writing the prefill-token KV slots after rotation (see below).
- `__init__` caches `self._has_concat_and_cache_mla_rope_fused = hasattr(vllm._custom_ops, "concat_and_cache_mla_rope_fused")` once, so `forward_impl` doesn't pay a `hasattr` lookup per token.
- `forward_impl`:
  - Resolves `impl_fuses_rope_in_decode = getattr(self.impl, "fuses_rope_in_decode", False)`.
  - **Mixed-batch prefill (ATOM parity)**: when `impl_fuses_rope_in_decode and num_mha_tokens > 0`, applies RoPE + writes prefill KV slots in one of two ways. Both paths read `positions` from `forward_context.additional_kwargs["mla_positions"]` (stashed by the model runner — see `gpu_model_runner.py`).
    - **Preferred (single kernel)**: when `_has_concat_and_cache_mla_rope_fused and kv_cache.numel() > 0`, calls `vllm._custom_ops.concat_and_cache_mla_rope_fused(positions[pf], q[pf, :, qk_nope:], k_pe[pf].squeeze(1), k_c_normed[pf], rotary_emb.cos_sin_cache, rotary_emb.is_neox_style, slot_mapping_pf.flatten(), kv_cache, kv_cache_dtype, _k_scale)`. The kernel rotates `q_pe` and `k_pe` in place AND writes the rotated `k_pe` + `k_c_normed[pf]` to `kv_cache` at `slot_mapping_pf`. Mirrors ATOM's `concat_and_cache_mla_rope_fused` branch (see `ATOM/atom/plugin/attention_mla.py:722`). Layout requirements (TORCH_CHECK-enforced in `csrc/cache_kernels_fused.cu`): `positions` 1D int64, `q_pe` 3D `[N, H, rot_dim]`, `k_pe` 2D `[N, rot_dim]`, `kv_c` 2D `[N, kv_lora_rank]`, `cos_sin_cache` 2D `[max_pos, rot_dim]` (concatenated), `slot_mapping` 1D int64.
    - **Fallback (two kernels)**: when the fused op is missing, calls `aiter.ops.rope.rope_cached_positions_2c_fwd_inplace` followed by `self.impl.do_kv_cache_update(...)`. The aiter kernel requires sbhd 4D inputs `[s, b, h, d]`, **4D cos/sin caches `[max_pos, 1, 1, rot_dim // 2]`**, and 2D positions `[s, b]` — the cos/sin caches stored in `_fused_rope_cos_cache` / `_fused_rope_sin_cache` are 2D for the decode kernel's signature, so we reshape with `.view(max_pos, 1, 1, -1)` at the call site. Forgetting this reshape was the `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 3)` from the first revision of the prefill fix (the kernel does `cos.size(3)` — see `csrc/kernels/rope/general_2c_cached_positions_fwd_kernels.cu:42`).
  - **Why we never call `self.rotary_emb(...)` here**: vLLM's `DeepseekScalingRotaryEmbedding.forward_hip` is just a passthrough to `forward_native`, which is the eager-PyTorch swarm of ~10 `elementwise_kernel_manual_unroll` launches per layer. ATOM gets away with calling its own `rotary_emb` because ATOM uses `aiter.rotary_embedding.RotaryEmbedding`, whose `forward_hip` actually dispatches to the aiter HIP kernel. We bypass the rotary module and call the kernel directly.
  - Sets `bmm_y_scale = None if impl_fuses_rope_in_decode else (self._q_scale if fp8_attention else None)` so the BMM stays BF16 — fused kernel quantizes on the fly.
  - When `impl_fuses_rope_in_decode and self.impl.dcp_world_size <= 1`, dispatches to `_fused_qk_rope_concat_and_cache_mla_decode(...)` which returns `mqa_q` (single concatenated FP8 tensor of shape `[B, N, kv_lora_rank + qk_rope_head_dim]`). DCP path keeps the old code (DCP+fused not supported yet).
  - The downstream `self.impl.forward_mqa(mqa_q, ...)` accepts either a `(ql_nope, q_pe)` tuple or a single tensor — with the fused path we pass the tensor and `forward_mqa`'s `if type(q) is tuple: ...` no-ops.
- New private method `_fused_qk_rope_concat_and_cache_mla_decode(self, mqa_ql_nope, mqa_q_pe, k_c_normed, k_pe, kv_cache, num_mqa_tokens, fp8_attention)`:
  - Reads `positions` from `forward_context.additional_kwargs["mla_positions"]` and `slot_mapping` from `forward_context.slot_mapping[layer_name]`; slices both to `[:num_mqa_tokens]`.
  - Allocates `mqa_q = torch.empty((B, N, kv_lora_rank + qk_rope_head_dim), dtype=current_platform.fp8_dtype() if fp8_attention else bf16)`.
  - Reshapes `kv_cache` into `[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]` for the kernel's expected layout.
  - Calls `rocm_aiter_ops.fused_qk_rope_concat_and_cache_mla_decode(...)` with `is_neox=self.rotary_emb.is_neox_style, is_nope_first=True`.

## Subtle points / gotchas

1. **Tool hallucination during the session.** During the prior session, the file-read tool occasionally returned content that did not match disk (and grep/awk briefly returned 0 hits for strings that *were* on disk). Trust `git diff`, `awk`, and `grep -c` over IDE Read output. The final on-disk state has been re-verified by `awk` slice reads.
2. **State-dict cleanliness.** `MLAAttention.rotary_emb` MUST be set with `object.__setattr__`, not `self.rotary_emb = ...`, otherwise `nn.Module.__setattr__` registers it as a submodule and the rotary `cos_sin_cache` buffer ends up in two state_dict paths (wrapper + mla_attn).
3. **Compile/cudagraph friendliness — `mla_positions` stash must live OUTSIDE the compiled forward.** First attempt put `additional_kwargs["mla_positions"] = positions` inside `MultiHeadLatentAttentionWrapper.forward` behind a `@torch._dynamo.disable(recursive=False)` helper (`_stash_mla_positions`). That works for piecewise/eager but **breaks fullgraph AOT compile** (e.g. `cudagraph_mode: FULL_AND_PIECEWISE`): `torch._dynamo.aot_compile_fullgraph` refuses to skip-inline a `torch.compiler.disable`'d function (`torch._dynamo.exc.Unsupported: gb0099`). Current design moves the stash to `gpu_model_runner.{execute_model,_dummy_run}` right after `set_forward_context(...)` and before invoking the model — the compiled bytecode never touches the dict, and dynamo never sees the mutation.
4. **BF16 BMM under fused path.** The aiter fused kernel reads `mqa_ql_nope` as `scalar_t` and quantizes itself; if the upstream `batched_gemm_a16wfp4` pre-quantizes (`y_scale=self._q_scale`) the kernel sees garbage. The `bmm_y_scale = None` branch in `forward_impl` is therefore load-bearing.
5. **DCP not yet supported.** The fused path only runs when `self.impl.dcp_world_size <= 1`. With DCP > 1 it falls back to the old `_decode_concat_quant_fp8_op` + standalone `concat_and_cache_mla` path inside `forward_impl`. The wrapper-side `concat_and_cache_mla` is still skipped for the whole batch under the flag, so the prefill-cache-write inside `forward_impl` (added in this branch) is what keeps prefill correctness in mixed batches. Re-check this when DCP is wired up.
6. **Head-count gate.** Aiter's kernel is tuned for an effective head count of 16 after the optional head-repeat. We allow `num_heads ∈ {4, 8, 16}` (which the impl repeats up to 16). If you see correctness regressions with other head counts, this is the gate.
7. **FP8 KV cache only.** The fused kernel writes FP8 to cache. Hence the `kv_cache_dtype.startswith("fp8")` guard.

## How to enable / smoke-test

```bash
export VLLM_ROCM_AITER_FUSED_MLA_DECODE=1
# also need the usual aiter MLA backend env (whatever your tree uses)
# e.g. VLLM_ROCM_USE_AITER=1, VLLM_ROCM_USE_AITER_MLA=1, --kv-cache-dtype fp8

# Run any DeepSeek-V2 / V3 decode workload, profile, and confirm the per-layer
# decode kernels collapse to the 3-kernel sequence above.
```

A profile reference for the *before* state lives at
`/home/ubuntu/pzhang12/kimi2.5/diff.log`. The vLLM-side `server_v18.log` is at
`/home/ubuntu/pzhang12/kimi2.5/vllm-tests/server_v18.log`.

## Continuing on a new machine

```bash
cd <repo-root>/vllm-19
git fetch upstream
git checkout feat/rocm-aiter-fused-mla-decode  # if branch was pushed
# OR if not pushed yet:
#   git checkout feat/rocm-fused-ar-rmsnorm
#   git checkout -b feat/rocm-aiter-fused-mla-decode
#   apply the diffs below
```

Anchors to grep when re-orienting in the code:

```
rg 'fuses_rope_in_decode|_fused_qk_rope_concat_and_cache_mla_decode|_fused_rope_cos_cache|mla_positions|VLLM_ROCM_AITER_FUSED_MLA_DECODE|fused_qk_rope_concat_and_cache_mla_decode|is_fused_mla_decode_enabled' vllm
```

Expected hits: `vllm/envs.py`, `vllm/_aiter_ops.py`,
`vllm/v1/attention/backends/mla/rocm_aiter_mla.py`,
`vllm/v1/worker/gpu_model_runner.py`,
`vllm/model_executor/layers/mla.py`,
`vllm/model_executor/layers/attention/mla_attention.py`.

## Kernel sequence (after prefill fix)

| Path | Kernel sequence |
|---|---|
| **Pure decode** | (1) `_batched_gemm_a16wfp4_kernel...` BMM (BF16 out), (2) `aiter::fuse_qk_rope_concat_and_cache_mla_per_head_kernel` (RoPE + concat + cache + Q FP8 quant), (3) `aiter::mla_a8w8_qh16_qseqlen1_gqaratio16_ps`. **Matches ATOM exactly.** |
| **Mixed-batch prefill slice (preferred)** | (1) `concat_and_cache_mla_rope_fused` (RoPE + KV cache write together). **Matches ATOM exactly.** |
| **Mixed-batch prefill slice (fallback)** | (1) `aiter::rope_cached_positions_2c_fwd_impl` (RoPE), (2) `vllm::concat_and_cache_mla_kernel` (KV cache write). Same kernel count as ATOM's `else` branch. |

The wrapper-side rotary in `MultiHeadLatentAttentionWrapper.forward` is **never** invoked under the fused-decode flag (no `triton_poi_fused_..._view_*` kernels in the trace). The decode-token slice is RoPE-rotated inside the fused decode kernel using real positions / real cos+sin caches (no identity-cache trick — the first revision tried that and ate a `FillFunctor<long>` per step from a `torch.zeros(...)` allocation).

## TODO / open items

- [ ] Run perf compare vs. ATOM (decode-only and mixed) and confirm the 3-kernel decode sequence + 1-kernel prefill in profiler.
- [ ] Decide whether to lift the `dcp_world_size <= 1` restriction (DCP + fused).
- [ ] Lint pass clean (`ReadLints` returned no errors); re-run after any further edits.
