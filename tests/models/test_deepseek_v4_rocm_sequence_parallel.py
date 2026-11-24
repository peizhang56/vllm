# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import SamplingParams
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="DeepSeek-V4 AMD model"
)

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
TP_SIZE = 2


def _setup_worker(worker, sequence_parallel: bool):
    """Record each layer's rows and attention input on steps of >= 1024 rows.

    QuickReduce stays configured, so the model builds its sequence-parallel
    layers, but its INT4 codec is turned off: on dummy weights the codec's
    error swamps any difference between the two layouts. The INT4
    reduce-scatter has its own test in tests/distributed.
    """
    from vllm.distributed import get_tp_group

    qr_comm = get_tp_group().device_communicator.qr_comm
    qr_comm.should_quick_allreduce = lambda inp: False
    qr_comm.should_quick_reduce_scatter = lambda inp: False

    model = worker.model_runner.model.model
    assert model.sequence_parallel
    model.sequence_parallel = sequence_parallel
    worker.layer_rows = []
    worker.attn_inputs = []

    def record_rows(module, args):
        worker.layer_rows.append((args[1].shape[0], args[0].shape[0]))

    def record_attn_input(module, args):
        if args[1].shape[0] >= 1024:
            worker.attn_inputs.append(args[1].float().cpu())

    for layer in model.layers:
        layer.register_forward_pre_hook(record_rows)
        layer.attn.register_forward_pre_hook(record_attn_input)


def test_sequence_parallel_shards_mhc_rows(vllm_runner, monkeypatch):
    """Steps of at least 1024 tokens run mHC on each rank's rows, and the
    sub-layers see the same input as with replicated rows."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")
    try:
        from aiter.ops.quick_reduce_scatter import SUPPORTED_WORLD_SIZES
    except ImportError:
        SUPPORTED_WORLD_SIZES = ()
    if TP_SIZE not in SUPPORTED_WORLD_SIZES:
        pytest.skip("AITER has no QuickReduce reduce-scatter for this world size.")

    prompt = {"prompt_token_ids": [2 + i % 997 for i in range(3073)]}
    num_layers = 4
    attn_inputs: list[list[torch.Tensor]] = []
    for sequence_parallel in (False, True):
        with vllm_runner(
            MODEL,
            load_format="dummy",
            hf_overrides={
                "num_hidden_layers": num_layers,
                "num_nextn_predict_layers": 0,
            },
            skip_tokenizer_init=True,
            tensor_parallel_size=TP_SIZE,
            enforce_eager=True,
            max_model_len=4096,
            max_num_batched_tokens=1025,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            block_size=256,
            kv_cache_dtype="fp8",
            gpu_memory_utilization=0.5,
        ) as runner:
            runner.llm.collective_rpc(
                _setup_worker, kwargs={"sequence_parallel": sequence_parallel}
            )
            runner.llm.generate(
                prompt, SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False
            )
            for rows in runner.llm.collective_rpc(lambda w: w.layer_rows):
                large = [(full, local) for full, local in rows if full >= 1024]
                assert large
                assert all(
                    local == (-(-full // TP_SIZE) if sequence_parallel else full)
                    for full, local in large
                )
            attn_inputs.append(runner.llm.collective_rpc(lambda w: w.attn_inputs)[0])

    replicated, sharded = attn_inputs
    assert len(replicated) == len(sharded) > 0
    # The first layer's attention input comes from mHC alone, so it matches
    # exactly; one layer later the reduce-scatter has run once.
    for step in range(0, len(replicated), num_layers):
        torch.testing.assert_close(sharded[step], replicated[step], atol=0, rtol=0)
        ref, out = replicated[step + 1], sharded[step + 1]
        assert (out - ref).norm() / ref.norm() < 1e-2
