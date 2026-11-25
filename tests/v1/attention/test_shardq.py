# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ token split (vllm/v1/attention/ops/shardq.py), its indexer upload,
and the row-local rows of a sequence-parallel step."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.attention.ops import shardq
from vllm.v1.attention.ops.shardq import build_shardq_split, zigzag_split

@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("lens", [[4096], [65, 300, 7000], [1000, 64, 64, 129]])
def test_zigzag_split_partitions_tokens(world, lens):
    qsl = np.r_[0, np.cumsum(lens)]
    token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
    per_rank = np.split(token_ids, np.cumsum(num_tokens_per_rank)[:-1])
    merged = np.sort(np.concatenate(per_rank))
    np.testing.assert_array_equal(merged, np.arange(qsl[-1]))
    for idx in per_rank:
        assert np.all(np.diff(idx) > 0)
    # Each rank takes one early and one late chunk of every request, so its
    # summed causal positions stay within a couple of tokens per request of
    # the mean.
    positions = [
        sum(int((idx[(idx >= a) & (idx < b)] - a).sum()) for a, b in zip(qsl, qsl[1:]))
        for idx in per_rank
    ]
    assert max(positions) - min(positions) <= sum(4 * n for n in lens)


@pytest.mark.parametrize("num_spec", [0, 2])
def test_build_shardq_split_after_indexer_decodes(monkeypatch, num_spec):
    """The MLA and indexer builders derive the same split from the batch:
    requests of at most 1 + num_spec tokens stay head-sharded, and the tokens
    after them are split once they reach sparse_mla_shard_q_min_tokens and a
    request is longer than the top-k."""
    monkeypatch.setattr(
        shardq, "get_tp_group", lambda: SimpleNamespace(world_size=4, rank_in_group=1)
    )
    query_lens = [1 + num_spec] * 3 + [100, 5000, 4000]
    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[q + 300 for q in query_lens], query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    attention_config = SimpleNamespace(sparse_mla_shard_q_min_tokens=9100)
    hf_config = SimpleNamespace(index_topk=5299)
    vllm_config = SimpleNamespace(
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None
        ),
        attention_config=attention_config,
        model_config=SimpleNamespace(hf_text_config=hf_config),
    )
    split = build_shardq_split(metadata, vllm_config)
    assert split is not None
    assert split.num_decode_tokens == 3 * (1 + num_spec)
    assert split.num_tokens_per_rank.sum() == 9100
    hf_config.index_topk = 5300
    assert build_shardq_split(metadata, vllm_config) is None
    hf_config.index_topk = 5299
    attention_config.sparse_mla_shard_q_min_tokens = 9101
    assert build_shardq_split(metadata, vllm_config) is None
    # Five tokens split into eight chunks, three of them empty, leaving rank 2
    # without tokens.
    attention_config.sparse_mla_shard_q_min_tokens = 1
    short = create_common_attn_metadata(
        BatchSpec(seq_lens=[5300], query_lens=[5]),
        block_size=16,
        device=torch.device("cpu"),
    )
    assert build_shardq_split(short, vllm_config) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_indexer_upload_survives_next_build(monkeypatch):
    """Building the next step must not change a still-queued upload."""
    from vllm.v1.attention.backends.mla.indexer import upload_shardq_indices

    monkeypatch.setattr(
        shardq, "get_tp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=0)
    )

    def upload(lens):
        split = shardq.ShardQSplit(0, *zigzag_split(np.array(lens), 2))
        qsl = torch.tensor(np.r_[0, np.cumsum(lens)], dtype=torch.int32)
        return upload_shardq_indices(split, qsl, torch.device("cuda"))[0], split

    torch.cuda._sleep(1 << 30)  # keep both copies queued
    first, split = upload([3000, 1200])
    upload([700])
    torch.accelerator.synchronize()
    np.testing.assert_array_equal(first.cpu().numpy(), split.local_token_ids)




@pytest.mark.parametrize("num_decode_tokens", [0, 5])
def test_rowlocal_rows_follow_sp_slices(monkeypatch, num_decode_tokens):
    """PrefillRowShard.rowlocal: inside a row-local call on a row-local step,
    rank k scores the prefill rows of its slice [k*s, (k+1)*s); otherwise
    there is no row-local shard."""
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.model_executor.layers.sparse_attn_indexer import PrefillRowShard

    world, num_prefill = 4, 1000
    m = num_decode_tokens + num_prefill
    s = -(-m // world)
    prefill = SimpleNamespace(shardq_rowlocal_rows=num_prefill)
    monkeypatch.setattr(
        sparse_attn_indexer, "get_tp_group", lambda: SimpleNamespace(world_size=world)
    )
    covered = []
    for rank in range(world):
        monkeypatch.setattr(
            sparse_attn_indexer, "get_tensor_model_parallel_rank", lambda r=rank: r
        )
        monkeypatch.setattr(sparse_attn_indexer, "_ROWLOCAL_ROWS", None)
        assert PrefillRowShard.rowlocal(prefill, num_decode_tokens) is None
        monkeypatch.setattr(sparse_attn_indexer, "_ROWLOCAL_ROWS", (rank * s, s))
        shard = PrefillRowShard.rowlocal(prefill, num_decode_tokens)
        assert shard.start == min(max(rank * s, num_decode_tokens), m)
        assert shard.stop == min(max((rank + 1) * s, num_decode_tokens), m)
        covered.extend(range(shard.start, shard.stop))
        assert (
            PrefillRowShard.rowlocal(
                SimpleNamespace(shardq_rowlocal_rows=None), num_decode_tokens
            )
            is None
        )
    assert covered == list(range(num_decode_tokens, m))
