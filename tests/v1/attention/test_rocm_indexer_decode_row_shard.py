# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm DeepSeek-V4 C4A decode indexer with its rows split across TP ranks
(`_decode_row_shard_topk` in rocm_aiter_mla_sparse.py) against the same op
with every rank scoring every row.

Each rank scores a contiguous share of the decode rows and the int32 top-k is
all-gathered, and the step skips the -1 fill of the rows the exchange writes.
So the sharded op must write every entry of every decode row, -1 past a row
shorter than k, still fill the padded rows past the decode rows with -1, and
pick the same columns per row as the replicated op. Equal logits at the cut
may be chosen differently, and then the chosen logits must be the same
multiset.
"""

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm sparse indexer test.", allow_module_level=True)

from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod

HEADS, DIM, PAGE, RATIO = 64, 128, 64, 4
NUM_PAGES = 4096
PREFIX = "model.layers.2.attn.indexer.k_cache"
SENTINEL = -7
# Step tokens past the decode rows, whose top-k rows the op fills with -1, and
# buffer rows past the step's tokens, which it must not touch.
PAD_ROWS, SPARE_ROWS = 3, 5


def _metadata(num_rows: int, requires_padding: bool = False, rows_per_req: int = 1):
    decode = SimpleNamespace(
        requires_padding=requires_padding,
        decode_lens=torch.full((num_rows // rows_per_req,), rows_per_req),
    )
    return SimpleNamespace(num_decode_tokens=num_rows, decode=decode)


@pytest.mark.parametrize(
    "case,expected",
    [
        ({}, True),
        ({"tp": 1}, False),
        ({"num_rows": sparse_mod._DECODE_ROW_SHARD_MIN_ROWS - 1}, False),
        ({"compress_ratio": 1}, False),
        ({"candidate_blocks": torch.empty(0)}, False),
        ({"use_pcp": True}, False),
        ({"requires_padding": True}, False),
        ({"rows_per_req": 2}, False),
    ],
    ids=[
        "applies",
        "tp1",
        "few_rows",
        "not_c4a",
        "candidates",
        "pcp",
        "padded",
        "native_mtp_rows",
    ],
)
def test_decode_row_shard_applies(monkeypatch, case, expected):
    """The sharded path needs TP > 1, enough flat single-query C4A rows, and
    no candidate blocks, PCP or padded decode rows."""
    case = {"tp": 2, "num_rows": 128, "compress_ratio": RATIO, **case}
    monkeypatch.setattr(
        sparse_mod, "get_tp_group", lambda: SimpleNamespace(world_size=case["tp"])
    )
    metadata = _metadata(
        case["num_rows"],
        case.get("requires_padding", False),
        case.get("rows_per_req", 1),
    )
    assert (
        sparse_mod._decode_row_shard_applies(
            metadata,
            case["compress_ratio"],
            case.get("candidate_blocks"),
            case.get("use_pcp", False),
        )
        is expected
    )


def _make_step(num_rows: int, topk: int, max_model_len: int, seed: int):
    """FP8 queries, head weights, compressed row lengths, block tables and a
    block-flat C4A indexer cache, identical on every rank. The first rows are
    a zero-length (CUDA-graph padding) row and rows around k and the full
    width."""
    gen = torch.Generator().manual_seed(seed)
    fp8 = current_platform.fp8_dtype()
    q = (torch.randn(num_rows, HEADS, DIM, generator=gen) * 0.5).to(fp8)
    weights = torch.randn(num_rows, HEADS, generator=gen)
    lens = torch.randint(1, max_model_len + 1, (num_rows,), generator=gen)
    special = torch.tensor([0, 1, topk - 1, topk, topk + 1, max_model_len])
    lens[: len(special)] = special[:num_rows]
    tables = torch.randint(
        0, NUM_PAGES, (num_rows, max_model_len // PAGE), generator=gen
    ).to(torch.int32)
    vals = (torch.randn(NUM_PAGES, PAGE * DIM, generator=gen) * 0.5).to(fp8)
    scales = torch.rand(NUM_PAGES, PAGE, generator=gen) + 0.5
    kv = torch.cat([vals.view(torch.uint8), scales.view(torch.uint8)], dim=1)
    return [
        t.cuda()
        for t in (
            q,
            weights,
            lens.to(torch.int32),
            tables,
            kv.view(NUM_PAGES, PAGE, DIM + 4),
        )
    ]


def _use_step(step, graph_mode):
    """Point the op's forward context at a decode-only step."""
    _, _, lens, tables, _ = step
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    num_rows = lens.shape[0]
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=tables,
        seq_lens=lens,
        decode_lens=torch.ones(num_rows, dtype=torch.int32, device="cuda"),
        requires_padding=False,
        schedule_metadata=torch.empty(0, dtype=torch.int32, device="cuda"),
    )
    meta = DeepseekV32IndexerMetadata(
        seq_lens=lens,
        max_seq_len=int(lens.max()) * RATIO,
        slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
        num_decodes=num_rows,
        num_decode_tokens=num_rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    context: Any = SimpleNamespace(
        attn_metadata={PREFIX: meta}, cudagraph_runtime_mode=graph_mode
    )
    sparse_mod.get_forward_context = lambda: context


def _run_op(step, topk: int, max_model_len: int, out: torch.Tensor, shard: bool):
    q, weights, lens, tables, kv = step
    sparse_mod.rocm_aiter_sparse_attn_indexer(
        torch.empty(lens.shape[0] + PAD_ROWS, 1, device="cuda"),
        PREFIX,
        kv,
        q,
        None,
        weights,
        128,
        "ue8m0",
        topk,
        DIM,
        max_model_len,
        1,
        out,
        skip_k_cache_insert=True,
        compress_ratio=RATIO,
        decode_row_shard=shard,
    )


def _new_buffer(num_rows: int, topk: int) -> torch.Tensor:
    return torch.full(
        (num_rows + PAD_ROWS + SPARE_ROWS, topk),
        SENTINEL,
        dtype=torch.int32,
        device="cuda",
    )


def _assert_same_topk(step, max_model_len, ref, got, num_rows, topk, label):
    q, weights, lens, tables, kv = step
    dense = sparse_mod.rocm_fp8_paged_mqa_logits(
        q.unsqueeze(1),
        kv.unsqueeze(-2),
        weights,
        lens,
        tables,
        torch.empty(0),
        max_model_len,
        compress_ratio=RATIO,
    ).cpu()
    ref, got, ends = ref.cpu(), got.cpu(), lens.cpu().tolist()
    assert (got[:num_rows] != SENTINEL).all(), f"{label}: unwritten top-k entries"
    assert (got[num_rows : num_rows + PAD_ROWS] == -1).all(), (
        f"{label}: step rows past the decode rows not filled with -1"
    )
    assert (got[num_rows + PAD_ROWS :] == SENTINEL).all(), (
        f"{label}: wrote rows past the step"
    )
    for r, end in enumerate(ends):
        g, w = got[r], ref[r]
        valid = g[g >= 0].long()
        assert ((g >= -1) & (g < end)).all(), f"{label} row {r}: bad entry"
        assert valid.numel() == min(end, topk), f"{label} row {r}: wrong count"
        assert valid.unique().numel() == valid.numel(), f"{label} row {r}: repeat"
        if end <= topk:
            # Every column is selected, then -1 padding.
            assert sorted(valid.tolist()) == list(range(end)), f"{label} row {r}"
            continue
        expected = w[w >= 0].long()
        if set(valid.tolist()) != set(expected.tolist()):
            assert sorted(dense[r, valid].tolist()) == sorted(
                dense[r, expected].tolist()
            ), f"{label} row {r}: different top-{topk} logits"


def _check(num_rows, topk, max_model_len, use_aiter, graph, seed) -> None:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import CUDAGraphMode
    from vllm.distributed.parallel_state import graph_capture

    rocm_aiter_ops._AITER_ENABLED = use_aiter
    label = f"rows={num_rows} k={topk} aiter={use_aiter} graph={graph}"
    mode = CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE
    step = _make_step(num_rows, topk, max_model_len, seed)
    _use_step(step, mode)

    sharded = []
    real_topk = sparse_mod._decode_row_shard_topk

    def record(*args, **kwargs):
        sharded.append(True)
        return real_topk(*args, **kwargs)

    sparse_mod._decode_row_shard_topk = record
    try:
        got = _new_buffer(num_rows, topk)
        _run_op(step, topk, max_model_len, got, shard=True)
        if graph:
            graph_obj = torch.cuda.CUDAGraph()
            with graph_capture(torch.device("cuda")) as ctx:
                got.fill_(SENTINEL)
                with torch.cuda.graph(graph_obj, stream=ctx.stream):
                    _run_op(step, topk, max_model_len, got, shard=True)
            # Replay on a new step of the same row count: new queries,
            # weights and lengths in the captured tensors.
            new_step = _make_step(num_rows, topk, max_model_len, seed + 1)
            for old, new in zip(step, new_step):
                old.copy_(new)
            got.fill_(SENTINEL)
            graph_obj.replay()
            torch.accelerator.synchronize()
            _use_step(step, mode)
        assert sharded, f"{label}: the step was not row-sharded"

        num_sharded = len(sharded)
        ref = _new_buffer(num_rows, topk)
        _run_op(step, topk, max_model_len, ref, shard=False)
        assert len(sharded) == num_sharded, f"{label}: the reference was sharded"
    finally:
        sparse_mod._decode_row_shard_topk = real_topk

    _assert_same_topk(step, max_model_len, ref, got, num_rows, topk, label)


def _run(rank: int, world: int, port: int) -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)
    init_workspace_manager(torch.device("cuda", rank))

    # 2 * world + 1 rows leave the last ranks' shares short (TP8) or empty.
    few_rows = 2 * world + 1
    min_rows = sparse_mod._DECODE_ROW_SHARD_MIN_ROWS
    sparse_mod._DECODE_ROW_SHARD_MIN_ROWS = few_rows
    try:
        _check(few_rows, 512, 4096, False, False, seed=0)
    finally:
        sparse_mod._DECODE_ROW_SHARD_MIN_ROWS = min_rows
    for seed, (num_rows, topk, max_model_len, use_aiter, graph) in enumerate(
        [
            (130, 512, 16384, False, False),
            (130, 1024, 16384, True, False),
            (256, 1024, 65536, False, False),
            (130, 512, 16384, False, True),
            (130, 1024, 16384, True, True),
        ],
        start=1,
    ):
        if use_aiter and not is_aiter_found_and_supported():
            continue
        _check(num_rows, topk, max_model_len, use_aiter, graph, seed=seed)


def _worker(rank: int, world: int, port: int) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    with set_current_vllm_config(config):
        _run(rank, world, port)


def _on_gfx950() -> bool:
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


@pytest.mark.parametrize("world", [2, 8])
def test_decode_row_shard_matches_replicated(world: int):
    if not _on_gfx950():
        pytest.skip("DeepSeek-V4 C4A decode indexer row sharding is tuned on gfx950.")
    if torch.accelerator.device_count() < world:
        pytest.skip(f"Needs {world} GPUs.")
    port = 29500 + (os.getpid() + world) % 1000
    mp.spawn(_worker, args=(world, port), nprocs=world, join=True)
