# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_tensor, dense_kv_cache_views
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    create_kv_cache_views,
)
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    _zero_kv_blocks_kernel,
)

DEVICE = current_platform.device_type


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
def test_attention_blocks_are_zeroed():
    device = torch.device(DEVICE)
    spec = AttentionSpec(block_size=2, num_kv_heads=1, head_size=1, dtype=torch.uint8)
    storage = torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device)
    layer_name = "draft.self_attn"
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[AttentionGroup(None, [layer_name], spec, 0)],
        kernel_block_sizes=[2],
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storage),
        },
        num_blocks=4,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(storage)
    expected[1] = 0
    assert torch.equal(storage, expected)


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
def test_layers_in_one_group_may_use_different_kernel_pages_per_block():
    """Derive kernel pages per block from each layer's allocation."""
    device = torch.device(DEVICE)
    num_blocks = 4
    spec = AttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
    )
    # Two kernel pages per logical block.
    wide = torch.ones((num_blocks * 2, 3), dtype=torch.int32, device=device)
    # One kernel page per logical block, carved out of a larger allocation so
    # an over-strided write lands in the guard region instead of faulting or
    # silently hitting another tensor.
    narrow_backing = torch.ones((num_blocks * 3, 5), dtype=torch.int32, device=device)
    narrow = narrow_backing[:num_blocks]

    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(
                None,
                ["wide", "narrow"],
                spec,
                0,
            )
        ],
        kernel_block_sizes=[1],
        static_forward_context={
            "wide": SimpleNamespace(kv_cache=wide),
            "narrow": SimpleNamespace(kv_cache=narrow),
        },
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected_wide = torch.ones_like(wide)
    expected_wide[2 * (num_blocks - 1) :] = 0
    assert torch.equal(wide, expected_wide)

    expected_narrow = torch.ones_like(narrow_backing)
    expected_narrow[num_blocks - 1] = 0
    assert torch.equal(narrow_backing, expected_narrow)


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device(DEVICE)
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        page_size_el // page_size_el,  # max_chunks = 1
        page_size_el,  # blk_size
        1,  # n_segs
    )

    stream = torch.Stream(device)
    with stream:
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        if current_platform.is_cuda_alike():
            torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device(DEVICE)
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device

    seg_page_sizes = [page_size_a, page_size_b]
    max_ps = max(seg_page_sizes)

    blk_size = min(1 << (max_ps - 1).bit_length(), 1024)

    zeroer._meta = (
        torch.tensor(
            [storage_a.data_ptr(), storage_b.data_ptr()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        (max_ps + blk_size - 1) // blk_size,
        blk_size,
        2,
    )

    stream = torch.Stream(device)
    with stream:
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
def test_packed_segment_zeros_only_its_last_block_page():
    """A packed KV segment steps by block stride but clears only its page."""
    device = torch.device(DEVICE)
    num_blocks = 4
    block_stride_el = 12
    page_size_el = 4
    page_offset_el = 3
    backing = torch.ones(
        (num_blocks, block_stride_el), dtype=torch.int32, device=device
    )

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor(
            [backing.data_ptr() + page_offset_el * backing.element_size()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor([block_stride_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(backing)
    expected[-1, page_offset_el : page_offset_el + page_size_el] = 0
    assert torch.equal(backing, expected)


def test_large_dsv4_launch_geometry(monkeypatch):
    """Keep the failing DSV4 shape efficient and within launch limits."""
    device = torch.device("cpu")
    n_blocks, n_segs = 6870, 181
    layer_names = [f"layer.{i}" for i in range(n_segs)]
    page_sizes = [9344 if i % 2 == 0 else 292 for i in range(n_segs)]
    spec = AttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
    )
    storages = {
        name: torch.ones((1, page_size), dtype=torch.int32)
        for name, page_size in zip(layer_names, page_sizes)
    }
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(None, [name], spec, group_id)
            for group_id, name in enumerate(layer_names)
        ],
        kernel_block_sizes=[1] * n_segs,
        static_forward_context={
            name: SimpleNamespace(kv_cache=storage)
            for name, storage in storages.items()
        },
        num_blocks=1,
    )

    assert zeroer._meta is not None
    _, _, seg_page_sizes, max_chunks, blk_size, n_segs = zeroer._meta
    assert sorted(seg_page_sizes.tolist()) == sorted(page_sizes)
    assert (max_chunks, blk_size, n_segs) == (10, 1024, 181)

    captured_grids = []

    class FakeKernel:
        def __getitem__(self, grid):
            captured_grids.append(grid)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", FakeKernel())
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: torch.tensor(values, dtype=torch.int64),
    )

    zeroer.zero_block_ids(list(range(n_blocks)))

    old_max_chunks = max(page_sizes) // 4
    assert math.prod((n_blocks, n_segs, old_max_chunks)) > 2**31 - 1
    assert captured_grids == [(n_blocks, n_segs, max_chunks)]


def _random_kv_layout(rng: random.Random, raw: torch.Tensor):
    """Hybrid KV cache groups carved out of `raw`.

    Groups may overlay one allocation with different page sizes and offsets.
    Under block-outermost layouts they share the packed block stride, and a
    BHLNC layer's page spans the block's other layers; otherwise each group's
    block stride is its own, and layouts with contiguous pages may split blocks
    into kernel blocks.
    """
    num_blocks = rng.randint(2, 5)
    groups: list[AttentionGroup] = []
    kernel_block_sizes: list[int] = []
    ctx: dict[str, SimpleNamespace] = {}
    offset = 0
    for _ in range(rng.randint(1, 3)):
        layout = rng.choice(list(KVCacheLayout))
        specs = [
            (
                AttentionSpec(
                    block_size=rng.choice([2, 4]),
                    num_kv_heads=rng.randint(1, 3),
                    head_size=4 * rng.randint(1, 4),
                    dtype=rng.choice([torch.int8, torch.float16, torch.float32]),
                ),
                rng.randint(1, 4),
            )
            for _ in range(rng.randint(1, 3))
        ]
        row = max(spec.page_size_bytes * num_layers for spec, num_layers in specs)
        alloc = raw[offset : offset + num_blocks * row]
        for spec, num_layers in specs:
            tensor = dense_kv_cache_tensor(alloc, spec, num_blocks, num_layers, layout)
            if layout.is_block_outermost:
                tensor = replace(
                    tensor,
                    block_stride=row,
                    offset=4 * rng.randint(0, (row - tensor.block_stride) // 4),
                )
            kernel_block_size = rng.choice([spec.block_size, spec.block_size // 2])
            try:
                views = create_kv_cache_views(
                    alloc, spec, num_blocks, layout, tensor, kernel_block_size
                )
            except ValueError:
                # Only layouts with contiguous pages can split blocks.
                kernel_block_size = spec.block_size
                views = create_kv_cache_views(alloc, spec, num_blocks, layout, tensor)
            names = [f"group{len(groups)}.{i}" for i in range(num_layers)]
            ctx |= {n: SimpleNamespace(kv_cache=v) for n, v in zip(names, views)}
            groups.append(AttentionGroup(None, names, spec, len(groups)))
            kernel_block_sizes.append(kernel_block_size)
        offset += num_blocks * row + 256
    assert offset <= raw.numel()
    return groups, kernel_block_sizes, ctx, num_blocks


def _zero_write_counts(meta, block_ids: list[int], raw: torch.Tensor) -> np.ndarray:
    """How many times the kernel writes each byte of `raw`."""
    seg_addrs, seg_block_strides, seg_page_sizes, *_ = meta
    counts = np.zeros(raw.numel(), dtype=np.int64)
    for addr, stride, page in zip(
        seg_addrs.tolist(), seg_block_strides.tolist(), seg_page_sizes.tolist()
    ):
        for block_id in block_ids:
            start = addr - raw.data_ptr() + block_id * stride * 4
            assert start >= 0 and start + page * 4 <= raw.numel()
            counts[start : start + page * 4] += 1
    return counts


def test_coalesced_segments_zero_the_same_bytes(monkeypatch):
    """Zeroing overlapping segments once must not change which bytes are zeroed,
    nor make the launch grid larger."""
    raw = torch.zeros(1 << 18, dtype=torch.int8)
    num_reduced = 0
    for seed in range(300):
        rng = random.Random(seed)
        args = _random_kv_layout(rng, raw)
        num_blocks = args[-1]
        block_ids = rng.sample(range(num_blocks), rng.randint(1, num_blocks))
        with monkeypatch.context() as m:
            m.setattr(worker_utils, "_coalesce_zero_segments", lambda *table: table)
            per_view = KVBlockZeroer(torch.device("cpu"), *args)._meta
        coalesced = KVBlockZeroer(torch.device("cpu"), *args)._meta
        assert per_view is not None and coalesced is not None

        before = _zero_write_counts(per_view, block_ids, raw)
        after = _zero_write_counts(coalesced, block_ids, raw)
        assert np.array_equal(before > 0, after > 0), seed
        assert after.sum() <= before.sum(), seed
        num_reduced += after.sum() < before.sum()
        *_, chunks_before, _, segs_before = per_view
        *_, chunks_after, _, segs_after = coalesced
        assert chunks_after <= chunks_before and segs_after <= segs_before, seed
    assert num_reduced > 0


def test_coalesced_launch_grid_stays_within_limits(monkeypatch):
    """Under BHLNC each layer's page spans the block's later layers. Their union
    is one 2 GiB interval per block; its pieces must fit the 65,535 limit of
    grid y and z."""
    device = torch.device("meta")
    num_layers, num_blocks = 2048, 2
    spec = AttentionSpec(
        block_size=256, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8, device=device
    )
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, KVCacheLayout.BHLNC)
    names = [f"layer.{i}" for i in range(num_layers)]
    tables = []
    coalesce = worker_utils._coalesce_zero_segments

    def spy(*table):
        tables.extend([table, coalesce(*table)])
        return tables[-1]

    monkeypatch.setattr(worker_utils, "_coalesce_zero_segments", spy)
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[AttentionGroup(None, names, spec, 0)],
        kernel_block_sizes=[spec.block_size],
        static_forward_context={
            n: SimpleNamespace(kv_cache=v) for n, v in zip(names, views)
        },
        num_blocks=num_blocks,
    )
    (_, _, view_pages), (_, _, pages) = tables
    # Uncut, the pages alone would need more than 65,535 chunks on grid z.
    assert math.ceil(max(view_pages) / 1024) > 65535
    assert sum(pages) * 4 == num_layers * spec.page_size_bytes

    assert zeroer._meta is not None
    *_, max_chunks, blk_size, n_segs = zeroer._meta
    assert n_segs <= 65535 and max_chunks <= 65535

    captured_grids = []

    class FakeKernel:
        def __getitem__(self, grid):
            captured_grids.append(grid)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", FakeKernel())
    zeroer.zero_block_ids([1])
    assert captured_grids == [(1, n_segs, max_chunks)]


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
@pytest.mark.parametrize("seed", range(10))
def test_coalesced_kernel_writes_the_same_bytes(monkeypatch, seed):
    """The real kernel leaves the same memory with and without coalescing."""
    device = torch.device(DEVICE)
    rng = random.Random(seed)
    raw = torch.empty(1 << 18, dtype=torch.int8, device=device)
    args = _random_kv_layout(rng, raw)
    num_blocks = args[-1]
    block_ids = rng.sample(range(num_blocks), rng.randint(1, num_blocks))
    with monkeypatch.context() as m:
        m.setattr(worker_utils, "_coalesce_zero_segments", lambda *table: table)
        per_view = KVBlockZeroer(device, *args)
    coalesced = KVBlockZeroer(device, *args)

    results = []
    for zeroer in (per_view, coalesced):
        raw.fill_(1)
        zeroer.zero_block_ids(block_ids)
        torch.accelerator.synchronize()
        results.append(raw.clone())
    assert torch.equal(results[0], results[1])
    assert (results[1] == 0).any()


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
def test_warmup_compiles_for_all_block_counts():
    """After warmup, no launch should trigger a first-request JIT compile.

    The block count is carried by the launch grid, so changing it must reuse
    the warmup's compiled kernel.
    """
    device = torch.device(DEVICE)
    num_blocks = 64
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,  # max_chunks
        page_size_el,  # blk_size
        1,  # n_segs
    )

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids(list(range(n_blocks)))
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device(DEVICE)
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="CUDA or XPU required",
)
@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_zeroes_exactly_one_block_per_layer(layout: KVCacheLayout):
    """The zeroer must zero every byte of the target block in every layer and nothing
    outside it — per head-group region under LHBNC, and never past the target block's
    tile under block-major layouts (no out-of-bounds writes, no clobbering)."""
    device = torch.device(DEVICE)
    num_blocks, num_layers = 4, 2
    spec = AttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32)
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    ).fill_(1)
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    groups = [
        AttentionGroup(
            backend=None,
            layer_names=[f"layer.{i}" for i in range(num_layers)],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    ctx = {f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(num_layers)}
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=iter(groups),
        kernel_block_sizes=[spec.block_size],
        static_forward_context=ctx,
        num_blocks=num_blocks,
    )
    zeroer.zero_block_ids([2])
    torch.accelerator.synchronize()

    for view in views:
        assert (view[2] == 0).all(), layout
        for b in (0, 1, 3):
            assert (view[b].view(torch.int8) == 1).all(), layout
    zero_bytes = int((raw == 0).sum().item())
    assert zero_bytes == num_layers * spec.page_size_bytes, layout
