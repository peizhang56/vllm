# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side tests for the LMCache recurrent-state sidecar.

These exercise the part that decides *correctness* on a hybrid model: a prefix
is only claimed from the external cache at a boundary where every recurrent
state group can be restored as well. The GPU copies are not covered here --
they are plain page copies -- but the boundary gate, the LRU slot map and the
destination-block arithmetic are.
"""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_hybrid_state import (
    HybridStateScheduler,
)
from vllm.v1.core.kv_cache_utils import BlockHash

BLOCK_SIZE = 16
PAGE_BYTES = 4096
NUM_STATE_GROUPS = 2
ATTN_GROUP = 2


def _kv_cache_config(num_blocks: int = 256):
    def group(n_layers: int):
        return SimpleNamespace(
            layer_names=[f"l{i}" for i in range(n_layers)],
            kv_cache_spec=SimpleNamespace(
                block_size=BLOCK_SIZE, page_size_bytes=PAGE_BYTES
            ),
        )

    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[group(2), group(2), group(3)],
    )


def _request(
    req_id: str,
    num_blocks: int,
    num_tokens: int | None = None,
    num_prompt_tokens: int | None = None,
):
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=num_tokens if num_tokens is not None else num_blocks * BLOCK_SIZE,
        num_prompt_tokens=(
            num_prompt_tokens
            if num_prompt_tokens is not None
            else num_blocks * BLOCK_SIZE
        ),
        num_computed_tokens=0,
        num_output_placeholders=0,
        block_hashes=[BlockHash(f"{req_id}-{i}".encode()) for i in range(num_blocks)],
    )


class _Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.is_null = False
        self.block_hash = None
        self.block_hash_num_tokens = None


class _Pool:
    hash_block_size = BLOCK_SIZE

    def __init__(self, n: int):
        self.blocks = [_Block(i) for i in range(n)]


def _cache(pool, block_id, block_hash, group_id, num_tokens):
    """Cache a block the way BlockPool does: group-tagged hash plus the prefix
    length it covers."""
    pool.blocks[block_id].block_hash = _tagged(block_hash, group_id)
    pool.blocks[block_id].block_hash_num_tokens = num_tokens


def _sched_output(req_id: str, block_ids, num_tokens: int, num_computed: int = 0):
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id=req_id, block_ids=block_ids, num_computed_tokens=num_computed
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set(), num_computed_tokens=[]
        ),
        num_scheduled_tokens={req_id: num_tokens},
    )


def _sched_output_cached(req_id: str, new_block_ids, num_tokens: int):
    """A step in which the request is already running (no new blocks)."""
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[req_id],
            new_block_ids=[new_block_ids],
            resumed_req_ids=set(),
            num_computed_tokens=[0],
        ),
        num_scheduled_tokens={req_id: num_tokens},
    )


def _scheduler(cpu_gb: float = 0.001, interval: int = BLOCK_SIZE):
    cfg = _kv_cache_config()
    return HybridStateScheduler(cfg, (0, 1), cpu_gb, interval)


def _blocks(block_ids_by_group):
    """Stand-in for KVCacheBlocks."""
    return SimpleNamespace(get_block_ids=lambda: block_ids_by_group)


def test_no_snapshots_means_no_external_hit():
    """The whole point: an empty state tier must refuse the prefix outright,
    not hand back attention KV that will run on someone else's state."""
    sched = _scheduler()
    req = _request("r0", 8)
    assert sched.clamp_external_hit(req, 0, 6 * BLOCK_SIZE) == 0


def test_hit_is_clamped_to_the_last_jointly_stored_boundary():
    sched = _scheduler()
    req = _request("r0", 8)
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)

    # Three blocks are allocated and hashed for both state groups; the fourth
    # is hashed for group 0 only -- a half-stored boundary that must not be
    # claimed.
    block_ids = ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33])
    sched.record_load(req, _blocks(block_ids), 0)
    for i in range(4):
        pool.blocks[10 + i].block_hash = _tagged(req.block_hashes[i], 0)
    for i in range(3):
        pool.blocks[20 + i].block_hash = _tagged(req.block_hashes[i], 1)
    req.num_computed_tokens = 4 * BLOCK_SIZE

    specs = sched.collect_saves(_sched_output("r0", block_ids, 64))
    assert len(specs) == 7  # 4 for group 0, 3 for group 1
    assert {s.group_id for s in specs} == {0, 1}

    # A 6-block hit is clamped back to the 3 blocks both groups can restore.
    assert sched.clamp_external_hit(req, 0, 6 * BLOCK_SIZE) == 3 * BLOCK_SIZE


def test_load_targets_the_running_state_block():
    """A recurrent state is not addressed by the prefix it covers: in align mode
    the kernel gathers its slot at (seq_len - 1) // block_size of the block
    table, where seq_len is the *end* of the step's tokens. The restore has to
    land there -- the same block vLLM's own local hit copy-on-writes into."""
    sched = _scheduler()
    req = _request("r0", 8)
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    block_ids = ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33])
    sched.record_load(req, _blocks(block_ids), 0)
    for g, base in ((0, 10), (1, 20)):
        for i in range(4):
            pool.blocks[base + i].block_hash = _tagged(req.block_hashes[i], g)
    req.num_computed_tokens = 4 * BLOCK_SIZE
    sched.collect_saves(_sched_output("r0", block_ids, 64))

    warm = _request("r0", 8)
    warm.block_hashes = req.block_hashes
    boundary = sched.clamp_external_hit(warm, 0, 4 * BLOCK_SIZE)
    assert boundary == 4 * BLOCK_SIZE

    # 8 blocks allocated; the step runs the remaining 4 blocks of the prompt,
    # so the state slot for this step is index (128 - 1) // 16 == 7.
    dest = (
        list(range(40, 48)),
        list(range(50, 58)),
        list(range(60, 68)),
    )
    sched.record_load(warm, _blocks(dest), boundary)
    meta = sched.build_meta(
        SimpleNamespace(),
        _sched_output("r0", dest, 4 * BLOCK_SIZE, num_computed=boundary),
    )
    assert {(s.group_id, s.gpu_block_id) for s in meta.loads} == {(0, 47), (1, 57)}


def test_a_load_for_a_request_this_step_does_not_schedule_is_dropped():
    """The destination block only exists once the step says how many tokens the
    request runs. No step, no safe destination."""
    sched = _scheduler()
    req = _request("r0", 4)
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    block_ids = ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33])
    sched.record_load(req, _blocks(block_ids), 0)
    for g, base in ((0, 10), (1, 20)):
        for i in range(4):
            _cache(pool, base + i, req.block_hashes[i], g, (i + 1) * BLOCK_SIZE)
    req.num_computed_tokens = 4 * BLOCK_SIZE
    sched.collect_saves(_sched_output("r0", block_ids, 64))

    warm = _request("r0", 4, num_tokens=128)
    warm.block_hashes = req.block_hashes
    boundary = sched.clamp_external_hit(warm, 0, 4 * BLOCK_SIZE)
    sched.record_load(warm, _blocks(block_ids), boundary)
    meta = sched.build_meta(SimpleNamespace(), _sched_output("other", block_ids, 16))
    assert meta.loads == []


def test_a_mismatched_allocation_refuses_the_state_load():
    """If vLLM allocates a different prefix than the gate planned for, guessing
    which snapshot applies would be silent corruption -- refuse instead."""
    sched = _scheduler()
    req = _request("r0", 8)
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    block_ids = ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33])
    sched.record_load(req, _blocks(block_ids), 0)
    for g, base in ((0, 10), (1, 20)):
        for i in range(4):
            pool.blocks[base + i].block_hash = _tagged(req.block_hashes[i], g)
    req.num_computed_tokens = 4 * BLOCK_SIZE
    sched.collect_saves(_sched_output("r0", block_ids, 64))

    sched.clamp_external_hit(req, 0, 4 * BLOCK_SIZE)
    sched.record_load(req, _blocks(block_ids), 2 * BLOCK_SIZE)
    meta = sched.build_meta(SimpleNamespace(), _sched_output("r0", block_ids, 0))
    assert meta.loads == []


def test_null_and_unhashed_blocks_are_not_snapshotted():
    sched = _scheduler()
    req = _request("r0", 4)
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    block_ids = ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33])
    sched.record_load(req, _blocks(block_ids), 0)
    # Align-mode padding at the front, one real hashed block at the tail.
    for i in range(3):
        pool.blocks[10 + i].is_null = True
        pool.blocks[20 + i].is_null = True
    pool.blocks[13].block_hash = _tagged(req.block_hashes[3], 0)
    pool.blocks[23].block_hash = _tagged(req.block_hashes[3], 1)
    req.num_computed_tokens = 4 * BLOCK_SIZE

    specs = sched.collect_saves(_sched_output("r0", block_ids, 64))
    assert [(s.group_id, s.gpu_block_id) for s in specs] == [(0, 13), (1, 23)]


def test_slots_are_reused_lru_and_evicted_hits_are_refused():
    # 0.001 GiB / 2 groups / (2 layers * 4096 B) = 131 slots per group.
    sched = _scheduler(cpu_gb=2 * 2 * 2 * PAGE_BYTES / (1 << 30))
    assert sched.num_slots[0] == 2

    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    req = _request("r0", 3)
    block_ids = ([10, 11, 12], [20, 21, 22], [30, 31, 32])
    sched.record_load(req, _blocks(block_ids), 0)
    for g, base in ((0, 10), (1, 20)):
        for i in range(3):
            pool.blocks[base + i].block_hash = _tagged(req.block_hashes[i], g)
    req.num_computed_tokens = 3 * BLOCK_SIZE

    # Both slots per group are taken by this step's own transfers, so the third
    # block cannot be snapshotted yet -- and must not be dropped either.
    specs = sched.collect_saves(_sched_output("r0", block_ids, 48))
    assert [(s.group_id, s.gpu_block_id) for s in specs] == [
        (0, 10),
        (0, 11),
        (1, 20),
        (1, 21),
    ]
    # A longer request sharing the same prefix: the cap at `num_tokens - 1`
    # would otherwise mask the slot limit being tested here.
    warm = _request("r0", 3, num_tokens=64)
    assert sched.clamp_external_hit(warm, 0, 3 * BLOCK_SIZE) == 2 * BLOCK_SIZE

    # Next step the reservation is released, so it is retried and evicts the
    # least recently used boundary. Boundary 3 is now the reachable one.
    sched.build_meta(SimpleNamespace(), _sched_output("r0", block_ids, 0))
    specs = sched.collect_saves(
        _sched_output_cached("r0", ([], [], []), 48)
    )
    assert [(s.group_id, s.gpu_block_id) for s in specs] == [(0, 12), (1, 22)]
    assert sched.clamp_external_hit(warm, 0, 3 * BLOCK_SIZE) == 3 * BLOCK_SIZE
    # The evicted boundary is no longer claimable.
    assert sched.clamp_external_hit(warm, 0, 1 * BLOCK_SIZE) == 0


def test_sparse_retention_snapshots_and_clamps_on_the_interval():
    """A recurrent snapshot costs the same whatever span it covers, so the tier
    keeps one per interval and clamps hits down to those boundaries."""
    sched = _scheduler(interval=2 * BLOCK_SIZE)
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    assert sched.snapshot_interval == 2 * BLOCK_SIZE
    req = _request("r0", 5, num_tokens=256, num_prompt_tokens=4 * BLOCK_SIZE)
    block_ids = ([10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34])
    sched.record_load(req, _blocks(block_ids), 0)
    for g, base in ((0, 10), (1, 20)):
        for i in range(5):
            pool.blocks[base + i].block_hash = _tagged(req.block_hashes[i], g)
    req.num_computed_tokens = 5 * BLOCK_SIZE

    # Only blocks 1 and 3 end on a 2-block boundary.
    specs = sched.collect_saves(_sched_output("r0", block_ids, 80))
    assert [(s.group_id, s.gpu_block_id) for s in specs] == [
        (0, 11),
        (0, 13),
        (1, 21),
        (1, 23),
    ]

    # A 3-block hit falls back to the 2-block boundary, not to block 3.
    assert sched.clamp_external_hit(req, 0, 3 * BLOCK_SIZE) == 2 * BLOCK_SIZE
    assert sched.clamp_external_hit(req, 0, 4 * BLOCK_SIZE) == 4 * BLOCK_SIZE
    # Below the first boundary there is nothing to claim.
    assert sched.clamp_external_hit(req, 0, 1 * BLOCK_SIZE) == 0


def test_the_prompt_tail_is_snapshotted_even_below_the_interval():
    """A prompt shorter than the snapshot interval would otherwise be
    snapshotted nowhere, and the tail is the boundary a repeat of the same
    prompt wants to resume from."""
    sched = _scheduler(interval=100 * BLOCK_SIZE)
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    req = _request("r0", 3)
    block_ids = ([10, 11, 12], [20, 21, 22], [30, 31, 32])
    sched.record_load(req, _blocks(block_ids), 0)
    for g, base in ((0, 10), (1, 20)):
        for i in range(3):
            _cache(pool, base + i, req.block_hashes[i], g, (i + 1) * BLOCK_SIZE)
    req.num_computed_tokens = 3 * BLOCK_SIZE

    specs = sched.collect_saves(_sched_output("r0", block_ids, 48))
    assert [(s.group_id, s.gpu_block_id) for s in specs] == [(0, 12), (1, 22)]

    warm = _request("r0", 3, num_tokens=64)
    warm.block_hashes = req.block_hashes
    assert sched.clamp_external_hit(warm, 0, 3 * BLOCK_SIZE) == 3 * BLOCK_SIZE


def test_a_partially_restorable_boundary_loads_nothing():
    """Reloading one group and leaving the other on the previous request's
    state is worse than not reloading at all."""
    sched = _scheduler()
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    req = _request("r0", 4, num_tokens=256)
    block_ids = ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33])
    sched.record_load(req, _blocks(block_ids), 0)
    for i in range(4):
        pool.blocks[10 + i].block_hash = _tagged(req.block_hashes[i], 0)
    for i in range(3):
        pool.blocks[20 + i].block_hash = _tagged(req.block_hashes[i], 1)
    req.num_computed_tokens = 4 * BLOCK_SIZE
    sched.collect_saves(_sched_output("r0", block_ids, 64))

    # Group 0 has block 3, group 1 does not. Approve the boundary by hand to
    # prove the load path refuses on its own rather than emitting half of it.
    sched._pending_boundary["r0"] = (4 * BLOCK_SIZE, 0)
    sched.record_load(req, _blocks(block_ids), 4 * BLOCK_SIZE)
    meta = sched.build_meta(SimpleNamespace(), _sched_output("r0", block_ids, 0))
    assert meta.loads == []


def test_a_load_the_gate_never_approved_is_refused():
    """`update_state_after_alloc` is called for every request, including ones
    whose hit never went through `clamp_external_hit`. Without a boundary of
    our own there is no way to know which snapshot applies."""
    sched = _scheduler()
    pool = _Pool(64)
    sched.bind_gpu_block_pool(pool)
    req = _request("r0", 4, num_tokens=256)
    block_ids = ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33])
    for g, base in ((0, 10), (1, 20)):
        for i in range(4):
            _cache(pool, base + i, req.block_hashes[i], g, (i + 1) * BLOCK_SIZE)
    sched.record_load(req, _blocks(block_ids), 0)
    req.num_computed_tokens = 4 * BLOCK_SIZE
    sched.collect_saves(_sched_output("r0", block_ids, 64))

    sched.record_load(req, _blocks(block_ids), 4 * BLOCK_SIZE)
    meta = sched.build_meta(SimpleNamespace(), _sched_output("r0", block_ids, 0))
    assert meta.loads == []


def _tagged(block_hash, group_id):
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    return make_block_hash_with_group_id(block_hash, group_id)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
