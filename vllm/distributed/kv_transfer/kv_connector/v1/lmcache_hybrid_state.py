# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recurrent-state sidecar for :class:`LMCacheConnectorV1` on hybrid models.

Why this exists
---------------
LMCache stores paged *attention* KV as token chunks. A hybrid model such as
Kimi-K3 runs 69 of its 93 layers on KimiDeltaAttention, whose per-request
recurrent state lives in separate ``MambaSpec`` KV cache groups that LMCache
has no format for. Restoring an attention prefix on top of whatever recurrent
state happens to be resident is *silent wrong output* -- no crash, no failed
load, just a corrupt answer. That is measured, not theoretical.

This module supplies the missing half:

* a small CPU tier holding the state groups' pages, keyed by the same block
  hashes vLLM already computes for its prefix cache, and
* a **joint boundary gate**: an external hit is clamped to a prefix at which
  *every* group -- attention and recurrent state alike -- can be restored.

That pairing is exactly what vLLM's in-tree ``SimpleCPUOffloadConnector`` gets
for free from its CPU-side ``KVCacheCoordinator``, and that connector is the
existence proof that the pairing is sufficient: on the same model, platform and
speculative-decoding config it reloads an evicted prefix correctly, where
LMCache alone returns garbage. The block-walking logic below deliberately
mirrors ``vllm/v1/simple_kv_offload/manager.py`` so the two stay comparable.

Layout note
-----------
Under the hybrid memory allocator all groups share **one** block-id space and
one set of backing buffers: buffer *k* is shared by the *k*-th layer of every
group, and a given page belongs to whichever group was allocated that block id.
So mirroring "page *b* of the buffers backing group *g*'s layers" captures
exactly group *g*'s state at block *b*, with no knowledge of the Mamba layout
required. Copies here are whole pages for that reason.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, make_block_hash_with_group_id

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Per-rank CPU budget for state snapshots, in GiB. One snapshot is one block
# for one group across that group's layers -- on Kimi-K3 at TP=8 that is
# 23 layers x 864 KiB ~= 19 MiB, so the default holds ~1700 of them.
DEFAULT_STATE_CPU_GB = 32.0

# Snapshot (and therefore allow a load at) only every Nth token boundary.
#
# A recurrent snapshot is the whole state, so it costs the same whether it
# covers 128 tokens or 128k: on Kimi-K3 one boundary is ~57 MiB across the
# three groups, i.e. ~2.2k tokens of reach per GiB if every block boundary is
# kept. Attention KV is ~200x denser, so dense state retention -- not the
# attention tier -- would set the size of the whole cache.
#
# Keeping one boundary per interval costs at most `interval` tokens of extra
# prefill on a hit, and buys a proportional increase in how far back the tier
# reaches. vLLM makes the same trade on the GPU side (`retention_interval` in
# MambaManager.get_retention_mask).
DEFAULT_SNAPSHOT_INTERVAL_TOKENS = 1024


@dataclass
class StateTransferSpec:
    """One page copy: group ``group_id``'s block ``gpu_block_id`` <-> ``cpu_slot``."""

    group_id: int
    gpu_block_id: int
    cpu_slot: int


@dataclass
class LMCacheHybridMetadata(KVConnectorMetadata):
    """LMCache's own metadata plus this step's state-page transfers.

    ``LMCacheConnectorV1.bind_connector_metadata`` unwraps this and binds
    ``inner`` as the connector metadata, so LMCache's adapter -- which reaches
    back through ``self._parent._get_connector_metadata()`` -- sees exactly what
    it built and is unaware of the sidecar.
    """

    inner: KVConnectorMetadata
    loads: list[StateTransferSpec] = field(default_factory=list)
    saves: list[StateTransferSpec] = field(default_factory=list)


@dataclass
class _LoadPlan:
    """An approved boundary, waiting for the step that says where it lands."""

    boundary: int
    block_ids: tuple[list[int], ...]
    slots: list[tuple[int, int]]


@dataclass
class _StoreState:
    """Per-request block ids accumulated from ``scheduler_output``."""

    request: "Request"
    block_ids: tuple[list[int], ...]
    num_scanned: list[int]
    # Prefix length, per group, at which this request last took a snapshot.
    last_snapshot: dict[int, int] = field(default_factory=dict)


class HybridStateScheduler:
    """Scheduler half: owns the CPU slot map and decides the joint boundary."""

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        state_group_ids: tuple[int, ...],
        cpu_gb_per_rank: float,
        snapshot_interval_tokens: int = DEFAULT_SNAPSHOT_INTERVAL_TOKENS,
    ):
        self.kv_cache_config = kv_cache_config
        self.state_group_ids = state_group_ids
        self.num_groups = len(kv_cache_config.kv_cache_groups)

        groups = kv_cache_config.kv_cache_groups
        self.block_sizes = [g.kv_cache_spec.block_size for g in groups]
        # Two different block sizes are in play and conflating them is the easy
        # mistake here. A *cache block* of a state group is `block_sizes[g]`
        # tokens wide -- under HMA that is inflated (1536 on Kimi-K3) so that
        # one page fits a whole recurrent state. Block *hashes*, though, are
        # still computed every `hash_block_size` tokens (128), and a state block
        # is cached under the hash of whatever prefix it actually holds, which
        # is why MambaManager sets `supports_fine_grained_hash_lookup`. So a
        # boundary is identified by a hash index, and the block that holds it
        # sits at `hash_index // (block_size // hash_block_size)`.
        # `hash_block_size` only becomes known when the block pool is bound.
        self.hash_block_size = min(self.block_sizes) if self.block_sizes else 1
        self.snapshot_interval = max(1, snapshot_interval_tokens)
        # Bytes for one snapshot of one group = one page per layer of it.
        self.group_slot_bytes = {
            g: groups[g].kv_cache_spec.page_size_bytes * len(groups[g].layer_names)
            for g in state_group_ids
        }

        budget = int(cpu_gb_per_rank * (1 << 30)) // max(1, len(state_group_ids))
        self.num_slots = {
            g: max(1, budget // self.group_slot_bytes[g]) for g in state_group_ids
        }

        # hash -> slot, in LRU order (oldest first). One map per state group;
        # hashes are already group-tagged, but keeping the maps separate makes
        # the per-group slot spaces independent.
        self._slots: dict[int, OrderedDict[BlockHashWithGroupId, int]] = {
            g: OrderedDict() for g in state_group_ids
        }
        self._free: dict[int, list[int]] = {
            g: list(range(self.num_slots[g])) for g in state_group_ids
        }
        # Slots touched by this step's transfers; never evict these.
        self._reserved: set[tuple[int, int]] = set()

        self._gpu_block_pool: "BlockPool | None" = None
        self._reqs_to_store: dict[str, _StoreState] = {}
        # req_id -> planned absolute boundary in tokens, from the clamp.
        # req_id -> (absolute boundary, locally computed prefix at gate time).
        # The second half has to be remembered rather than recomputed: by the
        # time `record_load` runs, `allocate_slots` has already hashed every
        # full block of the prompt, so "count the hashed blocks" reports the
        # whole prompt as locally cached.
        self._pending_boundary: dict[str, tuple[int, int]] = {}
        self._pending_load_plans: dict[str, _LoadPlan] = {}

        reach = next(iter(self.num_slots.values())) * self.snapshot_interval
        logger.info(
            "LMCache hybrid state tier: groups %s, %.1f MiB per snapshot, "
            "%d snapshots per group every %d tokens (%.1f GiB per rank, "
            "reaching %d tokens back)",
            list(state_group_ids),
            next(iter(self.group_slot_bytes.values())) / (1 << 20),
            next(iter(self.num_slots.values())),
            self.snapshot_interval,
            cpu_gb_per_rank,
            reach,
        )

    # ---------------- boundary gate ----------------

    def clamp_external_hit(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_hit_tokens: int,
    ) -> int:
        """Clamp LMCache's absolute hit length to a restorable joint boundary.

        Returns the largest boundary ``<= num_hit_tokens`` for which every state
        group has a stored snapshot, or ``num_computed_tokens`` (i.e. no
        external gain) when there is none. Snapshots are taken roughly every
        ``snapshot_interval`` tokens -- see the constant for why the state tier
        is deliberately sparse -- but at whichever hash boundaries the prefix
        cache actually produced, so this searches down rather than assuming a
        fixed stride.

        The result is always hash-block aligned and strictly below
        ``num_tokens``: that keeps LMCache off its "full prompt hit, recompute
        the last token" path, whose -1 would leave ``num_external_tokens``
        unaligned and break the back-trace in :meth:`record_load`.
        """
        hash_size = self.hash_block_size
        block_hashes = request.block_hashes

        max_tokens = min(num_hit_tokens, request.num_tokens - 1)
        k = min(max_tokens // hash_size, len(block_hashes))
        # Saves are throttled to one per interval, so the newest usable
        # boundary is at most that far below the hit. A couple of intervals of
        # slack covers boundaries the prefix cache happened not to produce.
        steps = 2 * cdiv(self.snapshot_interval, hash_size) + 2
        while k > 0 and steps > 0:
            bh = block_hashes[k - 1]
            if all(
                make_block_hash_with_group_id(bh, g) in self._slots[g]
                for g in self.state_group_ids
            ):
                break
            k -= 1
            steps -= 1
        else:
            k = 0

        boundary = k * hash_size
        if boundary <= num_computed_tokens:
            # Nothing to gain: vLLM already has at least this much locally, and
            # its own Mamba manager handles the local hit.
            self._pending_boundary.pop(request.request_id, None)
            return num_computed_tokens

        self._pending_boundary[request.request_id] = (boundary, num_computed_tokens)
        if boundary < num_hit_tokens:
            logger.debug(
                "Req %s: LMCache hit %d tokens, clamped to %d by the state "
                "boundary.",
                request.request_id,
                num_hit_tokens,
                boundary,
            )
        return boundary

    # ---------------- load path ----------------

    def record_load(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Emit the state-page loads that must accompany LMCache's KV load."""
        req_id = request.request_id
        planned = self._pending_boundary.pop(req_id, None)
        block_ids_by_group = blocks.get_block_ids()

        if req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = _StoreState(
                request=request,
                block_ids=tuple([] for _ in range(self.num_groups)),
                num_scanned=[0] * self.num_groups,
            )

        if num_external_tokens == 0:
            return

        if planned is None:
            logger.error(
                "Req %s: LMCache is loading %d tokens the state gate never "
                "approved; skipping the state load. The attention KV it is "
                "about to restore will run on stale recurrent state.",
                req_id,
                num_external_tokens,
            )
            return

        total, gated_computed = planned
        if gated_computed + num_external_tokens != total:
            # LMCache's own assert would already have fired on a short alloc;
            # if it ever does not, refuse to guess which snapshot applies.
            logger.error(
                "Req %s: state boundary %d != allocated prefix %d (%d local + "
                "%d external); skipping the state load. The attention KV "
                "LMCache is about to restore will run on stale recurrent state.",
                req_id,
                total,
                gated_computed + num_external_tokens,
                gated_computed,
                num_external_tokens,
            )
            return

        # Build the whole set first: a partially restored boundary -- some
        # groups reloaded, others left holding the previous request's state --
        # is worse than not reloading at all.
        planned_slots: list[tuple[int, int]] = []
        hash_size = self.hash_block_size
        if total % hash_size or total // hash_size > len(request.block_hashes):
            logger.error(
                "Req %s: prefix %d is not a hash boundary (%d) or is past the "
                "request's hashes; skipping the state load.",
                req_id, total, hash_size,
            )
            return
        hash_idx = total // hash_size - 1
        for g in self.state_group_ids:
            bh = make_block_hash_with_group_id(request.block_hashes[hash_idx], g)
            slot = self._slots[g].get(bh)
            if slot is None:
                logger.error(
                    "Req %s: no state snapshot for group %d at prefix %d, but "
                    "the boundary gate claimed one. Skipping the state load.",
                    req_id, g, total,
                )
                return
            self._slots[g].move_to_end(bh)
            planned_slots.append((g, slot))

        # Which GPU block to restore into is not knowable yet: it depends on how
        # many tokens the scheduler ends up giving this request in this step.
        # `build_meta` resolves it once `scheduler_output` says.
        self._pending_load_plans[req_id] = _LoadPlan(
            boundary=total, block_ids=block_ids_by_group, slots=planned_slots
        )
        for g, slot in planned_slots:
            self._reserved.add((g, slot))

    def _resolve_load_plans(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[StateTransferSpec]:
        """Turn each approved boundary into a copy into the block the kernel
        will actually read.

        A recurrent state is not addressed by the prefix it covers. In align
        mode the linear-attention backend gathers its state slot at
        ``(seq_len - 1) // block_size`` of the request's block table
        (``mamba_get_block_table_tensor``), where ``seq_len`` is the *end* of
        this step's tokens -- the running-state block, which the kernel reads
        the initial state from and then overwrites. vLLM's own local mamba hit
        does not leave the cached block where the hit landed either; it
        copy-on-writes it into that running block first (``_apply_cow``). The
        restore has to land in the same place, or the kernel starts the
        continuation from whatever the block happened to hold.
        """
        if not self._pending_load_plans:
            return []
        computed: dict[str, int] = {
            r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs
        }
        cached = scheduler_output.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))

        specs: list[StateTransferSpec] = []
        pool = self._gpu_block_pool
        for req_id, plan in self._pending_load_plans.items():
            scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            num_computed = computed.get(req_id)
            if not scheduled or num_computed is None:
                logger.error(
                    "Req %s: approved a state load but the request is not in "
                    "this step's scheduler output; dropping it.",
                    req_id,
                )
                self._release(plan)
                continue
            seq_len = num_computed + scheduled
            resolved: list[StateTransferSpec] = []
            for g, slot in plan.slots:
                idx = max(0, (seq_len - 1) // self.block_sizes[g])
                group_gpu_ids = plan.block_ids[g]
                if idx >= len(group_gpu_ids) or (
                    pool is not None and pool.blocks[group_gpu_ids[idx]].is_null
                ):
                    logger.error(
                        "Req %s: group %d running-state block %d (seq_len %d) "
                        "is missing or null; skipping the state load.",
                        req_id, g, idx, seq_len,
                    )
                    resolved = []
                    break
                resolved.append(
                    StateTransferSpec(
                        group_id=g, gpu_block_id=group_gpu_ids[idx], cpu_slot=slot
                    )
                )
            if not resolved:
                self._release(plan)
                continue
            logger.debug(
                "state load: req=%s boundary=%d -> seq_len %d, %s",
                req_id,
                plan.boundary,
                seq_len,
                [(s.group_id, s.gpu_block_id, s.cpu_slot) for s in resolved],
            )
            specs.extend(resolved)
        self._pending_load_plans.clear()
        return specs

    def _release(self, plan: "_LoadPlan") -> None:
        for g, slot in plan.slots:
            self._reserved.discard((g, slot))

    # ---------------- save path ----------------

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._gpu_block_pool = gpu_block_pool
        # Boundaries live on hash-block granularity, which is only knowable
        # from the pool: under HMA a state group's cache blocks are much wider
        # than the hashes that address them.
        self.hash_block_size = getattr(
            gpu_block_pool, "hash_block_size", self.hash_block_size
        )
        self.snapshot_interval = (
            cdiv(self.snapshot_interval, self.hash_block_size) * self.hash_block_size
        )
        reach = next(iter(self.num_slots.values())) * self.snapshot_interval
        logger.info(
            "LMCache hybrid state tier: hash block %d tokens, cache block %d, "
            "snapshot every %d tokens -- reaching ~%d tokens back per group.",
            self.hash_block_size,
            self.block_sizes[self.state_group_ids[0]],
            self.snapshot_interval,
            reach,
        )

    def request_finished(self, request_id: str) -> None:
        self._reqs_to_store.pop(request_id, None)
        self._pending_boundary.pop(request_id, None)

    def _alloc_slot(self, group_id: int) -> int | None:
        free = self._free[group_id]
        if free:
            return free.pop()
        # Evict the least recently used snapshot that is not in flight.
        for bh, slot in self._slots[group_id].items():
            if (group_id, slot) not in self._reserved:
                del self._slots[group_id][bh]
                return slot
        return None

    def _is_snapshot_boundary(
        self, request: "Request", covers: int, state: _StoreState, group_id: int
    ) -> bool:
        """Sparse retention: which boundaries are worth a whole state snapshot.

        A snapshot is the entire recurrent state whatever span it marks, so
        keeping one per block would cost ~200x what the attention tier costs
        for the same reach. Two kinds are kept:

        * one every ``snapshot_interval`` tokens, so a long prefix stays
          resumable somewhere near where a truncated re-send would land, and
        * the last boundary inside the prompt, which is where a repeat of the
          *same* prompt wants to resume and is by far the most valuable one.
          Without it a prompt shorter than the interval would be snapshotted
          nowhere at all. vLLM keeps the same boundary for the same reason --
          see ``reachable_boundaries`` in ``MambaManager.get_retention_mask``.
        """
        if covers <= state.last_snapshot.get(group_id, 0):
            return False
        if covers % self.snapshot_interval == 0:
            return True
        tail = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
        )
        return covers == tail

    def collect_saves(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[StateTransferSpec]:
        """Snapshot every newly hashed state block, mirroring
        ``SimpleCPUOffloadScheduler._prepare_eager_store_specs``."""
        pool = self._gpu_block_pool
        if pool is None:
            return []

        specs: list[StateTransferSpec] = []
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None:
                continue
            if preempted:
                state.block_ids = tuple([] for _ in range(self.num_groups))
                state.num_scanned = [0] * self.num_groups
            if new_block_id_groups:
                for g in range(min(self.num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        state.block_ids[g].extend(new_block_id_groups[g])

            if not scheduler_output.num_scheduled_tokens.get(req_id, 0):
                continue

            req = state.request
            # Only blocks whose KV is written and visible are snapshottable.
            confirmed = req.num_computed_tokens - req.num_output_placeholders

            for g in self.state_group_ids:
                block_size = self.block_sizes[g]
                ready = confirmed // block_size
                start = state.num_scanned[g]
                scannable = state.block_ids[g][start:ready]
                for offset, gpu_block_id in enumerate(scannable):
                    block = pool.blocks[gpu_block_id]
                    if block.is_null:
                        # Align-mode padding: no state lives here.
                        state.num_scanned[g] += 1
                        continue
                    bh = block.block_hash
                    if bh is None:
                        state.num_scanned[g] += 1
                        continue
                    # The prefix this block was cached under. A state block can
                    # be cached at a partial boundary, so take the recorded
                    # length rather than assuming the block is full.
                    covers = block.block_hash_num_tokens
                    if covers is None:
                        covers = (start + offset + 1) * block_size
                    if bh in self._slots[g]:
                        # Already covered; count it so the throttle below does
                        # not immediately re-snapshot the next boundary.
                        state.last_snapshot[g] = max(
                            state.last_snapshot.get(g, 0), covers
                        )
                        state.num_scanned[g] += 1
                        continue
                    if not self._is_snapshot_boundary(req, covers, state, g):
                        state.num_scanned[g] += 1
                        continue
                    slot = self._alloc_slot(g)
                    if slot is None:
                        # Every slot is spoken for by this step's transfers.
                        # Leave the cursor so the block is retried next step
                        # rather than silently never snapshotted.
                        break
                    state.num_scanned[g] += 1
                    state.last_snapshot[g] = covers
                    self._slots[g][bh] = slot
                    self._reserved.add((g, slot))
                    logger.debug(
                        "state snapshot: req=%s g=%d list_idx=%d block=%d "
                        "covers=%d computed=%d slot=%d",
                        req_id, g, start + offset, gpu_block_id, covers,
                        confirmed, slot,
                    )
                    specs.append(
                        StateTransferSpec(
                            group_id=g, gpu_block_id=gpu_block_id, cpu_slot=slot
                        )
                    )
        return specs

    def build_meta(
        self, inner: KVConnectorMetadata, scheduler_output: "SchedulerOutput"
    ) -> LMCacheHybridMetadata:
        loads = self._resolve_load_plans(scheduler_output)
        saves = self.collect_saves(scheduler_output)
        self._reserved.clear()
        return LMCacheHybridMetadata(inner=inner, loads=loads, saves=saves)


class HybridStateWorker:
    """Worker half: the CPU mirror and the page copies."""

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        state_group_ids: tuple[int, ...],
        num_slots: dict[int, int],
    ):
        self.kv_cache_config = kv_cache_config
        self.state_group_ids = state_group_ids
        self.num_slots = num_slots
        # group -> list of [num_blocks, page_bytes] int8 views, one per layer.
        self._gpu: dict[int, list[torch.Tensor]] = {}
        self._cpu: dict[int, list[torch.Tensor]] = {}

    def register_kv_caches(self, kv_caches: dict[str, object]) -> None:
        num_blocks = self.kv_cache_config.num_blocks
        groups = self.kv_cache_config.kv_cache_groups
        total_bytes = 0

        for g in self.state_group_ids:
            gpu_views: list[torch.Tensor] = []
            cpu_views: list[torch.Tensor] = []
            for name in groups[g].layer_names:
                value = kv_caches.get(name)
                if value is None:
                    continue
                # Mamba layers arrive either as one opaque page view or as a
                # list of views (conv/ssm) into the same buffer. Either way the
                # unit of transfer is the whole page, so work off the storage.
                tensor = value if isinstance(value, torch.Tensor) else value[0]
                storage = tensor.untyped_storage()
                raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(
                    storage, 0, (storage.nbytes(),)
                )
                gpu_view = raw.view(num_blocks, -1)
                gpu_views.append(gpu_view)
                cpu = torch.empty(
                    (self.num_slots[g], gpu_view.shape[1]),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=True,
                )
                cpu_views.append(cpu)
                total_bytes += cpu.numel()
            self._gpu[g] = gpu_views
            self._cpu[g] = cpu_views

        logger.info(
            "LMCache hybrid state tier: mirrored %d recurrent-state layer(s) "
            "across %d group(s), %.2f GiB of pinned CPU memory on this rank.",
            sum(len(v) for v in self._gpu.values()),
            len(self._gpu),
            total_bytes / (1 << 30),
        )

    def load(self, specs: list[StateTransferSpec]) -> None:
        """CPU -> GPU. Synchronous: the state layers are read early in the
        forward and one snapshot is ~19 MiB, i.e. ~1.5 ms at PCIe speed."""
        for spec in specs:
            for gpu, cpu in zip(self._gpu[spec.group_id], self._cpu[spec.group_id]):
                gpu[spec.gpu_block_id].copy_(cpu[spec.cpu_slot], non_blocking=False)

    def save(self, specs: list[StateTransferSpec]) -> None:
        """GPU -> CPU, after the forward has finished writing the pages."""
        for spec in specs:
            for gpu, cpu in zip(self._gpu[spec.group_id], self._cpu[spec.group_id]):
                cpu[spec.cpu_slot].copy_(gpu[spec.gpu_block_id], non_blocking=False)
