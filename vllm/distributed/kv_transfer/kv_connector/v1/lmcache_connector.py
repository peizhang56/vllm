# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_hybrid_state import (
    DEFAULT_SNAPSHOT_INTERVAL_TOKENS,
    DEFAULT_STATE_CPU_GB,
    HybridStateScheduler,
    HybridStateWorker,
    LMCacheHybridMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheKVEvents(KVConnectorKVEvents):
    """
    Concrete implementation of KVConnectorKVEvents using KVEventAggregator.
    """

    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "LMCacheKVEvents":
        """
        Aggregate KV events and retain only common events.
        """
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<LMCacheKVEvents events={self.get_all_events()}>"


class LMCacheConnectorV1(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        LMCache requires PIECEWISE CUDA graph mode when layerwise
        operations are enabled. The wait_for_layer_load and save_kv_layer
        methods perform actual async synchronization that cannot be
        captured in CUDA graphs.
        """
        return extra_config.get("use_layerwise", False)

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        assert vllm_config.kv_transfer_config is not None
        use_native = vllm_config.kv_transfer_config.get_from_extra_config(
            "use_native", False
        )
        if use_native:
            logger.info("Initializing native LMCache connector")
            # lazy import
            from vllm.distributed.kv_transfer.kv_connector.v1 import lmcache_integration

            _adapter = lmcache_integration.vllm_v1_adapter

            cls = _adapter.LMCacheConnectorV1Impl
        else:
            logger.info("Initializing latest dev LMCache connector")
            # lazy import
            from lmcache.integration.vllm.vllm_v1_adapter import (
                LMCacheConnectorV1Impl as LMCacheConnectorLatestImpl,
            )

            cls = LMCacheConnectorLatestImpl

        # Hybrid (Mamba/linear + attention) bookkeeping. LMCache's adapter is
        # token-chunk oriented and only understands paged attention KV, so on a
        # hybrid model it owns the attention group and something else has to
        # account for the recurrent state -- see `_state_group_ids` users.
        #
        # Must precede the adapter's construction: it reads
        # `attn_kv_cache_group_id` off us in its own `__init__`, and would
        # silently fall back to group 0 if we had not filled it in yet.
        self._attn_group_id, self._state_group_ids = self._split_kv_cache_groups(
            kv_cache_config
        )
        self._state_layer_names: frozenset[str] = frozenset(
            name
            for i in self._state_group_ids
            for name in kv_cache_config.kv_cache_groups[i].layer_names
        )
        self._allow_unsafe_hybrid_state = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "unsafe_hybrid_state", False
            )
        )
        # The state sidecar: a CPU tier for the recurrent-state groups plus the
        # joint-boundary gate that keeps LMCache's attention prefix and that
        # state in agreement. On by default for hybrid models -- without it the
        # only safe behaviour is to refuse loads entirely.
        self._state_scheduler: HybridStateScheduler | None = None
        self._state_worker: HybridStateWorker | None = None
        self._state_meta: LMCacheHybridMetadata | None = None
        enable_state_tier = bool(self._state_group_ids) and bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "hybrid_state_tier", True
            )
        )
        if enable_state_tier:
            cpu_gb = float(
                vllm_config.kv_transfer_config.get_from_extra_config(
                    "hybrid_state_cpu_gb", DEFAULT_STATE_CPU_GB
                )
            )
            interval = int(
                vllm_config.kv_transfer_config.get_from_extra_config(
                    "hybrid_state_interval_tokens",
                    DEFAULT_SNAPSHOT_INTERVAL_TOKENS,
                )
            )
            scheduler = HybridStateScheduler(
                kv_cache_config, self._state_group_ids, cpu_gb, interval
            )
            if role == KVConnectorRole.SCHEDULER:
                self._state_scheduler = scheduler
            else:
                # Slot ids are assigned by the scheduler and interpreted here,
                # so the worker only needs the sizing, which is derived from
                # the same config on every rank.
                self._state_worker = HybridStateWorker(
                    kv_cache_config, self._state_group_ids, scheduler.num_slots
                )

        self._warned_hybrid_state = False
        if self._state_group_ids:
            logger.info(
                "LMCache: hybrid KV cache layout detected -- attention group "
                "%d, recurrent-state group(s) %s. Loads are %s.",
                self._attn_group_id,
                list(self._state_group_ids),
                "ENABLED via the state tier"
                if enable_state_tier
                else "ENABLED (unsafe_hybrid_state)"
                if self._allow_unsafe_hybrid_state
                else "disabled -- no state transfer",
            )

        self._lmcache_engine = cls(vllm_config, role, self)

        self._kv_cache_events: LMCacheKVEvents | None = None

    @property
    def attn_kv_cache_group_id(self) -> int:
        """Index of the KV cache group LMCache owns.

        Read by the LMCache adapter, which receives one block-id list per
        group and must pick the one that indexes the tensors it registered.
        Group 0 on a dense model; on a hybrid model the Mamba groups come
        first, so it is not.
        """
        return self._attn_group_id

    @staticmethod
    def _split_kv_cache_groups(
        kv_cache_config: "KVCacheConfig",
    ) -> tuple[int, tuple[int, ...]]:
        """Return (attention group index, recurrent-state group indices).

        Mirrors ``GPUModelRunner._get_attention_kv_cache_gid``: the attention
        group is the first non-Mamba ``AttentionSpec`` group. On a dense model
        this is group 0 and the state tuple is empty, which is the pre-HMA
        behaviour.
        """
        attn_group_id: int | None = None
        state_group_ids: list[int] = []
        for i, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                state_group_ids.append(i)
            elif isinstance(spec, AttentionSpec) and attn_group_id is None:
                attn_group_id = i
        return attn_group_id if attn_group_id is not None else 0, tuple(state_group_ids)

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        if self._state_layer_names:
            # Mamba/linear layers are handed to connectors as an opaque
            # ``[num_blocks, 1, 1, page_size_bytes]`` int8 page view (see the
            # MambaSpec branch of ``GPUModelRunner._allocate_kv_cache_tensors``),
            # deliberately so that a connector "can register it without
            # special-casing Mamba". LMCache is not such a connector: its format
            # discovery only knows the 3-D MLA and 5-D MHA attention layouts and
            # raises on anything else --
            #
            #   ValueError: currently unsupported kv_caches format with list
            #   depth 1 and tensor dimension 4
            #       (lmcache/v1/gpu_connector/utils.py, in
            #        normalize_kv_and_discover_format)
            #
            # Hand it only the layers it can represent. Nothing is lost that it
            # could have used: without a state format there is no correct way to
            # transfer these pages, which is the same reason loads are gated in
            # `get_num_new_matched_tokens`.
            if self._state_worker is not None:
                # Not lost after all: the sidecar mirrors these pages itself.
                self._state_worker.register_kv_caches(kv_caches)
            kv_caches = {
                name: tensor
                for name, tensor in kv_caches.items()
                if name not in self._state_layer_names
            }
            logger.info(
                "LMCache: registering %d attention layer(s), skipping %d "
                "recurrent-state layer(s) that LMCache has no format for.",
                len(kv_caches),
                len(self._state_layer_names),
            )

        if hasattr(self._lmcache_engine, "register_kv_caches"):
            self._lmcache_engine.register_kv_caches(kv_caches)
        else:
            logger.warning(
                "LMCache engine does not support register_kv_caches, "
                "please check and use the latest version"
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        if self._state_worker is not None and self._state_meta is not None:
            # Restore the recurrent state before the attention KV it belongs
            # with. Both must land before the forward reads either.
            self._state_worker.load(self._state_meta.loads)
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if layer_name in self._state_layer_names:
            # Never registered -- see `register_kv_caches`.
            return
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if layer_name in self._state_layer_names:
            # Never registered -- see `register_kv_caches`.
            return
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()
        if self._state_worker is not None and self._state_meta is not None:
            # After the forward: the pages now hold the state for the prefix
            # the scheduler hashed them under.
            self._state_worker.save(self._state_meta.saves)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Split the sidecar's transfers off before LMCache sees the metadata.

        LMCache's adapter reads the bound metadata back through
        ``self._parent._get_connector_metadata()``, so it must be handed
        exactly the object it built.
        """
        if isinstance(connector_metadata, LMCacheHybridMetadata):
            self._state_meta = connector_metadata
            connector_metadata = connector_metadata.inner
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self._state_meta = None
        super().clear_connector_metadata()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        method = getattr(self._lmcache_engine, "get_block_ids_with_load_errors", None)
        if callable(method):
            return method()

        # Fallback for older versions that don't support this method
        return set()

    def get_kv_connector_kv_cache_events(self) -> LMCacheKVEvents | None:
        """
        Get the KV connector kv cache events collected during the last interval.
        """

        events = self._lmcache_engine.get_kv_events()  # type: ignore [attr-defined]
        if not events:
            return None

        blocks: list[BlockStored] = [
            BlockStored(
                block_hashes=e.block_hashes,
                parent_block_hash=e.parent_block_hash,
                token_ids=e.token_ids,
                lora_id=e.lora_id,
                block_size=e.block_size,
                medium=e.medium,
                lora_name=getattr(e, "lora_name", None),
            )
            for e in events
        ]

        lmcache_kv_events = LMCacheKVEvents(num_workers=1)
        lmcache_kv_events.add_events(blocks)
        return lmcache_kv_events

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if (
            self._state_group_ids
            and self._state_scheduler is None
            and not self._allow_unsafe_hybrid_state
        ):
            # Hybrid model (Mamba/linear + attention) and LMCache carries only
            # paged attention KV.
            #
            # Claiming external tokens here would restore an MLA/attention
            # prefix on top of whatever recurrent state happens to be resident.
            # vLLM only reconciles the per-group boundary when the connector
            # returns 0 (scheduler.py, `if hit_diverged and
            # num_external_computed_tokens == 0`), so any positive claim past
            # the local Mamba hit runs the KDA/SSM layers from a state that
            # never saw those tokens. That is silent wrong output -- no crash,
            # no failed-load metric, just a corrupt answer -- which is exactly
            # the failure already observed on the ATOM-native offload path.
            #
            # Refuse instead. Saves still happen, so the store side can be
            # measured, and the refusal is visible rather than silent. Set
            # kv_connector_extra_config={"unsafe_hybrid_state": true} to bypass
            # this for A/B experiments; it is not safe for serving.
            if not self._warned_hybrid_state:
                logger.warning(
                    "LMCache: refusing KV loads for this hybrid model. Groups "
                    "%s hold per-request recurrent state that LMCache cannot "
                    "restore, and reloading attention KV without it produces "
                    "silent wrong output. Saves are unaffected. Override with "
                    'kv_connector_extra_config={"unsafe_hybrid_state": true} '
                    "for experiments only.",
                    list(self._state_group_ids),
                )
                self._warned_hybrid_state = True
            return 0, False

        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        ), False

    def clamp_external_hit(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_hit_tokens: int,
    ) -> int:
        """Joint-boundary gate, called by LMCache's adapter.

        LMCache asserts that the scheduler accepts exactly the hit length it
        reported, so the clamp has to happen inside its own accounting rather
        than on the value this connector returns.
        """
        if self._state_scheduler is None:
            return num_hit_tokens
        return self._state_scheduler.clamp_external_hit(
            request, num_computed_tokens, num_hit_tokens
        )

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        """The sidecar walks the pool to find which state blocks are hashed
        (and therefore reusable) -- the same test vLLM's prefix cache uses."""
        if self._state_scheduler is not None:
            self._state_scheduler.bind_gpu_block_pool(gpu_block_pool)
        inner = getattr(self._lmcache_engine, "bind_gpu_block_pool", None)
        if inner is not None:
            inner(gpu_block_pool)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)
        if self._state_scheduler is not None:
            self._state_scheduler.record_load(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        inner = self._lmcache_engine.build_connector_meta(scheduler_output)
        if self._state_scheduler is None:
            return inner
        return self._state_scheduler.build_meta(inner, scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, LMCacheKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(
                kv_cache_events.get_number_of_workers()
            )
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        if self._state_scheduler is not None:
            self._state_scheduler.request_finished(request.request_id)
        return self._lmcache_engine.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA variant of ``request_finished``: one block-id list per KV cache
        group instead of a single flat list.

        Implementing this (via ``SupportsHMA``) is what keeps the hybrid KV
        cache manager enabled. Without it, ``supports_hma_config()`` is False,
        vLLM disables HMA and then tries to collapse Kimi-K3's MLA + Mamba
        specs into one type, which fails outright:
        ``Failed to promote local KV cache specs to one unified type``.

        LMCache stores paged attention KV, so it is handed the attention
        group's blocks. The recurrent-state groups are carried by the state
        sidecar (``lmcache_hybrid_state``), which also clamps every external
        hit to a boundary where both halves can be restored.
        """
        if self._state_scheduler is not None:
            self._state_scheduler.request_finished(request.request_id)
        return self._lmcache_engine.request_finished(
            request, block_ids[self._attn_group_id]
        )

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None
