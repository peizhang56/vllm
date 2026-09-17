# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    CACHE_AWARE = "cache_aware"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class CacheAwareRequestQueue(RequestQueue):
    """Shortest-uncached-prefill-first queue.

    Orders waiting requests by the prefill work that is *not* already resident
    in the prefix cache, so a request needing a long cold prefill does not
    head-of-line block requests whose prefix is already cached.

    An aging term subtracts `aging_tokens_per_second` worth of effective
    uncached tokens for every second a request has waited, so a cold request
    cannot be starved indefinitely by a stream of warm arrivals.

    Aging advances every waiting request at the same rate, so it shifts all
    scores by the same amount and cannot reorder them. Ordering is therefore
    stable between mutations, and the queue keeps a lazily sorted view instead
    of rescanning on every read: `schedule()` peeks and pops in a loop, so an
    O(N) scan per read would make a scheduling step O(N*K).

    The sort is still redone whenever membership changes or the scheduler
    starts a new step, since cache-hit estimates are refreshed per step and can
    reorder requests.
    """

    def __init__(
        self,
        uncached_tokens_fn: Callable[[Request], int],
        aging_tokens_per_second: float = 0.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._uncached_tokens_fn = uncached_tokens_fn
        self._aging_tokens_per_second = aging_tokens_per_second
        self._time_fn = time_fn
        # Insertion order is retained as the tie-breaker, so requests with
        # equal scores keep FCFS semantics.
        self._requests: deque[Request] = deque()
        # Queue-entry time per request, on the same clock as `time_fn`.
        # `Request.arrival_time` is wall-clock and cannot be mixed with it.
        self._queued_ts: dict[str, float] = {}
        # Probing the prefix cache is O(prefix blocks); memoize per step.
        self._probe_cache: dict[str, int] = {}
        # Score order is valid until membership or the probe estimates change.
        self._sorted = True
        # Monotonic arrival stamp, kept so a pop can be compared against what
        # FCFS would have chosen. Whether the policy ever overrides arrival
        # order depends on the workload, so it must be observable at runtime.
        self._arrival_seq = 0
        self._seq: dict[str, int] = {}
        self.num_reordered_pops = 0
        self.num_pops = 0

    def new_scheduling_step(self) -> None:
        """Drop memoized cache-hit estimates so the next step re-probes."""
        self._probe_cache.clear()
        self._sorted = False

    def uncached_tokens(self, request: Request) -> int:
        """Memoized uncached prefill length, for callers comparing queue heads."""
        return self._uncached_tokens(request)

    def _uncached_tokens(self, request: Request) -> int:
        cached = self._probe_cache.get(request.request_id)
        if cached is None:
            cached = self._uncached_tokens_fn(request)
            self._probe_cache[request.request_id] = cached
        return cached

    def _score(self, request: Request, now: float) -> float:
        """Effective uncached prefill length; lower is scheduled sooner."""
        uncached = self._uncached_tokens(request)
        if not self._aging_tokens_per_second:
            return float(uncached)
        waited = max(0.0, now - self._queued_at(request))
        return uncached - waited * self._aging_tokens_per_second

    def _queued_at(self, request: Request) -> float:
        return self._queued_ts.get(request.request_id, self._time_fn())

    def _stamp_arrival(self, request: Request) -> None:
        if request.request_id not in self._seq:
            self._seq[request.request_id] = self._arrival_seq
            self._arrival_seq += 1

    def _ensure_sorted(self) -> None:
        """Sort by score, cheaply, if nothing has invalidated the order.

        Sorting is stable, so equal scores retain insertion order and ties keep
        FCFS semantics.
        """
        if self._sorted:
            return
        now = self._time_fn()
        self._requests = deque(
            sorted(self._requests, key=lambda r: self._score(r, now))
        )
        self._sorted = True

    def add_request(self, request: Request) -> None:
        self._requests.append(request)
        self._queued_ts.setdefault(request.request_id, self._time_fn())
        self._stamp_arrival(request)
        self._sorted = False

    def pop_request(self) -> Request:
        if not self._requests:
            raise IndexError("pop from empty queue")
        self._ensure_sorted()
        request = self._requests.popleft()
        self.num_pops += 1
        oldest = min(
            (self._seq.get(r.request_id, 0) for r in self._requests),
            default=None,
        )
        if oldest is not None and self._seq.get(request.request_id, 0) > oldest:
            self.num_reordered_pops += 1
        self._queued_ts.pop(request.request_id, None)
        self._probe_cache.pop(request.request_id, None)
        self._seq.pop(request.request_id, None)
        return request

    def peek_request(self) -> Request:
        if not self._requests:
            raise IndexError("peek from an empty queue")
        self._ensure_sorted()
        return self._requests[0]

    def prepend_request(self, request: Request) -> None:
        # Re-entering requests (e.g. after preemption) restart their aging
        # clock here; without this their wait would be measured from "now" on
        # every score, leaving them permanently at zero aging credit.
        self._requests.appendleft(request)
        self._queued_ts.setdefault(request.request_id, self._time_fn())
        self._stamp_arrival(request)
        self._sorted = False

    def prepend_requests(self, requests: RequestQueue) -> None:
        self._requests.extendleft(requests)
        now = self._time_fn()
        for request in self._requests:
            self._queued_ts.setdefault(request.request_id, now)
            self._stamp_arrival(request)
        self._sorted = False

    def _forget(self, request: Request) -> None:
        self._queued_ts.pop(request.request_id, None)
        self._probe_cache.pop(request.request_id, None)
        self._seq.pop(request.request_id, None)

    def remove_request(self, request: Request) -> None:
        self._requests.remove(request)
        self._forget(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = requests if isinstance(requests, set) else set(requests)
        self._requests = deque(r for r in self._requests if r not in to_remove)
        for request in to_remove:
            self._forget(request)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        self._ensure_sorted()
        return iter(self._requests)


def create_request_queue(
    policy: SchedulingPolicy,
    uncached_tokens_fn: Callable[[Request], int] | None = None,
    aging_tokens_per_second: float = 0.0,
) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.CACHE_AWARE:
        if uncached_tokens_fn is None:
            raise ValueError(
                "cache_aware scheduling requires an uncached_tokens_fn probe"
            )
        return CacheAwareRequestQueue(
            uncached_tokens_fn=uncached_tokens_fn,
            aging_tokens_per_second=aging_tokens_per_second,
        )
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
