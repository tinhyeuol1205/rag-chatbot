"""Bounded, cancellation-aware dynamic batching.

The batcher is deliberately independent of PyTorch.  It accepts one or more
texts per HTTP request, coalesces requests for a short window and calls a
single inference function with a length-aware batch.  The request future keeps
the original cardinality and ordering, even when items from several requests
are executed together.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from model_server.errors import (
    ModelInferenceError,
    ModelNotReadyError,
    ModelQueueFullError,
    ModelQueueTimeoutError,
    ModelRequestTooLargeError,
)

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class BatchLimits:
    """Hard limits for both one request and all outstanding requests."""

    max_items: int
    max_tokens: int
    max_bytes: int
    max_pending_requests: int
    max_pending_items: int
    max_pending_bytes: int
    max_wait_ms: int
    queue_timeout_seconds: float
    max_items_per_request: int
    max_tokens_per_item: int
    online_max_burst_batches: int = 8
    max_items_per_request_per_batch: int | None = None


@dataclass
class _PendingRequest(Generic[T, R]):
    values: list[T]
    costs: list[int]
    request_bytes: int
    priority: str
    future: asyncio.Future[list[R]]
    enqueued_at: float
    cursor: int = 0
    cancelled: bool = False
    inflight: bool = False


class DynamicBatcher(Generic[T, R]):
    """One bounded worker loop shared by all requests for a model.

    ``infer`` may be synchronous (it is run in the event loop's default
    executor) or asynchronous.  A model server normally supplies an async
    callback that acquires the shared MPS arbiter before doing synchronous
    PyTorch work.
    """

    def __init__(
        self,
        name: str,
        infer: Callable[[list[T]], list[R] | Awaitable[list[R]]],
        limits: BatchLimits,
    ):
        self.name = name
        self._infer = infer
        self._limits = limits
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._queues: dict[str, deque[_PendingRequest[T, R]]] = {
            "online": deque(),
            "batch": deque(),
        }
        self._task: asyncio.Task[None] | None = None
        self._accepting = False
        self._online_burst = 0
        self._pending_requests = 0
        self._pending_items = 0
        self._pending_bytes = 0
        self._batches = 0
        self._items_processed = 0
        self._wasted_items = 0
        self._queue_wait_ms_total = 0.0
        self._batch_tokens_total = 0
        self._batch_bytes_total = 0
        self._last_error: str | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._accepting = True
        self._task = asyncio.create_task(self._run(), name=f"model-batcher:{self.name}")

    async def stop(self) -> None:
        self._accepting = False
        async with self._lock:
            queued = [item for queue in self._queues.values() for item in queue]
            for queue in self._queues.values():
                queue.clear()
            for item in queued:
                item.cancelled = True
                self._release_counters(item)
                if not item.future.done():
                    item.future.set_exception(ModelNotReadyError("Model service is shutting down"))
        self._wake.set()
        task = self._task
        if task is not None:
            await task
        self._task = None
        async with self._lock:
            self._pending_requests = 0
            self._pending_items = 0
            self._pending_bytes = 0

    async def submit(
        self,
        values: list[T],
        costs: list[int],
        *,
        priority: str = "online",
        timeout_seconds: float | None = None,
    ) -> list[R]:
        """Queue one request and return outputs in the caller's order."""
        if priority not in self._queues:
            priority = "online"
        if not values or len(values) != len(costs):
            raise ModelRequestTooLargeError("A model request must contain at least one item")
        request_bytes = sum(len(str(value).encode("utf-8")) for value in values)
        if len(values) > self._limits.max_items_per_request:
            raise ModelRequestTooLargeError("Request item count exceeds the configured limit")
        if request_bytes > self._limits.max_bytes:
            raise ModelRequestTooLargeError("Request byte size exceeds the configured limit")
        if any(cost <= 0 or cost > self._limits.max_tokens_per_item or cost > self._limits.max_tokens for cost in costs):
            raise ModelRequestTooLargeError("Request token length exceeds the configured limit")

        async with self._lock:
            if not self._accepting:
                raise ModelNotReadyError()
            if (
                self._pending_requests >= self._limits.max_pending_requests
                or self._pending_items + len(values) > self._limits.max_pending_items
                or self._pending_bytes + request_bytes > self._limits.max_pending_bytes
            ):
                raise ModelQueueFullError()
            loop = asyncio.get_running_loop()
            pending = _PendingRequest(
                values=list(values),
                costs=list(costs),
                request_bytes=request_bytes,
                priority=priority,
                future=loop.create_future(),
                enqueued_at=loop.time(),
            )
            self._queues[priority].append(pending)
            self._pending_requests += 1
            self._pending_items += len(values)
            self._pending_bytes += request_bytes
            self._wake.set()

        timeout = self._limits.queue_timeout_seconds if timeout_seconds is None else timeout_seconds
        try:
            # Shield means a cancelled HTTP task cannot cancel the worker's
            # future while the model is still executing.  ``cancel`` below
            # removes queued work and marks in-flight work as discardable.
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout=max(timeout, 0.001))
        except asyncio.TimeoutError as exc:
            await self.cancel(pending)
            raise ModelQueueTimeoutError() from exc
        except asyncio.CancelledError:
            await self.cancel(pending)
            raise

    async def cancel(self, pending: _PendingRequest[T, R]) -> None:
        """Cancel queued work or mark an in-flight request's result discardable."""
        async with self._lock:
            if pending.cancelled:
                return
            pending.cancelled = True
            if not pending.inflight:
                for queue in self._queues.values():
                    try:
                        queue.remove(pending)
                    except ValueError:
                        continue
                self._release_counters(pending)
                if not pending.future.done():
                    pending.future.cancel()

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accepting": self._accepting,
            "pending_requests": self._pending_requests,
            "pending_items": self._pending_items,
            "pending_bytes": self._pending_bytes,
            "online_queue": len(self._queues["online"]),
            "batch_queue": len(self._queues["batch"]),
            "batches": self._batches,
            "items_processed": self._items_processed,
            "wasted_items": self._wasted_items,
            "avg_batch_items": round(self._items_processed / self._batches, 2) if self._batches else 0.0,
            "avg_batch_padded_tokens": round(self._batch_tokens_total / self._batches, 2) if self._batches else 0.0,
            "avg_batch_bytes": round(self._batch_bytes_total / self._batches, 2) if self._batches else 0.0,
            "avg_queue_wait_ms": round(self._queue_wait_ms_total / self._batches, 2) if self._batches else 0.0,
            "last_error": self._last_error,
        }

    async def _run(self) -> None:
        while self._accepting or self._has_queued_work():
            await self._wait_for_work()
            if not self._has_queued_work():
                continue
            selected, work = await self._collect_batch()
            if not work:
                continue
            try:
                if inspect.iscoroutinefunction(self._infer):
                    result = self._infer(work)
                else:
                    result = await asyncio.to_thread(self._infer, work)
                if inspect.isawaitable(result):
                    outputs = await result
                else:
                    outputs = result
                if len(outputs) != len(work):
                    raise ModelInferenceError(
                        f"{self.name} inference returned {len(outputs)} outputs for {len(work)} inputs"
                    )
            except asyncio.CancelledError:
                await self._fail_selected(selected, ModelNotReadyError("Model service stopped"))
                raise
            except Exception as exc:  # noqa: BLE001 - convert model errors at one boundary
                self._last_error = type(exc).__name__
                error = exc if isinstance(exc, ModelInferenceError) else ModelInferenceError()
                await self._fail_selected(selected, error)
                continue
            await self._scatter(selected, outputs)

    async def _wait_for_work(self) -> None:
        if self._has_queued_work():
            return
        self._wake.clear()
        if self._accepting:
            await self._wake.wait()

    def _has_queued_work(self) -> bool:
        return bool(self._queues["online"] or self._queues["batch"])

    async def _collect_batch(
        self,
    ) -> tuple[list[tuple[_PendingRequest[T, R], list[tuple[int, T, int]]]], list[T]]:
        selected: list[tuple[_PendingRequest[T, R], list[tuple[int, T]]]] = []
        work: list[T] = []
        item_count = 0
        max_cost = 0
        batch_bytes = 0

        async with self._lock:
            first = self._pop_next_request()
            if first is not None:
                taken = self._take(first, self._per_request_batch_limit(), self._limits.max_tokens, 0, 0)
                if taken:
                    selected.append((first, taken))
                    work.extend(value for _, value in taken)
                    item_count = len(taken)
                    max_cost = max(first.costs[index] for index, _ in taken)
                    batch_bytes = sum(len(str(value).encode("utf-8")) for _, value in taken)
                else:
                    self._queues[first.priority].appendleft(first)

        # Coalescing happens outside the lock so request admission stays fast.
        if selected and self._limits.max_wait_ms > 0:
            await asyncio.sleep(self._limits.max_wait_ms / 1000)

        async with self._lock:
            while item_count < self._limits.max_items:
                candidate = self._pop_next_request()
                if candidate is None:
                    break
                remaining_items = min(
                    self._limits.max_items - item_count,
                    self._per_request_batch_limit(),
                )
                taken = self._take(candidate, remaining_items, self._limits.max_tokens, item_count, max_cost)
                if not taken:
                    self._queues[candidate.priority].appendleft(candidate)
                    break
                candidate_bytes = sum(len(str(value).encode("utf-8")) for _, value in taken)
                if batch_bytes + candidate_bytes > self._limits.max_bytes:
                    candidate.cursor -= len(taken)
                    candidate.inflight = False
                    self._queues[candidate.priority].appendleft(candidate)
                    break
                selected.append((candidate, taken))
                work.extend(value for _, value in taken)
                item_count += len(taken)
                batch_bytes += candidate_bytes
                max_cost = max(max_cost, max(candidate.costs[index] for index, _ in taken))
        # Bucket by token length to reduce padding.  Keep an explicit output
        # position for every item so duplicate texts and multi-item requests
        # still receive the right result after sorting.
        flattened: list[tuple[int, int, int, T]] = []
        for request_number, (pending, taken) in enumerate(selected):
            flattened.extend((pending.costs[index], request_number, index, value) for index, value in taken)
        flattened.sort(key=lambda row: row[0])
        work = [value for _, _, _, value in flattened]
        positioned: dict[int, list[tuple[int, T, int]]] = {n: [] for n in range(len(selected))}
        for output_position, (_, request_number, index, value) in enumerate(flattened):
            positioned[request_number].append((index, value, output_position))
        return [
            (pending, positioned[request_number])
            for request_number, (pending, _) in enumerate(selected)
        ], work

    def _pop_next_request(self) -> _PendingRequest[T, R] | None:
        online = self._queues["online"]
        batch = self._queues["batch"]
        if online and (not batch or self._online_burst < max(self._limits.online_max_burst_batches, 1)):
            self._online_burst += 1
            return online.popleft()
        if batch:
            self._online_burst = 0
            return batch.popleft()
        if online:
            self._online_burst += 1
            return online.popleft()
        return None

    def _take(
        self,
        pending: _PendingRequest[T, R],
        item_capacity: int,
        token_capacity: int,
        current_count: int,
        current_max_cost: int,
    ) -> list[tuple[int, T]]:
        if pending.cancelled or pending.cursor >= len(pending.values):
            return []
        start = pending.cursor
        taken: list[tuple[int, T]] = []
        max_cost = 0
        for index in range(start, min(len(pending.values), start + item_capacity)):
            cost = pending.costs[index]
            next_max = max(max_cost, cost)
            next_count = len(taken) + 1
            projected = max(current_max_cost, next_max) * (current_count + next_count)
            if taken and projected > token_capacity:
                break
            if not taken and projected > token_capacity:
                return []
            taken.append((index, pending.values[index]))
            max_cost = next_max
        pending.cursor += len(taken)
        pending.inflight = bool(taken)
        return taken

    async def _scatter(
        self,
        selected: list[tuple[_PendingRequest[T, R], list[tuple[int, T, int]]]],
        outputs: list[R],
    ) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            for pending, taken in selected:
                self._queue_wait_ms_total += max(0.0, (now - pending.enqueued_at) * 1000)
                if pending.cancelled:
                    self._release_counters(pending)
                    pending.inflight = False
                    self._wasted_items += len(taken)
                    continue
                if not hasattr(pending, "results"):
                    pending.results = [None] * len(pending.values)  # type: ignore[attr-defined]
                for index, _, output_position in taken:
                    pending.results[index] = outputs[output_position]  # type: ignore[attr-defined]
                pending.inflight = False
                if pending.cursor < len(pending.values) and self._accepting:
                    pending.enqueued_at = now
                    self._queues[pending.priority].append(pending)
                else:
                    self._release_counters(pending)
                    if not pending.future.done():
                        if pending.cursor < len(pending.values):
                            pending.future.set_exception(ModelNotReadyError("Model service is shutting down"))
                        else:
                            pending.future.set_result(list(pending.results))  # type: ignore[attr-defined]
            self._wake.set()
        self._batches += 1
        self._items_processed += len(outputs)
        if selected:
            max_cost = max(
                pending.costs[index]
                for pending, taken in selected
                for index, _, _ in taken
            )
            self._batch_tokens_total += max_cost * len(outputs)
            self._batch_bytes_total += sum(
                len(str(value).encode("utf-8"))
                for _, taken in selected
                for _, value, _ in taken
            )

    def _per_request_batch_limit(self) -> int:
        return max(
            1,
            min(
                self._limits.max_items,
                self._limits.max_items_per_request_per_batch or self._limits.max_items,
            ),
        )

    async def _fail_selected(
        self,
        selected: list[tuple[_PendingRequest[T, R], list[tuple[int, T, int]]]],
        error: Exception,
    ) -> None:
        async with self._lock:
            for pending, _ in selected:
                pending.inflight = False
                self._release_counters(pending)
                if not pending.future.done():
                    pending.future.set_exception(error)

    def _release_counters(self, pending: _PendingRequest[T, R]) -> None:
        if getattr(pending, "released", False):
            return
        pending.released = True  # type: ignore[attr-defined]
        self._pending_requests = max(0, self._pending_requests - 1)
        self._pending_items = max(0, self._pending_items - len(pending.values))
        self._pending_bytes = max(0, self._pending_bytes - pending.request_bytes)
