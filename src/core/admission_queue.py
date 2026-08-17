"""Distributed admission queue and LLM call reservations.

The API process does not start RAG work when the production queue is enabled.  A
Redis Streams consumer claims a job only after the bounded outstanding-job
counter admits it; the worker then reserves a paced set of LLM call slots before
calling the retriever.  This keeps the provider quota shared across replicas.

The inline implementation is intentionally a development adapter.  Production
must set ``RAG_EXECUTION_MODE=redis_worker`` and fail closed when Redis is not
available.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from core import get_logger
from core.config import settings
from core.errors import JobFailedError, JobTimeoutError, QueueFullError, QueueUnavailableError

logger = get_logger(__name__)

_current_reservation: contextvars.ContextVar[LLMReservation | None] = contextvars.ContextVar(
    "rag_llm_reservation", default=None
)


def llm_call_cost(history: list[tuple[str, str]] | None = None) -> int:
    """Return the conservative provider-call reservation for one RAG request."""
    # Multi-Query, HyDE and final answer are the normal calls.  Condense is
    # added only when history is present.  A reservation is deliberately an
    # upper bound; unused slots are never reused early in a way that can violate
    # the provider's rolling quota.
    return 3 + (1 if history else 0)


@dataclass(frozen=True)
class AdmissionJob:
    job_id: str
    query: str
    history: list[tuple[str, str]]
    llm_cost: int
    idempotency_key: str


class LLMReservation:
    """Paced reservation whose calls are consumed at the provider boundary."""

    def __init__(
        self,
        limiter: RateScheduler,
        reservation_id: str,
        cost: int,
        first_at: float,
        cancel_checker: Callable[[], bool] | None = None,
    ):
        self._limiter = limiter
        self.reservation_id = reservation_id
        self.cost = cost
        self.first_at = first_at
        self._cancel_checker = cancel_checker

    def wait_until_start(self) -> None:
        delay = self.first_at - time.time()
        if delay > 0:
            time.sleep(delay)

    def consume(self) -> None:
        """Consume one slot, blocking only until its globally paced timestamp."""
        if self._cancel_checker is not None and self._cancel_checker():
            raise JobTimeoutError("RAG job cancellation requested")
        self._limiter.consume(self.reservation_id)


@contextlib.contextmanager
def reservation_context(reservation: LLMReservation | None) -> Iterator[None]:
    token = _current_reservation.set(reservation)
    try:
        yield
    finally:
        _current_reservation.reset(token)


def current_reservation() -> LLMReservation | None:
    return _current_reservation.get()


def consume_llm_permit() -> None:
    """Consume one provider slot at the exact network-call boundary."""
    reservation = current_reservation()
    if reservation is None:
        if settings.RAG_EXECUTION_MODE == "redis_worker":
            raise QueueUnavailableError("LLM call attempted without an admission reservation")
        return
    reservation.consume()


class RateScheduler:
    """Redis-backed paced call scheduler with an in-memory test/dev adapter."""

    _RESERVE_LUA = """
    local now_parts = redis.call('TIME')
    local now = tonumber(now_parts[1]) * 1000 + math.floor(tonumber(now_parts[2]) / 1000)
    local next_at = tonumber(redis.call('GET', KEYS[1]) or '0')
    if next_at < now then next_at = now end
    local cost = tonumber(ARGV[1])
    local interval = tonumber(ARGV[2])
    local start_at = next_at
    local finish_at = next_at + (cost * interval)
    redis.call('SET', KEYS[1], tostring(finish_at), 'EX', ARGV[3])
    redis.call('HSET', KEYS[2], 'next_at', tostring(start_at), 'remaining', tostring(cost), 'interval', tostring(interval))
    redis.call('EXPIRE', KEYS[2], ARGV[3])
    return {tostring(start_at), tostring(interval)}
    """

    _CONSUME_LUA = """
    local remaining = tonumber(redis.call('HGET', KEYS[1], 'remaining') or '0')
    if remaining <= 0 then return {'empty', '0'} end
    local now_parts = redis.call('TIME')
    local now = tonumber(now_parts[1]) * 1000 + math.floor(tonumber(now_parts[2]) / 1000)
    local next_at = tonumber(redis.call('HGET', KEYS[1], 'next_at') or '0')
    if now < next_at then return {'wait', tostring(next_at)} end
    local interval = tonumber(redis.call('HGET', KEYS[1], 'interval') or '0')
    redis.call('HINCRBY', KEYS[1], 'remaining', -1)
    redis.call('HSET', KEYS[1], 'next_at', tostring(next_at + interval))
    return {'ok', tostring(next_at)}
    """

    def __init__(self, client=None):
        self._client = client
        self._lock = threading.Lock()
        self._next_at = 0.0
        self._reservations: dict[str, dict[str, float]] = {}
        self._reserve_script = client.register_script(self._RESERVE_LUA) if client is not None else None
        self._consume_script = client.register_script(self._CONSUME_LUA) if client is not None else None

    @property
    def interval_seconds(self) -> float:
        # Add a millisecond guard so calls at exact provider-window boundaries
        # do not create a 15/60s inclusive-window violation.
        return (settings.LLM_RATE_LIMIT_WINDOW_SECONDS + 0.001) / max(
            settings.LLM_RATE_LIMIT_CALLS,
            1,
        )

    def reserve(
        self,
        cost: int,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> LLMReservation:
        if cost <= 0:
            raise ValueError("LLM reservation cost must be positive")
        reservation_id = uuid.uuid4().hex
        ttl = max(int(settings.LLM_RESERVATION_TTL_SECONDS), 60)
        if self._client is not None:
            try:
                raw = self._reserve_script(
                    keys=[settings.REDIS_RATE_SCHEDULE_KEY, f"{settings.REDIS_RATE_RESERVATION_PREFIX}{reservation_id}"],
                    args=[cost, max(1, math.ceil(self.interval_seconds * 1000)), ttl],
                )
                first_at = float(raw[0]) / 1000.0
                return LLMReservation(self, reservation_id, cost, first_at, cancel_checker)
            except Exception as exc:  # noqa: BLE001 - convert every Redis/SDK error to a safe queue error
                self._raise_redis(exc)
        with self._lock:
            first_at = max(self._next_at, time.time())
            self._next_at = first_at + cost * self.interval_seconds
            self._reservations[reservation_id] = {
                "next_at": first_at,
                "remaining": float(cost),
                "interval": self.interval_seconds,
            }
        return LLMReservation(self, reservation_id, cost, first_at, cancel_checker)

    def consume(self, reservation_id: str) -> None:
        key = f"{settings.REDIS_RATE_RESERVATION_PREFIX}{reservation_id}"
        while True:
            if self._client is not None:
                try:
                    raw = self._consume_script(keys=[key], args=[])
                    state = raw[0].decode() if isinstance(raw[0], bytes) else str(raw[0])
                    value = float(raw[1])
                    if state == "ok":
                        return
                    if state == "empty":
                        raise RuntimeError("LLM reservation has no remaining call slots")
                    time.sleep(max(0.0, value / 1000.0 - time.time()))
                    continue
                except Exception as exc:
                    if isinstance(exc, RuntimeError):
                        raise
                    self._raise_redis(exc)
            with self._lock:
                reservation = self._reservations.get(reservation_id)
                if reservation is None or reservation["remaining"] <= 0:
                    raise RuntimeError("LLM reservation has no remaining call slots")
                now = time.time()
                next_at = reservation["next_at"]
                if now >= next_at:
                    reservation["remaining"] -= 1
                    reservation["next_at"] += reservation["interval"]
                    return
                delay = next_at - now
            time.sleep(delay)

    @staticmethod
    def _raise_redis(exc: Exception) -> None:
        raise QueueUnavailableError("Redis rate scheduler unavailable") from exc


class InlineAdmissionQueue:
    """Development adapter; runs a job immediately after a local reservation."""

    def __init__(self):
        self.rate = RateScheduler()

    def execute(
        self,
        handler: Callable[[str, list[tuple[str, str]]], Any],
        query: str,
        history: list[tuple[str, str]] | None = None,
    ) -> Any:
        reservation = self.rate.reserve(llm_call_cost(history))
        # Inline is a development adapter.  Do not sleep before retrieval in
        # local tests; the first LLM boundary still consumes the reservation
        # and subsequent calls are paced.  The production Redis worker waits
        # before invoking the handler, which is the hard admission contract.
        with reservation_context(reservation):
            return handler(query, history or [])


class RedisAdmissionQueue:
    """Bounded Redis Streams queue and worker-side result store."""

    _ENQUEUE_LUA = """
    local existing = redis.call('GET', KEYS[4])
    if existing then return {'existing', existing} end
    local count = tonumber(redis.call('GET', KEYS[2]) or '0')
    if count >= tonumber(ARGV[1]) then return {'full', tostring(count)} end
    local job_id = ARGV[2]
    local stream_id = redis.call('XADD', KEYS[1], '*', 'job_id', job_id)
    redis.call('INCR', KEYS[2])
    redis.call('HSET', KEYS[3],
      'job_id', job_id, 'query', ARGV[3], 'history', ARGV[4],
      'llm_cost', ARGV[5], 'state', 'queued', 'stream_id', stream_id)
    redis.call('EXPIRE', KEYS[3], ARGV[6])
    redis.call('SET', KEYS[4], job_id, 'EX', ARGV[6])
    return {'created', job_id}
    """

    _TERMINAL_LUA = """
    local state = redis.call('HGET', KEYS[3], 'state')
    if state == 'completed' or state == 'failed' or state == 'cancelled' then return 'already' end
    redis.call('HSET', KEYS[3], 'state', ARGV[1], 'result', ARGV[2], 'error_code', ARGV[3], 'error_message', ARGV[4])
    redis.call('DECR', KEYS[2])
    if ARGV[5] ~= '' then redis.call('XACK', KEYS[1], ARGV[6], ARGV[5]); redis.call('XDEL', KEYS[1], ARGV[5]) end
    redis.call('EXPIRE', KEYS[3], ARGV[7])
    return 'ok'
    """

    _CANCEL_LUA = """
    local state = redis.call('HGET', KEYS[3], 'state')
    if state == 'queued' or state == 'admitted' then
      redis.call('HSET', KEYS[3], 'state', 'cancelled')
      redis.call('DECR', KEYS[2])
      local stream_id = redis.call('HGET', KEYS[3], 'stream_id')
      if stream_id then redis.call('XDEL', KEYS[1], stream_id) end
      return 'cancelled'
    elseif state == 'running' then
      redis.call('HSET', KEYS[3], 'cancel_requested', '1')
      return 'requested'
    end
    return state or 'missing'
    """

    def __init__(self, client=None):
        if client is None:
            try:
                import redis

                client = redis.Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT_SECONDS,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                raise QueueUnavailableError("Redis client is not available") from exc
        self.client = client
        self.stream = settings.REDIS_QUEUE_STREAM
        self.group = settings.REDIS_QUEUE_GROUP
        self.rate = RateScheduler(client)
        self._enqueue_script = client.register_script(self._ENQUEUE_LUA)
        self._terminal_script = client.register_script(self._TERMINAL_LUA)
        self._cancel_script = client.register_script(self._CANCEL_LUA)
        self._ensure_group()

    def _ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001 - BUSYGROUP and transport errors need explicit handling
            if "BUSYGROUP" not in str(exc):
                self._raise_redis(exc)

    @staticmethod
    def _job_key(job_id: str) -> str:
        return f"{settings.REDIS_JOB_PREFIX}{job_id}"

    def enqueue(
        self,
        query: str,
        history: list[tuple[str, str]] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> AdmissionJob:
        job_id = uuid.uuid4().hex
        idem = idempotency_key or f"anonymous:{job_id}"
        try:
            raw = self._enqueue_script(
                keys=[self.stream, settings.REDIS_OUTSTANDING_KEY, self._job_key(job_id), f"{settings.REDIS_IDEMPOTENCY_PREFIX}{idem}"],
                args=[
                    settings.RAG_QUEUE_MAX_OUTSTANDING,
                    job_id,
                    query,
                    json.dumps(history or [], ensure_ascii=False),
                    llm_call_cost(history),
                    settings.RAG_JOB_TTL_SECONDS,
                ],
            )
        except Exception as exc:  # noqa: BLE001 - convert Redis script failures to a safe queue error
            self._raise_redis(exc)
        state = raw[0].decode() if isinstance(raw[0], bytes) else str(raw[0])
        returned_id = raw[1].decode() if isinstance(raw[1], bytes) else str(raw[1])
        if state == "full":
            raise QueueFullError("Redis admission queue is full")
        if state == "existing":
            job_id = returned_id
            data = self.client.hgetall(self._job_key(job_id))
            return self._job_from_hash(data, idem)
        return AdmissionJob(job_id, query, history or [], llm_call_cost(history), idem)

    def wait_result(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        wait_seconds = settings.RAG_JOB_MAX_WAIT_SECONDS if timeout is None else timeout
        end = time.monotonic() + wait_seconds
        key = self._job_key(job_id)
        while time.monotonic() < end:
            try:
                data = self.client.hgetall(key)
            except Exception as exc:  # noqa: BLE001 - convert Redis read failures to a safe queue error
                self._raise_redis(exc)
            state = self._text(self._hash_value(data, "state"))
            if state == "completed":
                return json.loads(self._text(self._hash_value(data, "result")) or "{}")
            if state == "failed":
                raise JobFailedError(self._text(self._hash_value(data, "error_message")) or "RAG job failed")
            if state == "cancelled":
                raise JobTimeoutError("RAG job was cancelled")
            time.sleep(settings.REDIS_RESULT_POLL_SECONDS)
        raise JobTimeoutError("RAG job result wait timed out")

    def run_once(
        self,
        handler: Callable[[str, list[tuple[str, str]]], Any],
        *,
        consumer: str | None = None,
        block_ms: int = 1000,
    ) -> str | None:
        consumer = consumer or f"worker-{uuid.uuid4().hex[:8]}"
        messages = self._claim_stale(consumer)
        if not messages:
            try:
                rows = self.client.xreadgroup(
                    self.group,
                    consumer,
                    {self.stream: ">"},
                    count=1,
                    block=block_ms,
                )
            except Exception as exc:  # noqa: BLE001 - convert Redis read failures to a safe queue error
                self._raise_redis(exc)
            if rows:
                _, messages = rows[0]
        if not messages:
            return None
        message_id, fields = messages[0]
        job_id = fields.get("job_id") or fields.get(b"job_id")
        if not job_id:
            self.client.xack(self.stream, self.group, message_id)
            return None
        data = self.client.hgetall(self._job_key(job_id))
        if not data:
            self.client.xack(self.stream, self.group, message_id)
            return job_id
        if self._text(self._hash_value(data, "state")) in {"cancelled", "completed", "failed"}:
            self.client.xack(self.stream, self.group, message_id)
            return job_id
        try:
            self.client.hset(self._job_key(job_id), mapping={"state": "admitted", "stream_id": message_id})
            # A disconnect can cancel a queued job while the message is being
            # claimed.  Re-check before reserving quota so a cancelled job is
            # not charged and never touches the RAG pipeline.
            if self._text(self.client.hget(self._job_key(job_id), "state")) == "cancelled":
                self.client.xack(self.stream, self.group, message_id)
                return job_id
            reservation = self.rate.reserve(
                int(self._text(self._hash_value(data, "llm_cost", 3))),
                cancel_checker=lambda: self._text(
                    self.client.hget(self._job_key(job_id), "cancel_requested")
                ) == "1",
            )
            reservation.wait_until_start()
            if self._text(self.client.hget(self._job_key(job_id), "state")) == "cancelled":
                self.client.xack(self.stream, self.group, message_id)
                return job_id
            self.client.hset(self._job_key(job_id), "state", "running")
            history = [tuple(item) for item in json.loads(self._text(self._hash_value(data, "history")) or "[]")]
            with reservation_context(reservation):
                result = handler(self._text(self._hash_value(data, "query", "")), history)
            self._terminal(job_id, message_id, "completed", _json_safe(result), "", "")
        except JobTimeoutError:
            self._terminal(job_id, message_id, "cancelled", {}, "job_timeout", "RAG job cancelled")
        except Exception as exc:
            logger.exception("RAG worker job failed", job_id=job_id, error_type=type(exc).__name__)
            self._terminal(job_id, message_id, "failed", {}, getattr(exc, "error_code", "job_failed"), "RAG job failed")
        return job_id

    def _claim_stale(self, consumer: str) -> list:
        """Reclaim a worker message left pending after a process crash."""
        try:
            result = self.client.xautoclaim(
                self.stream,
                self.group,
                consumer,
                min_idle_time=int(settings.RAG_WORKER_LEASE_SECONDS * 1000),
                start_id="0-0",
                count=1,
            )
        except AttributeError:
            return []
        except Exception as exc:  # noqa: BLE001 - old Redis servers may lack XAUTOCLAIM
            logger.warning("Redis stale-message reclaim failed", error_type=type(exc).__name__)
            return []
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            return result[1] or []
        return []

    def _terminal(self, job_id: str, message_id: str, state: str, result: dict[str, Any], code: str, message: str) -> None:
        try:
            self._terminal_script(
                keys=[self.stream, settings.REDIS_OUTSTANDING_KEY, self._job_key(job_id)],
                args=[state, json.dumps(result, ensure_ascii=False), code, message, message_id, self.group, settings.RAG_JOB_TTL_SECONDS],
            )
        except Exception as exc:  # noqa: BLE001 - cancellation update must fail closed
            self._raise_redis(exc)

    def cancel(self, job_id: str) -> str:
        try:
            return self._text(
                self._cancel_script(
                    keys=[self.stream, settings.REDIS_OUTSTANDING_KEY, self._job_key(job_id)],
                    args=[],
                )
            )
        except Exception as exc:  # noqa: BLE001 - cancellation update must fail closed
            self._raise_redis(exc)

    @staticmethod
    def _job_from_hash(data: dict[str, str], idem: str) -> AdmissionJob:
        return AdmissionJob(
            job_id=RedisAdmissionQueue._text(RedisAdmissionQueue._hash_value(data, "job_id", "")),
            query=RedisAdmissionQueue._text(RedisAdmissionQueue._hash_value(data, "query", "")),
            history=[tuple(item) for item in json.loads(RedisAdmissionQueue._text(RedisAdmissionQueue._hash_value(data, "history")) or "[]")],
            llm_cost=int(RedisAdmissionQueue._text(RedisAdmissionQueue._hash_value(data, "llm_cost", 3))),
            idempotency_key=idem,
        )

    @staticmethod
    def _hash_value(data: dict, key: str, default: Any = None) -> Any:
        return data.get(key, data.get(key.encode(), default))

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value or "")

    @staticmethod
    def _raise_redis(exc: Exception) -> None:
        raise QueueUnavailableError("Redis admission queue unavailable") from exc


def _json_safe(value: Any) -> dict[str, Any]:
    """Serialize an RAGResult without importing API modules into core."""
    if isinstance(value, dict):
        return value
    sources = []
    for source in getattr(value, "sources", []) or []:
        sources.append(source.as_dict() if hasattr(source, "as_dict") else dict(source))
    return {
        "answer": getattr(value, "answer", ""),
        "contexts": list(getattr(value, "contexts", []) or []),
        "sources": sources,
        "expanded_queries": list(getattr(value, "expanded_queries", []) or []),
        "num_candidates": int(getattr(value, "num_candidates", 0)),
    }


_queue_instance: RedisAdmissionQueue | InlineAdmissionQueue | None = None
_queue_lock = threading.Lock()


def get_admission_queue() -> RedisAdmissionQueue | InlineAdmissionQueue:
    """Return the process singleton queue adapter."""
    global _queue_instance
    if _queue_instance is None:
        with _queue_lock:
            if _queue_instance is None:
                if settings.RAG_EXECUTION_MODE == "redis_worker":
                    _queue_instance = RedisAdmissionQueue()
                else:
                    _queue_instance = InlineAdmissionQueue()
    return _queue_instance


def reset_admission_queue_for_tests() -> None:
    global _queue_instance
    with _queue_lock:
        _queue_instance = None
