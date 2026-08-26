"""Redis Streams worker for admitted RAG jobs.

Run with ``make run-worker`` (or ``python -m workers.rag_worker``).  The API
process only enqueues work in production; this process owns the retriever and
executes jobs after a paced LLM reservation has been acquired.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from api.chat import get_retriever
from core import get_logger
from core.admission_queue import RedisAdmissionQueue, get_admission_queue
from core.config import settings

logger = get_logger(__name__)


def _handle_job(query: str, history: list[tuple[str, str]]):
    """Execute the complete RAG request in the worker-owned retriever."""
    return get_retriever().query_with_context(query, history=history)


def _handle_job_stream(
    query: str,
    history: list[tuple[str, str]],
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Run RAG and forward each provider token to the job event stream."""
    answer_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    contexts: list[str] = []
    expanded_queries: list[str] = []
    num_candidates = 0
    for event in get_retriever().stream_with_sources(
        query,
        history=history,
        include_contexts=True,
    ):
        event_name = event.get("event") if isinstance(event, dict) else ""
        data = event.get("data", "") if isinstance(event, dict) else ""
        if event_name not in {"contexts", "metadata"}:
            emit(event)
        if event_name == "token":
            answer_parts.append(str(data.get("text", "") if isinstance(data, dict) else data))
        elif event_name == "sources":
            sources = list(data.get("sources", []) if isinstance(data, dict) else data or [])
        elif event_name == "contexts":
            contexts = [str(item) for item in data or []]
        elif event_name == "metadata":
            expanded_queries = [str(item) for item in data.get("expanded_queries", [])]
            num_candidates = int(data.get("num_candidates", 0))
    return {
        "answer": "".join(answer_parts),
        "sources": sources,
        "contexts": contexts,
        "expanded_queries": expanded_queries,
        "num_candidates": num_candidates or len(contexts),
    }


def run_worker(*, stop_event: threading.Event | None = None) -> None:
    """Consume Redis jobs until SIGTERM/Ctrl-C or the supplied event fires."""
    queue = get_admission_queue()
    if not isinstance(queue, RedisAdmissionQueue):
        raise TypeError("RAG worker requires RAG_EXECUTION_MODE=redis_worker")
    stop_event = stop_event or threading.Event()
    executor = ThreadPoolExecutor(
        max_workers=settings.RAG_WORKER_CONCURRENCY,
        thread_name_prefix="rag-job",
    )
    worker_id = f"{threading.current_thread().name}-{id(stop_event)}"
    heartbeat_interval = max(1.0, settings.RAG_WORKER_HEARTBEAT_TTL_SECONDS / 3)
    next_heartbeat = 0.0

    def refresh_heartbeat() -> None:
        nonlocal next_heartbeat
        now = time.monotonic()
        if now >= next_heartbeat:
            queue.set_worker_heartbeat(worker_id)
            next_heartbeat = now + heartbeat_interval

    def stop(_signum, _frame):
        stop_event.set()

    previous_int = previous_term = None
    if threading.current_thread() is threading.main_thread():
        previous_int = signal.signal(signal.SIGINT, stop)
        previous_term = signal.signal(signal.SIGTERM, stop)
    try:
        get_retriever().warmup()
        refresh_heartbeat()
        logger.info(
            "RAG worker started",
            concurrency=settings.RAG_WORKER_CONCURRENCY,
            stream=settings.REDIS_QUEUE_STREAM,
            group=settings.REDIS_QUEUE_GROUP,
        )
        # A small fixed pool prevents one slow GPU/LLM job from stopping all
        # consumers, while Redis outstanding admission remains the hard queue
        # bound.  ``run_once`` blocks only for the short XREADGROUP poll.
        futures = set()
        while not stop_event.is_set():
            refresh_heartbeat()
            while len(futures) < settings.RAG_WORKER_CONCURRENCY and not stop_event.is_set():
                consumer = f"worker-{threading.current_thread().name}-{len(futures)}"
                future = executor.submit(
                    queue.run_once,
                    _handle_job,
                    stream_handler=_handle_job_stream,
                    consumer=consumer,
                    block_ms=500,
                )
                futures.add(future)
                # Submit at most one extra read per loop; each call returns
                # after a message or the bounded 500ms poll.
                break
            done = {future for future in futures if future.done()}
            for future in done:
                futures.remove(future)
                try:
                    future.result()
                except Exception:
                    logger.exception("RAG worker poll failed")
            if not done and len(futures) >= settings.RAG_WORKER_CONCURRENCY:
                # Do not busy-spin while all consumers are waiting in Redis.
                stop_event.wait(0.05)
    finally:
        stop_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        if previous_int is not None and previous_term is not None:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)
        queue.clear_worker_heartbeat(worker_id)
        logger.info("RAG worker stopped")


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()
