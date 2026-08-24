#!/usr/bin/env python3
"""Profile RAG module latency and end-to-end SSE time to first token.

The in-process mode executes the real retriever and records inclusive spans for
each module.  It creates a request-local admission reservation so the configured
LLM pacing remains visible even when production uses the Redis worker mode.

The HTTP mode calls ``/chat/stream`` and measures the user-observed TTFT,
including API, queue, worker, retrieval, provider and SSE transport time.
No query, answer, context, token, API key or document content is written to the
report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from typing_extensions import Self

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class Span:
    name: str
    run: int
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)


class LatencyRecorder:
    """Thread-safe inclusive span collector used by parallel query transforms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._spans: list[Span] = []
        self.run = 0
        self.enabled = True

    def add(
        self,
        name: str,
        duration_ms: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        span = Span(
            name=name,
            run=self.run,
            duration_ms=round(duration_ms, 3),
            details=details or {},
        )
        with self._lock:
            self._spans.append(span)

    @contextmanager
    def span(
        self,
        name: str,
        details: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter_ns() - started) / 1_000_000, details)

    @property
    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def mark_llm_permit_granted(self, timestamp_ns: int) -> None:
        self._local.llm_permit_granted_ns = timestamp_ns

    def llm_permit_granted_ns(self) -> int | None:
        return getattr(self._local, "llm_permit_granted_ns", None)


def _percentile(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile; deterministic and dependency-free."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def summarize(spans: list[Span], runs: int) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = {}
    for span in spans:
        grouped.setdefault(span.name, []).append(span.duration_ms)
    report: dict[str, dict[str, float | int]] = {}
    for name, values in sorted(grouped.items()):
        report[name] = {
            "count": len(values),
            "calls_per_run": round(len(values) / max(runs, 1), 2),
            "min_ms": round(min(values), 2),
            "mean_ms": round(statistics.mean(values), 2),
            "p50_ms": round(_percentile(values, 0.50), 2),
            "p95_ms": round(_percentile(values, 0.95), 2),
            "max_ms": round(max(values), 2),
        }
    return report


class RetrieverProfiler:
    """Temporarily wrap one retriever instance without changing production code."""

    def __init__(self, retriever: Any, recorder: LatencyRecorder) -> None:
        self.retriever = retriever
        self.recorder = recorder
        self._restore: list[tuple[Any, str, Any]] = []

    def _patch(
        self,
        target: Any,
        method_name: str,
        span_name: str,
        details: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        original = getattr(target, method_name)

        def timed(*args: Any, **kwargs: Any) -> Any:
            metadata = details(args, kwargs) if details else None
            with self.recorder.span(span_name, metadata):
                return original(*args, **kwargs)

        self._restore.append((target, method_name, original))
        setattr(target, method_name, timed)

    def _patch_final_stream(self) -> None:
        target = self.retriever.llm
        original = target.generate_stream

        def timed(*args: Any, **kwargs: Any):
            source = original(*args, **kwargs)

            def iterate():
                started = time.perf_counter_ns()
                first = True
                try:
                    for token in source:
                        if first:
                            first = False
                            first_token_at = time.perf_counter_ns()
                            self.recorder.add(
                                "generation.stream_ttft",
                                (first_token_at - started) / 1_000_000,
                            )
                            permit_at = self.recorder.llm_permit_granted_ns()
                            self.recorder.add(
                                "generation.provider_ttft",
                                (first_token_at - (permit_at or started)) / 1_000_000,
                            )
                        yield token
                finally:
                    finished_at = time.perf_counter_ns()
                    permit_at = self.recorder.llm_permit_granted_ns()
                    self.recorder.add(
                        "generation.stream_total",
                        (finished_at - started) / 1_000_000,
                        {"produced_token": not first},
                    )
                    self.recorder.add(
                        "generation.provider_total",
                        (finished_at - (permit_at or started)) / 1_000_000,
                        {"produced_token": not first},
                    )

            return iterate()

        self._restore.append((target, "generate_stream", original))
        target.generate_stream = timed

    def __enter__(self) -> Self:
        self._patch(self.retriever, "retrieve", "retrieval.total")
        self._patch(self.retriever.condenser, "condense", "query.condense")
        self._patch(self.retriever.expander, "expand", "query.multi_query")
        self._patch(
            self.retriever.hyde,
            "_generate_hypothetical",
            "query.hyde_llm",
        )
        self._patch(self.retriever.hyde.embedder, "embed_single", "embedding.hyde")
        self._patch(self.retriever.hyde, "generate_embedding", "query.hyde_total")
        self._patch(self.retriever.searcher.dense.embedder, "embed_single", "embedding.search_query")
        self._patch(
            self.retriever.searcher,
            "search",
            "search.hybrid_total",
            lambda args, kwargs: {
                "channel": "hyde_dense" if kwargs.get("hyde_vector") is not None else "hybrid",
                "include_sparse": kwargs.get("include_sparse", True),
            },
        )
        qdrant = self.retriever.searcher.dense.qdrant
        self._patch(qdrant, "search_hybrid", "qdrant.hybrid_query")
        self._patch(qdrant, "search", "qdrant.dense_query")
        self._patch(self.retriever.reranker, "rerank", "rerank.total")
        self._patch(self.retriever.parent_resolver.qdrant, "get_by_ids", "qdrant.parent_lookup")
        self._patch(self.retriever.parent_resolver, "resolve", "parent_resolution.total")
        self._patch(self.retriever.assembler, "assemble", "context_assembly.total")
        self._patch_final_stream()

        # RRF is imported into the orchestrator module, so patch that exact alias.
        import retrieval.retriever as retriever_module

        original_fusion = retriever_module.rrf_fusion

        def timed_fusion(*args: Any, **kwargs: Any) -> Any:
            with self.recorder.span("fusion.rrf"):
                return original_fusion(*args, **kwargs)

        self._restore.append((retriever_module, "rrf_fusion", original_fusion))
        retriever_module.rrf_fusion = timed_fusion
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        for target, name, original in reversed(self._restore):
            setattr(target, name, original)
        self._restore.clear()


def _load_history(path: str | None) -> list[tuple[str, str]]:
    if not path:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("History must be a JSON list of [user, assistant] pairs")
    history: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2 or not all(isinstance(value, str) for value in item):
            raise ValueError("Every history item must be [user, assistant]")
        history.append((item[0], item[1]))
    return history


def _consume_stream(
    retriever: Any,
    query: str,
    history: list[tuple[str, str]],
    recorder: LatencyRecorder,
    *,
    request_started_ns: int | None = None,
) -> None:
    started = request_started_ns or time.perf_counter_ns()
    first_token = False
    event_count = 0
    for item in retriever.stream_with_sources(query, history=history):
        event_count += 1
        event_name = item.get("event") if isinstance(item, dict) else "token"
        data = item.get("data", "") if isinstance(item, dict) else item
        if event_name == "token" and data and not first_token:
            first_token = True
            recorder.add("rag.in_process_ttft", (time.perf_counter_ns() - started) / 1_000_000)
    recorder.add(
        "rag.in_process_total",
        (time.perf_counter_ns() - started) / 1_000_000,
        {"event_count": event_count, "produced_token": first_token},
    )
    if not first_token:
        raise RuntimeError("RAG stream completed without a non-empty token event")


def _instrument_reservation(reservation: Any, recorder: LatencyRecorder) -> None:
    original = reservation.consume

    def consume() -> None:
        started = time.perf_counter_ns()
        try:
            original()
        except BaseException:
            recorder.add("admission.llm_pacing_wait", (time.perf_counter_ns() - started) / 1_000_000)
            raise
        granted_at = time.perf_counter_ns()
        recorder.add("admission.llm_pacing_wait", (granted_at - started) / 1_000_000)
        recorder.mark_llm_permit_granted(granted_at)

    reservation.consume = consume


def benchmark_in_process(
    query: str,
    history: list[tuple[str, str]],
    runs: int,
    warmup_runs: int,
    pacing_mode: str,
) -> dict[str, Any]:
    from core.admission_queue import RateScheduler, get_admission_queue, llm_call_cost, reservation_context
    from core.config import settings
    from retrieval.retriever import RAGRetriever

    startup_started = time.perf_counter_ns()
    retriever = RAGRetriever()
    init_ms = (time.perf_counter_ns() - startup_started) / 1_000_000
    warmup_started = time.perf_counter_ns()
    retriever.warmup()
    model_warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000

    recorder = LatencyRecorder()
    shared_scheduler = get_admission_queue().rate if pacing_mode == "shared" else None

    def execute_once() -> None:
        request_started = time.perf_counter_ns()
        scheduler = shared_scheduler or RateScheduler()
        reservation = scheduler.reserve(llm_call_cost(history))
        initial_wait_started = time.perf_counter_ns()
        reservation.wait_until_start()
        recorder.add(
            "admission.initial_wait",
            (time.perf_counter_ns() - initial_wait_started) / 1_000_000,
        )
        _instrument_reservation(reservation, recorder)
        with reservation_context(reservation):
            _consume_stream(
                retriever,
                query,
                history,
                recorder,
                request_started_ns=request_started,
            )

    with RetrieverProfiler(retriever, recorder):
        recorder.enabled = False
        for _ in range(warmup_runs):
            execute_once()
        recorder.enabled = True

        for run in range(1, runs + 1):
            recorder.run = run
            execute_once()

    interval_ms = (
        (settings.LLM_RATE_LIMIT_WINDOW_SECONDS + 0.001)
        / max(settings.LLM_RATE_LIMIT_CALLS, 1)
        * 1000
    )
    call_cost = llm_call_cost(history)
    return {
        "startup": {
            "retriever_init_ms": round(init_ms, 2),
            "model_warmup_ms": round(model_warmup_ms, 2),
        },
        "admission_model": {
            "pacing_mode": pacing_mode,
            "llm_calls_per_request": call_cost,
            "configured_interval_ms": round(interval_ms, 2),
            "minimum_intra_request_pacing_before_final_call_ms": round(interval_ms * (call_cost - 1), 2),
            "note": (
                "Shared mode respects the configured admission scheduler but bypasses the job queue; "
                "measure full queue/API overhead with HTTP mode."
                if pacing_mode == "shared"
                else "Isolated mode excludes shared quota backlog; do not run it against a provider used by production."
            ),
        },
        "summary": summarize(recorder.spans, runs),
        "spans": [asdict(span) for span in recorder.spans],
    }


def _sse_events(lines: Iterator[str]) -> Iterator[tuple[str, str]]:
    event_name = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines or event_name != "message":
                yield event_name, "\n".join(data_lines)
            event_name, data_lines = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if data_lines or event_name != "message":
        yield event_name, "\n".join(data_lines)


def _has_nonempty_token(data: str) -> bool:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return bool(data)
    if isinstance(payload, dict):
        return bool(payload.get("text"))
    return bool(payload)


def _stream_endpoint(api_url: str) -> str:
    return api_url.rstrip("/") if api_url.rstrip("/").endswith("/chat/stream") else api_url.rstrip("/") + "/chat/stream"


def benchmark_http(
    api_url: str,
    api_key: str,
    query: str,
    history: list[tuple[str, str]],
    runs: int,
    warmup_runs: int,
    timeout: float,
) -> dict[str, Any]:
    recorder = LatencyRecorder()
    payload = {
        "query": query,
        "history": [
            message
            for user, assistant in history
            for message in (
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            )
        ],
    }
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["X-API-Key"] = api_key
    endpoint = _stream_endpoint(api_url)

    def one(client: httpx.Client, record: bool) -> None:
        started = time.perf_counter_ns()
        first_status = False
        first_status_at: int | None = None
        worker_started = False
        first_token = False
        saw_end = False
        with client.stream("POST", endpoint, json=payload, headers=headers) as response:
            response.raise_for_status()
            if record:
                recorder.add("http.response_headers", (time.perf_counter_ns() - started) / 1_000_000)
            for event_name, data in _sse_events(response.iter_lines()):
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                if event_name == "status" and not first_status:
                    first_status = True
                    first_status_at = time.perf_counter_ns()
                    if record:
                        recorder.add("http.first_status", elapsed_ms)
                if event_name == "status" and not worker_started:
                    try:
                        status_payload = json.loads(data)
                    except json.JSONDecodeError:
                        status_payload = {}
                    if isinstance(status_payload, dict) and status_payload.get("stage") == "started":
                        worker_started = True
                        if record:
                            recorder.add("http.worker_started", elapsed_ms)
                            recorder.add(
                                "admission.queue_wait_to_worker",
                                (time.perf_counter_ns() - (first_status_at or started)) / 1_000_000,
                            )
                if event_name == "token" and _has_nonempty_token(data) and not first_token:
                    first_token = True
                    if record:
                        recorder.add("rag.http_ttft", elapsed_ms)
                if event_name == "error":
                    raise RuntimeError("RAG API returned an SSE error event")
                if event_name == "end":
                    saw_end = True
                    break
        if record:
            recorder.add(
                "rag.http_total",
                (time.perf_counter_ns() - started) / 1_000_000,
                {"worker_started_event": worker_started},
            )
        if not first_token or not saw_end:
            raise RuntimeError("SSE stream did not contain both a non-empty token and end event")

    with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0))) as client:
        recorder.enabled = False
        for _ in range(warmup_runs):
            one(client, False)
        recorder.enabled = True
        for run in range(1, runs + 1):
            recorder.run = run
            one(client, True)

    parsed = urlparse(endpoint)
    return {
        "endpoint": f"{parsed.scheme}://{parsed.netloc}{parsed.path}",
        "summary": summarize(recorder.spans, runs),
        "spans": [asdict(span) for span in recorder.spans],
    }


def _safe_config() -> dict[str, Any]:
    from core.config import settings

    names = (
        "LLM_PROVIDER",
        "OPENAI_MODEL_ID",
        "GEMINI_MODEL_ID",
        "EMBEDDING_RUNTIME",
        "EMBEDDING_MODEL_ID",
        "RERANKER_RUNTIME",
        "RERANKER_MODEL_ID",
        "RAG_EXECUTION_MODE",
        "EXPAND_N_QUERY",
        "TOP_K",
        "RERANK_CANDIDATES",
        "KEEP_TOP_K",
        "MAX_CONTEXT_CHARS",
        "LLM_RATE_LIMIT_CALLS",
        "LLM_RATE_LIMIT_WINDOW_SECONDS",
    )
    return {name: getattr(settings, name) for name in names}


def _print_summary(title: str, summary: dict[str, dict[str, float | int]]) -> None:
    print(f"\n{title}")
    print("Inclusive spans can overlap (Multi-Query and HyDE run concurrently); do not sum rows.")
    print(f"{'span':38} {'calls/run':>9} {'mean ms':>11} {'p50 ms':>11} {'p95 ms':>11}")
    print("-" * 85)
    for name, stats in summary.items():
        print(
            f"{name:38} {stats['calls_per_run']:>9} "
            f"{stats['mean_ms']:>11.2f} {stats['p50_ms']:>11.2f} {stats['p95_ms']:>11.2f}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure per-module RAG latency and end-to-end SSE TTFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", required=True, help="Benchmark question (never written to the report)")
    parser.add_argument("--mode", choices=("in-process", "http", "both"), default="in-process")
    parser.add_argument("--api-url", default="http://127.0.0.1:8080", help="API base URL or /chat/stream URL")
    parser.add_argument(
        "--api-key-env",
        default="RAG_BENCHMARK_API_KEY",
        help="Environment variable containing X-API-Key; its value is never reported",
    )
    parser.add_argument("--history-file", help="Optional JSON file of [[user, assistant], ...]")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=0, help="Extra paid/unreported RAG calls")
    parser.add_argument(
        "--pacing",
        choices=("shared", "isolated"),
        default="shared",
        help="In-process LLM quota scheduler; isolated must not share a live production provider",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout in seconds")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.runs <= 0 or args.warmup_runs < 0 or args.timeout <= 0:
        parser.error("--runs and --timeout must be positive; --warmup-runs cannot be negative")
    if not args.query.strip():
        parser.error("--query cannot be blank")

    history = _load_history(args.history_file)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "history_turns": len(history),
        "config": _safe_config(),
        "privacy": "Query, history, answer, context, tokens and credentials are intentionally omitted.",
    }

    if args.mode in {"in-process", "both"}:
        report["in_process"] = benchmark_in_process(
            args.query,
            history,
            args.runs,
            args.warmup_runs,
            args.pacing,
        )
        _print_summary("In-process module and RAG latency", report["in_process"]["summary"])
        pacing = report["in_process"]["admission_model"]
        print(
            "\nConfigured pacing floor before final LLM call: "
            f"{pacing['minimum_intra_request_pacing_before_final_call_ms']:.2f} ms "
            f"({pacing['llm_calls_per_request']} calls/request)"
        )

    if args.mode in {"http", "both"}:
        api_key = os.getenv(args.api_key_env, "")
        report["http"] = benchmark_http(
            args.api_url,
            api_key,
            args.query,
            history,
            args.runs,
            args.warmup_runs,
            args.timeout,
        )
        _print_summary("HTTP/SSE end-to-end latency", report["http"]["summary"])

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
