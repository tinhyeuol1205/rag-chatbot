"""Unit tests for the dependency-free parts of the RAG latency benchmark."""

from __future__ import annotations

import httpx

import scripts.benchmark_rag_latency as benchmark
from scripts.benchmark_rag_latency import Span, _has_nonempty_token, _sse_events, _stream_endpoint, summarize


def test_sse_parser_preserves_multiline_data_and_event_order():
    lines = iter(
        [
            "event: status",
            'data: {"stage":"started"}',
            "",
            ": keep-alive",
            "event: token",
            'data: {"text":"Xin',
            'data: chào"}',
            "",
            "event: end",
            "data: {}",
            "",
        ]
    )

    assert list(_sse_events(lines)) == [
        ("status", '{"stage":"started"}'),
        ("token", '{"text":"Xin\nchào"}'),
        ("end", "{}"),
    ]


def test_summary_reports_repeated_module_calls_per_run():
    spans = [
        Span("search.hybrid_total", 1, 10.0),
        Span("search.hybrid_total", 1, 20.0),
        Span("search.hybrid_total", 2, 30.0),
        Span("search.hybrid_total", 2, 40.0),
    ]

    result = summarize(spans, runs=2)["search.hybrid_total"]

    assert result == {
        "count": 4,
        "calls_per_run": 2.0,
        "min_ms": 10.0,
        "mean_ms": 25.0,
        "p50_ms": 20.0,
        "p95_ms": 40.0,
        "max_ms": 40.0,
    }


def test_sse_token_detection_and_endpoint_normalization():
    assert _has_nonempty_token('{"text":"answer"}') is True
    assert _has_nonempty_token('{"text":""}') is False
    assert _stream_endpoint("http://127.0.0.1:8080") == "http://127.0.0.1:8080/chat/stream"
    assert _stream_endpoint("http://127.0.0.1:8080/chat/stream") == "http://127.0.0.1:8080/chat/stream"


def test_http_benchmark_measures_worker_queue_and_ttft(monkeypatch):
    body = (
        'event: status\ndata: {"message":"accepted"}\n\n'
        'event: status\ndata: {"stage":"started"}\n\n'
        'event: token\ndata: {"text":"answer"}\n\n'
        "event: sources\ndata: {\"sources\":[]}\n\n"
        "event: end\ndata: {}\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/stream"
        assert request.headers["X-API-Key"] == "secret"
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    real_client = httpx.Client

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(benchmark.httpx, "Client", client_factory)

    report = benchmark.benchmark_http(
        "http://rag.test",
        "secret",
        "question",
        [],
        runs=1,
        warmup_runs=0,
        timeout=10.0,
    )

    assert report["endpoint"] == "http://rag.test/chat/stream"
    assert report["summary"]["admission.queue_wait_to_worker"]["count"] == 1
    assert report["summary"]["rag.http_ttft"]["count"] == 1
    assert report["summary"]["rag.http_total"]["count"] == 1
