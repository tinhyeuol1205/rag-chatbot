"""PR15 contract tests for the thin FastAPI UI client."""

from __future__ import annotations

import json

import httpx
import pytest

from api.ui_client import (
    UIAPIClient,
    UIClientError,
    _parse_sse,
    normalize_history,
    sanitize_sources,
)


def test_history_is_normalized_and_bounded():
    history = [
        ["q1", "a1"],
        ["q2", "a2"],
        {"role": "user", "content": "dangling"},
    ]

    assert normalize_history(history[:2], max_turns=1) == [("q2", "a2")]
    assert normalize_history(history[2:], max_turns=3) == []


def test_sources_are_allowlisted_and_control_chars_removed():
    sources = sanitize_sources(
        [
            {
                "citation_id": "2",
                "file_name": "policy\n.md",
                "section_title": "<script>",
                "page_number": "4",
                "dataset_id": "secret",
            },
            {"file_name": ""},
        ]
    )

    assert sources == [
        {
            "citation_id": 2,
            "file_name": "policy.md",
            "section_title": "<script>",
            "page_number": 4,
        }
    ]


def test_sse_parser_handles_json_data_and_event_id():
    events = list(
        _parse_sse(
            [
                "id: 1-0",
                "event: token",
                f"data: {json.dumps({'text': 'Xin chào'}, ensure_ascii=False)}",
                "",
                "event: end",
                "data: {}",
                "",
            ]
        )
    )

    assert events == [
        {"event": "token", "data": {"text": "Xin chào"}, "id": "1-0"},
        {"event": "end", "data": {}},
    ]


def test_preflight_rejects_wrong_execution_mode():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ready"
        return httpx.Response(
            200,
            json={
                "status": "ready",
                "execution_mode": "inline",
                "embedding_runtime": "local",
                "reranker_runtime": "local",
                "worker_available": False,
            },
        )

    transport = httpx.MockTransport(handler)
    raw_client = httpx.Client(transport=transport, base_url="http://test")
    api = UIAPIClient("http://test", client=raw_client)

    with pytest.raises(UIClientError, match="execution_mode=redis_worker"):
        api.preflight("product")
    raw_client.close()


def test_stream_preserves_retry_after_on_overload():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "7"},
            json={"detail": {"code": "queue_full", "message": "busy"}},
        )

    raw_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    )
    api = UIAPIClient("http://test", client=raw_client)

    with pytest.raises(UIClientError) as caught:
        list(api.stream("question"))
    assert caught.value.code == "queue_full"
    assert caught.value.retry_after == "7"
    raw_client.close()
