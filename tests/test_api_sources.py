"""API contract tests for structured sources and SSE event types."""

import json

from fastapi.testclient import TestClient

import api.main as api_main
from core.config import settings
from retrieval.context.assembler import SourceRef
from retrieval.retriever import RAGResult


def test_chat_response_contains_sources(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret")
    monkeypatch.setattr(
        api_main,
        "chat_or_raise_with_sources",
        lambda _query: RAGResult(
            answer="Employees change it every 90 days [1]",
            sources=[SourceRef(1, "policy.md", "Password Policy", 4)],
        ),
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/chat",
        json={"query": "How often?"},
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Employees change it every 90 days [1]",
        "sources": [{
            "citation_id": 1,
            "file_name": "policy.md",
            "section_title": "Password Policy",
            "page_number": 4,
        }],
    }


def test_stream_has_sources_event_before_end(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret")

    def fake_events(_query):
        yield {"event": "token", "data": "answer"}
        yield {
            "event": "sources",
            "data": [{"citation_id": 1, "file_name": "policy.md"}],
        }

    monkeypatch.setattr(api_main, "chat_stream_events", fake_events)
    client = TestClient(api_main.app)

    response = client.post(
        "/chat/stream",
        json={"query": "q"},
        headers={"X-API-Key": "secret"},
    )

    body = response.text
    assert response.status_code == 200
    assert body.index("event: token") < body.index("event: sources")
    assert body.index("event: sources") < body.index("event: end")

    normalized_body = body.replace("\r\n", "\n")
    source_data = normalized_body.split("event: sources\ndata: ", 1)[1].split("\n\n", 1)[0]
    assert json.loads(source_data) == {
        "sources": [{"citation_id": 1, "file_name": "policy.md"}],
    }
