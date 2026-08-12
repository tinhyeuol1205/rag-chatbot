"""Regression tests for API-key enforcement on both chat endpoints."""

from fastapi.testclient import TestClient

import api.main as api_main
from core.config import settings
from retrieval.context.assembler import SourceRef
from retrieval.retriever import RAGResult


def _fake_stream(_query):
    yield "ok"


def test_both_chat_endpoints_require_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret")
    monkeypatch.setattr(
        api_main,
        "chat_or_raise_with_sources",
        lambda _q: RAGResult(answer="ok"),
    )
    monkeypatch.setattr(api_main, "chat_stream_events", _fake_stream)
    client = TestClient(api_main.app)

    response = client.post("/chat", json={"query": "q"})
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_api_key"

    response = client.post("/chat/stream", json={"query": "q"})
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_api_key"


def test_wrong_key_is_rejected_for_stream(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret")
    monkeypatch.setattr(api_main, "chat_stream_events", _fake_stream)
    client = TestClient(api_main.app)

    response = client.post(
        "/chat/stream",
        json={"query": "q"},
        headers={"X-API-Key": "not-secret"},
    )

    assert response.status_code == 401

def test_both_chat_endpoints_accept_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret")
    monkeypatch.setattr(
        api_main,
        "chat_or_raise_with_sources",
        lambda _q: RAGResult(
            answer="ok",
            sources=[SourceRef(1, "policy.md", "Password Policy", 2)],
        ),
    )
    monkeypatch.setattr(api_main, "chat_stream_events", _fake_stream)
    client = TestClient(api_main.app)
    headers = {"X-API-Key": "secret"}

    response = client.post("/chat", json={"query": "q"}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {
        "answer": "ok",
        "sources": [{
            "citation_id": 1,
            "file_name": "policy.md",
            "section_title": "Password Policy",
            "page_number": 2,
        }],
    }

    response = client.post("/chat/stream", json={"query": "q"}, headers=headers)
    assert response.status_code == 200
    assert "ok" in response.text


def test_dev_mode_keeps_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(
        api_main,
        "chat_or_raise_with_sources",
        lambda _q: RAGResult(answer="ok"),
    )
    client = TestClient(api_main.app)

    response = client.post("/chat", json={"query": "q"})
    assert response.status_code == 200


def test_request_cannot_select_an_arbitrary_dataset(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    client = TestClient(api_main.app)

    response = client.post(
        "/chat",
        json={"query": "q", "dataset_id": "another_team"},
    )

    assert response.status_code == 422
