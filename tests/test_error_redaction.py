"""Regression tests ensuring internal exception details stay server-side."""

from fastapi.testclient import TestClient

import api.main as api_main
from api.chat import chat, chat_stream
from core.config import settings
from core.errors import RetrievalError


class _BrokenRetriever:
    def query(self, *args, **kwargs):
        raise RetrievalError(
            "connection refused: http://qdrant.internal:6333?token=secret"
        )


def test_non_stream_does_not_leak_internal_error(monkeypatch):
    monkeypatch.setattr("api.chat.get_retriever", lambda: _BrokenRetriever())

    output = chat("question")

    assert "qdrant.internal" not in output
    assert "secret" not in output
    assert RetrievalError.public_message in output


def test_stream_does_not_leak_internal_error(monkeypatch):
    monkeypatch.setattr("api.chat.get_retriever", lambda: _BrokenRetriever())

    output = "".join(chat_stream("question"))

    assert "qdrant.internal" not in output
    assert "secret" not in output
    assert RetrievalError.public_message in output


def test_api_error_body_contains_only_public_contract(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret")

    def _raise(_query):
        raise RetrievalError(
            "connection refused: http://qdrant.internal:6333?token=secret"
        )

    monkeypatch.setattr(api_main, "chat_or_raise_with_sources", _raise)
    client = TestClient(api_main.app)

    response = client.post(
        "/chat",
        json={"query": "question"},
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 503
    body = response.json()
    assert body == {
        "detail": {
            "code": "retrieval_unavailable",
            "message": RetrievalError.public_message,
        }
    }
    assert "qdrant.internal" not in response.text
    assert "secret" not in response.text
