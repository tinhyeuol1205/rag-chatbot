"""PR15 Redis event-stream and FastAPI relay tests."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

import api.main as api_main
from core.admission_queue import AdmissionJob, RedisAdmissionQueue
from core.config import settings


class _EventRedis:
    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.sequence: dict[str, int] = {}

    def hincrby(self, key, field, amount):
        current = int(self.hashes.setdefault(key, {}).get(field, 0)) + amount
        self.hashes[key][field] = str(current)
        return current

    def xadd(self, stream, fields):
        next_id = f"{len(self.streams.setdefault(stream, [])) + 1}-0"
        self.streams[stream].append((next_id, dict(fields)))
        return next_id

    def hset(self, key, field=None, value=None, mapping=None):
        row = self.hashes.setdefault(key, {})
        if mapping is not None:
            row.update({str(k): str(v) for k, v in mapping.items()})
        else:
            row[str(field)] = str(value)

    def expire(self, _key, _ttl):
        return True

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def xread(self, streams, **_kwargs):
        stream, last_id = next(iter(streams.items()))
        last_number = int(str(last_id).split("-", 1)[0])
        messages = [row for row in self.streams.get(stream, []) if int(row[0].split("-", 1)[0]) > last_number]
        return [(stream, messages)] if messages else []


def test_redis_event_stream_preserves_sequence_and_payload(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_JOB_EVENT_PREFIX", "test:events:")
    monkeypatch.setattr(settings, "REDIS_JOB_PREFIX", "test:job:")
    redis = _EventRedis()
    queue = RedisAdmissionQueue.__new__(RedisAdmissionQueue)
    queue.client = redis

    queue.publish_event("job-1", {"event": "token", "data": {"text": "Xin chào"}})
    queue.publish_event("job-1", {"event": "end", "data": {}})
    redis.hashes["test:job:job-1"]["state"] = "completed"

    events, state = queue.read_events("job-1")

    assert state == "completed"
    assert [event["event"] for event in events] == ["token", "end"]
    assert [event["sequence"] for event in events] == ["1", "2"]
    assert events[0]["data"] == {"text": "Xin chào"}


@dataclass
class _FakeAPIQueue:
    events: list[dict]
    state: str = "completed"

    def enqueue(self, query, history, *, idempotency_key=None):
        assert query == "q"
        assert history == []
        return AdmissionJob("job-1", query, history, 3, idempotency_key or "generated")

    def read_events(self, _job_id, *, last_id="0-0", block_ms=500):
        assert block_ms > 0
        if last_id == "0-0":
            return self.events, self.state
        return [], self.state

    def cancel(self, _job_id):
        return "completed"


def test_product_sse_relays_worker_events(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(settings, "RAG_EXECUTION_MODE", "redis_worker")
    queue = _FakeAPIQueue(
        [
            {"id": "1-0", "event": "status", "data": {"stage": "started"}},
            {"id": "2-0", "event": "token", "data": {"text": "answer"}},
            {"id": "3-0", "event": "sources", "data": {"sources": []}},
            {"id": "4-0", "event": "end", "data": {}},
        ]
    )
    monkeypatch.setattr(api_main, "get_admission_queue", lambda: queue)
    client = TestClient(api_main.app)

    response = client.post("/chat/stream", json={"query": "q"})

    assert response.status_code == 200
    body = response.text.replace("\r\n", "\n")
    assert body.index("event: status") < body.index("event: token")
    assert body.index("event: token") < body.index("event: sources")
    assert body.index("event: sources") < body.index("event: end")
    assert 'data: {"text": "answer"}' in body


def test_product_sse_has_one_terminal_event_when_worker_state_is_failed(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(settings, "RAG_EXECUTION_MODE", "redis_worker")
    queue = _FakeAPIQueue([], state="failed")
    monkeypatch.setattr(api_main, "get_admission_queue", lambda: queue)
    client = TestClient(api_main.app)

    response = client.post("/chat/stream", json={"query": "q"})

    body = response.text
    assert response.status_code == 200
    assert body.count("event: end") == 1
    assert "event: error" in body
