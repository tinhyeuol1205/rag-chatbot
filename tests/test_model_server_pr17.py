from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from model_server.batcher import BatchLimits, DynamicBatcher
from model_server.errors import ModelQueueFullError, ModelQueueTimeoutError
from model_server.main import create_app
from model_server.runtime import MPSExecutionArbiter


def _limits(**overrides) -> BatchLimits:
    values = {
        "max_items": 8,
        "max_tokens": 128,
        "max_bytes": 10_000,
        "max_pending_requests": 32,
        "max_pending_items": 128,
        "max_pending_bytes": 100_000,
        "max_wait_ms": 2,
        "queue_timeout_seconds": 1,
        "max_items_per_request": 8,
        "max_tokens_per_item": 32,
    }
    values.update(overrides)
    return BatchLimits(**values)


def test_dynamic_batcher_coalesces_and_preserves_cardinality():
    async def scenario():
        calls: list[list[str]] = []

        async def infer(values: list[str]) -> list[str]:
            calls.append(values[:])
            await asyncio.sleep(0.001)
            return [value.upper() for value in values]

        batcher = DynamicBatcher("test", infer, _limits())
        await batcher.start()
        try:
            results = await asyncio.gather(*(batcher.submit([str(index)], [1]) for index in range(12)))
        finally:
            await batcher.stop()
        return results, calls, batcher.stats()

    results, calls, stats = asyncio.run(scenario())
    assert results == [[str(index).upper()] for index in range(12)]
    assert len(calls) < 12
    assert stats["pending_requests"] == 0


def test_dynamic_batcher_has_bounded_admission_and_timeout():
    async def scenario():
        async def infer(values: list[str]) -> list[str]:
            await asyncio.sleep(0.2)
            return values

        batcher = DynamicBatcher(
            "test",
            infer,
            _limits(max_pending_requests=1, max_pending_items=1, queue_timeout_seconds=0.03),
        )
        await batcher.start()
        first = asyncio.create_task(batcher.submit(["first"], [1]))
        await asyncio.sleep(0.005)
        try:
            with pytest.raises(ModelQueueFullError):
                await batcher.submit(["second"], [1])
            with pytest.raises(ModelQueueTimeoutError):
                await first
        finally:
            await batcher.stop()

    asyncio.run(scenario())


def test_dynamic_batcher_client_cancellation_releases_capacity():
    async def scenario():
        started = asyncio.Event()

        async def infer(values: list[str]) -> list[str]:
            started.set()
            await asyncio.sleep(0.05)
            return values

        batcher = DynamicBatcher("test", infer, _limits(max_pending_requests=1, max_pending_items=1))
        await batcher.start()
        request = asyncio.create_task(batcher.submit(["cancel-me"], [1]))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        # The synchronous/model callback is allowed to finish; capacity is
        # intentionally held while an in-flight forward cannot be cancelled.
        await asyncio.sleep(0.1)
        assert batcher.stats()["pending_requests"] == 0
        await batcher.stop()

    asyncio.run(scenario())


def test_mps_arbiter_holds_slot_until_cancelled_forward_finishes():
    async def scenario():
        arbiter = MPSExecutionArbiter(1)
        finished = asyncio.Event()

        async def forward():
            await asyncio.sleep(0.03)
            finished.set()
            return "done"

        task = asyncio.create_task(arbiter.execute(forward))
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert arbiter.stats()["active_batches"] == 0

    asyncio.run(scenario())


class _FakeRuntime:
    device = "cpu"

    def check(self):
        return self.device


class _FakeEmbedding:
    model_id = "BAAI/bge-m3"
    revision = "embedding-revision"
    dimension = 3
    normalize = True
    arbiter = None

    def load(self):
        return None

    def metadata(self):
        return {
            "model": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "ready": True,
        }

    def token_lengths(self, values):
        return [1 for _ in values]

    async def infer(self, values):
        return [[float(len(value)), 0.0, 1.0] for value in values]


class _FakeReranker:
    model_id = "BAAI/bge-reranker-v2-m3"
    revision = "reranker-revision"
    max_input_tokens = 8
    arbiter = None

    def load(self):
        return None

    def metadata(self):
        return {"model": self.model_id, "revision": self.revision, "ready": True}

    def token_lengths(self, query, documents):
        return [1 for _ in documents]

    async def infer(self, query, documents):
        return [float(len(document)) for document in documents]


def test_model_server_contract_and_metadata():
    app = create_app(
        embedding_runner=_FakeEmbedding(),
        reranker_runner=_FakeReranker(),
        runtime=_FakeRuntime(),
    )
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["status"] == "ready"
        metadata = client.get("/metadata").json()
        assert metadata["embedding_model"] == "BAAI/bge-m3"
        assert metadata["embedding_dimension"] == 3

        response = client.post(
            "/embed",
            json={
                "inputs": ["một", "hai"],
                "model": "BAAI/bge-m3",
                "revision": "embedding-revision",
                "priority": "online",
            },
        )
        assert response.status_code == 200
        assert len(response.json()["embeddings"]) == 2

        response = client.post(
            "/rerank",
            json={
                "query": "q",
                "documents": ["ngắn", "dài hơn"],
                "model": "BAAI/bge-reranker-v2-m3",
                "revision": "reranker-revision",
            },
        )
        assert response.status_code == 200
        assert response.json()["scores"] == [4.0, 7.0]

        mismatch = client.post("/embed", json={"inputs": ["x"], "model": "wrong"})
        assert mismatch.status_code == 409
