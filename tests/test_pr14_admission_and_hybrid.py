"""PR14 regressions: admission boundary and scoped native hybrid requests."""

from __future__ import annotations

from types import SimpleNamespace

from core.admission_queue import InlineAdmissionQueue, consume_llm_permit, llm_call_cost
from core.config import settings
from core.db.qdrant import QdrantConnector
from retrieval.scope import RetrievalScope
from retrieval.search.hybrid import HybridSearcher
from retrieval.search.sparse import sparse_document


def test_llm_reservation_is_consumed_inside_admitted_handler(monkeypatch):
    monkeypatch.setattr(settings, "LLM_RATE_LIMIT_CALLS", 100_000)
    queue = InlineAdmissionQueue()
    seen = []

    def handler(query, history):
        seen.append((query, history))
        consume_llm_permit()
        return "done"

    assert queue.execute(handler, "hello", [("u", "a")]) == "done"
    assert seen == [("hello", [("u", "a")])]
    assert llm_call_cost([]) == 3
    assert llm_call_cost([("u", "a")]) == 4


class _HybridQdrant:
    def __init__(self):
        self.kwargs = None

    def search_hybrid(self, *_args, **kwargs):
        self.kwargs = kwargs
        return [
            SimpleNamespace(
                id="allowed",
                score=0.9,
                payload={"dataset_id": "team_a", "content": "ok"},
            )
        ]


def test_native_hybrid_passes_identical_scope_filter_to_qdrant(monkeypatch):
    searcher = HybridSearcher.__new__(HybridSearcher)
    qdrant = _HybridQdrant()
    searcher.dense = SimpleNamespace(
        qdrant=qdrant,
        embedder=SimpleNamespace(embed_single=lambda _query: [0.1, 0.2]),
    )
    searcher.sparse = SimpleNamespace()
    scope = RetrievalScope(("team_a",))

    results = searcher.search("policy", top_k=2, scope=scope)

    assert [row["chunk_id"] for row in results] == ["allowed"]
    assert qdrant.kwargs["query_filter"] is not None
    assert qdrant.kwargs["query_filter"].must[0].key == "dataset_id"


def test_qdrant_hybrid_attaches_scope_to_both_prefetch_branches():
    class Client:
        def __init__(self):
            self.kwargs = None

        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(points=[])

    connector = QdrantConnector.__new__(QdrantConnector)
    client = Client()
    connector._client = client
    query_filter = RetrievalScope(("team_a",)).qdrant_filter()

    connector.search_hybrid(
        "child_chunks_active",
        dense_vector=[0.1, 0.2],
        sparse_query=sparse_document("policy"),
        sparse_vector_name="bm25",
        limit=5,
        prefetch_limit=10,
        query_filter=query_filter,
    )

    prefetch = client.kwargs["prefetch"]
    assert len(prefetch) == 2
    assert prefetch[0].filter is query_filter
    assert prefetch[1].filter is query_filter
    assert client.kwargs["query_filter"] is query_filter
