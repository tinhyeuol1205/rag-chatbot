"""Regression tests ensuring dense, sparse and parent lookup share one scope."""

from types import SimpleNamespace

from retrieval.context.parent_resolver import ParentResolver
from retrieval.scope import RetrievalScope
from retrieval.search.dense import DenseSearcher
from retrieval.search.sparse import SparseSearcher, invalidate_bm25_index

SCOPE = RetrievalScope(("team_a",))


class _DenseQdrant:
    def __init__(self):
        self.query_filter = None

    def search(self, **kwargs):
        self.query_filter = kwargs["query_filter"]
        return [
            SimpleNamespace(
                id="a",
                score=0.9,
                payload={"content": "allowed", "dataset_id": "team_a"},
            ),
            SimpleNamespace(
                id="b",
                score=0.8,
                payload={"content": "forbidden", "dataset_id": "team_b"},
            ),
        ]


def test_dense_search_passes_scope_filter_and_rejects_out_of_scope_results():
    searcher = DenseSearcher.__new__(DenseSearcher)
    searcher.qdrant = _DenseQdrant()
    searcher.embedder = SimpleNamespace(embed_single=lambda _query: [0.1, 0.2])

    results = searcher.search("question", scope=SCOPE)

    assert [result["chunk_id"] for result in results] == ["a"]
    condition = searcher.qdrant.query_filter.must[0]
    assert condition.key == "dataset_id"
    assert condition.match.any == ["team_a"]


class _SparseQdrant:
    def __init__(self):
        self.query_filter = None

    def search_sparse(self, **kwargs):
        self.query_filter = kwargs["query_filter"]
        # Simulate a backend/plugin that accidentally returns an extra point;
        # SparseSearcher must still apply the same defense-in-depth scope.
        return [
            SimpleNamespace(
                id="a1",
                payload={"content": "alpha policy", "dataset_id": "team_a"},
            ),
            SimpleNamespace(
                id="a2",
                payload={"content": "beta policy", "dataset_id": "team_a"},
            ),
            SimpleNamespace(
                id="b1",
                payload={"content": "alpha policy", "dataset_id": "team_b"},
            ),
            SimpleNamespace(
                id="legacy",
                payload={"content": "alpha policy"},
            ),
        ]


def test_sparse_index_is_scoped_and_legacy_points_are_excluded():
    invalidate_bm25_index()
    searcher = SparseSearcher.__new__(SparseSearcher)
    searcher.qdrant = _SparseQdrant()

    results = searcher.search("alpha", top_k=10, scope=SCOPE)

    assert all(result["dataset_id"] == "team_a" for result in results)
    assert {result["chunk_id"] for result in results} <= {"a1", "a2"}
    assert searcher.qdrant.query_filter.must[0].key == "dataset_id"


class _ParentQdrant:
    def get_by_ids(self, _collection, ids, *, query_filter=None):
        assert query_filter.must[0].key == "dataset_id"
        payloads = {
            "p-a": {"content": "allowed parent", "dataset_id": "team_a"},
            "p-b": {"content": "forbidden parent", "dataset_id": "team_b"},
        }
        return [
            SimpleNamespace(id=parent_id, payload=payloads[parent_id])
            for parent_id in ids
            if parent_id in payloads
        ]


def test_parent_resolution_rejects_out_of_scope_and_legacy_children():
    resolver = ParentResolver()
    resolver.qdrant = _ParentQdrant()
    children = [
        {
            "chunk_id": "c-a",
            "content": "allowed child",
            "parent_id": "p-a",
            "dataset_id": "team_a",
        },
        {
            "chunk_id": "c-b",
            "content": "forbidden child",
            "parent_id": "p-b",
            "dataset_id": "team_b",
        },
        {"chunk_id": "legacy", "content": "legacy child", "parent_id": None},
    ]

    results = resolver.resolve(children, scope=SCOPE)

    assert len(results) == 1
    assert results[0]["content"] == "allowed parent"
