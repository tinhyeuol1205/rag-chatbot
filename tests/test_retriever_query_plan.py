"""Regression tests for the retrieval query plan and RRF vote accounting."""

from retrieval.context.assembler import AssembledContext
from retrieval.retriever import RAGRetriever
from retrieval.scope import RetrievalScope


class _Condenser:
    def condense(self, query, _history):
        return query


class _Expander:
    def __init__(self, variants):
        self.variants = variants

    def expand(self, _query):
        return self.variants


class _Hyde:
    def __init__(self, vector):
        self.vector = vector

    def generate_embedding(self, _query):
        return self.vector


class _Searcher:
    def __init__(self):
        self.calls = []

    def search(self, query, *, scope, hyde_vector=None, include_sparse=True):
        self.calls.append({
            "query": query,
            "scope": scope,
            "hyde_vector": hyde_vector,
            "include_sparse": include_sparse,
        })
        channel = "hyde" if hyde_vector is not None else "hybrid"
        return [{
            "chunk_id": f"{channel}:{query}",
            "content": query,
            "parent_id": None,
            "dataset_id": scope.dataset_ids[0],
        }]


class _Reranker:
    def rerank(self, _query, documents):
        return documents


class _ParentResolver:
    def resolve(self, documents, *, scope):
        assert all(scope.allows(doc.get("dataset_id")) for doc in documents)
        return documents


class _Assembler:
    def assemble(self, _documents):
        return AssembledContext()


def _retriever(expander, hyde):
    retriever = RAGRetriever.__new__(RAGRetriever)
    retriever.scope = RetrievalScope(("sample_docs",))
    retriever.condenser = _Condenser()
    retriever.expander = expander
    retriever.hyde = hyde
    retriever.searcher = _Searcher()
    retriever.reranker = _Reranker()
    retriever.parent_resolver = _ParentResolver()
    retriever.assembler = _Assembler()
    return retriever


def test_original_query_is_not_searched_twice_when_hyde_is_enabled():
    retriever = _retriever(
        _Expander(["original question", "Alternative wording"]),
        _Hyde([0.1, 0.2]),
    )

    result = retriever.retrieve("original question")

    calls = retriever.searcher.calls
    assert [(call["query"], call["include_sparse"]) for call in calls] == [
        ("original question", True),
        ("original question", False),
        ("Alternative wording", True),
    ]
    assert result.expanded_queries == ["original question", "Alternative wording"]


def test_query_variants_are_deduplicated_using_unicode_normalization():
    retriever = _retriever(
        _Expander(["ＡＢＣ  policy", "abc policy", "Useful variant"]),
        _Hyde(None),
    )

    result = retriever.retrieve("ＡＢＣ policy")

    assert [call["query"] for call in retriever.searcher.calls] == [
        "ＡＢＣ policy",
        "Useful variant",
    ]
    assert result.expanded_queries == ["ＡＢＣ policy", "Useful variant"]


def test_expander_fallback_to_original_creates_one_hybrid_vote():
    retriever = _retriever(_Expander(["original question"]), _Hyde(None))

    retriever.retrieve("original question")

    assert len(retriever.searcher.calls) == 1
    assert retriever.searcher.calls[0]["include_sparse"] is True
