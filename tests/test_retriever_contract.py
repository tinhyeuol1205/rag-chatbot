"""Tests for the structured retrieval/context contract."""

from retrieval.context.assembler import ContextAssembler
from retrieval.retriever import NO_CONTEXT_MSG, RAGRetriever, RetrievalResult


def _retriever_with_result(assembled):
    retriever = RAGRetriever.__new__(RAGRetriever)
    result = RetrievalResult(
        assembled=assembled,
        expanded_queries=["q"],
        search_query="q",
        num_reranked_candidates=len(assembled.documents),
    )
    retriever.retrieve = lambda _query, history=None: result
    retriever._generate = lambda query, context, sources: "answer"
    retriever._generate_stream = lambda query, context, sources: iter(["answer"])
    return retriever


def test_evaluation_contexts_match_prompt_order():
    docs = [
        {"content": str(i), "file_name": f"f{i}.md"}
        for i in range(1, 6)
    ]
    assembled = ContextAssembler().assemble(docs)
    retriever = _retriever_with_result(assembled)

    result = retriever.query_with_context("q")

    assert result.contexts == ["1", "3", "5", "4", "2"]
    assert result.sources[0].citation_id == 1


def test_stream_emits_tokens_then_sources():
    docs = [{"content": "context", "file_name": "policy.md"}]
    assembled = ContextAssembler().assemble(docs)
    retriever = _retriever_with_result(assembled)

    events = list(retriever.stream_with_sources("q"))

    assert events[0] == {"event": "token", "data": "answer"}
    assert events[1] == {
        "event": "sources",
        "data": [{
            "citation_id": 1,
            "file_name": "policy.md",
            "section_title": None,
            "page_number": None,
        }],
    }


def test_empty_context_still_emits_empty_sources():
    retriever = _retriever_with_result(ContextAssembler().assemble([]))

    events = list(retriever.stream_with_sources("q"))

    assert events == [
        {"event": "token", "data": NO_CONTEXT_MSG},
        {"event": "sources", "data": []},
    ]
