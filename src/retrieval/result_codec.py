"""JSON-safe codec shared by API, worker and evaluation paths."""

from __future__ import annotations

from typing import Any


def result_to_payload(value: Any) -> dict[str, Any]:
    """Serialize an ``RAGResult`` or compatible object without raw errors."""
    if isinstance(value, dict):
        source_values = []
        for source in value.get("sources", []) or []:
            source_values.append(source.as_dict() if hasattr(source, "as_dict") else dict(source))
        return {
            "answer": str(value.get("answer", "")),
            "contexts": list(value.get("contexts", []) or []),
            "sources": source_values,
            "expanded_queries": list(value.get("expanded_queries", []) or []),
            "num_candidates": int(value.get("num_candidates", 0)),
        }
    sources = []
    for source in getattr(value, "sources", []) or []:
        sources.append(source.as_dict() if hasattr(source, "as_dict") else dict(source))
    return {
        "answer": str(getattr(value, "answer", "")),
        "contexts": list(getattr(value, "contexts", []) or []),
        "sources": sources,
        "expanded_queries": list(getattr(value, "expanded_queries", []) or []),
        "num_candidates": int(getattr(value, "num_candidates", 0)),
    }


def result_from_payload(payload: Any):
    """Restore a payload into the canonical ``RAGResult`` type."""
    from retrieval.context.assembler import SourceRef
    from retrieval.retriever import RAGResult

    if isinstance(payload, RAGResult):
        return payload
    data = payload if isinstance(payload, dict) else {}
    sources = []
    for source in data.get("sources", []) or []:
        if isinstance(source, SourceRef):
            sources.append(source)
        elif isinstance(source, dict):
            sources.append(SourceRef(**source))
    return RAGResult(
        answer=str(data.get("answer", "")),
        contexts=[str(item) for item in data.get("contexts", []) or []],
        sources=sources,
        expanded_queries=[str(item) for item in data.get("expanded_queries", []) or []],
        num_candidates=int(data.get("num_candidates", 0)),
    )
