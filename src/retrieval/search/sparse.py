"""Qdrant-native BM25 helpers.

Sparse text is indexed by Qdrant during ingestion and searched on the server.
The API process therefore never scrolls, tokenizes, or scores the full corpus.
"""

from __future__ import annotations

import re
import unicodedata

from qdrant_client.models import (
    Bm25Config,
    Document,
    TokenizerType,
)

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from retrieval.scope import RetrievalScope, default_scope

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Compatibility/debug tokenizer; online ranking is performed by Qdrant."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_RE.findall(normalized)


def _tokenizer() -> TokenizerType:
    """Resolve an explicitly configured Qdrant tokenizer."""
    try:
        return TokenizerType(settings.QDRANT_SPARSE_TOKENIZER.lower())
    except ValueError as exc:
        raise ValueError(
            f"Unsupported QDRANT_SPARSE_TOKENIZER: {settings.QDRANT_SPARSE_TOKENIZER}"
        ) from exc


def bm25_config() -> Bm25Config:
    """Return one canonical BM25 configuration for indexing and querying."""
    language = settings.QDRANT_SPARSE_LANGUAGE.strip().lower()
    # Qdrant 1.18.x uses the legacy ``language=none`` switch to disable the
    # default English stemmer and stop-word list.  The explicit
    # ``stemmer={type:none}`` representation was introduced in Qdrant 1.19 and
    # is rejected by the Docker version pinned by this repository.  Keep this
    # wire contract aligned with the deployed server and include it in the
    # ingestion fingerprint so a future 1.19 migration cannot mix vectors.
    return Bm25Config(
        k=settings.QDRANT_SPARSE_K,
        b=settings.QDRANT_SPARSE_B,
        avg_len=settings.QDRANT_SPARSE_AVG_LEN,
        tokenizer=_tokenizer(),
        language=language or None,
        lowercase=True,
        ascii_folding=False,
    )


def sparse_document(text: str) -> Document:
    """Build the native sparse-inference document used by Qdrant core."""
    return Document(
        text=text,
        model=settings.QDRANT_SPARSE_MODEL,
        options=bm25_config(),
    )


def format_results(raw_results: list, *, scope: RetrievalScope) -> list[dict]:
    """Convert Qdrant points and reject any out-of-scope backend result."""
    formatted: list[dict] = []
    for result in raw_results:
        payload = result.payload or {}
        if not scope.allows(payload.get("dataset_id")):
            continue
        formatted.append({
            "chunk_id": result.id,
            "content": payload.get("content", ""),
            "score": getattr(result, "score", 0.0),
            "parent_id": payload.get("parent_id"),
            "file_name": payload.get("file_name", ""),
            "source_uri": payload.get("source_uri") or payload.get("source_path", ""),
            "section_title": payload.get("section_title", ""),
            "page_number": payload.get("page_number"),
            "dataset_id": payload.get("dataset_id"),
            "generation_id": payload.get("generation_id"),
            "source": "sparse",
        })
    return formatted


def invalidate_bm25_index() -> None:
    """Compatibility no-op: native Qdrant indexes are visible after commit."""
    logger.debug("Native Qdrant sparse index requires no process-local invalidation")


class SparseSearcher:
    """Search Qdrant's native BM25 sparse index with the retrieval scope."""

    def __init__(self):
        self.qdrant = QdrantConnector()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        scope: RetrievalScope | None = None,
        collection_name: str | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K
        scope = scope or default_scope()
        if not query.strip():
            return []
        points = self.qdrant.search_sparse(
            collection_name=collection_name or settings.CHILD_COLLECTION,
            query=sparse_document(query),
            sparse_vector_name=settings.QDRANT_SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=scope.qdrant_filter(),
        )
        results = format_results(points, scope=scope)
        logger.info("Sparse search done", results=len(results))
        logger.info("Retrieval metric", retrieval_mode="sparse_only", result_count=len(results))
        return results
