"""
Dense Search — Tìm kiếm bằng vector similarity (cosine).

Cách hoạt động:
  1. Embed query → vector 384d
  2. Tìm trong Qdrant: vectors nào gần nhất (cosine similarity)
  3. Trả về top-K kết quả

Ưu điểm: Hiểu ngữ NGHĨA (synonym, paraphrase)
  "xe hơi" ≈ "ô tô" ≈ "automobile" → tìm được

Nhược điểm: Kém với từ khóa CHÍNH XÁC
  "TC-456" (mã ticket) → dense search không hiểu, coi như random text
"""

from __future__ import annotations

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from ingestion.embeddings import EmbeddingService
from retrieval.scope import RetrievalScope, default_scope

logger = get_logger(__name__)


class DenseSearcher:
    """Tìm kiếm bằng vector cosine similarity qua Qdrant."""

    def __init__(self):
        self.qdrant = QdrantConnector()
        self.embedder = EmbeddingService()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[dict]:
        """Search bằng query text.

        Args:
            query: Câu hỏi của user
            top_k: Số kết quả trả về (default: settings.TOP_K)

        Returns:
            List[dict] — mỗi dict có: chunk_id, content, score, metadata
        """
        top_k = top_k or settings.TOP_K
        scope = scope or default_scope()

        # Embed query → vector
        query_vector = self.embedder.embed_single(query)

        # Search trong Qdrant
        results = self.qdrant.search(
            collection_name=settings.CHILD_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=scope.qdrant_filter(),
        )

        return self._format_results(results, scope=scope)

    def search_by_vector(
        self,
        vector: list[float],
        top_k: int | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[dict]:
        """Search bằng vector có sẵn (dùng cho HyDE — đã embed sẵn).

        Args:
            vector: Vector embedding 384d
            top_k: Số kết quả trả về
        """
        top_k = top_k or settings.TOP_K
        scope = scope or default_scope()

        results = self.qdrant.search(
            collection_name=settings.CHILD_COLLECTION,
            query_vector=vector,
            limit=top_k,
            query_filter=scope.qdrant_filter(),
        )

        return self._format_results(results, scope=scope)

    def _format_results(
        self,
        raw_results,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[dict]:
        """Chuyển Qdrant results → dạng dict chuẩn."""
        scope = scope or default_scope()
        formatted = []
        for r in raw_results:
            payload = r.payload or {}
            if not scope.allows(payload.get("dataset_id")):
                continue
            formatted.append({
                "chunk_id": r.id,
                "content": payload.get("content", ""),
                "score": r.score,
                "parent_id": payload.get("parent_id"),
                "file_name": payload.get("file_name", ""),
                "section_title": payload.get("section_title", ""),
                "page_number": payload.get("page_number"),
                "dataset_id": payload.get("dataset_id"),
                "source": "dense",  # Đánh dấu nguồn tìm kiếm
            })
        return formatted
