from __future__ import annotations

"""
Cross-Encoder Reranking ★ — Kỹ thuật RAG #2.

Retrieval 2 giai đoạn (Two-Stage Retrieval):

  Giai đoạn 1: Hybrid Search → Top-20 (nhanh, nhưng chưa chính xác lắm)
  Giai đoạn 2: Reranker      → Top-5  (chậm hơn, nhưng CỰC KỲ chính xác)

Sự khác biệt giữa Bi-Encoder (Embedding) vs Cross-Encoder (Reranker):

  Bi-Encoder (dùng cho search):
    Query  → [Encoder] → vector_q ─┐
                                     ├─ cosine(v_q, v_d) = 0.82
    Document → [Encoder] → vector_d ─┘
    ✅ Nhanh (embed độc lập, so sánh nhanh)
    ❌ Bỏ qua tương tác từ-nối-từ giữa query và document

  Cross-Encoder (dùng cho rerank):
    [Query, Document] → [Transformer cùng lúc] → relevance_score = 0.95
    ❌ Chậm (phải xử lý TỪNG CẶP query-document)
    ✅ Chính xác cực cao (thấy tương tác giữa mọi từ)

  → Kết hợp: Bi-Encoder lọc 20 ứng viên, Cross-Encoder chọn 5 tốt nhất.

Model: BAAI/bge-reranker-v2-m3 — top đầu MTEB reranking benchmark

Tham khảo: rag_master.md — Module 5, mục 5.1
"""

from functools import lru_cache
from threading import Lock

from sentence_transformers import CrossEncoder

from core import get_logger
from core.config import settings

logger = get_logger(__name__)

# CrossEncoder không thread-safe khi predict song song → cần lock (liên quan P0-3)
_predict_lock = Lock()


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    """Load 1 lần duy nhất cho cả process (lru_cache module-level)."""
    logger.info("Loading reranker model", model=settings.RERANKER_MODEL_ID)
    model = CrossEncoder(
        settings.RERANKER_MODEL_ID,
        device=settings.EMBEDDING_DEVICE,   # ★ config này đang bị bỏ qua hoàn toàn
        max_length=512,                     # ★ chặn input dài → chậm bất định
    )
    logger.info("Reranker loaded")
    return model


class CrossEncoderReranker:
    """Rerank kết quả search bằng Cross-Encoder model."""

    @property
    def model(self) -> CrossEncoder:
        return _load_reranker()             # ★ cache module-level, không reload

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        """Rerank danh sách documents bằng Cross-Encoder.

        Args:
            query: Câu hỏi của user
            documents: Danh sách kết quả từ Hybrid Search

        Returns:
            Top KEEP_TOP_K documents, sắp xếp theo relevance score mới
        """
        if not documents:
            return []

        # ★ Cap số candidate — documents đã sort theo RRF nên cắt là an toàn
        candidates = documents[:settings.RERANK_CANDIDATES]
        if len(documents) > len(candidates):
            logger.info("Capped rerank candidates",
                        total=len(documents), kept=len(candidates))

        # Tạo pairs: [(query, doc_content), (query, doc_content), ...]
        pairs = [(query, doc["content"]) for doc in candidates]

        # Cross-Encoder scoring — chấm điểm từng cặp (lock vì không thread-safe)
        with _predict_lock:
            scores = self.model.predict(pairs, batch_size=settings.RERANK_BATCH_SIZE)

        # ★ KHÔNG mutate list/dict của caller (bug P3-4):
        # copy sang dict mới rồi mới sort, input ban đầu giữ nguyên
        scored = [{**doc, "rerank_score": float(s)} for doc, s in zip(candidates, scores)]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)

        # Giữ lại top KEEP_TOP_K
        top_docs = scored[:settings.KEEP_TOP_K]

        logger.info(
            "Reranking done",
            input_count=len(candidates),
            output_count=len(top_docs),
            top_score=round(top_docs[0]["rerank_score"], 4) if top_docs else 0,
        )

        return top_docs

