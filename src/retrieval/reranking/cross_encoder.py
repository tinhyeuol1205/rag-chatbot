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

from __future__ import annotations

from threading import Lock

import httpx
from sentence_transformers import CrossEncoder

from core import get_logger
from core.config import settings

logger = get_logger(__name__)

# CrossEncoder không thread-safe khi predict song song → cần lock (liên quan P0-3)
_predict_lock = Lock()

# Explicit global + lock thay vì lru_cache — same lý do embeddings.py (P2-14):
# load trùng model khi cold-start đồng thời → RAM nhân đôi/OOM.
_reranker_model: CrossEncoder | None = None
_reranker_model_lock = Lock()


def _load_reranker() -> CrossEncoder:
    """Load 1 lần duy nhất cho cả process. Thread-safe (lock + double-check)."""
    global _reranker_model
    if _reranker_model is None:             # fast path — không cần lock
        with _reranker_model_lock:          # slow path — lấy lock
            if _reranker_model is None:     # double-check LẠI sau khi có lock
                logger.info("Loading reranker model", model=settings.RERANKER_MODEL_ID)
                kwargs = {
                    "device": settings.RERANKER_DEVICE or settings.EMBEDDING_DEVICE,
                    "max_length": 512,
                }
                if settings.RERANKER_MODEL_REVISION:
                    kwargs["revision"] = settings.RERANKER_MODEL_REVISION
                model = CrossEncoder(
                    settings.RERANKER_MODEL_ID,
                    **kwargs,
                )
                _reranker_model = model     # publish chỉ sau load thành công
                logger.info("Reranker loaded")
    return _reranker_model


class CrossEncoderReranker:
    """Rerank through a GPU service in production or local model in dev."""

    @property
    def is_remote(self) -> bool:
        return settings.RERANKER_RUNTIME == "remote"

    @property
    def model(self) -> CrossEncoder:
        if self.is_remote:
            raise RuntimeError("Remote reranker runtime does not expose an in-process model")
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

        if self.is_remote:
            scores = self._remote_predict(query, [doc["content"] for doc in candidates], priority="online")
        else:
            # Local adapter only. GPU service concurrency is owned by the
            # inference deployment, not by an API-process limiter.
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

    @staticmethod
    def _remote_predict(query: str, documents: list[str], *, priority: str = "online") -> list[float]:
        url = settings.RERANKER_BASE_URL.rstrip("/") + "/rerank"
        payload = {
            "model": settings.RERANKER_MODEL_ID,
            "revision": settings.RERANKER_MODEL_REVISION,
            "query": query,
            "documents": documents,
            "priority": priority if priority in {"online", "batch"} else "online",
        }
        headers = {}
        if settings.MODEL_SERVER_API_KEY:
            headers["X-API-Key"] = settings.MODEL_SERVER_API_KEY
        try:
            request_kwargs = {"json": payload, "timeout": settings.RERANKER_HTTP_TIMEOUT_SECONDS}
            if headers:
                request_kwargs["headers"] = headers
            response = httpx.post(url, **request_kwargs)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            logger.warning("Remote reranker service failed", error_type=type(exc).__name__)
            raise ConnectionError("Reranker service unavailable") from exc
        values = body.get("scores") if isinstance(body, dict) else body
        if not isinstance(values, list) or len(values) != len(documents):
            raise ValueError("Reranker service returned invalid score cardinality")
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("Reranker service returned non-numeric scores") from exc
