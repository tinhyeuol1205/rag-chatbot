"""
Embedding Service — Chuyển text thành vector (384 dimensions).

Model: BAAI/bge-small-en-v1.5
- Chạy LOCAL (không tốn API tiền)
- 384 dimensions (nhỏ, nhanh, phù hợp demo)
- Chất lượng tốt trên MTEB benchmark

Tại sao không dùng OpenAI Embeddings?
- OpenAI tốn tiền cho mỗi lần embed
- bge-small-en chạy local, miễn phí, đủ chất lượng
- Trong production thật có thể switch sang OpenAI text-embedding-3-small

Usage:
    from ingestion.embeddings import EmbeddingService
    service = EmbeddingService()
    vectors = service.embed(["Hello world", "Another text"])
    # vectors.shape = (2, 384)
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterable, Iterator
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from core import get_logger
from core.config import settings

logger = get_logger(__name__)

# Explicit global + lock thay vì lru_cache: dễ chứng minh single-flight hơn
# (bug P2-14 — lru_cache cho phép wrapped function chạy lại nếu 2 thread cùng
# miss cache; sentence-transformer load trùng model → RAM tăng gấp đôi/OOM).
_embedding_model: SentenceTransformer | None = None
_embedding_model_lock = Lock()


def _load_model() -> SentenceTransformer:
    """Load 1 lần duy nhất cho cả process. Thread-safe (lock + double-check)."""
    global _embedding_model
    if _embedding_model is None:            # fast path — không cần lock
        with _embedding_model_lock:         # slow path — lấy lock
            if _embedding_model is None:    # double-check LẠI sau khi có lock
                logger.info("Loading embedding model", model=settings.EMBEDDING_MODEL_ID)
                kwargs = {"device": settings.EMBEDDING_DEVICE}
                if settings.INGEST_EMBEDDING_MODEL_REVISION:
                    kwargs["revision"] = settings.INGEST_EMBEDDING_MODEL_REVISION
                model = SentenceTransformer(settings.EMBEDDING_MODEL_ID, **kwargs)
                dimensions = getattr(model, "get_sentence_embedding_dimension", lambda: settings.EMBEDDING_SIZE)()
                if dimensions is not None and dimensions != settings.EMBEDDING_SIZE:
                    raise ValueError(
                        "Embedding model dimension does not match EMBEDDING_SIZE: "
                        f"model={dimensions}, configured={settings.EMBEDDING_SIZE}"
                    )
                _embedding_model = model    # publish chỉ sau load + validation thành công
                logger.info("Embedding model loaded", dimensions=dimensions or settings.EMBEDDING_SIZE)
    return _embedding_model


class EmbeddingService:
    """Wrapper mỏng quanh model đã cache ở module level.

    Tạo bao nhiêu instance cũng chỉ load model 1 lần (cache nằm ở _load_model).
    """

    @property
    def model(self) -> SentenceTransformer:
        return _load_model()

    @property
    def dimension(self) -> int:
        """Resolve the model dimension before a collection can be mutated."""
        dimension_fn = getattr(self.model, "get_sentence_embedding_dimension", None)
        if dimension_fn is None:
            return settings.EMBEDDING_SIZE
        dimension = dimension_fn()
        if dimension is None:
            raise ValueError("Embedding model did not expose a vector dimension")
        if dimension != settings.EMBEDDING_SIZE:
            raise ValueError(
                "Embedding model dimension does not match EMBEDDING_SIZE: "
                f"model={dimension}, configured={settings.EMBEDDING_SIZE}"
            )
        return int(dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed danh sách text → danh sách vectors.

        Args:
            texts: Danh sách strings cần embed

        Returns:
            List of vectors, mỗi vector có EMBEDDING_SIZE dimensions (384)
        """
        if not texts:
            return []
        vectors = self._encode_with_retry(texts)
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(
                "Embedding output has unexpected shape: "
                f"got={array.shape}, expected=(*, {self.dimension})"
            )
        return array.tolist()

    def embed_batches(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Embed an iterable in bounded float32 batches.

        The iterable is consumed incrementally; callers can serialize or upsert
        each returned array before asking for the next batch.
        """
        size = batch_size or settings.INGEST_EMBED_BATCH_SIZE
        if size <= 0:
            raise ValueError("embedding batch_size must be positive")
        pending: list[str] = []
        for text in texts:
            pending.append(text)
            if len(pending) >= size:
                yield self._encode_batch(pending)
                pending = []
        if pending:
            yield self._encode_batch(pending)

    def embed_single(self, text: str) -> list[float]:
        """Embed 1 text duy nhất → 1 vector."""
        return self.embed([text])[0]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        vectors = self._encode_with_retry(texts)
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(
                "Embedding batch has unexpected shape: "
                f"got={array.shape}, expected=(*, {self.dimension})"
            )
        return array

    def _encode_with_retry(self, texts: list[str]):
        """Retry only transient encoder/transport failures for an idempotent batch."""
        attempts = max(settings.INGEST_QDRANT_MAX_RETRIES, 0) + 1
        for attempt in range(attempts):
            try:
                return self.model.encode(
                    texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=min(len(texts), settings.INGEST_EMBED_BATCH_SIZE),
                    convert_to_numpy=True,
                )
            except Exception as exc:
                name = type(exc).__name__.lower()
                retryable = isinstance(exc, (TimeoutError, ConnectionError)) or "timeout" in name or "connection" in name
                if attempt >= attempts - 1 or not retryable:
                    raise
                delay = min(
                    settings.INGEST_RETRY_MAX_SECONDS,
                    settings.INGEST_RETRY_BASE_SECONDS * (2**attempt),
                )
                delay *= 0.8 + random.random() * 0.4
                logger.warning(
                    "Transient embedding failure; retrying",
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    delay_seconds=round(delay, 3),
                    error_type=type(exc).__name__,
                )
                time.sleep(delay)
