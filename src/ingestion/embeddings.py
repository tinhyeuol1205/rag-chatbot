"""Embedding service for BGE-M3 local/dev and remote GPU production modes."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable, Iterator
from threading import Lock

import httpx
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
                revision = settings.EMBEDDING_MODEL_REVISION or settings.INGEST_EMBEDDING_MODEL_REVISION
                if revision:
                    kwargs["revision"] = revision
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
    """One contract for ingestion and online query embedding.

    Production uses an HTTP GPU service so API/worker replicas do not each load
    BGE-M3 weights.  The local SentenceTransformer adapter remains available for
    development and deterministic unit tests.
    """

    @property
    def is_remote(self) -> bool:
        return settings.EMBEDDING_RUNTIME == "remote"

    @property
    def model(self) -> SentenceTransformer:
        if self.is_remote:
            raise RuntimeError("Remote embedding runtime does not expose an in-process model")
        return _load_model()

    @property
    def dimension(self) -> int:
        """Resolve the model dimension before a collection can be mutated."""
        if self.is_remote:
            return settings.EMBEDDING_SIZE
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

    def embed(self, texts: list[str], *, priority: str = "online") -> list[list[float]]:
        """Embed danh sách text → danh sách vectors.

        Args:
            texts: Danh sách strings cần embed

        Returns:
            List of vectors, mỗi vector có EMBEDDING_SIZE dimensions (1024 for BGE-M3)
        """
        if not texts:
            return []
        if self.is_remote:
            return self._embed_remote(texts, priority=priority)
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
                yield self._encode_batch(pending, priority="batch")
                pending = []
        if pending:
            yield self._encode_batch(pending, priority="batch")

    def embed_single(self, text: str) -> list[float]:
        """Embed 1 text duy nhất → 1 vector."""
        return self.embed([text])[0]

    def _encode_batch(self, texts: list[str], *, priority: str = "batch") -> np.ndarray:
        if self.is_remote:
            return np.asarray(self._embed_remote(texts, priority=priority), dtype=np.float32)
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
                    normalize_embeddings=settings.EMBEDDING_NORMALIZE,
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

    def _embed_remote(self, texts: list[str], *, priority: str = "online") -> list[list[float]]:
        """Call the pinned GPU embedding service contract.

        Accepted response shapes are the common TEI-style list of vectors and
        ``{"embeddings": [...]}``; anything else fails closed before Qdrant.
        """
        url = settings.EMBEDDING_BASE_URL.rstrip("/") + "/embed"
        payload = {
            "inputs": texts,
            "model": settings.EMBEDDING_MODEL_ID,
            "revision": settings.EMBEDDING_MODEL_REVISION or settings.INGEST_EMBEDDING_MODEL_REVISION,
            "normalize": settings.EMBEDDING_NORMALIZE,
            "priority": priority if priority in {"online", "batch"} else "online",
        }
        headers = {}
        if settings.MODEL_SERVER_API_KEY:
            headers["X-API-Key"] = settings.MODEL_SERVER_API_KEY
        try:
            request_kwargs = {"json": payload, "timeout": settings.EMBEDDING_HTTP_TIMEOUT_SECONDS}
            if headers:
                request_kwargs["headers"] = headers
            response = httpx.post(url, **request_kwargs)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            logger.warning("Remote embedding service failed", error_type=type(exc).__name__)
            raise ConnectionError("Embedding service unavailable") from exc
        vectors = body.get("embeddings") if isinstance(body, dict) else body
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("Embedding service returned an invalid batch cardinality")
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != settings.EMBEDDING_SIZE:
            raise ValueError(
                "Embedding service returned an unexpected dimension: "
                f"got={array.shape}, expected=(*, {settings.EMBEDDING_SIZE})"
            )
        return array.tolist()
