from __future__ import annotations

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

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from core import get_logger
from core.config import settings

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load 1 lần duy nhất cho cả process (lru_cache ở module level).

    Fix bug P1-1: class attribute + self._model = ... tạo instance attribute
    che class attribute → mỗi EmbeddingService() load lại model từ đầu.
    """
    logger.info("Loading embedding model", model=settings.EMBEDDING_MODEL_ID)
    model = SentenceTransformer(
        settings.EMBEDDING_MODEL_ID,
        device=settings.EMBEDDING_DEVICE,
    )
    logger.info("Embedding model loaded", dimensions=settings.EMBEDDING_SIZE)
    return model


class EmbeddingService:
    """Wrapper mỏng quanh model đã cache ở module level.

    Tạo bao nhiêu instance cũng chỉ load model 1 lần (cache nằm ở _load_model).
    """

    @property
    def model(self) -> SentenceTransformer:
        return _load_model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed danh sách text → danh sách vectors.

        Args:
            texts: Danh sách strings cần embed

        Returns:
            List of vectors, mỗi vector có EMBEDDING_SIZE dimensions (384)
        """
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,   # model BGE train/benchmark với vector đã normalize
            batch_size=32,
        )
        return vectors.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Embed 1 text duy nhất → 1 vector."""
        return self.embed([text])[0]
