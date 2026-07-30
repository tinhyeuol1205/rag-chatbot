from __future__ import annotations

"""
Qdrant Vector Database Connector — Singleton Pattern.

Pattern: Singleton (giống llm-twin-course/src/core/db/qdrant.py)
- Dù gọi QdrantConnector() 100 lần, chỉ tạo 1 connection duy nhất
- Tránh tạo quá nhiều connections → crash database

Cách hoạt động Singleton:
    connector_a = QdrantConnector()
    connector_b = QdrantConnector()
    assert connector_a is connector_b  # True! Cùng 1 object

Usage:
    from core.db import QdrantConnector
    qdrant = QdrantConnector()
    qdrant.search(collection_name="child_chunks", query_vector=[...], limit=5)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class QdrantConnector:
    """Singleton connector cho Qdrant vector database."""

    _instance: "QdrantConnector | None" = None
    _client: QdrantClient | None = None

    def __new__(cls) -> "QdrantConnector":
        """Singleton: chỉ tạo instance mới nếu chưa tồn tại."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> QdrantClient:
        """Lazy init: chỉ tạo connection khi thực sự cần dùng."""
        if self._client is None:
            self._client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
            logger.info(
                "Connected to Qdrant",
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
        return self._client

    # ----- Collection Management -----

    def create_vector_collection(self, collection_name: str) -> None:
        """Tạo collection CÓ vector index (cho child chunks — dùng để search)."""
        if self._collection_exists(collection_name):
            logger.info("Collection already exists", collection=collection_name)
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_SIZE,  # 384 dimensions
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created vector collection", collection=collection_name)

    def create_payload_collection(self, collection_name: str) -> None:
        """Tạo collection KHÔNG CÓ vector (cho parent chunks — chỉ lưu text)."""
        if self._collection_exists(collection_name):
            logger.info("Collection already exists", collection=collection_name)
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={},  # Không có vectors
        )
        logger.info("Created payload-only collection", collection=collection_name)

    # ----- Write Operations -----

    def upsert_points(self, collection_name: str, points: list[PointStruct]) -> None:
        """Insert hoặc update points vào collection."""
        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )
        logger.info("Upserted points", collection=collection_name, count=len(points))

    # ----- Read Operations -----

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
    ) -> list:
        """Tìm kiếm vector tương đồng (cosine similarity).

        qdrant-client v1.18+: dùng query_points() thay vì search().
        """
        from qdrant_client.models import models

        result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )
        return result.points

    def scroll_all(self, collection_name: str, limit: int = 10000) -> list:
        """Đọc tất cả points trong collection (phân trang)."""
        points, _ = self.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return points

    def get_by_ids(self, collection_name: str, ids: list[str]) -> list:
        """Lấy points theo danh sách IDs (dùng retrieve tiêu chuẩn)."""
        return self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )

    # ----- Helpers -----

    def _collection_exists(self, name: str) -> bool:
        collections = [c.name for c in self.client.get_collections().collections]
        return name in collections

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Qdrant connection closed")
