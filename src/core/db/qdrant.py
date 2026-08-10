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

from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class QdrantConnector:
    """Singleton connector cho Qdrant vector database (thread-safe)."""

    _instance: "QdrantConnector | None" = None
    _client: QdrantClient | None = None
    _lock = Lock()

    def __new__(cls) -> "QdrantConnector":
        """Singleton thread-safe: chỉ tạo instance mới nếu chưa tồn tại."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> QdrantClient:
        """Lazy init thread-safe: chỉ tạo connection khi thực sự cần dùng."""
        if self._client is None:            # fast path — không cần lock
            with self._lock:                # slow path — lấy lock
                if self._client is None:    # double-check LẠI sau khi có lock
                    self._client = QdrantClient(
                        host=settings.QDRANT_HOST,
                        port=settings.QDRANT_PORT,
                        timeout=30,         # ★ mặc định có thể treo rất lâu
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

    def delete_by_file_name(self, collection_name: str, file_name: str) -> None:
        """Xoá mọi point của 1 file — gọi TRƯỚC khi ingest lại file đó.

        Chống chunk rác vĩnh viễn (bug P1-6): content đổi → chunk_id đổi →
        point cũ không ai xoá. Fix: xoá toàn bộ point theo file_name trước khi
        ghi lại file.
        """
        if not self._collection_exists(collection_name):
            return
        self.client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(key="file_name", match=MatchValue(value=file_name))
                ])
            ),
            wait=True,
        )
        logger.info("Deleted existing points for file",
                    collection=collection_name, file=file_name)

    def create_payload_index(self, collection_name: str, field_name: str) -> None:
        """Index cho payload field — bắt buộc để filter/delete-by-filter chạy nhanh."""
        from qdrant_client.models import PayloadSchemaType
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Created payload index", collection=collection_name, field=field_name)
        except Exception as e:      # index đã tồn tại → bỏ qua
            logger.debug("Payload index skipped", field=field_name, error=str(e))

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

    def scroll_all(self, collection_name: str, batch_size: int = 1000,
                   max_points: int | None = None) -> list:
        """Đọc TẤT CẢ points trong collection (phân trang đúng cách).

        Fix bug P1-9: bản cũ chỉ đọc trang đầu (limit=10000) → mọi point sau
        #10000 biến mất khỏi BM25 index im lặng, không log, không lỗi.
        """
        all_points, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if offset is None:
                break
            if max_points is not None and len(all_points) >= max_points:
                logger.warning("scroll_all hit max_points cap — dữ liệu bị cắt",
                               collection=collection_name, cap=max_points,
                               fetched=len(all_points))
                break
        logger.info("Scrolled collection", collection=collection_name, total=len(all_points))
        return all_points

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
