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

from __future__ import annotations

import random
import re
import time
import unicodedata
from collections.abc import Callable, Iterator
from hashlib import blake2b
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    Document,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    HasVectorCondition,
    MatchValue,
    Modifier,
    PointIdsList,
    PointStruct,
    Prefetch,
    Rrf,
    RrfQuery,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from typing_extensions import Self

from core.config import settings
from core.logger import get_logger
from ingestion.batching import approximate_size, iter_batches

logger = get_logger(__name__)


class QdrantConnector:
    """Singleton connector cho Qdrant vector database (thread-safe)."""

    _instance: QdrantConnector | None = None
    _client: QdrantClient | None = None
    _lock = Lock()

    def __new__(cls) -> Self:
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
                        # Send native Document values to Qdrant core for BM25
                        # instead of requiring FastEmbed in the API/worker.
                        cloud_inference=True,
                    )
                    logger.info(
                        "Connected to Qdrant",
                        host=settings.QDRANT_HOST,
                        port=settings.QDRANT_PORT,
                    )
        return self._client

    # ----- Collection Management -----

    def create_vector_collection(
        self,
        collection_name: str,
        *,
        schema_fingerprint: str | None = None,
        vector_dimension: int | None = None,
    ) -> None:
        """Tạo collection CÓ vector index (cho child chunks — dùng để search)."""
        if self._collection_exists(collection_name):
            if schema_fingerprint or vector_dimension:
                self.validate_collection_schema(
                    collection_name,
                    schema_fingerprint=schema_fingerprint,
                    vector_dimension=vector_dimension or settings.EMBEDDING_SIZE,
                    expected_distance=Distance.COSINE,
                    require_sparse=True,
                )
            logger.info("Collection already exists", collection=collection_name)
            return

        self._with_retry(
            lambda: self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dimension or settings.EMBEDDING_SIZE,
                    distance=Distance.COSINE,
                ),
                sparse_vectors_config={
                    settings.QDRANT_SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=settings.QDRANT_SPARSE_ON_DISK,
                        ),
                        modifier=Modifier.IDF,
                    )
                },
                metadata=(
                    {"schema_fingerprint": schema_fingerprint}
                    if schema_fingerprint else None
                ),
            ),
            operation=f"create collection {collection_name}",
        )
        logger.info("Created vector collection", collection=collection_name)

    def create_payload_collection(
        self,
        collection_name: str,
        *,
        schema_fingerprint: str | None = None,
    ) -> None:
        """Tạo collection KHÔNG CÓ vector (cho parent chunks — chỉ lưu text)."""
        if self._collection_exists(collection_name):
            if schema_fingerprint:
                self.validate_collection_schema(
                    collection_name,
                    schema_fingerprint=schema_fingerprint,
                    vector_dimension=None,
                    expected_distance=None,
                )
            logger.info("Collection already exists", collection=collection_name)
            return

        self._with_retry(
            lambda: self.client.create_collection(
                collection_name=collection_name,
                vectors_config={},
                metadata=(
                    {"schema_fingerprint": schema_fingerprint}
                    if schema_fingerprint else None
                ),
            ),
            operation=f"create collection {collection_name}",
        )
        logger.info("Created payload-only collection", collection=collection_name)

    def create_generation_collections(
        self,
        generation_id: str,
        *,
        schema_fingerprint: str,
        vector_dimension: int,
    ) -> tuple[str, str]:
        """Create and validate isolated parent/child collections for a generation."""
        if not generation_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in generation_id
        ):
            raise ValueError("generation_id contains unsupported collection-name characters")
        child_name = f"{settings.CHILD_COLLECTION}__{generation_id}"
        parent_name = f"{settings.PARENT_COLLECTION}__{generation_id}"
        self.create_vector_collection(
            child_name,
            schema_fingerprint=schema_fingerprint,
            vector_dimension=vector_dimension,
        )
        self.create_payload_collection(parent_name, schema_fingerprint=schema_fingerprint)
        for collection in (child_name, parent_name):
            for field_name in ("dataset_id", "file_name", "source_uri", "generation_id"):
                self.create_payload_index(collection, field_name)
        for collection, dimension, distance in (
            (child_name, vector_dimension, Distance.COSINE),
            (parent_name, None, None),
        ):
            self.validate_collection_schema(
                collection,
                schema_fingerprint=schema_fingerprint,
                vector_dimension=dimension,
                expected_distance=distance,
                required_payload_indexes=("dataset_id", "file_name", "source_uri", "generation_id"),
                require_sparse=dimension is not None,
            )
        return child_name, parent_name

    # ----- Write Operations -----

    def upsert_points(
        self,
        collection_name: str,
        points: list[PointStruct],
        *,
        batch_size: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        """Insert/update points in bounded, retryable, idempotent requests."""
        if not points:
            return
        total = 0
        for batch in iter_batches(
            points,
            max_items=batch_size or settings.INGEST_QDRANT_WRITE_BATCH_SIZE,
            max_bytes=max_bytes or settings.INGEST_QDRANT_WRITE_MAX_BYTES,
            size_of=approximate_size,
        ):
            write_batch = self._prepare_local_points(batch)
            self._with_retry(
                lambda write_batch=write_batch: self.client.upsert(
                    collection_name=collection_name,
                    points=write_batch,
                    wait=True,
                ),
                operation=f"upsert {collection_name} ({len(batch)} points)",
            )
            total += len(batch)
        logger.info("Upserted points", collection=collection_name, count=total)

    def list_ids_by_source(
        self,
        collection_name: str,
        *,
        dataset_id: str,
        file_name: str,
        batch_size: int = 1000,
    ) -> set[str]:
        """Lấy toàn bộ point IDs của đúng dataset và relative file path."""
        if not self._collection_exists(collection_name):
            return set()

        source_filter = Filter(must=[
            FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id)),
            FieldCondition(key="file_name", match=MatchValue(value=file_name)),
        ])
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=source_filter,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(str(point.id) for point in points)
            if offset is None:
                break
        return ids

    def list_file_names_by_dataset(
        self,
        collection_name: str,
        *,
        dataset_id: str,
        batch_size: int = 1000,
    ) -> set[str]:
        """Lấy các relative file path đang lưu trong một dataset namespace."""
        if not self._collection_exists(collection_name):
            return set()

        dataset_filter = Filter(must=[
            FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id)),
        ])
        file_names: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=dataset_filter,
                limit=batch_size,
                offset=offset,
                with_payload=["file_name"],
                with_vectors=False,
            )
            file_names.update(
                point.payload.get("file_name")
                for point in points
                if point.payload and point.payload.get("file_name")
            )
            if offset is None:
                break
        return file_names

    def delete_by_ids(self, collection_name: str, ids: set[str]) -> None:
        """Xóa chính xác một tập point IDs; mutation lỗi được propagate."""
        if not ids or not self._collection_exists(collection_name):
            return
        total = 0
        for batch in iter_batches(
            sorted(ids),
            max_items=settings.INGEST_QDRANT_WRITE_BATCH_SIZE,
            max_bytes=settings.INGEST_QDRANT_WRITE_MAX_BYTES,
            size_of=lambda value: len(value) + 64,
        ):
            self._with_retry(
                lambda batch=batch: self.client.delete(
                    collection_name=collection_name,
                    points_selector=PointIdsList(points=batch),
                    wait=True,
                ),
                operation=f"delete {collection_name} ({len(batch)} points)",
            )
            total += len(batch)
        logger.info("Deleted points by IDs", collection=collection_name, count=total)

    def delete_by_file_name(
        self,
        collection_name: str,
        file_name: str,
        *,
        dataset_id: str | None = None,
    ) -> None:
        """Xoá mọi point của 1 file trong một dataset namespace.

        Chống chunk rác vĩnh viễn (bug P1-6): content đổi → chunk_id đổi →
        point cũ không ai xoá. Fix: xoá toàn bộ point theo file_name trước khi
        ghi lại file.
        """
        if dataset_id is None:
            raise ValueError(
                "dataset_id is required; an unscoped file delete could affect another dataset"
            )
        self.delete_by_file_name_scoped(
            collection_name,
            file_name,
            dataset_id=dataset_id,
        )

    def delete_by_file_name_scoped(
        self,
        collection_name: str,
        file_name: str,
        *,
        dataset_id: str,
    ) -> None:
        """Delete one file without crossing dataset boundaries.

        The old unscoped helper is intentionally disabled.  Keeping an unscoped
        delete available is too easy to misuse when multiple datasets share a
        collection.
        """
        if not dataset_id:
            raise ValueError("dataset_id is required for a scoped delete")
        if not self._collection_exists(collection_name):
            return
        self._with_retry(
            lambda: self.client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(must=[
                        FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id)),
                        FieldCondition(key="file_name", match=MatchValue(value=file_name)),
                    ])
                ),
                wait=True,
            ),
            operation=f"delete source {collection_name}/{file_name}",
        )
        logger.info("Deleted existing points for file",
                    collection=collection_name, file=file_name)

    def create_payload_index(self, collection_name: str, field_name: str) -> None:
        """Index cho payload field — bắt buộc để filter/delete-by-filter chạy nhanh."""
        from qdrant_client.http.exceptions import UnexpectedResponse
        from qdrant_client.models import PayloadSchemaType
        try:
            self._with_retry(
                lambda: self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                ),
                operation=f"create payload index {collection_name}/{field_name}",
            )
            logger.info("Created payload index", collection=collection_name, field=field_name)
        except UnexpectedResponse as exc:
            if exc.status_code == 409:      # index đã tồn tại → bỏ qua (idempotent)
                logger.debug("Payload index skipped (already exists)",
                             field=field_name)
            else:
                # Network/auth hoặc lỗi Qdrant khác phải propagate — không được
                # coi mọi exception là "đã tồn tại" (P3-8).
                logger.error("Failed to create payload index",
                             collection=collection_name, field=field_name,
                             status=exc.status_code)
                raise

    # ----- Read Operations -----

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        query_filter: Filter | None = None,
    ) -> list:
        """Tìm kiếm vector tương đồng (cosine similarity).

        qdrant-client v1.18+: dùng query_points() thay vì search().
        """
        result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            timeout=self._query_timeout(),
        )
        return result.points

    def search_sparse(
        self,
        collection_name: str,
        query,
        *,
        sparse_vector_name: str,
        limit: int = 10,
        query_filter: Filter | None = None,
    ) -> list:
        """Search a named sparse vector entirely inside Qdrant."""
        result = self.client.query_points(
            collection_name=collection_name,
            query=self._prepare_local_sparse(query),
            using=sparse_vector_name,
            limit=limit,
            query_filter=query_filter,
            timeout=self._query_timeout(),
        )
        return result.points

    def search_hybrid(
        self,
        collection_name: str,
        *,
        dense_vector: list[float],
        sparse_query,
        sparse_vector_name: str,
        limit: int,
        prefetch_limit: int,
        query_filter: Filter | None,
    ) -> list:
        """Fuse dense and sparse candidates server-side using scoped RRF."""
        prefetch = [
            Prefetch(
                query=dense_vector,
                filter=query_filter,
                limit=prefetch_limit,
            ),
            Prefetch(
                query=sparse_query,
                using=sparse_vector_name,
                filter=query_filter,
                limit=prefetch_limit,
            ),
        ]
        if self._is_local_client:
            prefetch[1] = Prefetch(
                query=self._prepare_local_sparse(sparse_query),
                using=sparse_vector_name,
                filter=query_filter,
                limit=prefetch_limit,
            )
        result = self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=RrfQuery(rrf=Rrf(k=61)),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            timeout=self._query_timeout(),
        )
        return result.points

    def scroll_all(
        self,
        collection_name: str,
        batch_size: int = 1000,
        max_points: int | None = None,
        scroll_filter: Filter | None = None,
    ) -> list:
        """Đọc TẤT CẢ points trong collection (phân trang đúng cách).

        Fix bug P1-9: bản cũ chỉ đọc trang đầu (limit=10000) → mọi point sau
        #10000 biến mất khỏi BM25 index im lặng, không log, không lỗi.
        """
        all_points = list(self.iter_scroll(
            collection_name,
            batch_size=batch_size,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            max_points=max_points,
        ))
        logger.info("Scrolled collection", collection=collection_name, total=len(all_points))
        return all_points

    def iter_scroll(
        self,
        collection_name: str,
        *,
        batch_size: int = 1000,
        scroll_filter: Filter | None = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
        max_points: int | None = None,
    ) -> Iterator:
        """Yield scroll results incrementally for copy/validation jobs."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        offset = None
        yielded = 0
        while True:
            points, offset = self._with_retry(
                lambda offset=offset: self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                ),
                operation=f"scroll {collection_name}",
            )
            for point in points:
                yield point
                yielded += 1
                if max_points is not None and yielded >= max_points:
                    logger.warning(
                        "iter_scroll hit max_points cap",
                        collection=collection_name,
                        cap=max_points,
                        fetched=yielded,
                    )
                    return
            if offset is None:
                return

    def get_by_ids(
        self,
        collection_name: str,
        ids: list[str],
        *,
        query_filter: Filter | None = None,
    ) -> list:
        """Lấy points theo IDs, tùy chọn kết hợp filter server-side.

        Qdrant ``retrieve`` không hỗ trợ payload filters.  Khi scope được truyền,
        dùng ``scroll`` với ``HasIdCondition`` để không fetch parent ngoài scope.
        """
        if query_filter is not None:
            scoped_filter = Filter(must=[query_filter, HasIdCondition(has_id=ids)])
            points, offset = [], None
            while True:
                batch, offset = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scoped_filter,
                    limit=max(len(ids), 1),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    timeout=self._query_timeout(),
                )
                points.extend(batch)
                if offset is None:
                    break
            return points
        points = self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
            timeout=self._query_timeout(),
        )
        return points

    def get_existing_ids(self, collection_name: str, ids: list[str]) -> set[str]:
        """Return IDs already persisted, closing the upsert/checkpoint gap."""
        if not ids or not self._collection_exists(collection_name):
            return set()
        existing: set[str] = set()
        for batch in iter_batches(
            ids,
            max_items=settings.INGEST_QDRANT_WRITE_BATCH_SIZE,
            max_bytes=settings.INGEST_QDRANT_WRITE_MAX_BYTES,
            size_of=lambda value: len(value) + 64,
        ):
            records = self._with_retry(
                lambda batch=batch: self.client.retrieve(
                    collection_name=collection_name,
                    ids=batch,
                    with_payload=False,
                    with_vectors=False,
                ),
                operation=f"verify IDs in {collection_name}",
            )
            existing.update(str(record.id) for record in records)
        return existing

    def copy_source_points(
        self,
        source_collection: str,
        target_collection: str,
        *,
        dataset_id: str,
        source_uri: str,
        with_vectors: bool,
        target_generation_id: str | None = None,
    ) -> int:
        """Copy one unchanged source between generations using bounded batches.

        The source payload is copied as-is except for ``generation_id``.  A
        generation is an immutable snapshot, so copied points must identify the
        target snapshot rather than retaining the source generation's audit
        value.
        """
        source_filter = Filter(must=[
            FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id)),
            FieldCondition(key="source_uri", match=MatchValue(value=source_uri)),
        ])
        pending: list[PointStruct] = []
        copied = 0
        for point in self.iter_scroll(
            source_collection,
            batch_size=settings.INGEST_QDRANT_WRITE_BATCH_SIZE,
            scroll_filter=source_filter,
            with_payload=True,
            with_vectors=with_vectors,
        ):
            vector = getattr(point, "vector", None) if with_vectors else {}
            payload = dict(point.payload or {})
            if target_generation_id is not None:
                payload["generation_id"] = target_generation_id
            pending.append(PointStruct(id=point.id, vector=vector or {}, payload=payload))
            if len(pending) >= settings.INGEST_QDRANT_WRITE_BATCH_SIZE:
                self.upsert_points(target_collection, pending)
                copied += len(pending)
                pending = []
        if pending:
            self.upsert_points(target_collection, pending)
            copied += len(pending)
        return copied

    def switch_aliases(
        self,
        *,
        child_collection: str,
        parent_collection: str,
    ) -> tuple[str | None, str | None]:
        """Atomically switch both retrieval aliases and return previous targets."""
        aliases = self.client.get_aliases().aliases
        targets = {alias.alias_name: alias.collection_name for alias in aliases}
        child_previous = targets.get(settings.CHILD_COLLECTION)
        parent_previous = targets.get(settings.PARENT_COLLECTION)

        def apply_switch() -> None:
            # Re-read aliases on every retry.  If the first request committed
            # but its response timed out, a second CreateAlias-only request
            # would otherwise fail with 409 even though the desired state is
            # already present.
            current = {
                alias.alias_name: alias.collection_name
                for alias in self.client.get_aliases().aliases
            }
            desired = {
                settings.CHILD_COLLECTION: child_collection,
                settings.PARENT_COLLECTION: parent_collection,
            }
            if all(current.get(name) == target for name, target in desired.items()):
                return
            actions = []
            for alias_name, collection_name in desired.items():
                if alias_name in current:
                    actions.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name)))
                actions.append(
                    CreateAliasOperation(
                        create_alias=CreateAlias(collection_name=collection_name, alias_name=alias_name)
                    )
                )
            self.client.update_collection_aliases(actions)

        self._with_retry(
            apply_switch,
            operation="atomic retrieval alias switch",
        )
        return child_previous, parent_previous

    def alias_target(self, alias_name: str) -> str | None:
        aliases = self.client.get_aliases().aliases
        return next((alias.collection_name for alias in aliases if alias.alias_name == alias_name), None)

    def validate_collection_schema(
        self,
        collection_name: str,
        *,
        schema_fingerprint: str | None,
        vector_dimension: int | None,
        expected_distance: Distance | None,
        required_payload_indexes: tuple[str, ...] = (),
        require_sparse: bool = False,
    ) -> None:
        """Fail closed when collection/model schema differs from the job."""
        info = self._with_retry(
            lambda: self.client.get_collection(collection_name),
            operation=f"validate collection {collection_name}",
        )
        vectors = info.config.params.vectors
        if vector_dimension is not None:
            if isinstance(vectors, dict) or vectors is None or vectors.size != vector_dimension:
                raise ValueError(
                    f"Collection {collection_name} vector dimension mismatch: "
                    f"actual={getattr(vectors, 'size', None)}, expected={vector_dimension}"
                )
            if expected_distance is not None and vectors.distance != expected_distance:
                raise ValueError(
                    f"Collection {collection_name} distance mismatch: "
                    f"actual={vectors.distance}, expected={expected_distance}"
                )
        elif vectors not in ({}, None) and not (isinstance(vectors, dict) and not vectors):
            raise ValueError(f"Collection {collection_name} unexpectedly has vectors")

        metadata = getattr(info.config, "metadata", None) or {}
        if schema_fingerprint and metadata.get("schema_fingerprint") != schema_fingerprint:
            raise ValueError(f"Collection {collection_name} schema fingerprint mismatch")
        missing = [field for field in required_payload_indexes if field not in info.payload_schema]
        if missing and self._is_local_client:
            logger.warning(
                "Local Qdrant does not expose payload indexes; skipping index validation",
                collection=collection_name,
                missing=missing,
            )
            missing = []
        if missing:
            raise ValueError(f"Collection {collection_name} missing payload indexes: {missing}")
        if require_sparse:
            sparse_vectors = getattr(info.config.params, "sparse_vectors", None) or {}
            sparse = sparse_vectors.get(settings.QDRANT_SPARSE_VECTOR_NAME)
            if sparse is None:
                raise ValueError(
                    f"Collection {collection_name} missing sparse vector "
                    f"{settings.QDRANT_SPARSE_VECTOR_NAME}"
                )
            if sparse.modifier != Modifier.IDF:
                raise ValueError(
                    f"Collection {collection_name} sparse modifier mismatch: "
                    f"actual={sparse.modifier}, expected={Modifier.IDF}"
                )

    def validate_sparse_coverage(
        self,
        collection_name: str,
        *,
        sparse_vector_name: str,
    ) -> None:
        """Require every child point to carry the sparse vector before activation."""
        total = self._with_retry(
            lambda: self.client.count(
                collection_name=collection_name,
                exact=True,
            ).count,
            operation=f"count {collection_name}",
        )
        sparse = self._with_retry(
            lambda: self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[HasVectorCondition(has_vector=sparse_vector_name)]
                ),
                exact=True,
            ).count,
            operation=f"count sparse coverage {collection_name}",
        )
        if sparse != total:
            raise ValueError(
                f"Collection {collection_name} sparse coverage mismatch: "
                f"sparse={sparse}, total={total}"
            )

    def delete_collection(self, collection_name: str) -> None:
        """Delete an unused staging collection; callers must resolve exact names."""
        if self._collection_exists(collection_name):
            self._with_retry(
                lambda: self.client.delete_collection(collection_name),
                operation=f"delete collection {collection_name}",
            )

    # ----- Helpers -----

    def _collection_exists(self, name: str) -> bool:
        collections = [c.name for c in self.client.get_collections().collections]
        return name in collections

    def _prepare_local_points(self, points: list[PointStruct]) -> list[PointStruct]:
        """Materialize native text documents for the in-memory dev backend.

        Production Qdrant core receives a Document and performs native BM25.
        The local client has no server inference unless FastEmbed is installed,
        so tests use a deterministic hashed sparse vector while still querying
        a bounded sparse index.
        """
        if not self._is_local_client:
            return points
        prepared: list[PointStruct] = []
        for point in points:
            vector = point.vector
            if not isinstance(vector, dict):
                prepared.append(point)
                continue
            converted = {
                name: self._prepare_local_sparse(value)
                for name, value in vector.items()
            }
            prepared.append(point.model_copy(update={"vector": converted}))
        return prepared

    @staticmethod
    def _local_sparse_vector(text: str) -> SparseVector:
        tokens = re.findall(
            r"[^\W_]+(?:[-_][^\W_]+)*",
            unicodedata.normalize("NFKC", text).casefold(),
            re.UNICODE,
        )
        frequencies: dict[int, float] = {}
        for token in tokens:
            index = int.from_bytes(
                blake2b(token.encode("utf-8"), digest_size=4).digest(),
                "big",
            )
            frequencies[index] = frequencies.get(index, 0.0) + 1.0
        indices = sorted(frequencies)
        return SparseVector(
            indices=indices,
            values=[frequencies[index] for index in indices],
        )

    def _prepare_local_sparse(self, query):
        if self._is_local_client and isinstance(query, Document):
            return self._local_sparse_vector(query.text)
        return query

    @property
    def _is_local_client(self) -> bool:
        return self.client.__dict__.get("_init_options", {}).get("location") == ":memory:"

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return status_code in {408, 425, 429} or status_code >= 500
        name = type(exc).__name__.lower()
        return (
            isinstance(exc, (TimeoutError, ConnectionError))
            or "timeout" in name
            or "connection" in name
            or "transport" in name
        )

    def _with_retry(self, func: Callable, *, operation: str):
        """Retry only transient failures; validation/auth errors fail immediately."""
        attempts = max(settings.INGEST_QDRANT_MAX_RETRIES, 0) + 1
        for attempt in range(attempts):
            try:
                return func()
            except Exception as exc:
                if attempt >= attempts - 1 or not self._is_retryable(exc):
                    raise
                delay = min(
                    settings.INGEST_RETRY_MAX_SECONDS,
                    settings.INGEST_RETRY_BASE_SECONDS * (2**attempt),
                )
                delay *= 0.8 + random.random() * 0.4
                logger.warning(
                    "Transient Qdrant operation failure; retrying",
                    operation=operation,
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    delay_seconds=round(delay, 3),
                    error_type=type(exc).__name__,
                )
                time.sleep(delay)

    @staticmethod
    def _query_timeout() -> int:
        """Return the configured transport timeout for Qdrant reads."""
        import math

        return max(1, math.ceil(settings.QDRANT_QUERY_TIMEOUT_SECONDS))

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Qdrant connection closed")
