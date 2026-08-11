"""
Ingestion Pipeline Orchestrator — Nối tất cả lại.

Luồng xử lý:
  1. Scan thư mục → tìm tất cả files (PDF, MD, DOCX)
  2. Parse mỗi file → list[RawDocument]
  3. Parent-Child Chunking → parent_chunks + child_chunks
  4. Embed child chunks → vectors 384d
  5. Store vào Qdrant:
     - child_chunks → collection có vectors (dùng để search)
     - parent_chunks → collection payload-only (dùng để trả context)

Usage:
    from ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline.run("data/sample_docs/")
    # Dùng sync=True một cách explicit nếu muốn prune file đã bị xóa.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client.models import PointStruct

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from ingestion.chunking.parent_child import parent_child_chunk
from ingestion.embeddings import EmbeddingService
from ingestion.models import Chunk, EmbeddedChunk, RawDocument
from ingestion.parsers import ParserDispatcher

logger = get_logger(__name__)


@dataclass
class ParseBatch:
    """Kết quả scan/parse, giữ cả file parse lỗi để sync không xóa nhầm."""

    discovered_files: set[str] = field(default_factory=set)
    documents_by_file: dict[str, list[RawDocument]] = field(default_factory=dict)
    failed_files: set[str] = field(default_factory=set)


@dataclass
class PreparedFile:
    """Dữ liệu đã chunk/embed hoàn tất nhưng chưa mutate Qdrant."""

    dataset_id: str
    file_name: str
    parents: list[Chunk]
    children: list[EmbeddedChunk]


@dataclass
class IngestionResult:
    """Tóm tắt run để CLI/operator biết sync có bị bỏ qua hay không."""

    dataset_id: str
    discovered_files: set[str] = field(default_factory=set)
    processed_files: set[str] = field(default_factory=set)
    failed_files: set[str] = field(default_factory=set)
    pruned_files: set[str] = field(default_factory=set)
    prune_skipped: bool = False


class IngestionPipeline:
    """Orchestrator cho toàn bộ ingestion flow."""

    def __init__(self):
        self.qdrant = QdrantConnector()
        self.embedder = EmbeddingService()

    def run(self, data_dir: str, *, sync: bool = False) -> IngestionResult:
        """Chạy pipeline safe-replace; chỉ prune file mất khi ``sync=True``."""

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Bước 0: Tạo collections trong Qdrant
        self._init_collections()

        # Bước 1: Scan & parse, giữ discovered_files kể cả file parse lỗi.
        parse_batch = self._parse_all_files(data_path)
        processed_files: set[str] = set()
        pruned_files: set[str] = set()
        bm25_may_be_dirty = False

        try:
            # Bước 2-5: Prepare hoàn toàn trong memory, sau đó commit từng file.
            for file_name in sorted(parse_batch.documents_by_file):
                documents = parse_batch.documents_by_file[file_name]
                logger.info("Processing file", file=file_name, raw_docs=len(documents))

                prepared = self._prepare_file(
                    settings.INGEST_DATASET_ID,
                    file_name,
                    documents,
                )
                if not prepared.parents and not prepared.children:
                    # Không coi parser/chunker trả rỗng là file đã bị xóa.
                    parse_batch.failed_files.add(file_name)
                    logger.warning("No chunks prepared, preserving existing points", file=file_name)
                    continue

                # Đặt cờ trước mutation: SDK có thể ghi một phần rồi mới raise.
                bm25_may_be_dirty = True
                self._commit_file(prepared)
                processed_files.add(file_name)

                logger.info(
                    "File processed",
                    file=file_name,
                    parents=len(prepared.parents),
                    children=len(prepared.children),
                )

            if sync:
                if parse_batch.failed_files:
                    logger.warning(
                        "Skipping stale-file prune because some files failed",
                        failed_files=sorted(parse_batch.failed_files),
                    )
                else:
                    def mark_bm25_dirty() -> None:
                        nonlocal bm25_may_be_dirty
                        # Callback chạy ngay trước delete đầu tiên; nếu delete
                        # kế tiếp raise thì BM25 vẫn được invalidate ở finally.
                        bm25_may_be_dirty = True

                    pruned_files = self._prune_removed_files(
                        settings.INGEST_DATASET_ID,
                        parse_batch.discovered_files,
                        on_mutation=mark_bm25_dirty,
                    )

            if parse_batch.failed_files:
                logger.warning(
                    "Ingestion completed with failed files",
                    failed_files=sorted(parse_batch.failed_files),
                )
            else:
                logger.info("Ingestion pipeline complete")

            return IngestionResult(
                dataset_id=settings.INGEST_DATASET_ID,
                discovered_files=parse_batch.discovered_files,
                processed_files=processed_files,
                failed_files=parse_batch.failed_files,
                pruned_files=pruned_files,
                prune_skipped=sync and bool(parse_batch.failed_files),
            )
        finally:
            # Invalidate only after a Qdrant mutation was attempted. If upsert
            # partially succeeded and raised, stale in-memory BM25 is unsafe.
            if bm25_may_be_dirty:
                self._invalidate_bm25_index()

    # ================================================================
    # Private methods — từng bước xử lý
    # ================================================================

    def _init_collections(self) -> None:
        """Tạo 2 collections trong Qdrant nếu chưa tồn tại."""
        self.qdrant.create_vector_collection(settings.CHILD_COLLECTION)
        self.qdrant.create_payload_collection(settings.PARENT_COLLECTION)
        # Index để safe-replace/sync filter đúng dataset + relative file.
        for coll in (settings.CHILD_COLLECTION, settings.PARENT_COLLECTION):
            for field_name in ("dataset_id", "file_name"):
                self.qdrant.create_payload_index(coll, field_name)

    def _parse_all_files(self, data_path: Path) -> ParseBatch:
        """Scan thư mục, parse tất cả files hỗ trợ.

        Returns:
            ``ParseBatch`` gồm discovered files, documents thành công và failed files.
            Dùng relative path làm key để 2 file cùng tên ở 2 thư mục con
            không đè nhau (bug P1-6).
        """
        supported = ParserDispatcher.supported_extensions()
        # ★ thêm is_file() — rglob("*") cũng trả về thư mục, làm parser crash
        files = [f for f in data_path.rglob("*")
                 if f.is_file() and f.suffix.lower() in supported]

        if not files:
            logger.warning("No supported files found", path=str(data_path), supported=supported)
            return ParseBatch()

        logger.info("Found files", count=len(files))

        discovered_files = {
            file_path.relative_to(data_path).as_posix()
            for file_path in files
        }
        result: dict[str, list[RawDocument]] = {}
        failed_files: set[str] = set()
        for file_path in sorted(files):
            rel = file_path.relative_to(data_path).as_posix()   # ★ unique key
            try:
                parser = ParserDispatcher.get_parser(file_path)
                documents = parser.parse(file_path)
            except Exception:
                # ★ 1 file hỏng KHÔNG được giết cả pipeline
                logger.exception("Failed to parse file, skipping", file=rel)
                failed_files.add(rel)
                continue
            if documents:
                # ★ đồng bộ file_name với key dùng để delete (bug P1-6)
                for d in documents:
                    d.metadata = d.metadata.model_copy(update={
                        "dataset_id": settings.INGEST_DATASET_ID,
                        "file_name": rel,
                        # ID phải ổn định giữa máy/deploy, không dùng absolute path.
                        "source_path": rel,
                    })
                result[rel] = documents
            else:
                logger.warning("Parser returned no documents", file=rel)
                failed_files.add(rel)

        logger.info(
            "Parsed files",
            found=len(files),
            parsed=len(result),
            failed=len(failed_files),
        )
        return ParseBatch(
            discovered_files=discovered_files,
            documents_by_file=result,
            failed_files=failed_files,
        )

    def _prepare_file(
        self,
        dataset_id: str,
        file_name: str,
        documents: list[RawDocument],
    ) -> PreparedFile:
        """Parse result → chunks → embeddings, chưa chạm Qdrant."""
        normalized_documents = []
        for document in documents:
            normalized_documents.append(document.model_copy(update={
                "metadata": document.metadata.model_copy(update={
                    "dataset_id": dataset_id,
                    "file_name": file_name,
                    "source_path": file_name,
                }),
            }))

        parents, children = parent_child_chunk(normalized_documents)
        embedded_children = self._embed_chunks(children)
        return PreparedFile(
            dataset_id=dataset_id,
            file_name=file_name,
            parents=parents,
            children=embedded_children,
        )

    def _commit_file(self, prepared: PreparedFile) -> None:
        """Upsert mới trước, chỉ xóa IDs cũ sau khi cả collections thành công."""
        old_parent_ids = self.qdrant.list_ids_by_source(
            settings.PARENT_COLLECTION,
            dataset_id=prepared.dataset_id,
            file_name=prepared.file_name,
        )
        old_child_ids = self.qdrant.list_ids_by_source(
            settings.CHILD_COLLECTION,
            dataset_id=prepared.dataset_id,
            file_name=prepared.file_name,
        )
        new_parent_ids = {chunk.chunk_id for chunk in prepared.parents}
        new_child_ids = {chunk.chunk_id for chunk in prepared.children}

        # Parent trước child để child mới không trỏ đến parent chưa tồn tại.
        self._store_parents(prepared.parents)
        self._store_children(prepared.children)

        # Delete stale IDs chỉ xảy ra sau khi cả hai upsert không raise.
        self.qdrant.delete_by_ids(
            settings.CHILD_COLLECTION,
            old_child_ids - new_child_ids,
        )
        self.qdrant.delete_by_ids(
            settings.PARENT_COLLECTION,
            old_parent_ids - new_parent_ids,
        )

    def _prune_removed_files(
        self,
        dataset_id: str,
        discovered_files: set[str],
        *,
        on_mutation: Callable[[], None] | None = None,
    ) -> set[str]:
        """Xóa file không còn trên disk, chỉ trong namespace dataset hiện tại."""
        stored_files = (
            self.qdrant.list_file_names_by_dataset(
                settings.CHILD_COLLECTION,
                dataset_id=dataset_id,
            )
            | self.qdrant.list_file_names_by_dataset(
                settings.PARENT_COLLECTION,
                dataset_id=dataset_id,
            )
        )
        removed_files = stored_files - discovered_files
        for file_name in sorted(removed_files):
            child_ids = self.qdrant.list_ids_by_source(
                settings.CHILD_COLLECTION,
                dataset_id=dataset_id,
                file_name=file_name,
            )
            parent_ids = self.qdrant.list_ids_by_source(
                settings.PARENT_COLLECTION,
                dataset_id=dataset_id,
                file_name=file_name,
            )
            if child_ids:
                if on_mutation:
                    on_mutation()
                self.qdrant.delete_by_ids(settings.CHILD_COLLECTION, child_ids)
            if parent_ids:
                if on_mutation:
                    on_mutation()
                self.qdrant.delete_by_ids(settings.PARENT_COLLECTION, parent_ids)

        if removed_files:
            logger.info("Pruned removed files", dataset_id=dataset_id, files=sorted(removed_files))
        return removed_files

    @staticmethod
    def _invalidate_bm25_index() -> None:
        """Invalidate BM25 sau mutation; ingestion vẫn không phụ thuộc retrieval."""
        try:
            from retrieval.search.sparse import invalidate_bm25_index
        except ImportError:
            return
        invalidate_bm25_index()

    def _embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed danh sách chunks → EmbeddedChunks (có vector)."""
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        vectors = self.embedder.embed(texts)

        embedded = []
        for chunk, vector in zip(chunks, vectors):
            embedded.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    embedding=vector,
                    parent_id=chunk.parent_id,
                    metadata=chunk.metadata,
                )
            )

        logger.info("Embedded chunks", count=len(embedded))
        return embedded

    def _store_children(self, chunks: list[EmbeddedChunk]) -> None:
        """Lưu child chunks vào Qdrant (CÓ vector)."""
        if not chunks:
            return

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "parent_id": chunk.parent_id,
                    "file_name": chunk.metadata.file_name,
                    "file_type": chunk.metadata.file_type,
                    "source_path": chunk.metadata.source_path,
                    "dataset_id": chunk.metadata.dataset_id,
                    "section_title": chunk.metadata.section_title,
                    "page_number": chunk.metadata.page_number,   # ★ THÊM
                },
            )
            for chunk in chunks
        ]

        self.qdrant.upsert_points(settings.CHILD_COLLECTION, points)

    def _store_parents(self, chunks: list[Chunk]) -> None:
        """Lưu parent chunks vào Qdrant (KHÔNG có vector, chỉ payload)."""
        if not chunks:
            return

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector={},  # Payload-only: không có vector
                payload={
                    "content": chunk.content,
                    "file_name": chunk.metadata.file_name,
                    "file_type": chunk.metadata.file_type,
                    "source_path": chunk.metadata.source_path,
                    "dataset_id": chunk.metadata.dataset_id,
                    "section_title": chunk.metadata.section_title,
                    "page_number": chunk.metadata.page_number,   # ★ THÊM
                },
            )
            for chunk in chunks
        ]

        self.qdrant.upsert_points(settings.PARENT_COLLECTION, points)
