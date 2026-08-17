"""
Ingestion Pipeline Orchestrator — Nối tất cả lại.

Luồng xử lý:
  1. Scan thư mục → tìm tất cả files (PDF, MD, DOCX)
  2. Parse mỗi file → list[RawDocument]
  3. Parent-Child Chunking → parent_chunks + child_chunks
  4. Embed child chunks → vectors 1024d (BGE-M3 default)
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

import inspect
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

from qdrant_client.models import Distance, PointStruct

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from ingestion.batching import iter_batches
from ingestion.chunking.parent_child import iter_parent_child_chunks, parent_child_chunk
from ingestion.embeddings import EmbeddingService
from ingestion.manifest import ManifestStore, pipeline_fingerprint, sha256_file
from ingestion.models import Chunk, EmbeddedChunk, ParseQuality, RawDocument
from ingestion.parsers import ParserDispatcher
from retrieval.search.sparse import sparse_document

logger = get_logger(__name__)


@dataclass
class ParseBatch:
    """Kết quả scan/parse, giữ cả file parse lỗi để sync không xóa nhầm."""

    discovered_files: set[str] = field(default_factory=set)
    documents_by_file: dict[str, list[RawDocument]] = field(default_factory=dict)
    failed_files: set[str] = field(default_factory=set)
    quality_by_file: dict[str, ParseQuality] = field(default_factory=dict)


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
    planned_pruned_files: set[str] = field(default_factory=set)
    dry_run: bool = False
    job_id: str | None = None
    generation_id: str | None = None
    skipped_files: set[str] = field(default_factory=set)
    quality_by_file: dict[str, dict] = field(default_factory=dict)


class IngestionPipeline:
    """Orchestrator cho toàn bộ ingestion flow."""

    def __init__(self):
        self.qdrant = QdrantConnector()
        self.embedder = EmbeddingService()
        self.manifest = ManifestStore()

    def _run_legacy(
        self,
        data_dir: str,
        *,
        sync: bool = False,
        allow_empty_source: bool = False,
        dry_run: bool = False,
    ) -> IngestionResult:
        """Chạy pipeline safe-replace với guard cho destructive sync.

        ``sync=True`` không được coi một source rỗng là trạng thái hợp lệ mặc
        định: mount sai hoặc listing lỗi có thể nếu không sẽ xóa toàn dataset.
        Caller phải truyền ``allow_empty_source=True`` một cách rõ ràng để prune
        về zero. ``dry_run`` chỉ lập kế hoạch, không tạo collection hay mutate DB.
        """

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not data_path.is_dir():
            raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

        # Bước 1: Scan & parse, giữ discovered_files kể cả file parse lỗi.
        parse_batch = self._parse_all_files(data_path)

        if sync and not parse_batch.discovered_files and not allow_empty_source:
            raise ValueError(
                "Refusing destructive sync from an empty source. "
                "Verify the mount/path, or pass allow_empty_source=True explicitly."
            )

        if dry_run:
            planned_pruned_files = (
                self._find_removed_files(
                    settings.INGEST_DATASET_ID,
                    parse_batch.discovered_files,
                )
                if sync and not parse_batch.failed_files
                else set()
            )
            logger.info(
                "Ingestion dry-run complete",
                dataset_id=settings.INGEST_DATASET_ID,
                discovered=len(parse_batch.discovered_files),
                failed=len(parse_batch.failed_files),
                would_prune=sorted(planned_pruned_files),
            )
            return IngestionResult(
                dataset_id=settings.INGEST_DATASET_ID,
                discovered_files=parse_batch.discovered_files,
                failed_files=parse_batch.failed_files,
                planned_pruned_files=planned_pruned_files,
                prune_skipped=sync and bool(parse_batch.failed_files),
                dry_run=True,
            )

        # Bước 0: Tạo collections trong Qdrant
        self._init_collections()

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

    def run(
        self,
        data_dir: str,
        *,
        sync: bool = False,
        allow_empty_source: bool = False,
        dry_run: bool = False,
        job_id: str | None = None,
        generation_id: str | None = None,
        resume: bool = True,
    ) -> IngestionResult:
        """Run bounded, manifest-backed, versioned ingestion.

        Objects created by older tests/callers with ``__new__`` do not have a
        manifest.  They deliberately retain the PR10 safe-replace path so the
        compatibility contract remains intact; normal construction always uses
        the versioned path.
        """
        if not hasattr(self, "manifest") or not settings.INGEST_VERSIONED:
            return self._run_legacy(
                data_dir,
                sync=sync,
                allow_empty_source=allow_empty_source,
                dry_run=dry_run,
            )
        return self._run_versioned(
            data_dir,
            sync=sync,
            allow_empty_source=allow_empty_source,
            dry_run=dry_run,
            job_id=job_id,
            generation_id=generation_id,
            resume=resume,
        )

    def _run_versioned(
        self,
        data_dir: str,
        *,
        sync: bool,
        allow_empty_source: bool,
        dry_run: bool,
        job_id: str | None,
        generation_id: str | None,
        resume: bool,
    ) -> IngestionResult:
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not data_path.is_dir():
            raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

        existing_job = self.manifest.get_job(job_id) if job_id else None
        if existing_job and resume:
            if existing_job["status"] == "committed":
                raise ValueError(
                    "Committed ingestion generations are immutable; start a new job/generation"
                )
            if generation_id and generation_id != existing_job["generation_id"]:
                raise ValueError("resume generation_id does not match the existing job")
            generation_id = generation_id or existing_job["generation_id"]
        else:
            generation_id = generation_id or uuid.uuid4().hex[:16]
        job_id = job_id or uuid.uuid4().hex
        if self.manifest.is_generation_committed(settings.INGEST_DATASET_ID, generation_id):
            raise ValueError(
                "Committed ingestion generations are immutable; start a new generation"
            )
        vector_dimension = self.embedder.dimension
        fingerprint = pipeline_fingerprint(embedding_dimension=vector_dimension)
        self._recover_alias_activation(fingerprint=fingerprint, vector_dimension=vector_dimension)
        if self.manifest.is_generation_committed(settings.INGEST_DATASET_ID, generation_id):
            raise ValueError(
                "Committed ingestion generations are immutable; start a new generation"
            )
        active_sources = self.manifest.list_sources(settings.INGEST_DATASET_ID)
        previous_generation = next(
            (
                record.active_generation
                for record in active_sources
                # A failed working run must not erase the last active
                # generation from the manifest.  The source row can be
                # ``failed`` while its ``active_generation`` still points at
                # the serving snapshot.
                if record.active_generation
            ),
            None,
        )
        previous_records = (
            self.manifest.list_sources(
                settings.INGEST_DATASET_ID,
                active_generation=previous_generation,
            )
            if previous_generation else []
        )
        if not resume and existing_job:
            raise ValueError(f"ingestion job already exists: {job_id}")
        self.manifest.create_job(
            job_id=job_id,
            dataset_id=settings.INGEST_DATASET_ID,
            generation_id=generation_id,
            fingerprint=fingerprint,
            active_generation_before=previous_generation,
        )

        if dry_run:
            return self._dry_run_versioned(
                data_path,
                sync=sync,
                allow_empty_source=allow_empty_source,
                job_id=job_id,
                generation_id=generation_id,
                fingerprint=fingerprint,
            )

        source_iterator = self._iter_source_files(data_path)
        first_source = next(source_iterator, None)
        if sync and first_source is None and not allow_empty_source:
            self.manifest.finish_job(job_id=job_id, status="failed")
            raise ValueError(
                "Refusing destructive sync from an empty source. "
                "Verify the mount/path, or pass allow_empty_source=True explicitly."
            )

        child_collection, parent_collection = self.qdrant.create_generation_collections(
            generation_id,
            schema_fingerprint=fingerprint,
            vector_dimension=vector_dimension,
        )
        previous_child = (
            f"{settings.CHILD_COLLECTION}__{previous_generation}"
            if previous_generation else self.qdrant.alias_target(settings.CHILD_COLLECTION)
        )
        previous_parent = (
            f"{settings.PARENT_COLLECTION}__{previous_generation}"
            if previous_generation else self.qdrant.alias_target(settings.PARENT_COLLECTION)
        )
        discovered: set[str] = set()
        active_source_uris: set[str] = set()
        processed: set[str] = set()
        skipped: set[str] = set()
        failed: set[str] = set()
        quality_by_file: dict[str, dict] = {}
        stale: set[str] = set()

        try:
            for source in chain(([first_source] if first_source else []), source_iterator):
                discovered.add(source["source_uri"])
                active_source_uris.add(source["source_uri"])
                source_uri = source["source_uri"]
                content_hash = sha256_file(source["path"])
                previous = self.manifest.record_discovered(
                    dataset_id=settings.INGEST_DATASET_ID,
                    source_uri=source_uri,
                    content_sha256=content_hash,
                    size_bytes=source["size_bytes"],
                    mtime_ns=source["mtime_ns"],
                    fingerprint=fingerprint,
                    generation_id=generation_id,
                )
                unchanged = bool(
                    previous
                    and previous.status == "committed"
                    and previous.content_sha256 == content_hash
                    and previous.fingerprint == fingerprint
                    and previous_generation
                    and previous.active_generation == previous_generation
                )
                if unchanged and previous_child and previous_parent:
                    copied_children = self.qdrant.copy_source_points(
                        previous_child,
                        child_collection,
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=source_uri,
                        with_vectors=True,
                        target_generation_id=generation_id,
                    )
                    copied_parents = self.qdrant.copy_source_points(
                        previous_parent,
                        parent_collection,
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=source_uri,
                        with_vectors=False,
                        target_generation_id=generation_id,
                    )
                    if copied_children < previous.child_count or copied_parents < previous.parent_count:
                        raise ValueError(
                            f"unchanged source is incomplete in previous generation: {source_uri}"
                        )
                    self.manifest.mark_source(
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=source_uri,
                        status="committed",
                        generation_id=generation_id,
                        parent_count=previous.parent_count,
                        child_count=previous.child_count,
                        quality=previous.quality,
                    )
                    skipped.add(source_uri)
                    continue

                try:
                    parser = ParserDispatcher.get_parser(source["path"])
                    documents, quality = parser.parse_with_quality(source["path"])
                    self._enforce_source_memory_budget(source_uri, documents)
                    quality_by_file[source_uri] = quality.as_dict()
                    if not self._quality_acceptable(quality, documents):
                        raise ValueError("parser_quality_threshold_exceeded")
                    documents = [self._normalize_document(document, source_uri, content_hash) for document in documents]
                    # A retry with the same generation may have left points from
                    # an older file revision. Remove only this staging source;
                    # immutable active generations are never mutated.
                    delete_source = getattr(self.qdrant, "delete_by_file_name_scoped", None)
                    if delete_source is not None:
                        delete_source(
                            child_collection,
                            source_uri,
                            dataset_id=settings.INGEST_DATASET_ID,
                        )
                        delete_source(
                            parent_collection,
                            source_uri,
                            dataset_id=settings.INGEST_DATASET_ID,
                        )
                    self.manifest.clear_source_checkpoints(
                        dataset_id=settings.INGEST_DATASET_ID,
                        generation_id=generation_id,
                        source_uri=source_uri,
                    )
                    parent_count, child_count = self._ingest_source(
                        documents,
                        source_uri=source_uri,
                        content_sha256=content_hash,
                        fingerprint=fingerprint,
                        generation_id=generation_id,
                        child_collection=child_collection,
                        parent_collection=parent_collection,
                    )
                    if parent_count <= 0 or child_count <= 0:
                        raise ValueError("no_chunks_emitted")
                    self.manifest.mark_source(
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=source_uri,
                        status="committed",
                        generation_id=generation_id,
                        parent_count=parent_count,
                        child_count=child_count,
                        quality=quality.as_dict(),
                    )
                    processed.add(source_uri)
                except Exception as exc:
                    failed.add(source_uri)
                    self.manifest.mark_source(
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=source_uri,
                        status="failed",
                        generation_id=generation_id,
                        quality=quality_by_file.get(source_uri),
                        error_code=type(exc).__name__,
                    )
                    logger.exception("Failed to ingest source", source=source_uri)

            # A non-sync run is additive: source files not present in this
            # invocation must be copied from the previous active generation.
            # Otherwise every ordinary incremental run would silently drop old
            # documents because staging starts empty.
            if not sync and previous_child and previous_parent:
                for record in previous_records:
                    if record.source_uri in discovered:
                        continue
                    if record.fingerprint != fingerprint:
                        raise ValueError(
                            "Cannot carry a source across ingestion schemas: "
                            f"{record.source_uri}. Run a full --sync backfill with "
                            "the complete corpus."
                        )
                    self.manifest.stage_active_source(
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=record.source_uri,
                        generation_id=generation_id,
                    )
                    copied_children = self.qdrant.copy_source_points(
                        previous_child,
                        child_collection,
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=record.source_uri,
                        with_vectors=True,
                        target_generation_id=generation_id,
                    )
                    copied_parents = self.qdrant.copy_source_points(
                        previous_parent,
                        parent_collection,
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=record.source_uri,
                        with_vectors=False,
                        target_generation_id=generation_id,
                    )
                    if (
                        record.status != "rolled_back"
                        and (copied_children < record.child_count or copied_parents < record.parent_count)
                    ):
                        raise ValueError(
                            f"carried-forward source is incomplete in previous generation: {record.source_uri}"
                        )
                    self.manifest.mark_source(
                        dataset_id=settings.INGEST_DATASET_ID,
                        source_uri=record.source_uri,
                        status="committed",
                        generation_id=generation_id,
                        parent_count=copied_parents,
                        child_count=copied_children,
                        quality=record.quality,
                    )
                    active_source_uris.add(record.source_uri)

            if sync:
                stale = (
                    {
                        record.source_uri
                        for record in self.manifest.list_sources(
                            settings.INGEST_DATASET_ID,
                            active_generation=previous_generation,
                        )
                        if record.source_uri not in discovered
                    }
                    if previous_generation else set()
                )
                if failed:
                    logger.warning("Skipping stale source removal because ingestion failed", failed=sorted(failed))

            if failed:
                self.manifest.finish_job(
                    job_id=job_id,
                    status="failed",
                    summary={"failed_files": sorted(failed), "generation_id": generation_id},
                )
                return IngestionResult(
                    dataset_id=settings.INGEST_DATASET_ID,
                    discovered_files=discovered,
                    processed_files=processed,
                    failed_files=failed,
                    job_id=job_id,
                    generation_id=generation_id,
                    skipped_files=skipped,
                    quality_by_file=quality_by_file,
                    prune_skipped=sync,
                )

            self._validate_generation(child_collection, parent_collection)
            self.qdrant.switch_aliases(
                child_collection=child_collection,
                parent_collection=parent_collection,
            )
            self.manifest.activate_generation(
                dataset_id=settings.INGEST_DATASET_ID,
                generation_id=generation_id,
                source_uris=active_source_uris,
            )
            self.manifest.mark_removed(
                dataset_id=settings.INGEST_DATASET_ID,
                source_uris=stale,
            )
            self.manifest.finish_job(
                job_id=job_id,
                status="committed",
                active_generation_after=generation_id,
                summary={
                    "processed_files": sorted(processed),
                    "skipped_files": sorted(skipped),
                    "generation_id": generation_id,
                },
            )
            self._prune_old_generations(generation_id)
            self._invalidate_bm25_index()
            return IngestionResult(
                dataset_id=settings.INGEST_DATASET_ID,
                discovered_files=discovered,
                processed_files=processed,
                skipped_files=skipped,
                generation_id=generation_id,
                job_id=job_id,
                quality_by_file=quality_by_file,
            )
        except Exception:
            self.manifest.finish_job(job_id=job_id, status="failed")
            raise

    def _dry_run_versioned(
        self,
        data_path: Path,
        *,
        sync: bool,
        allow_empty_source: bool,
        job_id: str,
        generation_id: str,
        fingerprint: str,
    ) -> IngestionResult:
        discovered: set[str] = set()
        skipped: set[str] = set()
        for source in self._iter_source_files(data_path):
            source_uri = source["source_uri"]
            discovered.add(source_uri)
            content_hash = sha256_file(source["path"])
            previous = self.manifest.get_source(settings.INGEST_DATASET_ID, source_uri)
            if previous and previous.status == "committed" and previous.content_sha256 == content_hash and previous.fingerprint == fingerprint:
                skipped.add(source_uri)
        active_records = self.manifest.list_sources(settings.INGEST_DATASET_ID)
        previous_generation = next(
            (
                record.active_generation
                for record in active_records
                if record.active_generation
            ),
            None,
        )
        planned_pruned_files = (
            {
                record.source_uri
                for record in self.manifest.list_sources(
                    settings.INGEST_DATASET_ID,
                    active_generation=previous_generation,
                )
                if record.source_uri not in discovered
            }
            if sync and previous_generation
            else set()
        )
        if sync and not discovered and not allow_empty_source:
            self.manifest.finish_job(job_id=job_id, status="failed")
            raise ValueError("Refusing destructive sync from an empty source")
        self.manifest.finish_job(
            job_id=job_id,
            status="dry_run",
            summary={
                "discovered": sorted(discovered),
                "skipped": sorted(skipped),
                "planned_pruned_files": sorted(planned_pruned_files),
            },
        )
        return IngestionResult(
            dataset_id=settings.INGEST_DATASET_ID,
            discovered_files=discovered,
            skipped_files=skipped,
            planned_pruned_files=planned_pruned_files,
            dry_run=True,
            job_id=job_id,
            generation_id=generation_id,
        )

    def _recover_alias_activation(self, *, fingerprint: str, vector_dimension: int) -> None:
        """Finish manifest activation after a crash between alias switch and commit.

        Alias updates are atomic in Qdrant, while the SQLite promotion follows
        them.  If the worker dies in that small gap, the next ingestion worker
        can safely reconcile only a generation whose job is still ``running``
        and whose source rows all carry committed working metadata.
        """
        try:
            child = self.qdrant.alias_target(settings.CHILD_COLLECTION)
            parent = self.qdrant.alias_target(settings.PARENT_COLLECTION)
        except AttributeError:
            return
        if not child or not parent:
            return
        child_prefix = f"{settings.CHILD_COLLECTION}__"
        parent_prefix = f"{settings.PARENT_COLLECTION}__"
        if not child.startswith(child_prefix) or not parent.startswith(parent_prefix):
            return
        child_generation = child[len(child_prefix):]
        parent_generation = parent[len(parent_prefix):]
        if not child_generation or child_generation != parent_generation:
            return
        job = self.manifest.get_job_for_generation(
            settings.INGEST_DATASET_ID,
            child_generation,
        )
        if not job or job["status"] != "running":
            return
        records = self.manifest.list_sources(settings.INGEST_DATASET_ID)
        working = {
            record.source_uri
            for record in records
            if record.working_generation == child_generation
            and record.working_status == "committed"
        }
        if not working:
            return
        # Confirm the aliased generation still has the expected native sparse schema
        # before making its metadata visible.
        self.qdrant.validate_collection_schema(
            child,
            schema_fingerprint=fingerprint,
            vector_dimension=vector_dimension,
            expected_distance=Distance.COSINE,
            require_sparse=True,
        )
        self.qdrant.validate_collection_schema(
            parent,
            schema_fingerprint=fingerprint,
            vector_dimension=None,
            expected_distance=None,
        )
        self.manifest.activate_generation(
            dataset_id=settings.INGEST_DATASET_ID,
            generation_id=child_generation,
            source_uris=working,
        )
        stale = {
            record.source_uri
            for record in records
            if record.active_generation and record.source_uri not in working
        }
        self.manifest.mark_removed(
            dataset_id=settings.INGEST_DATASET_ID,
            source_uris=stale,
        )
        self.manifest.finish_job(
            job_id=job["job_id"],
            status="committed",
            active_generation_after=child_generation,
            summary={"recovered_after_alias_switch": True, "generation_id": child_generation},
        )
        logger.warning(
            "Recovered manifest after alias switch",
            generation=child_generation,
            source_count=len(working),
        )

    def _iter_source_files(self, data_path: Path) -> Iterator[dict]:
        supported = set(ParserDispatcher.supported_extensions())
        for path in data_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in supported:
                relative = path.relative_to(data_path).as_posix()
                stat = path.stat()
                yield {
                    "path": path,
                    "source_uri": relative,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }

    @staticmethod
    def _normalize_document(document: RawDocument, source_uri: str, content_sha256: str) -> RawDocument:
        return document.model_copy(update={
            "metadata": document.metadata.model_copy(update={
                "dataset_id": settings.INGEST_DATASET_ID,
                "file_name": source_uri,
                "source_path": source_uri,
                "source_uri": source_uri,
                "content_sha256": content_sha256,
            }),
        })

    @staticmethod
    def _quality_acceptable(quality: ParseQuality, documents: list[RawDocument]) -> bool:
        if not documents:
            return False
        if not settings.INGEST_FAIL_ON_QUALITY:
            return True
        if any(len(document.content.strip()) < settings.INGEST_PARSER_MIN_TEXT_CHARS for document in documents):
            return False
        if quality.characters_emitted <= 0:
            return False
        if quality.pages_seen and quality.pages_empty / quality.pages_seen > settings.INGEST_PARSER_MAX_EMPTY_PAGE_RATIO:
            return False
        if quality.elements_seen and quality.unsupported_elements / quality.elements_seen > settings.INGEST_PARSER_MAX_UNSUPPORTED_RATIO:
            return False
        replacement_ratio = quality.replacement_characters / max(quality.characters_emitted, 1)
        return replacement_ratio <= settings.INGEST_PARSER_MAX_REPLACEMENT_RATIO

    @staticmethod
    def _enforce_source_memory_budget(
        source_uri: str,
        documents: list[RawDocument],
    ) -> None:
        """Fail closed when one parsed source is too large for safe staging.

        Parser libraries may materialize an entire PDF/DOCX before returning;
        this estimate cannot undo that allocation, but it prevents the rest of
        the pipeline from retaining another unbounded copy and makes the
        configured limit observable.  Truly huge files should be split or sent
        through a page-window parser worker.
        """
        estimated_bytes = sum(
            len(document.content.encode("utf-8")) * 2 + 1024
            for document in documents
        )
        limit = settings.INGEST_MAX_MEMORY_MB * 1024 * 1024
        if estimated_bytes > limit:
            raise MemoryError(
                f"source {source_uri} exceeds ingestion memory budget: "
                f"estimated={estimated_bytes} limit={limit}"
            )
        logger.info(
            "Parsed source memory estimate",
            source=source_uri,
            estimated_bytes=estimated_bytes,
            memory_budget_bytes=limit,
        )

    def _ingest_source(
        self,
        documents: list[RawDocument],
        *,
        source_uri: str,
        content_sha256: str,
        fingerprint: str,
        generation_id: str,
        child_collection: str,
        parent_collection: str,
    ) -> tuple[int, int]:
        parent_count = 0
        child_count = 0
        child_batch_index = 0
        for parent_batch_index, (parents, children) in enumerate(iter_parent_child_chunks(documents)):
            parent_points = [self._parent_point(chunk, generation_id) for chunk in parents]
            parent_key = f"{source_uri}:parent:{parent_batch_index}"
            parent_ids = [str(point.id) for point in parent_points]
            if not self._checkpoint_complete(
                source_uri, generation_id, parent_key, parent_collection, parent_ids, content_sha256, fingerprint
            ):
                self.qdrant.upsert_points(parent_collection, parent_points)
            self.manifest.record_batch(
                dataset_id=settings.INGEST_DATASET_ID,
                generation_id=generation_id,
                source_uri=source_uri,
                batch_key=parent_key,
                collection_name=parent_collection,
                point_ids=parent_ids,
                content_sha256=content_sha256,
                fingerprint=fingerprint,
            )
            parent_count += len(parent_points)

            for child_batch in iter_batches(
                children,
                max_items=settings.INGEST_EMBED_BATCH_SIZE,
                max_bytes=max(settings.INGEST_QDRANT_WRITE_MAX_BYTES, 1),
                size_of=lambda chunk: len(chunk.content.encode("utf-8")) + 256,
            ):
                child_ids = [chunk.chunk_id for chunk in child_batch]
                child_key = f"{source_uri}:child:{child_batch_index}"
                checkpointed = self._checkpoint_complete(
                    source_uri, generation_id, child_key, child_collection, child_ids, content_sha256, fingerprint
                )
                if not checkpointed:
                    vectors = self.embedder.embed([chunk.content for chunk in child_batch])
                    child_points = [self._child_point(chunk, vector, generation_id) for chunk, vector in zip(child_batch, vectors)]
                    self.qdrant.upsert_points(child_collection, child_points)
                self.manifest.record_batch(
                    dataset_id=settings.INGEST_DATASET_ID,
                    generation_id=generation_id,
                    source_uri=source_uri,
                    batch_key=child_key,
                    collection_name=child_collection,
                    point_ids=child_ids,
                    content_sha256=content_sha256,
                    fingerprint=fingerprint,
                )
                child_count += len(child_batch)
                child_batch_index += 1
        return parent_count, child_count

    def _checkpoint_complete(
        self,
        source_uri: str,
        generation_id: str,
        batch_key: str,
        collection_name: str,
        point_ids: list[str],
        content_sha256: str,
        fingerprint: str,
    ) -> bool:
        checkpoint = self.manifest.batch_committed(
            dataset_id=settings.INGEST_DATASET_ID,
            generation_id=generation_id,
            source_uri=source_uri,
            batch_key=batch_key,
            collection_name=collection_name,
            content_sha256=content_sha256,
            fingerprint=fingerprint,
        )
        candidate_ids = checkpoint or point_ids
        existing_ids = self.qdrant.get_existing_ids(collection_name, candidate_ids)
        return existing_ids >= set(point_ids)

    @staticmethod
    def _parent_point(chunk: Chunk, generation_id: str) -> PointStruct:
        metadata = chunk.metadata
        return PointStruct(
            id=chunk.chunk_id,
            vector={},
            payload={
                "content": chunk.content,
                "parent_id": None,
                "file_name": metadata.file_name,
                "file_type": metadata.file_type,
                "source_path": metadata.source_path,
                "source_uri": metadata.source_uri or metadata.source_path,
                "dataset_id": metadata.dataset_id,
                "generation_id": generation_id,
                "document_version": metadata.document_version,
                "content_sha256": metadata.content_sha256,
                "section_title": metadata.section_title,
                "page_number": metadata.page_number,
                "structural_anchor": metadata.structural_anchor,
            },
        )

    @staticmethod
    def _child_point(chunk: Chunk, vector: list[float], generation_id: str) -> PointStruct:
        metadata = chunk.metadata
        return PointStruct(
            id=chunk.chunk_id,
            vector={
                "": vector,
                settings.QDRANT_SPARSE_VECTOR_NAME: sparse_document(chunk.content),
            },
            payload={
                "content": chunk.content,
                "parent_id": chunk.parent_id,
                "file_name": metadata.file_name,
                "file_type": metadata.file_type,
                "source_path": metadata.source_path,
                "source_uri": metadata.source_uri or metadata.source_path,
                "dataset_id": metadata.dataset_id,
                "generation_id": generation_id,
                "document_version": metadata.document_version,
                "content_sha256": metadata.content_sha256,
                "section_title": metadata.section_title,
                "page_number": metadata.page_number,
                "structural_anchor": metadata.structural_anchor,
            },
        )

    def _validate_generation(self, child_collection: str, parent_collection: str) -> None:
        """Run bounded integrity checks before aliases can become visible."""
        self.qdrant.validate_sparse_coverage(
            child_collection,
            sparse_vector_name=settings.QDRANT_SPARSE_VECTOR_NAME,
        )
        # Sample child references and resolve exactly those parent IDs.  Keeping
        # the first 10k parent IDs would falsely reject a large generation when
        # a sampled child points to a parent outside that arbitrary prefix.
        sample_parent_ids: set[str] = set()
        for point in self.qdrant.iter_scroll(
            child_collection,
            batch_size=settings.INGEST_QDRANT_WRITE_BATCH_SIZE,
            with_payload=True,
            with_vectors=False,
            max_points=10_000,
        ):
            parent_id = (point.payload or {}).get("parent_id")
            if parent_id:
                sample_parent_ids.add(str(parent_id))
        sample_missing: list[str] = []
        for batch in iter_batches(
            sorted(sample_parent_ids),
            max_items=settings.INGEST_QDRANT_WRITE_BATCH_SIZE,
            max_bytes=settings.INGEST_QDRANT_WRITE_MAX_BYTES,
            size_of=lambda value: len(value) + 64,
        ):
            found = {str(point.id) for point in self.qdrant.get_by_ids(parent_collection, batch)}
            sample_missing.extend(sorted(set(batch) - found))
            if len(sample_missing) >= 10:
                break
        if sample_missing:
            raise ValueError(f"child-parent referential integrity failed: {sample_missing}")

    def rollback(self, generation_id: str) -> None:
        """Atomically restore retrieval aliases to a retained generation."""
        if not generation_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in generation_id
        ):
            raise ValueError("generation_id contains unsupported collection-name characters")
        child_collection = f"{settings.CHILD_COLLECTION}__{generation_id}"
        parent_collection = f"{settings.PARENT_COLLECTION}__{generation_id}"
        if not self.qdrant._collection_exists(child_collection) or not self.qdrant._collection_exists(parent_collection):
            raise ValueError(f"generation not found or incomplete: {generation_id}")
        fingerprint = pipeline_fingerprint(embedding_dimension=self.embedder.dimension)
        self.qdrant.validate_collection_schema(
            child_collection,
            schema_fingerprint=fingerprint,
            vector_dimension=self.embedder.dimension,
            expected_distance=Distance.COSINE,
            required_payload_indexes=("dataset_id", "file_name", "source_uri", "generation_id"),
            require_sparse=True,
        )
        self.qdrant.validate_collection_schema(
            parent_collection,
            schema_fingerprint=fingerprint,
            vector_dimension=None,
            expected_distance=None,
            required_payload_indexes=("dataset_id", "file_name", "source_uri", "generation_id"),
        )
        self._validate_generation(child_collection, parent_collection)
        self.qdrant.switch_aliases(
            child_collection=child_collection,
            parent_collection=parent_collection,
        )
        source_uris = (
            self.qdrant.list_file_names_by_dataset(
                child_collection,
                dataset_id=settings.INGEST_DATASET_ID,
            )
            | self.qdrant.list_file_names_by_dataset(
                parent_collection,
                dataset_id=settings.INGEST_DATASET_ID,
            )
        )
        self.manifest.mark_rollback(
            dataset_id=settings.INGEST_DATASET_ID,
            generation_id=generation_id,
            source_uris=source_uris,
        )
        self._invalidate_bm25_index()

    def _prune_old_generations(self, active_generation: str) -> None:
        """Delete committed staging generations outside the rollback window."""
        keep = set(
            self.manifest.committed_generations(
                settings.INGEST_DATASET_ID,
                limit=settings.INGEST_GENERATION_RETENTION,
            )
        )
        keep.add(active_generation)
        active_targets = {
            self.qdrant.alias_target(settings.CHILD_COLLECTION),
            self.qdrant.alias_target(settings.PARENT_COLLECTION),
        }
        for generation in self.manifest.committed_generations(
            settings.INGEST_DATASET_ID,
            limit=max(settings.INGEST_GENERATION_RETENTION + 100, 100),
        ):
            if generation in keep:
                continue
            child_collection = f"{settings.CHILD_COLLECTION}__{generation}"
            parent_collection = f"{settings.PARENT_COLLECTION}__{generation}"
            if child_collection in active_targets or parent_collection in active_targets:
                continue
            try:
                self.qdrant.delete_collection(child_collection)
                self.qdrant.delete_collection(parent_collection)
            except Exception:
                # Activation already succeeded; retention cleanup must not turn
                # a healthy ingestion into a failed job.  The next run retries.
                logger.exception("Failed to prune old ingestion generation", generation=generation)

    # ================================================================
    # Private methods — từng bước xử lý
    # ================================================================

    def _init_collections(self) -> None:
        """Tạo 2 collections trong Qdrant nếu chưa tồn tại."""
        dimension = getattr(self.embedder, "dimension", settings.EMBEDDING_SIZE)
        fingerprint = pipeline_fingerprint(embedding_dimension=dimension)
        vector_create = self.qdrant.create_vector_collection
        payload_create = self.qdrant.create_payload_collection
        if self._accepts_keyword(vector_create, "schema_fingerprint"):
            vector_create(
                settings.CHILD_COLLECTION,
                schema_fingerprint=fingerprint,
                vector_dimension=dimension,
            )
        else:
            # Lightweight PR11 test doubles retain the old one-argument API.
            vector_create(settings.CHILD_COLLECTION)
        if self._accepts_keyword(payload_create, "schema_fingerprint"):
            payload_create(
                settings.PARENT_COLLECTION,
                schema_fingerprint=fingerprint,
            )
        else:
            payload_create(settings.PARENT_COLLECTION)
        # Index để safe-replace/sync filter đúng dataset + relative file.
        for coll in (settings.CHILD_COLLECTION, settings.PARENT_COLLECTION):
            for field_name in ("dataset_id", "file_name"):
                self.qdrant.create_payload_index(coll, field_name)

    @staticmethod
    def _accepts_keyword(function: Callable, name: str) -> bool:
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            return False
        return name in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

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
                        "source_uri": rel,
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
                    "source_uri": file_name,
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
        removed_files = self._find_removed_files(dataset_id, discovered_files)
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

    def _find_removed_files(
        self,
        dataset_id: str,
        discovered_files: set[str],
    ) -> set[str]:
        """Return stale file names scoped to one dataset, without mutating Qdrant."""
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
        return stored_files - discovered_files

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
                vector={
                    "": chunk.embedding,
                    settings.QDRANT_SPARSE_VECTOR_NAME: sparse_document(chunk.content),
                },
                payload={
                    "content": chunk.content,
                    "parent_id": chunk.parent_id,
                    "file_name": chunk.metadata.file_name,
                    "file_type": chunk.metadata.file_type,
                    "source_path": chunk.metadata.source_path,
                    "source_uri": chunk.metadata.source_uri or chunk.metadata.source_path,
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
                    "source_uri": chunk.metadata.source_uri or chunk.metadata.source_path,
                    "dataset_id": chunk.metadata.dataset_id,
                    "section_title": chunk.metadata.section_title,
                    "page_number": chunk.metadata.page_number,   # ★ THÊM
                },
            )
            for chunk in chunks
        ]

        self.qdrant.upsert_points(settings.PARENT_COLLECTION, points)
