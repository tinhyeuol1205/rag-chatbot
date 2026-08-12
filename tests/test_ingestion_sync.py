"""Dataset-scoped directory sync tests."""

import pytest

from core.config import settings
from ingestion.models import Chunk, DocumentMetadata, EmbeddedChunk, RawDocument
from ingestion.pipeline import IngestionPipeline, IngestionResult, ParseBatch, PreparedFile


class _SyncQdrant:
    def __init__(self):
        self.points = {
            settings.PARENT_COLLECTION: {},
            settings.CHILD_COLLECTION: {},
        }
        self.deleted: list[tuple[str, str]] = []

    def create_vector_collection(self, _name):
        return None

    def create_payload_collection(self, _name):
        return None

    def create_payload_index(self, _name, _field):
        return None

    def seed(self, collection_name, point_id, *, dataset_id, file_name):
        self.points[collection_name][point_id] = {
            "dataset_id": dataset_id,
            "file_name": file_name,
        }

    def list_ids_by_source(self, collection_name, *, dataset_id, file_name):
        return {
            point_id
            for point_id, payload in self.points[collection_name].items()
            if payload["dataset_id"] == dataset_id and payload["file_name"] == file_name
        }

    def list_file_names_by_dataset(self, collection_name, *, dataset_id):
        return {
            payload["file_name"]
            for payload in self.points[collection_name].values()
            if payload["dataset_id"] == dataset_id
        }

    def upsert_points(self, collection_name, points):
        for point in points:
            self.points[collection_name][str(point.id)] = {
                "dataset_id": point.payload["dataset_id"],
                "file_name": point.payload["file_name"],
            }

    def delete_by_ids(self, collection_name, ids):
        for point_id in ids:
            self.deleted.append((collection_name, point_id))
            self.points[collection_name].pop(point_id, None)


def _pipeline(qdrant):
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.qdrant = qdrant
    pipeline.embedder = None
    pipeline._invalidate_bm25_index = lambda: None
    return pipeline


def _prepared(file_name: str, dataset_id: str = settings.INGEST_DATASET_ID) -> PreparedFile:
    metadata = DocumentMetadata(
        file_name=file_name,
        file_type="md",
        source_path=file_name,
        dataset_id=dataset_id,
    )
    parent = Chunk(content="parent", position="0:0", metadata=metadata, is_parent=True)
    child = Chunk(content="child", parent_id=parent.chunk_id, position="0:0:0", metadata=metadata)
    embedded = EmbeddedChunk(
        chunk_id=child.chunk_id,
        content=child.content,
        embedding=[0.1],
        parent_id=child.parent_id,
        metadata=metadata,
    )
    return PreparedFile(dataset_id, file_name, [parent], [embedded])


def _raw(file_name: str) -> RawDocument:
    return RawDocument(
        content="content",
        metadata=DocumentMetadata(file_name=file_name, file_type="md"),
    )


def test_sync_removes_deleted_file_in_same_dataset(tmp_path):
    qdrant = _SyncQdrant()
    qdrant.seed(settings.PARENT_COLLECTION, "removed-parent", dataset_id="sample_docs", file_name="removed.md")
    qdrant.seed(settings.CHILD_COLLECTION, "removed-child", dataset_id="sample_docs", file_name="removed.md")
    pipeline = _pipeline(qdrant)
    pipeline._parse_all_files = lambda _path: ParseBatch(
        discovered_files={"keep.md"},
        documents_by_file={"keep.md": [_raw("keep.md")]},
    )
    pipeline._prepare_file = lambda dataset_id, file_name, documents: _prepared(file_name, dataset_id)
    source_dir = tmp_path / "docs"
    source_dir.mkdir()

    result = pipeline.run(str(source_dir), sync=True)

    assert isinstance(result, IngestionResult)
    assert result.pruned_files == {"removed.md"}
    assert (settings.PARENT_COLLECTION, "removed-parent") in qdrant.deleted
    assert (settings.CHILD_COLLECTION, "removed-child") in qdrant.deleted


def test_sync_does_not_touch_other_dataset(tmp_path):
    qdrant = _SyncQdrant()
    qdrant.seed(settings.PARENT_COLLECTION, "hr-parent", dataset_id="hr_docs", file_name="policy.md")
    qdrant.seed(settings.CHILD_COLLECTION, "hr-child", dataset_id="hr_docs", file_name="policy.md")
    pipeline = _pipeline(qdrant)
    pipeline._parse_all_files = lambda _path: ParseBatch(discovered_files={"keep.md"})
    source_dir = tmp_path / "empty"
    source_dir.mkdir()

    result = pipeline.run(str(source_dir), sync=True)

    assert result.pruned_files == set()
    assert qdrant.deleted == []
    assert "hr-parent" in qdrant.points[settings.PARENT_COLLECTION]
    assert "hr-child" in qdrant.points[settings.CHILD_COLLECTION]


def test_parse_failure_is_not_treated_as_deleted_file(tmp_path):
    qdrant = _SyncQdrant()
    qdrant.seed(settings.PARENT_COLLECTION, "broken-parent", dataset_id="sample_docs", file_name="broken.md")
    qdrant.seed(settings.CHILD_COLLECTION, "removed-child", dataset_id="sample_docs", file_name="removed.md")
    pipeline = _pipeline(qdrant)
    pipeline._parse_all_files = lambda _path: ParseBatch(
        discovered_files={"keep.md", "broken.md"},
        documents_by_file={"keep.md": [_raw("keep.md")]},
        failed_files={"broken.md"},
    )
    pipeline._prepare_file = lambda dataset_id, file_name, documents: _prepared(file_name, dataset_id)
    source_dir = tmp_path / "docs"
    source_dir.mkdir()

    result = pipeline.run(str(source_dir), sync=True)

    assert result.prune_skipped is True
    assert result.failed_files == {"broken.md"}
    assert qdrant.deleted == []
    assert "broken-parent" in qdrant.points[settings.PARENT_COLLECTION]
    assert "removed-child" in qdrant.points[settings.CHILD_COLLECTION]


def test_empty_directory_requires_explicit_sync_to_prune(tmp_path):
    qdrant = _SyncQdrant()
    qdrant.seed(settings.PARENT_COLLECTION, "old-parent", dataset_id="sample_docs", file_name="old.md")
    qdrant.seed(settings.CHILD_COLLECTION, "old-child", dataset_id="sample_docs", file_name="old.md")
    pipeline = _pipeline(qdrant)
    pipeline._parse_all_files = lambda _path: ParseBatch()
    source_dir = tmp_path / "empty"
    source_dir.mkdir()

    no_sync_result = pipeline.run(str(source_dir), sync=False)
    assert no_sync_result.pruned_files == set()
    assert qdrant.deleted == []

    with pytest.raises(ValueError, match="empty source"):
        pipeline.run(str(source_dir), sync=True)

    sync_result = pipeline.run(str(source_dir), sync=True, allow_empty_source=True)
    assert sync_result.pruned_files == {"old.md"}
    assert {
        (settings.PARENT_COLLECTION, "old-parent"),
        (settings.CHILD_COLLECTION, "old-child"),
    } <= set(qdrant.deleted)


def test_non_directory_source_is_rejected(tmp_path):
    qdrant = _SyncQdrant()
    pipeline = _pipeline(qdrant)
    source_file = tmp_path / "not-a-directory.md"
    source_file.write_text("content", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        pipeline.run(str(source_file), sync=True, allow_empty_source=True)


def test_dry_run_reports_prune_without_mutating_qdrant(tmp_path):
    qdrant = _SyncQdrant()
    qdrant.seed(settings.PARENT_COLLECTION, "old-parent", dataset_id="sample_docs", file_name="old.md")
    qdrant.seed(settings.CHILD_COLLECTION, "old-child", dataset_id="sample_docs", file_name="old.md")
    pipeline = _pipeline(qdrant)
    pipeline._parse_all_files = lambda _path: ParseBatch(discovered_files={"keep.md"})
    source_dir = tmp_path / "docs"
    source_dir.mkdir()

    result = pipeline.run(str(source_dir), sync=True, dry_run=True)

    assert result.dry_run is True
    assert result.planned_pruned_files == {"old.md"}
    assert result.pruned_files == set()
    assert qdrant.deleted == []
