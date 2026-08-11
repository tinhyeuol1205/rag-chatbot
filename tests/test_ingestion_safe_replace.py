"""Failure-injection tests for ingestion prepare/commit ordering."""

import pytest

from core.config import settings
from ingestion.models import Chunk, DocumentMetadata, EmbeddedChunk, RawDocument
from ingestion.pipeline import IngestionPipeline, ParseBatch, PreparedFile


class _FakeQdrant:
    def __init__(self, *, fail_collection: str | None = None):
        self.points: dict[str, dict[str, dict]] = {
            settings.PARENT_COLLECTION: {},
            settings.CHILD_COLLECTION: {},
        }
        self.fail_collection = fail_collection
        self.calls: list[tuple[str, str]] = []
        self.deleted_by_collection: dict[str, set[str]] = {
            settings.PARENT_COLLECTION: set(),
            settings.CHILD_COLLECTION: set(),
        }

    @property
    def existing(self) -> set[str]:
        return {
            point_id
            for collection in self.points.values()
            for point_id in collection
        }

    def seed(self, collection_name: str, point_id: str, *, file_name: str = "doc.md"):
        self.points[collection_name][point_id] = {
            "id": point_id,
            "payload": {
                "dataset_id": settings.INGEST_DATASET_ID,
                "file_name": file_name,
            },
        }

    def create_vector_collection(self, _collection_name):
        return None

    def create_payload_collection(self, _collection_name):
        return None

    def create_payload_index(self, _collection_name, _field_name):
        return None

    def list_ids_by_source(self, collection_name, *, dataset_id, file_name):
        return {
            point_id
            for point_id, point in self.points[collection_name].items()
            if point["payload"].get("dataset_id") == dataset_id
            and point["payload"].get("file_name") == file_name
        }

    def list_file_names_by_dataset(self, collection_name, *, dataset_id):
        return {
            point["payload"].get("file_name")
            for point in self.points[collection_name].values()
            if point["payload"].get("dataset_id") == dataset_id
        }

    def upsert_points(self, collection_name, points):
        self.calls.append(("upsert", collection_name))
        if collection_name == self.fail_collection:
            raise RuntimeError(f"upsert failed for {collection_name}")
        for point in points:
            self.points[collection_name][str(point.id)] = {
                "id": str(point.id),
                "payload": point.payload,
            }

    def delete_by_ids(self, collection_name, ids):
        self.calls.append(("delete", collection_name))
        self.deleted_by_collection[collection_name].update(ids)
        for point_id in ids:
            self.points[collection_name].pop(point_id, None)


class _FailingEmbedder:
    def embed(self, _texts):
        raise RuntimeError("embedding failed")


def _pipeline(qdrant, embedder=None):
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.qdrant = qdrant
    pipeline.embedder = embedder
    pipeline._invalidate_bm25_index = lambda: None
    return pipeline


def _prepared_file() -> PreparedFile:
    metadata = DocumentMetadata(
        file_name="doc.md",
        file_type="md",
        source_path="doc.md",
        dataset_id=settings.INGEST_DATASET_ID,
    )
    parent = Chunk(content="new parent", position="0:0", metadata=metadata, is_parent=True)
    child = Chunk(
        content="new child",
        parent_id=parent.chunk_id,
        position="0:0:0",
        metadata=metadata,
    )
    embedded = EmbeddedChunk(
        chunk_id=child.chunk_id,
        content=child.content,
        embedding=[0.1, 0.2],
        parent_id=child.parent_id,
        metadata=metadata,
    )
    return PreparedFile(
        dataset_id=settings.INGEST_DATASET_ID,
        file_name="doc.md",
        parents=[parent],
        children=[embedded],
    )


def test_embedding_failure_does_not_delete_existing_points(tmp_path):
    qdrant = _FakeQdrant()
    qdrant.seed(settings.PARENT_COLLECTION, "old-parent")
    qdrant.seed(settings.CHILD_COLLECTION, "old-child")
    pipeline = _pipeline(qdrant, _FailingEmbedder())
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    pipeline._parse_all_files = lambda _path: ParseBatch(
        discovered_files={"doc.md"},
        documents_by_file={
            "doc.md": [RawDocument(
                content="document",
                metadata=DocumentMetadata(file_name="doc.md", file_type="md"),
            )],
        },
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        pipeline.run(str(source_dir))

    assert qdrant.deleted_by_collection[settings.PARENT_COLLECTION] == set()
    assert qdrant.deleted_by_collection[settings.CHILD_COLLECTION] == set()
    assert {"old-parent", "old-child"} <= qdrant.existing


def test_child_upsert_failure_does_not_delete_old_ids():
    qdrant = _FakeQdrant(fail_collection=settings.CHILD_COLLECTION)
    qdrant.seed(settings.PARENT_COLLECTION, "old-parent")
    qdrant.seed(settings.CHILD_COLLECTION, "old-child")
    pipeline = _pipeline(qdrant)

    with pytest.raises(RuntimeError, match="upsert failed"):
        pipeline._commit_file(_prepared_file())

    assert qdrant.deleted_by_collection[settings.PARENT_COLLECTION] == set()
    assert qdrant.deleted_by_collection[settings.CHILD_COLLECTION] == set()
    assert {"old-parent", "old-child"} <= qdrant.existing
    assert qdrant.calls[:2] == [
        ("upsert", settings.PARENT_COLLECTION),
        ("upsert", settings.CHILD_COLLECTION),
    ]


def test_success_deletes_only_stale_ids():
    qdrant = _FakeQdrant()
    qdrant.seed(settings.PARENT_COLLECTION, "old-parent")
    qdrant.seed(settings.CHILD_COLLECTION, "old-child")
    pipeline = _pipeline(qdrant)

    pipeline._commit_file(_prepared_file())

    assert qdrant.deleted_by_collection[settings.PARENT_COLLECTION] == {"old-parent"}
    assert qdrant.deleted_by_collection[settings.CHILD_COLLECTION] == {"old-child"}
    assert qdrant.calls == [
        ("upsert", settings.PARENT_COLLECTION),
        ("upsert", settings.CHILD_COLLECTION),
        ("delete", settings.CHILD_COLLECTION),
        ("delete", settings.PARENT_COLLECTION),
    ]


def test_prepare_commit_preserves_parent_before_child_order():
    qdrant = _FakeQdrant()
    pipeline = _pipeline(qdrant)

    pipeline._commit_file(_prepared_file())

    upserts = [call for call in qdrant.calls if call[0] == "upsert"]
    assert upserts == [
        ("upsert", settings.PARENT_COLLECTION),
        ("upsert", settings.CHILD_COLLECTION),
    ]


def test_both_collection_payloads_include_dataset_id():
    qdrant = _FakeQdrant()
    pipeline = _pipeline(qdrant)

    pipeline._commit_file(_prepared_file())

    for collection_name in (settings.PARENT_COLLECTION, settings.CHILD_COLLECTION):
        payloads = qdrant.points[collection_name].values()
        assert all(payload["payload"]["dataset_id"] == settings.INGEST_DATASET_ID for payload in payloads)
