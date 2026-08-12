"""PR11 bounded batching, manifest and generation-switch tests."""

import pytest
from qdrant_client import QdrantClient

from core.config import settings
from core.db.qdrant import QdrantConnector
from ingestion.batching import iter_batches
from ingestion.manifest import ManifestStore, pipeline_fingerprint
from ingestion.models import DocumentMetadata, RawDocument
from ingestion.parsers.markdown_parser import MarkdownParser
from ingestion.pipeline import IngestionPipeline


def test_iter_batches_enforces_count_and_byte_budget():
    batches = list(iter_batches(["a" * 10, "b" * 10, "c" * 10], max_items=2, max_bytes=25))

    assert [len(batch) for batch in batches] == [2, 1]
    assert all(sum(len(item) for item in batch) <= 20 for batch in batches)


def test_manifest_skip_requires_content_and_pipeline_fingerprint():
    store = ManifestStore(":memory:")
    fingerprint = pipeline_fingerprint(embedding_dimension=384)
    previous = store.record_discovered(
        dataset_id="docs",
        source_uri="a.md",
        content_sha256="hash-a",
        size_bytes=10,
        mtime_ns=1,
        fingerprint=fingerprint,
        generation_id="g1",
    )
    assert previous is None
    store.mark_source(
        dataset_id="docs",
        source_uri="a.md",
        status="committed",
        generation_id="g1",
        active_generation="g1",
    )

    old = store.get_source("docs", "a.md")
    assert old is not None and old.status == "committed"
    unchanged = store.record_discovered(
        dataset_id="docs",
        source_uri="a.md",
        content_sha256="hash-a",
        size_bytes=10,
        mtime_ns=2,
        fingerprint=fingerprint,
        generation_id="g2",
    )
    assert unchanged is not None and unchanged.content_sha256 == "hash-a"
    changed = store.record_discovered(
        dataset_id="docs",
        source_uri="a.md",
        content_sha256="hash-b",
        size_bytes=11,
        mtime_ns=3,
        fingerprint=fingerprint,
        generation_id="g3",
    )
    assert changed is not None
    assert store.get_source("docs", "a.md").status == "discovered"

    store.record_batch(
        dataset_id="docs",
        generation_id="g3",
        source_uri="a.md",
        batch_key="child:0",
        collection_name="children",
        point_ids=["p1"],
        content_sha256="hash-b",
        fingerprint=fingerprint,
    )
    assert store.batch_committed(
        dataset_id="docs",
        generation_id="g3",
        source_uri="a.md",
        batch_key="child:0",
        collection_name="children",
        content_sha256="hash-b",
        fingerprint=fingerprint,
    ) == ["p1"]


def test_markdown_stream_supports_tilde_fences_and_structural_anchor(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(
        "# Intro\ntext long enough for quality.\n\n~~~python\n# not a heading\n~~~\n"
        "\n## Next\nmore content here.\n",
        encoding="utf-8",
    )

    documents, quality = MarkdownParser().parse_with_quality(path)

    assert len(documents) == 2
    assert "# not a heading" in documents[0].content
    assert documents[0].metadata.structural_anchor == "line:1"
    assert documents[1].metadata.structural_anchor.startswith("line:")
    assert quality.characters_emitted > 0


def test_chunk_id_uses_stable_structural_anchor():
    first = RawDocument(
        content="stable section content",
        metadata=DocumentMetadata(
            file_name="a.md",
            file_type="md",
            source_path="a.md",
            source_uri="a.md",
            structural_anchor="heading:security",
            dataset_id="docs",
        ),
    )
    shifted = first.model_copy(update={
        "metadata": first.metadata.model_copy(update={"offset_start": 900}),
    })
    from ingestion.chunking.parent_child import parent_child_chunk

    assert parent_child_chunk([first])[0][0].chunk_id == parent_child_chunk([shifted])[0][0].chunk_id


def test_versioned_pipeline_skips_unchanged_and_rolls_back(tmp_path, monkeypatch):
    old_instance, old_client = QdrantConnector._instance, QdrantConnector._client
    QdrantConnector._instance = None
    QdrantConnector._client = QdrantClient(":memory:")
    try:
        qdrant = QdrantConnector()

        class FakeEmbedder:
            dimension = 384

            def __init__(self):
                self.calls = 0

            def embed(self, texts):
                self.calls += 1
                return [[0.01] * 384 for _ in texts]

        embedder = FakeEmbedder()
        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline.qdrant = qdrant
        pipeline.embedder = embedder
        pipeline.manifest = ManifestStore(":memory:")
        pipeline._invalidate_bm25_index = lambda: None
        source = tmp_path / "doc.md"
        source.write_text("# Policy\n\nThis document contains enough text to be indexed safely.\n", encoding="utf-8")

        first = pipeline.run(str(tmp_path), generation_id="g1")
        calls_after_first = embedder.calls
        second = pipeline.run(str(tmp_path), generation_id="g2")

        assert first.processed_files == {"doc.md"}
        assert second.skipped_files == {"doc.md"}
        assert embedder.calls == calls_after_first
        assert qdrant.alias_target(settings.CHILD_COLLECTION) == f"{settings.CHILD_COLLECTION}__g2"

        pipeline.rollback("g1")
        assert qdrant.alias_target(settings.CHILD_COLLECTION) == f"{settings.CHILD_COLLECTION}__g1"
    finally:
        QdrantConnector._instance, QdrantConnector._client = old_instance, old_client


def test_non_sync_generation_carries_forward_absent_sources(tmp_path):
    old_instance, old_client = QdrantConnector._instance, QdrantConnector._client
    QdrantConnector._instance = None
    QdrantConnector._client = QdrantClient(":memory:")
    try:
        qdrant = QdrantConnector()

        class FakeEmbedder:
            dimension = 384

            def embed(self, texts):
                return [[0.01] * 384 for _ in texts]

        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline.qdrant = qdrant
        pipeline.embedder = FakeEmbedder()
        pipeline.manifest = ManifestStore(":memory:")
        pipeline._invalidate_bm25_index = lambda: None
        (tmp_path / "old.md").write_text("old policy content that is long enough.", encoding="utf-8")
        pipeline.run(str(tmp_path), generation_id="g1")
        (tmp_path / "new.md").write_text("new policy content that is long enough.", encoding="utf-8")

        result = pipeline.run(str(tmp_path), generation_id="g2", sync=False)

        assert result.failed_files == set()
        points = list(qdrant.iter_scroll(
            f"{settings.CHILD_COLLECTION}__g2",
            with_payload=True,
            with_vectors=False,
        ))
        assert {point.payload["source_uri"] for point in points} == {"old.md", "new.md"}
    finally:
        QdrantConnector._instance, QdrantConnector._client = old_instance, old_client


def test_qdrant_retry_retries_transient_only(monkeypatch):
    connector = QdrantConnector.__new__(QdrantConnector)
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary")
        return "ok"

    monkeypatch.setattr(settings, "INGEST_QDRANT_MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "INGEST_RETRY_BASE_SECONDS", 0.0)
    assert connector._with_retry(operation, operation="test") == "ok"
    assert attempts["count"] == 3

    with pytest.raises(ValueError):
        connector._with_retry(lambda: (_ for _ in ()).throw(ValueError("schema")), operation="schema")
