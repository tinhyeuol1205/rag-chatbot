"""Fail-fast corpus and generation checks for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from evaluation.dataset import EvalSuite
from ingestion.manifest import ManifestStore, pipeline_fingerprint
from retrieval.scope import RetrievalScope

logger = get_logger(__name__)


class EvaluationPreflightError(ValueError):
    """Raised before a runner can make an LLM call."""


@dataclass(frozen=True)
class PreflightResult:
    suite_id: str
    suite_version: str
    dataset_id: str
    child_collection: str
    parent_collection: str
    child_generation: str | None
    parent_generation: str | None
    pipeline_fingerprint: str
    active_sources: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "dataset_id": self.dataset_id,
            "child_collection": self.child_collection,
            "parent_collection": self.parent_collection,
            "child_generation": self.child_generation,
            "parent_generation": self.parent_generation,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "active_sources": list(self.active_sources),
            **self.metadata,
        }


def run_preflight(
    suite: EvalSuite,
    *,
    connector: QdrantConnector | Any | None = None,
    manifest: ManifestStore | Any | None = None,
    require_configured_dataset: bool = True,
) -> PreflightResult:
    """Validate aliases, schema and source coverage without mutating state."""
    if require_configured_dataset and settings.INGEST_DATASET_ID != suite.dataset_id:
        raise EvaluationPreflightError(
            "evaluation suite dataset_id does not match configured INGEST_DATASET_ID"
        )
    if not suite.samples:
        raise EvaluationPreflightError("evaluation suite is empty")

    connector = connector or QdrantConnector()
    child_collection = connector.alias_target(settings.CHILD_COLLECTION)
    parent_collection = connector.alias_target(settings.PARENT_COLLECTION)
    if not child_collection or not parent_collection:
        raise EvaluationPreflightError("active Qdrant child/parent aliases are missing")
    child_generation = _generation_from_collection(child_collection, settings.CHILD_COLLECTION)
    parent_generation = _generation_from_collection(parent_collection, settings.PARENT_COLLECTION)
    if not child_generation or not parent_generation:
        raise EvaluationPreflightError(
            "active Qdrant aliases do not identify concrete versioned generations"
        )
    if child_generation and parent_generation and child_generation != parent_generation:
        raise EvaluationPreflightError("active child/parent aliases point to different generations")

    expected_fingerprint = pipeline_fingerprint(
        embedding_dimension=settings.EMBEDDING_SIZE,
        extra={"source_namespace": suite.dataset_id},
    )
    connector.validate_collection_schema(
        child_collection,
        schema_fingerprint=expected_fingerprint,
        vector_dimension=settings.EMBEDDING_SIZE,
        expected_distance=_distance_cosine(),
        required_payload_indexes=("dataset_id", "file_name", "source_uri", "generation_id"),
        require_sparse=True,
    )
    connector.validate_collection_schema(
        parent_collection,
        schema_fingerprint=expected_fingerprint,
        vector_dimension=None,
        expected_distance=None,
        required_payload_indexes=("dataset_id", "file_name", "source_uri", "generation_id"),
    )

    active_sources = _source_labels_from_qdrant(connector, parent_collection, suite.dataset_id)
    if manifest is not None:
        records = manifest.list_sources(suite.dataset_id, active_generation=parent_generation)
        manifest_sources = {
            _source_label(record.source_uri)
            for record in records
            if getattr(record, "status", "") == "committed"
        }
        # Qdrant is the serving source of truth.  Manifest metadata can enrich
        # diagnostics, but must not make a source appear active when its points
        # are absent from the aliased collection.
        if not active_sources and manifest_sources:
            logger.warning("Qdrant source scan returned no points; manifest cannot satisfy source coverage")
    expected = set(suite.expected_sources)
    missing = sorted(expected - set(active_sources))
    if missing:
        raise EvaluationPreflightError(
            f"active corpus {suite.dataset_id!r} is missing expected sources: {missing}"
        )

    return PreflightResult(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        dataset_id=suite.dataset_id,
        child_collection=child_collection,
        parent_collection=parent_collection,
        child_generation=child_generation,
        parent_generation=parent_generation,
        pipeline_fingerprint=expected_fingerprint,
        active_sources=tuple(sorted(active_sources)),
        metadata={"configured_dataset_id": settings.INGEST_DATASET_ID},
    )


def _source_labels_from_qdrant(connector: Any, collection: str, dataset_id: str) -> tuple[str, ...]:
    scope_filter = RetrievalScope((dataset_id,)).qdrant_filter()
    labels: set[str] = set()
    try:
        points = connector.iter_scroll(
            collection,
            batch_size=500,
            scroll_filter=scope_filter,
            with_payload=True,
            with_vectors=False,
            max_points=50_000,
        )
    except (AttributeError, TypeError):
        try:
            points = connector.scroll_all(collection, scroll_filter=scope_filter, max_points=50_000)
        except TypeError:
            points = connector.scroll_all(collection)
    for point in points:
        payload = getattr(point, "payload", None) or (point.get("payload", {}) if isinstance(point, dict) else {})
        if not isinstance(payload, dict) or payload.get("dataset_id") != dataset_id:
            continue
        value = payload.get("file_name") or payload.get("source_uri") or payload.get("source_path")
        if isinstance(value, str) and value.strip():
            labels.add(_source_label(value))
    return tuple(sorted(labels))


def _source_label(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").split("/")[-1]


def _generation_from_collection(collection: str, alias: str) -> str | None:
    prefix = f"{alias}__"
    if collection.startswith(prefix):
        return collection[len(prefix):] or None
    return None


def _distance_cosine():
    from qdrant_client.models import Distance

    return Distance.COSINE
