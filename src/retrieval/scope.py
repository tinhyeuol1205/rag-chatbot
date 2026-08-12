"""
Retrieval scope — server-side boundary for document visibility.

The ingestion ``dataset_id`` is a namespace, not a user-provided query field.  A
retriever receives a scope created by the server and forwards it to every search
and parent lookup.  Points without a dataset namespace are intentionally outside
all scopes.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchAny

from core.config import settings


@dataclass(frozen=True)
class RetrievalScope:
    """Immutable set of dataset namespaces a retriever may return."""

    dataset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(dataset_id, str) for dataset_id in self.dataset_ids):
            raise ValueError("RetrievalScope dataset_ids must contain strings")
        normalized = tuple(dict.fromkeys(dataset_id.strip() for dataset_id in self.dataset_ids))
        if not normalized or any(not dataset_id for dataset_id in normalized):
            raise ValueError("RetrievalScope requires at least one non-empty dataset_id")
        object.__setattr__(self, "dataset_ids", normalized)

    @classmethod
    def configured(cls) -> RetrievalScope:
        """Build the default server-side scope from application configuration."""
        return cls((settings.INGEST_DATASET_ID,))

    @property
    def cache_key(self) -> tuple[str, ...]:
        """Stable key for scoped in-process caches."""
        return self.dataset_ids

    def allows(self, dataset_id: object) -> bool:
        """Return whether a payload belongs to this scope.

        Missing or non-string values are rejected.  In particular, legacy points
        without ``dataset_id`` must not silently become part of the default scope.
        """
        return isinstance(dataset_id, str) and dataset_id in self.dataset_ids

    def qdrant_filter(self) -> Filter:
        """Create a Qdrant payload filter for this scope."""
        return Filter(
            must=[
                FieldCondition(
                    key="dataset_id",
                    match=MatchAny(any=list(self.dataset_ids)),
                )
            ]
        )

def default_scope() -> RetrievalScope:
    """Return the configured scope without exposing a request-controlled dataset."""
    return RetrievalScope.configured()
