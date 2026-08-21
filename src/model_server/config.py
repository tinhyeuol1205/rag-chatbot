"""Model-server configuration view.

The application-wide settings remain in :mod:`core.config` so the RAG API,
ingestion worker and model server share one model/revision contract.  This
small immutable view makes the server-specific values easy to inspect and
inject in tests without duplicating environment parsing.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import settings


@dataclass(frozen=True)
class ModelServerConfig:
    host: str
    port: int
    device: str
    workers: int
    api_key: str
    max_body_bytes: int
    max_pending_requests: int
    max_pending_items: int
    max_pending_bytes: int
    queue_timeout_seconds: float
    batch_wait_ms: int
    mps_max_inflight_batches: int
    online_max_burst_batches: int
    max_items_per_request_per_batch: int


def get_model_server_config() -> ModelServerConfig:
    """Return a validated snapshot of the current model-server settings."""
    return ModelServerConfig(
        host=settings.MODEL_SERVER_HOST,
        port=settings.MODEL_SERVER_PORT,
        device=settings.MODEL_SERVER_DEVICE,
        workers=settings.MODEL_SERVER_WORKERS,
        api_key=settings.MODEL_SERVER_API_KEY,
        max_body_bytes=settings.MODEL_SERVER_MAX_BODY_BYTES,
        max_pending_requests=settings.MODEL_SERVER_MAX_PENDING_REQUESTS,
        max_pending_items=settings.MODEL_SERVER_MAX_PENDING_ITEMS,
        max_pending_bytes=settings.MODEL_SERVER_MAX_PENDING_BYTES,
        queue_timeout_seconds=settings.MODEL_SERVER_QUEUE_TIMEOUT_SECONDS,
        batch_wait_ms=settings.MODEL_SERVER_BATCH_WAIT_MS,
        mps_max_inflight_batches=settings.MODEL_SERVER_MPS_MAX_INFLIGHT_BATCHES,
        online_max_burst_batches=settings.MODEL_SERVER_ONLINE_MAX_BURST_BATCHES,
        max_items_per_request_per_batch=settings.MODEL_SERVER_MAX_ITEMS_PER_REQUEST_PER_BATCH,
    )

