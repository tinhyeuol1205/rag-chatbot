"""Durable source manifest and resumable ingestion checkpoints.

The manifest is deliberately separate from vector collections.  It records the
source/fingerprint decision and the batches that have been committed, so a
worker can recover after a process or network failure without treating a
partially written generation as active data.

SQLite is used as the default control-plane store because it is available in
the standard library and supports WAL/transactions.  Production deployments
must place the file on a durable shared volume (or replace this class with the
project's transactional metadata service).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from core.config import settings


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally so hashing does not add corpus-sized memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_fingerprint(
    *,
    embedding_dimension: int,
    parser_version: str | None = None,
    chunker_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Return a stable fingerprint for every transformation affecting points."""
    payload: dict[str, Any] = {
        "pipeline_version": settings.INGEST_PIPELINE_VERSION,
        "schema_version": settings.INGEST_SCHEMA_VERSION,
        "parser_version": parser_version or settings.INGEST_PARSER_VERSION,
        "chunker_version": chunker_version or settings.INGEST_CHUNKER_VERSION,
        "embedding_model_id": settings.EMBEDDING_MODEL_ID,
        "embedding_model_revision": settings.INGEST_EMBEDDING_MODEL_REVISION,
        "embedding_dimension": embedding_dimension,
        "embedding_normalize": True,
        "parent_chunk_size": settings.PARENT_CHUNK_SIZE,
        "parent_chunk_overlap": settings.PARENT_CHUNK_OVERLAP,
        "child_chunk_size": settings.CHILD_CHUNK_SIZE,
        "child_chunk_overlap": settings.CHILD_CHUNK_OVERLAP,
        "document_window": settings.INGEST_DOCUMENT_WINDOW,
        "pdf_fast_strategy": settings.INGEST_PDF_FAST_STRATEGY,
        "pdf_ocr_strategy": settings.INGEST_PDF_OCR_STRATEGY,
        "parser_quality": {
            "max_empty_page_ratio": settings.INGEST_PARSER_MAX_EMPTY_PAGE_RATIO,
            "max_unsupported_ratio": settings.INGEST_PARSER_MAX_UNSUPPORTED_RATIO,
            "min_text_chars": settings.INGEST_PARSER_MIN_TEXT_CHARS,
            "max_replacement_ratio": settings.INGEST_PARSER_MAX_REPLACEMENT_RATIO,
        },
        "source_namespace": settings.INGEST_DATASET_ID,
    }
    if extra:
        payload.update(extra)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceRecord:
    dataset_id: str
    source_uri: str
    content_sha256: str
    size_bytes: int
    mtime_ns: int
    fingerprint: str
    status: str
    active_generation: str | None = None
    working_generation: str | None = None
    parent_count: int = 0
    child_count: int = 0
    quality: dict[str, Any] | None = None
    error_code: str | None = None
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SourceRecord:
        return cls(
            dataset_id=row["dataset_id"],
            source_uri=row["source_uri"],
            content_sha256=row["content_sha256"],
            size_bytes=row["size_bytes"],
            mtime_ns=row["mtime_ns"],
            fingerprint=row["fingerprint"],
            status=row["status"],
            active_generation=row["active_generation"],
            working_generation=row["working_generation"],
            parent_count=row["parent_count"] or 0,
            child_count=row["child_count"] or 0,
            quality=json.loads(row["quality_json"]) if row["quality_json"] else None,
            error_code=row["error_code"],
            updated_at=row["updated_at"] or 0.0,
        )


class ManifestStore:
    """Transactional source and batch manifest backed by SQLite."""

    _SQLITE_IN_BATCH = 500  # stay below SQLite's default 999 bind-variable limit

    def __init__(self, path: str | Path | None = None):
        self.path = str(path or settings.INGEST_MANIFEST_PATH)
        self._lock = RLock()
        self._memory_connection: sqlite3.Connection | None = None
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self.path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(
                    self.path,
                    check_same_thread=False,
                    timeout=30,
                )
                self._memory_connection.row_factory = sqlite3.Row
            return self._memory_connection
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS source_manifest (
                    dataset_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active_generation TEXT,
                    working_generation TEXT,
                    parent_count INTEGER NOT NULL DEFAULT 0,
                    child_count INTEGER NOT NULL DEFAULT 0,
                    quality_json TEXT,
                    error_code TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (dataset_id, source_uri)
                );
                CREATE INDEX IF NOT EXISTS idx_manifest_generation
                    ON source_manifest(dataset_id, active_generation);
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active_generation_before TEXT,
                    active_generation_after TEXT,
                    summary_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_checkpoints (
                    dataset_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    batch_key TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    point_ids_json TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    point_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (dataset_id, generation_id, source_uri, batch_key, collection_name)
                );
                """
            )
            connection.commit()
            if self.path != ":memory:":
                connection.close()

    def close(self) -> None:
        with self._lock:
            if self._memory_connection is not None:
                self._memory_connection.close()
                self._memory_connection = None

    def get_source(self, dataset_id: str, source_uri: str) -> SourceRecord | None:
        with self._lock:
            connection = self._connect()
            row = connection.execute(
                "SELECT * FROM source_manifest WHERE dataset_id=? AND source_uri=?",
                (dataset_id, source_uri),
            ).fetchone()
            if self.path != ":memory:":
                connection.close()
            return SourceRecord.from_row(row) if row else None

    def list_sources(
        self,
        dataset_id: str,
        *,
        active_generation: str | None = None,
    ) -> list[SourceRecord]:
        with self._lock:
            connection = self._connect()
            if active_generation is None:
                rows = connection.execute(
                    "SELECT * FROM source_manifest WHERE dataset_id=? ORDER BY source_uri",
                    (dataset_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM source_manifest WHERE dataset_id=? AND active_generation=? "
                    "ORDER BY source_uri",
                    (dataset_id, active_generation),
                ).fetchall()
            if self.path != ":memory:":
                connection.close()
            return [SourceRecord.from_row(row) for row in rows]

    def record_discovered(
        self,
        *,
        dataset_id: str,
        source_uri: str,
        content_sha256: str,
        size_bytes: int,
        mtime_ns: int,
        fingerprint: str,
        generation_id: str,
    ) -> SourceRecord | None:
        """Upsert discovery metadata and return the previous source record."""
        now = time.time()
        with self._lock:
            connection = self._connect()
            previous_row = connection.execute(
                "SELECT * FROM source_manifest WHERE dataset_id=? AND source_uri=?",
                (dataset_id, source_uri),
            ).fetchone()
            previous = SourceRecord.from_row(previous_row) if previous_row else None
            unchanged = bool(
                previous
                and previous.content_sha256 == content_sha256
                and previous.fingerprint == fingerprint
                and previous.status == "committed"
            )
            status = "committed" if unchanged else "discovered"
            connection.execute(
                """
                INSERT INTO source_manifest (
                    dataset_id, source_uri, content_sha256, size_bytes, mtime_ns,
                    fingerprint, status, active_generation, working_generation,
                    parent_count, child_count, quality_json, error_code, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, source_uri) DO UPDATE SET
                    content_sha256=excluded.content_sha256,
                    size_bytes=excluded.size_bytes,
                    mtime_ns=excluded.mtime_ns,
                    fingerprint=excluded.fingerprint,
                    status=excluded.status,
                    working_generation=excluded.working_generation,
                    error_code=CASE WHEN excluded.status='discovered' THEN NULL ELSE source_manifest.error_code END,
                    updated_at=excluded.updated_at
                """,
                (
                    dataset_id,
                    source_uri,
                    content_sha256,
                    size_bytes,
                    mtime_ns,
                    fingerprint,
                    status,
                    previous.active_generation if previous else None,
                    generation_id,
                    previous.parent_count if previous else 0,
                    previous.child_count if previous else 0,
                    json.dumps(previous.quality, sort_keys=True) if previous and previous.quality else None,
                    previous.error_code if previous else None,
                    now,
                ),
            )
            connection.commit()
            if self.path != ":memory:":
                connection.close()
            return previous

    def mark_source(
        self,
        *,
        dataset_id: str,
        source_uri: str,
        status: str,
        generation_id: str | None = None,
        active_generation: str | None = None,
        parent_count: int = 0,
        child_count: int = 0,
        quality: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            connection = self._connect()
            connection.execute(
                """
                UPDATE source_manifest SET
                    status=?,
                    working_generation=COALESCE(?, working_generation),
                    active_generation=COALESCE(?, active_generation),
                    parent_count=?, child_count=?, quality_json=?, error_code=?, updated_at=?
                WHERE dataset_id=? AND source_uri=?
                """,
                (
                    status,
                    generation_id,
                    active_generation,
                    parent_count,
                    child_count,
                    json.dumps(quality, sort_keys=True) if quality is not None else None,
                    error_code,
                    now,
                    dataset_id,
                    source_uri,
                ),
            )
            connection.commit()
            if self.path != ":memory:":
                connection.close()

    def record_batch(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        source_uri: str,
        batch_key: str,
        collection_name: str,
        point_ids: list[str],
        content_sha256: str,
        fingerprint: str,
    ) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute(
                """
                INSERT INTO batch_checkpoints (
                    dataset_id, generation_id, source_uri, batch_key,
                    collection_name, point_ids_json, content_sha256, fingerprint,
                    status, point_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'committed', ?, ?)
                ON CONFLICT(dataset_id, generation_id, source_uri, batch_key, collection_name)
                DO UPDATE SET status='committed', point_ids_json=excluded.point_ids_json,
                    point_count=excluded.point_count, updated_at=excluded.updated_at
                """,
                (
                    dataset_id,
                    generation_id,
                    source_uri,
                    batch_key,
                    collection_name,
                    json.dumps(sorted(point_ids)),
                    content_sha256,
                    fingerprint,
                    len(point_ids),
                    time.time(),
                ),
            )
            connection.commit()
            if self.path != ":memory:":
                connection.close()

    def batch_committed(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        source_uri: str,
        batch_key: str,
        collection_name: str,
        content_sha256: str,
        fingerprint: str,
    ) -> list[str] | None:
        with self._lock:
            connection = self._connect()
            row = connection.execute(
                """
                SELECT point_ids_json FROM batch_checkpoints
                WHERE dataset_id=? AND generation_id=? AND source_uri=? AND batch_key=?
                  AND collection_name=? AND content_sha256=? AND fingerprint=? AND status='committed'
                """,
                (
                    dataset_id,
                    generation_id,
                    source_uri,
                    batch_key,
                    collection_name,
                    content_sha256,
                    fingerprint,
                ),
            ).fetchone()
            if self.path != ":memory:":
                connection.close()
            return json.loads(row["point_ids_json"]) if row else None

    def create_job(
        self,
        *,
        job_id: str,
        dataset_id: str,
        generation_id: str,
        fingerprint: str,
        active_generation_before: str | None,
    ) -> None:
        now = time.time()
        with self._lock:
            connection = self._connect()
            connection.execute(
                """
                INSERT INTO ingest_jobs (
                    job_id, dataset_id, generation_id, fingerprint, status,
                    active_generation_before, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                ON CONFLICT(job_id) DO NOTHING
                """,
                (job_id, dataset_id, generation_id, fingerprint, active_generation_before, now, now),
            )
            connection.commit()
            if self.path != ":memory:":
                connection.close()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            connection = self._connect()
            row = connection.execute("SELECT * FROM ingest_jobs WHERE job_id=?", (job_id,)).fetchone()
            if self.path != ":memory:":
                connection.close()
            return dict(row) if row else None

    def finish_job(
        self,
        *,
        job_id: str,
        status: str,
        active_generation_after: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute(
                """
                UPDATE ingest_jobs SET status=?, active_generation_after=?,
                    summary_json=?, updated_at=? WHERE job_id=?
                """,
                (
                    status,
                    active_generation_after,
                    json.dumps(summary, sort_keys=True) if summary is not None else None,
                    time.time(),
                    job_id,
                ),
            )
            connection.commit()
            if self.path != ":memory:":
                connection.close()

    def committed_generations(self, dataset_id: str, *, limit: int) -> list[str]:
        """Return newest committed generations for retention/rollback tooling."""
        if limit <= 0:
            return []
        with self._lock:
            connection = self._connect()
            rows = connection.execute(
                "SELECT generation_id, MAX(updated_at) AS last_updated "
                "FROM ingest_jobs WHERE dataset_id=? AND status='committed' "
                "GROUP BY generation_id ORDER BY last_updated DESC LIMIT ?",
                (dataset_id, limit),
            ).fetchall()
            if self.path != ":memory:":
                connection.close()
            return [row["generation_id"] for row in rows]

    def activate_generation(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        source_uris: set[str],
    ) -> None:
        """Mark sources in a generation active and clear removed source rows."""
        with self._lock:
            connection = self._connect()
            connection.execute(
                "UPDATE source_manifest SET active_generation=NULL WHERE dataset_id=? AND active_generation=?",
                (dataset_id, generation_id),
            )
            if source_uris:
                ordered_sources = sorted(source_uris)
                for start in range(0, len(ordered_sources), self._SQLITE_IN_BATCH):
                    batch = ordered_sources[start:start + self._SQLITE_IN_BATCH]
                    placeholders = ",".join("?" for _ in batch)
                    connection.execute(
                        f"UPDATE source_manifest SET active_generation=?, status='committed', "
                        f"updated_at=?, working_generation=NULL WHERE dataset_id=? "
                        f"AND source_uri IN ({placeholders})",
                        (generation_id, time.time(), dataset_id, *batch),
                    )
            connection.commit()
            if self.path != ":memory:":
                connection.close()

    def mark_removed(self, *, dataset_id: str, source_uris: set[str]) -> None:
        """Mark sources absent from a successful sync as removed."""
        if not source_uris:
            return
        with self._lock:
            connection = self._connect()
            ordered_sources = sorted(source_uris)
            for start in range(0, len(ordered_sources), self._SQLITE_IN_BATCH):
                batch = ordered_sources[start:start + self._SQLITE_IN_BATCH]
                placeholders = ",".join("?" for _ in batch)
                connection.execute(
                    f"UPDATE source_manifest SET status='removed', active_generation=NULL, "
                    f"working_generation=NULL, updated_at=? WHERE dataset_id=? "
                    f"AND source_uri IN ({placeholders})",
                    (time.time(), dataset_id, *batch),
                )
            connection.commit()
            if self.path != ":memory:":
                connection.close()

    def mark_rollback(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        source_uris: set[str],
    ) -> None:
        """Point source state at a rolled-back generation and force revalidation.

        Source hashes/counts describe the latest successful ingest, not every
        retained generation.  Clearing them prevents the next incremental run
        from incorrectly skipping a file whose bytes match a newer generation
        while aliases serve the older rollback snapshot.
        """
        with self._lock:
            connection = self._connect()
            connection.execute(
                "UPDATE source_manifest SET status='removed', active_generation=NULL, "
                "working_generation=NULL, content_sha256='', fingerprint='', "
                "parent_count=0, child_count=0, quality_json=NULL, error_code=NULL, updated_at=? "
                "WHERE dataset_id=?",
                (time.time(), dataset_id),
            )
            if source_uris:
                ordered_sources = sorted(source_uris)
                for start in range(0, len(ordered_sources), self._SQLITE_IN_BATCH):
                    batch = ordered_sources[start:start + self._SQLITE_IN_BATCH]
                    placeholders = ",".join("?" for _ in batch)
                    connection.execute(
                        f"UPDATE source_manifest SET status='rolled_back', active_generation=?, "
                        f"working_generation=NULL, content_sha256='', fingerprint='', "
                        f"parent_count=0, child_count=0, quality_json=NULL, error_code=NULL, updated_at=? "
                        f"WHERE dataset_id=? AND source_uri IN ({placeholders})",
                        (generation_id, time.time(), dataset_id, *batch),
                    )
            connection.commit()
            if self.path != ":memory:":
                connection.close()
