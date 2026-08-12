"""
Entry point cho Ingestion Pipeline.

Chạy bằng: make ingest
Hoặc:      cd src && python -m ingestion.main
Đồng bộ xóa file cũ: python -m ingestion.main --sync
Xem trước thay đổi:  python -m ingestion.main --sync --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import get_logger
from ingestion.pipeline import IngestionPipeline, IngestionResult

logger = get_logger(__name__)

# Đường dẫn đến thư mục sample data
DATA_DIR = str(Path(__file__).parent.parent.parent / "data" / "sample_docs")
DEFAULT_SUMMARY_PATH = "data/ingest_runs/latest.json"


def _result_payload(result: IngestionResult) -> dict:
    """Convert set-heavy result data into a stable JSON operator artifact."""
    return {
        "dataset_id": result.dataset_id,
        "discovered_files": sorted(result.discovered_files),
        "processed_files": sorted(result.processed_files),
        "failed_files": sorted(result.failed_files),
        "pruned_files": sorted(result.pruned_files),
        "planned_pruned_files": sorted(result.planned_pruned_files),
        "prune_skipped": result.prune_skipped,
        "dry_run": result.dry_run,
    }


def _write_summary(result: IngestionResult, summary_path: str) -> None:
    """Persist an ingestion summary before returning a process status."""
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_result_payload(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_failure_summary(summary_path: str, error: Exception) -> None:
    """Persist a redacted failure artifact when the pipeline raises early."""
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"status": "failed", "error_type": type(error).__name__},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Delete files no longer present, scoped to INGEST_DATASET_ID",
    )
    parser.add_argument(
        "--allow-empty-source",
        action="store_true",
        help="Allow --sync to prune the dataset when no supported source files are found",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and show the ingestion/prune plan without mutating Qdrant",
    )
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_SUMMARY_PATH,
        help=f"JSON operator summary path (default: {DEFAULT_SUMMARY_PATH})",
    )
    args = parser.parse_args(argv)

    logger.info(
        "Starting ingestion pipeline",
        data_dir=args.data_dir,
        sync=args.sync,
        allow_empty_source=args.allow_empty_source,
        dry_run=args.dry_run,
    )
    try:
        pipeline = IngestionPipeline()
        result = pipeline.run(
            args.data_dir,
            sync=args.sync,
            allow_empty_source=args.allow_empty_source,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        logger.exception("Ingestion pipeline failed")
        try:
            _write_failure_summary(args.summary_path, exc)
        except Exception:
            logger.exception("Failed to write ingestion failure summary")
        return 1

    try:
        _write_summary(result, args.summary_path)
    except Exception:
        logger.exception("Failed to write ingestion summary")
        return 1

    if result.prune_skipped:
        logger.warning(
            "Ingestion completed without stale-file prune",
            failed_files=sorted(result.failed_files),
        )
    logger.info(
        "Done!",
        processed=len(result.processed_files),
        failed=len(result.failed_files),
        pruned=len(result.pruned_files),
        planned_pruned=len(result.planned_pruned_files),
        summary_path=args.summary_path,
    )
    return 1 if result.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
