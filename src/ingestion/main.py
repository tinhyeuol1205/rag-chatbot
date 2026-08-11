"""
Entry point cho Ingestion Pipeline.

Chạy bằng: make ingest
Hoặc:      cd src && python -m ingestion.main
Đồng bộ xóa file cũ: python -m ingestion.main --sync
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core import get_logger
from ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)

# Đường dẫn đến thư mục sample data
DATA_DIR = str(Path(__file__).parent.parent.parent / "data" / "sample_docs")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Delete files no longer present, scoped to INGEST_DATASET_ID",
    )
    args = parser.parse_args(argv)

    logger.info("Starting ingestion pipeline", data_dir=args.data_dir, sync=args.sync)
    pipeline = IngestionPipeline()
    result = pipeline.run(args.data_dir, sync=args.sync)
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
    )


if __name__ == "__main__":
    main()
