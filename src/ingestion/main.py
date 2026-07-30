from __future__ import annotations

"""
Entry point cho Ingestion Pipeline.

Chạy bằng: make ingest
Hoặc:      cd src && python -m ingestion.main
"""

from pathlib import Path

from core import get_logger
from ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)

# Đường dẫn đến thư mục sample data
DATA_DIR = str(Path(__file__).parent.parent.parent / "data" / "sample_docs")


def main():
    logger.info("Starting ingestion pipeline", data_dir=DATA_DIR)
    pipeline = IngestionPipeline()
    pipeline.run(DATA_DIR)
    logger.info("Done!")


if __name__ == "__main__":
    main()
