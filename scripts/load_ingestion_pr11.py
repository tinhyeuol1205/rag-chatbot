#!/usr/bin/env python3
"""Run a bounded ingestion load trial and emit JSON metrics.

This harness intentionally uses the configured real parser/embedder/Qdrant so
the report reflects deployment hardware.  It does not generate a million-file
fixture by default; operators should point it at a representative corpus or a
synthetic corpus generated outside the application.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from ingestion.pipeline import IngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure PR11 ingestion throughput and peak RSS")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--job-id", default="load-test")
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("data/ingest_runs/load-test.json"))
    args = parser.parse_args()

    started = time.monotonic()
    result = IngestionPipeline().run(
        str(args.data_dir),
        sync=True,
        job_id=args.job_id,
        generation_id=args.generation_id,
    )
    elapsed = max(time.monotonic() - started, 1e-9)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    peak_rss_mb = peak_rss / (1024 * 1024 if peak_rss > 10**9 else 1024)
    payload = {
        "job_id": result.job_id,
        "generation_id": result.generation_id,
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "discovered_files": len(result.discovered_files),
        "processed_files": len(result.processed_files),
        "skipped_files": len(result.skipped_files),
        "failed_files": sorted(result.failed_files),
        "throughput_files_per_second": round(len(result.processed_files) / elapsed, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if result.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())

