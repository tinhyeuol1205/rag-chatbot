# PR11 load-test report template

The repository includes `scripts/load_ingestion_pr11.py`. Run it against a
representative production corpus after starting the real Qdrant service:

```bash
PYTHONPATH=src rag/bin/python scripts/load_ingestion_pr11.py \
  /srv/corpus --job-id load-<date> \
  --output data/ingest_runs/load-<date>.json
```

Record the following for each deployment shape:

| Field | Target / observation |
|---|---|
| Corpus files and bytes |  |
| Parent/child chunks |  |
| Peak RSS | Must remain below `INGEST_MAX_MEMORY_MB` |
| Elapsed / files per second |  |
| Processed / skipped files |  |
| Qdrant max request bytes | Must remain below `INGEST_QDRANT_WRITE_MAX_BYTES` |
| Retry count / error rate |  |
| Failure injection batch | Resume must not re-embed committed IDs |
| Alias switch duration |  |

Acceptance requires a representative large corpus (target: 1M chunks or the
production-equivalent size), an injected timeout after a write batch, and a
second run with the same job ID. Keep the JSON output with the migration
artifacts and attach the exact configuration used.

