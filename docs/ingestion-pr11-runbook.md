# PR11 ingestion runbook

## Preconditions

1. Put `INGEST_MANIFEST_PATH` on a durable shared volume.
2. Pin `EMBEDDING_MODEL_ID`, `INGEST_EMBEDDING_MODEL_REVISION`, parser/chunker
   versions and embedding dimension in the deployment configuration.
3. Start the target Qdrant server and verify its payload indexes are available.
4. Keep the currently active aliases until the new generation passes validation.

## Staged backfill

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir /srv/corpus --sync --job-id backfill-<date> \
  --summary-path /srv/ingest-runs/backfill-<date>.json
```

The job writes concrete collections named `child_chunks_active__<generation>`
and `parent_chunks_active__<generation>`.  Files whose content hash and full
pipeline fingerprint match the previous generation are copied without parsing
or embedding.  A failed source prevents alias activation and is reported in the
summary/dead-letter manifest.

For a dry run:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir /srv/corpus --sync --dry-run \
  --summary-path /srv/ingest-runs/plan.json
```

## Resume

Re-run with the same `--job-id`.  Batch checkpoints are keyed by source,
generation, collection and deterministic batch key.  The worker verifies point
IDs in Qdrant before embedding/upserting a checkpointed batch, so a process
crash after a successful write is safe to resume.

## Activation and rollback

The two retrieval aliases are switched in one Qdrant alias operation only after:

- collection schema fingerprint, vector dimension and distance match;
- required payload indexes exist;
- sampled child-to-parent references resolve;
- no source failed.

To roll back, choose a retained generation and run:

```bash
PYTHONPATH=src rag/bin/python - <<'PY'
from ingestion.pipeline import IngestionPipeline
IngestionPipeline().rollback("<generation-id>")
PY
```

Then verify `/health`, a representative retrieval sample, and the ingestion
summary before deleting the failed generation.  Never delete a generation that
is still the target of either retrieval alias.

After a successful activation, committed generations beyond
`INGEST_GENERATION_RETENTION` are pruned automatically. Failed generations are
kept for inspection and are not counted as retained rollback snapshots.

## Failure handling

- Exit code `1` means at least one source failed or a safety/schema guard stopped
  the job.
- Do not pass `--allow-empty-source` unless the source mount was verified.
- Preserve the failed generation for inspection until the rollback/retention
  window expires.
- Investigate `error_code` and parse-quality counters in the manifest; retry only
  the affected sources after correction.
