"""CLI exit-code and operator-summary contract tests for ingestion."""

import json

import ingestion.main as ingestion_main
from ingestion.pipeline import IngestionResult


class _FakePipeline:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def run(self, data_dir, **kwargs):
        self.calls.append((data_dir, kwargs))
        if self.error:
            raise self.error
        return self.result


def test_partial_ingestion_returns_nonzero_and_writes_summary(monkeypatch, tmp_path):
    fake = _FakePipeline(
        result=IngestionResult(
            dataset_id="sample_docs",
            discovered_files={"good.md", "broken.md"},
            processed_files={"good.md"},
            failed_files={"broken.md"},
            prune_skipped=True,
        )
    )
    monkeypatch.setattr(ingestion_main, "IngestionPipeline", lambda: fake)
    summary_path = tmp_path / "run.json"

    exit_code = ingestion_main.main([
        "--data-dir", str(tmp_path),
        "--sync",
        "--summary-path", str(summary_path),
    ])

    assert exit_code == 1
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["failed_files"] == ["broken.md"]
    assert payload["prune_skipped"] is True
    assert fake.calls[0][1]["sync"] is True


def test_success_returns_zero_and_passes_explicit_safety_flags(monkeypatch, tmp_path):
    fake = _FakePipeline(result=IngestionResult(dataset_id="sample_docs", dry_run=True))
    monkeypatch.setattr(ingestion_main, "IngestionPipeline", lambda: fake)
    summary_path = tmp_path / "dry-run.json"

    exit_code = ingestion_main.main([
        "--data-dir", str(tmp_path),
        "--sync",
        "--allow-empty-source",
        "--dry-run",
        "--summary-path", str(summary_path),
    ])

    assert exit_code == 0
    assert fake.calls[0][1] == {
        "sync": True,
        "allow_empty_source": True,
        "dry_run": True,
    }
    assert json.loads(summary_path.read_text(encoding="utf-8"))["dry_run"] is True


def test_pipeline_exception_returns_nonzero_and_writes_redacted_failure_summary(
    monkeypatch,
    tmp_path,
):
    fake = _FakePipeline(error=RuntimeError("internal qdrant URL must not be persisted"))
    monkeypatch.setattr(ingestion_main, "IngestionPipeline", lambda: fake)
    summary_path = tmp_path / "failure.json"

    exit_code = ingestion_main.main([
        "--data-dir", str(tmp_path),
        "--summary-path", str(summary_path),
    ])

    assert exit_code == 1
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload == {"status": "failed", "error_type": "RuntimeError"}
