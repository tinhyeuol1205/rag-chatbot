"""Redacted, immutable evaluation run artifacts and quality decisions."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from core import get_logger
from core.config import settings
from evaluation.dataset import EvalSuite
from evaluation.preflight import PreflightResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class QualityDecision:
    passed: bool
    exit_code: int
    status: str
    failures: tuple[str, ...] = ()


def decide_quality(
    *,
    scores: dict[str, Any],
    results: list[Any],
    thresholds: dict[str, float],
    max_failed_samples: int,
    max_empty_context_samples: int,
    fallback_used: bool = False,
) -> QualityDecision:
    failed = [result for result in results if getattr(result, "status", "success") != "success"]
    empty = [
        result
        for result in results
        if getattr(result, "status", "success") == "success" and not getattr(result, "contexts", None)
    ]
    failures: list[str] = []
    if len(failed) > max_failed_samples:
        failures.append(f"failed_samples={len(failed)}>{max_failed_samples}")
    if len(empty) > max_empty_context_samples:
        failures.append(f"empty_context_samples={len(empty)}>{max_empty_context_samples}")
    if (failed or empty) and (
        len(failed) > max_failed_samples or len(empty) > max_empty_context_samples
    ):
        return QualityDecision(False, 3, "failed", tuple(failures))
    if fallback_used or "fallback" in str(scores.get("_mode", "")).lower():
        failures.append("simple_fallback_used")
        return QualityDecision(False, 4, "fallback", tuple(failures))
    if str(scores.get("_mode", "")).lower() in {"not_evaluated", "failed"}:
        failures.append("judge_metrics_not_evaluated")
        return QualityDecision(False, 4, "not_evaluated", tuple(failures))
    if not any(key in scores for key in ("context_precision", "faithfulness", "answer_relevancy")):
        failures.append("judge_metrics_not_evaluated")
        return QualityDecision(False, 4, "not_evaluated", tuple(failures))
    metric_keys = {key for key in thresholds}
    missing = sorted(key for key in metric_keys if key not in scores or not _is_finite(scores[key]))
    if missing:
        failures.append(f"metrics_not_evaluated={','.join(missing)}")
        return QualityDecision(False, 4, "not_evaluated", tuple(failures))
    for metric, threshold in thresholds.items():
        value = float(scores[metric])
        if value < threshold:
            failures.append(f"{metric}={value:.6f}<{threshold:.6f}")
    if failures:
        return QualityDecision(False, 5, "failed", tuple(failures))
    return QualityDecision(True, 0, "passed", ())


def build_artifact(
    *,
    suite: EvalSuite,
    results: list[Any],
    scores: dict[str, Any],
    thresholds: dict[str, float],
    decision: QualityDecision,
    preflight: PreflightResult | None = None,
    include_content: bool = False,
    run_id: str,
) -> dict[str, Any]:
    """Build schema v2 without raw content unless explicitly opted in."""
    _validate_finite_tree(scores)
    for key, value in scores.items():
        if key != "per_sample" and not key.startswith("_") and isinstance(value, (int, float)):
            _json_number(value)
    failed = [result for result in results if getattr(result, "status", "success") != "success"]
    empty = [
        result
        for result in results
        if getattr(result, "status", "success") == "success" and not getattr(result, "contexts", None)
    ]
    aggregate = {
        key: _json_number(value)
        for key, value in scores.items()
        if key != "per_sample" and not key.startswith("_") and _is_finite(value)
    }
    aggregate.update(_deterministic_from_results(results))
    samples = []
    for result in results:
        sample = getattr(result, "sample", None)
        item: dict[str, Any] = {
            "sample_id": getattr(sample, "sample_id", getattr(result, "sample_id", "")),
            "status": getattr(result, "status", "success"),
            "slice": getattr(sample, "slice", "default"),
            "answerable": bool(getattr(sample, "answerable", True)),
            "num_contexts": len(getattr(result, "contexts", []) or []),
            "num_candidates": int(getattr(result, "num_candidates", 0)),
            "source_ids": list(getattr(result, "source_ids", ()) or ()),
            "latency_seconds": _json_number(getattr(result, "latency_seconds", 0.0)),
            "scores": {
                key: _json_number(value)
                for key, value in {
                    **(getattr(result, "scores", {}) or {}),
                    **(getattr(result, "deterministic", {}) or {}),
                }.items()
                if _is_finite(value)
            },
        }
        for key, value in (getattr(result, "scores", {}) or {}).items():
            if isinstance(value, (int, float)):
                _json_number(value)
        if getattr(result, "error_code", None):
            item["error_code"] = str(result.error_code)
        if getattr(result, "error_type", None):
            item["error_type"] = str(result.error_type)
        if include_content:
            item["content"] = {
                "question": getattr(result, "question", ""),
                "answer": getattr(result, "answer", ""),
                "reference": getattr(result, "ground_truth", ""),
                "contexts": list(getattr(result, "contexts", []) or []),
            }
        samples.append(item)

    payload: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": decision.status,
        "exit_code": decision.exit_code,
        "suite": {
            **suite.as_dict(include_content=False),
            "fingerprint": suite.fingerprint,
        },
        "code": _git_metadata(),
        "corpus": preflight.as_dict() if preflight is not None else {},
        "models": {
            "generator": {
                "provider": settings.LLM_PROVIDER,
                "model": _candidate_model(),
                "base_host": _redact_host(
                    settings.OPENAI_BASE_URL if settings.LLM_PROVIDER.lower() == "openai" else ""
                ),
            },
            "judge": {
                "provider": settings.EVAL_JUDGE_PROVIDER,
                "model": settings.EVAL_JUDGE_MODEL,
                "base_host": _redact_host(settings.EVAL_JUDGE_BASE_URL),
            },
            "embedding": {
                "model": settings.EMBEDDING_MODEL_ID,
                "revision": settings.EMBEDDING_MODEL_REVISION or settings.INGEST_EMBEDDING_MODEL_REVISION,
                "runtime": settings.EMBEDDING_RUNTIME,
            },
            "reranker": {
                "model": settings.RERANKER_MODEL_ID,
                "revision": settings.RERANKER_MODEL_REVISION,
                "runtime": settings.RERANKER_RUNTIME,
            },
        },
        "retrieval": {
            "top_k": settings.TOP_K,
            "rerank_candidates": settings.RERANK_CANDIDATES,
            "keep_top_k": settings.KEEP_TOP_K,
            "expand_n_query": settings.EXPAND_N_QUERY,
            "max_context_chars": settings.MAX_CONTEXT_CHARS,
        },
        "counts": {
            "total": len(results),
            "success": len(results) - len(failed),
            "failed": len(failed),
            "empty_context": len(empty),
        },
        "aggregate": aggregate,
        "thresholds": {key: _json_number(value) for key, value in thresholds.items()},
        "quality": {
            "passed": decision.passed,
            "failures": list(decision.failures),
            "diagnostics": _safe_diagnostics(scores),
        },
        "samples": samples,
    }
    if include_content:
        logger.warning("Evaluation artifact includes raw content; use only in a trusted local directory")
    return payload


def write_artifact(payload: dict[str, Any], *, out_dir: str | Path, suite: EvalSuite) -> tuple[Path, Path]:
    """Write an immutable run file and refresh the convenience latest copy."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}-{_safe_name(payload.get('code', {}).get('git_sha', 'nogit'))}-{_safe_name(suite.suite_id)}-v{_safe_name(suite.version)}"
    path = directory / f"{stem}.json"
    suffix = 1
    while path.exists():
        path = directory / f"{stem}-{suffix}.json"
        suffix += 1
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    latest = directory / "latest.json"
    latest.write_text(encoded, encoding="utf-8")
    return path, latest


def _deterministic_from_results(results: list[Any]) -> dict[str, float]:
    from evaluation.deterministic import compute_deterministic_metrics

    values = compute_deterministic_metrics(results)
    return {
        key: _json_number(value)
        for key, value in values.items()
        if key != "per_sample" and _is_finite(value)
    }


def _candidate_model() -> str:
    if settings.LLM_PROVIDER.lower() == "gemini":
        return settings.GEMINI_MODEL_ID
    return settings.OPENAI_MODEL_ID


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[2],
                check=False,
                timeout=2,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    sha = run("rev-parse", "HEAD") or "unknown"
    dirty = bool(run("status", "--porcelain"))
    return {"git_sha": sha, "dirty": dirty}


def _redact_host(value: str) -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.hostname:
        return parsed.hostname
    return value.split("/", 1)[0].split("?", 1)[0]


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text) or "unknown"


def _safe_diagnostics(scores: dict[str, Any]) -> dict[str, str]:
    """Keep only bounded diagnostic tokens; never persist raw exception text."""
    diagnostics: dict[str, str] = {}
    for key in ("error_code", "error_type", "error_stage", "error_metric"):
        value = scores.get(key)
        if not isinstance(value, str) or not value or len(value) > 80:
            continue
        if all(char.isalnum() or char in "._-" for char in value):
            diagnostics[key] = value
    return diagnostics


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_finite_tree(value: Any) -> None:
    """Reject non-finite numbers even when they are nested per-sample scores."""
    if isinstance(value, dict):
        for child in value.values():
            _validate_finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_tree(child)
    elif isinstance(value, (int, float)):
        _json_number(value)


def _json_number(value: Any) -> float | int:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("evaluation artifact cannot contain NaN or infinity")
    return int(number) if number.is_integer() else number
