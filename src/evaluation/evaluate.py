"""Strict RAG evaluation CLI.

Run with::

    make evaluate
    PYTHONPATH=src rag/bin/python -m evaluation.evaluate \
      --suite data/eval/suites/kiemhiep-kimdung.json

The command exits non-zero for corpus/config errors, RAG failures, judge
failures/fallbacks and threshold misses.  Release artifacts are redacted by
default; ``--include-content`` is for trusted local diagnostics only.
"""

from __future__ import annotations

import argparse
import math
import uuid
from typing import Any

from core import get_logger
from core.config import settings
from evaluation.artifacts import (
    QualityDecision,
    build_artifact,
    decide_quality,
    write_artifact,
)
from evaluation.dataset import EVAL_SUITE, EvalSuite, EvaluationSuiteError, load_suite
from evaluation.deterministic import compute_deterministic_metrics
from evaluation.metrics import EvaluationMetricsError, evaluate_results
from evaluation.preflight import EvaluationPreflightError, PreflightResult, run_preflight
from evaluation.runner import EvaluationRunner

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_EXECUTION = 3
EXIT_METRICS = 4
EXIT_THRESHOLD = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict RAG quality evaluation")
    parser.add_argument(
        "--suite",
        default="data/eval/suites/kiemhiep-kimdung.json",
        help="versioned JSON evaluation suite",
    )
    parser.add_argument("--output-dir", default="data/eval_runs")
    parser.add_argument("--execution-mode", choices=("inline", "redis_worker"), default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--min-context-precision", type=float, default=None)
    parser.add_argument("--min-faithfulness", type=float, default=None)
    parser.add_argument("--min-answer-relevancy", type=float, default=None)
    parser.add_argument("--min-source-recall", type=float, default=None)
    parser.add_argument("--max-failed-samples", type=int, default=0)
    parser.add_argument("--max-empty-context-samples", type=int, default=0)
    parser.add_argument(
        "--allow-simple-fallback",
        action="store_true",
        help="local diagnostic only; fallback always exits with code 4",
    )
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="include raw question/answer/reference/context in a trusted local artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_error = _validate_args(args)
    if validation_error is not None:
        _print_failure(validation_error, EXIT_CONFIG)
        return EXIT_CONFIG
    run_id = uuid.uuid4().hex
    try:
        suite = _load_requested_suite(args.suite)
    except (EvaluationSuiteError, OSError) as exc:
        _print_failure("Evaluation suite is invalid", EXIT_CONFIG)
        logger.error("Evaluation suite rejected", error_type=type(exc).__name__)
        return EXIT_CONFIG

    preflight: PreflightResult | None = None
    try:
        preflight = run_preflight(suite)
    except EvaluationPreflightError as exc:
        decision = QualityDecision(False, EXIT_CONFIG, "failed", ("preflight_failed",))
        _write_failure_artifact(
            suite=suite,
            results=[],
            scores={"_mode": "not_evaluated"},
            decision=decision,
            preflight=None,
            args=args,
            run_id=run_id,
        )
        logger.error("Evaluation preflight rejected", error_type=type(exc).__name__)
        return EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001 - preflight is a public config boundary
        decision = QualityDecision(False, EXIT_CONFIG, "failed", ("preflight_unavailable",))
        _write_failure_artifact(
            suite=suite,
            results=[],
            scores={"_mode": "not_evaluated"},
            decision=decision,
            preflight=None,
            args=args,
            run_id=run_id,
        )
        logger.error("Evaluation preflight unavailable", error_type=type(exc).__name__)
        return EXIT_CONFIG

    runner = EvaluationRunner(
        suite,
        execution_mode=args.execution_mode,
        timeout_seconds=args.timeout_seconds,
        run_id=run_id,
    )
    results = runner.run_all()
    deterministic = compute_deterministic_metrics(results)
    scores: dict[str, Any] = {}
    fallback_allowed = bool(args.allow_simple_fallback or settings.EVAL_ALLOW_SIMPLE_FALLBACK)
    try:
        scores = evaluate_results(results, allow_simple_fallback=fallback_allowed)
    except EvaluationMetricsError as exc:
        scores = {"_mode": "not_evaluated", **exc.as_diagnostics()}
        logger.error("Evaluation judge failed", **exc.as_diagnostics())
    # Deterministic metrics are useful even if judge metrics fail, but never turn
    # a judge failure into a pass.
    for key, value in deterministic.items():
        if key != "per_sample":
            scores[key] = value
    thresholds = _thresholds(args)
    decision = decide_quality(
        scores=scores,
        results=results,
        thresholds=thresholds,
        max_failed_samples=args.max_failed_samples,
        max_empty_context_samples=args.max_empty_context_samples,
        fallback_used=fallback_allowed and "fallback" in str(scores.get("_mode", "")).lower(),
    )
    _write_failure_artifact(
        suite=suite,
        results=results,
        scores=scores,
        decision=decision,
        preflight=preflight,
        args=args,
        run_id=run_id,
    )
    _print_report(suite, results, scores, decision)
    return decision.exit_code


def _load_requested_suite(path: str) -> EvalSuite:
    if path in {"builtin", "default"}:
        return EVAL_SUITE
    return load_suite(path)


def _thresholds(args: argparse.Namespace) -> dict[str, float]:
    values = {
        "context_precision": args.min_context_precision,
        "faithfulness": args.min_faithfulness,
        "answer_relevancy": args.min_answer_relevancy,
        "source_recall": args.min_source_recall,
    }
    return {key: value for key, value in values.items() if value is not None}


def _validate_args(args: argparse.Namespace) -> str | None:
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        return "--timeout-seconds must be positive"
    if args.max_failed_samples < 0 or args.max_empty_context_samples < 0:
        return "sample failure bounds cannot be negative"
    for name, value in (
        ("--min-context-precision", args.min_context_precision),
        ("--min-faithfulness", args.min_faithfulness),
        ("--min-answer-relevancy", args.min_answer_relevancy),
        ("--min-source-recall", args.min_source_recall),
    ):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            return f"{name} must be a finite value between 0 and 1"
    return None


def _write_failure_artifact(
    *,
    suite: EvalSuite,
    results: list[Any],
    scores: dict[str, Any],
    decision: QualityDecision,
    preflight: PreflightResult | None,
    args: argparse.Namespace,
    run_id: str,
) -> None:
    try:
        payload = build_artifact(
            suite=suite,
            results=results,
            scores=scores,
            thresholds=_thresholds(args),
            decision=decision,
            preflight=preflight,
            include_content=bool(args.include_content),
            run_id=run_id,
        )
        path, _latest = write_artifact(payload, out_dir=args.output_dir, suite=suite)
        logger.info("Evaluation artifact saved", path=str(path), status=decision.status)
    except Exception as exc:  # noqa: BLE001 - artifact failure must not hide gate code
        logger.error("Could not write evaluation artifact", error_type=type(exc).__name__)


def _print_report(
    suite: EvalSuite,
    results: list[Any],
    scores: dict[str, Any],
    decision: QualityDecision,
) -> None:
    """Print metadata and aggregate scores only; never raw user content."""
    print("\n" + "=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)
    print(f"Suite: {suite.identity}")
    print(f"Samples: {len(results)}")
    print(f"Status: {decision.status} (exit {decision.exit_code})")
    print("\n--- Aggregate Scores ---")
    for metric, value in scores.items():
        if metric == "per_sample" or metric.startswith("_"):
            continue
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    if decision.failures:
        print("\nFailures: " + "; ".join(decision.failures))
    print("=" * 70)


def _print_failure(message: str, code: int) -> None:
    print(f"{message} (exit {code})")


if __name__ == "__main__":
    raise SystemExit(main())
