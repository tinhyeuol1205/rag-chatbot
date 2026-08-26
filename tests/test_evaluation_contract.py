from __future__ import annotations

import asyncio
import json

import pytest

from evaluation.artifacts import build_artifact, decide_quality, write_artifact
from evaluation.dataset import EVAL_SUITE, EvalSample, EvalSuite, EvaluationSuiteError, load_suite
from evaluation.metrics import (
    EvaluationMetricsError,
    _build_judge_llm,
    _classify_provider_error,
    _metric_value,
)
from evaluation.preflight import EvaluationPreflightError, run_preflight
from evaluation.runner import EvalSampleResult, EvaluationRunner


def test_builtin_suite_is_corpus_pinned_and_stable():
    assert EVAL_SUITE.suite_id == "kiemhiep-kimdung-anh-hung-xa-dieu"
    assert EVAL_SUITE.version == "1"
    assert EVAL_SUITE.dataset_id == "kiemhiep_kimdung"
    assert EVAL_SUITE.expected_sources == ("anh_hung_xa_dieu.pdf",)
    assert {sample.sample_id for sample in EVAL_SUITE.samples} == {
        "naming-tinh-khang",
        "guo-jing-childhood",
        "jiangnan-freaks-wager",
        "chen-xuanfeng-defeat",
        "guo-jing-meets-huang-rong",
        "huang-rong-family",
        "dragon-palms-teacher",
        "wanyan-honglie-identity",
        "yang-kang-parentage",
        "beggars-sect-leadership",
    }


def test_suite_rejects_duplicate_id_question_and_unsafe_source():
    sample = EvalSample("a", "same question", "reference", ("doc.md",))
    with pytest.raises(EvaluationSuiteError, match="duplicate sample_id"):
        EvalSuite("suite", "1", "docs", ("doc.md",), (sample, EvalSample("a", "other", "ref", ("doc.md",))))
    with pytest.raises(EvaluationSuiteError, match="duplicate normalized question"):
        EvalSuite("suite", "1", "docs", ("doc.md",), (sample, EvalSample("b", " SAME   QUESTION ", "ref", ("doc.md",))))
    with pytest.raises(EvaluationSuiteError, match="unsafe source"):
        EvalSuite("suite", "1", "docs", ("../doc.md",), (EvalSample("a", "q", "ref"),))


def test_json_loader_rejects_unknown_fields(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"suite_id": "s", "version": "1", "dataset_id": "d", "samples": [], "unknown": 1}))
    with pytest.raises(EvaluationSuiteError, match="unknown suite fields"):
        load_suite(path)


def test_jsonl_loader_accepts_header_plus_samples(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"suite_id": "s", "version": "1", "dataset_id": "d", "expected_sources": ["doc.md"]}),
                json.dumps({"sample_id": "a", "question": "q", "reference": "r", "relevant_sources": ["doc.md"]}),
            ]
        )
    )
    assert load_suite(path).samples[0].sample_id == "a"


def test_redis_runner_uses_queue_payload_and_never_constructs_local_retriever(monkeypatch):
    sample = EVAL_SUITE.samples[0]

    class Queue:
        def enqueue(self, query, history, *, idempotency_key=None):
            assert query == sample.question
            assert history == []
            return type("Job", (), {"job_id": "job-1"})()

        def wait_result(self, job_id, timeout):
            assert job_id == "job-1"
            assert timeout > 0
            return {
                "answer": "answer",
                "contexts": ["context"],
                "sources": [{"citation_id": 1, "file_name": "anh_hung_xa_dieu.pdf"}],
                "expanded_queries": [sample.question],
                "num_candidates": 1,
            }

    def forbidden_factory(_scope):
        raise AssertionError("redis evaluation must not construct a local retriever")

    result = EvaluationRunner(
        EVAL_SUITE,
        queue=Queue(),
        execution_mode="redis_worker",
        retriever_factory=forbidden_factory,
    ).run(sample)
    assert result.status == "success"
    assert result.contexts == ["context"]
    assert result.source_ids == ("anh_hung_xa_dieu.pdf",)


def test_runner_failure_does_not_store_exception_message():
    sample = EVAL_SUITE.samples[0]

    class Queue:
        def enqueue(self, *_args, **_kwargs):
            raise RuntimeError("secret qdrant URL and token")

        def wait_result(self, *_args, **_kwargs):
            raise AssertionError("enqueue should fail first")

    result = EvaluationRunner(EVAL_SUITE, queue=Queue(), execution_mode="redis_worker").run(sample)
    assert result.status == "failed"
    assert result.error_code == "evaluation_sample_failed"
    assert result.error_type == "RuntimeError"
    assert not hasattr(result, "error_message")


def test_redis_runner_cancels_timed_out_job():
    from core.errors import JobTimeoutError

    sample = EVAL_SUITE.samples[0]

    class Queue:
        cancelled = None

        def enqueue(self, *_args, **_kwargs):
            return type("Job", (), {"job_id": "job-timeout"})()

        def wait_result(self, *_args, **_kwargs):
            raise JobTimeoutError("internal timeout")

        def cancel(self, job_id):
            self.cancelled = job_id

    queue = Queue()
    result = EvaluationRunner(EVAL_SUITE, queue=queue, execution_mode="redis_worker").run(sample)
    assert result.status == "failed"
    assert queue.cancelled == "job-timeout"


def test_quality_decision_rejects_fallback_and_missing_judge():
    result = EvalSampleResult(EVAL_SUITE.samples[0], "success", answer="a", contexts=["c"])
    fallback = decide_quality(
        scores={"_mode": "SIMPLE_FALLBACK (local diagnostic only)"},
        results=[result],
        thresholds={},
        max_failed_samples=0,
        max_empty_context_samples=0,
    )
    assert fallback.exit_code == 4
    missing = decide_quality(
        scores={"_mode": "not_evaluated"},
        results=[result],
        thresholds={},
        max_failed_samples=0,
        max_empty_context_samples=0,
    )
    assert missing.exit_code == 4


def test_default_artifact_is_redacted_and_json_standard(tmp_path):
    result = EvalSampleResult(
        EVAL_SUITE.samples[0],
        "success",
        answer="private answer",
        contexts=["private context"],
        sources=[{"file_name": "anh_hung_xa_dieu.pdf"}],
        scores={"context_precision": 0.8},
    )
    decision = decide_quality(
        scores={"context_precision": 0.8},
        results=[result],
        thresholds={"context_precision": 0.8},
        max_failed_samples=0,
        max_empty_context_samples=0,
    )
    payload = build_artifact(
        suite=EVAL_SUITE,
        results=[result],
        scores={"context_precision": 0.8},
        thresholds={"context_precision": 0.8},
        decision=decision,
        run_id="run-1",
    )
    encoded = json.dumps(payload, allow_nan=False)
    assert "private answer" not in encoded
    assert "private context" not in encoded
    assert "Khưu Xử Cơ" not in encoded
    path, latest = write_artifact(payload, out_dir=tmp_path, suite=EVAL_SUITE)
    assert path != latest
    assert path.exists() and latest.exists()


def test_artifact_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="NaN|infinity"):
        build_artifact(
            suite=EVAL_SUITE,
            results=[],
            scores={"faithfulness": float("nan")},
            thresholds={},
            decision=decide_quality(
                scores={"_mode": "not_evaluated"},
                results=[],
                thresholds={},
                max_failed_samples=0,
                max_empty_context_samples=0,
            ),
            run_id="run-nan",
        )


def test_artifact_keeps_only_safe_judge_diagnostics():
    scores = {
        "_mode": "not_evaluated",
        "error_code": "judge_metric_timeout",
        "error_type": "TimeoutError",
        "error_stage": "metric",
        "error_metric": "faithfulness",
        "error_message": "secret API key and endpoint",
    }
    payload = build_artifact(
        suite=EVAL_SUITE,
        results=[],
        scores=scores,
        thresholds={},
        decision=decide_quality(
            scores=scores,
            results=[],
            thresholds={},
            max_failed_samples=0,
            max_empty_context_samples=0,
        ),
        run_id="run-diagnostics",
    )
    assert payload["quality"]["diagnostics"] == {
        "error_code": "judge_metric_timeout",
        "error_type": "TimeoutError",
        "error_stage": "metric",
        "error_metric": "faithfulness",
    }
    assert "secret API key" not in json.dumps(payload)


def test_metric_timeout_has_stable_safe_diagnostics(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EVAL_TIMEOUT_SECONDS", 0.001)
    with pytest.raises(EvaluationMetricsError) as caught:
        asyncio.run(_metric_value(asyncio.sleep(0.05), "faithfulness"))
    assert caught.value.as_diagnostics() == {
        "error_code": "judge_metric_timeout",
        "error_type": "TimeoutError",
        "error_stage": "metric",
        "error_metric": "faithfulness",
    }


def test_provider_error_classifier_uses_status_without_exception_message():
    class ProviderFailure(Exception):
        status_code = 429

    assert _classify_provider_error(ProviderFailure("secret provider response")) == "judge_rate_limited"


def test_provider_error_classifier_unwraps_instructor_retry_rate_limit():
    class ProviderFailure(Exception):
        status_code = 429

    try:
        try:
            raise ProviderFailure("secret quota response")
        except ProviderFailure as cause:
            raise RuntimeError("retry wrapper") from cause
    except RuntimeError as exc:
        assert _classify_provider_error(exc) == "judge_rate_limited"


def test_provider_error_classifier_exposes_missing_dependency_code():
    class ConfigurationError(Exception):
        pass

    try:
        try:
            raise ModuleNotFoundError("jsonref")
        except ModuleNotFoundError as cause:
            raise ConfigurationError("optional schema dependency is missing") from cause
    except ConfigurationError as exc:
        assert _classify_provider_error(exc) == "judge_dependency_missing"


def test_gemini_judge_uses_google_genai_async_transport(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EVAL_JUDGE_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "EVAL_JUDGE_MODEL", "gemini-test")
    monkeypatch.setattr(settings, "EVAL_JUDGE_API_KEY", "dummy-key")
    judge = _build_judge_llm()
    assert judge.provider == "google"
    assert judge.is_async is True
    assert type(judge.client).__name__ == "AsyncInstructor"


def test_worker_stream_result_retains_exact_contexts(monkeypatch):
    from workers import rag_worker

    class Retriever:
        def stream_with_sources(self, _query, *, history, include_contexts):
            assert history == []
            assert include_contexts is True
            yield {"event": "token", "data": "answer"}
            yield {"event": "contexts", "data": ["prompt context"]}
            yield {"event": "metadata", "data": {"expanded_queries": ["q"], "num_candidates": 1}}
            yield {"event": "sources", "data": [{"citation_id": 1, "file_name": "anh_hung_xa_dieu.pdf"}]}

    emitted = []
    monkeypatch.setattr(rag_worker, "get_retriever", lambda: Retriever())
    payload = rag_worker._handle_job_stream("q", [], emitted.append)
    assert payload["contexts"] == ["prompt context"]
    assert payload["expanded_queries"] == ["q"]
    assert [event["event"] for event in emitted] == ["token", "sources"]


def test_preflight_rejects_configured_corpus_mismatch(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "INGEST_DATASET_ID", "other-corpus")
    with pytest.raises(EvaluationPreflightError, match="dataset_id"):
        run_preflight(EVAL_SUITE)


def test_preflight_checks_generation_schema_and_source_coverage(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "INGEST_DATASET_ID", "kiemhiep_kimdung")

    class Point:
        def __init__(self):
            self.payload = {"dataset_id": "kiemhiep_kimdung", "file_name": "other-book.pdf"}

    class Connector:
        def alias_target(self, alias):
            return f"{alias}__generation-1"

        def validate_collection_schema(self, *args, **kwargs):
            return None

        def iter_scroll(self, *_args, **_kwargs):
            return iter([Point()])

    with pytest.raises(EvaluationPreflightError, match="missing expected sources"):
        run_preflight(EVAL_SUITE, connector=Connector(), manifest=None)
