"""Strict RAG quality metrics and project-runtime adapters.

RAGAS is intentionally imported lazily.  This keeps unit tests and the
deterministic diagnostic path usable without importing optional judge SDKs, while
the release path fails closed when a configured judge cannot be constructed.
"""

from __future__ import annotations

import asyncio
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ragas.embeddings.base import BaseRagasEmbedding

from core import get_logger
from core.config import settings

logger = get_logger(__name__)


class EvaluationMetricsError(RuntimeError):
    """Raised when strict judge/metric evaluation cannot complete.

    Diagnostics are deliberately limited to stable codes and type names.  The
    original exception message is never exported because SDK errors can contain
    API keys, endpoints, prompts, or model responses.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "judge_evaluation_failed",
        stage: str = "judge",
        metric: str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.stage = stage
        self.metric = metric
        self.error_type = error_type or type(self).__name__

    def as_diagnostics(self) -> dict[str, str]:
        values = {
            "error_code": self.error_code,
            "error_type": self.error_type,
            "error_stage": self.stage,
        }
        if self.metric:
            values["error_metric"] = self.metric
        return values


@dataclass
class EvalResult:
    """A metric input plus optional status metadata.

    The first five fields preserve the pre-PR18 constructor used by notebooks
    and tests.  New runners populate the stable sample and source contract.
    """

    question: str
    answer: str
    ground_truth: str
    contexts: list[str]
    context_relevance: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    sample_id: str = ""
    relevant_sources: tuple[str, ...] = ()
    answerable: bool = True
    status: str = "success"
    sources: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class ProjectRagasEmbedding(BaseRagasEmbedding):
    """RAGAS embedding adapter backed by the project's EmbeddingService.

    It preserves local/remote model, revision, normalization and dimension
    settings and sends judge-generated batches with ``priority=batch``.
    """

    def __init__(self, service: Any | None = None):
        super().__init__()
        if service is None:
            from ingestion.embeddings import EmbeddingService

            service = EmbeddingService()
        self.service = service

    def embed_text(self, text: str, **_: Any) -> list[float]:
        return list(self.service.embed([text], priority="batch")[0])

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    def embed_texts(self, texts: list[str], **_: Any) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.service.embed(list(texts), priority="batch")
        return [list(vector) for vector in vectors]

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_texts, texts, **kwargs)


def evaluate_results(
    results: Iterable[Any],
    *,
    allow_simple_fallback: bool = False,
) -> dict[str, Any]:
    """Evaluate successful, non-empty samples with modern RAGAS metrics.

    Failed or empty-context samples are reported by the caller and are never
    included in aggregate quality metrics.  Strict mode propagates judge/API/
    schema errors.  The fallback is an explicitly labelled local diagnostic and
    never returns RAGAS metric keys.
    """
    values = list(results)
    eligible = [
        value
        for value in values
        if getattr(value, "status", "success") == "success"
        and getattr(value, "contexts", None)
        and getattr(value, "answer", "").strip()
    ]
    if not eligible:
        raise EvaluationMetricsError(
            "no successful non-empty samples available for judge evaluation",
            error_code="judge_no_eligible_samples",
            stage="input",
        )
    try:
        scores = _ragas_evaluate(eligible)
    except Exception as exc:
        if not allow_simple_fallback:
            if isinstance(exc, EvaluationMetricsError):
                raise
            raise EvaluationMetricsError(
                "RAGAS judge evaluation failed",
                error_code=_classify_provider_error(exc),
                stage="judge",
                error_type=type(exc).__name__,
            ) from exc
        logger.warning("Using explicitly requested simple evaluation fallback", error_type=type(exc).__name__)
        return _simple_evaluate(eligible)
    return scores


def evaluate_with_ragas(results: list[EvalResult], *, allow_simple_fallback: bool = False) -> dict[str, Any]:
    """Compatibility wrapper with strict-by-default semantics."""
    return evaluate_results(results, allow_simple_fallback=allow_simple_fallback)


def _ragas_evaluate(results: list[Any]) -> dict[str, Any]:
    """Run modern collections metrics with an explicit concurrency bound."""
    try:
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness
    except Exception as exc:  # pragma: no cover - exercised in environments without RAGAS
        raise EvaluationMetricsError(
            "modern RAGAS collections API is unavailable",
            error_code="ragas_collections_unavailable",
            stage="setup",
            error_type=type(exc).__name__,
        ) from exc
    judge = _build_judge_llm()
    embeddings = _build_judge_embeddings()
    try:
        metrics = (
            ContextPrecision(llm=judge),
            Faithfulness(llm=judge),
            AnswerRelevancy(llm=judge, embeddings=embeddings),
        )
    except Exception as exc:
        raise EvaluationMetricsError(
            "could not construct RAGAS evaluation metrics",
            error_code="judge_metric_construction_failed",
            stage="setup",
            error_type=type(exc).__name__,
        ) from exc
    # ``asyncio.run`` cannot be called by notebooks with an active loop.  Run
    # the bounded coroutine in a short-lived helper thread in that case instead
    # of weakening strictness or creating unbounded tasks.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        per_sample = asyncio.run(_score_all(results, metrics))
    else:
        per_sample = _run_coroutine_in_thread(_score_all(results, metrics))
    aggregate: dict[str, float] = {}
    for result, sample_scores in zip(results, per_sample):
        _assign_scores(result, sample_scores)
        for name, value in sample_scores.items():
            aggregate.setdefault(name, 0.0)
            aggregate[name] += value
    if not per_sample:
        raise EvaluationMetricsError(
            "RAGAS returned no sample scores",
            error_code="judge_no_scores",
            stage="aggregate",
        )
    for name in list(aggregate):
        aggregate[name] /= len(per_sample)
        _ensure_finite(name, aggregate[name])
    aggregate["num_evaluated_samples"] = len(per_sample)
    aggregate["_mode"] = "ragas_collections_v2"
    aggregate["per_sample"] = [dict(scores) for scores in per_sample]
    return aggregate


async def _score_all(results: list[Any], metrics: tuple[Any, ...]) -> list[dict[str, float]]:
    semaphore = asyncio.Semaphore(max(1, int(settings.EVAL_MAX_WORKERS)))

    async def score_one(result: Any) -> dict[str, float]:
        async with semaphore:
            common = {
                "user_input": result.question,
                "reference": result.ground_truth,
                "retrieved_contexts": list(result.contexts),
            }
            values: dict[str, float] = {}
            # Explicit sequential metric calls keep each sample's provider
            # request count bounded; the semaphore bounds cross-sample fan-out.
            values["context_precision"] = await _metric_value(
                metrics[0].ascore(**common), "context_precision"
            )
            values["faithfulness"] = await _metric_value(
                metrics[1].ascore(
                    user_input=result.question,
                    response=result.answer,
                    retrieved_contexts=list(result.contexts),
                ),
                "faithfulness",
            )
            values["answer_relevancy"] = await _metric_value(
                metrics[2].ascore(user_input=result.question, response=result.answer),
                "answer_relevancy",
            )
            return values

    return await asyncio.gather(*(score_one(result) for result in results))


async def _metric_value(awaitable: Any, metric_name: str) -> float:
    try:
        value = await asyncio.wait_for(awaitable, timeout=settings.EVAL_TIMEOUT_SECONDS)
    except EvaluationMetricsError:
        raise
    except Exception as exc:
        raise EvaluationMetricsError(
            "RAGAS metric request failed",
            error_code=_classify_provider_error(exc),
            stage="metric",
            metric=metric_name,
            error_type=type(exc).__name__,
        ) from exc
    score = getattr(value, "value", value)
    try:
        number = float(score)
    except (TypeError, ValueError) as exc:
        raise EvaluationMetricsError(
            "RAGAS metric returned a non-numeric value",
            error_code="judge_metric_non_numeric",
            stage="metric",
            metric=metric_name,
            error_type=type(exc).__name__,
        ) from exc
    if not math.isfinite(number):
        raise EvaluationMetricsError(
            "RAGAS metric returned a non-finite value",
            error_code="judge_metric_non_finite",
            stage="metric",
            metric=metric_name,
        )
    return number


def _assign_scores(result: Any, scores: dict[str, float]) -> None:
    if hasattr(result, "scores"):
        result.scores = dict(scores)
    # Preserve the old field names for callers that still render them.
    if hasattr(result, "context_relevance"):
        result.context_relevance = scores.get("context_precision", 0.0)
    if hasattr(result, "faithfulness"):
        result.faithfulness = scores.get("faithfulness", 0.0)
    if hasattr(result, "answer_relevance"):
        result.answer_relevance = scores.get("answer_relevancy", 0.0)


def _build_judge_llm():
    """Build an explicit, independent structured-output judge.

    No ``None`` is returned: RAGAS must never silently choose OpenAI or a
    candidate model when the evaluation provider is absent/misconfigured.
    """
    provider = settings.EVAL_JUDGE_PROVIDER.strip().lower()
    model = settings.EVAL_JUDGE_MODEL.strip()
    api_key = settings.EVAL_JUDGE_API_KEY.strip()
    if provider not in {"openai", "gemini"}:
        raise EvaluationMetricsError(
            "EVAL_JUDGE_PROVIDER must be explicitly set to openai or gemini",
            error_code="judge_config_invalid",
            stage="config",
        )
    if not model:
        raise EvaluationMetricsError(
            "EVAL_JUDGE_MODEL must be explicitly pinned",
            error_code="judge_config_invalid",
            stage="config",
        )
    if not api_key:
        raise EvaluationMetricsError(
            "EVAL_JUDGE_API_KEY is required for the configured judge",
            error_code="judge_config_invalid",
            stage="config",
        )
    try:
        from ragas.llms import llm_factory
        if provider == "openai":
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": settings.EVAL_TIMEOUT_SECONDS,
                "max_retries": settings.EVAL_MAX_RETRIES,
            }
            if settings.EVAL_JUDGE_BASE_URL.strip():
                kwargs["base_url"] = settings.EVAL_JUDGE_BASE_URL.strip()
            client = AsyncOpenAI(**kwargs)
            return llm_factory(model, provider="openai", client=client, adapter="auto")
        import instructor
        from google import genai
        from ragas.llms import InstructorLLM

        client = genai.Client(api_key=api_key)
        # ``instructor.from_genai(..., use_async=True)`` keeps the original
        # Client for SDK compatibility and routes each request through this
        # async transport.  Passing the default sync wrapper to RAGAS would
        # make every collections metric fail at ``llm.agenerate``.
        patched_client = instructor.from_genai(client, use_async=True)
        return InstructorLLM(client=patched_client, model=model, provider="google")
    except EvaluationMetricsError:
        raise
    except Exception as exc:
        raise EvaluationMetricsError(
            "could not construct configured evaluation judge",
            error_code="judge_construction_failed",
            stage="judge_setup",
            error_type=type(exc).__name__,
        ) from exc


def _build_judge_embeddings():
    """Return the project embedding adapter (local or remote), never a second model."""
    try:
        return ProjectRagasEmbedding()
    except Exception as exc:
        raise EvaluationMetricsError(
            "could not construct project evaluation embeddings",
            error_code="judge_embedding_construction_failed",
            stage="embedding_setup",
            error_type=type(exc).__name__,
        ) from exc


def _simple_evaluate(results: Iterable[Any]) -> dict[str, Any]:
    """Explicit local diagnostic fallback; not a quality-gate metric."""
    values = list(results)
    if not values:
        return {}
    keyword_scores = []
    for result in values:
        answer_words = _tokens(getattr(result, "answer", ""))
        truth_words = _tokens(getattr(result, "ground_truth", ""))
        if truth_words:
            keyword_scores.append(len(answer_words & truth_words) / len(truth_words))
    return {
        "_mode": "SIMPLE_FALLBACK (local diagnostic only)",
        "num_samples": len(values),
        "avg_answer_length": sum(len(getattr(value, "answer", "")) for value in values) / len(values),
        "avg_keyword_overlap": sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0,
        "has_context_ratio": sum(1 for value in values if getattr(value, "contexts", None)) / len(values),
    }


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return set(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def _ensure_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise EvaluationMetricsError(
            f"metric {name} is not finite",
            error_code="judge_metric_non_finite",
            stage="aggregate",
            metric=name,
        )


def _classify_provider_error(exc: BaseException) -> str:
    """Map SDK failures to stable codes without inspecting sensitive messages."""
    chain = tuple(_exception_chain(exc))
    if any(isinstance(item, (asyncio.TimeoutError, TimeoutError)) for item in chain):
        return "judge_metric_timeout"

    type_names = tuple(type(item).__name__.casefold() for item in chain)
    if any(
        _exception_status(item) == 429
        or any(marker in type_name for marker in ("ratelimit", "resourceexhausted", "toomanyrequests"))
        for item, type_name in zip(chain, type_names)
    ):
        return "judge_rate_limited"
    if any(isinstance(item, ModuleNotFoundError) for item in chain):
        return "judge_dependency_missing"

    if any(
        _exception_status(item) in {401, 403}
        or any(
            marker in type_name
            for marker in ("authentication", "unauthorized", "permissiondenied", "forbidden")
        )
        for item, type_name in zip(chain, type_names)
    ):
        return "judge_auth_failed"
    if any(
        any(marker in type_name for marker in ("connection", "transport", "network"))
        for type_name in type_names
    ):
        return "judge_connection_failed"
    if any(
        any(
            marker in type_name
            for marker in ("validation", "parse", "parsing", "instructor", "structuredoutput", "jsondecode")
        )
        for type_name in type_names
    ):
        return "judge_structured_output_failed"
    if any("configuration" in type_name or "config" in type_name for type_name in type_names):
        return "judge_configuration_failed"
    return "judge_metric_failed"


def _exception_chain(exc: BaseException):
    """Yield an exception and safe nested causes, including Instructor retries."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for related in (current.__cause__, current.__context__):
            if isinstance(related, BaseException):
                pending.append(related)
        for attempt in getattr(current, "failed_attempts", ()) or ():
            related = getattr(attempt, "exception", None)
            if isinstance(related, BaseException):
                pending.append(related)


def _exception_status(exc: BaseException) -> int | None:
    candidates = (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    for value in candidates:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _run_coroutine_in_thread(coroutine):
    import threading

    result: list[Any] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except Exception as exc:  # noqa: BLE001 - propagate coroutine errors after joining
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]
