"""Execute evaluation samples through the configured RAG admission topology."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

from core import get_logger
from core.admission_queue import (
    InlineAdmissionQueue,
    RedisAdmissionQueue,
    get_admission_queue,
)
from core.config import settings
from core.errors import JobTimeoutError
from evaluation.dataset import EvalSample, EvalSuite
from retrieval.result_codec import result_from_payload
from retrieval.retriever import RAGResult, RAGRetriever
from retrieval.scope import RetrievalScope

logger = get_logger(__name__)


@dataclass
class EvalSampleResult:
    """In-memory result; content is deliberately omitted by release artifacts."""

    sample: EvalSample
    status: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    num_candidates: int = 0
    latency_seconds: float = 0.0
    error_code: str | None = None
    error_type: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    deterministic: dict[str, float] = field(default_factory=dict)

    @property
    def empty_context(self) -> bool:
        return self.status == "success" and not self.contexts

    @property
    def question(self) -> str:
        return self.sample.question

    @property
    def ground_truth(self) -> str:
        return self.sample.reference

    @property
    def source_ids(self) -> tuple[str, ...]:
        values: list[str] = []
        for source in self.sources:
            if not isinstance(source, dict):
                continue
            name = source.get("file_name") or source.get("source_uri")
            if isinstance(name, str) and name.strip():
                label = PurePath(name.replace("\\", "/")).name
                if label and label not in values:
                    values.append(label)
        return tuple(values)

    def as_metric_input(self):
        """Return the legacy metric shape without exposing it to artifacts."""
        from evaluation.metrics import EvalResult

        return EvalResult(
            question=self.sample.question,
            answer=self.answer,
            ground_truth=self.sample.reference,
            contexts=list(self.contexts),
            sample_id=self.sample.sample_id,
            relevant_sources=self.sample.relevant_sources,
            answerable=self.sample.answerable,
        )


class EvaluationRunner:
    """Run samples using inline admission or the Redis worker contract.

    The Redis path never constructs a retriever in the evaluator process.  The
    worker owns model clients, reservation context and the exact contexts from
    the final prompt.  This is the same boundary used by product traffic.
    """

    def __init__(
        self,
        suite: EvalSuite,
        *,
        queue: InlineAdmissionQueue | RedisAdmissionQueue | None = None,
        retriever_factory: Callable[[RetrievalScope], RAGRetriever] | None = None,
        execution_mode: str | None = None,
        timeout_seconds: float | None = None,
        run_id: str | None = None,
    ) -> None:
        self.suite = suite
        self.queue = queue
        self.execution_mode = execution_mode or settings.RAG_EXECUTION_MODE
        if self.execution_mode not in {"inline", "redis_worker"}:
            raise ValueError("evaluation execution_mode must be inline or redis_worker")
        self.timeout_seconds = (
            settings.RAG_JOB_MAX_WAIT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        if self.timeout_seconds <= 0:
            raise ValueError("evaluation timeout_seconds must be positive")
        self.run_id = run_id or uuid.uuid4().hex
        self._retriever_factory = retriever_factory or (lambda scope: RAGRetriever(scope=scope))
        self._retriever: RAGRetriever | None = None

    def run(self, sample: EvalSample) -> EvalSampleResult:
        started = time.monotonic()
        try:
            payload = self._execute(sample)
            result = result_from_payload(payload)
            if not isinstance(result, RAGResult):
                raise TypeError("RAG worker returned an invalid result payload")
            sample_result = EvalSampleResult(
                sample=sample,
                status="success",
                answer=result.answer,
                contexts=list(result.contexts),
                sources=[source.as_dict() if hasattr(source, "as_dict") else dict(source) for source in result.sources],
                expanded_queries=list(result.expanded_queries),
                num_candidates=result.num_candidates,
            )
        except Exception as exc:  # noqa: BLE001 - result records safe type/code only
            sample_result = EvalSampleResult(
                sample=sample,
                status="failed",
                error_code=_safe_error_code(exc),
                error_type=type(exc).__name__,
            )
        sample_result.latency_seconds = max(0.0, time.monotonic() - started)
        logger.info(
            "Evaluation sample completed",
            sample_id=sample.sample_id,
            status=sample_result.status,
            num_contexts=len(sample_result.contexts),
            latency_ms=round(sample_result.latency_seconds * 1000, 2),
        )
        return sample_result

    def run_all(self) -> list[EvalSampleResult]:
        return [self.run(sample) for sample in self.suite.samples]

    def _execute(self, sample: EvalSample) -> Any:
        queue = self.queue or get_admission_queue()
        if self.execution_mode == "inline":
            if not isinstance(queue, InlineAdmissionQueue) and not callable(getattr(queue, "execute", None)):
                raise TypeError("inline evaluation requires InlineAdmissionQueue")
            retriever = self._get_retriever()
            return queue.execute(
                lambda query, history: retriever.query_with_context(query, history=history),
                sample.question,
                [],
            )
        if not (
            isinstance(queue, RedisAdmissionQueue)
            or (
                callable(getattr(queue, "enqueue", None))
                and callable(getattr(queue, "wait_result", None))
            )
        ):
            raise TypeError("redis_worker evaluation requires RedisAdmissionQueue")
        job = queue.enqueue(
            sample.question,
            [],
            idempotency_key=f"evaluation:{self.run_id}:{sample.sample_id}",
        )
        try:
            return queue.wait_result(job.job_id, self.timeout_seconds)
        except JobTimeoutError:
            # Match the API contract: release a queued outstanding slot or
            # request cancellation for a job that is already running.
            try:
                queue.cancel(job.job_id)
            except Exception:  # noqa: BLE001 - original timeout is the safe result
                logger.warning("Could not cancel timed-out evaluation job", error_type="cancel_failed")
            raise

    def _get_retriever(self) -> RAGRetriever:
        if self._retriever is None:
            scope = RetrievalScope((self.suite.dataset_id,))
            self._retriever = self._retriever_factory(scope)
        return self._retriever


def _safe_error_code(exc: Exception) -> str:
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_.:-]{1,80}", code):
        return code
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "evaluation_timeout"
    if "queue" in name or "redis" in name:
        return "evaluation_queue_error"
    return "evaluation_sample_failed"
