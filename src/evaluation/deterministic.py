"""Judge-free retrieval, citation and abstention metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import PurePath
from typing import Any


def compute_deterministic_metrics(results: Iterable[Any]) -> dict[str, Any]:
    values = list(results)
    if not values:
        return {}
    per_sample: list[dict[str, float]] = []
    successful = [value for value in values if getattr(value, "status", "success") == "success"]
    for value in successful:
        expected = _labels(getattr(getattr(value, "sample", None), "relevant_sources", ()))
        actual = _labels(_source_ids(value))
        source_recall = 1.0 if not expected else len(expected & actual) / len(expected)
        citation_recall = source_recall
        answerable = bool(getattr(getattr(value, "sample", None), "answerable", True))
        abstained = _is_abstention(value)
        abstention_accuracy = 1.0 if answerable != abstained else 0.0
        sample_metrics = {
            "source_recall": _finite(source_recall),
            "citation_source_recall": _finite(citation_recall),
            "abstention_accuracy": _finite(abstention_accuracy),
            "empty_context": 1.0 if not getattr(value, "contexts", None) else 0.0,
        }
        if hasattr(value, "deterministic"):
            value.deterministic = dict(sample_metrics)
        per_sample.append(sample_metrics)
    denominator = len(values)
    empty_context_ratio = sum(
        1 for value in values if getattr(value, "status", "success") == "success" and not getattr(value, "contexts", None)
    ) / denominator
    aggregate: dict[str, float | int | str | list[dict[str, float]]] = {
        "source_recall": _mean(per_sample, "source_recall"),
        "citation_source_recall": _mean(per_sample, "citation_source_recall"),
        "abstention_accuracy": _mean(per_sample, "abstention_accuracy"),
        "empty_context_ratio": _finite(empty_context_ratio),
        "num_successful_samples": len(successful),
        "per_sample": per_sample,
    }
    return aggregate


def _source_ids(value: Any) -> list[str]:
    if hasattr(value, "source_ids"):
        return list(value.source_ids)
    sources = getattr(value, "sources", []) or []
    output: list[str] = []
    for source in sources:
        if isinstance(source, dict):
            source = source.get("file_name") or source.get("source_uri") or ""
        if source:
            output.append(str(source))
    return output


def _labels(values: Any) -> set[str]:
    labels: set[str] = set()
    for value in values or ():
        if not isinstance(value, str):
            continue
        normalized = value.replace("\\", "/").rstrip("/")
        labels.add(PurePath(normalized).name.casefold())
    return labels


def _is_abstention(value: Any) -> bool:
    if not getattr(value, "contexts", None):
        return True
    answer = str(getattr(value, "answer", "")).casefold()
    markers = (
        "không tìm thấy thông tin",
        "không có đủ thông tin",
        "cannot answer from the provided context",
        "i don't have enough information",
        "no relevant context",
    )
    return any(marker in answer for marker in markers)


def _mean(values: list[dict[str, float]], key: str) -> float:
    if not values:
        return 0.0
    return _finite(sum(value[key] for value in values) / len(values))


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("deterministic evaluation metric is not finite")
    return float(value)
