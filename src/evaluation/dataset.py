"""Versioned, corpus-pinned evaluation suites.

An evaluation question is only meaningful when it is evaluated against the
corpus from which its reference answer was written.  The old implementation
used an untyped list and silently inherited ``INGEST_DATASET_ID``.  PR18 makes
that contract explicit and keeps a small compatibility projection named
``EVAL_DATASET`` for callers that still import it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvaluationSuiteError(ValueError):
    """Raised when a suite violates the evaluation contract."""


_SAFE_SOURCE = re.compile(r"^[^/\\][^/\\]*$")
_ALLOWED_SUITE_KEYS = {"suite_id", "version", "dataset_id", "expected_sources", "samples"}
_ALLOWED_SAMPLE_KEYS = {
    "sample_id",
    "question",
    "reference",
    "ground_truth",  # accepted only as a backwards-compatible alias
    "relevant_sources",
    "slice",
    "answerable",
}


@dataclass(frozen=True)
class EvalSample:
    """One deterministic evaluation case."""

    sample_id: str
    question: str
    reference: str
    relevant_sources: tuple[str, ...] = ()
    slice: str = "default"
    answerable: bool = True

    @property
    def ground_truth(self) -> str:
        """Compatibility name used by the pre-PR18 evaluator."""
        return self.reference

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "reference": self.reference,
            "relevant_sources": list(self.relevant_sources),
            "slice": self.slice,
            "answerable": self.answerable,
        }


@dataclass(frozen=True)
class EvalSuite:
    """Immutable evaluation contract pinned to one dataset namespace."""

    suite_id: str
    version: str
    dataset_id: str
    expected_sources: tuple[str, ...]
    samples: tuple[EvalSample, ...]

    def __post_init__(self) -> None:
        _require_text("suite_id", self.suite_id)
        _require_text("version", self.version)
        _require_text("dataset_id", self.dataset_id)
        # Canonicalize identity fields before fingerprinting.  This prevents
        # two semantically identical files from producing different suite IDs
        # merely because an operator added surrounding whitespace.
        object.__setattr__(self, "suite_id", self.suite_id.strip())
        object.__setattr__(self, "version", self.version.strip())
        object.__setattr__(self, "dataset_id", self.dataset_id.strip())
        if not self.samples:
            raise EvaluationSuiteError("suite must contain at least one sample")
        source_names = _unique_safe_sources(self.expected_sources, "expected_sources")
        object.__setattr__(self, "expected_sources", source_names)
        ids: set[str] = set()
        questions: set[str] = set()
        normalized_samples: list[EvalSample] = []
        for sample in self.samples:
            if not isinstance(sample, EvalSample):
                raise EvaluationSuiteError("samples must contain EvalSample values")
            _require_text("sample_id", sample.sample_id)
            _require_text("question", sample.question)
            _require_text("reference", sample.reference)
            sample_id = sample.sample_id.strip()
            if sample_id in ids:
                raise EvaluationSuiteError(f"duplicate sample_id: {sample_id}")
            ids.add(sample_id)
            question_key = _normalize_question(sample.question)
            if question_key in questions:
                raise EvaluationSuiteError("duplicate normalized question in suite")
            questions.add(question_key)
            sources = _unique_safe_sources(sample.relevant_sources, "relevant_sources")
            if any(source not in source_names for source in sources):
                missing = sorted(set(sources) - set(source_names))
                raise EvaluationSuiteError(
                    f"sample {sample.sample_id} references sources outside expected_sources: {missing}"
                )
            normalized_samples.append(
                EvalSample(
                    sample_id=sample_id,
                    question=sample.question.strip(),
                    reference=sample.reference.strip(),
                    relevant_sources=sources,
                    slice=sample.slice.strip() or "default",
                    answerable=bool(sample.answerable),
                )
            )
        object.__setattr__(self, "samples", tuple(normalized_samples))

    @property
    def identity(self) -> str:
        """Stable human-readable identity used in artifact names."""
        return f"{self.suite_id}:v{self.version}:{self.dataset_id}"

    @property
    def fingerprint(self) -> str:
        """Hash the validated suite content without exposing it in artifacts."""
        encoded = json.dumps(
            self.as_dict(include_content=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "suite_id": self.suite_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "expected_sources": list(self.expected_sources),
            "samples": [],
        }
        if include_content:
            payload["samples"] = [sample.as_dict() for sample in self.samples]
        else:
            payload["samples"] = [
                {
                    "sample_id": sample.sample_id,
                    "slice": sample.slice,
                    "answerable": sample.answerable,
                    "relevant_sources": list(sample.relevant_sources),
                }
                for sample in self.samples
            ]
        return payload


def load_suite(path: str | Path) -> EvalSuite:
    """Load JSON or header-plus-samples JSONL and validate the contract."""
    suite_path = Path(path)
    try:
        raw_text = suite_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationSuiteError(f"could not read evaluation suite: {suite_path}") from exc
    if suite_path.suffix.lower() == ".jsonl":
        payload = _jsonl_payload(raw_text)
    else:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise EvaluationSuiteError(f"invalid evaluation suite JSON: {suite_path}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationSuiteError("evaluation suite root must be an object")
    unknown = set(payload) - _ALLOWED_SUITE_KEYS
    if unknown:
        raise EvaluationSuiteError(f"unknown suite fields: {sorted(unknown)}")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise EvaluationSuiteError("suite.samples must be an array")
    expected_sources = payload.get("expected_sources", [])
    if not isinstance(expected_sources, list):
        raise EvaluationSuiteError("suite.expected_sources must be an array")
    samples = tuple(_sample_from_mapping(item) for item in raw_samples)
    return EvalSuite(
        suite_id=str(payload.get("suite_id", "")),
        version=str(payload.get("version", "")),
        dataset_id=str(payload.get("dataset_id", "")),
        expected_sources=tuple(str(value) for value in expected_sources),
        samples=samples,
    )


def _jsonl_payload(raw_text: str) -> Mapping[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(raw_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationSuiteError(f"invalid evaluation suite JSONL at line {line_number}") from exc
        if not isinstance(value, Mapping):
            raise EvaluationSuiteError(f"evaluation suite JSONL line {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise EvaluationSuiteError("evaluation suite JSONL is empty")
    header = dict(rows[0])
    if not {"suite_id", "version", "dataset_id"}.issubset(header):
        raise EvaluationSuiteError("JSONL suite requires a first-line suite header")
    if "samples" in header:
        raise EvaluationSuiteError("JSONL suite header must not contain samples")
    unknown = set(header) - (_ALLOWED_SUITE_KEYS - {"samples"})
    if unknown:
        raise EvaluationSuiteError(f"unknown suite fields: {sorted(unknown)}")
    header["samples"] = [dict(row) for row in rows[1:]]
    return header


def _sample_from_mapping(item: Any) -> EvalSample:
    if not isinstance(item, Mapping):
        raise EvaluationSuiteError("each suite sample must be an object")
    unknown = set(item) - _ALLOWED_SAMPLE_KEYS
    if unknown:
        raise EvaluationSuiteError(f"unknown sample fields: {sorted(unknown)}")
    reference = item.get("reference", item.get("ground_truth", ""))
    if (
        "reference" in item
        and "ground_truth" in item
        and str(item["reference"]).strip() != str(item["ground_truth"]).strip()
    ):
        raise EvaluationSuiteError("reference and ground_truth disagree")
    sources = item.get("relevant_sources", [])
    if not isinstance(sources, list):
        raise EvaluationSuiteError("sample.relevant_sources must be an array")
    return EvalSample(
        sample_id=str(item.get("sample_id", "")),
        question=str(item.get("question", "")),
        reference=str(reference),
        relevant_sources=tuple(str(value) for value in sources),
        slice=str(item.get("slice", "default")),
        answerable=bool(item.get("answerable", True)),
    )


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationSuiteError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise EvaluationSuiteError(f"{name} contains a NUL character")


def _unique_safe_sources(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise EvaluationSuiteError(f"{field_name} must be a sequence")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise EvaluationSuiteError(f"{field_name} contains an empty source")
        source = value.strip()
        # Source labels are deliberately not paths.  This rejects absolute,
        # parent-traversal and platform-specific separator forms.
        if not _SAFE_SOURCE.fullmatch(source) or source in {".", ".."}:
            raise EvaluationSuiteError(f"unsafe source label in {field_name}: {source!r}")
        if source not in normalized:
            normalized.append(source)
    return tuple(normalized)


def _normalize_question(value: str) -> str:
    return " ".join(value.casefold().split())


# The versioned JSON is the single source of truth for both the CLI default and
# callers that request the built-in suite.  This prevents the Python constant
# and release suite file from drifting apart.
_DEFAULT_SUITE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "eval"
    / "suites"
    / "kiemhiep-kimdung.json"
)
EVAL_SUITE = load_suite(_DEFAULT_SUITE_PATH)


# Backwards-compatible projection for notebooks and older callers.
EVAL_DATASET = [
    {"question": sample.question, "ground_truth": sample.reference}
    for sample in EVAL_SUITE.samples
]
