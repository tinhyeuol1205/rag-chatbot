"""Bounded batching helpers shared by ingestion and the Qdrant connector.

Point count alone is not a safe request-size limit: one PDF table or a large
payload can make a small point batch exceed the HTTP/proxy limit.  The helpers
therefore enforce both a count and an approximate serialized byte budget while
keeping at most one batch alive at a time.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def approximate_size(value: object) -> int:
    """Return a conservative UTF-8 JSON size estimate for a request item."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))


def iter_batches(
    values: Iterable[T],
    *,
    max_items: int,
    max_bytes: int,
    size_of: Callable[[T], int] = approximate_size,
) -> Iterator[list[T]]:
    """Yield bounded batches without materializing the input iterable.

    A single item larger than ``max_bytes`` is yielded alone.  It cannot be
    split without changing its semantics, so the caller receives a warning or
    server-side rejection rather than an infinite batching loop.
    """
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    batch: list[T] = []
    batch_bytes = 0
    for value in values:
        value_bytes = max(size_of(value), 1)
        exceeds_items = len(batch) >= max_items
        exceeds_bytes = bool(batch) and batch_bytes + value_bytes > max_bytes
        if exceeds_items or exceeds_bytes:
            yield batch
            batch = []
            batch_bytes = 0

        batch.append(value)
        batch_bytes += value_bytes

    if batch:
        yield batch

