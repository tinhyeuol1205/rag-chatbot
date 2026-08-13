"""
Base Parser — Abstract interface cho tất cả document parsers.

Mọi parser (PDF, MD, DOCX) đều phải implement method parse().
Đây là Strategy Pattern: cùng interface, khác implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from ingestion.models import ParseQuality, RawDocument


class BaseParser(ABC):
    """Abstract base class — mọi parser phải implement parse()."""

    @abstractmethod
    def parse(self, file_path: Path) -> list[RawDocument]:
        """Đọc file → trả về danh sách RawDocument.

        Mỗi RawDocument thường = 1 trang (PDF) hoặc 1 section (Markdown).
        Trả về list vì 1 file có thể có nhiều trang/sections.
        """

    def iter_documents(self, file_path: Path) -> Iterator[RawDocument]:
        """Stream parsed documents; legacy parsers are wrapped by default."""
        yield from self.parse(file_path)

    def parse_with_quality(self, file_path: Path) -> tuple[list[RawDocument], ParseQuality]:
        """Compatibility helper returning documents plus baseline quality counters."""
        documents = list(self.iter_documents(file_path))
        chars = sum(len(document.content) for document in documents)
        return documents, ParseQuality(
            documents_emitted=len(documents),
            characters_emitted=chars,
        )
