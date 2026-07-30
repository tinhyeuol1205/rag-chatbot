from __future__ import annotations

"""
Base Parser — Abstract interface cho tất cả document parsers.

Mọi parser (PDF, MD, DOCX) đều phải implement method parse().
Đây là Strategy Pattern: cùng interface, khác implementation.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ingestion.models import RawDocument


class BaseParser(ABC):
    """Abstract base class — mọi parser phải implement parse()."""

    @abstractmethod
    def parse(self, file_path: Path) -> list[RawDocument]:
        """Đọc file → trả về danh sách RawDocument.

        Mỗi RawDocument thường = 1 trang (PDF) hoặc 1 section (Markdown).
        Trả về list vì 1 file có thể có nhiều trang/sections.
        """
        pass
