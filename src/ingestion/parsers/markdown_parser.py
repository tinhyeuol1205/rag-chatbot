"""Markdown Parser — Đọc file .md, chia theo headers (## hoặc #)."""

from __future__ import annotations

import re
from pathlib import Path

from core import get_logger
from ingestion.models import DocumentMetadata, RawDocument
from ingestion.parsers.base import BaseParser

logger = get_logger(__name__)


class MarkdownParser(BaseParser):
    def parse(self, file_path: Path) -> list[RawDocument]:
        logger.info("Parsing Markdown", file=file_path.name)

        text = file_path.read_text(encoding="utf-8")

        # Split theo header cấp 1-3 (#, ##, ###), BỎ QUA các dòng # nằm trong
        # fenced code block (```...```) — bug P1-2: # trong code bị split sai,
        # và ### không được tách section.
        in_fence = False
        sections: list[str] = []
        current: list[str] = []
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            is_header = (not in_fence) and re.match(r"#{1,3}\s", line)
            if is_header and current:
                sections.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            sections.append("\n".join(current))

        documents = []
        for section in sections:
            content = section.strip()
            if not content:
                continue

            # Trích xuất tiêu đề section
            first_line = content.split("\n")[0]
            section_title = first_line.lstrip("#").strip() if first_line.startswith("#") else None

            documents.append(
                RawDocument(
                    content=content,
                    metadata=DocumentMetadata(
                        file_name=file_path.name,
                        # ★ .txt ≠ md (dispatcher map .txt → MarkdownParser)
                        file_type=file_path.suffix.lstrip(".").lower() or "md",
                        section_title=section_title,
                        source_path=str(file_path),
                    ),
                )
            )

        logger.info("Markdown parsed", file=file_path.name, sections=len(documents))
        return documents
