from __future__ import annotations

"""Markdown Parser — Đọc file .md, chia theo headers (## hoặc #)."""

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

        # Chia theo header cấp 1 hoặc 2 (# hoặc ##)
        # Regex: tìm dòng bắt đầu bằng # (sau newline)
        sections = re.split(r"\n(?=#{1,2}\s)", text)

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
                        file_type="md",
                        section_title=section_title,
                        source_path=str(file_path),
                    ),
                )
            )

        logger.info("Markdown parsed", file=file_path.name, sections=len(documents))
        return documents
