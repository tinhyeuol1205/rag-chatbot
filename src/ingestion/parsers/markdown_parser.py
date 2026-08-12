"""Markdown Parser — Đọc file .md, chia theo headers (## hoặc #)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from core import get_logger
from ingestion.models import DocumentMetadata, ParseQuality, RawDocument
from ingestion.parsers.base import BaseParser

logger = get_logger(__name__)


class MarkdownParser(BaseParser):
    def parse(self, file_path: Path) -> list[RawDocument]:
        return list(self.iter_documents(file_path))

    def iter_documents(self, file_path: Path) -> Iterator[RawDocument]:
        logger.info("Parsing Markdown", file=file_path.name)
        section_lines: list[str] = []
        section_start = 0
        section_title: str | None = None
        in_fence = False
        fence_marker: str | None = None

        def emit(end_line: int) -> RawDocument | None:
            content = "\n".join(section_lines).strip()
            if not content:
                return None
            first_line = content.split("\n", 1)[0]
            title = section_title or (
                first_line.lstrip("#").strip() if first_line.startswith("#") else None
            )
            return RawDocument(
                content=content,
                metadata=DocumentMetadata(
                    file_name=file_path.name,
                    file_type=file_path.suffix.lstrip(".").lower() or "md",
                    section_title=title,
                    source_path=str(file_path),
                    source_uri=str(file_path),
                    structural_anchor=f"line:{section_start + 1}",
                    offset_start=section_start,
                    offset_end=end_line,
                ),
            )

        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle):
                line = raw_line.rstrip("\n")
                stripped = line.lstrip()
                fence_match = re.match(r"(`{3,}|~{3,})", stripped)
                if fence_match:
                    marker = fence_match.group(1)[0]
                    if not in_fence:
                        in_fence = True
                        fence_marker = marker
                    elif marker == fence_marker:
                        in_fence = False
                        fence_marker = None

                is_header = not in_fence and bool(re.match(r"#{1,6}\s", line))
                if is_header and section_lines:
                    document = emit(line_number)
                    if document is not None:
                        yield document
                    section_lines = []
                    section_start = line_number
                    section_title = line.lstrip("#").strip()
                elif is_header:
                    section_title = line.lstrip("#").strip()
                    section_start = line_number
                section_lines.append(line)

        document = emit(section_start + len(section_lines))
        if document is not None:
            yield document

        logger.info("Markdown parsed", file=file_path.name)

    def parse_with_quality(self, file_path: Path) -> tuple[list[RawDocument], ParseQuality]:
        documents = list(self.iter_documents(file_path))
        chars = sum(len(document.content) for document in documents)
        return documents, ParseQuality(
            documents_emitted=len(documents),
            characters_emitted=chars,
            elements_seen=len(documents),
            replacement_characters=sum(document.content.count("\ufffd") for document in documents),
        )
