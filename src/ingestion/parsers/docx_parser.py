"""DOCX Parser — Đọc file Word (.docx), trích xuất text."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from core import get_logger
from core.config import settings
from ingestion.models import DocumentMetadata, ParseQuality, RawDocument
from ingestion.parsers.base import BaseParser

logger = get_logger(__name__)


class DocxParser(BaseParser):
    def parse(self, file_path: Path) -> list[RawDocument]:
        return list(self.iter_documents(file_path))

    def iter_documents(self, file_path: Path) -> Iterator[RawDocument]:
        # Import lazy — tránh load unstructured khi không cần
        from unstructured.partition.docx import partition_docx

        logger.info("Parsing DOCX", file=file_path.name)
        elements = partition_docx(filename=str(file_path))
        yield from self._documents_from_elements(elements, file_path)

    @staticmethod
    def _documents_from_elements(elements, file_path: Path) -> Iterator[RawDocument]:
        window: list[str] = []
        window_start = 0
        section_title: str | None = None
        for index, element in enumerate(elements):
            text = str(element).strip()
            if not text:
                continue
            category = type(element).__name__
            if category.lower() in {"title", "header", "heading"}:
                section_title = text
            window.append(text)
            if len(window) >= settings.INGEST_DOCUMENT_WINDOW:
                yield RawDocument(
                    content="\n\n".join(window),
                    metadata=DocumentMetadata(
                        file_name=file_path.name,
                        file_type="docx",
                        section_title=section_title,
                        source_path=str(file_path),
                        source_uri=str(file_path),
                        structural_anchor=f"element:{window_start}",
                        offset_start=window_start,
                        offset_end=index + 1,
                    ),
                )
                window = []
                window_start = index + 1
        if window:
            yield RawDocument(
                content="\n\n".join(window),
                metadata=DocumentMetadata(
                    file_name=file_path.name,
                    file_type="docx",
                    section_title=section_title,
                    source_path=str(file_path),
                    source_uri=str(file_path),
                    structural_anchor=f"element:{window_start}",
                    offset_start=window_start,
                    offset_end=len(elements),
                ),
            )

    def parse_with_quality(self, file_path: Path) -> tuple[list[RawDocument], ParseQuality]:
        from unstructured.partition.docx import partition_docx

        elements = partition_docx(filename=str(file_path))
        quality = ParseQuality(elements_seen=len(elements))
        for element in elements:
            category = type(element).__name__.lower()
            if "table" in category:
                quality.table_elements += 1
            elif "image" in category:
                quality.image_elements += 1
            elif "caption" in category:
                quality.caption_elements += 1
            elif category not in {"title", "header", "heading", "narrativetext", "listitem", "text"}:
                quality.unsupported_elements += 1
        documents = list(self._documents_from_elements(elements, file_path))
        quality.documents_emitted = len(documents)
        quality.characters_emitted = sum(len(document.content) for document in documents)
        quality.replacement_characters = sum(document.content.count("\ufffd") for document in documents)
        if not documents:
            quality.pages_empty = 1
        return documents, quality
