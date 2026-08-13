"""PDF Parser — Đọc file PDF, trích xuất text theo từng trang."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from core import get_logger
from core.config import settings
from ingestion.models import DocumentMetadata, ParseQuality, RawDocument
from ingestion.parsers.base import BaseParser

logger = get_logger(__name__)


class PDFParser(BaseParser):
    def parse(self, file_path: Path) -> list[RawDocument]:
        return list(self.iter_documents(file_path))

    def _partition(self, file_path: Path, strategy: str):
        # Import lazy — tránh load unstructured khi không cần
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(filename=str(file_path), strategy=strategy)

    def iter_documents(self, file_path: Path) -> Iterator[RawDocument]:
        logger.info("Parsing PDF", file=file_path.name)
        documents, _ = self.parse_with_quality(file_path)
        yield from documents

    def parse_with_quality(self, file_path: Path) -> tuple[list[RawDocument], ParseQuality]:
        elements = self._partition(file_path, settings.INGEST_PDF_FAST_STRATEGY)
        documents, quality = self._documents_from_elements(elements, file_path)
        page_count = self._page_count(file_path)
        fast_pages = {
            document.metadata.page_number
            for document in documents
            if document.metadata.page_number is not None
        }
        if not documents or (page_count and len(fast_pages) < page_count):
            try:
                ocr_elements = self._partition(file_path, settings.INGEST_PDF_OCR_STRATEGY)
            except Exception:
                logger.exception("PDF OCR fallback failed", file=file_path.name)
            else:
                ocr_documents, ocr_quality = self._documents_from_elements(ocr_elements, file_path)
                existing_pages = {
                    document.metadata.page_number
                    for document in documents
                    if document.metadata.page_number is not None
                }
                # Keep fast text for pages it already extracted and append only
                # OCR pages that were missing, avoiding duplicate citations.
                documents.extend(
                    document
                    for document in ocr_documents
                    if document.metadata.page_number not in existing_pages
                )
                quality = quality.merge(ocr_quality)
                quality.documents_emitted = len(documents)
                quality.characters_emitted = sum(len(document.content) for document in documents)
                quality.replacement_characters = sum(
                    document.content.count("\ufffd") for document in documents
                )
                quality.pages_seen = max(page_count, quality.pages_seen, ocr_quality.pages_seen)
                quality.pages_empty = max(
                    quality.pages_seen
                    - len({document.metadata.page_number for document in documents}),
                    0,
                )
                quality.ocr_pages = len({
                    document.metadata.page_number
                    for document in ocr_documents
                    if document.metadata.page_number not in existing_pages
                })
        if page_count:
            extracted_pages = len({
                document.metadata.page_number
                for document in documents
                if document.metadata.page_number is not None
            })
            quality.pages_seen = max(quality.pages_seen, page_count)
            quality.pages_empty = max(page_count - extracted_pages, 0)
        return documents, quality

    @staticmethod
    def _page_count(file_path: Path) -> int:
        """Read only the PDF page index when available; tolerate missing extras."""
        try:
            from pypdf import PdfReader

            return len(PdfReader(str(file_path), strict=False).pages)
        except Exception:  # noqa: BLE001 - page counting is optional quality metadata
            return 0

    @staticmethod
    def _documents_from_elements(elements, file_path: Path) -> tuple[list[RawDocument], ParseQuality]:
        pages: dict[int, list[str]] = {}
        page_boxes: dict[int, tuple[float, float, float, float]] = {}
        quality = ParseQuality(elements_seen=len(elements))
        for element in elements:
            metadata = getattr(element, "metadata", None)
            page = getattr(metadata, "page_number", None) or 1
            text = str(element).strip()
            pages.setdefault(page, []).append(text)
            coordinates = getattr(metadata, "coordinates", None)
            points = getattr(coordinates, "points", None)
            if points:
                try:
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                except (IndexError, TypeError, ValueError):
                    xs, ys = [], []
                if xs and ys:
                    box = (min(xs), min(ys), max(xs), max(ys))
                    previous_box = page_boxes.get(page)
                    page_boxes[page] = (
                        box if previous_box is None else (
                            min(previous_box[0], box[0]),
                            min(previous_box[1], box[1]),
                            max(previous_box[2], box[2]),
                            max(previous_box[3], box[3]),
                        )
                    )
            category = type(element).__name__.lower()
            if "table" in category:
                quality.table_elements += 1
            elif "image" in category:
                quality.image_elements += 1
            elif "caption" in category:
                quality.caption_elements += 1
            elif not any(token in category for token in ("title", "text", "list", "header", "footer")):
                quality.unsupported_elements += 1
        quality.pages_seen = max(pages, default=0)
        quality.pages_empty = sum(not "\n".join(values).strip() for values in pages.values())
        documents = []
        for page_num, texts in sorted(pages.items()):
            content = "\n\n".join(text for text in texts if text).strip()
            if content:
                documents.append(
                    RawDocument(
                        content=content,
                        metadata=DocumentMetadata(
                            file_name=file_path.name,
                            file_type="pdf",
                            page_number=page_num,
                            source_path=str(file_path),
                            source_uri=str(file_path),
                            structural_anchor=f"page:{page_num}",
                            bbox=page_boxes.get(page_num),
                            element_type="page",
                        ),
                    )
                )
        quality.documents_emitted = len(documents)
        quality.characters_emitted = sum(len(document.content) for document in documents)
        quality.replacement_characters = sum(document.content.count("\ufffd") for document in documents)
        return documents, quality
