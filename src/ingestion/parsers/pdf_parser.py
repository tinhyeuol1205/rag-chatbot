"""Streaming PDF parser with bounded internal page windows.

The public source remains one PDF.  Internally the parser writes only a small
page slice to a spooled temporary file before invoking Unstructured.  This is
important because ``partition_pdf`` returns an in-memory element list and does
not expose an end-page argument.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from core import get_logger
from core.config import settings
from ingestion.models import DocumentMetadata, ParseQuality, ParseWindow, RawDocument
from ingestion.parsers.base import BaseParser

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """Parse one PDF without materializing the complete document."""

    def parse(self, file_path: Path) -> list[RawDocument]:
        """Legacy materializing API; production ingestion uses ``iter_windows``."""
        return [document for window in self.iter_windows(file_path) for document in window.documents]

    def _partition(
        self,
        file_path: Path,
        strategy: str,
        *,
        file: BinaryIO | None = None,
        starting_page_number: int = 1,
    ):
        # Import lazy — tránh load unstructured khi không cần.
        from unstructured.partition.pdf import partition_pdf

        kwargs = {
            "strategy": strategy,
            "starting_page_number": starting_page_number,
        }
        if file is None:
            kwargs["filename"] = str(file_path)
        else:
            # Unstructured requires exactly one of ``filename`` and ``file``.
            # The bytes handed to it are always the bounded spool, never the
            # source PDF; source identity is restored in RawDocument metadata.
            kwargs["file"] = file
        return partition_pdf(**kwargs)

    def iter_documents(self, file_path: Path) -> Iterator[RawDocument]:
        logger.info("Parsing PDF with page windows", file=file_path.name)
        for window in self.iter_windows(file_path):
            yield from window.documents

    def iter_windows(
        self,
        file_path: Path,
        *,
        start_page: int | None = None,
    ) -> Iterator[ParseWindow]:
        """Yield one bounded ParseWindow at a time.

        ``start_page`` is used by resumable ingestion to avoid parsing windows
        that were already committed and verified in the manifest.
        """
        reader = self._reader(file_path)
        page_count = len(reader.pages)
        self._validate_page_count(file_path, page_count)
        first_page = max(1, start_page or 1)
        if first_page > page_count:
            return

        window_size = settings.INGEST_PDF_PAGE_WINDOW
        for start in range(first_page, page_count + 1, window_size):
            end = min(start + window_size - 1, page_count)
            started = time.monotonic()
            window = self._parse_window(reader, file_path, start, end)
            elapsed = time.monotonic() - started
            if elapsed > settings.INGEST_PDF_WINDOW_TIMEOUT_SECONDS:
                raise TimeoutError("pdf_window_timeout")
            yield window

    def parse_with_quality(self, file_path: Path) -> tuple[list[RawDocument], ParseQuality]:
        """Compatibility API that deliberately materializes all windows."""
        documents: list[RawDocument] = []
        quality = ParseQuality()
        for window in self.iter_windows(file_path):
            documents.extend(window.documents)
            quality = quality.merge(window.quality)
        return documents, quality

    @staticmethod
    def _reader(file_path: Path):
        try:
            from pypdf import PdfReader

            return PdfReader(str(file_path), strict=False)
        except Exception as exc:
            raise ValueError(f"Unable to read PDF source ({type(exc).__name__})") from exc

    @staticmethod
    def _page_count(file_path: Path) -> int:
        """Compatibility helper used by older quality tests."""
        try:
            return len(PDFParser._reader(file_path).pages)
        except Exception:  # noqa: BLE001 - optional quality metadata
            return 0

    @staticmethod
    def _validate_page_count(file_path: Path, page_count: int) -> None:
        if page_count <= 0:
            raise ValueError("pdf_has_no_pages")
        if page_count > settings.INGEST_PDF_MAX_PAGES:
            raise ValueError(
                f"pdf_page_limit_exceeded:{page_count}>{settings.INGEST_PDF_MAX_PAGES}"
            )

    @contextmanager
    def _window_file(self, reader, page_numbers: Sequence[int]):
        """Create and clean a bounded, owner-only PDF spool for selected pages."""
        from pypdf import PdfWriter

        with tempfile.SpooledTemporaryFile(
            max_size=settings.INGEST_PDF_SPOOL_MAX_MB * 1024 * 1024,
            mode="w+b",
        ) as spool:
            writer = PdfWriter()
            for page_number in page_numbers:
                writer.add_page(reader.pages[page_number - 1])
            writer.write(spool)
            spool.seek(0)
            yield spool

    def _parse_window(
        self,
        reader,
        file_path: Path,
        start_page: int,
        end_page: int,
    ) -> ParseWindow:
        pages = list(range(start_page, end_page + 1))
        local_to_global = {local: page for local, page in enumerate(pages, start=1)}
        with self._window_file(reader, pages) as window_file:
            fast_elements = self._partition(
                file_path,
                settings.INGEST_PDF_FAST_STRATEGY,
                file=window_file,
                starting_page_number=start_page,
            )
        fast_documents, fast_quality = self._documents_from_elements(
            fast_elements,
            file_path,
            page_map=local_to_global,
            strict_page_metadata=True,
        )
        del fast_elements

        fast_by_page = {document.metadata.page_number: document for document in fast_documents}
        usable_pages = {
            page
            for page, document in fast_by_page.items()
            if page is not None and len(document.content.strip()) >= settings.INGEST_PARSER_MIN_TEXT_CHARS
        }
        if settings.INGEST_PDF_OCR_MODE == "always":
            ocr_targets = set(pages)
        elif settings.INGEST_PDF_OCR_MODE == "missing_pages":
            ocr_targets = set(pages) - usable_pages
        else:
            ocr_targets = set()

        merged_by_page = dict(fast_by_page)
        ocr_page_numbers: set[int] = set()
        ocr_quality = ParseQuality()
        if ocr_targets:
            for group in self._contiguous_groups(sorted(ocr_targets), settings.INGEST_PDF_OCR_PAGE_WINDOW):
                group_map = {local: page for local, page in enumerate(group, start=1)}
                with self._window_file(reader, group) as ocr_file:
                    ocr_elements = self._partition(
                        file_path,
                        settings.INGEST_PDF_OCR_STRATEGY,
                        file=ocr_file,
                        starting_page_number=group[0],
                    )
                ocr_documents, group_quality = self._documents_from_elements(
                    ocr_elements,
                    file_path,
                    page_map=group_map,
                    strict_page_metadata=True,
                )
                del ocr_elements
                ocr_quality = ocr_quality.merge(group_quality)
                for document in ocr_documents:
                    page = document.metadata.page_number
                    if page in ocr_targets:
                        merged_by_page[page] = document
                        ocr_page_numbers.add(page)

        documents = [
            merged_by_page[page]
            for page in pages
            if page in merged_by_page and merged_by_page[page].content.strip()
        ]
        quality = ParseQuality(
            pages_seen=len(pages),
            pages_empty=sum(
                page not in merged_by_page or not merged_by_page[page].content.strip()
                for page in pages
            ),
            ocr_pages=len(ocr_page_numbers),
            table_elements=fast_quality.table_elements + ocr_quality.table_elements,
            image_elements=fast_quality.image_elements + ocr_quality.image_elements,
            caption_elements=fast_quality.caption_elements + ocr_quality.caption_elements,
            unsupported_elements=fast_quality.unsupported_elements + ocr_quality.unsupported_elements,
            elements_seen=fast_quality.elements_seen + ocr_quality.elements_seen,
            documents_emitted=len(documents),
            characters_emitted=sum(len(document.content) for document in documents),
            replacement_characters=sum(document.content.count("\ufffd") for document in documents),
        )
        estimated_bytes = sum(len(document.content.encode("utf-8")) * 2 + 1024 for document in documents)
        logger.info(
            "Parsed PDF window",
            file=file_path.name,
            page_start=start_page,
            page_end=end_page,
            pages_emitted=len(documents),
            ocr_pages=sorted(ocr_page_numbers),
            estimated_bytes=estimated_bytes,
        )
        return ParseWindow(
            start_page=start_page,
            end_page=end_page,
            documents=documents,
            quality=quality,
            estimated_bytes=estimated_bytes,
            ocr_pages=tuple(sorted(ocr_page_numbers)),
            ordinal=(start_page - 1) // settings.INGEST_PDF_PAGE_WINDOW,
        )

    @staticmethod
    def _contiguous_groups(pages: list[int], max_size: int) -> Iterator[list[int]]:
        if max_size <= 0:
            raise ValueError("INGEST_PDF_OCR_PAGE_WINDOW must be positive")
        if not pages:
            return
        group: list[int] = [pages[0]]
        for page in pages[1:]:
            if page == group[-1] + 1 and len(group) < max_size:
                group.append(page)
            else:
                yield group
                group = [page]
        yield group

    @staticmethod
    def _documents_from_elements(
        elements,
        file_path: Path,
        *,
        page_map: Mapping[int, int] | None = None,
        strict_page_metadata: bool = False,
    ) -> tuple[list[RawDocument], ParseQuality]:
        """Group elements by global page and preserve page-level coordinates."""
        elements = list(elements)
        pages: dict[int, list[str]] = {}
        page_boxes: dict[int, tuple[float, float, float, float]] = {}
        quality = ParseQuality(elements_seen=len(elements))
        global_pages = set(page_map.values()) if page_map else set()
        local_pages = set(page_map) if page_map else set()
        for element in elements:
            metadata = getattr(element, "metadata", None)
            raw_page = getattr(metadata, "page_number", None)
            try:
                parsed_page = int(raw_page or 0)
            except (TypeError, ValueError):
                parsed_page = 0
            if raw_page is None or parsed_page <= 0:
                if strict_page_metadata:
                    raise ValueError("pdf_element_missing_page_metadata")
                page = 1
            else:
                raw_page = parsed_page
                if page_map and raw_page in local_pages:
                    page = page_map[raw_page]
                elif page_map and raw_page in global_pages:
                    page = raw_page
                else:
                    page = raw_page
            if page_map and page not in global_pages:
                raise ValueError("pdf_element_page_out_of_window")

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
                    page_boxes[page] = box if previous_box is None else (
                        min(previous_box[0], box[0]),
                        min(previous_box[1], box[1]),
                        max(previous_box[2], box[2]),
                        max(previous_box[3], box[3]),
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

        documents: list[RawDocument] = []
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
        quality.pages_seen = len(global_pages) if page_map else max(pages, default=0)
        quality.pages_empty = sum(
            not "\n".join(values).strip() for values in pages.values()
        )
        quality.documents_emitted = len(documents)
        quality.characters_emitted = sum(len(document.content) for document in documents)
        quality.replacement_characters = sum(document.content.count("\ufffd") for document in documents)
        return documents, quality
