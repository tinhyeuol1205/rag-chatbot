from __future__ import annotations

"""PDF Parser — Đọc file PDF, trích xuất text theo từng trang."""

from pathlib import Path

from core import get_logger
from ingestion.models import DocumentMetadata, RawDocument
from ingestion.parsers.base import BaseParser

logger = get_logger(__name__)


class PDFParser(BaseParser):
    def parse(self, file_path: Path) -> list[RawDocument]:
        # Import lazy — tránh load unstructured khi không cần
        from unstructured.partition.pdf import partition_pdf

        logger.info("Parsing PDF", file=file_path.name)

        # partition_pdf trả về list các Element (Title, NarrativeText, Table, ...)
        elements = partition_pdf(filename=str(file_path), strategy="fast")

        # Nhóm các elements theo page_number
        pages: dict[int, list[str]] = {}
        for el in elements:
            page = el.metadata.page_number or 1
            pages.setdefault(page, []).append(str(el))

        # Mỗi page → 1 RawDocument
        documents = []
        for page_num, texts in sorted(pages.items()):
            content = "\n\n".join(texts).strip()
            if content:
                documents.append(
                    RawDocument(
                        content=content,
                        metadata=DocumentMetadata(
                            file_name=file_path.name,
                            file_type="pdf",
                            page_number=page_num,
                            source_path=str(file_path),
                        ),
                    )
                )

        logger.info("PDF parsed", file=file_path.name, pages=len(documents))
        return documents
