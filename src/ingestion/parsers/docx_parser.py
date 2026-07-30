from __future__ import annotations

"""DOCX Parser — Đọc file Word (.docx), trích xuất text."""

from pathlib import Path

from core import get_logger
from ingestion.models import DocumentMetadata, RawDocument
from ingestion.parsers.base import BaseParser

logger = get_logger(__name__)


class DocxParser(BaseParser):
    def parse(self, file_path: Path) -> list[RawDocument]:
        # Import lazy — tránh load unstructured khi không cần
        from unstructured.partition.docx import partition_docx

        logger.info("Parsing DOCX", file=file_path.name)

        elements = partition_docx(filename=str(file_path))

        # Gộp tất cả elements thành 1 document
        # (DOCX thường không có page_number rõ ràng như PDF)
        content = "\n\n".join(str(el) for el in elements if str(el).strip())

        if not content.strip():
            logger.warning("DOCX is empty", file=file_path.name)
            return []

        documents = [
            RawDocument(
                content=content,
                metadata=DocumentMetadata(
                    file_name=file_path.name,
                    file_type="docx",
                    source_path=str(file_path),
                ),
            )
        ]

        logger.info("DOCX parsed", file=file_path.name, elements=len(elements))
        return documents
