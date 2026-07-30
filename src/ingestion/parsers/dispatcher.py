from __future__ import annotations

"""
Parser Dispatcher — Tự động chọn parser phù hợp theo đuôi file.

Pattern: Dispatcher (giống llm-twin-course/data_crawling/dispatcher.py)
  CrawlerDispatcher nhận URL  → chọn MediumCrawler / GithubCrawler
  ParserDispatcher  nhận Path → chọn PDFParser / MarkdownParser
"""

from pathlib import Path

from core import get_logger
from ingestion.parsers.base import BaseParser
from ingestion.parsers.docx_parser import DocxParser
from ingestion.parsers.markdown_parser import MarkdownParser
from ingestion.parsers.pdf_parser import PDFParser

logger = get_logger(__name__)

# Registry: mapping đuôi file → parser class
_PARSER_REGISTRY: dict[str, type[BaseParser]] = {
    ".pdf": PDFParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".txt": MarkdownParser,  # Txt xử lý giống Markdown
    ".docx": DocxParser,
}


class ParserDispatcher:
    """Nhận file path, tự động chọn parser phù hợp."""

    @staticmethod
    def get_parser(file_path: Path) -> BaseParser:
        suffix = file_path.suffix.lower()
        parser_cls = _PARSER_REGISTRY.get(suffix)

        if parser_cls is None:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {list(_PARSER_REGISTRY.keys())}"
            )

        logger.info("Selected parser", file=file_path.name, parser=parser_cls.__name__)
        return parser_cls()

    @staticmethod
    def supported_extensions() -> list[str]:
        return list(_PARSER_REGISTRY.keys())
