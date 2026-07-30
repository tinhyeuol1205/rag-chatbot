"""
Document Parsers — Đọc file và trích xuất text.

Pattern: Strategy (giống llm-twin-course/data_crawling/crawlers/base.py)
- BaseParser định nghĩa interface chung
- Mỗi loại file có parser riêng (PDF, MD, DOCX)
- ParserDispatcher tự động chọn parser phù hợp dựa trên đuôi file

Tương tự CrawlerDispatcher trong llm-twin-course:
  CrawlerDispatcher.get_crawler(url)  → MediumCrawler / GithubCrawler
  ParserDispatcher.get_parser(path)   → PDFParser / MarkdownParser
"""

from .dispatcher import ParserDispatcher

__all__ = ["ParserDispatcher"]
