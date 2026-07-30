from __future__ import annotations

"""
Custom Exceptions cho RAG Chatbot.

Tại sao cần custom exceptions?
- Phân biệt rõ lỗi từ đâu (parsing? retrieval? config?)
- Caller có thể catch theo loại cụ thể thay vì catch chung Exception
- Log message rõ ràng hơn khi debug

Usage:
    from core.errors import ParsingError
    raise ParsingError("Cannot parse file: corrupted PDF")
"""


class RAGChatbotError(Exception):
    """Base exception — tất cả lỗi trong project kế thừa từ đây."""
    pass


class ConfigurationError(RAGChatbotError):
    """Thiếu hoặc sai config (API key, DB connection, ...)."""
    pass


class ParsingError(RAGChatbotError):
    """Lỗi khi parse document (file hỏng, format không hỗ trợ, ...)."""
    pass


class IngestionError(RAGChatbotError):
    """Lỗi trong ingestion pipeline (chunking, embedding, store, ...)."""
    pass


class RetrievalError(RAGChatbotError):
    """Lỗi trong retrieval pipeline (search, rerank, ...)."""
    pass
