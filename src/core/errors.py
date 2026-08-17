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

from __future__ import annotations


class RAGChatbotError(Exception):
    """Base exception với message nội bộ và contract an toàn cho client.

    ``str(exc)`` được giữ cho log/debug nội bộ. API/UI phải dùng
    ``error_code`` và ``public_message`` để không vô tình trả exception của SDK,
    URL nội bộ hoặc chi tiết hạ tầng ra ngoài.
    """

    error_code = "rag_error"
    public_message = (
        "Xin lỗi, hệ thống đang gặp sự cố khi xử lý câu hỏi. "
        "Vui lòng thử lại sau ít phút."
    )


class ConfigurationError(RAGChatbotError):
    """Thiếu hoặc sai config (API key, DB connection, ...)."""

    error_code = "configuration_error"
    public_message = "Hệ thống chưa được cấu hình đúng. Vui lòng liên hệ quản trị viên."


class ParsingError(RAGChatbotError):
    """Lỗi khi parse document (file hỏng, format không hỗ trợ, ...)."""

    error_code = "parsing_error"
    public_message = "Không thể đọc một tài liệu trong kho dữ liệu."


class IngestionError(RAGChatbotError):
    """Lỗi trong ingestion pipeline (chunking, embedding, store, ...)."""

    error_code = "ingestion_error"
    public_message = "Không thể cập nhật kho tài liệu. Vui lòng thử lại sau."


class RetrievalError(RAGChatbotError):
    """Lỗi trong retrieval pipeline (search, rerank, ...)."""

    error_code = "retrieval_unavailable"
    public_message = "Kho tài liệu tạm thời không khả dụng. Vui lòng thử lại sau."


class QueueFullError(RAGChatbotError):
    """The distributed admission queue has no outstanding-job capacity."""

    error_code = "queue_full"
    public_message = "Hệ thống đang bận. Vui lòng thử lại sau."


class QueueUnavailableError(RAGChatbotError):
    """Redis could not be reached; production must fail closed."""

    error_code = "queue_unavailable"
    public_message = "Hệ thống xếp hàng tạm thời không khả dụng. Vui lòng thử lại sau."


class JobTimeoutError(RAGChatbotError):
    """A queued job exceeded its result wait/retention contract."""

    error_code = "job_timeout"
    public_message = "Yêu cầu mất quá nhiều thời gian xử lý. Vui lòng thử lại."


class JobFailedError(RAGChatbotError):
    """The worker completed a job with an internal pipeline failure."""

    error_code = "job_failed"
    public_message = "Không thể hoàn tất yêu cầu. Vui lòng thử lại sau ít phút."
