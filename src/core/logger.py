from __future__ import annotations

"""
Structured logging với structlog.

Tại sao dùng structlog thay vì logging tiêu chuẩn?
- Log có cấu trúc key=value, dễ parse bằng máy (ELK, Datadog)
- Tự động thêm context (module name, log level)
- Output đẹp hơn trong terminal (có màu)

Usage:
    from core import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document", file="report.pdf", chunks=5)
    # Output: [info] Processing document  file=report.pdf  chunks=5  module=ingestion.pipeline
"""

import structlog


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Tạo logger instance, bind với tên module để biết log từ đâu."""

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),  # Output đẹp có màu trong terminal
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger().bind(module=name)
