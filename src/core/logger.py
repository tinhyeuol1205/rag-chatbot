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
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def _configure() -> None:
    """Configure structlog ĐÚNG 1 LẦN cho cả process.

    Bug P3-2: bản cũ gọi structlog.configure() mỗi lần get_logger() (khoảng 20 lần),
    không có log level (debug luôn in), in ra stdout (trộn output chương trình).
    """
    global _configured
    if _configured:
        return

    from core.config import settings

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.LOG_JSON
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),   # timestamp để debug latency
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,          # ★ cho logger.exception() in traceback
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),   # ★ level filtering
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr), # ★ stderr
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str):
    """Tạo logger instance, bind với tên module để biết log từ đâu."""
    _configure()
    return structlog.get_logger().bind(module=name)
