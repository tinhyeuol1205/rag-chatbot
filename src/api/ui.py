"""Gradio thin UI for the FastAPI-backed dev/product demos.

Run with ``make run-ui UI_DEMO_MODE=dev`` or
``make run-ui UI_DEMO_MODE=product``.  This module intentionally contains no
retriever, Qdrant, Redis queue, or model-runtime imports: FastAPI is the only
backend boundary visible to the browser-facing UI process.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Iterator
from typing import Any

import gradio as gr

from api.ui_client import UIAPIClient, UIClientError, normalize_history, sanitize_sources
from core import get_logger
from core.config import settings

logger = get_logger(__name__)

EXAMPLE_QUESTIONS = [
    "How many days of annual leave do employees get?",
    "What equipment does the company provide for remote workers?",
    "What is the Git branching strategy?",
    "How should security incidents be reported?",
    "What happens during the first day of onboarding?",
    "What is the password policy?",
    "How many code review approvals are needed?",
]

_client: UIAPIClient | None = None


def get_ui_client() -> UIAPIClient:
    """Create one HTTP client per UI process and reuse its connection pool."""
    global _client
    if _client is None:
        _client = UIAPIClient()
    return _client


def respond(
    message: str,
    chat_history: Iterable[Any] | None,
    *,
    client: UIAPIClient | None = None,
) -> Iterator[str]:
    """Yield the accumulated answer as FastAPI SSE token events arrive."""
    api = client or get_ui_client()
    history = normalize_history(chat_history, max_turns=settings.UI_HISTORY_MAX_TURNS)
    response = ""
    try:
        for event in api.stream(message, history):
            event_name = event.get("event", "message")
            data = event.get("data", {})
            if event_name == "token":
                token = data.get("text", "") if isinstance(data, dict) else data
                response += str(token)
            elif event_name == "sources":
                source_rows = data.get("sources", []) if isinstance(data, dict) else data
                response += _format_sources(sanitize_sources(source_rows))
            elif event_name == "error":
                response += f"\n\n⚠️ {_error_text(data)}"
            if event_name in {"token", "sources", "error"}:
                yield response
    except UIClientError as exc:
        retry_hint = f" (Retry-After: {exc.retry_after}s)" if exc.retry_after else ""
        logger.warning("UI API request failed", error_code=exc.code, status_code=exc.status_code)
        yield f"⚠️ {exc.message}{retry_hint}"
    except Exception:
        logger.exception("UI callback failed")
        yield "⚠️ Không thể kết nối tới API. Vui lòng thử lại sau."


def _format_sources(sources: list[dict[str, Any]]) -> str:
    """Render only allowlisted, escaped citation metadata as plain Markdown."""
    if not sources:
        return ""
    lines = ["\n\nSources:"]
    for source in sources:
        parts = [html.escape(str(source.get("file_name", "Unknown")))]
        section = source.get("section_title")
        if section:
            parts.append(html.escape(str(section)))
        if source.get("page_number") is not None:
            parts.append(f"p.{html.escape(str(source['page_number']))}")
        citation = html.escape(str(source.get("citation_id", "?")))
        lines.append(f"- [{citation}] {' → '.join(parts)}")
    return "\n".join(lines)


def _error_text(data: Any) -> str:
    """Extract a safe API error message and preserve retry guidance."""
    if isinstance(data, dict):
        message = str(data.get("message", "Yêu cầu không hoàn tất."))
    else:
        message = str(data or "Yêu cầu không hoàn tất.")
    return message


def create_ui(client: UIAPIClient | None = None) -> gr.ChatInterface:
    """Create the Gradio view; all backend calls remain in ``UIAPIClient``."""
    api = client or get_ui_client()

    def respond_from_gradio(message: str, history: list[Any]):
        yield from respond(message, history, client=api)

    demo = gr.ChatInterface(
        fn=respond_from_gradio,
        title="🤖 RAG Chatbot — Internal Knowledge Base",
        description=(
            "Ask questions about company policies and engineering practices.\n"
            "Powered by **FastAPI + Advanced RAG**."
        ),
        examples=EXAMPLE_QUESTIONS,
        cache_examples=False,
    )
    # This is only a small browser/UI transport bound.  Redis admission and the
    # provider quota remain authoritative in product mode; Gradio never calls
    # the RAG pipeline directly.
    demo.queue(
        max_size=settings.RAG_QUEUE_MAX_OUTSTANDING,
        default_concurrency_limit=settings.RAG_WORKER_CONCURRENCY,
    )
    return demo


def main() -> None:
    """Validate topology before listening, then launch the configured UI."""
    api = get_ui_client()
    state = api.preflight(settings.UI_DEMO_MODE)
    logger.info(
        "Starting thin RAG UI",
        demo_mode=settings.UI_DEMO_MODE,
        execution_mode=state.execution_mode,
        api_base_url=settings.UI_API_BASE_URL,
    )
    auth = None
    if settings.UI_AUTH_USERNAME and settings.UI_AUTH_PASSWORD:
        auth = (settings.UI_AUTH_USERNAME, settings.UI_AUTH_PASSWORD)
    try:
        create_ui(api).launch(
            server_name=settings.UI_HOST,
            server_port=settings.UI_PORT,
            share=settings.UI_PUBLIC_SHARE,
            auth=auth,
            show_error=False,
        )
    finally:
        api.close()


if __name__ == "__main__":
    main()
