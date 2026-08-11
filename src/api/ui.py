"""
Gradio Chat UI — Giao diện chat trực quan.

Features:
  - Chat interface với streaming response
  - Hiển thị tên dự án và mô tả
  - Ví dụ câu hỏi mẫu để user thử nhanh

Chạy bằng: make run-ui
Hoặc:      cd src && python -m api.ui
"""

from __future__ import annotations

import gradio as gr

from api.chat import chat_stream_events, get_retriever
from core import get_logger

logger = get_logger(__name__)

# Câu hỏi mẫu để user thử nhanh
EXAMPLE_QUESTIONS = [
    "How many days of annual leave do employees get?",
    "What equipment does the company provide for remote workers?",
    "What is the Git branching strategy?",
    "How should security incidents be reported?",
    "What happens during the first day of onboarding?",
    "What is the password policy?",
    "How many code review approvals are needed?",
]


def respond(message: str, chat_history: list):
    """Xử lý message từ user, trả về streaming response (multi-turn).

    Args:
        message: Tin nhắn mới từ user
        chat_history: Lịch sử chat (Gradio format) — dùng để condense follow-up

    Yields:
        Từng token để Gradio hiển thị streaming
    """
    # Streaming: ghép token và render sources sau khi generation kết thúc.
    response = ""
    for event in chat_stream_events(message, history=chat_history):
        if isinstance(event, str):
            # Tương thích với caller/test cũ nếu event adapter bị thay thế.
            response += event
        elif event.get("event") == "token":
            token = event.get("data", "")
            response += str(token.get("text", "") if isinstance(token, dict) else token)
        elif event.get("event") == "sources":
            sources = event.get("data", [])
            if isinstance(sources, dict):
                sources = sources.get("sources", [])
            response += _format_sources(sources)
        elif event.get("event") == "error":
            error = event.get("data", {})
            if isinstance(error, dict):
                message_text = error.get("message", "An unexpected error occurred.")
            else:
                message_text = str(error) if error else "An unexpected error occurred."
            response += f"\n\n⚠️ {message_text}"
        yield response


def _format_sources(sources: list[dict]) -> str:
    """Render source metadata thành Markdown ở cuối câu trả lời."""
    if not sources:
        return ""
    lines = ["\n\nSources:"]
    for source in sources:
        parts = [source.get("file_name") or "Unknown"]
        if source.get("section_title"):
            parts.append(source["section_title"])
        if source.get("page_number") is not None:
            parts.append(f"p.{source['page_number']}")
        lines.append(f"- [{source.get('citation_id')}] {' → '.join(parts)}")
    return "\n".join(lines)


def create_ui() -> gr.ChatInterface:
    """Tạo Gradio ChatInterface."""

    demo = gr.ChatInterface(
        fn=respond,
        title="🤖 RAG Chatbot — Internal Knowledge Base",
        description="Ask questions about company policies and engineering practices.\nPowered by **Advanced RAG** (Hybrid Search, Reranking, Parent-Child Retrieval).",
        examples=EXAMPLE_QUESTIONS,
        cache_examples=False,
    )

    return demo


def main():
    logger.info("Starting Gradio UI")
    # Fail-fast: config/model errors hiện ở đây, không phải request đầu tiên
    # (bug P2-14 — warmup load embedding/reranker model 1 lần).
    get_retriever().warmup()
    demo = create_ui()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
