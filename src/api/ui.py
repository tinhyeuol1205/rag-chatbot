from __future__ import annotations

"""
Gradio Chat UI — Giao diện chat trực quan.

Features:
  - Chat interface với streaming response
  - Hiển thị tên dự án và mô tả
  - Ví dụ câu hỏi mẫu để user thử nhanh

Chạy bằng: make run-ui
Hoặc:      cd src && python -m api.ui
"""

import gradio as gr

from api.chat import chat_stream
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
    """Xử lý message từ user, trả về streaming response.

    Args:
        message: Tin nhắn mới từ user
        chat_history: Lịch sử chat (Gradio format)

    Yields:
        Từng token để Gradio hiển thị streaming
    """
    # Streaming: ghép từng token vào response
    response = ""
    for token in chat_stream(message):
        response += token
        yield response


def create_ui() -> gr.ChatInterface:
    """Tạo Gradio ChatInterface (Gradio 6.x API)."""

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
    demo = create_ui()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()

