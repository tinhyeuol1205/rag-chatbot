from __future__ import annotations

"""
Chat Orchestrator — Kết nối RAGRetriever với API/UI layer.

Tách riêng khỏi retriever để:
- Quản lý conversation history (nếu cần)
- Format response cho API/UI
- Handle errors gracefully
"""

from core import get_logger
from retrieval.retriever import RAGRetriever

logger = get_logger(__name__)

# Singleton retriever — load models 1 lần, dùng cho mọi request
_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    """Lazy init retriever (tránh load models khi import)."""
    global _retriever
    if _retriever is None:
        logger.info("Initializing RAG Retriever...")
        _retriever = RAGRetriever()
        logger.info("RAG Retriever ready")
    return _retriever


def chat(query: str) -> str:
    """Xử lý câu hỏi và trả về câu trả lời (non-streaming).

    Args:
        query: Câu hỏi của user

    Returns:
        Câu trả lời từ RAG pipeline
    """
    if not query.strip():
        return "Please enter a question."

    try:
        retriever = get_retriever()
        answer = retriever.query(query, stream=False)
        return answer
    except Exception as e:
        logger.error("Chat error", error=str(e))
        return f"❌ Error: {str(e)}"


def chat_stream(query: str):
    """Xử lý câu hỏi và trả về câu trả lời (streaming — từng token).

    Args:
        query: Câu hỏi của user

    Yields:
        Từng token của câu trả lời
    """
    if not query.strip():
        yield "Please enter a question."
        return

    try:
        retriever = get_retriever()
        yield from retriever.query(query, stream=True)
    except Exception as e:
        logger.error("Chat stream error", error=str(e))
        yield f"❌ Error: {str(e)}"
