from __future__ import annotations

"""
Chat Orchestrator — Kết nối RAGRetriever với API/UI layer.

Tách riêng khỏi retriever để:
- Quản lý conversation history (nếu cần)
- Format response cho API/UI
- Handle errors gracefully
"""

from core import get_logger
from core.errors import RAGChatbotError
from retrieval.retriever import RAGRetriever

logger = get_logger(__name__)

USER_FACING_ERROR = (
    "Xin lỗi, hệ thống đang gặp sự cố khi xử lý câu hỏi. "
    "Vui lòng thử lại sau ít phút."
)

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
    """Xử lý câu hỏi và trả về câu trả lời (non-streaming). Không raise.

    Args:
        query: Câu hỏi của user

    Returns:
        Câu trả lời từ RAG pipeline
    """
    if not query.strip():
        return "Please enter a question."

    try:
        return get_retriever().query(query, stream=False)
    except RAGChatbotError as e:
        logger.exception("Chat failed (known error)")
        return f"⚠️ {e}"          # lỗi mình tự raise → message đã an toàn, hữu ích
    except Exception:
        logger.exception("Chat failed (unexpected)")   # ★ full traceback vào log
        return USER_FACING_ERROR                        # ★ không leak ra ngoài


def chat_or_raise(query: str) -> str:
    """Bản cho API layer — raise để FastAPI trả HTTP status đúng."""
    if not query.strip():
        return "Please enter a question."
    return get_retriever().query(query, stream=False)


def chat_stream(query: str):
    """Xử lý câu hỏi và trả về câu trả lời (streaming — từng token). Không raise.

    Args:
        query: Câu hỏi của user

    Yields:
        Từng token của câu trả lời
    """
    if not query.strip():
        yield "Please enter a question."
        return

    try:
        yield from get_retriever().query(query, stream=True)
    except RAGChatbotError as e:
        logger.exception("Chat stream failed (known error)")
        yield f"⚠️ {e}"
    except Exception:
        logger.exception("Chat stream failed (unexpected)")
        yield USER_FACING_ERROR
