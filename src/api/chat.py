"""
Chat Orchestrator — Kết nối RAGRetriever với API/UI layer.

Tách riêng khỏi retriever để:
- Quản lý conversation history (nếu cần)
- Format response cho API/UI
- Handle errors gracefully
"""

from __future__ import annotations

from threading import Lock

from core import get_logger
from core.errors import RAGChatbotError
from retrieval.retriever import RAGResult, RAGRetriever

logger = get_logger(__name__)

USER_FACING_ERROR = (
    "Xin lỗi, hệ thống đang gặp sự cố khi xử lý câu hỏi. "
    "Vui lòng thử lại sau ít phút."
)

# Singleton retriever — load models 1 lần, dùng cho mọi request.
# Lock + double-checked: nhiều request cold-start cùng lúc cũng chỉ tạo 1 instance
# (bug P2-14: check-then-create không lock → 2 thread cùng khởi tạo retriever/model).
_retriever: RAGRetriever | None = None
_retriever_lock = Lock()


def get_retriever() -> RAGRetriever:
    """Lazy init retriever (tránh load models khi import). Thread-safe."""
    global _retriever
    if _retriever is None:          # fast path — không cần lock
        with _retriever_lock:       # slow path — lấy lock
            if _retriever is None:  # double-check LẠI sau khi có lock
                logger.info("Initializing RAG Retriever...")
                instance = RAGRetriever()
                _retriever = instance   # publish chỉ sau init thành công
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
    except RAGChatbotError as exc:
        logger.exception("Chat failed (known error)", error_code=exc.error_code)
        return f"⚠️ {exc.public_message}"
    except Exception:
        logger.exception("Chat failed (unexpected)")   # ★ full traceback vào log
        return USER_FACING_ERROR                        # ★ không leak ra ngoài


def chat_or_raise(query: str, history: list[tuple[str, str]] | None = None) -> str:
    """Bản cho API layer — raise để FastAPI trả HTTP status đúng."""
    if not query.strip():
        return "Please enter a question."
    return get_retriever().query(query, stream=False, history=history)


def chat_or_raise_with_sources(
    query: str,
    history: list[tuple[str, str]] | None = None,
) -> RAGResult:
    """API variant trả answer cùng sources đã thực sự đưa vào prompt."""
    if not query.strip():
        return RAGResult(answer="Please enter a question.")
    return get_retriever().query_with_context(query, history=history)


def chat_stream(query: str, history: list | None = None):
    """Xử lý câu hỏi và trả về câu trả lời (streaming — từng token). Không raise.

    Args:
        query: Câu hỏi của user
        history: Lịch sử chat (Gradio format) cho multi-turn — condense trước khi retrieval

    Yields:
        Từng token của câu trả lời
    """
    if not query.strip():
        yield "Please enter a question."
        return

    try:
        yield from get_retriever().query(
            query, stream=True, history=_normalize_history(history or [])
        )
    except RAGChatbotError as exc:
        logger.exception("Chat stream failed (known error)", error_code=exc.error_code)
        # PR 7 sẽ chuẩn hóa event error riêng cho SSE. Ở PR 6 tối thiểu không
        # để message nội bộ của exception đi vào stream như token trả lời.
        yield f"⚠️ {exc.public_message}"
    except Exception:
        logger.exception("Chat stream failed (unexpected)")
        yield USER_FACING_ERROR


def chat_stream_events(query: str, history: list | None = None):
    """Stream token/source/error events cho API SSE và Gradio UI.

    ``chat_stream`` được giữ để tương thích với callers chỉ cần token. Event API
    mới giúp client phân biệt token, sources và lỗi thay vì trộn mọi thứ thành text.
    """
    if not query.strip():
        yield {"event": "token", "data": "Please enter a question."}
        yield {"event": "sources", "data": []}
        return

    try:
        yield from get_retriever().stream_with_sources(
            query,
            history=_normalize_history(history or []),
        )
    except RAGChatbotError as exc:
        logger.exception("Chat event stream failed (known error)", error_code=exc.error_code)
        yield {
            "event": "error",
            "data": {
                "code": exc.error_code,
                "message": exc.public_message,
            },
        }
    except Exception:
        logger.exception("Chat event stream failed (unexpected)")
        yield {
            "event": "error",
            "data": {
                "code": "internal_error",
                "message": USER_FACING_ERROR,
            },
        }


def _normalize_history(chat_history: list) -> list[tuple[str, str]]:
    """Chuẩn hoá format history của Gradio về list[(user, assistant)].

    Gradio 4.x: [[user, bot], ...] (tuples)
    Gradio 5.x+: [{'role', 'content'}, ...] (messages dict)
    """
    if not chat_history:
        return []
    if isinstance(chat_history[0], dict):
        pairs, pending = [], None
        for msg in chat_history:
            if msg.get("role") == "user":
                pending = msg.get("content", "")
            elif msg.get("role") == "assistant" and pending is not None:
                pairs.append((pending, msg.get("content", "")))
                pending = None
        return pairs
    return [(u, a) for u, a in chat_history if u and a]
