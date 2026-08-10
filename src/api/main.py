"""
FastAPI Backend — REST API + SSE streaming.

Endpoints:
  POST /chat          — Chat (JSON response)
  POST /chat/stream   — Chat (SSE streaming response)
  GET  /health        — Health check

Chạy bằng: make run-api
Hoặc:      uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from api.chat import USER_FACING_ERROR, chat_or_raise, chat_stream
from core import get_logger
from core.config import settings
from core.errors import RAGChatbotError

logger = get_logger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="Internal Knowledge Base Assistant with Advanced RAG",
    version="0.1.0",
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Auth tối thiểu — bỏ qua nếu API_KEY rỗng (dev mode)."""
    if not settings.API_KEY:          # dev mode: bỏ qua
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")

# CORS — cho phép Gradio UI gọi API (chỉ origin cụ thể, không "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# --- Request/Response Models ---

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)   # ★ chặn query khổng lồ


class ChatResponse(BaseModel):
    answer: str


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat_endpoint(request: ChatRequest):
    """Chat endpoint — trả về JSON response đầy đủ.

    KHÔNG dùng async def: chat() là sync nặng (LLM + inference CPU).
    Để FastAPI đẩy vào threadpool thay vì block event loop (bug P0-3).
    """
    try:
        return ChatResponse(answer=chat_or_raise(request.query))
    except RAGChatbotError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail=USER_FACING_ERROR)


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Chat streaming endpoint — trả về SSE (Server-Sent Events).

    Lưu ý: retrieval pipeline (multi-query + HyDE + search + rerank) chạy TRƯỚC
    token đầu tiên, nên TTFT ≈ thời gian retrieval (vài giây), không phải < 500ms.
    Event "status" được gửi ngay để client biết request đã được nhận.
    """
    async def generate():
        yield {"event": "status", "data": "Đang tìm kiếm tài liệu..."}
        # iterate_in_threadpool: sync generator chạy trong thread, không block loop
        async for token in iterate_in_threadpool(chat_stream(request.query)):
            yield {"data": token}
        yield {"event": "end", "data": ""}     # tín hiệu kết thúc cho client

    return EventSourceResponse(generate())
