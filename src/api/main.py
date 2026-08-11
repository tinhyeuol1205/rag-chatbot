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

import json
from secrets import compare_digest

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from api.chat import USER_FACING_ERROR, chat_or_raise_with_sources, chat_stream_events
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
    if x_api_key is None or not compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_api_key", "message": "Invalid API key"},
            headers={"WWW-Authenticate": "ApiKey"},
        )

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


class SourceResponse(BaseModel):
    citation_id: int
    file_name: str
    section_title: str | None = None
    page_number: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)


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
        result = chat_or_raise_with_sources(request.query)
        return ChatResponse(
            answer=result.answer,
            sources=[SourceResponse(**source.as_dict()) for source in result.sources],
        )
    except RAGChatbotError as exc:
        logger.exception("Known error in /chat", error_code=exc.error_code)
        raise HTTPException(
            status_code=503,
            detail={"code": exc.error_code, "message": exc.public_message},
        ) from exc
    except Exception:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail=USER_FACING_ERROR)


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream_endpoint(request: ChatRequest):
    """Chat streaming endpoint — trả về SSE (Server-Sent Events).

    Lưu ý: retrieval pipeline (multi-query + HyDE + search + rerank) chạy TRƯỚC
    token đầu tiên, nên TTFT ≈ thời gian retrieval (vài giây), không phải < 500ms.
    Event "status" được gửi ngay để client biết request đã được nhận.
    """
    async def generate():
        yield {
            "event": "status",
            "data": json.dumps(
                {"message": "Đang tìm kiếm tài liệu..."},
                ensure_ascii=False,
            ),
        }
        # iterate_in_threadpool: sync generator chạy trong thread, không block loop.
        async for item in iterate_in_threadpool(chat_stream_events(request.query)):
            # Backward-compatible với generator test/caller cũ chỉ yield string.
            if isinstance(item, str):
                event_name, data = "token", item
            else:
                event_name = str(item.get("event", "token"))
                data = item.get("data", "")

            if event_name == "token" and isinstance(data, str):
                data = {"text": data}
            elif event_name == "sources" and isinstance(data, list):
                data = {"sources": data}
            elif event_name == "end":
                data = {}
            elif isinstance(data, str):
                # Preserve custom event payloads while keeping the SSE contract JSON.
                data = {"message": data}
            yield {
                "event": event_name,
                "data": json.dumps(data, ensure_ascii=False),
            }
        yield {"event": "end", "data": "{}"}     # tín hiệu kết thúc cho client

    return EventSourceResponse(generate())
