from __future__ import annotations

"""
FastAPI Backend — REST API + SSE streaming.

Endpoints:
  POST /chat          — Chat (JSON response)
  POST /chat/stream   — Chat (SSE streaming response)
  GET  /health        — Health check

Chạy bằng: make run-api
Hoặc:      uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.chat import chat, chat_stream

app = FastAPI(
    title="RAG Chatbot API",
    description="Internal Knowledge Base Assistant with Advanced RAG",
    version="0.1.0",
)

# CORS — cho phép Gradio UI gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint — trả về JSON response đầy đủ."""
    answer = chat(request.query)
    return ChatResponse(answer=answer)


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Chat streaming endpoint — trả về SSE (Server-Sent Events).

    Client nhận từng token realtime, giảm thời gian chờ đợi.
    Time to First Token (TTFT) thường < 500ms.
    """
    async def generate():
        for token in chat_stream(request.query):
            yield {"data": token}

    return EventSourceResponse(generate())
