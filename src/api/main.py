"""
FastAPI Backend — REST API + SSE streaming.

Endpoints:
  POST /chat          — Chat (JSON response)
  POST /chat/stream   — Chat (SSE streaming response)
  GET  /health        — Health check

Chạy bằng: make run-api (host 127.0.0.1:8080)
Hoặc:      uvicorn api.main:app --host 127.0.0.1 --port 8080 --reload
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.models import Distance
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.chat import (
    USER_FACING_ERROR,
    chat_or_raise_with_sources,
    chat_stream_events,
    get_retriever,
)
from core import get_logger
from core.admission_queue import InlineAdmissionQueue, get_admission_queue
from core.config import settings
from core.db import QdrantConnector
from core.errors import JobTimeoutError, QueueFullError, QueueUnavailableError, RAGChatbotError

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warmup models trước khi nhận request — không block event loop.

    get_retriever() là sync + nặng (load embedding/reranker model) → chạy trong
    threadpool. KHÔNG gọi external paid LLM ở startup (bug P2-14).
    """
    retriever = await run_in_threadpool(get_retriever)
    await run_in_threadpool(retriever.warmup)
    logger.info("Application ready")
    try:
        yield
    finally:
        QdrantConnector().close()


app = FastAPI(
    title="RAG Chatbot API",
    description="Internal Knowledge Base Assistant with Advanced RAG",
    version="0.1.0",
    lifespan=lifespan,
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
    allow_headers=["Content-Type", "X-API-Key", "Idempotency-Key"],
)


# --- Request/Response Models ---

class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)   # ★ chặn query khổng lồ
    history: list[HistoryMessage] = Field(default_factory=list, max_length=16)


class SourceResponse(BaseModel):
    citation_id: int
    file_name: str
    section_title: str | None = None
    page_number: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)


def _run_chat_job(query: str, history: list[tuple[str, str]]) -> object:
    try:
        return chat_or_raise_with_sources(query, history=history)
    except TypeError as exc:
        # Keep small test/integration adapters written against the PR11
        # one-argument contract working without swallowing real TypeErrors
        # raised inside the chat implementation.
        if "unexpected keyword argument 'history'" not in str(exc):
            raise
        return chat_or_raise_with_sources(query)


def _run_stream_job(query: str, history: list[tuple[str, str]]) -> list:
    try:
        return list(chat_stream_events(query, history=history))
    except TypeError as exc:
        if "unexpected keyword argument 'history'" not in str(exc):
            raise
        return list(chat_stream_events(query))


def _result_from_payload(payload: object):
    """Restore a worker JSON result without coupling Redis to API models."""
    from retrieval.context.assembler import SourceRef
    from retrieval.retriever import RAGResult

    if isinstance(payload, RAGResult):
        return payload
    data = payload if isinstance(payload, dict) else {}
    return RAGResult(
        answer=str(data.get("answer", "")),
        contexts=list(data.get("contexts", []) or []),
        sources=[SourceRef(**source) for source in data.get("sources", []) or []],
        expanded_queries=list(data.get("expanded_queries", []) or []),
        num_candidates=int(data.get("num_candidates", 0)),
    )


def _history_pairs(history: list[HistoryMessage]) -> list[tuple[str, str]]:
    """Convert typed API messages into the retriever's bounded turn format."""
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for message in history[-settings.UI_HISTORY_MAX_TURNS * 2 :]:
        if message.role == "user":
            pending_user = message.content.strip()
        elif pending_user:
            pairs.append((pending_user, message.content.strip()))
            pending_user = None
    return pairs


async def _submit_chat(
    query: str,
    history: list[tuple[str, str]] | None = None,
    idempotency_key: str | None = None,
):
    queue = get_admission_queue()
    history = history or []
    if isinstance(queue, InlineAdmissionQueue):
        payload = await asyncio.to_thread(queue.execute, _run_chat_job, query, history)
    else:
        if idempotency_key is None:
            job = await asyncio.to_thread(queue.enqueue, query, history)
        else:
            job = await asyncio.to_thread(
                queue.enqueue,
                query,
                history,
                idempotency_key=idempotency_key,
            )
        try:
            payload = await asyncio.to_thread(queue.wait_result, job.job_id, settings.RAG_JOB_MAX_WAIT_SECONDS)
        except JobTimeoutError:
            # If the worker has not started yet this atomically releases the
            # outstanding slot and prevents a late GPU/LLM charge.  A running
            # job only receives a cancellation request and is not retried.
            await asyncio.to_thread(queue.cancel, job.job_id)
            raise
        except asyncio.CancelledError:
            await asyncio.to_thread(queue.cancel, job.job_id)
            raise
    return _result_from_payload(payload)


def _normalise_event(item):
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
        data = {"message": data}
    return event_name, data


def _result_events(result) -> list[dict]:
    return [
        {"event": "token", "data": result.answer},
        {"event": "sources", "data": [source.as_dict() for source in result.sources]},
        {"event": "end", "data": {}},
    ]


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    """Readiness for Redis admission, GPU services and the active Qdrant schema."""
    worker_available = False
    try:
        if settings.RAG_EXECUTION_MODE == "redis_worker":
            queue = get_admission_queue()
            queue.client.ping()
            worker_available = queue.worker_available()
            if not worker_available:
                raise ValueError("RAG worker heartbeat is missing")
        connector = QdrantConnector()
        child = connector.alias_target(settings.CHILD_COLLECTION)
        parent = connector.alias_target(settings.PARENT_COLLECTION)
        if not child or not parent:
            raise ValueError("active aliases are missing")
        connector.validate_collection_schema(
            child,
            schema_fingerprint=None,
            vector_dimension=settings.EMBEDDING_SIZE,
            expected_distance=Distance.COSINE,
            require_sparse=True,
        )
        if settings.EMBEDDING_RUNTIME == "remote":
            response = await asyncio.to_thread(
                httpx.get,
                settings.EMBEDDING_BASE_URL.rstrip("/") + "/health",
                timeout=settings.EMBEDDING_HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        if settings.RERANKER_RUNTIME == "remote":
            response = await asyncio.to_thread(
                httpx.get,
                settings.RERANKER_BASE_URL.rstrip("/") + "/health",
                timeout=settings.RERANKER_HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
    except Exception as exc:
        logger.warning("Readiness check failed", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={"code": "not_ready", "message": "Hệ thống chưa sẵn sàng phục vụ"},
        ) from exc
    return {
        "status": "ready",
        "app_env": settings.APP_ENV,
        "execution_mode": settings.RAG_EXECUTION_MODE,
        "embedding_runtime": settings.EMBEDDING_RUNTIME,
        "reranker_runtime": settings.RERANKER_RUNTIME,
        "worker_available": worker_available,
        "child_collection": child,
        "parent_collection": parent,
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat_endpoint(
    request: ChatRequest,
    idempotency_key: str | None = Header(default=None, max_length=128, alias="Idempotency-Key"),
):
    """Chat endpoint — trả về JSON response đầy đủ.

    Chỉ enqueue/wait ở API; worker mới chạy RAG sau Redis admission.
    """
    try:
        result = await _submit_chat(
            request.query,
            _history_pairs(request.history),
            idempotency_key=idempotency_key,
        )
        return ChatResponse(
            answer=result.answer,
            sources=[SourceResponse(**source.as_dict()) for source in result.sources],
        )
    except QueueFullError as exc:
        raise HTTPException(
            status_code=429,
            detail={"code": exc.error_code, "message": exc.public_message},
            headers={"Retry-After": "1"},
        ) from exc
    except QueueUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": exc.error_code, "message": exc.public_message},
        ) from exc
    except JobTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={"code": exc.error_code, "message": exc.public_message},
        ) from exc
    except RAGChatbotError as exc:
        logger.exception("Known error in /chat", error_code=exc.error_code)
        raise HTTPException(
            status_code=503,
            detail={"code": exc.error_code, "message": exc.public_message},
        ) from exc
    except (ConnectionError, TimeoutError) as exc:
        logger.exception("Inference or provider transport unavailable")
        raise HTTPException(
            status_code=503,
            detail={"code": "upstream_unavailable", "message": USER_FACING_ERROR},
        ) from exc
    except Exception:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail=USER_FACING_ERROR)


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream_endpoint(
    request: ChatRequest,
    idempotency_key: str | None = Header(default=None, max_length=128, alias="Idempotency-Key"),
):
    """Chat streaming endpoint — trả về SSE (Server-Sent Events).

    Lưu ý: retrieval pipeline (multi-query + HyDE + search + rerank) chạy TRƯỚC
    token đầu tiên, nên TTFT ≈ thời gian retrieval (vài giây), không phải < 500ms.
    Event "status" được gửi ngay để client biết request đã được nhận.
    """
    queue = get_admission_queue()
    pending_job = None
    history = _history_pairs(request.history)
    if not isinstance(queue, InlineAdmissionQueue):
        try:
            if idempotency_key is None:
                pending_job = await asyncio.to_thread(queue.enqueue, request.query, history)
            else:
                pending_job = await asyncio.to_thread(
                    queue.enqueue,
                    request.query,
                    history,
                    idempotency_key=idempotency_key,
                )
        except QueueFullError as exc:
            raise HTTPException(
                status_code=429,
                detail={"code": exc.error_code, "message": exc.public_message},
                headers={"Retry-After": "1"},
            ) from exc
        except QueueUnavailableError as exc:
            raise HTTPException(status_code=503, detail={"code": exc.error_code, "message": exc.public_message}) from exc

    async def generate():
        try:
            yield {
                "event": "status",
                "data": json.dumps(
                    {"message": "Đang tìm kiếm tài liệu..."},
                    ensure_ascii=False,
                ),
            }
            if isinstance(queue, InlineAdmissionQueue):
                events = await asyncio.to_thread(
                    queue.execute,
                    _run_stream_job,
                    request.query,
                    history,
                )
                for item in events:
                    event_name, data = _normalise_event(item)
                    yield {
                        "event": event_name,
                        "data": json.dumps(data, ensure_ascii=False),
                    }
            else:
                events = []
                last_id = "0-0"
                deadline = time.monotonic() + settings.RAG_JOB_MAX_WAIT_SECONDS
                terminal = False
                while time.monotonic() < deadline and not terminal:
                    new_events, state = await asyncio.to_thread(
                        queue.read_events,
                        pending_job.job_id,
                        last_id=last_id,
                        block_ms=min(500, max(50, int((deadline - time.monotonic()) * 1000))),
                    )
                    for item in new_events:
                        events.append(item)
                        last_id = str(item.get("id", last_id))
                        event_name, data = _normalise_event(item)
                        payload = {
                            "event": event_name,
                            "data": json.dumps(data, ensure_ascii=False),
                        }
                        if item.get("id"):
                            payload["id"] = str(item["id"])
                        yield payload
                        if event_name == "end":
                            terminal = True
                            break
                    if terminal:
                        break
                    if state in {"failed", "cancelled"} and not new_events:
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {
                                    "code": "job_failed" if state == "failed" else "job_timeout",
                                    "message": USER_FACING_ERROR,
                                },
                                ensure_ascii=False,
                            ),
                        }
                        terminal_event = {"event": "end", "data": {}}
                        events.append(terminal_event)
                        yield {"event": "end", "data": "{}"}
                        terminal = True
                if not terminal:
                    raise JobTimeoutError("RAG job result wait timed out")
            last_event = events[-1] if events else None
            if not isinstance(last_event, dict) or _normalise_event(last_event)[0] != "end":
                yield {"event": "end", "data": "{}"}
        except RAGChatbotError as exc:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"code": exc.error_code, "message": exc.public_message},
                    ensure_ascii=False,
                ),
            }
            yield {"event": "end", "data": "{}"}
        except Exception:
            logger.exception("Unhandled error in /chat/stream")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"code": "internal_error", "message": USER_FACING_ERROR},
                    ensure_ascii=False,
                ),
            }
            yield {"event": "end", "data": "{}"}
        finally:
            # Closing a client-side SSE generator before the first worker poll
            # must not leave an outstanding Redis job behind.
            if pending_job is not None:
                try:
                    await asyncio.to_thread(queue.cancel, pending_job.job_id)
                except Exception:  # noqa: BLE001 - cleanup must not mask disconnect
                    logger.warning("Failed to cancel disconnected RAG job")

    return EventSourceResponse(generate(), send_timeout=settings.SSE_SEND_TIMEOUT_SECONDS)
