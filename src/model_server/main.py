"""FastAPI model server for BGE-M3 and bge-reranker-v2-m3.

Run with ``make run-model-server``.  The process is intentionally single
worker: HTTP requests are concurrent, while the shared dynamic batchers and
MPS arbiter keep model execution bounded.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from core.config import settings
from model_server.batcher import BatchLimits, DynamicBatcher
from model_server.embedding import EmbeddingRunner
from model_server.errors import ModelAuthenticationError, ModelContractMismatchError, ModelServerError
from model_server.reranker import RerankerRunner
from model_server.runtime import MPSExecutionArbiter, TorchRuntime
from model_server.schemas import EmbedRequest, EmbedResponse, ModelMetadata, RerankRequest, RerankResponse


def _api_key_is_valid(value: str | None) -> bool:
    configured = settings.MODEL_SERVER_API_KEY
    return not configured or value == configured


def _error_response(error: ModelServerError) -> JSONResponse:
    headers: dict[str, str] = {}
    if error.retry_after is not None:
        headers["Retry-After"] = str(error.retry_after)
    if error.status_code == 401:
        headers["WWW-Authenticate"] = "ApiKey"
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.error_code, "message": error.message}},
        headers=headers,
    )


def _limits(
    *,
    max_items: int,
    max_tokens: int,
    max_items_per_request: int,
    max_tokens_per_item: int,
) -> BatchLimits:
    return BatchLimits(
        max_items=max_items,
        max_tokens=max_tokens,
        max_bytes=settings.MODEL_SERVER_MAX_PENDING_BYTES,
        max_pending_requests=settings.MODEL_SERVER_MAX_PENDING_REQUESTS,
        max_pending_items=settings.MODEL_SERVER_MAX_PENDING_ITEMS,
        max_pending_bytes=settings.MODEL_SERVER_MAX_PENDING_BYTES,
        max_wait_ms=settings.MODEL_SERVER_BATCH_WAIT_MS,
        queue_timeout_seconds=settings.MODEL_SERVER_QUEUE_TIMEOUT_SECONDS,
        max_items_per_request=max_items_per_request,
        max_tokens_per_item=max_tokens_per_item,
        online_max_burst_batches=settings.MODEL_SERVER_ONLINE_MAX_BURST_BATCHES,
        max_items_per_request_per_batch=settings.MODEL_SERVER_MAX_ITEMS_PER_REQUEST_PER_BATCH,
    )


def create_app(
    *,
    embedding_runner: EmbeddingRunner | Any | None = None,
    reranker_runner: RerankerRunner | Any | None = None,
    runtime: TorchRuntime | Any | None = None,
    start_models: bool = True,
) -> FastAPI:
    """Create an app; dependency injection keeps contract tests weight-free."""

    runtime = runtime or TorchRuntime()
    arbiter = MPSExecutionArbiter(settings.MODEL_SERVER_MPS_MAX_INFLIGHT_BATCHES)
    embedding_runner = embedding_runner or EmbeddingRunner(runtime=runtime, arbiter=arbiter)
    reranker_runner = reranker_runner or RerankerRunner(runtime=runtime, arbiter=arbiter)

    async def startup() -> None:
        if not start_models:
            app.state.ready = True
        else:
            await asyncio.to_thread(runtime.check)
            await asyncio.to_thread(embedding_runner.load)
            await asyncio.to_thread(reranker_runner.load)
            warmup_embedding = getattr(embedding_runner, "warmup", None)
            warmup_reranker = getattr(reranker_runner, "warmup", None)
            if warmup_embedding is not None:
                await asyncio.to_thread(warmup_embedding)
            if warmup_reranker is not None:
                await asyncio.to_thread(warmup_reranker)
            app.state.ready = True
        app.state.embedding_batcher = DynamicBatcher(
            "embedding",
            embedding_runner.infer,
            _limits(
                max_items=settings.EMBEDDING_DYNAMIC_BATCH_MAX_ITEMS,
                max_tokens=settings.EMBEDDING_DYNAMIC_BATCH_MAX_TOKENS,
                max_items_per_request=settings.EMBEDDING_DYNAMIC_BATCH_MAX_ITEMS,
                max_tokens_per_item=settings.EMBEDDING_MAX_INPUT_TOKENS,
            ),
        )
        app.state.reranker_batcher = DynamicBatcher(
            "reranker",
            lambda pairs: _rerank_infer(reranker_runner, pairs),
            _limits(
                max_items=settings.RERANKER_DYNAMIC_BATCH_MAX_ITEMS,
                max_tokens=settings.RERANKER_DYNAMIC_BATCH_MAX_TOKENS,
                max_items_per_request=settings.RERANKER_DYNAMIC_BATCH_MAX_ITEMS,
                max_tokens_per_item=settings.RERANKER_MAX_INPUT_TOKENS,
            ),
        )
        await app.state.embedding_batcher.start()
        await app.state.reranker_batcher.start()

    async def shutdown() -> None:
        for name in ("embedding_batcher", "reranker_batcher"):
            batcher = getattr(app.state, name, None)
            if batcher is not None:
                await batcher.stop()
        app.state.ready = False

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            await startup()
            yield
        finally:
            await shutdown()

    app = FastAPI(
        title="RAG BGE Model Server",
        version="17.0.0",
        description="Bounded dynamic batching for BGE-M3 and bge-reranker-v2-m3",
        lifespan=lifespan,
    )
    app.state.ready = False
    app.state.embedding_runner = embedding_runner
    app.state.reranker_runner = reranker_runner
    app.state.runtime = runtime

    @app.middleware("http")
    async def enforce_body_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.MODEL_SERVER_MAX_BODY_BYTES:
                    from model_server.errors import ModelRequestTooLargeError

                    return _error_response(ModelRequestTooLargeError("Request body is too large"))
            except ValueError:
                pass
        return await call_next(request)

    @app.exception_handler(ModelServerError)
    async def model_error_handler(_request: Request, exc: ModelServerError):
        return _error_response(exc)

    def require_key(x_api_key: str | None) -> None:
        if not _api_key_is_valid(x_api_key):
            raise ModelAuthenticationError()

    def ensure_ready() -> None:
        if not app.state.ready:
            from model_server.errors import ModelNotReadyError

            raise ModelNotReadyError()

    def check_embed_contract(request: EmbedRequest) -> None:
        runner = app.state.embedding_runner
        if request.model is not None and request.model != runner.model_id:
            raise ModelContractMismatchError()
        if request.revision is not None and request.revision != runner.revision:
            raise ModelContractMismatchError()
        if request.normalize is not None and bool(request.normalize) != bool(runner.normalize):
            raise ModelContractMismatchError("Requested normalize setting does not match the loaded model")
        if len(request.inputs) > settings.EMBEDDING_DYNAMIC_BATCH_MAX_ITEMS:
            from model_server.errors import ModelRequestTooLargeError

            raise ModelRequestTooLargeError("Embedding request has too many inputs")
        if sum(len(value.encode("utf-8")) for value in request.inputs) > settings.MODEL_SERVER_MAX_BODY_BYTES:
            from model_server.errors import ModelRequestTooLargeError

            raise ModelRequestTooLargeError("Embedding request body exceeds the configured limit")

    def check_rerank_contract(request: RerankRequest) -> None:
        runner = app.state.reranker_runner
        if request.model is not None and request.model != runner.model_id:
            raise ModelContractMismatchError()
        if request.revision is not None and request.revision != runner.revision:
            raise ModelContractMismatchError()
        if len(request.documents) > settings.RERANKER_DYNAMIC_BATCH_MAX_ITEMS:
            from model_server.errors import ModelRequestTooLargeError

            raise ModelRequestTooLargeError("Rerank request has too many documents")
        if (
            len(request.query.encode("utf-8"))
            + sum(len(value.encode("utf-8")) for value in request.documents)
            > settings.MODEL_SERVER_MAX_BODY_BYTES
        ):
            from model_server.errors import ModelRequestTooLargeError

            raise ModelRequestTooLargeError("Rerank request body exceeds the configured limit")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if not app.state.ready:
            from model_server.errors import ModelNotReadyError

            raise ModelNotReadyError()
        return {
            "status": "ready",
            "device": getattr(runtime, "device", settings.MODEL_SERVER_DEVICE),
            "embedding": app.state.embedding_runner.metadata(),
            "reranker": app.state.reranker_runner.metadata(),
            "queues": {
                "embedding": app.state.embedding_batcher.stats(),
                "reranker": app.state.reranker_batcher.stats(),
            },
        }

    @app.get("/metadata", response_model=ModelMetadata)
    async def metadata(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        require_key(x_api_key)
        runner = app.state.embedding_runner
        reranker = app.state.reranker_runner
        return ModelMetadata(
            runtime="pytorch",
            device=getattr(runtime, "device", settings.MODEL_SERVER_DEVICE),
            embedding_model=runner.model_id,
            embedding_revision=runner.revision,
            embedding_dimension=runner.dimension,
            embedding_normalize=runner.normalize,
            embedding_max_input_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
            reranker_model=reranker.model_id,
            reranker_revision=reranker.revision,
            reranker_max_input_tokens=reranker.max_input_tokens,
            max_inflight_batches=settings.MODEL_SERVER_MPS_MAX_INFLIGHT_BATCHES,
        )

    @app.get("/metrics")
    async def metrics(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        require_key(x_api_key)
        return {
            "embedding": app.state.embedding_batcher.stats(),
            "reranker": app.state.reranker_batcher.stats(),
            "arbiter": getattr(getattr(app.state.embedding_runner, "arbiter", None), "stats", dict)(),
        }

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        require_key(x_api_key)
        ensure_ready()
        check_embed_contract(request)
        runner = app.state.embedding_runner
        costs = await asyncio.to_thread(runner.token_lengths, request.inputs)
        vectors = await app.state.embedding_batcher.submit(
            request.inputs,
            costs,
            priority=request.priority,
        )
        return EmbedResponse(
            embeddings=vectors,
            model=runner.model_id,
            revision=runner.revision,
            dimension=runner.dimension,
        )

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(request: RerankRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        require_key(x_api_key)
        ensure_ready()
        check_rerank_contract(request)
        runner = app.state.reranker_runner
        costs = await asyncio.to_thread(runner.token_lengths, request.query, request.documents)
        pairs = [(request.query, document) for document in request.documents]
        scores = await app.state.reranker_batcher.submit(
            pairs,
            costs,
            priority=request.priority,
        )
        return RerankResponse(scores=scores, model=runner.model_id, revision=runner.revision)

    return app


async def _rerank_infer(runner: RerankerRunner, pairs: list[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    query = pairs[0][0]
    if any(item[0] != query for item in pairs):
        # The reranker contract has one query per request.  Mixed-query batches
        # are still safe; execute each group in order and restore cardinality.
        outputs: list[float] = []
        start = 0
        while start < len(pairs):
            current_query = pairs[start][0]
            end = start + 1
            while end < len(pairs) and pairs[end][0] == current_query:
                end += 1
            outputs.extend(await runner.infer(current_query, [doc for _, doc in pairs[start:end]]))
            start = end
        return outputs
    return await runner.infer(query, [document for _, document in pairs])


app = create_app()
