"""
Central configuration — Tất cả settings đọc từ file .env

Pattern: Pydantic Settings (giống llm-twin-course/src/core/config.py)
- Tự động đọc .env file
- Type validation (port phải là int, API key phải là str)
- Giá trị mặc định cho mọi biến

Usage:
    from core.config import settings
    print(settings.OPENAI_API_KEY)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.errors import ConfigurationError  # ★ dùng exception đang bị bỏ không

# Tìm thư mục gốc dự án (chứa .env file)
# __file__ = src/core/config.py → parent.parent.parent = rag-chatbot/
ROOT_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # Bỏ qua biến .env không khai báo ở đây
    )

    @model_validator(mode="after")
    def _check_provider_credentials(self):
        provider = self.LLM_PROVIDER.lower()
        if provider not in {"openai", "gemini"}:
            raise ConfigurationError(
                f"LLM_PROVIDER='{self.LLM_PROVIDER}' không hợp lệ. Dùng 'openai' hoặc 'gemini'."
            )
        # base_url tự host (vLLM) thường dùng key giả "EMPTY" → chấp nhận
        if provider == "openai" and not self.OPENAI_API_KEY and not self.OPENAI_BASE_URL:
            raise ConfigurationError("LLM_PROVIDER=openai nhưng thiếu OPENAI_API_KEY.")
        if provider == "gemini" and not self.GEMINI_API_KEY:
            raise ConfigurationError("LLM_PROVIDER=gemini nhưng thiếu GEMINI_API_KEY.")
        positive_limits = {
            "INGEST_EMBED_BATCH_SIZE": self.INGEST_EMBED_BATCH_SIZE,
            "INGEST_DOCUMENT_WINDOW": self.INGEST_DOCUMENT_WINDOW,
            "INGEST_QDRANT_WRITE_BATCH_SIZE": self.INGEST_QDRANT_WRITE_BATCH_SIZE,
            "INGEST_QDRANT_WRITE_MAX_BYTES": self.INGEST_QDRANT_WRITE_MAX_BYTES,
            "INGEST_QDRANT_MAX_RETRIES": self.INGEST_QDRANT_MAX_RETRIES,
            "INGEST_GENERATION_RETENTION": self.INGEST_GENERATION_RETENTION,
            "INGEST_MAX_MEMORY_MB": self.INGEST_MAX_MEMORY_MB,
            "RAG_QUEUE_MAX_OUTSTANDING": self.RAG_QUEUE_MAX_OUTSTANDING,
            "RAG_JOB_MAX_WAIT_SECONDS": self.RAG_JOB_MAX_WAIT_SECONDS,
            "RAG_JOB_TTL_SECONDS": self.RAG_JOB_TTL_SECONDS,
            "RAG_WORKER_CONCURRENCY": self.RAG_WORKER_CONCURRENCY,
            "LLM_RATE_LIMIT_CALLS": self.LLM_RATE_LIMIT_CALLS,
            "LLM_RATE_LIMIT_WINDOW_SECONDS": self.LLM_RATE_LIMIT_WINDOW_SECONDS,
            "LLM_HTTP_TIMEOUT_SECONDS": self.LLM_HTTP_TIMEOUT_SECONDS,
            "EMBEDDING_HTTP_TIMEOUT_SECONDS": self.EMBEDDING_HTTP_TIMEOUT_SECONDS,
            "RERANKER_HTTP_TIMEOUT_SECONDS": self.RERANKER_HTTP_TIMEOUT_SECONDS,
            "LLM_RESERVATION_TTL_SECONDS": self.LLM_RESERVATION_TTL_SECONDS,
            "QDRANT_QUERY_TIMEOUT_SECONDS": self.QDRANT_QUERY_TIMEOUT_SECONDS,
            "SSE_SEND_TIMEOUT_SECONDS": self.SSE_SEND_TIMEOUT_SECONDS,
            "LLM_MAX_OUTPUT_TOKENS": self.LLM_MAX_OUTPUT_TOKENS,
            "EMBEDDING_SIZE": self.EMBEDDING_SIZE,
            "QDRANT_HYBRID_PREFETCH_LIMIT": self.QDRANT_HYBRID_PREFETCH_LIMIT,
            "QDRANT_SPARSE_AVG_LEN": self.QDRANT_SPARSE_AVG_LEN,
            "MAX_CONTEXT_CHARS": self.MAX_CONTEXT_CHARS,
            "UI_CONNECT_TIMEOUT_SECONDS": self.UI_CONNECT_TIMEOUT_SECONDS,
            "UI_REQUEST_TIMEOUT_SECONDS": self.UI_REQUEST_TIMEOUT_SECONDS,
            "UI_SSE_IDLE_TIMEOUT_SECONDS": self.UI_SSE_IDLE_TIMEOUT_SECONDS,
            "UI_PORT": self.UI_PORT,
            "UI_MAX_INPUT_CHARS": self.UI_MAX_INPUT_CHARS,
            "UI_HISTORY_MAX_TURNS": self.UI_HISTORY_MAX_TURNS,
            "REDIS_JOB_EVENT_TTL_SECONDS": self.REDIS_JOB_EVENT_TTL_SECONDS,
            "RAG_WORKER_HEARTBEAT_TTL_SECONDS": self.RAG_WORKER_HEARTBEAT_TTL_SECONDS,
        }
        invalid = [name for name, value in positive_limits.items() if value <= 0]
        if invalid:
            raise ConfigurationError(f"Ingestion limits must be positive: {invalid}")
        retry_delays = {
            "INGEST_RETRY_BASE_SECONDS": self.INGEST_RETRY_BASE_SECONDS,
            "INGEST_RETRY_MAX_SECONDS": self.INGEST_RETRY_MAX_SECONDS,
        }
        invalid_delays = [name for name, value in retry_delays.items() if value <= 0]
        if self.INGEST_RETRY_MAX_SECONDS < self.INGEST_RETRY_BASE_SECONDS:
            invalid_delays.append("INGEST_RETRY_MAX_SECONDS")
        if invalid_delays:
            raise ConfigurationError(
                "Ingestion retry delays must be positive and max >= base: "
                f"{invalid_delays}"
            )
        ratios = {
            "INGEST_PARSER_MAX_EMPTY_PAGE_RATIO": self.INGEST_PARSER_MAX_EMPTY_PAGE_RATIO,
            "INGEST_PARSER_MAX_UNSUPPORTED_RATIO": self.INGEST_PARSER_MAX_UNSUPPORTED_RATIO,
            "INGEST_PARSER_MAX_REPLACEMENT_RATIO": self.INGEST_PARSER_MAX_REPLACEMENT_RATIO,
        }
        invalid_ratios = [name for name, value in ratios.items() if not 0 <= value <= 1]
        if invalid_ratios:
            raise ConfigurationError(f"Ingestion quality ratios must be between 0 and 1: {invalid_ratios}")
        if self.RAG_EXECUTION_MODE not in {"inline", "redis_worker"}:
            raise ConfigurationError("RAG_EXECUTION_MODE must be inline or redis_worker")
        if self.UI_DEMO_MODE not in {"dev", "product"}:
            raise ConfigurationError("UI_DEMO_MODE must be dev or product")
        if self.UI_PUBLIC_SHARE and not (self.UI_AUTH_USERNAME.strip() and self.UI_AUTH_PASSWORD):
            raise ConfigurationError(
                "UI_PUBLIC_SHARE requires both UI_AUTH_USERNAME and UI_AUTH_PASSWORD"
            )
        if self.UI_DEMO_MODE == "product":
            if self.RAG_EXECUTION_MODE != "redis_worker":
                raise ConfigurationError("Product UI requires RAG_EXECUTION_MODE=redis_worker")
            if not self.API_KEY.strip() or not self.UI_API_KEY.strip():
                raise ConfigurationError("Product UI requires API_KEY and UI_API_KEY")
        if self.REDIS_JOB_EVENT_TTL_SECONDS < self.RAG_JOB_MAX_WAIT_SECONDS:
            raise ConfigurationError(
                "REDIS_JOB_EVENT_TTL_SECONDS must cover RAG_JOB_MAX_WAIT_SECONDS"
            )
        if self.EMBEDDING_RUNTIME not in {"local", "remote"}:
            raise ConfigurationError("EMBEDDING_RUNTIME must be local or remote")
        if self.RERANKER_RUNTIME not in {"local", "remote"}:
            raise ConfigurationError("RERANKER_RUNTIME must be local or remote")
        if self.EMBEDDING_RUNTIME == "remote" and not self.EMBEDDING_BASE_URL.strip():
            raise ConfigurationError("EMBEDDING_RUNTIME=remote requires EMBEDDING_BASE_URL")
        if self.RERANKER_RUNTIME == "remote" and not self.RERANKER_BASE_URL.strip():
            raise ConfigurationError("RERANKER_RUNTIME=remote requires RERANKER_BASE_URL")
        if self.APP_ENV.lower() in {"production", "prod"}:
            if self.RAG_EXECUTION_MODE != "redis_worker":
                raise ConfigurationError("Production requires RAG_EXECUTION_MODE=redis_worker")
            if self.EMBEDDING_RUNTIME != "remote" or self.RERANKER_RUNTIME != "remote":
                raise ConfigurationError("Production requires remote GPU embedding and reranker services")
            if not self.EMBEDDING_MODEL_REVISION.strip() or not self.RERANKER_MODEL_REVISION.strip():
                raise ConfigurationError("Production requires immutable embedding/reranker model revisions")
            if self.EMBEDDING_MODEL_ID.lower() != "baai/bge-m3" or self.EMBEDDING_SIZE != 1024:
                raise ConfigurationError("Production PR14 requires BAAI/bge-m3 with EMBEDDING_SIZE=1024")
            if self.RERANKER_MODEL_ID.lower() != "baai/bge-reranker-v2-m3":
                raise ConfigurationError("Production PR14 requires BAAI/bge-reranker-v2-m3")
            if self.INGEST_SCHEMA_VERSION != "3":
                raise ConfigurationError("Production PR14 requires INGEST_SCHEMA_VERSION=3")
        if self.QDRANT_SPARSE_K <= 0 or not 0 <= self.QDRANT_SPARSE_B <= 1:
            raise ConfigurationError(
                "QDRANT_SPARSE_K must be positive and QDRANT_SPARSE_B must be between 0 and 1"
            )
        return self

    # --- LLM Provider (chọn "openai" hoặc "gemini") ---
    LLM_PROVIDER: str = "openai"

    # --- OpenAI / NVIDIA NIM ---
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = ""

    # --- Google Gemini ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_ID: str = "gemini-2.5-flash"

    # --- Qdrant ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # --- Embedding Model / GPU inference contract ---
    EMBEDDING_MODEL_ID: str = "BAAI/bge-m3"
    EMBEDDING_MODEL_REVISION: str = ""
    EMBEDDING_SIZE: int = 1024
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_RUNTIME: str = "local"  # remote in production; local is dev/test adapter
    EMBEDDING_BASE_URL: str = "http://embedding-gpu:8080"
    EMBEDDING_HTTP_TIMEOUT_SECONDS: float = 30.0
    EMBEDDING_NORMALIZE: bool = True

    # --- Reranker Model ---
    RERANKER_MODEL_ID: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MODEL_REVISION: str = ""
    RERANK_CANDIDATES: int = 30    # Số candidate tối đa đưa vào cross-encoder
    RERANK_BATCH_SIZE: int = 16
    RERANKER_DEVICE: str = ""  # empty -> reuse EMBEDDING_DEVICE in local mode
    RERANKER_RUNTIME: str = "local"  # remote in production; local is dev/test adapter
    RERANKER_BASE_URL: str = "http://reranker-gpu:8080"
    RERANKER_HTTP_TIMEOUT_SECONDS: float = 30.0

    # --- API ---
    CORS_ORIGINS: list[str] = ["http://localhost:7860", "http://127.0.0.1:7860"]
    API_KEY: str = ""     # để trống = tắt auth (dev); set giá trị = bật auth

    # --- Thin Gradio UI (PR15) ---
    # UI không kết nối trực tiếp tới retriever/Redis/Qdrant.  Nó chỉ gọi FastAPI.
    UI_DEMO_MODE: str = "dev"  # dev -> inline API; product -> Redis worker API
    UI_API_BASE_URL: str = "http://127.0.0.1:8080"
    UI_HOST: str = "127.0.0.1"
    UI_PORT: int = 7860
    UI_CONNECT_TIMEOUT_SECONDS: float = 3.0
    UI_REQUEST_TIMEOUT_SECONDS: float = 90.0
    UI_SSE_IDLE_TIMEOUT_SECONDS: float = 45.0
    UI_MAX_INPUT_CHARS: int = 2_000
    UI_HISTORY_MAX_TURNS: int = 3
    UI_API_KEY: str = ""  # service credential; không gửi xuống browser
    UI_AUTH_USERNAME: str = ""
    UI_AUTH_PASSWORD: str = ""
    UI_PUBLIC_SHARE: bool = False  # chỉ bật explicit với basic auth

    # --- Logging ---
    LOG_LEVEL: str = "INFO"     # DEBUG / INFO / WARNING / ERROR
    LOG_JSON: bool = False      # True → JSON output (cho ELK/Datadog)

    # --- RAG Parameters ---
    TOP_K: int = 20          # Lấy bao nhiêu kết quả ban đầu (trước reranking)
    KEEP_TOP_K: int = 5      # Giữ lại bao nhiêu sau reranking
    EXPAND_N_QUERY: int = 3  # Tạo bao nhiêu biến thể câu hỏi
    MAX_CONTEXT_CHARS: int = 24_000  # ~6k token — an toàn cho model 8k+

    # --- Redis admission queue / distributed provider quota ---
    APP_ENV: str = "development"
    RAG_EXECUTION_MODE: str = "inline"  # redis_worker is mandatory in production
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_KEY_PREFIX: str = "rag:default"
    REDIS_QUEUE_STREAM: str = "rag:default:jobs"
    REDIS_QUEUE_GROUP: str = "rag-workers"
    REDIS_OUTSTANDING_KEY: str = "rag:default:outstanding"
    REDIS_RATE_SCHEDULE_KEY: str = "rag:default:llm_schedule"
    REDIS_RATE_RESERVATION_PREFIX: str = "rag:default:reservation:"
    REDIS_JOB_PREFIX: str = "rag:default:job:"
    REDIS_IDEMPOTENCY_PREFIX: str = "rag:default:idempotency:"
    REDIS_JOB_EVENT_PREFIX: str = "rag:default:events:"
    REDIS_JOB_EVENT_TTL_SECONDS: int = 300
    REDIS_CONNECT_TIMEOUT_SECONDS: float = 2.0
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 2.0
    REDIS_RESULT_POLL_SECONDS: float = 0.1
    RAG_QUEUE_MAX_OUTSTANDING: int = 32
    RAG_JOB_MAX_WAIT_SECONDS: float = 180.0
    RAG_JOB_TTL_SECONDS: int = 600
    RAG_WORKER_CONCURRENCY: int = 2
    RAG_WORKER_LEASE_SECONDS: int = 300
    RAG_WORKER_HEARTBEAT_TTL_SECONDS: int = 10
    LLM_RATE_LIMIT_CALLS: int = 15
    LLM_RATE_LIMIT_WINDOW_SECONDS: float = 60.0
    LLM_RESERVATION_TTL_SECONDS: int = 600
    LLM_HTTP_TIMEOUT_SECONDS: float = 60.0
    QDRANT_QUERY_TIMEOUT_SECONDS: float = 8.0
    SSE_SEND_TIMEOUT_SECONDS: float = 15.0

    # Generation output cap remains a cost/latency control.  PR14 removes only
    # token-aware input-context accounting from the assembler.
    LLM_MAX_OUTPUT_TOKENS: int = 1_024

    # --- Chunking Parameters ---
    CHILD_CHUNK_SIZE: int = 400      # Chunk nhỏ (search chính xác)
    CHILD_CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 2000    # Chunk lớn (context đầy đủ cho LLM)
    PARENT_CHUNK_OVERLAP: int = 200

    # --- Collection Names (Qdrant) ---
    # Stable aliases used by retrieval.  Concrete generation collections are
    # created by the versioned ingestion pipeline and switched atomically to
    # these names.  The old names remain configurable for migration tooling.
    CHILD_COLLECTION: str = "child_chunks_active"
    PARENT_COLLECTION: str = "parent_chunks_active"
    LEGACY_CHILD_COLLECTION: str = "child_chunks"
    LEGACY_PARENT_COLLECTION: str = "parent_chunks"

    # --- Versioned ingestion ---
    INGEST_VERSIONED: bool = True
    INGEST_MANIFEST_PATH: str = "data/ingest_runs/manifest.sqlite3"
    INGEST_PIPELINE_VERSION: str = "14.0"
    INGEST_SCHEMA_VERSION: str = "3"
    INGEST_PARSER_VERSION: str = "2"
    INGEST_CHUNKER_VERSION: str = "2"
    INGEST_EMBEDDING_MODEL_REVISION: str = ""
    INGEST_EMBED_BATCH_SIZE: int = 32
    INGEST_DOCUMENT_WINDOW: int = 8
    INGEST_QDRANT_WRITE_BATCH_SIZE: int = 256
    INGEST_QDRANT_WRITE_MAX_BYTES: int = 4_000_000
    INGEST_QDRANT_MAX_RETRIES: int = 5
    INGEST_RETRY_BASE_SECONDS: float = 0.25
    INGEST_RETRY_MAX_SECONDS: float = 8.0
    INGEST_GENERATION_RETENTION: int = 2
    INGEST_MAX_MEMORY_MB: int = 2048
    INGEST_PARSER_MAX_EMPTY_PAGE_RATIO: float = 0.5
    INGEST_PARSER_MAX_UNSUPPORTED_RATIO: float = 0.5
    INGEST_PARSER_MIN_TEXT_CHARS: int = 8
    INGEST_PARSER_MAX_REPLACEMENT_RATIO: float = 0.02
    INGEST_FAIL_ON_QUALITY: bool = True
    INGEST_PDF_FAST_STRATEGY: str = "fast"
    INGEST_PDF_OCR_STRATEGY: str = "hi_res"

    # --- Qdrant native BM25 / server-side hybrid retrieval ---
    QDRANT_SPARSE_VECTOR_NAME: str = "bm25"
    QDRANT_SPARSE_MODEL: str = "Qdrant/bm25"
    QDRANT_SPARSE_TOKENIZER: str = "multilingual"
    QDRANT_SPARSE_LANGUAGE: str = "none"
    QDRANT_SPARSE_K: float = 1.2
    QDRANT_SPARSE_B: float = 0.75
    QDRANT_SPARSE_AVG_LEN: int = 256
    QDRANT_SPARSE_ON_DISK: bool = True
    QDRANT_HYBRID_PREFETCH_LIMIT: int = 40

    # --- Ingestion namespace ---
    # Dùng để sync đúng dataset, không đụng points của source directory khác.
    INGEST_DATASET_ID: str = "sample_docs"


# Singleton instance — import từ bất kỳ đâu đều dùng cùng 1 object
settings = Settings()
