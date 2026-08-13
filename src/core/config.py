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

    # --- Embedding Model ---
    EMBEDDING_MODEL_ID: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_SIZE: int = 384
    EMBEDDING_DEVICE: str = "cpu"

    # --- Reranker Model ---
    RERANKER_MODEL_ID: str = "BAAI/bge-reranker-v2-m3"
    RERANK_CANDIDATES: int = 30    # Số candidate tối đa đưa vào cross-encoder
    RERANK_BATCH_SIZE: int = 16

    # --- API ---
    CORS_ORIGINS: list[str] = ["http://localhost:7860", "http://127.0.0.1:7860"]
    API_KEY: str = ""     # để trống = tắt auth (dev); set giá trị = bật auth

    # --- Logging ---
    LOG_LEVEL: str = "INFO"     # DEBUG / INFO / WARNING / ERROR
    LOG_JSON: bool = False      # True → JSON output (cho ELK/Datadog)

    # --- RAG Parameters ---
    TOP_K: int = 20          # Lấy bao nhiêu kết quả ban đầu (trước reranking)
    KEEP_TOP_K: int = 5      # Giữ lại bao nhiêu sau reranking
    EXPAND_N_QUERY: int = 3  # Tạo bao nhiêu biến thể câu hỏi
    MAX_CONTEXT_CHARS: int = 24_000  # ~6k token — an toàn cho model 8k+

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
    INGEST_PIPELINE_VERSION: str = "11.0"
    INGEST_SCHEMA_VERSION: str = "1"
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

    # --- Ingestion namespace ---
    # Dùng để sync đúng dataset, không đụng points của source directory khác.
    INGEST_DATASET_ID: str = "sample_docs"


# Singleton instance — import từ bất kỳ đâu đều dùng cùng 1 object
settings = Settings()
