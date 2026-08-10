from __future__ import annotations

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

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.errors import ConfigurationError     # ★ dùng exception đang bị bỏ không

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
    CHILD_COLLECTION: str = "child_chunks"
    PARENT_COLLECTION: str = "parent_chunks"


# Singleton instance — import từ bất kỳ đâu đều dùng cùng 1 object
settings = Settings()
