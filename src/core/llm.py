"""
LLM Service — Unified interface cho nhiều LLM providers.

Hỗ trợ 2 providers:
  - "openai": OpenAI API / NVIDIA NIM / bất kỳ OpenAI-compatible API
  - "gemini": Google Gemini Interactions API (google-genai SDK 2.x)

Chọn provider bằng biến LLM_PROVIDER trong .env:
  LLM_PROVIDER=openai   → dùng OpenAI client
  LLM_PROVIDER=gemini   → dùng Gemini Interactions API

Design Pattern: Strategy — cùng interface, khác implementation.

Usage:
    from core.llm import get_llm_service
    llm = get_llm_service()
    answer = llm.generate("What is the policy?", system="You are a helpful assistant.")
    for token in llm.generate_stream("Hello", system="You are helpful."):
        print(token, end="")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Lock

from core import get_logger
from core.admission_queue import consume_llm_permit
from core.config import settings
from core.errors import ConfigurationError

logger = get_logger(__name__)


class BaseLLMService(ABC):
    """Abstract base — mọi LLM provider phải implement generate()."""

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Gọi LLM và trả về response text."""

    @abstractmethod
    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        """Gọi LLM và trả về generator (streaming)."""


class OpenAILLMService(BaseLLMService):
    """LLM provider dùng OpenAI-compatible API (OpenAI, NVIDIA NIM, etc.)."""

    def __init__(self):
        from openai import OpenAI

        kwargs = {"api_key": settings.OPENAI_API_KEY, "max_retries": 0, "timeout": settings.LLM_HTTP_TIMEOUT_SECONDS}
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL

        self._client = OpenAI(**kwargs)
        self._model = settings.OPENAI_MODEL_ID
        logger.info("OpenAI LLM initialized", model=self._model)

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        consume_llm_permit()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response.choices:
            logger.warning("LLM returned no choices")
            return ""
        choice = response.choices[0]
        content = choice.message.content
        if content is None:
            # finish_reason == "content_filter" / tool_call-only / reasoning model
            # chỉ điền reasoning_content → content None
            logger.warning("LLM returned empty content",
                           finish_reason=choice.finish_reason)
            return ""
        return content.strip()

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        consume_llm_permit()
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        try:
            for chunk in stream:
                if not chunk.choices:  # usage-only chunk → bỏ qua (vLLM/NIM/Azure)
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()


class GeminiLLMService(BaseLLMService):
    """LLM provider dùng Google Gemini Interactions API (google-genai 2.x).

    Verified chạy thật với SDK 2.15.0 + key (xem review/standalone.md P0-2).

    Khác biệt so với OpenAI:
      - system_prompt → system_instruction (top-level kwarg, KHÔNG nằm trong messages)
      - user_prompt → input
      - Response: interaction.output_text
      - Config (temperature, max_output_tokens) → generation_config (dict)
    """

    def __init__(self):
        from google import genai
        from google.genai import types

        kwargs = {}
        if settings.GEMINI_API_KEY:
            kwargs["api_key"] = settings.GEMINI_API_KEY

        kwargs["http_options"] = types.HttpOptions(
            timeout=int(settings.LLM_HTTP_TIMEOUT_SECONDS * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        self._client = genai.Client(**kwargs)
        self._model = settings.GEMINI_MODEL_ID
        logger.info("Gemini LLM initialized", model=self._model)

    def _build_config(self, temperature: float, max_tokens: int) -> dict:
        return {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        kwargs = {
            "model": self._model,
            "input": user_prompt,
            "generation_config": self._build_config(temperature, max_tokens),
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt

        consume_llm_permit()
        interaction = self._client.interactions.create(**kwargs)
        return (interaction.output_text or "").strip()

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        """Gemini streaming qua Interactions API.

        Lưu ý: event 'step.delta' có thể là 'thought_signature' (thinking) — KHÔNG có text,
        phải lọc bằng `delta.type == 'text'` (code hiện tại đã làm đúng).
        """
        kwargs = {
            "model": self._model,
            "input": user_prompt,
            "stream": True,
            "generation_config": self._build_config(temperature, max_tokens),  # ★ THÊM max_tokens
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt

        consume_llm_permit()
        stream = self._client.interactions.create(**kwargs)
        try:
            for event in stream:
                if (
                    event.event_type == "step.delta"
                    and event.delta.type == "text"
                    and event.delta.text
                ):
                    yield event.delta.text
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()


# ================================================================
# Factory — chọn provider dựa trên LLM_PROVIDER config
# ================================================================

_llm_instance: BaseLLMService | None = None
_llm_lock = Lock()


def get_llm_service() -> BaseLLMService:
    """Singleton factory — tạo LLM service dựa trên LLM_PROVIDER config.

    Thread-safe (lock + double-check): cold-start đồng thời chỉ tạo 1 client
    (bug P2-14), tránh mở nhiều connection/API client giống nhau.

    Returns:
        OpenAILLMService nếu LLM_PROVIDER="openai"
        GeminiLLMService nếu LLM_PROVIDER="gemini"
    """
    global _llm_instance
    if _llm_instance is None:               # fast path — không cần lock
        with _llm_lock:                     # slow path — lấy lock
            if _llm_instance is None:       # double-check LẠI sau khi có lock
                provider = settings.LLM_PROVIDER.lower()

                if provider == "gemini":
                    _llm_instance = GeminiLLMService()
                elif provider == "openai":
                    _llm_instance = OpenAILLMService()
                else:
                    raise ConfigurationError(
                        f"LLM_PROVIDER='{settings.LLM_PROVIDER}' không hợp lệ. "
                        f"Dùng 'openai' hoặc 'gemini'."
                    )

                logger.info("LLM Service created", provider=provider)

    return _llm_instance
