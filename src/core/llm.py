from __future__ import annotations

"""
LLM Service — Unified interface cho nhiều LLM providers.

Hỗ trợ 2 providers:
  - "openai": OpenAI API / NVIDIA NIM / bất kỳ OpenAI-compatible API
  - "gemini": Google Gemini Interactions API (google-genai SDK)

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

from abc import ABC, abstractmethod

from core import get_logger
from core.config import settings

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
        pass

    @abstractmethod
    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
    ):
        """Gọi LLM và trả về generator (streaming)."""
        pass


class OpenAILLMService(BaseLLMService):
    """LLM provider dùng OpenAI-compatible API (OpenAI, NVIDIA NIM, etc.)."""

    def __init__(self):
        from openai import OpenAI

        kwargs = {"api_key": settings.OPENAI_API_KEY}
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

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GeminiLLMService(BaseLLMService):
    """LLM provider dùng Google Gemini Interactions API.

    Khác biệt so với OpenAI:
      - system_prompt → truyền vào system_instruction (không nằm trong messages)
      - user_prompt → truyền vào input
      - Response: interaction.outputs[-1].text

    Bonus: Multi-turn tự động bằng previous_interaction_id
    """

    def __init__(self):
        from google import genai

        kwargs = {}
        if settings.GEMINI_API_KEY:
            kwargs["api_key"] = settings.GEMINI_API_KEY

        self._client = genai.Client(**kwargs)
        self._model = settings.GEMINI_MODEL_ID
        logger.info("Gemini LLM initialized", model=self._model)

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
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt

        kwargs["generation_config"] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        interaction = self._client.interactions.create(**kwargs)
        return interaction.output_text

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
    ):
        """Gemini streaming qua Interactions API."""

        kwargs = {
            "model": self._model,
            "input": user_prompt,
            "stream": True,
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt

        kwargs["generation_config"] = {
            "temperature": temperature,
        }
        stream = self._client.interactions.create(**kwargs)
        for event in stream:
            if event.event_type == "step.delta":
                if event.delta.type == "text":
                    yield event.delta.text


# ================================================================
# Factory — chọn provider dựa trên LLM_PROVIDER config
# ================================================================

_llm_instance: BaseLLMService | None = None


def get_llm_service() -> BaseLLMService:
    """Singleton factory — tạo LLM service dựa trên LLM_PROVIDER config.

    Returns:
        OpenAILLMService nếu LLM_PROVIDER="openai"
        GeminiLLMService nếu LLM_PROVIDER="gemini"
    """
    global _llm_instance
    if _llm_instance is None:
        provider = settings.LLM_PROVIDER.lower()

        if provider == "gemini":
            _llm_instance = GeminiLLMService()
        elif provider == "openai":
            _llm_instance = OpenAILLMService()
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{provider}'. Use 'openai' or 'gemini'."
            )

        logger.info("LLM Service created", provider=provider)

    return _llm_instance
