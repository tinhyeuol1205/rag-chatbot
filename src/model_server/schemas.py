"""Strict wire schemas for the embedding/reranker HTTP contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Priority = Literal["online", "batch"]


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[str] = Field(min_length=1)
    model: str | None = None
    revision: str | None = None
    normalize: bool | None = None
    priority: Priority = "online"

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("inputs must contain non-empty strings")
        return value


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    model: str | None = None
    revision: str | None = None
    priority: Priority = "online"

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must be non-empty")
        return value

    @field_validator("documents")
    @classmethod
    def _validate_documents(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("documents must contain non-empty strings")
        return value


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    revision: str
    dimension: int


class RerankResponse(BaseModel):
    scores: list[float]
    model: str
    revision: str


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str = "rag-model-server"
    runtime: str
    device: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    embedding_normalize: bool
    embedding_max_input_tokens: int
    reranker_model: str
    reranker_revision: str
    reranker_max_input_tokens: int
    max_inflight_batches: int
