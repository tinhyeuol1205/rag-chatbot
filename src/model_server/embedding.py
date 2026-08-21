"""BGE-M3 model adapter used by the dynamic embedding batcher."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.config import settings
from model_server.errors import ModelInferenceError, ModelNotReadyError
from model_server.runtime import MPSExecutionArbiter, TorchRuntime


class EmbeddingRunner:
    """One lazily loaded SentenceTransformer instance per server process."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        revision: str | None = None,
        dimension: int | None = None,
        normalize: bool | None = None,
        runtime: TorchRuntime | None = None,
        arbiter: MPSExecutionArbiter | None = None,
    ):
        self.model_id = model_id or settings.EMBEDDING_MODEL_ID
        self.revision = revision if revision is not None else (
            settings.EMBEDDING_MODEL_REVISION or settings.INGEST_EMBEDDING_MODEL_REVISION
        )
        self.dimension = int(dimension or settings.EMBEDDING_SIZE)
        self.normalize = settings.EMBEDDING_NORMALIZE if normalize is None else bool(normalize)
        self.runtime = runtime or TorchRuntime()
        self.arbiter = arbiter or MPSExecutionArbiter(settings.MODEL_SERVER_MPS_MAX_INFLIGHT_BATCHES)
        self.model: Any = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        device = self.runtime.check()
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"device": device}
            if self.revision:
                kwargs["revision"] = self.revision
            model = SentenceTransformer(self.model_id, **kwargs)
            model.eval()
            actual_dimension = model.get_sentence_embedding_dimension()
            if actual_dimension is not None and int(actual_dimension) != self.dimension:
                raise ModelNotReadyError(
                    f"Embedding dimension mismatch: model={actual_dimension}, configured={self.dimension}"
                )
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "model_max_length"):
                # Do not let a model's huge sentinel value defeat the server
                # admission cap; this only changes truncation behaviour.
                tokenizer.model_max_length = min(
                    int(getattr(tokenizer, "model_max_length", settings.EMBEDDING_MAX_INPUT_TOKENS)),
                    settings.EMBEDDING_MAX_INPUT_TOKENS,
                )
            self.model = model
            self._loaded = True
        except ModelNotReadyError:
            raise
        except Exception as exc:
            raise ModelNotReadyError("Unable to load embedding model") from exc

    @property
    def ready(self) -> bool:
        return self._loaded and self.runtime.is_ready and self.model is not None

    def token_lengths(self, texts: list[str]) -> list[int]:
        self.load()
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return [max(1, min(settings.EMBEDDING_MAX_INPUT_TOKENS, len(text) // 4 + 1)) for text in texts]
        try:
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=settings.EMBEDDING_MAX_INPUT_TOKENS,
                add_special_tokens=True,
            )
            return [max(1, min(settings.EMBEDDING_MAX_INPUT_TOKENS, len(ids))) for ids in encoded["input_ids"]]
        except (KeyError, TypeError, ValueError):
            return [max(1, min(settings.EMBEDDING_MAX_INPUT_TOKENS, len(text) // 4 + 1)) for text in texts]

    async def infer(self, texts: list[str]) -> list[list[float]]:
        self.load()

        def forward() -> list[list[float]]:
            try:
                import torch

                with torch.inference_mode():
                    vectors = self.model.encode(
                        texts,
                        batch_size=len(texts),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                    )
                array = np.asarray(vectors, dtype=np.float32)
                if array.ndim != 2 or array.shape != (len(texts), self.dimension):
                    raise ModelInferenceError(
                        f"Embedding output shape {array.shape} does not match ({len(texts)}, {self.dimension})"
                    )
                return array.tolist()
            except ModelInferenceError:
                raise
            except Exception as exc:
                raise ModelInferenceError("Embedding inference failed") from exc

        return await self.arbiter.execute(forward)

    def warmup(self) -> None:
        """Run one representative short forward before readiness is exposed."""
        self.load()
        try:
            import torch

            with torch.inference_mode():
                self.model.encode(
                    ["warmup: quy định nội bộ và điều kiện áp dụng"],
                    batch_size=1,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize,
                )
        except Exception as exc:
            raise ModelNotReadyError("Embedding model warmup failed") from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "ready": self.ready,
        }
