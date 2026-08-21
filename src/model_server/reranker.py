"""BGE-reranker-v2-m3 model adapter used by the rerank batcher."""

from __future__ import annotations

from typing import Any

from core.config import settings
from model_server.errors import ModelInferenceError, ModelNotReadyError
from model_server.runtime import MPSExecutionArbiter, TorchRuntime


class RerankerRunner:
    """One CrossEncoder instance with bounded MPS forward passes."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        revision: str | None = None,
        runtime: TorchRuntime | None = None,
        arbiter: MPSExecutionArbiter | None = None,
    ):
        self.model_id = model_id or settings.RERANKER_MODEL_ID
        self.revision = revision if revision is not None else settings.RERANKER_MODEL_REVISION
        self.max_input_tokens = settings.RERANKER_MAX_INPUT_TOKENS
        self.runtime = runtime or TorchRuntime()
        self.arbiter = arbiter or MPSExecutionArbiter(settings.MODEL_SERVER_MPS_MAX_INFLIGHT_BATCHES)
        self.model: Any = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        device = self.runtime.check()
        try:
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {
                "device": device,
                "max_length": self.max_input_tokens,
            }
            if self.revision:
                kwargs["revision"] = self.revision
            self.model = CrossEncoder(self.model_id, **kwargs)
            getattr(self.model, "model", self.model).eval()
            self._loaded = True
        except ModelNotReadyError:
            raise
        except Exception as exc:
            raise ModelNotReadyError("Unable to load reranker model") from exc

    @property
    def ready(self) -> bool:
        return self._loaded and self.runtime.is_ready and self.model is not None

    def token_lengths(self, query: str, documents: list[str]) -> list[int]:
        self.load()
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return [
                max(1, min(self.max_input_tokens, (len(query) + len(document)) // 4 + 2))
                for document in documents
            ]
        try:
            encoded = tokenizer(
                [query] * len(documents),
                documents,
                truncation=True,
                max_length=self.max_input_tokens,
                add_special_tokens=True,
            )
            return [max(1, min(self.max_input_tokens, len(ids))) for ids in encoded["input_ids"]]
        except (KeyError, TypeError, ValueError):
            return [
                max(1, min(self.max_input_tokens, (len(query) + len(document)) // 4 + 2))
                for document in documents
            ]

    async def infer(self, query: str, documents: list[str]) -> list[float]:
        self.load()

        def forward() -> list[float]:
            try:
                import torch

                pairs = [[query, document] for document in documents]
                with torch.inference_mode():
                    values = self.model.predict(
                        pairs,
                        batch_size=min(settings.RERANK_BATCH_SIZE, len(pairs)),
                        show_progress_bar=False,
                    )
                scores = [float(value) for value in values]
                if len(scores) != len(documents):
                    raise ModelInferenceError("Reranker output cardinality mismatch")
                return scores
            except ModelInferenceError:
                raise
            except Exception as exc:
                raise ModelInferenceError("Reranker inference failed") from exc

        return await self.arbiter.execute(forward)

    def warmup(self) -> None:
        """Run one query/document pair before readiness is exposed."""
        self.load()
        try:
            import torch

            with torch.inference_mode():
                self.model.predict(
                    [["warmup: điều kiện áp dụng", "warmup: văn bản quy định"]],
                    batch_size=1,
                    show_progress_bar=False,
                )
        except Exception as exc:
            raise ModelNotReadyError("Reranker model warmup failed") from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "revision": self.revision,
            "max_input_tokens": self.max_input_tokens,
            "ready": self.ready,
        }
