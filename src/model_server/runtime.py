"""PyTorch device checks and the single-process MPS execution gate."""

from __future__ import annotations

import asyncio
import inspect
import os
import platform
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TypeVar

from core.config import settings
from model_server.errors import ModelInferenceError, ModelNotReadyError, MPSUnavailableError

T = TypeVar("T")


class MPSExecutionArbiter:
    """Bound the number of simultaneous forward passes on one Mac GPU.

    MPS uses unified memory; allowing many independent forwards can exhaust
    memory even when the HTTP server accepts requests concurrently.  The
    default is one in-flight batch, while the semaphore remains configurable
    for measured hardware.
    """

    def __init__(self, max_inflight_batches: int = 1):
        if max_inflight_batches <= 0:
            raise ValueError("max_inflight_batches must be positive")
        self.max_inflight_batches = max_inflight_batches
        self._semaphore = asyncio.Semaphore(max_inflight_batches)
        self._active = 0
        self._completed = 0

    async def execute(self, function: Callable[[], T | Awaitable[T]]) -> T:
        async with self._semaphore:
            self._active += 1
            try:
                if inspect.iscoroutinefunction(function):
                    task = asyncio.create_task(function())
                else:
                    # PyTorch's forward pass is synchronous.  Move the
                    # complete call (not just its return value) off the loop.
                    task = asyncio.create_task(asyncio.to_thread(function))
                try:
                    result = await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A cancelled HTTP request must not release the MPS slot
                    # while its non-cancellable PyTorch thread is still using
                    # unified memory.  Wait for it, then propagate cancel.
                    with suppress(BaseException):
                        await task
                    raise
                if inspect.isawaitable(result):
                    return await result
                return result
            except Exception as exc:
                if isinstance(exc, ModelInferenceError):
                    raise
                raise ModelInferenceError() from exc
            finally:
                self._active = max(0, self._active - 1)
                self._completed += 1

    def stats(self) -> dict[str, int]:
        return {
            "max_inflight_batches": self.max_inflight_batches,
            "active_batches": self._active,
            "completed_batches": self._completed,
        }


class TorchRuntime:
    """Lazy torch capability check, with product fail-closed semantics."""

    def __init__(self, device: str | None = None, allow_cpu_fallback: bool | None = None):
        self.device = (device or settings.MODEL_SERVER_DEVICE).lower()
        self.allow_cpu_fallback = (
            settings.MODEL_SERVER_ALLOW_CPU_FALLBACK if allow_cpu_fallback is None else allow_cpu_fallback
        )
        self._checked = False
        self._available = False

    @property
    def is_ready(self) -> bool:
        return self._checked and self._available

    def check(self) -> str:
        """Validate the configured device without loading model weights."""
        try:
            import torch
        except Exception as exc:  # pragma: no cover - dependency is project-pinned
            raise ModelNotReadyError("PyTorch is not installed") from exc
        if self.device not in {"cpu", "mps"}:
            raise ModelNotReadyError("MODEL_SERVER_DEVICE must be cpu or mps")
        if self.device == "mps":
            if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
                raise MPSUnavailableError("MPS model server requires an Apple Silicon macOS host")
            fallback = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "0").lower()
            if settings.APP_ENV.lower() in {"production", "prod"} and fallback in {"1", "true", "yes", "on"}:
                raise ModelNotReadyError("Production MPS server must disable PyTorch CPU fallback")
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            available = bool(mps and mps.is_built() and mps.is_available())
            if not available:
                if not self.allow_cpu_fallback:
                    raise MPSUnavailableError("PyTorch MPS is not available on this host")
                self.device = "cpu"
        self._checked = True
        self._available = True
        return self.device

    def torch_device(self):
        self.check()
        import torch

        return torch.device(self.device)
