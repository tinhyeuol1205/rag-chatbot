"""Public, machine-readable errors returned by the model server."""

from __future__ import annotations


class ModelServerError(Exception):
    """Base error with an HTTP-safe code and retry hint."""

    status_code = 503
    error_code = "model_server_error"
    public_message = "Model service is unavailable"
    retry_after: int | None = None

    def __init__(self, message: str | None = None):
        super().__init__(message or self.public_message)
        self.message = message or self.public_message


class ModelQueueFullError(ModelServerError):
    status_code = 429
    error_code = "model_queue_full"
    public_message = "Model service is busy; retry later"
    retry_after = 1


class ModelQueueTimeoutError(ModelServerError):
    status_code = 504
    error_code = "model_queue_timeout"
    public_message = "Model service queue deadline expired"


class ModelNotReadyError(ModelServerError):
    status_code = 503
    error_code = "model_not_ready"
    public_message = "Model service is not ready"
    retry_after = 1


class MPSUnavailableError(ModelNotReadyError):
    error_code = "mps_unavailable"


class ModelContractMismatchError(ModelServerError):
    status_code = 409
    error_code = "model_contract_mismatch"
    public_message = "Requested model contract does not match the loaded model"


class ModelAuthenticationError(ModelServerError):
    status_code = 401
    error_code = "model_authentication_failed"
    public_message = "Invalid model server API key"


class ModelRequestTooLargeError(ModelServerError):
    status_code = 413
    error_code = "request_too_large"
    public_message = "Model request exceeds the configured batch limits"


class ModelInferenceError(ModelServerError):
    status_code = 503
    error_code = "inference_failed"
    public_message = "Model inference failed"
    retry_after = 1
