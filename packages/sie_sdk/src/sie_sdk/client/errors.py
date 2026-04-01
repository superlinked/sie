"""SIE SDK error classes."""

from __future__ import annotations


class SIEError(Exception):
    """Base exception for SIE SDK errors."""


class SIEConnectionError(SIEError):
    """Error connecting to the SIE server."""


class RequestError(SIEError):
    """Error in the request (4xx responses)."""

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ServerError(SIEError):
    """Error from the server (5xx responses)."""

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ProvisioningError(SIEError):
    """Error when capacity is not available and provisioning timed out.

    Raised when:
    - Server returns 202 (no capacity, provisioning)
    - wait_for_capacity=False (caller doesn't want to wait)
    - Or provisioning timeout exceeded

    Attributes:
        gpu: The GPU type that was requested.
        retry_after: Suggested retry delay from server (if provided).
    """

    def __init__(
        self,
        message: str,
        *,
        gpu: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.gpu = gpu
        self.retry_after = retry_after


class PoolError(SIEError):
    """Error related to resource pool operations.

    Raised when:
    - Pool creation fails (e.g., insufficient capacity)
    - Pool not found
    - Pool in invalid state (e.g., expired)
    - Pool lease renewal fails

    Attributes:
        pool_name: Name of the pool.
        state: Current pool state (if known).
    """

    def __init__(
        self,
        message: str,
        *,
        pool_name: str | None = None,
        state: str | None = None,
    ) -> None:
        super().__init__(message)
        self.pool_name = pool_name
        self.state = state


class LoraLoadingError(SIEError):
    """Error when LoRA adapter is loading and retry limit exceeded.

    Raised when:
    - Server returns 503 with LORA_LOADING code
    - Retry limit is exceeded

    Attributes:
        lora: The LoRA adapter that was requested.
        model: The model the LoRA was requested for.
    """

    def __init__(
        self,
        message: str,
        *,
        lora: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.lora = lora
        self.model = model


class ModelLoadingError(SIEError):
    """Error when model is loading and retry limit exceeded.

    Raised when:
    - Server returns 503 with MODEL_LOADING code
    - Retry limit is exceeded

    Attributes:
        model: The model that was requested.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
