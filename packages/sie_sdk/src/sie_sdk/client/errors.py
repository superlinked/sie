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


class ModelLoadFailedError(ServerError):
    """Error when the server reports a recorded model-load failure.

    Distinct from :class:`ModelLoadingError` — this is raised on the
    first response (no retry budget consumed) when the server returns
    HTTP ``502 MODEL_LOAD_FAILED``. The server uses this code for both:

    - **Permanent-class failures** (``GATED``, ``NOT_FOUND``,
      ``DEPENDENCY``, ``UNKNOWN``) where retrying would waste time and
      operator intervention is required (e.g. set ``HF_TOKEN``, accept
      the model license, upgrade ``transformers``). These carry
      ``permanent=True``.
    - **Transient classes in active cooldown** (``OOM``, ``NETWORK``)
      where the registry is suppressing retries for a finite window so
      the load loop does not hot-spin. These carry ``permanent=False``;
      the failure auto-expires and a subsequent request will trigger a
      fresh load attempt.

    Either way the server omits the ``Retry-After`` header so the SDK
    short-circuits its ``MODEL_LOADING`` retry budget and surfaces the
    error immediately. Callers can either catch :class:`ServerError`
    generally (preserves legacy 5xx handling) or catch
    :class:`ModelLoadFailedError` specifically and branch on
    :attr:`permanent` / :attr:`error_class` for tailored remediation.

    Attributes:
        model: The model that was requested.
        error_class: Server-side classification (``GATED``, ``OOM``,
            ``DEPENDENCY``, ``NOT_FOUND``, ``NETWORK``, ``UNKNOWN``).
            Use this to route to remediation paths (surface an
            "HF_TOKEN" hint for ``GATED``, retry later for ``OOM``).
        permanent: Whether the failure is non-retryable per server
            policy. ``True`` indicates a terminal failure that will not
            auto-clear — an operator must fix the underlying cause.
            ``False`` indicates a server-side cooldown over a transient
            condition; retrying after the cooldown window will succeed
            once the underlying issue resolves.
        attempts: How many load attempts the server has logged.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        error_class: str | None = None,
        permanent: bool = True,
        attempts: int = 1,
    ) -> None:
        super().__init__(message, code="MODEL_LOAD_FAILED", status_code=502)
        self.model = model
        self.error_class = error_class
        self.permanent = permanent
        self.attempts = attempts


class ResourceExhaustedError(ServerError):
    """Error when the server has exhausted its OOM-recovery strategies.

    Raised when:
    - Server returns 503 with RESOURCE_EXHAUSTED code
    - SDK retry limit is exceeded

    Subclass of :class:`ServerError` so callers that already catch
    ``ServerError`` continue to behave correctly; new code can catch
    ``ResourceExhaustedError`` specifically to react to sustained GPU
    pressure (e.g., back off, route elsewhere, scale up).

    Attributes:
        model: The model that was requested.
        retries: Number of retry attempts made before giving up.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        retries: int = 0,
    ) -> None:
        super().__init__(message, code="RESOURCE_EXHAUSTED", status_code=503)
        self.model = model
        self.retries = retries
