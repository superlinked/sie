"""SIE SDK error classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sie_sdk.types import RequestMetadata


class SIEError(Exception):
    """Base exception for SIE SDK errors."""


class SIEConnectionError(SIEError):
    """Error connecting to the SIE server."""


class RequestError(SIEError):
    """Terminal request or response-contract error.

    This covers 4xx responses plus unexpected redirects, malformed successful
    response bodies, and other terminal response-shape violations.

    ``request`` contains canonical request, usage, and debit metadata parsed
    from the terminal response headers when the server supplied any.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status_code: int | None = None,
        *,
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.param = param
        self.request = request


class ServerError(SIEError):
    """Error from the server (5xx responses).

    ``request`` contains canonical request, usage, and debit metadata parsed
    from the terminal response headers when the server supplied any.
    ``retry_after`` preserves a validated server retry hint in seconds when
    the terminal error carries one; callers must not infer that retrying a
    non-idempotent request is safe merely because the hint is present.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status_code: int | None = None,
        *,
        param: str | None = None,
        request: RequestMetadata | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.param = param
        self.request = request
        self.retry_after = retry_after


class ProvisioningError(SIEError):
    """Error when capacity is not available and provisioning timed out.

    Raised when:
    - Server returns 503 with PROVISIONING code
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
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.gpu = gpu
        self.retry_after = retry_after
        self.param = param


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
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code="MODEL_LOAD_FAILED", status_code=502, param=param, request=request)
        self.model = model
        self.error_class = error_class
        self.permanent = permanent
        self.attempts = attempts


class InputTooLongError(RequestError):
    """Error when the request input exceeds the model's maximum token capacity.

    Raised when the server returns HTTP ``400 INPUT_TOO_LONG`` for an
    extraction request. Distinct from generic ``RequestError`` so callers
    can branch on token-budget failures specifically (e.g. truncate the
    input client-side, switch to a longer-context model, or surface a
    targeted error to the end user) without parsing the error code.

    Subclass of :class:`RequestError` so existing 4xx handlers continue
    to work; new code can catch :class:`InputTooLongError` for tailored
    handling.

    Attributes:
        model: The model that was requested.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code="INPUT_TOO_LONG", status_code=400, param=param, request=request)
        self.model = model


class EstimateUnroutableError(ServerError):
    """The gateway cannot PRICE the request, so it will not run it either.

    Raised by :meth:`SIEClient.estimate` when the dry run answers ``503``:
    the active rate book declares no rate for the request's
    (model, profile, operation, region) identity, or the planner cannot bound
    one of the dimensions that book DOES price (an unbounded generation input,
    say, or a sealed lane whose GPU class is unresolved).

    This is deliberately the same verdict the real request would get — the
    estimate runs the live planner — so treat it as "this request is not
    sellable right now", not as an estimator limitation. :attr:`args` carries
    the planner's own reason, which names the unpriced identity or the
    dimension it could not bound.

    Subclass of :class:`ServerError` so existing 5xx handlers keep working.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code=code, status_code=503, param=param, request=request)


class RateLimitError(RequestError):
    """Error when the gateway rate-limits the caller and retries are exhausted.

    Raised when the gateway returns HTTP ``429 TOO_MANY_REQUESTS`` (code
    ``RATE_LIMIT``, per-key or per-account, default-on) and the SDK's
    bounded, ``Retry-After``-honoring retry budget (capped by
    ``provision_timeout_s``) is spent. The SDK honors the server's
    ``Retry-After`` on each attempt before giving up.

    Subclass of :class:`RequestError` so existing 4xx handlers keep working;
    new code can catch :class:`RateLimitError` specifically to back off at a
    higher level, shed load, or route elsewhere.

    Attributes:
        retry_after: The last ``Retry-After`` hint the server supplied
            (seconds), if any.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = "RATE_LIMIT",
        retry_after: float | None = None,
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code=code, status_code=429, param=param, request=request)
        self.retry_after = retry_after


class InsufficientCreditsError(RequestError):
    """Error when the account has insufficient credits to run the request.

    Raised when the gateway returns HTTP ``402 PAYMENT_REQUIRED`` with code
    ``INSUFFICIENT_CREDITS``. This is a TERMINAL billing failure — the SDK
    never retries it, because retrying a credit failure would be wrong.

    Subclass of :class:`RequestError` so existing 4xx handlers keep working;
    new code can catch :class:`InsufficientCreditsError` specifically to
    surface a top-up prompt or halt a batch.
    """

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code="INSUFFICIENT_CREDITS", status_code=402, param=param, request=request)


class SpendLimitError(RequestError):
    """Error when the API key's configured spend limit is exceeded.

    Raised when the gateway returns HTTP ``402 PAYMENT_REQUIRED`` with code
    ``KEY_SPEND_LIMIT_EXCEEDED``. This is a TERMINAL policy failure — the SDK
    never retries it. Distinct from :class:`InsufficientCreditsError`: the
    account may have credits, but this key has hit its own spend cap.

    Subclass of :class:`RequestError` so existing 4xx handlers keep working.
    """

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code="KEY_SPEND_LIMIT_EXCEEDED", status_code=402, param=param, request=request)


class AccountInactiveError(RequestError):
    """Error when the account is not permitted to submit work.

    Raised when the gateway returns HTTP ``403 FORBIDDEN`` with code
    ``ACCOUNT_SUSPENDED`` or ``ACCOUNT_PENDING_REVIEW``. This is a TERMINAL
    account-state failure — the SDK never retries it, because the account
    must be activated/reinstated out of band before work is accepted.

    Subclass of :class:`RequestError` so existing 4xx handlers keep working;
    new code can catch :class:`AccountInactiveError` and branch on
    :attr:`code` (``ACCOUNT_SUSPENDED`` vs ``ACCOUNT_PENDING_REVIEW``).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code=code, status_code=403, param=param, request=request)


class AccountStateUnavailableError(ServerError):
    """Error when the gateway cannot resolve the account's admission state.

    Raised when the gateway returns HTTP ``503`` with code
    ``ACCOUNT_STATE_UNAVAILABLE`` — a fail-closed infrastructure signal (the
    control plane could not resolve account state), distinct from a customer
    suspension. Surfaced as a typed TERMINAL error rather than being retried
    on the SDK's admission ladder: a caller may re-issue the whole request,
    but the SDK does not silently loop on an unresolved account state.

    Subclass of :class:`ServerError` so existing 5xx handlers keep working.
    """

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code="ACCOUNT_STATE_UNAVAILABLE", status_code=503, param=param, request=request)


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
        retry_after: Last validated server retry hint, in seconds.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        retries: int = 0,
        param: str | None = None,
        request: RequestMetadata | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            code="RESOURCE_EXHAUSTED",
            status_code=503,
            param=param,
            request=request,
            retry_after=retry_after,
        )
        self.model = model
        self.retries = retries


class IncompleteBatchError(ServerError):
    """A successful (HTTP 200) batch response dropped or added items.

    The gateway's queue path returns mixed-success batches as ``200`` carrying
    only the successful items — a per-item failure is dropped from the body,
    not surfaced as an error envelope. Batch responses are positional (item
    ``id`` is optional), so a shortened body silently shifts every item after
    the dropped one: a zip-inputs-to-outputs consumer would store results
    against the wrong inputs. The SDK guards the 1:1 input↔output contract on
    every batch response and raises this instead of returning a desynced list.

    Subclass of :class:`ServerError` — the server violated the response-shape
    contract even though the HTTP status was 200 — so existing ``ServerError``
    handlers keep working (this refines the untyped guard from #1526).
    ``status_code`` is ``None``: the response was not an HTTP error. Callers
    can catch :class:`IncompleteBatchError` specifically and retry item-wise
    (single-item batches get per-item error visibility) using
    :attr:`missing_ids` when available.

    Attributes:
        expected: Number of items submitted in this HTTP request.
        received: Number of items the response body carried.
        model: The model that was requested.
        missing_ids: Ids of submitted items absent from the response — only
            when ids identify every item on both sides (every submitted item
            carried an ``id`` and every returned item echoed one), ``None``
            otherwise.
        request_id: Gateway request id (``x-sie-request-id``) when the
            response carried one; quote it when reporting the incident.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        expected: int,
        received: int,
        model: str | None = None,
        missing_ids: list[str] | None = None,
        request: RequestMetadata | None = None,
    ) -> None:
        super().__init__(message, code=code, request=request)
        self.expected = expected
        self.received = received
        self.model = model
        self.missing_ids = missing_ids
        self.request_id: str | None = (request or {}).get("id")


class JobFailedError(SIEError):
    """A job reached a non-successful terminal state (``failed``/``suspended``/``cancelled``).

    Raised by :meth:`SIEClient.jobs.wait` / :meth:`SIEAsyncClient.jobs.wait`
    only when called with ``raise_on_failure=True``; the default remains
    back-compatible and returns the terminal status doc unchanged. The
    gateway's terminal reason rides ``outcome`` / ``error_code`` on the status
    doc (see :class:`sie_sdk.types.JobStatus`); this surfaces them so a caller
    can branch on the failure without re-reading the doc.

    Attributes:
        job_id: The job that failed.
        state: The terminal state (``failed``, ``suspended``, or ``cancelled``).
        outcome: The gateway's terminal outcome (e.g. ``reexecution_required``),
            or ``None`` when the status doc carried none.
        error_code: The gateway's terminal error code (e.g.
            ``RESULT_HANDLE_EXPIRED``), or ``None`` when absent.
    """

    def __init__(
        self,
        message: str,
        *,
        job_id: str | None = None,
        state: str | None = None,
        outcome: str | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.state = state
        self.outcome = outcome
        self.error_code = error_code
