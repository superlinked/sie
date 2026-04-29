import logging
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Any, TypeVar

import msgspec
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from sie_server.api.serialization import MsgPackResponse, _convert_for_json
from sie_server.core.oom import is_oom_error
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import QueueFullError
from sie_server.observability.metrics import record_request
from sie_server.observability.tracing import get_current_trace_id
from sie_server.types.responses import ErrorCode

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from sie_server.core.registry import ModelRegistry

# Default Retry-After (seconds) returned alongside RESOURCE_EXHAUSTED when no
# engine-config override is supplied. Kept small enough that an SDK using the
# default backoff (5 → 10 → 20 s) will typically land within the LRU-eviction
# + recovery window. Operators can override per-cluster via
# ``engine_config.oom_recovery.retry_after_s`` (env: SIE_OOM_RECOVERY__RETRY_AFTER_S).
_OOM_DEFAULT_RETRY_AFTER_S = 5


def oom_retry_after_from_registry(registry: "ModelRegistry | None") -> int:
    """Resolve the OOM ``Retry-After`` value from a registry's engine config.

    Falls back to ``_OOM_DEFAULT_RETRY_AFTER_S`` when:

    - The registry is ``None`` (early app startup, edge tests).
    - The registry has no engine config attached (legacy embed-only test
      harnesses).
    - The chained attribute access does not yield an ``int`` — protects
      tests that use ``MagicMock(spec=ModelRegistry)`` without explicitly
      stubbing ``engine_config``. Without this fallback the ``Retry-After``
      header would receive ``str(<Mock>)``, breaking the SDK's parsing.

    Centralised so the three API route handlers stay one-liners.
    """
    if registry is None or registry.engine_config is None:
        return _OOM_DEFAULT_RETRY_AFTER_S
    value = registry.engine_config.oom_recovery.retry_after_s
    if not isinstance(value, int):
        # Defensive: a misconfigured test fixture / future schema change
        # would otherwise silently emit a garbage Retry-After header.
        return _OOM_DEFAULT_RETRY_AFTER_S
    return value


logger = logging.getLogger(__name__)

T = TypeVar("T")


# Content types
MSGPACK_CONTENT_TYPE = "application/msgpack"
JSON_CONTENT_TYPE = "application/json"

# Version negotiation headers
SERVER_VERSION_HEADER = "X-SIE-Server-Version"
SDK_VERSION_HEADER = "X-SIE-SDK-Version"


def _get_server_version() -> str:
    try:
        return pkg_version("sie-server")
    except PackageNotFoundError:
        return "unknown"


_SERVER_VERSION = _get_server_version()

_sdk_version_warned: set[str] = set()

_MIN_VERSION_PARTS = 2  # major.minor required for skew check


def _check_sdk_version(http_request: Request) -> None:
    sdk_version = http_request.headers.get(SDK_VERSION_HEADER)
    if not sdk_version:
        return
    try:
        sdk_parts = sdk_version.split(".")
        server_parts = _SERVER_VERSION.split(".")
        if len(sdk_parts) < _MIN_VERSION_PARTS or len(server_parts) < _MIN_VERSION_PARTS:
            return
        sdk_major, sdk_minor = int(sdk_parts[0]), int(sdk_parts[1])
        server_major, server_minor = int(server_parts[0]), int(server_parts[1])
        key = f"{sdk_major}.{sdk_minor}"
        if key in _sdk_version_warned:
            return
        if sdk_major != server_major or abs(sdk_minor - server_minor) > 1:
            logger.warning(
                "SDK version skew: client sent %s, server is %s",
                sdk_version,
                _SERVER_VERSION,
            )
            _sdk_version_warned.add(key)
    except (ValueError, IndexError):
        pass


def _mask_api_key(key: str) -> str:
    """Mask an API key, showing only the last 4 characters."""
    mask = "****"
    if len(key) <= len(mask):
        return mask
    return f"{mask}{key[-4:]}"


@dataclass(frozen=True)
class RequestContext:
    """Request-scoped context for structured logging."""

    request_id: str
    api_key: str | None
    queue_depth: int | None


def extract_request_context(
    http_request: Request,
    model: str,
    registry: "ModelRegistry",
) -> RequestContext:
    """Extract request context from HTTP request for structured logging.

    Args:
        http_request: FastAPI request object.
        model: Model name (for queue depth lookup).
        registry: ModelRegistry to get queue depth.

    Returns:
        RequestContext with request_id, masked api_key, and queue_depth.
    """
    _check_sdk_version(http_request)

    # request_id: from X-Request-ID header or generate new
    request_id = http_request.headers.get("x-request-id") or str(uuid.uuid4())

    # api_key: from Authorization header, masked
    auth_header = http_request.headers.get("authorization")
    api_key: str | None = None
    if auth_header:
        token = (auth_header[7:] if auth_header.lower().startswith("bearer ") else auth_header).strip()
        if token:
            api_key = _mask_api_key(token)

    # queue_depth: from worker's pending_count
    queue_depth: int | None = None
    worker = registry.get_worker(model)
    if worker is not None:
        queue_depth = worker.pending_count

    return RequestContext(request_id=request_id, api_key=api_key, queue_depth=queue_depth)


class ContentNegotiator:
    """Handles HTTP content negotiation for msgpack/JSON."""

    @staticmethod
    def wants_msgpack(accept: str | None) -> bool:
        """Check if client prefers msgpack based on Accept header.

        Default to msgpack if no preference specified (per DESIGN.md).
        """
        if not accept:
            return True  # Default to msgpack

        # Parse Accept header and check preferences
        accept_lower = accept.lower()
        if MSGPACK_CONTENT_TYPE in accept_lower:
            return True
        if "application/x-msgpack" in accept_lower:
            return True
        return JSON_CONTENT_TYPE not in accept_lower

    @staticmethod
    def is_msgpack_request(content_type: str | None) -> bool:
        """Check if request body is msgpack based on Content-Type header."""
        if not content_type:
            return False
        content_type_lower = content_type.lower()
        return MSGPACK_CONTENT_TYPE in content_type_lower or "application/x-msgpack" in content_type_lower


class RequestParser:
    """Parses and validates HTTP request bodies via msgspec."""

    @staticmethod
    async def parse(http_request: Request, type: type[T]) -> T:
        """Parse the request body into a typed msgspec Struct.

        Supports both msgpack (default) and JSON (fallback) based on
        Content-Type header.

        Args:
            http_request: FastAPI Request object.
            type: The msgspec.Struct type to parse into.

        Returns:
            Parsed and validated request as a typed Struct instance.

        Raises:
            HTTPException: 400 if parsing or validation fails.
        """
        content_type = http_request.headers.get("content-type")
        body = await http_request.body()

        try:
            if ContentNegotiator.is_msgpack_request(content_type):
                return msgspec.msgpack.decode(body, type=type)
            return msgspec.json.decode(body, type=type)
        except (msgspec.ValidationError, msgspec.DecodeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": ErrorCode.INVALID_INPUT.value,
                    "message": str(e),
                },
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": ErrorCode.INVALID_INPUT.value,
                    "message": f"Failed to parse request body: {e}",
                },
            ) from e


class ModelStateChecker:
    """Validates model state before inference."""

    def __init__(self, registry: "ModelRegistry", model: str, span: "Span") -> None:
        """Initialize checker.

        Args:
            registry: ModelRegistry instance.
            model: Model name to check.
            span: OpenTelemetry span for error attributes.
        """
        self.registry = registry
        self.model = model
        self.span = span

    def check_exists(self) -> None:
        """Check if model exists in registry.

        Raises:
            HTTPException: 404 if model not found.
        """
        if not self.registry.has_model(self.model):
            self.span.set_attribute("error", "model_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": ErrorCode.MODEL_NOT_FOUND.value,
                    "message": f"Model '{self.model}' not found",
                },
            )

    def check_not_unloading(self) -> None:
        """Check if model is being unloaded.

        Raises:
            HTTPException: 503 if model is unloading.
        """
        if self.registry.is_unloading(self.model):
            self.span.set_attribute("error", "model_unloading")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": ErrorCode.MODEL_NOT_LOADED.value,
                    "message": f"Model '{self.model}' is unloading",
                },
            )

    def check_not_loading(self) -> None:
        """Check if model is currently loading.

        Raises:
            HTTPException: 503 with Retry-After if model is loading.
        """
        if self.registry.is_loading(self.model):
            self.span.set_attribute("error", "model_loading")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": ErrorCode.MODEL_LOADING.value,
                    "message": f"Model '{self.model}' is loading, please retry",
                },
                headers={"Retry-After": "5"},
            )

    def check_not_failed(self) -> None:
        """Check whether the model has a recorded terminal load failure.

        When the registry has a sticky :class:`LoadFailure` still in
        cooldown, return ``502 MODEL_LOAD_FAILED`` with **no**
        ``Retry-After`` header. The SDK uses the absence of
        ``Retry-After`` (combined with the dedicated error code) to
        bypass its 5-minute ``MODEL_LOADING`` retry budget and surface
        a :class:`ModelLoadFailedError` to the caller immediately.

        This is the symptom-side fix for sie-test#85: previously a
        gated/dep-missing model would loop the SDK for 5 minutes; now
        the server tells the SDK "don't bother" the first time it asks.

        Raises:
            HTTPException: 502 ``MODEL_LOAD_FAILED`` if registry holds a
                non-expired failure record.
        """
        if not self.registry.is_failed(self.model):
            return
        failure = self.registry.get_failure(self.model)
        # Defensive: ``is_failed`` is True implies a record exists, but a
        # concurrent ``clear_failure`` could have raced us to the dict.
        if failure is None:  # pragma: no cover — narrow race window
            return
        self.span.set_attribute("error", "model_load_failed")
        self.span.set_attribute("model_load_error_class", failure.error_class.value)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": ErrorCode.MODEL_LOAD_FAILED.value,
                "message": (
                    f"Model '{self.model}' failed to load "
                    f"({failure.error_class.value}, attempts={failure.attempts}): {failure.message}"
                ),
                "error_class": failure.error_class.value,
                "attempts": failure.attempts,
                "permanent": failure.is_permanent,
            },
            # Intentionally no Retry-After header — the SDK uses its
            # absence to short-circuit the MODEL_LOADING retry budget.
        )

    async def ensure_loaded(self, device: str) -> None:
        """Start loading model if not loaded, raise 503 to retry.

        Args:
            device: Device to load model on (cpu, cuda, mps).

        Raises:
            HTTPException: 502 ``MODEL_LOAD_FAILED`` (non-retryable) if a
                terminal failure is already recorded; otherwise 503
                ``MODEL_LOADING`` with ``Retry-After`` so the SDK retries.
        """
        if self.registry.is_loaded(self.model):
            return

        # Short-circuit recorded terminal failures BEFORE kicking off
        # another doomed background load. Without this guard, every
        # request would retrigger ``start_load_async`` and the SDK
        # would burn its 5-minute MODEL_LOADING budget on a known-bad
        # configuration (sie-test#85).
        self.check_not_failed()

        logger.info("Starting background load for model %s on device %s", self.model, device)
        await self.registry.start_load_async(self.model, device=device)

        # ``start_load_async`` is a no-op when a failure was recorded
        # mid-flight (race between ``check_not_failed`` and this call).
        # Re-check and surface the terminal error rather than telling
        # the SDK to retry.
        self.check_not_failed()

        # Race window: a *concurrent* request could have driven the load
        # to completion between the initial ``is_loaded`` check above
        # and this point. In that case we should return success to the
        # caller (no need to make them retry on a 503) — the model is
        # already serving.
        if self.registry.is_loaded(self.model):
            return

        self.span.set_attribute("error", "model_loading")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": ErrorCode.MODEL_LOADING.value,
                "message": f"Model '{self.model}' is loading, please retry",
            },
            headers={"Retry-After": "5"},
        )

    async def validate_ready(self, device: str) -> None:
        """Run all state checks to ensure model is ready for inference.

        Checks in order: exists, not unloading, not failed, not loading,
        ensure loaded. ``check_not_failed`` runs before ``check_not_loading``
        so a recorded terminal failure short-circuits the SDK's retry
        loop even if a stale ``_loading`` flag races with it.

        Args:
            device: Device to load model on if needed.

        Raises:
            HTTPException: Various status codes based on model state.
        """
        self.check_exists()
        self.check_not_unloading()
        self.check_not_failed()
        self.check_not_loading()
        await self.ensure_loaded(device)


class InferenceErrorHandler:
    """Handles common inference errors and converts to HTTP responses."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        span: "Span",
        ctx: RequestContext | None = None,
        oom_retry_after_s: int = _OOM_DEFAULT_RETRY_AFTER_S,
    ) -> None:
        """Initialize handler.

        Args:
            model: Model name for metrics.
            endpoint: Endpoint name for metrics (encode, score, extract).
            span: OpenTelemetry span for error attributes.
            ctx: Optional request context for structured logging.
            oom_retry_after_s: Value for the ``Retry-After`` header on
                ``RESOURCE_EXHAUSTED`` (503) responses. Defaults to the
                module constant; route handlers pass
                ``registry.engine_config.oom_recovery.retry_after_s`` so
                operators can tune it per cluster.
        """
        self.model = model
        self.endpoint = endpoint
        self.span = span
        self.ctx = ctx
        self.oom_retry_after_s = oom_retry_after_s

    def _log_kwargs(self) -> dict[str, Any]:
        """Build kwargs for record_request from context."""
        if self.ctx is None:
            return {}
        return {
            "request_id": self.ctx.request_id,
            "api_key": self.ctx.api_key,
            "queue_depth": self.ctx.queue_depth,
        }

    def handle_queue_full(self, error: QueueFullError) -> HTTPException:
        """Handle queue full backpressure error.

        Args:
            error: QueueFullError from worker.

        Returns:
            HTTPException with 503 status.
        """
        self.span.set_attribute("error", "queue_full")
        record_request(model=self.model, endpoint=self.endpoint, status="error", **self._log_kwargs())
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": ErrorCode.MODEL_NOT_LOADED.value,
                "message": str(error),
            },
        )

    def handle_value_error(self, error: ValueError) -> HTTPException:
        """Handle invalid input errors.

        Args:
            error: ValueError from adapter/worker.

        Returns:
            HTTPException with 400 status.
        """
        self.span.set_attribute("error", "invalid_input")
        record_request(model=self.model, endpoint=self.endpoint, status="error", **self._log_kwargs())
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_INPUT.value,
                "message": str(error),
            },
        )

    def handle_inference_error(self, error: Exception, operation: str = "Inference") -> HTTPException:
        """Handle generic inference errors.

        OOM errors are mapped to 503 ``RESOURCE_EXHAUSTED`` with a
        ``Retry-After`` header so the SDK can auto-retry. Everything else
        keeps the legacy 500 ``INFERENCE_ERROR`` mapping.

        Args:
            error: Exception from inference.
            operation: Operation name for error message (Inference, Extraction, Scoring).

        Returns:
            HTTPException with 503 (OOM) or 500 (other).
        """
        # Recognise both the worker's ``ResourceExhaustedError`` (recovery
        # exhausted) and any OOM that escaped without recovery (recovery
        # disabled or a code path the executor doesn't wrap). Both deserve
        # the same 503 treatment.
        if is_oom_error(error):
            logger.warning(
                "%s OOM for model %s, returning 503 RESOURCE_EXHAUSTED: %s",
                operation,
                self.model,
                error,
            )
            self.span.set_attribute("error", "resource_exhausted")
            record_request(model=self.model, endpoint=self.endpoint, status="error", **self._log_kwargs())
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": ErrorCode.RESOURCE_EXHAUSTED.value,
                    "message": f"{operation} temporarily unavailable due to resource pressure: {error}",
                },
                headers={"Retry-After": str(self.oom_retry_after_s)},
            )

        logger.exception("%s error for model %s", operation, self.model)
        self.span.set_attribute("error", "inference_error")
        record_request(model=self.model, endpoint=self.endpoint, status="error", **self._log_kwargs())
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": ErrorCode.INFERENCE_ERROR.value,
                "message": f"{operation} error: {error}",
            },
        )


class ResponseBuilder:
    """Builds HTTP responses with common headers."""

    @staticmethod
    def build_headers(timing: RequestTiming | None = None) -> dict[str, str]:
        """Build response headers with trace ID and timing.

        Args:
            timing: Optional RequestTiming to include timing headers.

        Returns:
            Headers dict with X-Trace-ID and timing headers.
        """
        headers: dict[str, str] = {SERVER_VERSION_HEADER: _SERVER_VERSION}

        # Add trace ID
        trace_id = get_current_trace_id()
        if trace_id:
            headers["X-Trace-ID"] = trace_id

        # Add timing headers
        if timing is not None:
            timing.finish()
            timing_headers = timing.to_headers()
            headers.update(timing_headers)

        return headers

    @staticmethod
    def build_response(
        content: Any,
        accept: str | None,
        headers: dict[str, str],
        *,
        convert_for_json: bool = False,
    ) -> MsgPackResponse | JSONResponse:
        """Build response with content negotiation.

        Args:
            content: Response content (TypedDict or dict).
            accept: Accept header value.
            headers: Response headers.
            convert_for_json: If True, convert numpy arrays for JSON response.

        Returns:
            MsgPackResponse or JSONResponse based on Accept header.
        """
        if ContentNegotiator.wants_msgpack(accept):
            return MsgPackResponse(content=content, headers=headers)

        # JSON fallback
        if convert_for_json:
            content = _convert_for_json(dict(content))

        return JSONResponse(content=content, headers=headers)
