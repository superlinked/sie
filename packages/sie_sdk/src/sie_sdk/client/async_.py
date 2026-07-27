"""Asynchronous SIE Engine Client.

Provides an async Python client for the Search Inference Engine server.

Async variants for all client methods.

Example:
    >>> async with SIEAsyncClient("http://localhost:8080") as client:
    ...     result = await client.encode("bge-m3", {"text": "Hello world"})
    ...     print(result["dense"].shape)
    (1024,)

    >>> # With defaults for all requests
    >>> async with SIEAsyncClient(
    ...     "http://gateway:8080",
    ...     gpu="l4",
    ...     options={"normalize": True},
    ... ) as client:
    ...     result = await client.encode("bge-m3", {"text": "Hello"})  # uses l4

    >>> # With resource pool for isolated capacity
    >>> async with SIEAsyncClient(
    ...     "http://gateway:8080",
    ...     pool={"name": "eval-bench", "gpus": {"l4": 2}},
    ... ) as client:
    ...     result = await client.encode("bge-m3", {"text": "Hello"}, gpu="eval-bench/l4")
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import warnings
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Literal, Self, cast, overload
from urllib.parse import urlencode

import aiohttp
import msgpack
import msgpack_numpy as m

from sie_sdk.audio import convert_item_audio
from sie_sdk.documents import convert_item_document
from sie_sdk.files import resolve_upload
from sie_sdk.images import ImageLike, convert_images_for_json, convert_item_images
from sie_sdk.jobs import TERMINAL_JOB_STATES, build_job_body, decode_chunk_bytes, job_chunks
from sie_sdk.types import (
    Batch,
    CapacityInfo,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    Connection,
    ConnectionCreated,
    ConnectionRevoked,
    EncodeResult,
    ExtractResult,
    File,
    FileDeleted,
    GenerateChunk,
    GenerateGrammar,
    GenerateImage,
    GenerateResult,
    Item,
    JobResults,
    JobStatus,
    JobSubmitResult,
    ModelInfo,
    OutputType,
    PoolInfo,
    PoolSpec,
    RequestMetadata,
    ResponseInputMessage,
    ResponseResult,
    ScoreResult,
    StatusMessage,
    WorkerInfo,
)

from ._shared import (
    DEFAULT_LEASE_RENEWAL_INTERVAL_S,
    DEFAULT_PROVISION_TIMEOUT_S,
    HTTP_CLIENT_ERROR,
    HTTP_GATEWAY_TIMEOUT,
    JSON_CONTENT_TYPE,
    LORA_LOADING_DEFAULT_DELAY_S,
    LORA_LOADING_ERROR_CODE,
    LORA_LOADING_MAX_RETRIES,
    MODEL_LOADING_DEFAULT_DELAY_S,
    MODEL_LOADING_ERROR_CODE,
    MSGPACK_CONTENT_TYPE,
    PROVISIONING_ERROR_CODE,
    RESOURCE_EXHAUSTED_ERROR_CODE,
    RESOURCE_EXHAUSTED_MAX_RETRIES,
    SDK_VERSION_HEADER,
    SERVER_VERSION_HEADER,
    _coerce_token_count,
    attach_request_metadata,
    base_url_accepts_origin_credentials,
    build_chat_body,
    build_responses_body,
    check_version_skew,
    compute_oom_backoff,
    compute_retry_delay,
    convert_score_images_for_wire,
    copy_base_url_headers,
    get_error_code,
    get_retry_after,
    get_sdk_version,
    handle_error,
    is_transient_connect_error,
    next_stream_retry_delay,
    parse_encode_results,
    parse_extract_results,
    parse_gpu_param,
    parse_request_metadata,
    parse_score_result,
    parse_terminal_json_object,
    provisioning_retry_delay,
    raise_if_input_too_long,
    raise_if_model_load_failed,
    request_matches_base_url_origin,
    retry_after_or_default,
    sse_chunk_error,
    sse_headers,
    validate_encode_result_count,
    validate_generate_grammar,
    validate_generate_request_body,
    websocket_matches_base_url_origin,
)
from ._sse import aiter_sse_payloads
from .errors import (
    LoraLoadingError,
    ModelLoadingError,
    PoolError,
    ProvisioningError,
    RequestError,
    ResourceExhaustedError,
    ServerError,
    SIEConnectionError,
)

logger = logging.getLogger(__name__)


class _PoolCreationInFlight:
    def __init__(self, future: asyncio.Future[None]) -> None:
        self.future = future


_PoolTaskEntry = asyncio.Task[None] | _PoolCreationInFlight


def _complete_pool_creation(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _fail_pool_creation(future: asyncio.Future[None], exc: BaseException) -> None:
    if future.done():
        return
    if isinstance(exc, asyncio.CancelledError):
        future.cancel()
        return
    future.set_exception(exc)
    # The creating caller raises the same exception directly. If no concurrent
    # caller awaits the shared future, consume it here to avoid loop warnings.
    with contextlib.suppress(asyncio.CancelledError):
        future.exception()


# Mid-flight transport errors retried under `wait_for_capacity=True`:
# the request was in flight and the peer severed the connection before a
# complete response arrived (proxy idle timeout, rolling restart,
# ECONNRESET). `ClientConnectorError` is retried separately at each call
# site to preserve its distinct "Failed to connect" message.
# Call-site `except` order: `_RETRYABLE_TRANSPORT_ERRORS` →
# `ClientConnectorError` → `(ClientError, OSError)` (first-match
# routing requires most-specific first).
_RETRYABLE_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ServerTimeoutError,
    aiohttp.ClientPayloadError,
)

_LEASE_RENEWAL_MAX_RETRIES = 5

# NOTE: msgpack_numpy.patch() is called lazily in SIEAsyncClient.__init__
# (see sync.py for details).
_NUMPY_PATCHED = False


def _parse_generate_result_async(
    data: dict[str, Any],
    *,
    request: RequestMetadata | None = None,
) -> GenerateResult:
    """Build a :class:`GenerateResult` from the gateway's JSON envelope.

    ``model`` and ``text`` are required strings; missing or null values are
    surfaced as :class:`RequestError` so silent data loss does not look
    like an empty completion.
    """
    model = data.get("model")
    if not isinstance(model, str):
        msg = f"Generate response missing string 'model' field: got {type(model).__name__}"
        raise RequestError(msg, request=request)
    text = data.get("text")
    if not isinstance(text, str):
        msg = f"Generate response missing string 'text' field: got {type(text).__name__}"
        raise RequestError(msg, request=request)
    result: GenerateResult = {
        "model": model,
        "text": text,
    }
    finish = data.get("finish_reason")
    if isinstance(finish, str):
        result["finish_reason"] = finish  # type: ignore[typeddict-item]
    usage = data.get("usage")
    if isinstance(usage, dict):
        result["usage"] = {
            "prompt_tokens": _coerce_token_count(usage.get("prompt_tokens")),
            "completion_tokens": _coerce_token_count(usage.get("completion_tokens")),
            "total_tokens": _coerce_token_count(usage.get("total_tokens")),
        }
    attempt_id = data.get("attempt_id")
    if isinstance(attempt_id, str):
        result["attempt_id"] = attempt_id
    ttft = data.get("ttft_ms")
    if isinstance(ttft, (int, float)):
        result["ttft_ms"] = float(ttft)
    tpot = data.get("tpot_ms")
    if isinstance(tpot, (int, float)):
        result["tpot_ms"] = float(tpot)
    return result


async def _handle_oom_retry(
    response: _AioResponse,
    *,
    start_time: float,
    oom_retries: int,
    max_oom_retries: int,
    timeout: float,  # noqa: ASYNC109 — provision_timeout_s budget, not a per-call timeout
    model: str,
) -> int:
    """Async sibling of sync ``_handle_oom_retry``.

    Sleeps through one ``RESOURCE_EXHAUSTED`` retry and returns the next
    ``oom_retries`` counter, or raises ``ResourceExhaustedError`` when the
    retry budget / ``provision_timeout_s`` budget is exhausted, *or* when
    the next backoff would consume the rest of the budget without
    leaving room for the retried request to run. The latter guard
    surfaces sustained OOM as ``ResourceExhaustedError`` rather than
    letting the outer loop's ``remaining <= 0`` branch raise
    ``ProvisioningError`` and mask the root cause.

    See ``sync._handle_oom_retry`` for full rationale.
    """
    elapsed = time.monotonic() - start_time
    if oom_retries >= max_oom_retries or elapsed >= timeout:
        msg = f"Server resource exhausted after {oom_retries} retry attempt(s) for model '{model}'"
        raise ResourceExhaustedError(
            msg,
            model=model,
            retries=oom_retries,
            request=parse_request_metadata(response.headers),
        )
    retry_after = get_retry_after(response)
    raw_delay = compute_oom_backoff(retry_after, oom_retries)
    remaining = timeout - elapsed
    if raw_delay >= remaining:
        logger.warning(
            "Server resource exhausted; remaining budget %.1fs < next backoff %.1fs (attempt %d/%d, elapsed: %.1fs, timeout: %.1fs)",
            remaining,
            raw_delay,
            oom_retries + 1,
            max_oom_retries,
            elapsed,
            timeout,
        )
        msg = f"Server resource exhausted after {oom_retries} retry attempt(s) for model '{model}'"
        raise ResourceExhaustedError(
            msg,
            model=model,
            retries=oom_retries,
            request=parse_request_metadata(response.headers),
        )
    delay = raw_delay
    # First retry surfaces at WARNING so a user with default log level
    # can see "the SDK is retrying you" — without this they may spend
    # hours debugging "slow inference" not realising auto-retry is in
    # flight. Subsequent retries stay at INFO to avoid log spam at scale.
    log_fn = logger.warning if oom_retries == 0 else logger.info
    log_fn(
        "Server resource exhausted, retrying in %.1fs (attempt %d/%d, elapsed: %.1fs, timeout: %.1fs)",
        delay,
        oom_retries + 1,
        max_oom_retries,
        elapsed,
        timeout,
    )
    await asyncio.sleep(delay)
    return oom_retries + 1


async def _aiter_text_lines(content: aiohttp.StreamReader) -> AsyncIterator[str]:
    """Decode an aiohttp byte ``StreamReader`` into newline-stripped text lines.

    aiohttp async-iterates ``content`` as byte chunks split on newline
    boundaries (each includes the trailing newline). We decode UTF-8 and strip
    the line ending so the SSE parser sees the same per-line shape as httpx's
    ``iter_lines``.
    """
    async for raw in content:
        yield raw.decode("utf-8").rstrip("\r\n")


class _AioResponse:
    """Adapts an aiohttp response to the interface ``_shared.py`` helpers expect.

    The shared helpers (``handle_error``, ``get_retry_after``, ``get_error_code``)
    access ``.status_code``, ``.content``, ``.headers``, ``.text``, and ``.json()``
    which are synchronous on ``httpx.Response``.  This wrapper eagerly reads the
    body once and exposes the same synchronous API so the helpers work unchanged
    for both the sync (httpx) and async (aiohttp) clients.
    """

    __slots__ = ("_json_cache", "_text", "content", "headers", "status_code")

    def __init__(self, status: int, content: bytes, headers: Any) -> None:
        self.status_code = status
        self.content = content
        self.headers = headers
        self._text: str | None = None
        self._json_cache: Any = None

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = self.content.decode("utf-8", errors="replace")
        return self._text

    def json(self) -> Any:
        if self._json_cache is None:
            self._json_cache = json.loads(self.content)
        return self._json_cache


class SIEAsyncClient:
    """Async client for the Search Inference Engine.

    Async variants for all client methods.

    Args:
        base_url: Base URL of the SIE server (e.g., "http://localhost:8080").
        timeout_s: Request timeout in seconds (default: 30.0).
        api_key: Optional API key for authentication (sent as Bearer token).
        gpu: GPU type for requests (e.g., "l4", "a100-80gb"). Can be overridden per-call.
        options: Options dict for requests. Merged with per-call options (per-call wins).
        pool: Resource pool spec for isolated capacity. Created lazily on first request.
            Format: {"name": "pool-name", "gpus": {"l4": 2, "a100-40gb": 1}}.
        base_url_headers: Optional additional headers for the configured gateway
            origin. Values are copied at construction and never forwarded to a
            control-plane URL, external payload-store reference, or redirect
            target. Same-origin capability refs receive only these edge headers.

    Example:
        >>> async with SIEAsyncClient("http://localhost:8080") as client:
        ...     result = await client.encode("bge-m3", {"text": "Hello world"})
        ...     print(result["dense"].shape)
        (1024,)

        >>> # With defaults for all requests
        >>> async with SIEAsyncClient(
        ...     "http://gateway:8080",
        ...     gpu="l4",
        ...     options={"normalize": True},
        ... ) as client:
        ...     result = await client.encode("bge-m3", {"text": "Hello"})  # uses l4

        >>> # With resource pool for isolated capacity
        >>> async with SIEAsyncClient(
        ...     "http://gateway:8080",
        ...     pool={"name": "eval-bench", "gpus": {"l4": 2}},
        ... ) as client:
        ...     result = await client.encode("bge-m3", {"text": "Hello"}, gpu="eval-bench/l4")
    """

    _version_warning_logged = False

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        api_key: str | None = None,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        pool: PoolSpec | None = None,
        max_connections: int | None = None,
        max_concurrency: int | None = None,
        control_plane_url: str | None = None,
        org: str | None = None,
        base_url_headers: Mapping[str, str] | None = None,
    ) -> None:
        # Ensure msgpack-numpy hooks are installed (once per process).
        # Done lazily here instead of at module level to avoid monkey-patching
        # msgpack in processes that import sie_sdk but never use the client.
        global _NUMPY_PATCHED
        if not _NUMPY_PATCHED:
            m.patch()
            _NUMPY_PATCHED = True

        # Normalize base_url (remove trailing slash)
        self._base_url = base_url.rstrip("/")
        self._base_url_headers = copy_base_url_headers(base_url_headers)
        if self._base_url_headers and not base_url_accepts_origin_credentials(self._base_url):
            msg = "base_url_headers require an absolute https base_url without embedded credentials"
            raise ValueError(msg)
        self._timeout = timeout_s
        self._default_gpu = gpu
        self._default_options = options
        self._api_key = api_key
        # Control-plane base URL + org for the connections namespace (connector
        # auth lives on the control plane, not the keyed gateway).
        self._control_plane_url = control_plane_url.rstrip("/") if control_plane_url else None
        self._org = org

        # Multi-pool state: track in-flight creations and lease renewal tasks
        self._pools: dict[str, _PoolTaskEntry] = {}
        self._pools_lock = asyncio.Lock()

        # Legacy pool state (DEPRECATED - for backward compatibility)
        self._pool_spec = pool
        self._pool_created = False
        self._pool_lock = asyncio.Lock()
        self._lease_renewal_task: asyncio.Task[None] | None = None

        # Validate pool spec (legacy)
        if pool is not None and "name" not in pool:
            msg = "Pool spec must have 'name' key"
            raise ValueError(msg)

        # Build headers
        headers = {
            "Content-Type": MSGPACK_CONTENT_TYPE,
            "Accept": MSGPACK_CONTENT_TYPE,
            SDK_VERSION_HEADER: get_sdk_version(),
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._headers = headers.copy()

        self._max_connections = max_connections or 100
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

        # First-class batch + connector surface.
        self.jobs = _AsyncJobs(self)
        self.connections = _AsyncConnections(self)
        # OpenAI-compatible Files + Batches surface — a
        # `base_url` swap makes an `openai` batch caller work unchanged.
        self.files = _AsyncFiles(self)
        self.batches = _AsyncBatches(self)

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create the aiohttp session on first use (requires a running loop)."""
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=self._max_connections,
                limit_per_host=self._max_connections,
                keepalive_timeout=90,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                base_url=self._base_url,
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers=self._headers,
            )
        return self._session

    def __del__(self) -> None:
        """Warn if the client was not closed explicitly."""
        if not getattr(self, "_closed", True):
            warnings.warn(
                f"Unclosed {self.__class__.__name__}. Call 'await client.close()' "
                "or use 'async with' to avoid resource leaks.",
                ResourceWarning,
                stacklevel=1,
            )

    @property
    def base_url(self) -> str:
        """Return the base URL of the SIE server."""
        return self._base_url

    # ------------------------------------------------------------------
    # Low-level HTTP helpers (thin wrappers around aiohttp)
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _throttle(self) -> AsyncIterator[None]:
        """Acquire the concurrency semaphore if configured, else no-op."""
        if self._semaphore is not None:
            async with self._semaphore:
                yield
        else:
            yield

    async def _post(
        self,
        url: str,
        *,
        data: bytes | None = None,
        json_data: Any = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
        include_base_url_headers: bool = True,
    ) -> _AioResponse:
        kw: dict[str, Any] = {}
        if data is not None:
            kw["data"] = data
        if json_data is not None:
            kw["json"] = json_data
        request_headers = self._headers_for_request(url, headers, include_base_url_headers=include_base_url_headers)
        if request_headers:
            kw["headers"] = request_headers
        if timeout_s is not None:
            kw["timeout"] = aiohttp.ClientTimeout(total=timeout_s)
        kw["allow_redirects"] = False
        async with self._throttle(), self._ensure_session().post(url, **kw) as resp:
            body = await resp.read()
            return _AioResponse(resp.status, body, resp.headers)

    async def _get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        include_base_url_headers: bool = True,
    ) -> _AioResponse:
        kw: dict[str, Any] = {}
        request_headers = self._headers_for_request(url, headers, include_base_url_headers=include_base_url_headers)
        if request_headers:
            kw["headers"] = request_headers
        kw["allow_redirects"] = False
        async with self._throttle(), self._ensure_session().get(url, **kw) as resp:
            body = await resp.read()
            return _AioResponse(resp.status, body, resp.headers)

    async def _delete(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        include_base_url_headers: bool = True,
    ) -> _AioResponse:
        kw: dict[str, Any] = {}
        request_headers = self._headers_for_request(url, headers, include_base_url_headers=include_base_url_headers)
        if request_headers:
            kw["headers"] = request_headers
        kw["allow_redirects"] = False
        async with self._throttle(), self._ensure_session().delete(url, **kw) as resp:
            body = await resp.read()
            return _AioResponse(resp.status, body, resp.headers)

    def _headers_for_request(
        self,
        url: str,
        headers: Mapping[str, str] | None,
        *,
        include_base_url_headers: bool = True,
    ) -> dict[str, str]:
        """Build detached request headers with exact-origin edge credentials."""
        merged = dict(headers or {})
        if include_base_url_headers and self._base_url_headers and request_matches_base_url_origin(self._base_url, url):
            merged.update(self._base_url_headers)
        return merged

    def _check_server_version(self, response: _AioResponse) -> None:
        if SIEAsyncClient._version_warning_logged:
            return
        server_version = response.headers.get(SERVER_VERSION_HEADER)
        if not server_version:
            return
        sdk_version = self._headers.get(SDK_VERSION_HEADER, "unknown")
        warning = check_version_skew(sdk_version, server_version)
        if warning:
            logger.warning(warning)
            SIEAsyncClient._version_warning_logged = True

    def _resolve_gpu(self, gpu: str | None) -> str | None:
        """Resolve GPU, using default if not specified."""
        return gpu if gpu is not None else self._default_gpu

    def _resolve_options(self, options: dict[str, Any] | None) -> dict[str, Any] | None:
        """Resolve options, merging with defaults (per-call takes precedence)."""
        if self._default_options is None:
            return options
        if options is None:
            return self._default_options
        # Merge: defaults first, then per-call overrides
        return {**self._default_options, **options}

    async def _resolve_pool_and_gpu(self, gpu: str | None) -> tuple[str | None, str | None]:
        """Resolve pool name and GPU type from gpu parameter.

        Handles the gpu="pool_name/gpu_type" format and ensures pool is
        created if the pool name matches our configured pool.

        Args:
            gpu: GPU string, either "pool_name/gpu_type" or just "gpu_type".

        Returns:
            Tuple of (pool_name, gpu_type) to use for routing.
        """
        resolved_gpu = self._resolve_gpu(gpu)

        # If no GPU specified but pool is configured, still use pool routing
        if resolved_gpu is None:
            if self._pool_spec:
                await self._ensure_pool_created()
                return self._pool_spec["name"], None
            return None, None

        pool_name, gpu_type = parse_gpu_param(resolved_gpu)

        # If pool name in gpu param matches our pool, ensure it's created
        if pool_name and self._pool_spec and pool_name == self._pool_spec.get("name"):
            await self._ensure_pool_created()

        return pool_name, gpu_type

    async def _ensure_pool_created(self) -> None:
        """Ensure the pool is created (lazy initialization)."""
        if self._pool_spec is None:
            return

        async with self._pool_lock:
            if self._pool_created:
                return

            pool_name = self._pool_spec["name"]
            logger.info("Creating pool '%s'", pool_name)

            # Build pool creation request
            request_body: dict[str, Any] = {"name": pool_name}
            if "gpus" in self._pool_spec:
                request_body["gpus"] = self._pool_spec["gpus"]
            if "gpu_caps" in self._pool_spec:
                request_body["gpu_caps"] = self._pool_spec["gpu_caps"]
            if "queue_pool" in self._pool_spec:
                request_body["queue_pool"] = self._pool_spec["queue_pool"]
            if "bundle" in self._pool_spec:
                request_body["bundle"] = self._pool_spec["bundle"]
            if self._pool_spec.get("minimum_worker_count") is not None:
                request_body["minimum_worker_count"] = self._pool_spec["minimum_worker_count"]
            if self._pool_spec.get("pinned_models") is not None:
                request_body["pinned_models"] = self._pool_spec["pinned_models"]

            try:
                response = await self._post(
                    "/v1/pools",
                    json_data=request_body,
                    headers={"Content-Type": JSON_CONTENT_TYPE, "Accept": JSON_CONTENT_TYPE},
                )
            except (aiohttp.ClientError, OSError) as e:
                msg = f"Failed to create pool '{pool_name}': connection error: {e}"
                raise SIEConnectionError(msg) from e

            if response.status_code >= HTTP_CLIENT_ERROR:
                try:
                    data = response.json()
                    error_msg = data.get("detail", {}).get("message", str(data))
                except (ValueError, KeyError):
                    error_msg = response.text
                msg = f"Failed to create pool '{pool_name}': {error_msg}"
                raise PoolError(msg, pool_name=pool_name)

            # Pool created successfully
            data = response.json()
            state = data.get("status", {}).get("state", "unknown")
            logger.info("Pool '%s' created with state '%s'", pool_name, state)

            self._pool_created = True

            # Start lease renewal task
            await self._start_lease_renewal()

    async def _start_lease_renewal(self) -> None:
        """Start the async lease renewal task."""
        if self._pool_spec is None or self._lease_renewal_task is not None:
            return

        self._lease_renewal_task = asyncio.create_task(
            self._lease_renewal_loop(),
            name=f"pool-lease-{self._pool_spec['name']}",
        )
        logger.debug("Started lease renewal task for pool '%s'", self._pool_spec["name"])

    async def _lease_renewal_loop(self) -> None:
        """Async task loop to renew pool lease."""
        if self._pool_spec is None:
            return

        pool_name = self._pool_spec["name"]

        while True:
            try:
                await asyncio.sleep(DEFAULT_LEASE_RENEWAL_INTERVAL_S)
            except asyncio.CancelledError:
                logger.debug("Lease renewal task cancelled for pool '%s'", pool_name)
                return
            last_error: Exception | None = None
            for attempt in range(_LEASE_RENEWAL_MAX_RETRIES):
                try:
                    response = await self._post(
                        f"/v1/pools/{pool_name}/renew",
                        headers={"Accept": JSON_CONTENT_TYPE},
                    )
                    if response.status_code >= HTTP_CLIENT_ERROR:
                        logger.warning(
                            "Failed to renew lease for pool '%s': HTTP %d (attempt %d/%d)",
                            pool_name,
                            response.status_code,
                            attempt + 1,
                            _LEASE_RENEWAL_MAX_RETRIES,
                        )
                    else:
                        logger.debug("Renewed lease for pool '%s'", pool_name)
                        break
                except asyncio.CancelledError:
                    logger.debug("Lease renewal task cancelled for pool '%s'", pool_name)
                    return
                except (aiohttp.ClientError, OSError) as e:
                    last_error = e
                    logger.warning(
                        "Error renewing lease for pool '%s': %s (attempt %d/%d)",
                        pool_name,
                        e,
                        attempt + 1,
                        _LEASE_RENEWAL_MAX_RETRIES,
                    )
                try:
                    await asyncio.sleep(min(2.0**attempt, 10.0))
                except asyncio.CancelledError:
                    return
            else:
                if last_error:
                    logger.error(
                        "All %d lease renewal attempts failed for pool '%s': %s",
                        _LEASE_RENEWAL_MAX_RETRIES,
                        pool_name,
                        last_error,
                    )

    async def _cleanup_pool(self) -> None:
        """Cleanup legacy pool resources on client close."""
        # Cancel legacy lease renewal task
        if self._lease_renewal_task is not None:
            self._lease_renewal_task.cancel()
            try:
                await self._lease_renewal_task
            except asyncio.CancelledError:
                pass  # Task cancelled, expected
            self._lease_renewal_task = None

    async def _cleanup_all_pools(self) -> None:
        """Cleanup all pool lease renewal tasks."""
        # Cancel all new-style pool tasks
        async with self._pools_lock:
            for _pool_name, entry in list(self._pools.items()):
                if isinstance(entry, _PoolCreationInFlight):
                    entry.future.cancel()
                    continue
                entry.cancel()
                try:
                    await entry
                except asyncio.CancelledError:
                    pass
            self._pools.clear()

        # Also cleanup legacy pool
        await self._cleanup_pool()

    async def create_pool(
        self,
        name: str,
        gpus: dict[str, int] | None = None,
        gpu_caps: dict[str, int] | None = None,
        bundle: str | None = None,
        minimum_worker_count: int | None = None,
        pinned_models: list[str] | None = None,
        *,
        queue_pool: str | None = None,
    ) -> None:
        """Create or update a resource pool for isolated capacity.

        Args:
            name: Pool name (used in gpu="pool_name/machine_profile" routing).
                The gateway stores and routes pool names in lowercase.
            gpus: Optional machine profile requirements for pool readiness, e.g.,
                {"l4": 2, "l4-spot": 1}.
            gpu_caps: Optional maximum assigned workers per machine profile, e.g.,
                {"l4": 4}. If omitted, all matching workers can be assigned.
            bundle: Optional bundle filter. When set, only workers running this
                bundle will be assigned to the pool.
            minimum_worker_count: Per-pool warm floor (minimum machines kept warm).
                The gateway emits canonical ``sie.gateway.pool.warm_floor``
                telemetry; the collector exposes ``sie_gateway_pool_warm_floor``
                to KEDA. Defaults to 0 (scale to zero).
            pinned_models: Optional set of model ids to keep loaded so the first
                request to them pays no cold model-load. Each id must be a model the
                gateway already tracks and may be profile-qualified
                (``model-name:profile_name``); unknown ids are rejected. Defaults to none.
            queue_pool: Optional Helm/NATS queue namespace backing this logical
                pool. Defaults to "default", drawing from base capacity.

        Raises:
            PoolError: If pool creation fails (e.g., invalid machine profile).
            SIEConnectionError: If unable to connect to the server.
        """
        if minimum_worker_count is not None and minimum_worker_count < 0:
            msg = "minimum_worker_count must be >= 0"
            raise ValueError(msg)

        creation: _PoolCreationInFlight | None = None
        wait_for_creation: asyncio.Future[None] | None = None
        reserved_slot = False
        already_tracking = False
        async with self._pools_lock:
            existing_entry = self._pools.get(name)
            if isinstance(existing_entry, _PoolCreationInFlight):
                logger.debug("Pool '%s' creation already in flight", name)
                wait_for_creation = existing_entry.future
            else:
                already_tracking = existing_entry is not None
                if not already_tracking:
                    creation = _PoolCreationInFlight(asyncio.get_running_loop().create_future())
                    self._pools[name] = creation
                    reserved_slot = True

        if wait_for_creation is not None:
            await asyncio.shield(wait_for_creation)
            return

        assert creation is not None or not reserved_slot

        logger.info(
            "Creating/updating pool '%s' with gpus=%s, gpu_caps=%s, bundle=%s",
            name,
            gpus,
            gpu_caps,
            bundle,
        )

        request_body: dict[str, Any] = {"name": name}
        if gpus is not None:
            request_body["gpus"] = gpus
        if gpu_caps is not None:
            request_body["gpu_caps"] = gpu_caps
        if queue_pool:
            request_body["queue_pool"] = queue_pool
        if bundle:
            request_body["bundle"] = bundle
        if minimum_worker_count is not None:
            request_body["minimum_worker_count"] = minimum_worker_count
        if pinned_models is not None:
            request_body["pinned_models"] = pinned_models

        try:
            response = await self._post(
                "/v1/pools",
                json_data=request_body,
                headers={"Content-Type": JSON_CONTENT_TYPE, "Accept": JSON_CONTENT_TYPE},
            )
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to create pool '{name}': connection error: {e}"
            error = SIEConnectionError(msg)
            if reserved_slot and creation is not None:
                async with self._pools_lock:
                    if self._pools.get(name) is creation:
                        self._pools.pop(name, None)
                _fail_pool_creation(creation.future, error)
            raise error from e
        except asyncio.CancelledError as e:
            if reserved_slot and creation is not None:
                async with self._pools_lock:
                    if self._pools.get(name) is creation:
                        self._pools.pop(name, None)
                _fail_pool_creation(creation.future, e)
            raise

        if response.status_code >= HTTP_CLIENT_ERROR:
            try:
                data = response.json()
                error_msg = data.get("detail", {}).get("message", str(data))
            except (ValueError, KeyError):
                error_msg = response.text
            msg = f"Failed to create pool '{name}': {error_msg}"
            error = PoolError(msg, pool_name=name)
            if reserved_slot and creation is not None:
                async with self._pools_lock:
                    if self._pools.get(name) is creation:
                        self._pools.pop(name, None)
                _fail_pool_creation(creation.future, error)
            raise error

        data = response.json()
        state = data.get("status", {}).get("state", "unknown")
        logger.info("Pool '%s' created/updated with state '%s'", name, state)

        # Start lease renewal task for this pool if this client is not
        # already tracking it. Repeated create_pool calls intentionally still
        # POST so callers can update gpus/gpu_caps on the gateway.
        if reserved_slot and creation is not None:
            try:
                await self._start_pool_lease_renewal(name, creation)
            except asyncio.CancelledError as e:
                async with self._pools_lock:
                    if self._pools.get(name) is creation:
                        self._pools.pop(name, None)
                _fail_pool_creation(creation.future, e)
                raise
            _complete_pool_creation(creation.future)

    async def _start_pool_lease_renewal(self, pool_name: str, creation: _PoolCreationInFlight) -> None:
        """Start lease renewal task for a pool."""
        async with self._pools_lock:
            if self._pools.get(pool_name) is creation:
                task = asyncio.create_task(
                    self._pool_lease_renewal_loop(pool_name),
                    name=f"pool-lease-{pool_name}",
                )
                self._pools[pool_name] = task
            else:
                return
        logger.debug("Started lease renewal task for pool '%s'", pool_name)

    async def _pool_lease_renewal_loop(self, pool_name: str) -> None:
        """Async task loop to renew pool lease."""
        while True:
            try:
                await asyncio.sleep(DEFAULT_LEASE_RENEWAL_INTERVAL_S)
            except asyncio.CancelledError:
                logger.debug("Lease renewal task cancelled for pool '%s'", pool_name)
                return
            last_error: Exception | None = None
            for attempt in range(_LEASE_RENEWAL_MAX_RETRIES):
                try:
                    response = await self._post(
                        f"/v1/pools/{pool_name}/renew",
                        headers={"Accept": JSON_CONTENT_TYPE},
                    )
                    if response.status_code >= HTTP_CLIENT_ERROR:
                        logger.warning(
                            "Failed to renew lease for pool '%s': HTTP %d (attempt %d/%d)",
                            pool_name,
                            response.status_code,
                            attempt + 1,
                            _LEASE_RENEWAL_MAX_RETRIES,
                        )
                    else:
                        logger.debug("Renewed lease for pool '%s'", pool_name)
                        break
                except asyncio.CancelledError:
                    logger.debug("Lease renewal task cancelled for pool '%s'", pool_name)
                    return
                except (aiohttp.ClientError, OSError) as e:
                    last_error = e
                    logger.warning(
                        "Error renewing lease for pool '%s': %s (attempt %d/%d)",
                        pool_name,
                        e,
                        attempt + 1,
                        _LEASE_RENEWAL_MAX_RETRIES,
                    )
                try:
                    await asyncio.sleep(min(2.0**attempt, 10.0))
                except asyncio.CancelledError:
                    return
            else:
                if last_error:
                    logger.error(
                        "All %d lease renewal attempts failed for pool '%s': %s",
                        _LEASE_RENEWAL_MAX_RETRIES,
                        pool_name,
                        last_error,
                    )

    async def get_pool(self, name: str | None = None) -> PoolInfo | None:
        """Get information about a pool.

        Args:
            name: Pool name to look up. If None, uses the legacy constructor pool.

        Returns:
            PoolInfo if pool exists, None otherwise.
        """
        # Determine pool name (new API or legacy)
        if name is not None:
            pool_name = name
        elif self._pool_spec is not None:
            pool_name = self._pool_spec["name"]
        else:
            return None

        try:
            response = await self._get(
                f"/v1/pools/{pool_name}",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to get pool '{pool_name}': connection error: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code == 404:
            return None

        if response.status_code >= HTTP_CLIENT_ERROR:
            try:
                data = response.json()
                detail = data.get("detail", {})
                if isinstance(detail, str):
                    error_msg = detail
                elif isinstance(detail, dict):
                    error_msg = detail.get("message", str(data))
                else:
                    error_msg = str(data)
            except (ValueError, KeyError):
                error_msg = response.text
            msg = f"Failed to get pool '{pool_name}': {error_msg}"
            raise PoolError(msg, pool_name=pool_name)

        data = response.json()
        return PoolInfo(
            name=data.get("name", pool_name),
            spec=data.get("spec", {}),
            status=data.get("status", {}),
        )

    async def delete_pool(self, name: str | None = None) -> bool:
        """Delete a pool.

        Args:
            name: Pool name to delete. If None, uses the legacy constructor pool.

        Returns:
            True if pool was deleted, False if pool didn't exist.
        """
        # Determine pool name (new API or legacy)
        if name is not None:
            pool_name = name
        elif self._pool_spec is not None:
            pool_name = self._pool_spec["name"]
        else:
            return False

        # Stop lease renewal task for this pool. If creation is in flight,
        # wait for that POST to settle before issuing DELETE so it cannot
        # recreate the pool after deletion.
        entry: _PoolTaskEntry | None = None
        async with self._pools_lock:
            current_entry = self._pools.get(pool_name)
            if isinstance(current_entry, _PoolCreationInFlight):
                entry = current_entry
            else:
                entry = self._pools.pop(pool_name, None)

        if isinstance(entry, _PoolCreationInFlight):
            with contextlib.suppress(asyncio.CancelledError, PoolError, SIEConnectionError):
                await asyncio.shield(entry.future)
            async with self._pools_lock:
                entry = self._pools.pop(pool_name, None)

        if isinstance(entry, asyncio.Task):
            entry.cancel()
            try:
                await entry
            except asyncio.CancelledError:
                pass

        # Also handle legacy pool cleanup if this is the legacy pool
        if self._pool_spec is not None and pool_name == self._pool_spec.get("name"):
            await self._cleanup_pool()

        try:
            response = await self._delete(
                f"/v1/pools/{pool_name}",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to delete pool '{pool_name}': connection error: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code == 404:
            if self._pool_spec is not None and pool_name == self._pool_spec.get("name"):
                self._pool_created = False
            return False

        if response.status_code >= HTTP_CLIENT_ERROR:
            try:
                data = response.json()
                error_msg = data.get("detail", {}).get("message", str(data))
            except (ValueError, KeyError):
                error_msg = response.text
            msg = f"Failed to delete pool '{pool_name}': {error_msg}"
            raise PoolError(msg, pool_name=pool_name)

        if self._pool_spec is not None and pool_name == self._pool_spec.get("name"):
            self._pool_created = False
        logger.info("Deleted pool '%s'", pool_name)
        return True

    async def close(self) -> None:
        """Close the HTTP session and cleanup pool resources."""
        await self._cleanup_all_pools()
        if self._session is not None:
            await self._session.close()
        self._closed = True

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager."""
        await self.close()

    # Use overload for proper type hints when single item vs list
    @overload
    async def encode(
        self,
        model: str,
        items: Item,
        *,
        output_types: list[OutputType] | None = None,
        instruction: str | None = None,
        output_dtype: str | None = None,
        is_query: bool | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> EncodeResult: ...

    @overload
    async def encode(
        self,
        model: str,
        items: list[Item],
        *,
        output_types: list[OutputType] | None = None,
        instruction: str | None = None,
        output_dtype: str | None = None,
        is_query: bool | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> list[EncodeResult]: ...

    async def encode(
        self,
        model: str,
        items: Item | list[Item],
        *,
        output_types: list[OutputType] | None = None,
        instruction: str | None = None,
        output_dtype: str | None = None,
        is_query: bool | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> EncodeResult | list[EncodeResult]:
        """Async version of encode(). See SIEClient.encode() for details."""
        # Track if single item was passed
        single_item = not isinstance(items, list)
        items_list = [items] if single_item else items

        # Convert images to JPEG bytes for transport.
        # Only copy items that have images — text-only items are passed through directly
        items_for_wire = [
            convert_item_images({**item}) if "images" in item else item  # ty: ignore[invalid-argument-type]
            for item in items_list
        ]

        # Build request body
        request_body: dict[str, Any] = {"items": items_for_wire}

        # Resolve defaults and pool
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)
        resolved_options = self._resolve_options(options)

        # Merge is_query into options if provided
        if is_query is not None:
            if resolved_options is None:
                resolved_options = {"is_query": is_query}
            else:
                resolved_options = {**resolved_options, "is_query": is_query}

        # Add params if any are non-default
        params: dict[str, Any] = {}
        if output_types is not None:
            params["output_types"] = output_types
        if instruction is not None:
            params["instruction"] = instruction
        if output_dtype is not None:
            params["output_dtype"] = output_dtype
        if resolved_options is not None:
            params["options"] = resolved_options
        if params:
            request_body["params"] = params

        # Serialize with msgpack
        body = msgpack.packb(request_body, use_bin_type=True)

        # Build headers with optional GPU and pool routing
        headers: dict[str, str] = {}
        if resolved_gpu:
            headers["X-SIE-MACHINE-PROFILE"] = resolved_gpu
        if pool_name:
            headers["X-SIE-Pool"] = pool_name

        # Set up provisioning timeout
        timeout = provision_timeout_s if provision_timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()

        # Local retry counter for LoRA loading (model loading uses time-based timeout only)
        lora_retries = 0
        # Retry counter for server-side OOM (RESOURCE_EXHAUSTED).
        oom_retries = 0

        # Retry loop for retryable provisioning/capacity responses.
        while True:
            # Compute per-request timeout: cap to remaining provision time
            # This ensures a single hanging request can't exceed the overall timeout
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                msg = f"Provision timeout ({timeout:.1f}s) exceeded before request could be sent"
                raise ProvisioningError(msg, gpu=resolved_gpu)
            request_timeout = min(self._timeout, remaining)

            try:
                response = await self._post(
                    f"/v1/encode/{model}",
                    data=body,
                    headers=headers,
                    timeout_s=request_timeout,
                )
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                if wait_for_capacity:
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Transient transport error",
                        error=e,
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                if isinstance(e, TimeoutError):
                    msg = f"Request timed out: {e}"
                else:
                    msg = (
                        f"Connection lost mid-request ({type(e).__name__}); "
                        f"the peer closed the connection before sending a complete response: {e}"
                    )
                raise SIEConnectionError(msg) from e
            except aiohttp.ClientConnectorError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e

            # Short-circuit terminal load failures (sie-test#85).
            raise_if_model_load_failed(response, model=model)

            # Handle 503 with LORA_LOADING or MODEL_LOADING - auto-retry
            if response.status_code == 503:
                error_code = get_error_code(response)
                if error_code == PROVISIONING_ERROR_CODE:
                    actual_delay = provisioning_retry_delay(
                        response,
                        gpu=resolved_gpu,
                        wait_for_capacity=wait_for_capacity,
                        start_time=start_time,
                        timeout=timeout,
                    )
                    logger.debug(
                        "Provisioning in progress, retrying in %.1fs (timeout: %.1fs)",
                        actual_delay,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

                if error_code == LORA_LOADING_ERROR_CODE:
                    lora_retries += 1

                    if lora_retries > LORA_LOADING_MAX_RETRIES:
                        # Extract lora from options for error message
                        lora_name = resolved_options.get("lora") if resolved_options else None
                        msg = f"LoRA loading timeout after {lora_retries} retries"
                        raise LoraLoadingError(msg, lora=str(lora_name) if lora_name else None, model=model)

                    # Wait and retry
                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, LORA_LOADING_DEFAULT_DELAY_S)
                    logger.debug(
                        "LoRA loading, retrying in %.1fs (attempt %d/%d)",
                        delay,
                        lora_retries,
                        LORA_LOADING_MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue

                if error_code == MODEL_LOADING_ERROR_CODE:
                    # Check if we've exceeded the provision timeout
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)

                    # Wait and retry, respecting remaining time
                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Model loading in progress, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

                if error_code == RESOURCE_EXHAUSTED_ERROR_CODE:
                    oom_retries = await _handle_oom_retry(
                        response,
                        start_time=start_time,
                        oom_retries=oom_retries,
                        max_oom_retries=max_oom_retries,
                        timeout=timeout,
                        model=model,
                    )
                    continue

            # Handle 504 (gateway timeout): queued work was published, but the
            # gateway did not receive a worker result before its deadline.
            # Encode/score/extract are idempotent, so callers that opted into
            # wait_for_capacity can retry within provision_timeout_s.
            if response.status_code == HTTP_GATEWAY_TIMEOUT and wait_for_capacity:
                elapsed = time.monotonic() - start_time
                if elapsed < timeout:
                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Gateway timeout (504), retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

            # Handle errors
            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)

            # Success - break out of retry loop
            break

        self._check_server_version(response)

        # Deserialize response
        response_data = msgpack.unpackb(response.content, raw=False)

        # Get timing info if present
        timing = response_data.get("timing")

        # Parse results and inject timing into each
        results = parse_encode_results(response_data["items"])
        # Guard the 1:1 input↔output contract before any positional access
        # (``results[0]`` below, or batch reassembly in callers). A desynced
        # count otherwise surfaces as a context-free ``IndexError`` (#1526).
        validate_encode_result_count(
            results,
            len(items_list),
            model,
            request=parse_request_metadata(response.headers),
        )
        if timing:
            for result in results:
                result["timing"] = timing

        attach_request_metadata(results, response.headers)

        # Return single result if single item was passed
        return results[0] if single_item else results

    async def list_models(self) -> list[ModelInfo]:
        """Async version of list_models(). See SIEClient.list_models() for details."""
        try:
            response = await self._get(
                "/v1/models",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except TimeoutError as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to connect to {self._base_url}: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)

        data = response.json()
        return data["models"]

    async def get_model(self, model: str) -> ModelInfo:
        """Async version of get_model(). See SIEClient.get_model() for details."""
        try:
            response = await self._get(
                f"/v1/models/{model}",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except TimeoutError as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to connect to {self._base_url}: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)

        return response.json()

    async def _detect_endpoint_type(self) -> Literal["cluster", "worker"]:
        """Detect whether base_url is a gateway (cluster) or worker endpoint."""
        try:
            response = await self._get(
                "/health",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except (aiohttp.ClientError, OSError):
            return "worker"

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                return "worker"
            if isinstance(payload, dict) and payload.get("type") == "gateway":
                return "cluster"
        return "worker"

    def _ws_url(self, path: str) -> str:
        """Build websocket URL from base_url."""
        if self._base_url.startswith("https://"):
            scheme = "wss://"
            rest = self._base_url[len("https://") :]
        elif self._base_url.startswith("http://"):
            scheme = "ws://"
            rest = self._base_url[len("http://") :]
        else:
            scheme = "ws://"
            rest = self._base_url
        return f"{scheme}{rest}{path}"

    def _websocket_headers(self, websocket_url: str) -> dict[str, str]:
        """Build edge-confined headers for one WebSocket handshake."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._base_url_headers and websocket_matches_base_url_origin(self._base_url, websocket_url):
            headers.update(self._base_url_headers)
        return headers

    async def watch(
        self,
        *,
        mode: Literal["auto", "cluster", "worker"] = "auto",
    ) -> AsyncIterator[StatusMessage]:
        """Stream real-time status updates from the server or gateway."""
        import websockets

        class _NoRedirectConnect(websockets.connect):
            """WebSocket connector that never forwards edge credentials to redirects."""

            def process_redirect(self, exc: Exception) -> Exception:
                return exc

        if mode == "auto":
            detected = await self._detect_endpoint_type()
            paths = ["/ws/cluster-status"] if detected == "cluster" else ["/ws/status"]
        elif mode == "cluster":
            paths = ["/ws/cluster-status"]
        else:
            paths = ["/ws/status"]

        for path in paths:
            ws_url = self._ws_url(path)
            try:
                async with _NoRedirectConnect(
                    ws_url,
                    additional_headers=self._websocket_headers(ws_url),
                ) as ws:
                    async for message in ws:
                        if isinstance(message, bytes):
                            payload = message.decode("utf-8")
                        else:
                            payload = message
                        data = json.loads(payload)
                        yield data
                return
            except websockets.exceptions.InvalidStatus as e:
                raise RequestError(f"WebSocket connection failed: {e.response.status_code}") from e
            except (websockets.WebSocketException, OSError, json.JSONDecodeError) as e:
                raise SIEConnectionError(f"WebSocket error: {e}") from e

    async def get_capacity(self, *, gpu: str | None = None) -> CapacityInfo:
        """Async version of get_capacity(). See SIEClient.get_capacity() for details."""
        try:
            response = await self._get(
                "/health",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except TimeoutError as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to connect to {self._base_url}: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)

        data = response.json()

        # Check if this is a gateway (has 'type': 'gateway') or worker
        if data.get("type") != "gateway":
            msg = "get_capacity() requires a gateway endpoint. This appears to be a worker."
            raise RequestError(msg, code="not_gateway", status_code=400)

        # Build CapacityInfo
        cluster = data.get("cluster", {})
        workers_data = data.get("workers", [])

        # Filter by GPU if specified
        if gpu:
            gpu_lower = gpu.lower()
            workers_data = [w for w in workers_data if w.get("gpu", "").lower() == gpu_lower]

        workers: list[WorkerInfo] = [
            WorkerInfo(
                name=w.get("name", ""),
                url=w.get("url", ""),
                gpu=w.get("gpu", ""),
                gpu_count=w.get("gpu_count", 0),
                ready_gpu_slots=w.get(
                    "ready_gpu_slots",
                    w.get("gpu_count", 1 if w.get("healthy", False) else 0),
                ),
                healthy=w.get("healthy", False),
                queue_depth=w.get("queue_depth", 0),
                pending_cost=w.get("pending_cost", 0),
                inflight_batches=w.get("inflight_batches", 0),
                loaded_models=w.get("loaded_models", []),
                memory_used_bytes=w.get("memory_used_bytes", 0),
                memory_total_bytes=w.get("memory_total_bytes", 0),
                bundle=w.get("bundle", ""),
                bundle_config_hash=w.get("bundle_config_hash", ""),
            )
            for w in workers_data
        ]

        return CapacityInfo(
            status=data.get("status", "unknown"),
            worker_count=len(workers) if gpu else cluster.get("worker_count", 0),
            gpu_count=cluster.get("gpu_count", 0),
            models_loaded=cluster.get("models_loaded", 0),
            configured_gpu_types=data.get("configured_gpu_types", []),
            live_gpu_types=data.get("live_gpu_types", []),
            workers=workers,
        )

    async def wait_for_capacity(
        self,
        gpu: str,
        *,
        model: str | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float = 5.0,
    ) -> CapacityInfo:
        """Async version of wait_for_capacity(). See SIEClient.wait_for_capacity() for details."""
        timeout = timeout_s if timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()

        # If model is specified, use encode with wait_for_capacity to trigger
        # both scale-up and model loading
        if model:
            await self.encode(
                model,
                Item(text="warmup"),
                gpu=gpu,
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            # After successful encode, get capacity info
            return await self.get_capacity(gpu=gpu)

        # Otherwise, poll capacity until workers are available
        while True:
            try:
                capacity = await self.get_capacity(gpu=gpu)
                if capacity.get("worker_count", 0) > 0:
                    return capacity
            except (SIEConnectionError, RequestError):
                pass  # Keep trying

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                msg = f"Timeout after {elapsed:.1f}s waiting for GPU '{gpu}' capacity"
                raise ProvisioningError(msg, gpu=gpu)

            # Wait before next poll
            remaining = timeout - elapsed
            delay = min(poll_interval_s, remaining)
            await asyncio.sleep(delay)

    async def score(
        self,
        model: str,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> ScoreResult:
        """Score items against a query using a reranker model.

        Async version of :meth:`SIEClient.score`. See that method for full
        parameter documentation.
        """
        # Resolve defaults and pool
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)
        resolved_options = self._resolve_options(options)
        query_for_wire, items_for_wire = convert_score_images_for_wire(query, items)

        # Build request body
        request_body: dict[str, Any] = {
            "query": query_for_wire,
            "items": items_for_wire,
        }
        if instruction is not None:
            request_body["instruction"] = instruction
        if resolved_options is not None:
            request_body["options"] = resolved_options

        # Serialize with msgpack
        body = msgpack.packb(request_body, use_bin_type=True)

        # Build headers with optional GPU and pool routing
        headers: dict[str, str] = {}
        if resolved_gpu:
            headers["X-SIE-MACHINE-PROFILE"] = resolved_gpu
        if pool_name:
            headers["X-SIE-Pool"] = pool_name

        # Set up provisioning timeout
        timeout = provision_timeout_s if provision_timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()

        # Model loading uses time-based timeout only (no retry counter)
        # OOM retry counter (RESOURCE_EXHAUSTED) — bounded with exponential backoff.
        oom_retries = 0

        # Retry loop for retryable provisioning/capacity responses.
        while True:
            # Compute per-request timeout: cap to remaining provision time
            # This ensures a single hanging request can't exceed the overall timeout
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                msg = f"Provision timeout ({timeout:.1f}s) exceeded before request could be sent"
                raise ProvisioningError(msg, gpu=resolved_gpu)
            request_timeout = min(self._timeout, remaining)

            try:
                response = await self._post(
                    f"/v1/score/{model}",
                    data=body,
                    headers=headers,
                    timeout_s=request_timeout,
                )
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                if wait_for_capacity:
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Transient transport error",
                        error=e,
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                if isinstance(e, TimeoutError):
                    msg = f"Request timed out: {e}"
                else:
                    msg = (
                        f"Connection lost mid-request ({type(e).__name__}); "
                        f"the peer closed the connection before sending a complete response: {e}"
                    )
                raise SIEConnectionError(msg) from e
            except aiohttp.ClientConnectorError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e

            # Short-circuit terminal load failures (sie-test#85).
            raise_if_model_load_failed(response, model=model)

            # Handle 503 with MODEL_LOADING - auto-retry
            if response.status_code == 503:
                error_code = get_error_code(response)
                if error_code == PROVISIONING_ERROR_CODE:
                    actual_delay = provisioning_retry_delay(
                        response,
                        gpu=resolved_gpu,
                        wait_for_capacity=wait_for_capacity,
                        start_time=start_time,
                        timeout=timeout,
                    )
                    logger.debug(
                        "Provisioning in progress, retrying in %.1fs (timeout: %.1fs)",
                        actual_delay,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

                if error_code == MODEL_LOADING_ERROR_CODE:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)

                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Model loading in progress, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

                if error_code == RESOURCE_EXHAUSTED_ERROR_CODE:
                    oom_retries = await _handle_oom_retry(
                        response,
                        start_time=start_time,
                        oom_retries=oom_retries,
                        max_oom_retries=max_oom_retries,
                        timeout=timeout,
                        model=model,
                    )
                    continue

            # Handle 504 (gateway timeout). See encode() above for rationale.
            if response.status_code == HTTP_GATEWAY_TIMEOUT and wait_for_capacity:
                elapsed = time.monotonic() - start_time
                if elapsed < timeout:
                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Gateway timeout (504), retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)

            break

        self._check_server_version(response)

        response_data = msgpack.unpackb(response.content, raw=False)

        result = parse_score_result(response_data)
        attach_request_metadata([result], response.headers)
        return result

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        max_new_tokens: int,
        images: Sequence[ImageLike | GenerateImage | dict[str, Any]] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        grammar: GenerateGrammar | Mapping[str, Any] | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        routing_key: str | None = None,
        prompt_cache_key: str | None = None,
        safety_identifier: str | None = None,
        lora_adapter: str | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> GenerateResult:
        """Async sibling of :meth:`SIEClient.generate`.

        See the sync docstring for parameter semantics. This method
        awaits the aggregated outcome; use :meth:`stream_generate` for
        SIE-native chunk streaming.
        """
        resolved_grammar = validate_generate_grammar(grammar) if grammar is not None else None
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)

        safe_model = model.replace("/", "__")

        resolved_options = self._resolve_options(options)
        request_body: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        if images is not None:
            request_body["images"] = convert_images_for_json(images)
        if stop is not None:
            request_body["stop"] = stop
        optional_fields = {
            "temperature": temperature,
            "top_p": top_p,
            "options": resolved_options,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "grammar": resolved_grammar,
            "seed": seed,
            "logit_bias": logit_bias,
            "routing_key": routing_key,
            "prompt_cache_key": prompt_cache_key,
            "safety_identifier": safety_identifier,
            "lora_adapter": lora_adapter,
        }
        request_body.update({key: value for key, value in optional_fields.items() if value is not None})
        validate_generate_request_body(request_body)

        body = json.dumps(request_body).encode("utf-8")
        headers: dict[str, str] = {"content-type": JSON_CONTENT_TYPE, "accept": JSON_CONTENT_TYPE}
        if resolved_gpu:
            headers["X-SIE-MACHINE-PROFILE"] = resolved_gpu
        if pool_name:
            headers["X-SIE-Pool"] = pool_name

        timeout = provision_timeout_s if provision_timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()
        oom_retries = 0

        while True:
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                msg = f"Provision timeout ({timeout:.1f}s) exceeded before request could be sent"
                raise ProvisioningError(msg, gpu=resolved_gpu)
            request_timeout_s = min(self._timeout, remaining)

            try:
                async with (
                    self._throttle(),
                    self._ensure_session().post(
                        f"/v1/generate/{safe_model}",
                        data=body,
                        headers=self._headers_for_request(f"/v1/generate/{safe_model}", headers),
                        timeout=aiohttp.ClientTimeout(total=request_timeout_s),
                        allow_redirects=False,
                    ) as raw,
                ):
                    content = await raw.read()
                    response = _AioResponse(raw.status, content, raw.headers)
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                # Generation is NOT idempotent and carries no dedup key.
                # These errors fire after the request body was sent (read
                # timeout, peer reset, payload error), so the worker may
                # already be — or have finished — generating. Retrying would
                # issue a *second* billable generation with a different
                # completion, so surface the error instead of re-running.
                # (The idempotent encode/score/extract paths still retry.)
                msg = f"Request failed: {e}"
                raise SIEConnectionError(msg) from e
            except aiohttp.ClientConnectorError as e:
                # Connect errors fail before the request is sent, so no
                # generation could have started — safe to retry.
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                # Catch-all transport failure. For the non-idempotent
                # generate path we do NOT retry: unlike a pure connect
                # failure (ClientConnectorError above), a generic
                # ClientError/OSError (e.g. ECONNRESET) can fire after the
                # request was sent, so the worker may already have generated.
                # Retrying would double-bill an inference. Surface instead.
                msg = f"Request failed: {e}"
                raise SIEConnectionError(msg) from e

            raise_if_model_load_failed(response, model=model)

            if response.status_code == 503:
                error_code = get_error_code(response)
                if error_code == PROVISIONING_ERROR_CODE:
                    delay = provisioning_retry_delay(
                        response,
                        gpu=resolved_gpu,
                        wait_for_capacity=wait_for_capacity,
                        start_time=start_time,
                        timeout=timeout,
                    )
                    await asyncio.sleep(delay)
                    continue

                if error_code == MODEL_LOADING_ERROR_CODE:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)
                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
                    await asyncio.sleep(min(delay, timeout - elapsed))
                    continue
                if error_code == RESOURCE_EXHAUSTED_ERROR_CODE:
                    oom_retries = await _handle_oom_retry(
                        response,
                        start_time=start_time,
                        oom_retries=oom_retries,
                        max_oom_retries=max_oom_retries,
                        timeout=timeout,
                        model=model,
                    )
                    continue

            # Do NOT retry 504 here. Unlike the idempotent encode/score/extract
            # paths (which keep the 504 retry block), generation is NOT
            # idempotent and carries no dedup key. A 504 GATEWAY_TIMEOUT is a
            # *post-publish* timeout: the work item is already on the queue and
            # a worker may be — or have finished — generating. Retrying would
            # issue a SECOND billable generation with a different completion, so
            # surface it as a terminal ServerError instead (same reasoning as
            # the mid-flight transport-error block above). The pre-execution
            # 503 MODEL_LOADING / PROVISIONING retries above remain because
            # those fire *before* any generation can have started.
            if response.status_code == HTTP_GATEWAY_TIMEOUT:
                msg = (
                    "Gateway timed out (504) after the generate request was published to the "
                    "queue; a worker may already be generating. Not retried because generation "
                    "is non-idempotent (retrying could double-bill). Re-issue manually if needed."
                )
                raise ServerError(
                    msg,
                    code=get_error_code(response),
                    status_code=response.status_code,
                    request=parse_request_metadata(response.headers),
                )

            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)
            break

        self._check_server_version(response)

        data = parse_terminal_json_object(response, owner="generate")
        result = _parse_generate_result_async(data, request=parse_request_metadata(response.headers))
        attach_request_metadata([result], response.headers)
        return result

    async def responses(
        self,
        model: str,
        input: str | Sequence[ResponseInputMessage],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> ResponseResult:
        """Async counterpart of :meth:`SIEClient.responses`.

        The gateway's Responses MVP is stateless, text-only, and
        non-streaming. Retry behavior matches the sync method: only explicit
        pre-execution capacity signals and connect-before-send failures are
        retried; mid-flight failures and post-publish 504 responses are
        terminal because generation is non-idempotent.
        """
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)
        body = json.dumps(
            build_responses_body(
                model,
                input,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        ).encode("utf-8")
        headers: dict[str, str] = {"content-type": JSON_CONTENT_TYPE, "accept": JSON_CONTENT_TYPE}
        if resolved_gpu:
            headers["X-SIE-MACHINE-PROFILE"] = resolved_gpu
        if pool_name:
            headers["X-SIE-Pool"] = pool_name

        timeout = provision_timeout_s if provision_timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()
        oom_retries = 0
        while True:
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                msg = f"Provision timeout ({timeout:.1f}s) exceeded before request could be sent"
                raise ProvisioningError(msg, gpu=resolved_gpu)
            try:
                async with (
                    self._throttle(),
                    self._ensure_session().post(
                        "/v1/responses",
                        data=body,
                        headers=self._headers_for_request("/v1/responses", headers),
                        timeout=aiohttp.ClientTimeout(total=min(self._timeout, remaining)),
                        allow_redirects=False,
                    ) as raw,
                ):
                    content = await raw.read()
                    response = _AioResponse(raw.status, content, raw.headers)
            except aiohttp.ClientConnectorError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                msg = f"Request failed: {e}"
                raise SIEConnectionError(msg) from e

            if response.status_code == 200:
                break
            delay, oom_retries = next_stream_retry_delay(
                response,
                model=model,
                gpu=resolved_gpu,
                wait_for_capacity=wait_for_capacity,
                start_time=start_time,
                timeout=timeout,
                oom_retries=oom_retries,
                max_oom_retries=max_oom_retries,
            )
            await asyncio.sleep(delay)

        self._check_server_version(response)
        data = parse_terminal_json_object(response, owner="Responses")
        attach_request_metadata([data], response.headers)
        return cast("ResponseResult", data)

    async def chat_completions(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: dict[str, Any] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        n: int | None = None,
        best_of: int | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        logit_bias: dict[str, float] | None = None,
        seed: int | None = None,
        user: str | None = None,
        safety_identifier: str | None = None,
        lora_adapter: str | None = None,
        gpu: str | None = None,
        extra_body: dict[str, Any] | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> ChatCompletion:
        """Non-streaming OpenAI-compatible chat completion (``/v1/chat/completions``).

        Async counterpart of :meth:`SIEClient.chat_completions`. For token
        streaming use :meth:`stream_chat_completions`. Generation is
        non-idempotent, so only pre-execution 503 PROVISIONING / MODEL_LOADING responses are retried;
        a 504 surfaces as :class:`ServerError`.

        Typed kwargs cover the full gateway-supported field set (see
        :func:`build_chat_body` for the canonical list); ``extra_body`` is
        still merged last for forward-compat fields the typed surface does
        not name yet.
        """
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)
        body = json.dumps(
            build_chat_body(
                model,
                messages,
                stream=False,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                response_format=response_format,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=n,
                best_of=best_of,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                logit_bias=logit_bias,
                seed=seed,
                user=user,
                safety_identifier=safety_identifier,
                lora_adapter=lora_adapter,
                extra_body=extra_body,
            )
        ).encode("utf-8")
        headers: dict[str, str] = {"content-type": JSON_CONTENT_TYPE, "accept": JSON_CONTENT_TYPE}
        if resolved_gpu:
            headers["X-SIE-MACHINE-PROFILE"] = resolved_gpu
        if pool_name:
            headers["X-SIE-Pool"] = pool_name

        timeout = provision_timeout_s if provision_timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()
        oom_retries = 0
        while True:
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                msg = f"Provision timeout ({timeout:.1f}s) exceeded before request could be sent"
                raise ProvisioningError(msg, gpu=resolved_gpu)
            try:
                async with (
                    self._throttle(),
                    self._ensure_session().post(
                        "/v1/chat/completions",
                        data=body,
                        headers=self._headers_for_request("/v1/chat/completions", headers),
                        timeout=aiohttp.ClientTimeout(total=min(self._timeout, remaining)),
                        allow_redirects=False,
                    ) as raw,
                ):
                    content = await raw.read()
                    response = _AioResponse(raw.status, content, raw.headers)
            except aiohttp.ClientConnectorError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time, timeout=timeout, error_label="Connect error", error=e
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                # Non-idempotent: a mid-flight failure may already have started
                # a generation, so surface it instead of silently re-running.
                msg = f"Request failed: {e}"
                raise SIEConnectionError(msg) from e

            if response.status_code == 200:
                break
            delay, oom_retries = next_stream_retry_delay(
                response,
                model=model,
                gpu=resolved_gpu,
                wait_for_capacity=wait_for_capacity,
                start_time=start_time,
                timeout=timeout,
                oom_retries=oom_retries,
                max_oom_retries=max_oom_retries,
            )
            await asyncio.sleep(delay)

        self._check_server_version(response)
        data = parse_terminal_json_object(response, owner="chat completion")
        attach_request_metadata([data], response.headers)
        return cast("ChatCompletion", data)

    async def stream_chat_completions(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: dict[str, Any] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        n: int | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        logit_bias: dict[str, float] | None = None,
        seed: int | None = None,
        user: str | None = None,
        safety_identifier: str | None = None,
        lora_adapter: str | None = None,
        stream_options: dict[str, Any] | None = None,
        gpu: str | None = None,
        extra_body: dict[str, Any] | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Streaming OpenAI-compatible chat completion.

        Async counterpart of :meth:`SIEClient.stream_chat_completions`. Yields
        :class:`ChatCompletionChunk` events; raises :class:`ServerError` on a
        mid-stream error chunk. Breaking out of the iterator early closes the
        stream so the worker stops generating.

        ``best_of`` is intentionally not exposed on the streaming surface —
        the gateway rejects ``best_of`` together with ``stream: true``.
        """
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)
        body = json.dumps(
            build_chat_body(
                model,
                messages,
                stream=True,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                response_format=response_format,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=n,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                logit_bias=logit_bias,
                seed=seed,
                user=user,
                safety_identifier=safety_identifier,
                lora_adapter=lora_adapter,
                stream_options=stream_options,
                extra_body=extra_body,
            )
        ).encode("utf-8")
        headers = sse_headers(resolved_gpu, pool_name)
        async for chunk in self._stream_sse_chunks(
            "/v1/chat/completions",
            body,
            headers,
            model=model,
            resolved_gpu=resolved_gpu,
            wait_for_capacity=wait_for_capacity,
            provision_timeout_s=provision_timeout_s,
            max_oom_retries=max_oom_retries,
        ):
            yield chunk

    async def stream_generate(
        self,
        model: str,
        prompt: str,
        *,
        max_new_tokens: int,
        images: Sequence[ImageLike | GenerateImage | dict[str, Any]] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        grammar: GenerateGrammar | Mapping[str, Any] | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        routing_key: str | None = None,
        prompt_cache_key: str | None = None,
        safety_identifier: str | None = None,
        lora_adapter: str | None = None,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> AsyncIterator[GenerateChunk]:
        """Streaming SIE-native generation (``/v1/generate/{model}``).

        Async counterpart of :meth:`SIEClient.stream_generate`.
        """
        resolved_grammar = validate_generate_grammar(grammar) if grammar is not None else None
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)
        safe_model = model.replace("/", "__")
        resolved_options = self._resolve_options(options)
        req: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "stream": True,
        }
        if images is not None:
            req["images"] = convert_images_for_json(images)
        if stop is not None:
            req["stop"] = stop
        optional_fields = {
            "frequency_penalty": frequency_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "options": resolved_options,
            "presence_penalty": presence_penalty,
            "grammar": resolved_grammar,
            "seed": seed,
            "logit_bias": logit_bias,
            "routing_key": routing_key,
            "prompt_cache_key": prompt_cache_key,
            "safety_identifier": safety_identifier,
            "lora_adapter": lora_adapter,
        }
        req.update({key: value for key, value in optional_fields.items() if value is not None})
        if logprobs:
            req["logprobs"] = True
            if top_logprobs is not None:
                req["top_logprobs"] = top_logprobs
        if extra_body:
            req.update(extra_body)
        validate_generate_request_body(req)
        if not req.get("logprobs"):
            req.pop("top_logprobs", None)
        req.update(prompt=prompt, max_new_tokens=max_new_tokens, stream=True)
        body = json.dumps(req).encode("utf-8")
        headers = sse_headers(resolved_gpu, pool_name)
        async for chunk in self._stream_sse_chunks(
            f"/v1/generate/{safe_model}",
            body,
            headers,
            model=model,
            resolved_gpu=resolved_gpu,
            wait_for_capacity=wait_for_capacity,
            provision_timeout_s=provision_timeout_s,
            max_oom_retries=max_oom_retries,
        ):
            yield chunk

    async def _stream_sse_chunks(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        *,
        model: str,
        resolved_gpu: str | None,
        wait_for_capacity: bool,
        provision_timeout_s: float | None,
        max_oom_retries: int,
    ) -> AsyncIterator[Any]:
        """Open an aiohttp SSE stream (with pre-stream provisioning retry) and yield chunks.

        Shared by :meth:`stream_chat_completions` and :meth:`stream_generate`.
        Only the *pre-stream* response is retried (503 PROVISIONING / MODEL_LOADING); once bytes flow
        a failure is terminal (non-idempotent). The ``async with`` keeps the
        connection open while the caller consumes the generator.
        """
        timeout = provision_timeout_s if provision_timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()
        oom_retries = 0
        while True:
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                msg = f"Provision timeout ({timeout:.1f}s) exceeded before request could be sent"
                raise ProvisioningError(msg, gpu=resolved_gpu)
            retry_delay: float | None = None
            try:
                async with (
                    self._throttle(),
                    self._ensure_session().post(
                        url,
                        data=body,
                        headers=self._headers_for_request(url, headers),
                        timeout=aiohttp.ClientTimeout(total=min(self._timeout, remaining)),
                        allow_redirects=False,
                    ) as raw,
                ):
                    if raw.status != 200:
                        content = await raw.read()
                        response = _AioResponse(raw.status, content, raw.headers)
                        retry_delay, oom_retries = next_stream_retry_delay(
                            response,
                            model=model,
                            gpu=resolved_gpu,
                            wait_for_capacity=wait_for_capacity,
                            start_time=start_time,
                            timeout=timeout,
                            oom_retries=oom_retries,
                            max_oom_retries=max_oom_retries,
                        )
                    else:
                        self._check_server_version(_AioResponse(raw.status, b"", raw.headers))
                        async for payload in aiter_sse_payloads(_aiter_text_lines(raw.content)):
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError as e:
                                msg = f"Malformed SSE chunk from server: {e}"
                                raise RequestError(msg) from e
                            if isinstance(chunk, dict):
                                err = sse_chunk_error(chunk)
                                if err is not None:
                                    code, message = err
                                    raise ServerError(message, code=code)
                            yield chunk
                        return
            except aiohttp.ClientConnectorError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time, timeout=timeout, error_label="Connect error", error=e
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError, TimeoutError) as e:
                # Mid-stream/transport failure: non-idempotent, do not retry.
                msg = f"Connection lost during stream ({type(e).__name__}): {e}"
                raise SIEConnectionError(msg) from e
            # Reached only on the non-200 pre-stream retry path.
            if retry_delay is not None:
                await asyncio.sleep(retry_delay)

    # Use overload for proper type hints when single item vs list
    @overload
    async def extract(
        self,
        model: str,
        items: Item,
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> ExtractResult: ...

    @overload
    async def extract(
        self,
        model: str,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> list[ExtractResult]: ...

    async def extract(
        self,
        model: str,
        items: Item | list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        wait_for_capacity: bool = True,
        provision_timeout_s: float | None = None,
        max_oom_retries: int = RESOURCE_EXHAUSTED_MAX_RETRIES,
    ) -> ExtractResult | list[ExtractResult]:
        """Async version of extract(). See SIEClient.extract() for details."""
        # Track if single item was passed
        single_item = not isinstance(items, list)
        items_list = [items] if single_item else items

        # Convert media and documents to wire format (bytes + format hint)
        items_for_wire = []
        for item in items_list:
            wire_item: dict[str, Any] = {**item}  # ty: ignore[invalid-argument-type]
            if "images" in wire_item:
                wire_item = convert_item_images(wire_item)
            if "audio" in wire_item:
                wire_item = convert_item_audio(wire_item)
            if "document" in wire_item:
                wire_item = convert_item_document(wire_item)
            items_for_wire.append(wire_item)

        # Build request body
        request_body: dict[str, Any] = {"items": items_for_wire}

        # Resolve defaults and pool
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)
        resolved_options = self._resolve_options(options)

        # Add params if any are non-default
        params: dict[str, Any] = {}
        if labels is not None:
            params["labels"] = labels
        if output_schema is not None:
            params["output_schema"] = output_schema
        if instruction is not None:
            params["instruction"] = instruction
        if resolved_options is not None:
            params["options"] = resolved_options
        if params:
            request_body["params"] = params

        # Serialize with msgpack
        body = msgpack.packb(request_body, use_bin_type=True)

        # Build headers with optional GPU and pool routing
        headers: dict[str, str] = {}
        if resolved_gpu:
            headers["X-SIE-MACHINE-PROFILE"] = resolved_gpu
        if pool_name:
            headers["X-SIE-Pool"] = pool_name

        # Set up provisioning timeout
        timeout = provision_timeout_s if provision_timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()

        # Model loading uses time-based timeout only (no retry counter)
        # OOM retry counter (RESOURCE_EXHAUSTED) — bounded with exponential backoff.
        oom_retries = 0

        # Retry loop for retryable provisioning/capacity responses.
        while True:
            # Compute per-request timeout: cap to remaining provision time
            # This ensures a single hanging request can't exceed the overall timeout
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                msg = f"Provision timeout ({timeout:.1f}s) exceeded before request could be sent"
                raise ProvisioningError(msg, gpu=resolved_gpu)
            request_timeout = min(self._timeout, remaining)

            try:
                response = await self._post(
                    f"/v1/extract/{model}",
                    data=body,
                    headers=headers,
                    timeout_s=request_timeout,
                )
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                if wait_for_capacity:
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Transient transport error",
                        error=e,
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                if isinstance(e, TimeoutError):
                    msg = f"Request timed out: {e}"
                else:
                    msg = (
                        f"Connection lost mid-request ({type(e).__name__}); "
                        f"the peer closed the connection before sending a complete response: {e}"
                    )
                raise SIEConnectionError(msg) from e
            except aiohttp.ClientConnectorError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        await asyncio.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e

            # Short-circuit terminal load failures (sie-test#85).
            raise_if_model_load_failed(response, model=model)

            # Short-circuit token-budget overruns (#849).
            raise_if_input_too_long(response, model=model)

            # Handle 503 with MODEL_LOADING - auto-retry
            if response.status_code == 503:
                error_code = get_error_code(response)
                if error_code == PROVISIONING_ERROR_CODE:
                    actual_delay = provisioning_retry_delay(
                        response,
                        gpu=resolved_gpu,
                        wait_for_capacity=wait_for_capacity,
                        start_time=start_time,
                        timeout=timeout,
                    )
                    logger.debug(
                        "Provisioning in progress, retrying in %.1fs (timeout: %.1fs)",
                        actual_delay,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

                if error_code == MODEL_LOADING_ERROR_CODE:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)

                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Model loading in progress, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

                if error_code == RESOURCE_EXHAUSTED_ERROR_CODE:
                    oom_retries = await _handle_oom_retry(
                        response,
                        start_time=start_time,
                        oom_retries=oom_retries,
                        max_oom_retries=max_oom_retries,
                        timeout=timeout,
                        model=model,
                    )
                    continue

            # Handle 504 (gateway timeout). See encode() above for rationale.
            if response.status_code == HTTP_GATEWAY_TIMEOUT and wait_for_capacity:
                elapsed = time.monotonic() - start_time
                if elapsed < timeout:
                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Gateway timeout (504), retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    await asyncio.sleep(actual_delay)
                    continue

            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)

            break

        self._check_server_version(response)

        response_data = msgpack.unpackb(response.content, raw=False)

        results = parse_extract_results(response_data["items"])

        attach_request_metadata(results, response.headers)

        return results[0] if single_item else results


# ---------------------------------------------------------------------------
# Jobs + connections namespaces — async variants of the sync
# namespaces; reuse the client's aiohttp session and bearer auth.
# ---------------------------------------------------------------------------


class _AsyncNamespace:
    """Shared JSON transport for the async jobs/connections namespaces."""

    def __init__(self, client: SIEAsyncClient) -> None:
        self._c = client

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: Any = None,
        timeout_s: float | None = None,
        include_base_url_headers: bool = True,
    ) -> Any:
        """One JSON request over the client's aiohttp session (bearer auth reused).

        ``timeout_s`` overrides the session default for a single call — long-running
        POSTs (jobs.submit / batches.create) floor it to 120s so a large preflight
        does not abort while the server keeps working.
        """
        try:
            if method == "GET":
                response = await self._c._get(
                    url,
                    headers={"Accept": JSON_CONTENT_TYPE},
                    include_base_url_headers=include_base_url_headers,
                )
            elif method == "DELETE":
                response = await self._c._delete(
                    url,
                    headers={"Accept": JSON_CONTENT_TYPE},
                    include_base_url_headers=include_base_url_headers,
                )
            else:
                # Pre-serialize so the Content-Type is unambiguously application/json.
                response = await self._c._post(
                    url,
                    data=json.dumps(json_body).encode("utf-8"),
                    headers={"Accept": JSON_CONTENT_TYPE, "Content-Type": JSON_CONTENT_TYPE},
                    timeout_s=timeout_s,
                    include_base_url_headers=include_base_url_headers,
                )
        except TimeoutError as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to connect to {url}: {e}"
            raise SIEConnectionError(msg) from e
        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)
        self._c._check_server_version(response)
        if not response.content:
            return {}
        return response.json()


class _AsyncJobs(_AsyncNamespace):
    """Async batch class — see :class:`SIEClient` ``.jobs`` for the contract."""

    async def submit(
        self,
        *,
        source: Any,
        model: str,
        operation: str = "encode",
        sink: Any = None,
        connection: str | None = None,
        sink_connection: str | None = None,
        field_map: Mapping[str, Any] | None = None,
        output_field: str | None = None,
        when: Any = None,
        output_types: Sequence[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> JobSubmitResult:
        """Async ``jobs.submit``. See :class:`SIEClient` ``.jobs.submit`` for details."""
        body = build_job_body(
            source=source,
            operation=operation,
            model=model,
            sink=sink,
            connection=connection,
            sink_connection=sink_connection,
            field_map=field_map,
            output_field=output_field,
            when=when,
            output_types=output_types,
            options=options,
        )
        return await self._request_json("POST", "/v1/jobs", json_body=body, timeout_s=max(self._c._timeout, 120.0))

    async def get(self, job_id: str) -> JobStatus:
        """Async ``jobs.get``."""
        return await self._request_json("GET", f"/v1/jobs/{job_id}")

    async def list(self) -> Sequence[JobStatus]:
        """Async ``jobs.list``."""
        data = await self._request_json("GET", "/v1/jobs")
        if isinstance(data, dict):
            return data.get("data", [])
        return data if isinstance(data, list) else []

    async def cancel(self, job_id: str) -> JobStatus:
        """Async ``jobs.cancel``."""
        return await self._request_json("POST", f"/v1/jobs/{job_id}/cancel")

    async def results(self, job_id: str) -> JobResults:
        """Async ``jobs.results`` — read + decode the finished job's chunk refs."""
        job = await self.get(job_id)
        chunks = job_chunks(job)
        items = []
        for chunk in chunks:
            ref = chunk.get("ref")
            if chunk.get("state") != "succeeded" or not ref:
                continue
            items.extend(decode_chunk_bytes(await self._read_ref(ref)))
        dims = next((it["dims"] for it in items if it.get("dims")), None)
        return {
            "job_id": job.get("id", job_id),
            "state": job.get("state"),
            "total_items": job.get("total_items"),
            "settled_credits": job.get("settled_credits"),
            "chunks": chunks,
            "retrieved": len(items),
            "dims": dims,
            "items": items,
        }

    async def wait(self, job_id: str, *, timeout_s: float = 600.0, poll_s: float = 2.0) -> JobStatus:
        """Poll ``get`` until the job reaches a terminal state or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        while True:
            job = await self.get(job_id)
            if job.get("state") in TERMINAL_JOB_STATES:
                return job
            if time.monotonic() >= deadline:
                msg = f"job {job_id} still {job.get('state')!r} after {timeout_s:.0f}s"
                raise RequestError(msg, code="job_wait_timeout", status_code=504)
            await asyncio.sleep(poll_s)

    async def _read_ref(self, ref: str) -> bytes:
        """Retrieve a chunk's payload-store ref (local path or http(s) URL, POC).

        http(s) refs are fetched without the client's ``Authorization`` header.
        An exact gateway-origin capability ref receives the configured edge
        headers (for example Modal proxy auth); external refs remain bare. Ref
        redirects are never followed, so neither credential can reach a redirect
        target.
        """
        if ref.startswith(("http://", "https://")):
            timeout = aiohttp.ClientTimeout(total=self._c._timeout)
            headers = {"Accept": "application/octet-stream"}
            if self._c._base_url_headers and request_matches_base_url_origin(self._c._base_url, ref):
                headers.update(self._c._base_url_headers)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(ref, headers=headers, allow_redirects=False) as resp,
            ):
                response = _AioResponse(resp.status, await resp.read(), resp.headers)
            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)
            return response.content
        path = Path(ref)
        if await asyncio.to_thread(path.exists):
            return await asyncio.to_thread(path.read_bytes)
        msg = f"cannot retrieve payload-store ref {ref!r} (POC reads local-path and http(s) refs)"
        raise RequestError(msg, code="bad_ref", status_code=400)


class _AsyncConnections(_AsyncNamespace):
    """Async org-scoped connections — see :class:`SIEClient` ``.connections``."""

    def _base(self) -> str:
        if not self._c._control_plane_url:
            msg = "connections require control_plane_url on the client: SIEAsyncClient(..., control_plane_url=..., org=...)"
            raise ValueError(msg)
        if not self._c._org:
            msg = "connections require org on the client: SIEAsyncClient(..., org=...)"
            raise ValueError(msg)
        return f"{self._c._control_plane_url}/internal/orgs/{self._c._org}/connections"

    async def add(self, name: str, type: str, secret: str) -> ConnectionCreated:
        """Async ``connections.add``."""
        body = {"type": type, "name": name, "secret": secret}
        return await self._request_json("POST", self._base(), json_body=body, include_base_url_headers=False)

    async def list(self) -> Sequence[Connection]:
        """Async ``connections.list`` (secrets redacted)."""
        data = await self._request_json("GET", self._base(), include_base_url_headers=False)
        if isinstance(data, dict):
            return data.get("connections", [])
        return data if isinstance(data, list) else []

    async def revoke(self, name: str) -> ConnectionRevoked:
        """Async ``connections.revoke``."""
        return await self._request_json("DELETE", f"{self._base()}/{name}", include_base_url_headers=False)


# ---------------------------------------------------------------------------
# Files + batches namespaces — async variants of the sync
# namespaces; reuse the client's aiohttp session and bearer auth. Method
# names/args mirror `openai.files` / `openai.batches`.
# ---------------------------------------------------------------------------


class _AsyncFiles(_AsyncNamespace):
    """Async OpenAI-compatible Files API — see :class:`SIEClient` ``.files``."""

    async def upload(
        self,
        file: str | Path | bytes | bytearray | IO[bytes],
        *,
        purpose: str = "batch",
        filename: str | None = None,
    ) -> File:
        """Async ``files.upload`` (``POST /v1/files``)."""
        # resolve_upload does blocking path/file-like reads; off-load it so a
        # large upload does not stall the event loop.
        content, name = await asyncio.to_thread(resolve_upload, file, filename)
        query = urlencode({"purpose": purpose, "filename": name})
        try:
            response = await self._c._post(
                f"/v1/files?{query}",
                data=content,
                headers={"Accept": JSON_CONTENT_TYPE, "Content-Type": "application/jsonl"},
                timeout_s=max(self._c._timeout, 120.0),
            )
        except TimeoutError as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to connect to {self._c._base_url}: {e}"
            raise SIEConnectionError(msg) from e
        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)
        self._c._check_server_version(response)
        return response.json()

    async def create(
        self,
        *,
        file: str | Path | bytes | bytearray | IO[bytes],
        purpose: str = "batch",
    ) -> File:
        """OpenAI-exact alias for :meth:`upload` (``files.create(file=, purpose=)``)."""
        return await self.upload(file, purpose=purpose)

    async def retrieve(self, file_id: str) -> File:
        """Async ``files.retrieve`` (``GET /v1/files/{id}``)."""
        return await self._request_json("GET", f"/v1/files/{file_id}")

    async def content(self, file_id: str) -> bytes:
        """Async ``files.content`` — download a file's raw bytes (``GET /v1/files/{id}/content``)."""
        try:
            response = await self._c._get(
                f"/v1/files/{file_id}/content",
                headers={"Accept": "application/jsonl"},
            )
        except TimeoutError as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        except (aiohttp.ClientError, OSError) as e:
            msg = f"Failed to connect to {self._c._base_url}: {e}"
            raise SIEConnectionError(msg) from e
        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)
        return response.content

    async def delete(self, file_id: str) -> FileDeleted:
        """Async ``files.delete`` (``DELETE /v1/files/{id}``; additive OpenAI-parity)."""
        return await self._request_json("DELETE", f"/v1/files/{file_id}")


class _AsyncBatches(_AsyncNamespace):
    """Async OpenAI-compatible Batch API — see :class:`SIEClient` ``.batches``."""

    async def create(
        self,
        *,
        input_file_id: str,
        endpoint: str = "/v1/embeddings",
        completion_window: str = "24h",
        metadata: dict[str, Any] | None = None,
    ) -> Batch:
        """Async ``batches.create`` (``POST /v1/batches``)."""
        body: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata is not None:
            body["metadata"] = metadata
        return await self._request_json("POST", "/v1/batches", json_body=body, timeout_s=max(self._c._timeout, 120.0))

    async def retrieve(self, batch_id: str) -> Batch:
        """Async ``batches.retrieve`` (``GET /v1/batches/{id}``)."""
        return await self._request_json("GET", f"/v1/batches/{batch_id}")

    async def list(self) -> Sequence[Batch]:
        """Async ``batches.list`` (``GET /v1/batches``; additive OpenAI-parity)."""
        data = await self._request_json("GET", "/v1/batches")
        if isinstance(data, dict):
            return data.get("data", [])
        return data if isinstance(data, list) else []

    async def cancel(self, batch_id: str) -> Batch:
        """Async ``batches.cancel`` (``POST /v1/batches/{id}/cancel``; additive OpenAI-parity)."""
        return await self._request_json("POST", f"/v1/batches/{batch_id}/cancel")
