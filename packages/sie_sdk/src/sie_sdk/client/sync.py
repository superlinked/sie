"""Synchronous SIE Engine Client.

Provides a Python client for the Search Inference Engine server.

- Synchronous encode() method
- Accepts Item or list[Item], returns matching shape
- Uses msgpack for efficient serialization with native numpy support
- Returns numpy arrays directly

Example:
    >>> from sie_sdk import SIEClient
    >>> client = SIEClient("http://localhost:8080")
    >>> result = client.encode("bge-m3", {"text": "Hello world"})
    >>> result["dense"]  # np.ndarray, shape [1024]

GPU Selection and Auto-Retry:
    >>> # Request specific GPU, auto-retry while scaling up
    >>> client = SIEClient("http://gateway:8080")
    >>> result = client.encode(
    ...     "bge-m3",
    ...     {"text": "Hello"},
    ...     gpu="l4",
    ...     wait_for_capacity=True,  # Auto-retry explicit capacity 503s and transient transport errors
    ...     provision_timeout_s=900,  # Wait up to 15 min
    ... )

Resource Pools:
    >>> # Create pool for isolated capacity
    >>> client = SIEClient("http://gateway:8080")
    >>> client.create_pool("eval-bench", {"l4": 2})  # 2 L4 GPUs
    >>> result = client.encode("bge-m3", {"text": "Hello"}, gpu="eval-bench/l4")
    >>> client.delete_pool("eval-bench")  # Cleanup when done
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import threading
import time
import weakref
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import IO, Any, Literal, Self, cast, overload
from urllib.parse import urlencode

import httpx
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
    MODEL_REVISION_HEADER,
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
from ._sse import iter_sse_payloads
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


# Mid-flight transport errors retried under `wait_for_capacity=True`:
# the request was in flight and the peer severed the connection before a
# complete response arrived (proxy idle timeout, rolling restart,
# TCP reset). `httpx.ConnectError` is retried separately at each call
# site to preserve its distinct "Failed to connect" message.
_RETRYABLE_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

_LEASE_RENEWAL_MAX_RETRIES = 5

# NOTE: msgpack_numpy.patch() is called lazily in SIEClient.__init__
# to avoid monkey-patching the global msgpack module at import time.
# This prevents overhead in processes that import sie_sdk but don't use
# the client (e.g., the gateway only needs sie_sdk.types/queue_types).
_NUMPY_PATCHED = False


def _parse_generate_result(
    data: dict[str, Any],
    *,
    request: RequestMetadata | None = None,
) -> GenerateResult:
    """Build a :class:`GenerateResult` from the gateway's JSON envelope.

    The gateway shape (streaming): ``{model, text, finish_reason, usage,
    attempt_id, ttft_ms?, tpot_ms?}``. Tolerant of extra fields for
    forward compatibility with future surface extensions. ``model`` and
    ``text`` are required strings; a missing or null value is surfaced
    as :class:`RequestError` so silent data loss does not look like an
    empty completion.
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


def _close_transport(transport: httpx.Client) -> None:
    """Safety net: close httpx transport if SIEClient.close() was not called.

    This is a module-level function (not a method) so it does not prevent
    garbage collection of the SIEClient instance.
    """
    with contextlib.suppress(Exception):
        transport.close()


def _attach_origin_scoped_headers(
    request: httpx.Request,
    *,
    base_url: str,
    headers: Mapping[str, str],
) -> None:
    """Attach edge headers only to a non-control-plane request at the gateway origin."""
    if request.extensions.get("sie_skip_base_url_headers"):
        return
    if request_matches_base_url_origin(base_url, str(request.url)):
        request.headers.update(headers)


def _handle_oom_retry(
    response: httpx.Response,
    *,
    start_time: float,
    oom_retries: int,
    max_oom_retries: int,
    timeout: float,
    model: str,
) -> int:
    """Sleep through one ``RESOURCE_EXHAUSTED`` retry and return the next
    ``oom_retries`` counter, or raise ``ResourceExhaustedError`` when the
    retry / ``provision_timeout_s`` budget is exhausted.

    Centralises the bounded-exponential-backoff path shared by ``encode``,
    ``score`` and ``extract``: ``oom_retries`` counts *completed* retries
    so the (oom_retries+1)-th attempt is the one we're about to make.
    ``compute_oom_backoff(attempt=oom_retries)`` returns the delay BEFORE
    that attempt; the typical sequence (no ``Retry-After``) is
    5 → 10 → 20 → 30s capped, so three retries take ~35s total. Distinct
    from MODEL_LOADING: the model is already resident, the request just
    lost the race for compute resources.
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
    # Sustained OOM: the next backoff would consume the rest of the
    # provision-timeout budget without leaving room for the retried
    # request to actually run. Surface the *root cause*
    # (``ResourceExhaustedError``) now rather than sleeping the budget
    # away and letting the outer loop's ``remaining <= 0`` branch raise
    # ``ProvisioningError`` — that masquerade was the original
    # complaint: a server stuck at OOM would surface to callers as
    # "provisioning timeout" with no hint that the real failure was
    # capacity exhaustion.
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
    time.sleep(delay)
    return oom_retries + 1


class SIEClient:
    """Client for the Search Inference Engine.

    Args:
        base_url: Base URL of the SIE server (e.g., "http://localhost:8080").
        timeout_s: Request timeout in seconds (default: 30.0).
        api_key: Optional API key for authentication (sent as Bearer token).
        gpu: Default GPU/machine profile for requests (e.g., "l4", "l4-spot").
            Can be overridden per-call.
        options: Options dict for requests. Merged with per-call options (per-call wins).
        pool: DEPRECATED. Use create_pool() instead. Resource pool spec for isolated
            capacity. Format: {"name": "pool-name", "gpus": {"l4": 2}}.
        max_connections: Optional maximum number of HTTP connections. Uses
            httpx defaults when omitted.
        base_url_headers: Optional additional headers for the configured gateway
            origin. Values are copied at construction and never forwarded to a
            control-plane URL, external payload-store reference, or redirect
            target. Same-origin capability refs receive only these edge headers.

    Example:
        >>> client = SIEClient("http://localhost:8080")
        >>> result = client.encode("bge-m3", {"text": "Hello world"})
        >>> print(result["dense"].shape)
        (1024,)

        >>> # With defaults for all requests
        >>> client = SIEClient(
        ...     "http://gateway:8080",
        ...     gpu="l4",
        ...     options={"normalize": True},
        ... )
        >>> result = client.encode("bge-m3", {"text": "Hello"})  # uses l4
        >>> result = client.encode("bge-m3", {"text": "Hello"}, gpu="a100")  # overrides

        >>> # With resource pool for isolated capacity (new API)
        >>> client = SIEClient("http://gateway:8080")
        >>> client.create_pool("eval-bench", {"l4": 2})
        >>> result = client.encode("bge-m3", {"text": "Hello"}, gpu="eval-bench/l4")
        >>> client.delete_pool("eval-bench")
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

        # Multi-pool state: track created pools and their lease renewal threads
        # Key: pool name, Value: (lease_thread, stop_event)
        self._pools: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._pools_lock = threading.Lock()

        # Legacy pool state (DEPRECATED - for backward compatibility)
        self._pool_spec = pool
        self._pool_created = False
        self._pool_lock = threading.Lock()
        self._lease_renewal_thread: threading.Thread | None = None
        self._lease_renewal_stop = threading.Event()

        # Note: LoRA and model loading retry counters are now local to each method
        # to avoid interference between concurrent requests

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

        client_kwargs: dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": timeout_s,
            "headers": headers,
            "follow_redirects": False,
        }
        if self._base_url_headers:
            client_kwargs["event_hooks"] = {
                "request": [
                    partial(
                        _attach_origin_scoped_headers,
                        base_url=self._base_url,
                        headers=self._base_url_headers,
                    )
                ]
            }
        if max_connections is not None:
            client_kwargs["limits"] = httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            )
        self._client = httpx.Client(**client_kwargs)
        # Per-thread request evidence lets benchmark callers account for SDK
        # retries without changing normal response payloads.
        self._request_state = threading.local()

        # Safety net: ensure transport is closed even if close() is never called.
        # Uses weakref.finalize so the reference doesn't prevent GC.
        self._finalizer = weakref.finalize(self, _close_transport, self._client)

        # Register cleanup on interpreter exit
        if pool is not None:
            atexit.register(self._cleanup_pool)

        # First-class batch + connector surface.
        self.jobs = _SyncJobs(self)
        self.connections = _SyncConnections(self)
        # OpenAI-compatible Files + Batches surface — a
        # `base_url` swap makes an `openai` batch caller work unchanged.
        self.files = _SyncFiles(self)
        self.batches = _SyncBatches(self)

    @property
    def base_url(self) -> str:
        """Return the base URL of the SIE server."""
        return self._base_url

    @property
    def last_retry_count(self) -> int:
        """Return SDK retries performed by the latest call in this thread."""
        return int(getattr(self._request_state, "last_retry_count", 0))

    @property
    def last_model_revision(self) -> str | None:
        """Return the deployed execution revision observed on the latest call in this thread.

        The property name is retained for wire compatibility with
        ``X-SIE-Model-Revision``.
        """
        value = getattr(self._request_state, "last_model_revision", None)
        return str(value) if value is not None else None

    def _reset_retry_count(self) -> None:
        self._request_state.last_retry_count = 0
        self._request_state.last_model_revision = None

    def _record_retry(self) -> None:
        self._request_state.last_retry_count = self.last_retry_count + 1

    def _check_server_version(self, response: httpx.Response) -> None:
        revision = response.headers.get(MODEL_REVISION_HEADER)
        self._request_state.last_model_revision = revision if isinstance(revision, str) and revision else None
        if SIEClient._version_warning_logged:
            return
        server_version = response.headers.get(SERVER_VERSION_HEADER)
        if not server_version:
            return
        sdk_version = self._headers.get(SDK_VERSION_HEADER, "unknown")
        warning = check_version_skew(sdk_version, server_version)
        if warning:
            logger.warning(warning)
            SIEClient._version_warning_logged = True

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

    def _resolve_pool_and_gpu(self, gpu: str | None) -> tuple[str | None, str | None]:
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
                self._ensure_pool_created()
                return self._pool_spec["name"], None
            return None, None

        pool_name, gpu_type = parse_gpu_param(resolved_gpu)

        # If pool name in gpu param matches our pool, ensure it's created
        if pool_name and self._pool_spec and pool_name == self._pool_spec.get("name"):
            self._ensure_pool_created()

        return pool_name, gpu_type

    def _ensure_pool_created(self) -> None:
        """Ensure the pool is created (lazy initialization).

        Thread-safe - uses lock to prevent multiple creation attempts.
        Starts lease renewal background thread after pool creation.
        """
        if self._pool_spec is None:
            return

        with self._pool_lock:
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
                response = self._client.post(
                    "/v1/pools",
                    json=request_body,
                    headers={"Content-Type": JSON_CONTENT_TYPE, "Accept": JSON_CONTENT_TYPE},
                )
            except httpx.ConnectError as e:
                msg = f"Failed to create pool '{pool_name}': connection error: {e}"
                raise PoolError(msg, pool_name=pool_name) from e

            if response.status_code >= HTTP_CLIENT_ERROR:
                # Parse error
                try:
                    data = response.json()
                    error_msg = data.get("detail", {}).get("message", str(data))
                except (ValueError, KeyError):
                    error_msg = response.text
                msg = f"Failed to create pool '{pool_name}': {error_msg}"
                raise PoolError(msg, pool_name=pool_name)

            # Pool created successfully
            data = response.json()
            # Handle nested structure: data["status"]["state"]
            state = data.get("status", {}).get("state", "unknown")
            logger.info("Pool '%s' created with state '%s'", pool_name, state)

            self._pool_created = True

            # Start lease renewal thread
            self._start_lease_renewal()

    def _start_lease_renewal(self) -> None:
        """Start the background lease renewal thread."""
        if self._pool_spec is None or self._lease_renewal_thread is not None:
            return

        self._lease_renewal_stop.clear()
        self._lease_renewal_thread = threading.Thread(
            target=self._lease_renewal_loop,
            name=f"pool-lease-{self._pool_spec['name']}",
            daemon=True,
        )
        self._lease_renewal_thread.start()
        logger.debug("Started lease renewal thread for pool '%s'", self._pool_spec["name"])

    def _lease_renewal_loop(self) -> None:
        """Background thread loop to renew pool lease."""
        if self._pool_spec is None:
            return

        pool_name = self._pool_spec["name"]

        while not self._lease_renewal_stop.wait(timeout=DEFAULT_LEASE_RENEWAL_INTERVAL_S):
            last_error: Exception | None = None
            for attempt in range(_LEASE_RENEWAL_MAX_RETRIES):
                try:
                    response = self._client.post(
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
                except (httpx.HTTPError, OSError) as e:
                    last_error = e
                    logger.warning(
                        "Error renewing lease for pool '%s': %s (attempt %d/%d)",
                        pool_name,
                        e,
                        attempt + 1,
                        _LEASE_RENEWAL_MAX_RETRIES,
                    )
                backoff = min(2.0**attempt, 10.0)
                if self._lease_renewal_stop.wait(timeout=backoff):
                    return
            else:
                if last_error:
                    logger.error(
                        "All %d lease renewal attempts failed for pool '%s': %s",
                        _LEASE_RENEWAL_MAX_RETRIES,
                        pool_name,
                        last_error,
                    )

    def _cleanup_pool(self) -> None:
        """Cleanup legacy pool resources on client close."""
        # Stop legacy lease renewal thread
        if self._lease_renewal_thread is not None:
            self._lease_renewal_stop.set()
            self._lease_renewal_thread.join(timeout=5.0)
            self._lease_renewal_thread = None

        # Note: Pool deletion is not done here - pools are GC'd by gateway
        # after inactivity. This allows pool reuse if client reconnects.

    def _cleanup_all_pools(self) -> None:
        """Cleanup all pool lease renewal threads."""
        # Stop all new-style pool threads
        with self._pools_lock:
            for pool_name, (thread, stop_event) in list(self._pools.items()):
                stop_event.set()
                thread.join(timeout=5.0)
            self._pools.clear()

        # Also cleanup legacy pool
        self._cleanup_pool()

    def create_pool(
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

        Pools reserve exclusive GPU capacity for your workload. Use them for:
        - Benchmarks that need consistent performance
        - Evaluations that shouldn't compete with production traffic
        - Isolated environments for testing

        Args:
            name: Pool name (used in gpu="pool_name/machine_profile" routing).
                The gateway stores and routes pool names in lowercase.
            gpus: Optional machine profile requirements for pool readiness, e.g.,
                {"l4": 2, "l4-spot": 1}.
                Keys are machine profile names from cluster config.
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

        Example:
            >>> client.create_pool("eval", {"l4": 2}, bundle="default", minimum_worker_count=1)
            >>> result = client.encode("bge-m3", {"text": "Hello"}, gpu="eval/l4")
            >>> client.delete_pool("eval")
        """
        with self._pools_lock:
            already_tracking = name in self._pools

        if minimum_worker_count is not None and minimum_worker_count < 0:
            msg = "minimum_worker_count must be >= 0"
            raise ValueError(msg)

        logger.info(
            "Creating/updating pool '%s' with gpus=%s, gpu_caps=%s, bundle=%s",
            name,
            gpus,
            gpu_caps,
            bundle,
        )

        # Build pool creation request
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
            response = self._client.post(
                "/v1/pools",
                json=request_body,
                headers={"Content-Type": JSON_CONTENT_TYPE, "Accept": JSON_CONTENT_TYPE},
            )
        except httpx.ConnectError as e:
            msg = f"Failed to create pool '{name}': connection error: {e}"
            raise PoolError(msg, pool_name=name) from e

        if response.status_code >= HTTP_CLIENT_ERROR:
            # Parse error
            try:
                data = response.json()
                error_msg = data.get("detail", {}).get("message", str(data))
            except (ValueError, KeyError):
                error_msg = response.text
            msg = f"Failed to create pool '{name}': {error_msg}"
            raise PoolError(msg, pool_name=name)

        # Pool created successfully
        data = response.json()
        state = data.get("status", {}).get("state", "unknown")
        logger.info("Pool '%s' created/updated with state '%s'", name, state)

        # Start lease renewal thread for this pool if this client is not
        # already tracking it. Repeated create_pool calls intentionally still
        # POST so callers can update gpus/gpu_caps on the gateway.
        if not already_tracking:
            self._start_pool_lease_renewal(name)

    def _start_pool_lease_renewal(self, pool_name: str) -> None:
        """Start lease renewal thread for a pool."""
        with self._pools_lock:
            if pool_name in self._pools:
                return
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._pool_lease_renewal_loop,
                args=(pool_name, stop_event),
                name=f"pool-lease-{pool_name}",
                daemon=True,
            )
            self._pools[pool_name] = (thread, stop_event)
            thread.start()

        logger.debug("Started lease renewal thread for pool '%s'", pool_name)

    def _pool_lease_renewal_loop(self, pool_name: str, stop_event: threading.Event) -> None:
        """Background thread loop to renew pool lease."""
        while not stop_event.wait(timeout=DEFAULT_LEASE_RENEWAL_INTERVAL_S):
            last_error: Exception | None = None
            for attempt in range(_LEASE_RENEWAL_MAX_RETRIES):
                try:
                    response = self._client.post(
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
                except (httpx.HTTPError, OSError) as e:
                    last_error = e
                    logger.warning(
                        "Error renewing lease for pool '%s': %s (attempt %d/%d)",
                        pool_name,
                        e,
                        attempt + 1,
                        _LEASE_RENEWAL_MAX_RETRIES,
                    )
                backoff = min(2.0**attempt, 10.0)
                if stop_event.wait(timeout=backoff):
                    return
            else:
                if last_error:
                    logger.error(
                        "All %d lease renewal attempts failed for pool '%s': %s",
                        _LEASE_RENEWAL_MAX_RETRIES,
                        pool_name,
                        last_error,
                    )

    def get_pool(self, name: str | None = None) -> PoolInfo | None:
        """Get information about a pool.

        Args:
            name: Pool name to look up. If None, uses the legacy constructor pool.

        Returns:
            PoolInfo if pool exists, None otherwise.

        Raises:
            SIEConnectionError: If unable to connect to the server.
            PoolError: If pool lookup fails.

        Example:
            >>> client.create_pool("eval", {"l4": 2})
            >>> info = client.get_pool("eval")
            >>> print(f"Pool state: {info['status']['state']}, workers: {len(info['status']['assigned_workers'])}")
        """
        # Determine pool name (new API or legacy)
        if name is not None:
            pool_name = name
        elif self._pool_spec is not None:
            pool_name = self._pool_spec["name"]
        else:
            return None

        try:
            response = self._client.get(
                f"/v1/pools/{pool_name}",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except httpx.ConnectError as e:
            msg = f"Failed to get pool '{pool_name}': connection error: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code == 404:
            # Pool doesn't exist yet (not created)
            return None

        if response.status_code >= HTTP_CLIENT_ERROR:
            try:
                data = response.json()
                detail = data.get("detail", {})
                # Handle both string and dict detail formats
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
        # Return the nested structure directly (matches PoolInfo TypedDict)
        return PoolInfo(
            name=data.get("name", pool_name),
            spec=data.get("spec", {}),
            status=data.get("status", {}),
        )

    def delete_pool(self, name: str | None = None) -> bool:
        """Delete a pool.

        This explicitly releases pool resources. Normally pools are GC'd
        automatically after inactivity, so this is only needed for
        immediate cleanup.

        Args:
            name: Pool name to delete. If None, uses the legacy constructor pool.

        Returns:
            True if pool was deleted, False if pool didn't exist.

        Raises:
            SIEConnectionError: If unable to connect to the server.
            PoolError: If pool deletion fails.

        Example:
            >>> client.create_pool("eval", {"l4": 2})
            >>> # ... use pool ...
            >>> client.delete_pool("eval")
            True
        """
        # Determine pool name (new API or legacy)
        if name is not None:
            pool_name = name
        elif self._pool_spec is not None:
            pool_name = self._pool_spec["name"]
        else:
            return False

        # Stop lease renewal thread for this pool
        with self._pools_lock:
            if pool_name in self._pools:
                thread, stop_event = self._pools.pop(pool_name)
                stop_event.set()
                thread.join(timeout=5.0)

        # Also handle legacy pool cleanup if this is the legacy pool
        if self._pool_spec is not None and pool_name == self._pool_spec.get("name"):
            self._cleanup_pool()

        try:
            response = self._client.delete(
                f"/v1/pools/{pool_name}",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except httpx.ConnectError as e:
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

    def close(self) -> None:
        """Close the HTTP client and cleanup pool resources."""
        self._cleanup_all_pools()
        self._client.close()
        self._finalizer.detach()  # Prevent double-close from GC finalizer

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager."""
        self.close()

    # Use overload for proper type hints when single item vs list
    @overload
    def encode(
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
    def encode(
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

    def encode(
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
        """Encode items into vector representations.

        Args:
            model: Model name to use for encoding (e.g., "bge-m3").
            items: Single Item or list of Items to encode.
            output_types: Which outputs to return: ["dense"], ["sparse"], ["dense", "sparse", "multivector"].
                         Default: ["dense"].
            instruction: Task instruction for instruction-tuned models.
            output_dtype: Output dtype: "float32", "float16", "int8", "uint8", "binary", "ubinary".
            is_query: Whether this is a query embedding (vs document). Affects some models
                     that use asymmetric encoding (e.g., BGE, E5). Default: None (model default).
            options: Runtime options dict. Can include "profile" to select a named profile,
                    or individual options like "muvera", "normalize", etc.
            gpu: Target GPU type (e.g., "l4", "a100-80gb"). Routes request to workers
                with matching GPU. Required when using the gateway with multiple GPU pools.
            wait_for_capacity: When True (default), auto-retry transient "not
                enough capacity yet" responses under ``provision_timeout_s`` —
                ``503 PROVISIONING`` (scale-from-zero provisioning);
                ``504`` gateway result timeouts for idempotent queue paths; local
                ``httpx`` read/connect/pool timeouts; and transient
                mid-flight transport errors (``RemoteProtocolError``,
                ``ReadError``, ``WriteError`` — peer severed the connection
                before a complete response arrived). Retries honour the
                server's ``Retry-After`` header when present. When False,
                these surface immediately as ``ProvisioningError`` /
                ``ServerError`` / ``SIEConnectionError``.
                Note: ``503 MODEL_LOADING``, ``503 LORA_LOADING`` and
                ``503 RESOURCE_EXHAUSTED`` are retried regardless of this
                flag — the worker has already accepted the request and is
                loading the target model/adapter or recovering from
                transient capacity exhaustion. Their budgets are documented
                under ``ModelLoadingError`` / ``LoraLoadingError`` /
                ``ResourceExhaustedError`` below; the ``RESOURCE_EXHAUSTED``
                branch can be disabled by passing ``max_oom_retries=0``.
            provision_timeout_s: Maximum time to wait for capacity when wait_for_capacity=True.
                Default: 900 seconds (15 minutes).
            max_oom_retries: Public retry knob capping the number of
                ``503 RESOURCE_EXHAUSTED`` (server-side OOM) retries. Each
                retry uses bounded exponential backoff
                (``compute_oom_backoff``); the SDK also stops retrying
                early if the next backoff would exhaust
                ``provision_timeout_s``. ``RESOURCE_EXHAUSTED`` retries
                run regardless of ``wait_for_capacity``; set
                ``max_oom_retries=0`` to disable them entirely (the first
                OOM surfaces immediately as ``ResourceExhaustedError``).
                Default: ``RESOURCE_EXHAUSTED_MAX_RETRIES`` (3).

        Returns:
            EncodeResult if single item was passed, list[EncodeResult] if list was passed.
            Each result contains the requested output types as numpy arrays.

        Raises:
            RequestError: If the request is invalid (4xx response).
            ServerError: If the server encounters an error (5xx response).
            SIEConnectionError: If unable to connect to the server.
            ProvisioningError: If ``wait_for_capacity=False`` and the gateway
                returns ``503 PROVISIONING`` (scale-from-zero provisioning), or if provisioning
                retries exceed ``provision_timeout_s``.
            ModelLoadingError: If ``503`` ``MODEL_LOADING`` retries exceed
                ``provision_timeout_s`` during worker-side cold-loading of the
                target model. Note: this branch retries regardless of
                ``wait_for_capacity``.
            LoraLoadingError: If ``503`` ``LORA_LOADING`` retries exhaust the
                (short, fixed) retry budget. Note: this branch retries
                regardless of ``wait_for_capacity``.
            ResourceExhaustedError: If ``503`` ``RESOURCE_EXHAUSTED``
                retries exhaust ``max_oom_retries`` or the next backoff
                would exhaust ``provision_timeout_s``. Note: this branch
                retries regardless of ``wait_for_capacity`` unless
                ``max_oom_retries=0``.

        Example:
            >>> # Single item
            >>> result = client.encode("bge-m3", {"text": "Hello"})
            >>> result["dense"]  # np.ndarray

            >>> # Batch
            >>> results = client.encode("bge-m3", [{"text": "Hello"}, {"text": "World"}])
            >>> len(results)  # 2

            >>> # With GPU selection (for gateway)
            >>> result = client.encode("bge-m3", {"text": "Hello"}, gpu="l4")

            >>> # Auto-wait for capacity during scale-up
            >>> result = client.encode(
            ...     "bge-m3",
            ...     {"text": "Hello"},
            ...     gpu="l4",
            ...     wait_for_capacity=True,
            ...     provision_timeout_s=900,  # Wait up to 15 min
            ... )

            >>> # Query embedding with instruction (for E5, GTE-Qwen, etc.)
            >>> result = client.encode(
            ...     "gte-qwen2-7b",
            ...     {"text": "What is ML?"},
            ...     instruction="Retrieve passages that answer the question",
            ...     is_query=True,
            ... )

            >>> # Multimodal (CLIP, SigLIP, etc.)
            >>> result = client.encode(
            ...     "openai/clip-vit-base-patch32",
            ...     {"images": ["photo.jpg"]},
            ... )
        """
        self._reset_retry_count()
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
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)
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
        # Retry counter for server-side OOM (RESOURCE_EXHAUSTED). Bounded by
        # ``RESOURCE_EXHAUSTED_MAX_RETRIES`` so a stuck-at-OOM server cannot
        # cause unbounded blocking; each retry uses bounded exponential
        # backoff via ``compute_oom_backoff``.
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
                response = self._client.post(
                    f"/v1/encode/{model}", content=body, headers=headers, timeout=request_timeout
                )
            except httpx.ConnectError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                if wait_for_capacity:
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Transient transport error",
                        error=e,
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                if isinstance(e, httpx.TimeoutException):
                    msg = f"Request timed out: {e}"
                else:
                    msg = (
                        f"Connection lost mid-request ({type(e).__name__}); "
                        f"the peer closed the connection before sending a complete response: {e}"
                    )
                raise SIEConnectionError(msg) from e

            # Short-circuit terminal load failures BEFORE engaging the
            # MODEL_LOADING retry budget. The server emits 502
            # MODEL_LOAD_FAILED for permanent classes (gated repos,
            # missing deps) — retrying would waste 5 minutes on a
            # known-bad config (sie-test#85).
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
                    self._record_retry()
                    time.sleep(actual_delay)
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
                    self._record_retry()
                    time.sleep(delay)
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
                    self._record_retry()
                    time.sleep(actual_delay)
                    continue

                if error_code == RESOURCE_EXHAUSTED_ERROR_CODE:
                    oom_retries = _handle_oom_retry(
                        response,
                        start_time=start_time,
                        oom_retries=oom_retries,
                        max_oom_retries=max_oom_retries,
                        timeout=timeout,
                        model=model,
                    )
                    self._record_retry()
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
                    self._record_retry()
                    time.sleep(actual_delay)
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
        response_model = response_data.get("model")
        if isinstance(response_model, str) and response_model:
            for result in results:
                result["model"] = response_model
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

    def list_models(self) -> list[ModelInfo]:
        """List available models with their capabilities.

        Returns:
            List of ModelInfo dicts with name, loaded status, inputs, outputs, and dims.

        Raises:
            SIEConnectionError: If unable to connect to the server.
            ServerError: If the server encounters an error.

        Example:
            >>> models = client.list_models()
            >>> for m in models:
            ...     print(f"{m['name']}: {m['outputs']}")
            bge-m3: ['dense', 'sparse', 'multivector']
        """
        try:
            response = self._client.get(
                "/v1/models",
                headers={"Accept": JSON_CONTENT_TYPE},  # Models endpoint returns JSON
            )
        except httpx.ConnectError as e:
            msg = f"Failed to connect to {self._base_url}: {e}"
            raise SIEConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)

        data = response.json()
        return data["models"]

    def get_model(self, model: str) -> ModelInfo:
        """Get details for a specific model.

        Returns model metadata including dimensions, supported inputs/outputs,
        loaded status, and profiles. This is a lightweight call that reads
        from model config — it does not load the model or trigger inference.

        Args:
            model: Model name (e.g., "BAAI/bge-m3").

        Returns:
            ModelInfo dict with name, dims, inputs, outputs, loaded, etc.

        Raises:
            RequestError: If the model is not found (404).
            SIEConnectionError: If unable to connect to the server.
            ServerError: If the server encounters an error.

        Example:
            >>> info = client.get_model("BAAI/bge-m3")
            >>> info["dims"]["dense"]
            1024
        """
        try:
            response = self._client.get(
                f"/v1/models/{model}",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except httpx.ConnectError as e:
            msg = f"Failed to connect to {self._base_url}: {e}"
            raise SIEConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)

        return response.json()

    def _detect_endpoint_type(self) -> Literal["cluster", "worker"]:
        """Detect whether base_url is a gateway (cluster) or worker endpoint."""
        try:
            response = self._client.get(
                "/health",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except httpx.HTTPError:
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

    def watch(
        self,
        *,
        mode: Literal["auto", "cluster", "worker"] = "auto",
    ) -> Iterator[StatusMessage]:
        """Stream real-time status updates from the server or gateway.

        Args:
            mode: "cluster" connects to /ws/cluster-status, "worker" to /ws/status.
                "auto" detects gateway vs worker via /health.

        Yields:
            StatusMessage updates (ClusterStatusMessage or WorkerStatusMessage).
        """
        from websockets.exceptions import InvalidStatus, WebSocketException
        from websockets.sync.client import connect

        if mode == "auto":
            detected = self._detect_endpoint_type()
            paths = ["/ws/cluster-status"] if detected == "cluster" else ["/ws/status"]
        elif mode == "cluster":
            paths = ["/ws/cluster-status"]
        else:
            paths = ["/ws/status"]

        for path in paths:
            ws_url = self._ws_url(path)
            try:
                with connect(ws_url, additional_headers=self._websocket_headers(ws_url)) as ws:
                    for message in ws:
                        if isinstance(message, bytes):
                            payload = message.decode("utf-8")
                        else:
                            payload = message
                        data = json.loads(payload)
                        yield data
                return
            except InvalidStatus as e:
                status = (
                    getattr(e, "status_code", None)
                    or getattr(e, "status", None)
                    or getattr(getattr(e, "response", None), "status_code", None)
                )
                raise RequestError(f"WebSocket connection failed: {status}") from e
            except WebSocketException as e:
                raise SIEConnectionError(f"WebSocket error: {e}") from e
            except (OSError, json.JSONDecodeError) as e:
                raise SIEConnectionError(f"WebSocket error: {e}") from e

    def get_capacity(self, *, gpu: str | None = None) -> CapacityInfo:
        """Get current cluster capacity information.

        Queries the gateway's /health endpoint for cluster state. Useful for
        checking if specific GPU types are available before sending requests.

        Args:
            gpu: Optional filter to check specific GPU type availability.

        Returns:
            CapacityInfo with worker count, GPU types, and worker details.
            If gpu is specified, only workers with matching GPU are included.

        Raises:
            SIEConnectionError: If unable to connect to the server.
            ServerError: If the server encounters an error.
            RequestError: If the endpoint is not available (e.g., worker, not gateway).

        Example:
            >>> # Check cluster state
            >>> capacity = client.get_capacity()
            >>> print(f"Workers: {capacity['worker_count']}, GPUs: {capacity['live_gpu_types']}")
            Workers: 4, GPUs: ['l4', 'a100-80gb']

            >>> # Check if L4 GPUs are available
            >>> capacity = client.get_capacity(gpu="l4")
            >>> if capacity["worker_count"] > 0:
            ...     print("L4 workers available")
        """
        try:
            response = self._client.get(
                "/health",
                headers={"Accept": JSON_CONTENT_TYPE},
            )
        except httpx.ConnectError as e:
            msg = f"Failed to connect to {self._base_url}: {e}"
            raise SIEConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request timed out: {e}"
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

    def wait_for_capacity(
        self,
        gpu: str,
        *,
        model: str | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float = 5.0,
    ) -> CapacityInfo:
        """Wait for GPU capacity to become available.

        Polls the gateway until workers with the specified GPU type are online.
        This is useful for pre-warming the cluster before running benchmarks.

        Note: This triggers capacity requests by sending a warmup encode request
        with wait_for_capacity=True. If you just want to check capacity without
        triggering scale-up, use get_capacity() instead.

        Args:
            gpu: GPU type to wait for (e.g., "l4", "a100-80gb").
            model: Optional model to use for warmup request. If provided, sends
                a warmup encode request which may trigger model loading.
            timeout_s: Maximum time to wait for capacity. Default: 300s (5 min).
            poll_interval_s: How often to check capacity. Default: 5s.

        Returns:
            CapacityInfo once capacity is available.

        Raises:
            ProvisioningError: If timeout is exceeded waiting for capacity.
            SIEConnectionError: If unable to connect to the server.

        Example:
            >>> # Wait for L4 capacity before running benchmarks
            >>> capacity = client.wait_for_capacity("l4", timeout_s=300)
            >>> print(f"Ready with {capacity['worker_count']} L4 workers")

            >>> # Wait and pre-load a model
            >>> capacity = client.wait_for_capacity("l4", model="bge-m3")
        """
        timeout = timeout_s if timeout_s is not None else DEFAULT_PROVISION_TIMEOUT_S
        start_time = time.monotonic()

        # If model is specified, use encode with wait_for_capacity to trigger
        # both scale-up and model loading
        if model:
            self.encode(
                model,
                Item(text="warmup"),
                gpu=gpu,
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            # After successful encode, get capacity info
            return self.get_capacity(gpu=gpu)

        # Otherwise, poll capacity until workers are available
        while True:
            try:
                capacity = self.get_capacity(gpu=gpu)
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
            time.sleep(delay)

    def score(
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

        Sends query and items to the server, which encodes and computes scores
        (cross-encoder or MaxSim depending on the model).

        For client-side MaxSim with pre-encoded multivectors, use
        :func:`sie_sdk.scoring.maxsim` directly.

        Args:
            model: Model name to use for scoring (must support reranking).
            query: Query item (e.g., ``{"text": "query text"}``).
            items: List of items to score against the query.
            instruction: Optional instruction for instruction-tuned models.
            options: Runtime options dict. Can include "profile" to select a named profile.
            gpu: Target GPU type (e.g., "l4", "a100-80gb"). Routes request to workers
                with matching GPU.
            wait_for_capacity: When True (default), auto-retry transient "not
                enough capacity yet" responses under ``provision_timeout_s`` —
                ``503 PROVISIONING`` (scale-from-zero provisioning);
                ``504`` gateway result timeouts for idempotent queue paths; local
                ``httpx`` read/connect/pool timeouts; and transient
                mid-flight transport errors (``RemoteProtocolError``,
                ``ReadError``, ``WriteError`` — peer severed the connection
                before a complete response arrived). Retries honour the
                server's ``Retry-After`` header when present. When False,
                these surface immediately as ``ProvisioningError`` /
                ``ServerError`` / ``SIEConnectionError``.
                Note: ``503 MODEL_LOADING`` and ``503 RESOURCE_EXHAUSTED``
                are retried regardless of this flag — the worker has
                already accepted the request and is loading the target
                model or recovering from transient capacity exhaustion.
                Their budgets are documented under ``ModelLoadingError`` /
                ``ResourceExhaustedError`` below; the
                ``RESOURCE_EXHAUSTED`` branch can be disabled by passing
                ``max_oom_retries=0``.
            provision_timeout_s: Maximum time to wait for capacity when wait_for_capacity=True.
                Default: 900 seconds (15 minutes).
            max_oom_retries: Public retry knob capping the number of
                ``503 RESOURCE_EXHAUSTED`` (server-side OOM) retries. Each
                retry uses bounded exponential backoff
                (``compute_oom_backoff``); the SDK also stops retrying
                early if the next backoff would exhaust
                ``provision_timeout_s``. ``RESOURCE_EXHAUSTED`` retries
                run regardless of ``wait_for_capacity``; set
                ``max_oom_retries=0`` to disable them entirely.
                Default: ``RESOURCE_EXHAUSTED_MAX_RETRIES`` (3).

        Returns:
            ScoreResult containing the model name, query_id, and sorted scores.
            Scores are sorted by relevance (descending), with rank 0 being most relevant.

        Raises:
            RequestError: If the request is invalid (4xx response).
            ServerError: If the server encounters an error (5xx response).
            SIEConnectionError: If unable to connect to the server.
            ProvisioningError: If ``wait_for_capacity=False`` and the gateway
                returns ``503 PROVISIONING`` (scale-from-zero provisioning), or if provisioning
                retries exceed ``provision_timeout_s``.
            ModelLoadingError: If ``503`` ``MODEL_LOADING`` retries exceed
                ``provision_timeout_s`` during worker-side cold-loading of the
                target model. Note: this branch retries regardless of
                ``wait_for_capacity``.
            ResourceExhaustedError: If ``503`` ``RESOURCE_EXHAUSTED``
                retries exhaust ``max_oom_retries`` or the next backoff
                would exhaust ``provision_timeout_s``. Note: this branch
                retries regardless of ``wait_for_capacity`` unless
                ``max_oom_retries=0``.

        Example:
            >>> result = client.score(
            ...     "bge-reranker-v2",
            ...     query={"text": "What is machine learning?"},
            ...     items=[{"text": "ML is AI..."}, {"text": "Python is..."}],
            ... )
        """
        self._reset_retry_count()
        # Resolve defaults and pool
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)
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
                response = self._client.post(
                    f"/v1/score/{model}", content=body, headers=headers, timeout=request_timeout
                )
            except httpx.ConnectError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                if wait_for_capacity:
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Transient transport error",
                        error=e,
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                if isinstance(e, httpx.TimeoutException):
                    msg = f"Request timed out: {e}"
                else:
                    msg = (
                        f"Connection lost mid-request ({type(e).__name__}); "
                        f"the peer closed the connection before sending a complete response: {e}"
                    )
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
                    self._record_retry()
                    time.sleep(actual_delay)
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
                    self._record_retry()
                    time.sleep(actual_delay)
                    continue

                if error_code == RESOURCE_EXHAUSTED_ERROR_CODE:
                    oom_retries = _handle_oom_retry(
                        response,
                        start_time=start_time,
                        oom_retries=oom_retries,
                        max_oom_retries=max_oom_retries,
                        timeout=timeout,
                        model=model,
                    )
                    self._record_retry()
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
                    self._record_retry()
                    time.sleep(actual_delay)
                    continue

            # Handle errors
            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)

            # Success - break out of retry loop
            break

        self._check_server_version(response)

        # Deserialize response
        response_data = msgpack.unpackb(response.content, raw=False)

        # Build ScoreResult
        result = parse_score_result(response_data)
        attach_request_metadata([result], response.headers)
        return result

    def generate(
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
        """Generate text from a prompt (walking-skeleton SDK surface).

        The SDK does not currently expose streaming chunks to the caller;
        the worker streams chunks to the gateway, the gateway aggregates,
        and the SDK returns the assembled result plus SIE-native timing
        metadata (``ttft_ms``, ``tpot_ms``, ``attempt_id``).

        Args:
            model: Model name. HF-style IDs such as
                ``"Qwen/Qwen3-4B-Instruct-2507"`` are normalized to the
                gateway's SIE-safe path id
                ``"Qwen__Qwen3-4B-Instruct-2507"`` for this endpoint.
            prompt: Raw prompt string. Chat-template rendering, if any,
                is performed by the worker — this surface has no chat-template
                helpers in the SDK (use the OpenAI SDK against
                ``/v1/chat/completions`` for chat-shaped requests).
            max_new_tokens: Hard cap on output tokens.
            images: Optional native image inputs. When present, the worker
                renders one user turn containing the images and ``prompt``
                through the model's own chat template before generation.
            temperature: Sampling temperature override. Omit to use the
                selected model profile's default.
            top_p: Nucleus sampling cutoff override. Omit to use the selected
                model profile's default.
            stop: Optional list of stop strings.
            frequency_penalty: OpenAI-compatible frequency penalty in ``[-2, 2]``.
            presence_penalty: OpenAI-compatible presence penalty in ``[-2, 2]``.
            grammar: Optional native structured-output grammar. Set exactly
                one of ``json_schema``, ``regex``, or ``ebnf``.
            seed: Optional signed 64-bit per-request sampling seed. Exact
                reproducibility depends on the active generation backend and
                deployment configuration.
            logit_bias: Optional token-id-to-bias map.
            routing_key: Optional stable request routing key.
            prompt_cache_key: Optional prompt cache affinity key.
            safety_identifier: Optional opaque privacy-preserving safety id.
            lora_adapter: Optional served LoRA adapter name.
            options: Governed generation runtime options. Client defaults are
                shallow-merged below per-call values, and explicit typed
                sampler arguments win. Select non-default profiles through the
                ``model:profile`` identity rather than ``options.profile``.
            gpu: Target GPU type / pool spec; see ``encode``.
            wait_for_capacity: Auto-retry 503 PROVISIONING / MODEL_LOADING responses
                under ``provision_timeout_s``. See ``encode``. Unlike the
                idempotent encode/score/extract paths, a ``504`` gateway
                timeout is NOT retried here: generation is non-idempotent
                and a 504 is a post-publish timeout, so retrying could
                double-bill an inference. A ``504`` is surfaced as a
                terminal :class:`ServerError`.
            provision_timeout_s: Maximum time to wait for capacity.

        Returns:
            :class:`GenerateResult` with text, usage, finish_reason, and
            timing metadata.

        Raises:
            ServerError: On a ``504`` gateway timeout (post-publish, not
                retried) or other 5xx responses.
        """
        self._reset_retry_count()
        resolved_grammar = validate_generate_grammar(grammar) if grammar is not None else None
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)

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
            request_timeout = min(self._timeout, remaining)

            try:
                response = self._client.post(
                    f"/v1/generate/{safe_model}",
                    content=body,
                    headers=headers,
                    timeout=request_timeout,
                )
            except httpx.ConnectError as e:
                # ``ConnectError`` fails *before* the request is sent, so no
                # generation could have started — safe to retry.
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                # Unlike the idempotent encode/score/extract paths, generation
                # is NOT idempotent and carries no dedup key. By the time these
                # mid-flight errors fire (read/write timeout, peer reset) the
                # request body has already been sent and the worker may be — or
                # have finished — generating. Retrying would issue a *second*
                # billable generation with a different completion, so surface
                # the error instead of silently re-running.
                if isinstance(e, httpx.TimeoutException):
                    msg = f"Request timed out: {e}"
                else:
                    msg = (
                        f"Connection lost mid-request ({type(e).__name__}); "
                        f"the peer closed the connection before sending a complete response: {e}"
                    )
                raise SIEConnectionError(msg) from e

            raise_if_model_load_failed(response, model=model)

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
                    self._record_retry()
                    time.sleep(actual_delay)
                    continue

                if error_code == MODEL_LOADING_ERROR_CODE:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)
                    retry_after = get_retry_after(response)
                    delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
                    actual_delay = min(delay, timeout - elapsed)
                    self._record_retry()
                    time.sleep(actual_delay)
                    continue
                if error_code == RESOURCE_EXHAUSTED_ERROR_CODE:
                    oom_retries = _handle_oom_retry(
                        response,
                        start_time=start_time,
                        oom_retries=oom_retries,
                        max_oom_retries=max_oom_retries,
                        timeout=timeout,
                        model=model,
                    )
                    self._record_retry()
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
        result = _parse_generate_result(data, request=parse_request_metadata(response.headers))
        attach_request_metadata([result], response.headers)
        return result

    def responses(
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
        """Create one non-streaming OpenAI-compatible Responses result.

        The gateway's Responses MVP is stateless and text-only. ``input`` may
        be a raw prompt or a sequence of role/content messages; content-part
        arrays accept text parts only. Streaming, tools, stateful threading,
        reasoning, metadata, instructions, and multimodal parts are not
        supported by this surface.

        Generation is non-idempotent. Only explicit pre-execution 503
        PROVISIONING / MODEL_LOADING / RESOURCE_EXHAUSTED responses and
        connect-before-send failures are retried. Mid-flight transport errors
        and post-publish 504 responses are terminal so a retry cannot
        double-bill or create a second completion.

        Args:
            model: Model name to use for generation.
            input: Raw prompt string or typed role/content messages.
            max_output_tokens: Optional hard cap on output tokens.
            temperature: Optional sampling temperature override.
            top_p: Optional nucleus sampling cutoff override.
            seed: Optional signed 64-bit per-request sampling seed.
            gpu: Target GPU type or pool spec; see :meth:`encode`.
            wait_for_capacity: Auto-retry ``503 PROVISIONING`` responses
                under ``provision_timeout_s``. Worker-side model loading and
                resource-exhaustion retries follow their own bounded policies.
            provision_timeout_s: Maximum time to wait for capacity.
            max_oom_retries: Maximum number of worker-side
                ``503 RESOURCE_EXHAUSTED`` retries.

        Returns:
            A typed :class:`ResponseResult`.

        Raises:
            ProvisioningError: If capacity is unavailable and waiting is
                disabled, or the provisioning timeout is exceeded.
            ModelLoadingError: If worker-side model-loading retries time out.
            ResourceExhaustedError: If worker-side OOM retries are exhausted.
            ServerError: On post-publish ``504`` or other server errors.
            SIEConnectionError: If the request cannot be sent safely or the
                connection fails after sending.
        """
        self._reset_retry_count()
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)
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
                response = self._client.post(
                    "/v1/responses",
                    content=body,
                    headers=headers,
                    timeout=min(self._timeout, remaining),
                )
            except httpx.ConnectError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                msg = f"Connection lost mid-request ({type(e).__name__}): {e}"
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
            self._record_retry()
            time.sleep(delay)

        self._check_server_version(response)
        data = parse_terminal_json_object(response, owner="Responses")
        attach_request_metadata([data], response.headers)
        return cast("ResponseResult", data)

    def chat_completions(
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

        Mirrors the subset of OpenAI's ``chat.completions.create`` the gateway
        honours. For token streaming use :meth:`stream_chat_completions`.
        Generation is non-idempotent, so — like :meth:`generate` — only
        pre-execution 503 PROVISIONING / MODEL_LOADING responses are retried; a 504 (post-publish)
        surfaces as :class:`ServerError`.

        Typed kwargs cover the full gateway-supported field set (see
        :func:`build_chat_body` for the canonical list); ``extra_body`` is
        still merged last for forward-compat fields the typed surface does
        not name yet.
        """
        self._reset_retry_count()
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)
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
                response = self._client.post(
                    "/v1/chat/completions",
                    content=body,
                    headers=headers,
                    timeout=min(self._timeout, remaining),
                )
            except httpx.ConnectError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time, timeout=timeout, error_label="Connect error", error=e
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                # Non-idempotent: a mid-flight failure may have already started
                # a generation, so surface it instead of silently re-running.
                msg = f"Connection lost mid-request ({type(e).__name__}): {e}"
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
            self._record_retry()
            time.sleep(delay)

        self._check_server_version(response)
        data = parse_terminal_json_object(response, owner="chat completion")
        attach_request_metadata([data], response.headers)
        return cast("ChatCompletion", data)

    def stream_chat_completions(
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
    ) -> Iterator[ChatCompletionChunk]:
        """Streaming OpenAI-compatible chat completion.

        Yields :class:`ChatCompletionChunk` events. Pass
        ``stream_options={"include_usage": True}`` to receive a final
        usage-only chunk (``choices: []``) before the stream ends. Raises
        :class:`ServerError` if the gateway emits a mid-stream error chunk;
        breaking out of the iterator early closes the stream so the worker
        stops generating.

        ``best_of`` is intentionally not exposed on the streaming surface —
        the gateway rejects ``best_of`` together with ``stream: true``.
        """
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)
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
        yield from self._stream_sse_chunks(
            "/v1/chat/completions",
            body,
            headers,
            model=model,
            resolved_gpu=resolved_gpu,
            wait_for_capacity=wait_for_capacity,
            provision_timeout_s=provision_timeout_s,
            max_oom_retries=max_oom_retries,
        )

    def stream_generate(
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
    ) -> Iterator[GenerateChunk]:
        """Streaming SIE-native generation (``/v1/generate/{model}``).

        Yields :class:`GenerateChunk` events; the terminal chunk carries
        ``done: true`` plus ``usage`` / ``ttft_ms``. Error semantics match
        :meth:`stream_chat_completions`.
        """
        resolved_grammar = validate_generate_grammar(grammar) if grammar is not None else None
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)
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
        yield from self._stream_sse_chunks(
            f"/v1/generate/{safe_model}",
            body,
            headers,
            model=model,
            resolved_gpu=resolved_gpu,
            wait_for_capacity=wait_for_capacity,
            provision_timeout_s=provision_timeout_s,
            max_oom_retries=max_oom_retries,
        )

    def _stream_sse_chunks(
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
    ) -> Iterator[Any]:
        """Open an SSE stream (with pre-stream provisioning retry) and yield chunks.

        Shared by :meth:`stream_chat_completions` and :meth:`stream_generate`.
        Only the *pre-stream* response is retried (503 PROVISIONING / MODEL_LOADING); once bytes start
        flowing a failure is terminal (non-idempotent). Yielding inside the
        ``with`` keeps the stream open while the caller consumes it; an early
        ``break`` tears the context down and the worker sees the disconnect.
        """
        self._reset_retry_count()
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
                with self._client.stream(
                    "POST", url, content=body, headers=headers, timeout=min(self._timeout, remaining)
                ) as response:
                    if response.status_code != 200:
                        # Buffer the body so the decision helper can read the
                        # error code, then either sleep-retry or raise.
                        response.read()
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
                        self._record_retry()
                    else:
                        self._check_server_version(response)
                        for payload in iter_sse_payloads(response.iter_lines()):
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
            except httpx.ConnectError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time, timeout=timeout, error_label="Connect error", error=e
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                # Mid-stream/connection failure: non-idempotent, do not retry.
                msg = f"Connection lost during stream ({type(e).__name__}): {e}"
                raise SIEConnectionError(msg) from e
            # Reached only on the non-200 pre-stream retry path.
            if retry_delay is not None:
                time.sleep(retry_delay)

    # Use overload for proper type hints when single item vs list
    @overload
    def extract(
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
    def extract(
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

    def extract(
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
        """Extract entities or structured data from items.

        Args:
            model: Model name to use for extraction (e.g., "gliner-multi-v2.1").
            items: Single Item or list of Items to extract from.
            labels: Entity types to extract (e.g., ["person", "organization"]).
            output_schema: JSON schema for structured extraction output.
            instruction: Optional instruction for extraction.
            options: Runtime options dict. Can include "profile" to select a named profile.
            gpu: Target GPU type (e.g., "l4", "a100-80gb"). Routes request to workers
                with matching GPU.
            wait_for_capacity: When True (default), auto-retry transient "not
                enough capacity yet" responses under ``provision_timeout_s`` —
                ``503 PROVISIONING`` (scale-from-zero provisioning);
                ``504`` gateway result timeouts for idempotent queue paths; local
                ``httpx`` read/connect/pool timeouts; and transient
                mid-flight transport errors (``RemoteProtocolError``,
                ``ReadError``, ``WriteError`` — peer severed the connection
                before a complete response arrived). Retries honour the
                server's ``Retry-After`` header when present. When False,
                these surface immediately as ``ProvisioningError`` /
                ``ServerError`` / ``SIEConnectionError``.
                Note: ``503 MODEL_LOADING`` and ``503 RESOURCE_EXHAUSTED``
                are retried regardless of this flag — the worker has
                already accepted the request and is loading the target
                model or recovering from transient capacity exhaustion.
                Their budgets are documented under ``ModelLoadingError`` /
                ``ResourceExhaustedError`` below; the
                ``RESOURCE_EXHAUSTED`` branch can be disabled by passing
                ``max_oom_retries=0``.
            provision_timeout_s: Maximum time to wait for capacity when wait_for_capacity=True.
                Default: 900 seconds (15 minutes).
            max_oom_retries: Public retry knob capping the number of
                ``503 RESOURCE_EXHAUSTED`` (server-side OOM) retries. Each
                retry uses bounded exponential backoff
                (``compute_oom_backoff``); the SDK also stops retrying
                early if the next backoff would exhaust
                ``provision_timeout_s``. ``RESOURCE_EXHAUSTED`` retries
                run regardless of ``wait_for_capacity``; set
                ``max_oom_retries=0`` to disable them entirely.
                Default: ``RESOURCE_EXHAUSTED_MAX_RETRIES`` (3).

        Returns:
            ExtractResult if single item was passed, list[ExtractResult] if list was passed.
            Each result contains entities (list of EntityResult) and optional data dict.

        Raises:
            RequestError: If the request is invalid (4xx response).
            ServerError: If the server encounters an error (5xx response).
            SIEConnectionError: If unable to connect to the server.
            ProvisioningError: If ``wait_for_capacity=False`` and the gateway
                returns ``503 PROVISIONING`` (scale-from-zero provisioning), or if provisioning
                retries exceed ``provision_timeout_s``.
            ModelLoadingError: If ``503`` ``MODEL_LOADING`` retries exceed
                ``provision_timeout_s`` during worker-side cold-loading of the
                target model. Note: this branch retries regardless of
                ``wait_for_capacity``.
            ResourceExhaustedError: If ``503`` ``RESOURCE_EXHAUSTED``
                retries exhaust ``max_oom_retries`` or the next backoff
                would exhaust ``provision_timeout_s``. Note: this branch
                retries regardless of ``wait_for_capacity`` unless
                ``max_oom_retries=0``.

        Example:
            >>> # Single item
            >>> result = client.extract(
            ...     "gliner-multi-v2.1",
            ...     {"text": "Apple was founded by Steve Jobs."},
            ...     labels=["person", "organization"],
            ... )
            >>> for entity in result["entities"]:
            ...     print(f"{entity['text']} ({entity['label']})")
            Apple (organization)
            Steve Jobs (person)

            >>> # Batch
            >>> results = client.extract(
            ...     "gliner-multi-v2.1",
            ...     [{"text": "Tesla CEO Elon Musk..."}, {"text": "Google's Sundar Pichai..."}],
            ...     labels=["person", "organization"],
            ... )
        """
        self._reset_retry_count()
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
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)
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
                response = self._client.post(
                    f"/v1/extract/{model}", content=body, headers=headers, timeout=request_timeout
                )
            except httpx.ConnectError as e:
                if wait_for_capacity and is_transient_connect_error(e):
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Connect error",
                        error=e,
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except _RETRYABLE_TRANSPORT_ERRORS as e:
                if wait_for_capacity:
                    delay_s = compute_retry_delay(
                        start_time=start_time,
                        timeout=timeout,
                        error_label="Transient transport error",
                        error=e,
                    )
                    if delay_s is not None:
                        self._record_retry()
                        time.sleep(delay_s)
                        continue
                if isinstance(e, httpx.TimeoutException):
                    msg = f"Request timed out: {e}"
                else:
                    msg = (
                        f"Connection lost mid-request ({type(e).__name__}); "
                        f"the peer closed the connection before sending a complete response: {e}"
                    )
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
                    self._record_retry()
                    time.sleep(actual_delay)
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
                    self._record_retry()
                    time.sleep(actual_delay)
                    continue

                if error_code == RESOURCE_EXHAUSTED_ERROR_CODE:
                    oom_retries = _handle_oom_retry(
                        response,
                        start_time=start_time,
                        oom_retries=oom_retries,
                        max_oom_retries=max_oom_retries,
                        timeout=timeout,
                        model=model,
                    )
                    self._record_retry()
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
                    self._record_retry()
                    time.sleep(actual_delay)
                    continue

            # Handle errors
            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)

            # Success - break out of retry loop
            break

        self._check_server_version(response)

        # Deserialize response
        response_data = msgpack.unpackb(response.content, raw=False)

        # Parse results
        results = parse_extract_results(response_data["items"])
        response_model = response_data.get("model")
        if isinstance(response_model, str) and response_model:
            for result in results:
                result["model"] = response_model

        attach_request_metadata(results, response.headers)

        # Return single result if single item was passed
        return results[0] if single_item else results


# ---------------------------------------------------------------------------
# Jobs + connections namespaces — reuse the client's transport
# and bearer auth. Jobs ride the keyed gateway (`/v1/jobs`); connections ride
# the control plane (`/internal/orgs/{org}/connections`).
# ---------------------------------------------------------------------------


class _SyncNamespace:
    """Shared JSON transport for the sync jobs/connections namespaces."""

    def __init__(self, client: SIEClient) -> None:
        self._c = client

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: Any = None,
        timeout_s: float | None = None,
        include_base_url_headers: bool = True,
    ) -> Any:
        """One JSON request over the client's httpx transport (bearer auth reused)."""
        headers = {"Accept": JSON_CONTENT_TYPE, "Content-Type": JSON_CONTENT_TYPE}
        try:
            response = self._c._client.request(
                method,
                url,
                json=json_body,
                headers=headers,
                timeout=timeout_s if timeout_s is not None else self._c._timeout,
                extensions={"sie_skip_base_url_headers": not include_base_url_headers},
            )
        except httpx.ConnectError as e:
            msg = f"Failed to connect to {url}: {e}"
            raise SIEConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)
        self._c._check_server_version(response)
        if not response.content:
            return {}
        return response.json()


class _SyncJobs(_SyncNamespace):
    """The batch class — ``POST/GET /v1/jobs`` on the keyed gateway.

    ``submit``/``get``/``results`` bind to the shipped ``/v1/jobs`` contract;
    ``list``/``cancel`` bind to the additive ``GET /v1/jobs`` /
    ``POST /v1/jobs/{id}/cancel`` completion of that REST surface (``/v1``
    additive-only — the shipped E2E is the ``sie.process`` submit/get/results
    path).
    """

    def submit(
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
        """Submit a batch job (``POST /v1/jobs``); returns the created-job envelope.

        ``source`` is either inline items (a list, or a bare string = one text
        item) or a connector ``scheme://<connection>/…`` URI (incl.
        the internal ``upload://<file-id>`` push-to-us source); ``sink``
        is ``"return"`` (default), ``"inplace"``, or a connector URI; ``when`` is
        ``"now"`` (default), ``"schedule:<cron>"``, or ``"watch:<source>"``.
        ``field_map`` names the uniform source slots
        (``{"id_field", "input_field", "carry", "input_type"}``) and
        ``output_field`` the sink target — connector jobs only; the
        per-connector URI params keep working as aliases. ``options`` carries
        the per-item options plus the op inputs (operation matrix:
        score → ``options["query"]``, extract → ``options["labels"]`` /
        ``options["output_schema"]``, generate → sampling such as
        ``max_new_tokens``), forwarded as-is.
        """
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
        return self._request_json("POST", "/v1/jobs", json_body=body, timeout_s=max(self._c._timeout, 120.0))

    def get(self, job_id: str) -> JobStatus:
        """Fetch a job's public status doc (``GET /v1/jobs/{id}``)."""
        return self._request_json("GET", f"/v1/jobs/{job_id}")

    def list(self) -> Sequence[JobStatus]:
        """List the org's jobs (``GET /v1/jobs``; scoped to the key's org)."""
        data = self._request_json("GET", "/v1/jobs")
        if isinstance(data, dict):
            return data.get("data", [])
        return data if isinstance(data, list) else []

    def cancel(self, job_id: str) -> JobStatus:
        """Cancel a job (``POST /v1/jobs/{id}/cancel``); the hold's remainder releases."""
        return self._request_json("POST", f"/v1/jobs/{job_id}/cancel")

    def results(self, job_id: str) -> JobResults:
        """Retrieve a finished job's chunk refs and decode the per-item results.

        Reads each succeeded chunk's TTL'd payload-store ref (local path or
        http(s) URL in the POC) and unpacks the msgpack ``WorkResult`` array;
        dense embeddings decode to numpy arrays (like :meth:`encode`).
        """
        job = self.get(job_id)
        chunks = job_chunks(job)
        items = []
        for chunk in chunks:
            ref = chunk.get("ref")
            if chunk.get("state") != "succeeded" or not ref:
                continue
            items.extend(decode_chunk_bytes(self._read_ref(ref)))
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

    def wait(self, job_id: str, *, timeout_s: float = 600.0, poll_s: float = 2.0) -> JobStatus:
        """Poll ``get`` until the job reaches a terminal state or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        while True:
            job = self.get(job_id)
            if job.get("state") in TERMINAL_JOB_STATES:
                return job
            if time.monotonic() >= deadline:
                msg = f"job {job_id} still {job.get('state')!r} after {timeout_s:.0f}s"
                raise RequestError(msg, code="job_wait_timeout", status_code=504)
            time.sleep(poll_s)

    def _read_ref(self, ref: str) -> bytes:
        """Retrieve a chunk's payload-store ref (local path or http(s) URL, POC).

        http(s) refs are fetched without the client's ``Authorization`` header.
        An exact gateway-origin capability ref receives the configured edge
        headers (for example Modal proxy auth); external refs remain bare. Ref
        redirects are never followed, so neither credential can reach a redirect
        target.
        """
        if ref.startswith(("http://", "https://")):
            headers = {"Accept": "application/octet-stream"}
            if self._c._base_url_headers and request_matches_base_url_origin(self._c._base_url, ref):
                headers.update(self._c._base_url_headers)
            response = httpx.get(
                ref,
                headers=headers,
                timeout=self._c._client.timeout,
                follow_redirects=False,
            )
            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)
            return response.content
        path = Path(ref)
        if path.exists():
            return path.read_bytes()
        msg = f"cannot retrieve payload-store ref {ref!r} (POC reads local-path and http(s) refs)"
        raise RequestError(msg, code="bad_ref", status_code=400)


class _SyncConnections(_SyncNamespace):
    """Org-scoped connections (connector auth by name) on the control plane.

    Requires ``control_plane_url`` + ``org`` on the client; the secret is sent on
    ``add`` and is never returned by ``list`` (only the job-runner resolve path
    sees it).
    """

    def _base(self) -> str:
        if not self._c._control_plane_url:
            msg = "connections require control_plane_url on the client: SIEClient(..., control_plane_url=..., org=...)"
            raise ValueError(msg)
        if not self._c._org:
            msg = "connections require org on the client: SIEClient(..., org=...)"
            raise ValueError(msg)
        return f"{self._c._control_plane_url}/internal/orgs/{self._c._org}/connections"

    def add(self, name: str, type: str, secret: str) -> ConnectionCreated:
        """Create an org-scoped connection (connector auth). ``type`` is the connector family."""
        body = {"type": type, "name": name, "secret": secret}
        return self._request_json("POST", self._base(), json_body=body, include_base_url_headers=False)

    def list(self) -> Sequence[Connection]:
        """List the org's active connections (secrets redacted)."""
        data = self._request_json("GET", self._base(), include_base_url_headers=False)
        if isinstance(data, dict):
            return data.get("connections", [])
        return data if isinstance(data, list) else []

    def revoke(self, name: str) -> ConnectionRevoked:
        """Revoke (soft-delete) a connection; frees the name for reuse."""
        return self._request_json("DELETE", f"{self._base()}/{name}", include_base_url_headers=False)


# ---------------------------------------------------------------------------
# Files + batches namespaces — the OpenAI-compatible file/batch
# surface on the keyed gateway. Method names/args mirror `openai.files` /
# `openai.batches` so switching an OpenAI-batch caller to the SDK is mechanical.
# ---------------------------------------------------------------------------


class _SyncFiles(_SyncNamespace):
    """OpenAI-compatible Files API — ``POST/GET /v1/files`` on the keyed gateway.

    Mirrors ``openai.files``: :meth:`upload` (aliased :meth:`create` for a
    byte-for-byte ``openai`` → ``sie_sdk`` swap), :meth:`retrieve`,
    :meth:`content` (raw bytes), and :meth:`delete`. Bytes live in the gateway's
    file store the batch path reads/writes; ``purpose="batch"`` is the only
    purpose the store serves today. :meth:`delete` is the additive OpenAI-parity
    completion of the surface (served when the delete-on-complete seam lands).
    """

    def upload(
        self,
        file: str | Path | bytes | bytearray | IO[bytes],
        *,
        purpose: str = "batch",
        filename: str | None = None,
    ) -> File:
        """Upload a file (``POST /v1/files``); returns the created File object.

        ``file`` is a filesystem path, raw bytes, or a binary file-like object
        (the same inputs OpenAI's ``files.create(file=...)`` accepts);
        ``purpose`` is ``"batch"`` (the only purpose the mode-B store serves).
        """
        content, name = resolve_upload(file, filename)
        query = urlencode({"purpose": purpose, "filename": name})
        headers = {"Accept": JSON_CONTENT_TYPE, "Content-Type": "application/jsonl"}
        try:
            response = self._c._client.post(
                f"/v1/files?{query}",
                content=content,
                headers=headers,
                timeout=max(self._c._timeout, 120.0),
            )
        except httpx.ConnectError as e:
            msg = f"Failed to connect to {self._c._base_url}: {e}"
            raise SIEConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)
        self._c._check_server_version(response)
        return response.json()

    def create(
        self,
        *,
        file: str | Path | bytes | bytearray | IO[bytes],
        purpose: str = "batch",
    ) -> File:
        """OpenAI-exact alias for :meth:`upload` (``files.create(file=, purpose=)``)."""
        return self.upload(file, purpose=purpose)

    def retrieve(self, file_id: str) -> File:
        """Fetch a file's metadata (``GET /v1/files/{id}``)."""
        return self._request_json("GET", f"/v1/files/{file_id}")

    def content(self, file_id: str) -> bytes:
        """Download a file's raw bytes (``GET /v1/files/{id}/content``)."""
        try:
            response = self._c._client.get(
                f"/v1/files/{file_id}/content",
                headers={"Accept": "application/jsonl"},
            )
        except httpx.ConnectError as e:
            msg = f"Failed to connect to {self._c._base_url}: {e}"
            raise SIEConnectionError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Request timed out: {e}"
            raise SIEConnectionError(msg) from e
        if response.status_code >= HTTP_CLIENT_ERROR:
            handle_error(response)
        return response.content

    def delete(self, file_id: str) -> FileDeleted:
        """Delete a file (``DELETE /v1/files/{id}``; additive OpenAI-parity surface)."""
        return self._request_json("DELETE", f"/v1/files/{file_id}")


class _SyncBatches(_SyncNamespace):
    """OpenAI-compatible Batch API — ``POST/GET /v1/batches`` on the keyed gateway.

    Mirrors ``openai.batches``: :meth:`create`, :meth:`retrieve`, :meth:`list`,
    :meth:`cancel`. A batch is a job over an uploaded file's JSONL lines run
    on the batch lane (same ``spawn_encode_job`` engine as
    ``/v1/jobs``). :meth:`list` / :meth:`cancel` are the additive OpenAI-parity
    completion of the surface (served when the batch store hardens; the shipped path is
    create → retrieve → download the output file via ``client.files.content``).
    """

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str = "/v1/embeddings",
        completion_window: str = "24h",
        metadata: dict[str, Any] | None = None,
    ) -> Batch:
        """Create a batch (``POST /v1/batches``); returns the Batch object.

        ``input_file_id`` is a ``file-…`` id from :meth:`SIEClient.files.upload`
        with ``purpose="batch"``; ``endpoint`` is ``/v1/embeddings`` (the only
        endpoint the encode-only jobs POC serves).
        """
        body: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata is not None:
            body["metadata"] = metadata
        return self._request_json("POST", "/v1/batches", json_body=body, timeout_s=max(self._c._timeout, 120.0))

    def retrieve(self, batch_id: str) -> Batch:
        """Fetch a batch's status (``GET /v1/batches/{id}``)."""
        return self._request_json("GET", f"/v1/batches/{batch_id}")

    def list(self) -> Sequence[Batch]:
        """List the org's batches (``GET /v1/batches``; additive OpenAI-parity)."""
        data = self._request_json("GET", "/v1/batches")
        if isinstance(data, dict):
            return data.get("data", [])
        return data if isinstance(data, list) else []

    def cancel(self, batch_id: str) -> Batch:
        """Cancel a batch (``POST /v1/batches/{id}/cancel``; additive OpenAI-parity)."""
        return self._request_json("POST", f"/v1/batches/{batch_id}/cancel")
