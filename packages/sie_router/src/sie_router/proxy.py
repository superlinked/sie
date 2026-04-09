import asyncio
import hmac
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import msgpack
import msgpack_numpy
import orjson
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from sie_sdk.types import PoolListItem, PoolResponse

from sie_router.health import CONFIGURED_GPU_TYPES
from sie_router.metrics import REQUEST_COUNT, REQUEST_LATENCY, record_pending_demand
from sie_router.model_registry import (
    BundleConflictError,
    ModelNotFoundError,
    ModelRegistry,
    parse_model_spec,
)
from sie_router.pools import DEFAULT_POOL_NAME, DefaultPoolProtectedError, PoolManager, parse_gpu_param
from sie_router.registry import WorkerRegistry
from sie_router.responses import pool_to_list_item, pool_to_response
from sie_router.types import AuditEntry, ProvisioningResponse
from sie_router.version import ROUTER_VERSION, SDK_VERSION_HEADER
from sie_router.work_publisher import NoConsumersError, WorkPublisher

# NOTE: We do NOT call msgpack_numpy.patch() globally.  That monkey-patches
# msgpack.packb/unpackb with numpy-aware wrappers that add overhead on every
# call (~0.6% of total CPU).  The router never serializes numpy arrays; it
# passes worker result blobs through as raw bytes on the msgpack response path.
# Only the JSON response path needs numpy deserialization, and that uses
# msgpack_numpy.unpackb explicitly (see ``_unpack_with_numpy``).

# Cluster routing mode: "direct" (HTTP forwarding, default) or "queue" (NATS JetStream).
# Read once at import time; override via monkeypatch in tests.
CLUSTER_ROUTING_MODE = os.environ.get("SIE_CLUSTER_ROUTING", "direct").strip().lower()

# Maximum recursion depth for _convert_numpy_for_json to prevent stack overflow
_MAX_CONVERSION_DEPTH = 128

# Maximum msgpack buffer size for untrusted data (256 MB)
_MAX_MSGPACK_BUFFER_SIZE = 256 * 1024 * 1024

# Payloads below this threshold are parsed directly on the event loop
# (json.loads / msgpack.unpackb on small data takes <30μs — less than the
# ~50μs overhead of asyncio.to_thread).  Larger payloads are offloaded.
_INLINE_PARSE_THRESHOLD = 65536  # 64 KB


def _response_content_key(operation: str) -> str:
    """Return the top-level response key for the given operation.

    Score responses use ``"scores"`` (matching ``ScoreResult``);
    all other operations use ``"items"``.
    """
    return "scores" if operation == "score" else "items"


def _safe_msgpack_unpack(data: bytes) -> Any:
    """Unpack msgpack data with size limits to prevent memory exhaustion.

    Uses ``msgpack.unpackb`` directly instead of the streaming
    ``Unpacker`` class — more efficient for single-shot deserialization.
    """
    if len(data) > _MAX_MSGPACK_BUFFER_SIZE:
        raise ValueError(f"msgpack payload too large: {len(data)} bytes (max {_MAX_MSGPACK_BUFFER_SIZE})")
    return msgpack.unpackb(data, raw=False)


logger = logging.getLogger(__name__)


def _unpack_with_numpy(data: bytes) -> Any:
    """Unpack msgpack data that may contain numpy ext types (worker results).

    Uses ``msgpack_numpy.unpackb`` directly instead of the global monkey-patch
    so the hot-path ``msgpack.unpackb`` stays fast for non-numpy data.
    Only called on the JSON response path (inside ``asyncio.to_thread``).
    """
    if len(data) > _MAX_MSGPACK_BUFFER_SIZE:
        raise ValueError(f"msgpack payload too large: {len(data)} bytes (max {_MAX_MSGPACK_BUFFER_SIZE})")
    return msgpack_numpy.unpackb(data, raw=False)


def _convert_numpy_for_json(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    import numpy as np

    if _depth > _MAX_CONVERSION_DEPTH:
        return obj  # Bail out to avoid stack overflow
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy_for_json(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_for_json(v, _depth + 1) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


class _AsyncIterStream(httpx.AsyncByteStream):
    """Wrap an async iterator for httpx content= compatibility."""

    def __init__(self, it: AsyncIterator[bytes]) -> None:
        self._it = it

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._it:
            yield chunk

    async def aclose(self) -> None:
        return


audit_logger = logging.getLogger("sie_router.audit")

router = APIRouter(tags=["proxy"])

# Headers that should not be forwarded
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
}

# Default Retry-After for 202 responses (seconds)
DEFAULT_RETRY_AFTER = 120

# Auth configuration (router-level static token)
AUTH_MODE = os.environ.get("SIE_AUTH_MODE", "").strip().lower() or "none"
_TOKENS = os.environ.get("SIE_AUTH_TOKENS", "") or os.environ.get("SIE_AUTH_TOKEN", "")
AUTH_TOKENS = {t.strip() for t in _TOKENS.split(",") if t.strip()}
AUTH_EXEMPT_PATHS = {"/health", "/healthz", "/readyz", "/metrics"}

_sdk_version_warned: set[str] = set()


def _check_sdk_version(request: Request) -> None:
    sdk_version = request.headers.get(SDK_VERSION_HEADER)
    if not sdk_version:
        return
    try:
        sdk_parts = sdk_version.split(".")
        router_parts = ROUTER_VERSION.split(".")
        if len(sdk_parts) < 2 or len(router_parts) < 2:
            return
        sdk_major, sdk_minor = int(sdk_parts[0]), int(sdk_parts[1])
        router_major, router_minor = int(router_parts[0]), int(router_parts[1])
        key = f"{sdk_major}.{sdk_minor}"
        if key in _sdk_version_warned:
            return
        if sdk_major != router_major or abs(sdk_minor - router_minor) > 1:
            logger.warning(
                "SDK version skew: client sent %s, router is %s",
                sdk_version,
                ROUTER_VERSION,
            )
            _sdk_version_warned.add(key)
            # Cap to prevent unbounded growth from malicious version strings
            if len(_sdk_version_warned) > 100:
                _sdk_version_warned.clear()
    except (ValueError, IndexError):
        pass


def _mask_token(token: str) -> str:
    """Mask a token for audit logging, showing only last 4 characters."""
    if len(token) <= 4:
        return "****"
    return f"****{token[-4:]}"


def _require_auth(request: Request) -> None:
    """Enforce router-level auth when configured."""
    if AUTH_MODE != "static":
        request.state.token_id = None
        return

    if request.url.path in AUTH_EXEMPT_PATHS:
        return

    if not AUTH_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Router auth enabled but no tokens configured"},
        )

    header = request.headers.get("authorization", "").strip()
    if not header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Missing Authorization header"},
        )

    token = header
    if header.lower().startswith("bearer "):
        token = header[7:].strip()

    if not any(hmac.compare_digest(token, valid) for valid in AUTH_TOKENS):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Invalid token"},
        )

    request.state.token_id = _mask_token(token)


def _emit_audit_log(
    request: Request,
    *,
    endpoint: str,
    model: str | None = None,
    pool: str | None = None,
    gpu: str | None = None,
    worker: str | None = None,
    status: int | str,
    latency_ms: float | None = None,
    body_bytes: int | None = None,
    event: str = "api_request",
) -> None:
    """Emit a structured audit log entry for an API request."""
    token_id = getattr(request.state, "token_id", None)
    method = request.method
    path = request.url.path
    resolved_status = int(status) if isinstance(status, str) else status

    # Build human-readable message
    parts = [method, path, str(status)]
    if latency_ms is not None:
        parts.append(f"{latency_ms:.1f}ms")
    message = " ".join(parts)

    entry = AuditEntry(
        event=event,
        method=method,
        endpoint=endpoint,
        status=resolved_status,
        token_id=token_id,
        model=model,
        pool=pool,
        gpu=gpu,
        worker=worker,
        latency_ms=latency_ms,
        body_bytes=body_bytes,
    )

    audit_logger.info(message, extra=entry.to_dict())


# Headers stripped when proxying streamed bodies (in addition to hop-by-hop)
STREAMED_BODY_STRIP_HEADERS = HOP_BY_HOP_HEADERS | {"content-length"}


def _filter_headers(headers: dict[str, str], *, strip_content_length: bool = False) -> dict[str, str]:
    """Filter out hop-by-hop headers for forwarding."""
    excluded = STREAMED_BODY_STRIP_HEADERS if strip_content_length else HOP_BY_HOP_HEADERS
    return {k: v for k, v in headers.items() if k.lower() not in excluded}


def _resolve_machine_profile(gpu: str, configured_gpu_types: list[str]) -> str:
    """Resolve a bare GPU type to its spot variant if one is configured.

    Users may send ``X-SIE-MACHINE-PROFILE: l4`` but the cluster only provisions
    spot workers (``l4-spot``).  This function transparently maps the bare name
    to the ``-spot`` variant when:
      1. The bare name itself is **not** in the configured list, **and**
      2. A ``<gpu>-spot`` entry **is** in the configured list.

    All comparisons are case-insensitive; the returned value preserves the case
    of the configured entry so that downstream metrics/KEDA labels match exactly.
    """
    cfg_lower = {g.lower(): g for g in configured_gpu_types}
    if gpu.lower() in cfg_lower:
        return cfg_lower[gpu.lower()]
    spot_key = f"{gpu.lower()}-spot"
    if spot_key in cfg_lower:
        logger.info("Resolved machine_profile '%s' → '%s'", gpu, cfg_lower[spot_key])
        return cfg_lower[spot_key]
    return gpu


def _make_provisioning_response(gpu: str) -> JSONResponse:
    """Create a 202 Accepted response for provisioning.

    Returns 202 with Retry-After header when GPU capacity is being provisioned.
    """
    response = ProvisioningResponse(
        status="provisioning",
        gpu=gpu,
        estimated_wait_s=180,
        message=f"No worker available for GPU type '{gpu}'. Provisioning in progress.",
    )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": response.status,
            "gpu": response.gpu,
            "estimated_wait_s": response.estimated_wait_s,
            "message": response.message,
        },
        headers={"Retry-After": str(DEFAULT_RETRY_AFTER)},
    )


def _make_unconfigured_gpu_response(gpu: str, configured: list[str]) -> JSONResponse:
    """Create a 503 response for GPU type not configured in cluster.

    Returns 503 when the requested GPU type is not available and will never be provisioned.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "gpu_not_configured",
            "gpu": gpu,
            "configured_gpu_types": configured,
            "message": f"GPU type '{gpu}' is not configured in this cluster. "
            f"Available types: {', '.join(configured) if configured else 'none'}. "
            "Contact your administrator to add this GPU type.",
        },
    )


async def _forward_request(
    http_client: httpx.AsyncClient,
    worker_url: str,
    method: str,
    path: str,
    headers: dict[str, str],
    request_stream: AsyncIterator[bytes],
    timeout_s: float = 300.0,
) -> Response:
    """Forward a request to a worker.

    Args:
        http_client: Shared httpx client for connection pooling.
        worker_url: Worker base URL.
        method: HTTP method.
        path: Request path.
        headers: Request headers (filtered).
        request_stream: Async iterator of request body chunks (streamed to worker).
        timeout_s: Request timeout in seconds.

    Returns:
        Response from the worker.
    """
    url = f"{worker_url}{path}"

    req = http_client.build_request(
        method=method,
        url=url,
        headers=headers,
        content=_AsyncIterStream(request_stream),
        timeout=httpx.Timeout(timeout_s),
    )

    response = await http_client.send(req, stream=True)

    # Filter response headers
    response_headers = _filter_headers(dict(response.headers), strip_content_length=True)

    async def _stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        except httpx.StreamError:
            logger.error("Stream interrupted from %s", url)
        finally:
            await response.aclose()

    return StreamingResponse(
        content=_stream_body(),
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.headers.get("content-type"),
    )


async def _proxy_request(
    request: Request,
    model: str,
    path: str,
    x_machine_profile: str | None = None,
    x_sie_pool: str | None = None,
) -> Response:
    """Common proxy logic for encode/score/extract endpoints.

    Args:
        request: Incoming FastAPI request.
        model: Model name from path (may include bundle prefix).
        path: Full path to forward.
        x_machine_profile: Machine profile header (may include pool: "pool_name/profile").
        x_sie_pool: Pool name header (alternative to embedding in GPU param).

    Returns:
        Response from worker, 202 if provisioning, 404 if model unknown,
        409 if bundle incompatible, or 503 if GPU not configured.
    """
    _require_auth(request)
    _check_sdk_version(request)
    registry: WorkerRegistry = request.app.state.registry
    pool_manager: PoolManager | None = getattr(request.app.state, "pool_manager", None)
    model_registry: ModelRegistry | None = getattr(request.app.state, "model_registry", None)

    # Parse bundle from model spec (e.g., "default:/org/model" -> bundle="default", model="org/model")
    bundle_override, model_name = parse_model_spec(model)

    # Resolve bundle using ModelRegistry (priority-based or explicit override)
    # Fast-fail for unknown models (404) or incompatible overrides (409)
    # These errors do NOT trigger autoscaling
    if model_registry is not None:
        try:
            bundle = model_registry.resolve_bundle(model_name, bundle_override)
        except ModelNotFoundError:
            logger.warning("Model not found: %s", model_name)
            endpoint = path.split("/")[2] if len(path.split("/")) > 2 else "unknown"
            _emit_audit_log(request, endpoint=endpoint, model=model_name, status=404)
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "model_not_found",
                    "model": model_name,
                    "message": f"Model '{model_name}' is not available in any bundle. "
                    "Check /v1/models for available models.",
                },
            )
        except BundleConflictError as e:
            logger.warning(
                "Bundle '%s' does not support model '%s' (compatible: %s)",
                bundle_override,
                model_name,
                e.compatible_bundles,
            )
            endpoint = path.split("/")[2] if len(path.split("/")) > 2 else "unknown"
            _emit_audit_log(request, endpoint=endpoint, model=model_name, status=409)
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "error": "bundle_conflict",
                    "model": model_name,
                    "requested_bundle": bundle_override,
                    "compatible_bundles": e.compatible_bundles,
                    "message": f"Bundle '{bundle_override}' does not support model '{model_name}'. "
                    f"Compatible bundles: {', '.join(e.compatible_bundles)}.",
                },
            )
    else:
        # Fallback when ModelRegistry not initialized (e.g., during startup or tests)
        bundle = bundle_override or "default"

    # Parse pool from GPU param (e.g., "eval-l4/l4" -> pool="eval-l4", gpu="l4")
    pool_name = x_sie_pool
    gpu = x_machine_profile

    if gpu and "/" in gpu:
        parsed_pool, parsed_gpu = parse_gpu_param(gpu)
        if parsed_pool:
            pool_name = parsed_pool
            gpu = parsed_gpu

    # Resolve bare GPU types to spot variants (e.g., "l4" → "l4-spot")
    # when the spot variant exists in configured GPU types.
    if gpu and CONFIGURED_GPU_TYPES:
        gpu = _resolve_machine_profile(gpu, CONFIGURED_GPU_TYPES)

    # Use default pool if none specified and pool manager is available
    effective_pool = pool_name or DEFAULT_POOL_NAME

    # Validate GPU is configured in cluster (fast rejection for unconfigured GPUs)
    if gpu and CONFIGURED_GPU_TYPES:
        configured_lower = {g.lower() for g in CONFIGURED_GPU_TYPES}
        if gpu.lower() not in configured_lower:
            logger.warning("GPU type '%s' not configured (available: %s)", gpu, CONFIGURED_GPU_TYPES)
            endpoint = path.split("/")[2] if len(path.split("/")) > 2 else "unknown"
            _emit_audit_log(
                request,
                endpoint=endpoint,
                model=model_name,
                gpu=gpu,
                pool=effective_pool,
                status=503,
            )
            return _make_unconfigured_gpu_response(gpu, CONFIGURED_GPU_TYPES)

    # Queue mode: publish to NATS JetStream instead of forwarding to worker.
    # Workers subscribe to pool-level streams keyed by their SIE_POOL env var
    # (e.g., "l4-spot-default").  Resolve the NATS pool name from the worker
    # registry so the published subject matches the worker's subscription.
    if CLUSTER_ROUTING_MODE == "queue":
        queue_pool = pool_name or registry.resolve_queue_pool(bundle, gpu) or gpu or effective_pool or "_default"
        return await _queue_request(
            request=request,
            model_name=model_name,
            profile_id="default",  # Profile resolved by workers from model config
            bundle=bundle,
            pool_name=queue_pool,
            machine_profile=gpu or "",
            path=path,
        )

    # Pool-aware routing (explicit pool or default pool)
    worker = None
    if pool_manager:
        # Use explicit pool if specified, otherwise use default pool
        routing_pool_name = pool_name or DEFAULT_POOL_NAME
        pool = pool_manager.get_pool(routing_pool_name)
        if pool and pool.is_active:
            # Route to workers in the pool
            pool_workers = pool.get_workers_for_gpu(gpu) if gpu else pool.status.assigned_workers
            if pool_workers:
                # Select best worker within the pool (model affinity, bundle, load balance)
                pool_worker_urls = {w.url for w in pool_workers}
                worker = registry.select_worker(
                    gpu=gpu,
                    bundle=bundle,
                    model=model_name,
                    worker_urls=pool_worker_urls,
                )
            if worker is None:
                logger.info("Pool %s has no workers for GPU %s, bundle %s", routing_pool_name, gpu, bundle)
        elif pool and not pool.is_active:
            logger.info("Pool %s is not active (state: %s)", routing_pool_name, pool.status.state)
        elif routing_pool_name != DEFAULT_POOL_NAME:
            # Only log warning for explicitly requested pools, not for default
            logger.warning("Pool %s not found", routing_pool_name)

    # Fall back to general routing if pool routing failed or no pool manager
    if worker is None:
        worker = registry.select_worker(gpu=gpu, bundle=bundle, model=model_name)

    if worker is None:
        # No worker available - determine GPU type for demand recording
        demand_gpu = gpu

        # If no explicit GPU but pool is specified, extract GPU from pool spec
        if not demand_gpu and effective_pool and pool_manager:
            pool = pool_manager.get_pool(effective_pool)
            if pool and pool.spec.gpus:
                # Use the first GPU type from the pool spec
                demand_gpu = next(iter(pool.spec.gpus.keys()))
                if CONFIGURED_GPU_TYPES:
                    demand_gpu = _resolve_machine_profile(demand_gpu, CONFIGURED_GPU_TYPES)
                logger.debug("Extracted GPU '%s' from pool '%s' spec", demand_gpu, effective_pool)

        if demand_gpu:
            # GPU/machine_profile specified (or derived from pool) but no capacity - return 202
            logger.info(
                "No worker for machine_profile %s, bundle %s (pool=%s), returning 202",
                demand_gpu,
                bundle,
                effective_pool,
            )
            # Record demand for KEDA scale-from-zero with machine_profile, bundle, pool labels
            # demand_gpu is the machine_profile (e.g., "l4-spot", "a100-40gb")
            # effective_pool is the pool name (enables per-pool demand tracking for scaling)
            record_pending_demand(demand_gpu, bundle, effective_pool or "default")
            endpoint = path.split("/")[2] if len(path.split("/")) > 2 else "unknown"
            _emit_audit_log(
                request,
                endpoint=endpoint,
                model=model_name,
                gpu=demand_gpu,
                pool=effective_pool,
                status=202,
            )
            return _make_provisioning_response(demand_gpu)

        # No GPU specified and no pool with GPU spec - return 503
        logger.warning("No healthy workers available")
        endpoint = path.split("/")[2] if len(path.split("/")) > 2 else "unknown"
        _emit_audit_log(request, endpoint=endpoint, model=model_name, pool=effective_pool, status=503)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "No healthy workers available"},
        )

    # Forward request to worker
    http_client: httpx.AsyncClient = request.app.state.http_client
    headers = _filter_headers(dict(request.headers), strip_content_length=True)
    body_bytes = 0

    async def counting_stream() -> AsyncIterator[bytes]:
        nonlocal body_bytes
        async for chunk in request.stream():
            body_bytes += len(chunk)
            yield chunk

    # Rebuild path with model_name (without bundle prefix) for the worker
    # Original path format: /v1/{endpoint}/{model_with_possible_bundle}
    # We need: /v1/{endpoint}/{model_name}
    if bundle_override:
        # Strip bundle prefix from path for forwarding to worker
        forward_path = path.replace(f"{bundle_override}:/", "", 1)
    else:
        forward_path = path

    logger.debug("Forwarding %s to worker %s (bundle=%s, pool=%s)", forward_path, worker.url, bundle, pool_name)

    # Extract endpoint name for metrics (e.g., "/v1/encode/model" -> "encode")
    endpoint = forward_path.split("/")[2] if len(forward_path.split("/")) > 2 else "unknown"
    machine_profile = worker.machine_profile or "unknown"
    start_time = time.monotonic()
    status_code = "500"

    try:
        response = await _forward_request(
            http_client=http_client,
            worker_url=worker.url,
            method=request.method,
            path=forward_path,
            headers=headers,
            request_stream=counting_stream(),
        )

        # Record request for QPS tracking
        registry.record_request(worker.url)
        status_code = str(response.status_code)

        # Add worker identification header for per-worker metrics tracking
        response.headers["X-SIE-Worker"] = worker.name or worker.url

        return response

    except httpx.TimeoutException:
        status_code = "504"
        logger.error("Request to %s timed out", worker.url)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"message": "Request to worker timed out"},
        ) from None
    except httpx.RequestError as e:
        status_code = "502"
        logger.error("Error forwarding to %s: %s", worker.url, e)
        # Mark worker unhealthy on connection error
        await registry.mark_unhealthy(worker.url)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": f"Worker connection error: {e}"},
        ) from e
    finally:
        elapsed = time.monotonic() - start_time
        REQUEST_COUNT.labels(endpoint=endpoint, status=status_code, machine_profile=machine_profile).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, machine_profile=machine_profile).observe(elapsed)
        _emit_audit_log(
            request,
            endpoint=endpoint,
            model=model_name,
            pool=effective_pool,
            gpu=gpu,
            worker=worker.name,
            status=status_code,
            latency_ms=round(elapsed * 1000, 1),
            body_bytes=body_bytes,
        )


def _compute_timing_headers(
    results: list[Any],
    publish_ms: float,
    wait_ms: float,
) -> dict[str, str]:
    """Compute aggregate timing headers from queue-mode results."""
    headers: dict[str, str] = {}

    # Router-side timing
    headers["X-Queue-Publish-Time"] = f"{publish_ms:.1f}"
    headers["X-Queue-Wait-Time"] = f"{wait_ms:.1f}"

    # Worker-side timing (aggregate across all items)
    successful = [r for r in results if r.get("success")]
    if successful:
        # Use max values (worst-case latency across items)
        queue_ms = max(r.get("queue_ms", 0) for r in successful)
        inference_ms = max(r.get("inference_ms", 0) for r in successful)
        tokenization_ms = max(r.get("tokenization_ms", 0) for r in successful)
        postprocessing_ms = max(r.get("postprocessing_ms", 0) for r in successful)
        payload_fetch_ms = max(r.get("payload_fetch_ms", 0) for r in successful)

        headers["X-Queue-Time"] = f"{queue_ms:.1f}"
        headers["X-Inference-Time"] = f"{inference_ms:.1f}"
        if tokenization_ms > 0:
            headers["X-Tokenization-Time"] = f"{tokenization_ms:.1f}"
        if postprocessing_ms > 0:
            headers["X-Postprocessing-Time"] = f"{postprocessing_ms:.1f}"
        if payload_fetch_ms > 0:
            headers["X-Payload-Fetch-Time"] = f"{payload_fetch_ms:.1f}"

    return headers


async def _queue_request(
    request: Request,
    model_name: str,
    profile_id: str,
    bundle: str,
    pool_name: str,
    machine_profile: str,
    path: str,
) -> Response:
    """Handle request via NATS JetStream work queue (cluster queue mode).

    Parses the request body, decomposes it into work items, publishes to
    JetStream, and waits for results. This replaces ``_forward_request``
    when ``SIE_CLUSTER_ROUTING=queue``.

    """
    work_publisher: WorkPublisher | None = getattr(request.app.state, "work_publisher", None)
    if work_publisher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "nats_unavailable", "message": "NATS JetStream not configured"},
        )

    # Parse request body
    body = await request.body()
    content_type = request.headers.get("content-type", "")

    try:
        if len(body) <= _INLINE_PARSE_THRESHOLD:
            # Small payload: parse on the event loop (faster than thread hop)
            if "msgpack" in content_type:
                parsed = _safe_msgpack_unpack(body)
            else:
                parsed = orjson.loads(body)
        elif "msgpack" in content_type:
            parsed = await asyncio.to_thread(_safe_msgpack_unpack, body)
        else:
            if len(body) > _MAX_MSGPACK_BUFFER_SIZE:
                raise ValueError(f"JSON payload too large: {len(body)} bytes (max {_MAX_MSGPACK_BUFFER_SIZE})")
            parsed = await asyncio.to_thread(orjson.loads, body)
    except (msgpack.UnpackValueError, orjson.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "parse_error", "message": f"Invalid request body: {e}"},
        ) from e

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "parse_error", "message": "Request body must be a JSON object"},
        )
    if "items" in parsed and not isinstance(parsed["items"], list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "parse_error", "message": "'items' must be an array"},
        )
    if "query" in parsed and not isinstance(parsed["query"], dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "parse_error", "message": "'query' must be an object"},
        )

    # Compute expected bundle_config_hash for stale-worker gating
    model_registry: ModelRegistry | None = getattr(request.app.state, "model_registry", None)
    bundle_config_hash = ""
    if model_registry is not None:
        try:
            bundle_config_hash = model_registry.compute_bundle_config_hash(bundle)
        except Exception:  # noqa: BLE001
            logger.debug("Could not compute bundle_config_hash for %s", bundle)

    # Determine operation from path
    parts = path.strip("/").split("/")
    operation = parts[1] if len(parts) > 1 else "encode"

    endpoint = operation
    start_time = time.monotonic()
    status_code = "500"

    publish_elapsed_ms = 0.0
    wait_elapsed_ms = 0.0

    try:
        if operation == "encode":
            items = parsed.get("items", [])
            params = parsed.get("params") or {}
            publish_start = time.monotonic()
            submit_task = asyncio.ensure_future(
                work_publisher.submit_encode(
                    model_id=model_name,
                    profile_id=profile_id,
                    pool_name=pool_name,
                    machine_profile=machine_profile,
                    items=items,
                    output_types=params.get("output_types"),
                    instruction=params.get("instruction"),
                    is_query=params.get("is_query", False),
                    options=params.get("options"),
                    bundle_config_hash=bundle_config_hash,
                )
            )
            results = await submit_task
            total_elapsed_ms = (time.monotonic() - publish_start) * 1000
            publish_elapsed_ms = total_elapsed_ms * 0.1  # Approximate: publish is ~10% of total
            wait_elapsed_ms = total_elapsed_ms
        elif operation == "score":
            query = parsed.get("query", {})
            items = parsed.get("items", [])
            publish_start = time.monotonic()
            submit_task = asyncio.ensure_future(
                work_publisher.submit_score(
                    model_id=model_name,
                    profile_id=profile_id,
                    pool_name=pool_name,
                    machine_profile=machine_profile,
                    query=query,
                    items=items,
                    instruction=parsed.get("instruction"),
                    options=parsed.get("options"),
                    bundle_config_hash=bundle_config_hash,
                )
            )
            results = await submit_task
            total_elapsed_ms = (time.monotonic() - publish_start) * 1000
            publish_elapsed_ms = total_elapsed_ms * 0.1
            wait_elapsed_ms = total_elapsed_ms
        elif operation == "extract":
            items = parsed.get("items", [])
            params = parsed.get("params") or {}
            publish_start = time.monotonic()
            submit_task = asyncio.ensure_future(
                work_publisher.submit_extract(
                    model_id=model_name,
                    profile_id=profile_id,
                    pool_name=pool_name,
                    machine_profile=machine_profile,
                    items=items,
                    labels=params.get("labels"),
                    output_schema=params.get("output_schema"),
                    instruction=params.get("instruction"),
                    options=params.get("options"),
                    bundle_config_hash=bundle_config_hash,
                )
            )
            results = await submit_task
            total_elapsed_ms = (time.monotonic() - publish_start) * 1000
            publish_elapsed_ms = total_elapsed_ms * 0.1
            wait_elapsed_ms = total_elapsed_ms
        else:
            raise HTTPException(status_code=400, detail={"error": f"Unknown operation: {operation}"})

        # Reassemble response from work results.
        # Each result_msgpack is an opaque msgpack blob from the worker,
        # serialized with msgpack-numpy (contains numpy ext types).
        result_blobs: list[bytes] = []
        errors = []
        for r in results:
            if r.get("success"):
                raw_blob: bytes | None = r.get("result_msgpack")
                if raw_blob is None:
                    # Worker reported success but omitted result payload — treat as error
                    logger.warning(
                        "Work item %s reported success but missing result_msgpack",
                        r.get("work_item_id", "unknown"),
                    )
                    errors.append(
                        {
                            "item_index": r.get("item_index"),
                            "error": "Worker reported success but returned no result payload",
                        }
                    )
                else:
                    result_blobs.append(raw_blob)
            else:
                errors.append({"item_index": r.get("item_index"), "error": r.get("error")})

        if errors and not result_blobs:
            # All items failed
            status_code = "500"
            return JSONResponse(
                status_code=500,
                content={"error": "all_items_failed", "details": errors},
            )

        status_code = "200"

        # Compute aggregate timing headers from worker results
        timing_headers = _compute_timing_headers(results, publish_elapsed_ms, wait_elapsed_ms)
        response_headers = {"X-SIE-Version": ROUTER_VERSION, **timing_headers}

        # Return as msgpack if client accepts it, otherwise JSON
        if "msgpack" in request.headers.get("accept", ""):
            # Build the response at the byte level to avoid deserializing
            # the worker blobs (which contain numpy ext types).
            # Manually assemble: {"model": ..., "scores"|"items": [...], "errors"?: ...}
            packer = msgpack.Packer(use_bin_type=True, autoreset=False)
            n_keys = 3 if errors else 2
            packer.pack_map_header(n_keys)
            packer.pack("model")
            packer.pack(model_name)
            packer.pack(_response_content_key(operation))
            if len(result_blobs) == 1 and operation == "score":
                # Score returns a single blob containing the complete list;
                # write it directly to avoid wrapping in an extra array.
                parts = [packer.bytes()]
                packer.reset()
                parts.append(result_blobs[0])
            else:
                packer.pack_array_header(len(result_blobs))
                parts = [packer.bytes()]
                packer.reset()
                # Embed each result blob directly — they are already valid msgpack values
                parts.extend(result_blobs)
            if errors:
                packer.pack("errors")
                packer.pack(errors)
                parts.append(packer.bytes())
                packer.reset()
            response_bytes = b"".join(parts)
            return Response(
                content=response_bytes,
                media_type="application/x-msgpack",
                headers=response_headers,
            )

        # JSON path: must deserialize blobs (JSON can't embed binary).
        # Offload to thread pool — this involves msgpack deserialization,
        # numpy→list conversion, and is CPU-bound for large batches.
        def _build_json_items(blobs: list[bytes], is_score: bool) -> list:
            items = []
            for blob in blobs:
                item_data = _unpack_with_numpy(blob)
                items.append(_convert_numpy_for_json(item_data))
            # Score returns a single blob containing the complete list;
            # unwrap to avoid nesting it in an extra array.
            if is_score and len(items) == 1 and isinstance(items[0], list):
                return items[0]
            return items

        result_items = await asyncio.to_thread(_build_json_items, result_blobs, operation == "score")
        response_data: dict[str, Any] = {
            "model": model_name,
            _response_content_key(operation): result_items,
        }
        if errors:
            response_data["errors"] = errors
        return JSONResponse(
            content=response_data,
            headers=response_headers,
        )

    except TimeoutError:
        status_code = "504"
        return JSONResponse(
            status_code=504,
            content={"error": "timeout", "message": "Work items not completed within timeout"},
            headers={"X-SIE-Version": ROUTER_VERSION},
        )
    except NoConsumersError as e:
        status_code = "503"
        logger.info("No consumers for model %s — returning 503", e.model_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": "no_consumers",
                "message": str(e),
            },
            headers={"Retry-After": str(DEFAULT_RETRY_AFTER), "X-SIE-Version": ROUTER_VERSION},
        )
    except RuntimeError as e:
        if "backpressure" in str(e).lower():
            status_code = "503"
            return JSONResponse(
                status_code=503,
                content={"error": "service_unavailable", "message": str(e)},
                headers={"Retry-After": "5", "X-SIE-Version": ROUTER_VERSION},
            )
        raise
    finally:
        elapsed = time.monotonic() - start_time
        REQUEST_COUNT.labels(endpoint=endpoint, status=status_code, machine_profile=machine_profile or "queue").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, machine_profile=machine_profile or "queue").observe(elapsed)
        _emit_audit_log(
            request,
            endpoint=endpoint,
            model=model_name,
            pool=pool_name,
            gpu=machine_profile,
            status=status_code,
            latency_ms=round(elapsed * 1000, 1),
        )


@router.api_route(
    "/v1/encode/{model:path}",
    methods=["POST"],
    responses={
        200: {"description": "Embeddings generated successfully"},
        202: {"description": "GPU capacity not available, provisioning in progress"},
        502: {"description": "Worker connection error"},
        503: {"description": "No healthy workers available"},
        504: {"description": "Worker request timed out"},
    },
)
async def proxy_encode(
    model: str,
    request: Request,
    x_machine_profile: str | None = Header(None, alias="X-SIE-MACHINE-PROFILE"),
    x_sie_pool: str | None = Header(None, alias="X-SIE-Pool"),
) -> Response:
    """Proxy encode request to a worker.

    Routes to a worker based on GPU type, pool, and model affinity.
    Pool can be specified via X-SIE-Pool header or in GPU param (e.g., "pool_name/l4").
    Returns 202 if GPU type is specified but no capacity available.
    """
    return await _proxy_request(
        request=request,
        model=model,
        path=f"/v1/encode/{model}",
        x_machine_profile=x_machine_profile,
        x_sie_pool=x_sie_pool,
    )


@router.api_route(
    "/v1/score/{model:path}",
    methods=["POST"],
    responses={
        200: {"description": "Scores generated successfully"},
        202: {"description": "GPU capacity not available, provisioning in progress"},
        502: {"description": "Worker connection error"},
        503: {"description": "No healthy workers available"},
        504: {"description": "Worker request timed out"},
    },
)
async def proxy_score(
    model: str,
    request: Request,
    x_machine_profile: str | None = Header(None, alias="X-SIE-MACHINE-PROFILE"),
    x_sie_pool: str | None = Header(None, alias="X-SIE-Pool"),
) -> Response:
    """Proxy score request to a worker."""
    return await _proxy_request(
        request=request,
        model=model,
        path=f"/v1/score/{model}",
        x_machine_profile=x_machine_profile,
        x_sie_pool=x_sie_pool,
    )


@router.api_route(
    "/v1/extract/{model:path}",
    methods=["POST"],
    responses={
        200: {"description": "Extraction completed successfully"},
        202: {"description": "GPU capacity not available, provisioning in progress"},
        502: {"description": "Worker connection error"},
        503: {"description": "No healthy workers available"},
        504: {"description": "Worker request timed out"},
    },
)
async def proxy_extract(
    model: str,
    request: Request,
    x_machine_profile: str | None = Header(None, alias="X-SIE-MACHINE-PROFILE"),
    x_sie_pool: str | None = Header(None, alias="X-SIE-Pool"),
) -> Response:
    """Proxy extract request to a worker."""
    return await _proxy_request(
        request=request,
        model=model,
        path=f"/v1/extract/{model}",
        x_machine_profile=x_machine_profile,
        x_sie_pool=x_sie_pool,
    )


@router.get("/v1/models")
async def proxy_models(request: Request) -> dict[str, Any]:
    """List all available models with cluster status.

    Returns the complete model catalog from ModelRegistry (available even when
    workers are scaled to zero) combined with live status from connected workers.

    Format matches the server's /v1/models response with added cluster info.
    """
    _require_auth(request)
    worker_registry: WorkerRegistry = request.app.state.registry
    model_registry: ModelRegistry | None = getattr(request.app.state, "model_registry", None)

    # Get models currently loaded on workers
    model_workers = worker_registry.get_models()

    # If ModelRegistry is available, return complete catalog
    if model_registry is not None:
        all_model_names = model_registry.list_models()
        models = []
        for model_name in all_model_names:
            model_info = model_registry.get_model_info(model_name)
            worker_urls = model_workers.get(model_name, [])
            models.append(
                {
                    "name": model_name,
                    "bundles": model_info.bundles if model_info else [],
                    "worker_count": len(worker_urls),
                    "workers": worker_urls,
                    "loaded": len(worker_urls) > 0,
                }
            )
        return {"models": models}

    # Fallback: return only models from connected workers
    models = []
    for model_name, worker_urls in model_workers.items():
        models.append(
            {
                "name": model_name,
                "worker_count": len(worker_urls),
                "workers": worker_urls,
                "loaded": True,
            }
        )

    return {"models": models}


# =============================================================================
# Pool API endpoints
# =============================================================================


@router.post("/v1/pools", status_code=201)
async def create_pool(
    request: Request,
    body: dict[str, Any],
) -> PoolResponse:
    """Create a resource pool (idempotent).

    Request body:
        name: Pool name (required)
        gpus: GPU requirements, e.g., {"l4": 2, "a100-40gb": 1}

    Returns:
        Pool info including name, spec, and status.
    """
    _require_auth(request)
    pool_manager: PoolManager | None = getattr(request.app.state, "pool_manager", None)
    if not pool_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Pool management not enabled"},
        )

    name = body.get("name")
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Pool name is required"},
        )

    gpus = body.get("gpus", {})
    bundle = body.get("bundle")  # Optional bundle filter
    minimum_worker_count = body.get("minimum_worker_count", 0)
    if not isinstance(minimum_worker_count, int) or isinstance(minimum_worker_count, bool):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "minimum_worker_count must be an integer"},
        )
    if minimum_worker_count < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "minimum_worker_count must be >= 0"},
        )

    pool = await pool_manager.create_pool(
        name=name, gpus=gpus, bundle=bundle, minimum_worker_count=minimum_worker_count
    )

    # Try to assign workers if pool is pending
    if pool.status.state.value == "pending":
        registry: WorkerRegistry = request.app.state.registry
        assigned_urls = pool_manager.get_assigned_worker_urls()
        available = [
            (w.name, w.url, w.machine_profile, w.bundle) for w in registry.healthy_workers if w.url not in assigned_urls
        ]
        if pool_manager.assign_workers(pool, available):
            # Update pool in K8s if using K8s mode
            if pool_manager._use_kubernetes:
                await pool_manager._update_pool_in_k8s(pool)

    _emit_audit_log(request, endpoint="pools", pool=name, status=201, event="pool_create")
    return pool_to_response(pool)


@router.get("/v1/pools/{name}")
async def get_pool(request: Request, name: str) -> PoolResponse:
    """Get pool status by name.

    If the pool is pending, tries to assign available workers to it.
    This enables scale-from-zero: when KEDA scales up workers in response
    to pending_demand metrics, the next GET will assign them to the pool.
    """
    _require_auth(request)
    pool_manager: PoolManager | None = getattr(request.app.state, "pool_manager", None)
    if not pool_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Pool management not enabled"},
        )

    pool = pool_manager.get_pool(name)
    if not pool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": f"Pool '{name}' not found"},
        )

    # Try to assign workers if pool is pending (enables scale-from-zero)
    if pool.status.state.value == "pending":
        registry: WorkerRegistry = request.app.state.registry
        assigned_urls = pool_manager.get_assigned_worker_urls()
        available = [
            (w.name, w.url, w.machine_profile, w.bundle) for w in registry.healthy_workers if w.url not in assigned_urls
        ]
        if pool_manager.assign_workers(pool, available):
            # Update pool in K8s if using K8s mode
            if pool_manager._use_kubernetes:
                await pool_manager._update_pool_in_k8s(pool)
            logger.info("Assigned workers to pool '%s' on GET (scale-from-zero)", name)

    return pool_to_response(pool)


@router.get("/v1/pools")
async def list_pools(request: Request) -> dict[str, list[PoolListItem]]:
    """List all pools."""
    _require_auth(request)
    pool_manager: PoolManager | None = getattr(request.app.state, "pool_manager", None)
    if not pool_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Pool management not enabled"},
        )

    pools: list[PoolListItem] = [pool_to_list_item(pool) for pool in pool_manager.pools.values()]

    return {"pools": pools}


@router.delete("/v1/pools/{name}")
async def delete_pool(request: Request, name: str) -> dict[str, Any]:
    """Delete a pool."""
    _require_auth(request)
    pool_manager: PoolManager | None = getattr(request.app.state, "pool_manager", None)
    if not pool_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Pool management not enabled"},
        )

    try:
        deleted = await pool_manager.delete_pool(name)
    except DefaultPoolProtectedError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": f"Cannot delete the default pool '{DEFAULT_POOL_NAME}'"},
        ) from None

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": f"Pool '{name}' not found"},
        )

    _emit_audit_log(request, endpoint="pools", pool=name, status=200, event="pool_delete")
    return {"message": f"Pool '{name}' deleted"}


@router.post("/v1/pools/{name}/renew")
async def renew_pool_lease(
    request: Request,
    name: str,
) -> dict[str, Any]:
    """Renew a pool's lease."""
    _require_auth(request)
    pool_manager: PoolManager | None = getattr(request.app.state, "pool_manager", None)
    if not pool_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Pool management not enabled"},
        )

    renewed = await pool_manager.renew_lease(name)
    if not renewed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": f"Pool '{name}' not found"},
        )

    return {"message": f"Lease renewed for pool '{name}'"}
