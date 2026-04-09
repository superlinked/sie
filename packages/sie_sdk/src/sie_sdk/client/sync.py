"""Synchronous SIE Engine Client.

Provides a Python client for the Search Inference Engine server.

Per DESIGN.md Section 8.1:
- Synchronous encode() method (async variants in M5)
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
    >>> client = SIEClient("http://router:8080")
    >>> result = client.encode(
    ...     "bge-m3",
    ...     {"text": "Hello"},
    ...     gpu="l4",
    ...     wait_for_capacity=True,  # Auto-retry on 202
    ...     provision_timeout_s=900,  # Wait up to 15 min
    ... )

Resource Pools (per DESIGN.md Section 10.3):
    >>> # Create pool for isolated capacity
    >>> client = SIEClient("http://router:8080")
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
from collections.abc import Iterator
from typing import Any, Literal, Self, overload

import httpx
import msgpack
import msgpack_numpy as m

from sie_sdk.images import convert_item_images
from sie_sdk.types import (
    CapacityInfo,
    EncodeResult,
    ExtractResult,
    Item,
    ModelInfo,
    OutputType,
    PoolInfo,
    PoolSpec,
    ScoreResult,
    StatusMessage,
    WorkerInfo,
)

from ._shared import (
    DEFAULT_LEASE_RENEWAL_INTERVAL_S,
    DEFAULT_PROVISION_TIMEOUT_S,
    DEFAULT_RETRY_DELAY_S,
    HTTP_ACCEPTED,
    HTTP_CLIENT_ERROR,
    JSON_CONTENT_TYPE,
    LORA_LOADING_DEFAULT_DELAY_S,
    LORA_LOADING_ERROR_CODE,
    LORA_LOADING_MAX_RETRIES,
    MODEL_LOADING_DEFAULT_DELAY_S,
    MODEL_LOADING_ERROR_CODE,
    MSGPACK_CONTENT_TYPE,
    SDK_VERSION_HEADER,
    SERVER_VERSION_HEADER,
    check_version_skew,
    get_retry_after,
    get_sdk_version,
    handle_error,
    parse_encode_results,
    parse_extract_results,
    parse_gpu_param,
    parse_score_result,
)
from .errors import (
    LoraLoadingError,
    ModelLoadingError,
    PoolError,
    ProvisioningError,
    RequestError,
    SIEConnectionError,
)

logger = logging.getLogger(__name__)

_LEASE_RENEWAL_MAX_RETRIES = 5

# NOTE: msgpack_numpy.patch() is called lazily in SIEClient.__init__
# to avoid monkey-patching the global msgpack module at import time.
# This prevents overhead in processes that import sie_sdk but don't use
# the client (e.g., the router only needs sie_sdk.types/queue_types).
_NUMPY_PATCHED = False


def _close_transport(transport: httpx.Client) -> None:
    """Safety net: close httpx transport if SIEClient.close() was not called.

    This is a module-level function (not a method) so it does not prevent
    garbage collection of the SIEClient instance.
    """
    with contextlib.suppress(Exception):
        transport.close()


class SIEClient:
    """Client for the Search Inference Engine.

    Per DESIGN.md Section 8.1 and 9.6.

    Args:
        base_url: Base URL of the SIE server (e.g., "http://localhost:8080").
        timeout_s: Request timeout in seconds (default: 30.0).
        api_key: Optional API key for authentication (sent as Bearer token).
        gpu: Default GPU/machine profile for requests (e.g., "l4", "l4-spot").
            Can be overridden per-call.
        options: Options dict for requests. Merged with per-call options (per-call wins).
        pool: DEPRECATED. Use create_pool() instead. Resource pool spec for isolated
            capacity. Format: {"name": "pool-name", "gpus": {"l4": 2}}.

    Example:
        >>> client = SIEClient("http://localhost:8080")
        >>> result = client.encode("bge-m3", {"text": "Hello world"})
        >>> print(result["dense"].shape)
        (1024,)

        >>> # With defaults for all requests
        >>> client = SIEClient(
        ...     "http://router:8080",
        ...     gpu="l4",
        ...     options={"normalize": True},
        ... )
        >>> result = client.encode("bge-m3", {"text": "Hello"})  # uses l4
        >>> result = client.encode("bge-m3", {"text": "Hello"}, gpu="a100")  # overrides

        >>> # With resource pool for isolated capacity (new API)
        >>> client = SIEClient("http://router:8080")
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
        self._timeout = timeout_s
        self._default_gpu = gpu
        self._default_options = options
        self._api_key = api_key

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

        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_s,
            headers=headers,
        )

        # Safety net: ensure transport is closed even if close() is never called.
        # Uses weakref.finalize so the reference doesn't prevent GC.
        self._finalizer = weakref.finalize(self, _close_transport, self._client)

        # Register cleanup on interpreter exit
        if pool is not None:
            atexit.register(self._cleanup_pool)

    @property
    def base_url(self) -> str:
        """Return the base URL of the SIE server."""
        return self._base_url

    def _check_server_version(self, response: httpx.Response) -> None:
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
            if "bundle" in self._pool_spec:
                request_body["bundle"] = self._pool_spec["bundle"]
            if self._pool_spec.get("minimum_worker_count") is not None:
                request_body["minimum_worker_count"] = self._pool_spec["minimum_worker_count"]

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

        # Note: Pool deletion is not done here - pools are GC'd by router
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
        gpus: dict[str, int],
        bundle: str | None = None,
        minimum_worker_count: int | None = None,
    ) -> None:
        """Create a resource pool for isolated capacity.

        Pools reserve exclusive GPU capacity for your workload. Use them for:
        - Benchmarks that need consistent performance
        - Evaluations that shouldn't compete with production traffic
        - Isolated environments for testing

        Args:
            name: Pool name (used in gpu="pool_name/machine_profile" routing).
            gpus: Machine profile requirements, e.g., {"l4": 2, "l4-spot": 1}.
                Keys are machine profile names from cluster config.
            bundle: Optional bundle filter. When set, only workers running this
                bundle will be assigned to the pool.
            minimum_worker_count: Desired minimum number of warm workers in the pool.
                Stored in pool spec and forwarded to the router; enforcement depends
                on cluster autoscaler configuration. Defaults to 0 (scale to zero).

        Raises:
            PoolError: If pool creation fails (e.g., invalid machine profile).
            SIEConnectionError: If unable to connect to the server.

        Example:
            >>> client.create_pool("eval", {"l4": 2}, bundle="default", minimum_worker_count=1)
            >>> result = client.encode("bge-m3", {"text": "Hello"}, gpu="eval/l4")
            >>> client.delete_pool("eval")
        """
        with self._pools_lock:
            if name in self._pools:
                logger.debug("Pool '%s' already tracked, skipping creation", name)
                return

        if minimum_worker_count is not None and minimum_worker_count < 0:
            msg = "minimum_worker_count must be >= 0"
            raise ValueError(msg)

        logger.info("Creating pool '%s' with gpus=%s, bundle=%s", name, gpus, bundle)

        # Build pool creation request
        request_body: dict[str, Any] = {"name": name, "gpus": gpus}
        if bundle:
            request_body["bundle"] = bundle
        if minimum_worker_count is not None:
            request_body["minimum_worker_count"] = minimum_worker_count

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
        logger.info("Pool '%s' created with state '%s'", name, state)

        # Start lease renewal thread for this pool
        self._start_pool_lease_renewal(name)

    def _start_pool_lease_renewal(self, pool_name: str) -> None:
        """Start lease renewal thread for a pool."""
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._pool_lease_renewal_loop,
            args=(pool_name, stop_event),
            name=f"pool-lease-{pool_name}",
            daemon=True,
        )
        thread.start()

        with self._pools_lock:
            self._pools[pool_name] = (thread, stop_event)

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
    ) -> EncodeResult | list[EncodeResult]:
        """Encode items into vector representations.

        Per DESIGN.md Section 8.1.

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
                with matching GPU. Required when using the router with multiple GPU pools.
            wait_for_capacity: If True, automatically retry on 202 (provisioning) until
                capacity is available or timeout. If False, raise ProvisioningError on 202.
            provision_timeout_s: Maximum time to wait for capacity when wait_for_capacity=True.
                Default: 900 seconds (15 minutes).

        Returns:
            EncodeResult if single item was passed, list[EncodeResult] if list was passed.
            Each result contains the requested output types as numpy arrays.

        Raises:
            RequestError: If the request is invalid (4xx response).
            ServerError: If the server encounters an error (5xx response).
            SIEConnectionError: If unable to connect to the server.
            ProvisioningError: If wait_for_capacity=False and server returns 202,
                or if provisioning times out.

        Example:
            >>> # Single item
            >>> result = client.encode("bge-m3", {"text": "Hello"})
            >>> result["dense"]  # np.ndarray

            >>> # Batch
            >>> results = client.encode("bge-m3", [{"text": "Hello"}, {"text": "World"}])
            >>> len(results)  # 2

            >>> # With GPU selection (for router)
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
        # Track if single item was passed
        single_item = not isinstance(items, list)
        items_list = [items] if single_item else items

        # Convert images to JPEG bytes for transport (per design.md Section 4.3)
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

        # Retry loop for 202 (provisioning) responses
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
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except httpx.TimeoutException as e:
                # Retry on timeout if wait_for_capacity is enabled and we haven't exceeded provision timeout.
                # This handles the case where model loading takes longer than the HTTP socket timeout.
                if wait_for_capacity:
                    elapsed = time.monotonic() - start_time
                    if elapsed < timeout:
                        delay = MODEL_LOADING_DEFAULT_DELAY_S
                        remaining = timeout - elapsed
                        actual_delay = min(delay, remaining)
                        logger.info(
                            "Request timed out, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs): %s",
                            actual_delay,
                            elapsed,
                            timeout,
                            e,
                        )
                        time.sleep(actual_delay)
                        continue
                msg = f"Request timed out: {e}"
                raise SIEConnectionError(msg) from e

            # Handle 202 (provisioning) - capacity not available
            if response.status_code == HTTP_ACCEPTED:
                retry_after = get_retry_after(response)

                if not wait_for_capacity:
                    msg = f"No capacity available for GPU '{resolved_gpu}'. Server is provisioning."
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                # Check if we've exceeded the timeout
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    msg = f"Provisioning timeout after {elapsed:.1f}s waiting for GPU '{resolved_gpu}'"
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                # Wait and retry
                delay = retry_after or DEFAULT_RETRY_DELAY_S
                remaining = timeout - elapsed
                actual_delay = min(delay, remaining)
                logger.debug(
                    "Waiting %.1fs for capacity (elapsed: %.1fs, timeout: %.1fs)",
                    actual_delay,
                    elapsed,
                    timeout,
                )
                time.sleep(actual_delay)
                continue

            # Handle 503 with LORA_LOADING or MODEL_LOADING - auto-retry
            if response.status_code == 503:
                from ._shared import get_error_code

                error_code = get_error_code(response)
                if error_code == LORA_LOADING_ERROR_CODE:
                    lora_retries += 1

                    if lora_retries > LORA_LOADING_MAX_RETRIES:
                        # Extract lora from options for error message
                        lora_name = resolved_options.get("lora") if resolved_options else None
                        msg = f"LoRA loading timeout after {lora_retries} retries"
                        raise LoraLoadingError(msg, lora=str(lora_name) if lora_name else None, model=model)

                    # Wait and retry
                    retry_after = get_retry_after(response)
                    delay = retry_after or LORA_LOADING_DEFAULT_DELAY_S
                    logger.debug(
                        "LoRA loading, retrying in %.1fs (attempt %d/%d)",
                        delay,
                        lora_retries,
                        LORA_LOADING_MAX_RETRIES,
                    )
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
                    delay = retry_after or MODEL_LOADING_DEFAULT_DELAY_S
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Model loading in progress, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    time.sleep(actual_delay)
                    continue

                # Generic 503 (no healthy workers) - retry if wait_for_capacity
                # This handles scale-from-zero when pools are PENDING and have no workers yet
                if wait_for_capacity:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        # Timeout exceeded, let handle_error raise the exception
                        pass
                    else:
                        retry_after = get_retry_after(response)
                        delay = retry_after or DEFAULT_RETRY_DELAY_S
                        remaining = timeout - elapsed
                        actual_delay = min(delay, remaining)
                        logger.debug(
                            "No healthy workers, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                            actual_delay,
                            elapsed,
                            timeout,
                        )
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
        if timing:
            for result in results:
                result["timing"] = timing

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
        """Detect whether base_url is a router (cluster) or worker endpoint."""
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
            if isinstance(payload, dict) and payload.get("type") == "router":
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

    def watch(
        self,
        *,
        mode: Literal["auto", "cluster", "worker"] = "auto",
    ) -> Iterator[StatusMessage]:
        """Stream real-time status updates from the server or router.

        Args:
            mode: "cluster" connects to /ws/cluster-status, "worker" to /ws/status.
                "auto" detects router vs worker via /health.

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

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_error: Exception | None = None
        for path in paths:
            ws_url = self._ws_url(path)
            try:
                with connect(ws_url, additional_headers=headers) as ws:
                    for message in ws:
                        if isinstance(message, bytes):
                            payload = message.decode("utf-8")
                        else:
                            payload = message
                        data = json.loads(payload)
                        yield data
                return
            except InvalidStatus as e:
                last_error = e
                status = (
                    getattr(e, "status_code", None)
                    or getattr(e, "status", None)
                    or getattr(getattr(e, "response", None), "status_code", None)
                )
                raise RequestError(f"WebSocket connection failed: {status}") from e
            except WebSocketException as e:
                last_error = e
                raise SIEConnectionError(f"WebSocket error: {e}") from e
            except (OSError, json.JSONDecodeError) as e:
                last_error = e
                raise SIEConnectionError(f"WebSocket error: {e}") from e

        if last_error:
            raise SIEConnectionError(f"WebSocket connection failed: {last_error}") from last_error

    def get_capacity(self, *, gpu: str | None = None) -> CapacityInfo:
        """Get current cluster capacity information.

        Queries the router's /health endpoint for cluster state. Useful for
        checking if specific GPU types are available before sending requests.

        Args:
            gpu: Optional filter to check specific GPU type availability.

        Returns:
            CapacityInfo with worker count, GPU types, and worker details.
            If gpu is specified, only workers with matching GPU are included.

        Raises:
            SIEConnectionError: If unable to connect to the server.
            ServerError: If the server encounters an error.
            RequestError: If the endpoint is not available (e.g., worker, not router).

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

        # Check if this is a router (has 'type': 'router') or worker
        if data.get("type") != "router":
            msg = "get_capacity() requires a router endpoint. This appears to be a worker."
            raise RequestError(msg, code="not_router", status_code=400)

        # Build CapacityInfo
        cluster = data.get("cluster", {})
        workers_data = data.get("workers", [])

        # Filter by GPU if specified
        if gpu:
            gpu_lower = gpu.lower()
            workers_data = [w for w in workers_data if w.get("gpu", "").lower() == gpu_lower]

        workers: list[WorkerInfo] = [
            WorkerInfo(
                url=w.get("url", ""),
                gpu=w.get("gpu", ""),
                healthy=w.get("healthy", False),
                queue_depth=w.get("queue_depth", 0),
                loaded_models=w.get("loaded_models", []),
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

        Polls the router until workers with the specified GPU type are online.
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
            wait_for_capacity: If True, automatically retry on 202 (provisioning) until
                capacity is available or timeout.
            provision_timeout_s: Maximum time to wait for capacity when wait_for_capacity=True.
                Default: 900 seconds (15 minutes).

        Returns:
            ScoreResult containing the model name, query_id, and sorted scores.
            Scores are sorted by relevance (descending), with rank 0 being most relevant.

        Raises:
            RequestError: If the request is invalid (4xx response).
            ServerError: If the server encounters an error (5xx response).
            SIEConnectionError: If unable to connect to the server.
            ProvisioningError: If wait_for_capacity=False and server returns 202,
                or if provisioning times out.

        Example:
            >>> result = client.score(
            ...     "bge-reranker-v2",
            ...     query={"text": "What is machine learning?"},
            ...     items=[{"text": "ML is AI..."}, {"text": "Python is..."}],
            ... )
        """
        # Resolve defaults and pool
        pool_name, resolved_gpu = self._resolve_pool_and_gpu(gpu)
        resolved_options = self._resolve_options(options)

        # Build request body
        request_body: dict[str, Any] = {
            "query": query,
            "items": items,
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

        # Retry loop for 202 (provisioning) responses
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
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except httpx.TimeoutException as e:
                # Retry on timeout if wait_for_capacity is enabled and we haven't exceeded provision timeout.
                # This handles the case where model loading takes longer than the HTTP socket timeout.
                if wait_for_capacity:
                    elapsed = time.monotonic() - start_time
                    if elapsed < timeout:
                        delay = MODEL_LOADING_DEFAULT_DELAY_S
                        remaining = timeout - elapsed
                        actual_delay = min(delay, remaining)
                        logger.info(
                            "Request timed out, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs): %s",
                            actual_delay,
                            elapsed,
                            timeout,
                            e,
                        )
                        time.sleep(actual_delay)
                        continue
                msg = f"Request timed out: {e}"
                raise SIEConnectionError(msg) from e

            # Handle 202 (provisioning) - capacity not available
            if response.status_code == HTTP_ACCEPTED:
                retry_after = get_retry_after(response)

                if not wait_for_capacity:
                    msg = f"No capacity available for GPU '{resolved_gpu}'. Server is provisioning."
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                # Check if we've exceeded the timeout
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    msg = f"Provisioning timeout after {elapsed:.1f}s waiting for GPU '{resolved_gpu}'"
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                # Wait and retry
                delay = retry_after or DEFAULT_RETRY_DELAY_S
                remaining = timeout - elapsed
                actual_delay = min(delay, remaining)
                logger.debug(
                    "Waiting %.1fs for capacity (elapsed: %.1fs, timeout: %.1fs)",
                    actual_delay,
                    elapsed,
                    timeout,
                )
                time.sleep(actual_delay)
                continue

            # Handle 503 with MODEL_LOADING - auto-retry
            if response.status_code == 503:
                from ._shared import get_error_code

                error_code = get_error_code(response)
                if error_code == MODEL_LOADING_ERROR_CODE:
                    # Check if we've exceeded the provision timeout
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)

                    # Wait and retry, respecting remaining time
                    retry_after = get_retry_after(response)
                    delay = retry_after or MODEL_LOADING_DEFAULT_DELAY_S
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Model loading in progress, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    time.sleep(actual_delay)
                    continue

                # Generic 503 (no healthy workers) - retry if wait_for_capacity
                # This handles scale-from-zero when pools are PENDING and have no workers yet
                if wait_for_capacity:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        # Timeout exceeded, let handle_error raise the exception
                        pass
                    else:
                        retry_after = get_retry_after(response)
                        delay = retry_after or DEFAULT_RETRY_DELAY_S
                        remaining = timeout - elapsed
                        actual_delay = min(delay, remaining)
                        logger.debug(
                            "No healthy workers, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                            actual_delay,
                            elapsed,
                            timeout,
                        )
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
        return parse_score_result(response_data)

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
            wait_for_capacity: If True, automatically retry on 202 (provisioning) until
                capacity is available or timeout.
            provision_timeout_s: Maximum time to wait for capacity when wait_for_capacity=True.
                Default: 900 seconds (15 minutes).

        Returns:
            ExtractResult if single item was passed, list[ExtractResult] if list was passed.
            Each result contains entities (list of EntityResult) and optional data dict.

        Raises:
            RequestError: If the request is invalid (4xx response).
            ServerError: If the server encounters an error (5xx response).
            SIEConnectionError: If unable to connect to the server.
            ProvisioningError: If wait_for_capacity=False and server returns 202,
                or if provisioning times out.

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
        # Track if single item was passed
        single_item = not isinstance(items, list)
        items_list = [items] if single_item else items

        # Convert images to wire format (same as encode)
        items_for_wire = [convert_item_images({**item}) if "images" in item else item for item in items_list]  # ty: ignore[invalid-argument-type]

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

        # Retry loop for 202 (provisioning) responses
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
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e
            except httpx.TimeoutException as e:
                # Retry on timeout if wait_for_capacity is enabled and we haven't exceeded provision timeout.
                # This handles the case where model loading takes longer than the HTTP socket timeout.
                if wait_for_capacity:
                    elapsed = time.monotonic() - start_time
                    if elapsed < timeout:
                        delay = MODEL_LOADING_DEFAULT_DELAY_S
                        remaining = timeout - elapsed
                        actual_delay = min(delay, remaining)
                        logger.info(
                            "Request timed out, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs): %s",
                            actual_delay,
                            elapsed,
                            timeout,
                            e,
                        )
                        time.sleep(actual_delay)
                        continue
                msg = f"Request timed out: {e}"
                raise SIEConnectionError(msg) from e

            # Handle 202 (provisioning) - capacity not available
            if response.status_code == HTTP_ACCEPTED:
                retry_after = get_retry_after(response)

                if not wait_for_capacity:
                    msg = f"No capacity available for GPU '{resolved_gpu}'. Server is provisioning."
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                # Check if we've exceeded the timeout
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    msg = f"Provisioning timeout after {elapsed:.1f}s waiting for GPU '{resolved_gpu}'"
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                # Wait and retry
                delay = retry_after or DEFAULT_RETRY_DELAY_S
                remaining = timeout - elapsed
                actual_delay = min(delay, remaining)
                logger.debug(
                    "Waiting %.1fs for capacity (elapsed: %.1fs, timeout: %.1fs)",
                    actual_delay,
                    elapsed,
                    timeout,
                )
                time.sleep(actual_delay)
                continue

            # Handle 503 with MODEL_LOADING - auto-retry
            if response.status_code == 503:
                from ._shared import get_error_code

                error_code = get_error_code(response)
                if error_code == MODEL_LOADING_ERROR_CODE:
                    # Check if we've exceeded the provision timeout
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)

                    # Wait and retry, respecting remaining time
                    retry_after = get_retry_after(response)
                    delay = retry_after or MODEL_LOADING_DEFAULT_DELAY_S
                    remaining = timeout - elapsed
                    actual_delay = min(delay, remaining)
                    logger.info(
                        "Model loading in progress, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                        actual_delay,
                        elapsed,
                        timeout,
                    )
                    time.sleep(actual_delay)
                    continue

                # Generic 503 (no healthy workers) - retry if wait_for_capacity
                # This handles scale-from-zero when pools are PENDING and have no workers yet
                if wait_for_capacity:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        # Timeout exceeded, let handle_error raise the exception
                        pass
                    else:
                        retry_after = get_retry_after(response)
                        delay = retry_after or DEFAULT_RETRY_DELAY_S
                        remaining = timeout - elapsed
                        actual_delay = min(delay, remaining)
                        logger.debug(
                            "No healthy workers, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs)",
                            actual_delay,
                            elapsed,
                            timeout,
                        )
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

        # Return single result if single item was passed
        return results[0] if single_item else results
