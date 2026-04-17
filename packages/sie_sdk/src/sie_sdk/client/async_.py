"""Asynchronous SIE Engine Client.

Provides an async Python client for the Search Inference Engine server.

Per DESIGN.md Section 8.1 and 9.6 - Async variants for all methods.

Example:
    >>> async with SIEAsyncClient("http://localhost:8080") as client:
    ...     result = await client.encode("bge-m3", {"text": "Hello world"})
    ...     print(result["dense"].shape)
    (1024,)

    >>> # With defaults for all requests
    >>> async with SIEAsyncClient(
    ...     "http://router:8080",
    ...     gpu="l4",
    ...     options={"normalize": True},
    ... ) as client:
    ...     result = await client.encode("bge-m3", {"text": "Hello"})  # uses l4

    >>> # With resource pool for isolated capacity
    >>> async with SIEAsyncClient(
    ...     "http://router:8080",
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
from collections.abc import AsyncIterator
from typing import Any, Literal, Self, overload

import aiohttp
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

# NOTE: msgpack_numpy.patch() is called lazily in SIEAsyncClient.__init__
# (see sync.py for details).
_NUMPY_PATCHED = False


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

    Per DESIGN.md Section 8.1 and 9.6 - Async variants for all methods.

    Args:
        base_url: Base URL of the SIE server (e.g., "http://localhost:8080").
        timeout_s: Request timeout in seconds (default: 30.0).
        api_key: Optional API key for authentication (sent as Bearer token).
        gpu: GPU type for requests (e.g., "l4", "a100-80gb"). Can be overridden per-call.
        options: Options dict for requests. Merged with per-call options (per-call wins).
        pool: Resource pool spec for isolated capacity. Created lazily on first request.
            Format: {"name": "pool-name", "gpus": {"l4": 2, "a100-40gb": 1}}.

    Example:
        >>> async with SIEAsyncClient("http://localhost:8080") as client:
        ...     result = await client.encode("bge-m3", {"text": "Hello world"})
        ...     print(result["dense"].shape)
        (1024,)

        >>> # With defaults for all requests
        >>> async with SIEAsyncClient(
        ...     "http://router:8080",
        ...     gpu="l4",
        ...     options={"normalize": True},
        ... ) as client:
        ...     result = await client.encode("bge-m3", {"text": "Hello"})  # uses l4

        >>> # With resource pool for isolated capacity
        >>> async with SIEAsyncClient(
        ...     "http://router:8080",
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

        # Multi-pool state: track created pools and their lease renewal tasks
        # Key: pool name, Value: asyncio.Task for lease renewal
        self._pools: dict[str, asyncio.Task[None]] = {}
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
        if not self._closed:
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
    ) -> _AioResponse:
        kw: dict[str, Any] = {}
        if data is not None:
            kw["data"] = data
        if json_data is not None:
            kw["json"] = json_data
        if headers:
            kw["headers"] = headers
        if timeout_s is not None:
            kw["timeout"] = aiohttp.ClientTimeout(total=timeout_s)
        async with self._throttle(), self._ensure_session().post(url, **kw) as resp:
            body = await resp.read()
            return _AioResponse(resp.status, body, resp.headers)

    async def _get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> _AioResponse:
        kw: dict[str, Any] = {}
        if headers:
            kw["headers"] = headers
        async with self._throttle(), self._ensure_session().get(url, **kw) as resp:
            body = await resp.read()
            return _AioResponse(resp.status, body, resp.headers)

    async def _delete(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> _AioResponse:
        kw: dict[str, Any] = {}
        if headers:
            kw["headers"] = headers
        async with self._throttle(), self._ensure_session().delete(url, **kw) as resp:
            body = await resp.read()
            return _AioResponse(resp.status, body, resp.headers)

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
            if "bundle" in self._pool_spec:
                request_body["bundle"] = self._pool_spec["bundle"]
            if self._pool_spec.get("minimum_worker_count") is not None:
                request_body["minimum_worker_count"] = self._pool_spec["minimum_worker_count"]

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
            for pool_name, task in list(self._pools.items()):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._pools.clear()

        # Also cleanup legacy pool
        await self._cleanup_pool()

    async def create_pool(
        self,
        name: str,
        gpus: dict[str, int],
        bundle: str | None = None,
        minimum_worker_count: int | None = None,
    ) -> None:
        """Create a resource pool for isolated capacity.

        Args:
            name: Pool name (used in gpu="pool_name/machine_profile" routing).
            gpus: Machine profile requirements, e.g., {"l4": 2, "l4-spot": 1}.
            bundle: Optional bundle filter. When set, only workers running this
                bundle will be assigned to the pool.
            minimum_worker_count: Desired minimum number of warm workers in the pool.
                Stored in pool spec and forwarded to the router; enforcement depends
                on cluster autoscaler configuration. Defaults to 0 (scale to zero).

        Raises:
            PoolError: If pool creation fails (e.g., invalid machine profile).
            SIEConnectionError: If unable to connect to the server.
        """
        async with self._pools_lock:
            if name in self._pools:
                logger.debug("Pool '%s' already tracked, skipping creation", name)
                return
            # Reserve the name to prevent concurrent create_pool racing
            self._pools[name] = None  # type: ignore[assignment]

        if minimum_worker_count is not None and minimum_worker_count < 0:
            async with self._pools_lock:
                self._pools.pop(name, None)
            msg = "minimum_worker_count must be >= 0"
            raise ValueError(msg)

        logger.info("Creating pool '%s' with gpus=%s, bundle=%s", name, gpus, bundle)

        request_body: dict[str, Any] = {"name": name, "gpus": gpus}
        if bundle:
            request_body["bundle"] = bundle
        if minimum_worker_count is not None:
            request_body["minimum_worker_count"] = minimum_worker_count

        try:
            response = await self._post(
                "/v1/pools",
                json_data=request_body,
                headers={"Content-Type": JSON_CONTENT_TYPE, "Accept": JSON_CONTENT_TYPE},
            )
        except (aiohttp.ClientError, OSError) as e:
            async with self._pools_lock:
                self._pools.pop(name, None)
            msg = f"Failed to create pool '{name}': connection error: {e}"
            raise SIEConnectionError(msg) from e

        if response.status_code >= HTTP_CLIENT_ERROR:
            async with self._pools_lock:
                self._pools.pop(name, None)
            try:
                data = response.json()
                error_msg = data.get("detail", {}).get("message", str(data))
            except (ValueError, KeyError):
                error_msg = response.text
            msg = f"Failed to create pool '{name}': {error_msg}"
            raise PoolError(msg, pool_name=name)

        data = response.json()
        state = data.get("status", {}).get("state", "unknown")
        logger.info("Pool '%s' created with state '%s'", name, state)

        # Start lease renewal task for this pool
        await self._start_pool_lease_renewal(name)

    async def _start_pool_lease_renewal(self, pool_name: str) -> None:
        """Start lease renewal task for a pool."""
        task = asyncio.create_task(
            self._pool_lease_renewal_loop(pool_name),
            name=f"pool-lease-{pool_name}",
        )
        async with self._pools_lock:
            self._pools[pool_name] = task
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

        # Stop lease renewal task for this pool
        async with self._pools_lock:
            if pool_name in self._pools:
                task = self._pools.pop(pool_name)
                task.cancel()
                try:
                    await task
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
    ) -> EncodeResult | list[EncodeResult]:
        """Async version of encode(). See SIEClient.encode() for details."""
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
                response = await self._post(
                    f"/v1/encode/{model}",
                    data=body,
                    headers=headers,
                    timeout_s=request_timeout,
                )
            except TimeoutError as e:
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
                        await asyncio.sleep(actual_delay)
                        continue
                msg = f"Request timed out: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                msg = f"Failed to connect to {self._base_url}: {e}"
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
                await asyncio.sleep(actual_delay)
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
                    delay = retry_after or MODEL_LOADING_DEFAULT_DELAY_S
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
        if timing:
            for result in results:
                result["timing"] = timing

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
        """Detect whether base_url is a router (cluster) or worker endpoint."""
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

    async def watch(
        self,
        *,
        mode: Literal["auto", "cluster", "worker"] = "auto",
    ) -> AsyncIterator[StatusMessage]:
        """Stream real-time status updates from the server or router."""
        import websockets

        if mode == "auto":
            detected = await self._detect_endpoint_type()
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
                async with websockets.connect(ws_url, additional_headers=headers) as ws:
                    async for message in ws:
                        if isinstance(message, bytes):
                            payload = message.decode("utf-8")
                        else:
                            payload = message
                        data = json.loads(payload)
                        yield data
                return
            except websockets.exceptions.InvalidStatus as e:
                last_error = e
                raise RequestError(f"WebSocket connection failed: {e.response.status_code}") from e
            except (websockets.WebSocketException, OSError, json.JSONDecodeError) as e:
                last_error = e
                raise SIEConnectionError(f"WebSocket error: {e}") from e

        if last_error:
            raise SIEConnectionError(f"WebSocket connection failed: {last_error}") from last_error

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
    ) -> ScoreResult:
        """Score items against a query using a reranker model.

        Async version of :meth:`SIEClient.score`. See that method for full
        parameter documentation.
        """
        # Resolve defaults and pool
        pool_name, resolved_gpu = await self._resolve_pool_and_gpu(gpu)
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
                response = await self._post(
                    f"/v1/score/{model}",
                    data=body,
                    headers=headers,
                    timeout_s=request_timeout,
                )
            except TimeoutError as e:
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
                        await asyncio.sleep(actual_delay)
                        continue
                msg = f"Request timed out: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e

            # Handle 202 (provisioning) - capacity not available
            if response.status_code == HTTP_ACCEPTED:
                retry_after = get_retry_after(response)

                if not wait_for_capacity:
                    msg = f"No capacity available for GPU '{resolved_gpu}'. Server is provisioning."
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    msg = f"Provisioning timeout after {elapsed:.1f}s waiting for GPU '{resolved_gpu}'"
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                delay = retry_after or DEFAULT_RETRY_DELAY_S
                remaining = timeout - elapsed
                actual_delay = min(delay, remaining)
                logger.debug(
                    "Waiting %.1fs for capacity (elapsed: %.1fs, timeout: %.1fs)",
                    actual_delay,
                    elapsed,
                    timeout,
                )
                await asyncio.sleep(actual_delay)
                continue

            # Handle 503 with MODEL_LOADING - auto-retry
            if response.status_code == 503:
                from ._shared import get_error_code

                error_code = get_error_code(response)
                if error_code == MODEL_LOADING_ERROR_CODE:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)

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
                    await asyncio.sleep(actual_delay)
                    continue

                if wait_for_capacity:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
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
                        await asyncio.sleep(actual_delay)
                        continue

            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)

            break

        self._check_server_version(response)

        response_data = msgpack.unpackb(response.content, raw=False)

        return parse_score_result(response_data)

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
    ) -> ExtractResult | list[ExtractResult]:
        """Async version of extract(). See SIEClient.extract() for details."""
        # Track if single item was passed
        single_item = not isinstance(items, list)
        items_list = [items] if single_item else items

        # Convert images to wire format (same as encode)
        items_for_wire = [
            convert_item_images({**item}) if "images" in item else item  # ty: ignore[invalid-argument-type]
            for item in items_list
        ]

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
                response = await self._post(
                    f"/v1/extract/{model}",
                    data=body,
                    headers=headers,
                    timeout_s=request_timeout,
                )
            except TimeoutError as e:
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
                        await asyncio.sleep(actual_delay)
                        continue
                msg = f"Request timed out: {e}"
                raise SIEConnectionError(msg) from e
            except (aiohttp.ClientError, OSError) as e:
                msg = f"Failed to connect to {self._base_url}: {e}"
                raise SIEConnectionError(msg) from e

            # Handle 202 (provisioning) - capacity not available
            if response.status_code == HTTP_ACCEPTED:
                retry_after = get_retry_after(response)

                if not wait_for_capacity:
                    msg = f"No capacity available for GPU '{resolved_gpu}'. Server is provisioning."
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    msg = f"Provisioning timeout after {elapsed:.1f}s waiting for GPU '{resolved_gpu}'"
                    raise ProvisioningError(msg, gpu=resolved_gpu, retry_after=retry_after)

                delay = retry_after or DEFAULT_RETRY_DELAY_S
                remaining = timeout - elapsed
                actual_delay = min(delay, remaining)
                logger.debug(
                    "Waiting %.1fs for capacity (elapsed: %.1fs, timeout: %.1fs)",
                    actual_delay,
                    elapsed,
                    timeout,
                )
                await asyncio.sleep(actual_delay)
                continue

            # Handle 503 with MODEL_LOADING - auto-retry
            if response.status_code == 503:
                from ._shared import get_error_code

                error_code = get_error_code(response)
                if error_code == MODEL_LOADING_ERROR_CODE:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                        raise ModelLoadingError(msg, model=model)

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
                    await asyncio.sleep(actual_delay)
                    continue

                if wait_for_capacity:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
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
                        await asyncio.sleep(actual_delay)
                        continue

            if response.status_code >= HTTP_CLIENT_ERROR:
                handle_error(response)

            break

        self._check_server_version(response)

        response_data = msgpack.unpackb(response.content, raw=False)

        results = parse_extract_results(response_data["items"])

        return results[0] if single_item else results
