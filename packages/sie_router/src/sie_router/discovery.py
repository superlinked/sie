"""Worker discovery and WebSocket connection management.

Supports multiple discovery methods:
- Static: Worker URLs from config file
- Kubernetes: Watch service endpoints via K8s API (event-driven, no polling)
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import orjson
import websockets
from websockets.exceptions import ConnectionClosedError

from sie_router.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class DiscoveryEventType(Enum):
    """Type of discovery event."""

    ADDED = "added"
    REMOVED = "removed"


@dataclass
class DiscoveryEvent:
    """Event from worker discovery watch."""

    event_type: DiscoveryEventType
    url: str


class WorkerDiscovery(ABC):
    """Abstract base for worker discovery."""

    @abstractmethod
    async def get_worker_urls(self) -> list[str]:
        """Get list of worker URLs to connect to."""
        ...

    @property
    def supports_watch(self) -> bool:
        """Whether this discovery supports event-driven watch."""
        return False

    async def watch(self) -> AsyncIterator[DiscoveryEvent]:
        """Watch for worker discovery events.

        Override in subclasses that support watch.
        Yields DiscoveryEvent for each added/removed worker.
        """
        raise NotImplementedError("This discovery does not support watch")
        # Make this an async generator
        yield  # type: ignore[misc]


class StaticDiscovery(WorkerDiscovery):
    """Static worker discovery from config."""

    def __init__(self, worker_urls: list[str]) -> None:
        """Initialize with static list of worker URLs.

        Args:
            worker_urls: List of worker URLs (e.g., ["http://worker-0:8080"]).
        """
        self._worker_urls = worker_urls

    async def get_worker_urls(self) -> list[str]:
        """Return the static list of worker URLs."""
        return self._worker_urls


class KubernetesDiscovery(WorkerDiscovery):
    """Kubernetes-based worker discovery.

    Uses Kubernetes watch API for event-driven discovery of worker endpoints.
    When a worker pod starts/stops, the router is notified immediately
    (no polling delay).
    """

    def __init__(
        self,
        *,
        namespace: str = "default",
        service_name: str = "sie-worker",
        port: int = 8080,
    ) -> None:
        """Initialize Kubernetes discovery.

        Args:
            namespace: Kubernetes namespace.
            service_name: Name of the worker service.
            port: Port workers are listening on.
        """
        self._namespace = namespace
        self._service_name = service_name
        self._port = port
        # Type annotation uses Any to avoid importing kubernetes at load time
        self._client: Any = None
        self._known_urls: set[str] = set()

    def _init_client(self) -> None:
        """Initialize Kubernetes client lazily."""
        if self._client is not None:
            return

        import kubernetes

        try:
            kubernetes.config.load_incluster_config()
            logger.info("Using in-cluster K8s config for discovery")
        except kubernetes.config.ConfigException:
            kubernetes.config.load_kube_config()
            logger.info("Using kubeconfig for K8s discovery")

        self._client = kubernetes.client.CoreV1Api()

    def _endpoints_to_urls(self, endpoints: Any) -> set[str]:
        """Extract worker URLs from Kubernetes endpoints object."""
        urls: set[str] = set()
        if endpoints.subsets:
            for subset in endpoints.subsets:
                if subset.addresses:
                    for addr in subset.addresses:
                        url = f"http://{addr.ip}:{self._port}"
                        urls.add(url)
        return urls

    async def get_worker_urls(self) -> list[str]:
        """Get worker URLs from Kubernetes endpoints."""
        try:
            self._init_client()

            # Get endpoints for the service
            endpoints = self._client.read_namespaced_endpoints(
                name=self._service_name,
                namespace=self._namespace,
            )

            urls = self._endpoints_to_urls(endpoints)
            self._known_urls = urls
            return list(urls)

        except ImportError:
            logger.error("kubernetes package not installed")
            return []
        except (AttributeError, ValueError, OSError) as e:
            logger.error("Failed to discover workers via K8s: %s", e)
            return []

    @property
    def supports_watch(self) -> bool:
        """Kubernetes discovery supports event-driven watch."""
        return True

    async def watch(self) -> AsyncIterator[DiscoveryEvent]:
        """Watch Kubernetes endpoints for worker changes.

        Yields DiscoveryEvent immediately when workers are added/removed.
        Uses short watch timeouts and reconnects to avoid blocking.
        """
        from kubernetes import watch
        from kubernetes.client.rest import ApiException

        self._init_client()
        w = watch.Watch()
        resource_version: str | None = None
        backoff_s = 1.0

        while True:
            try:
                if resource_version is None:
                    endpoints_list = await asyncio.to_thread(
                        self._client.list_namespaced_endpoints,
                        namespace=self._namespace,
                        field_selector=f"metadata.name={self._service_name}",
                    )
                    endpoints = endpoints_list.items[0] if endpoints_list.items else None
                    current_urls = self._endpoints_to_urls(endpoints) if endpoints else set()
                    self._known_urls = current_urls
                    resource_version = endpoints_list.metadata.resource_version

                logger.info(
                    "Starting K8s endpoints watch: %s/%s",
                    self._namespace,
                    self._service_name,
                )

                # Run the synchronous watch in a thread pool
                def watch_endpoints() -> list[tuple[str, Any]]:
                    """Watch endpoints in thread, return list of (event_type, endpoints)."""
                    events = []
                    stream = w.stream(
                        self._client.list_namespaced_endpoints,
                        namespace=self._namespace,
                        field_selector=f"metadata.name={self._service_name}",
                        resource_version=resource_version,
                        timeout_seconds=30,  # Short timeout to yield control
                    )
                    for event in stream:
                        events.append((event["type"], event["object"]))
                    return events

                events = await asyncio.to_thread(watch_endpoints)

                for event_type, endpoints in events:
                    current_urls = self._endpoints_to_urls(endpoints)
                    if endpoints.metadata and endpoints.metadata.resource_version:
                        resource_version = endpoints.metadata.resource_version

                    if event_type in ("ADDED", "MODIFIED"):
                        # Find new workers
                        added = current_urls - self._known_urls
                        for url in added:
                            logger.info("K8s watch: worker added %s", url)
                            yield DiscoveryEvent(DiscoveryEventType.ADDED, url)

                        # Find removed workers
                        removed = self._known_urls - current_urls
                        for url in removed:
                            logger.info("K8s watch: worker removed %s", url)
                            yield DiscoveryEvent(DiscoveryEventType.REMOVED, url)

                        self._known_urls = current_urls

                    elif event_type == "DELETED":
                        # Service endpoints deleted - all workers gone
                        for url in self._known_urls:
                            logger.info("K8s watch: worker removed (endpoints deleted) %s", url)
                            yield DiscoveryEvent(DiscoveryEventType.REMOVED, url)
                        self._known_urls = set()

                # Brief yield before restarting watch
                await asyncio.sleep(0.1)
                backoff_s = 1.0

            except asyncio.CancelledError:
                logger.info("K8s endpoints watch cancelled")
                raise
            except ApiException as e:
                if e.status == 410:
                    logger.warning("K8s endpoints watch expired (410 Gone); relisting")
                    resource_version = None
                    continue
                logger.error("K8s endpoints watch API error: %s", e)
                jitter = random.uniform(0, backoff_s * 0.2)  # noqa: S311 — retry jitter, not security
                await asyncio.sleep(backoff_s + jitter)
                backoff_s = min(backoff_s * 1.5, 60.0)
            except (ValueError, KeyError, OSError) as e:
                logger.error("K8s endpoints watch error: %s", e)
                jitter = random.uniform(0, backoff_s * 0.2)  # noqa: S311 — retry jitter, not security
                await asyncio.sleep(backoff_s + jitter)
                backoff_s = min(backoff_s * 1.5, 60.0)
            except Exception as e:  # noqa: BLE001 — K8s API errors are varied
                # Catch-all for unexpected errors (e.g., urllib3 MaxRetryError
                # when K8s API is transiently unreachable)
                logger.error("K8s endpoints watch unexpected error: %s", e)
                jitter = random.uniform(0, backoff_s * 0.2)  # noqa: S311 — retry jitter, not security
                await asyncio.sleep(backoff_s + jitter)
                backoff_s = min(backoff_s * 1.5, 60.0)


class WorkerConnectionManager:
    """Manages WebSocket connections to workers.

    Connects to each worker's /ws/status endpoint to receive state updates.
    Handles reconnection on disconnect.

    Uses event-driven discovery (Kubernetes watch) when available,
    falling back to polling for static discovery.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        discovery: WorkerDiscovery,
        *,
        reconnect_delay_s: float = 5.0,
        discovery_interval_s: float = 30.0,
    ) -> None:
        """Initialize the connection manager.

        Args:
            registry: Worker registry to update.
            discovery: Worker discovery mechanism.
            reconnect_delay_s: Delay before reconnecting to a worker.
            discovery_interval_s: Interval for re-discovering workers (fallback for non-watch).
        """
        self._registry = registry
        self._discovery = discovery
        self._reconnect_delay_s = reconnect_delay_s
        self._discovery_interval_s = discovery_interval_s

        self._running = False
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._discovery_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._removed_urls: set[str] = set()

    async def start(self) -> None:
        """Start connecting to workers."""
        self._running = True

        # Initial discovery
        urls = await self._discovery.get_worker_urls()
        logger.info("Discovered %d workers", len(urls))

        for url in urls:
            self._start_connection(url)

        # Start background tasks
        # Use watch-based discovery if supported (Kubernetes), else poll
        if self._discovery.supports_watch:
            logger.info("Using event-driven worker discovery (watch)")
            self._discovery_task = asyncio.create_task(self._watch_loop())
        else:
            logger.info("Using polling-based worker discovery (interval=%ds)", self._discovery_interval_s)
            self._discovery_task = asyncio.create_task(self._discovery_loop())

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop all connections."""
        self._running = False

        # Cancel discovery and heartbeat tasks
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cancel all connection tasks
        for task in self._tasks.values():
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        self._tasks.clear()

    def _start_connection(self, url: str) -> None:
        """Start a connection task for a worker URL."""
        if url not in self._tasks or self._tasks[url].done():
            self._tasks[url] = asyncio.create_task(self._connect_to_worker(url))

    async def _connect_to_worker(self, url: str) -> None:
        """Connect to a worker and receive state updates.

        Args:
            url: Worker HTTP URL (e.g., "http://worker:8080").
        """
        # Convert HTTP URL to WebSocket URL
        ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/status"

        while self._running:
            try:
                logger.info("Connecting to worker %s", ws_url)
                async with websockets.connect(ws_url) as ws:
                    logger.info("Connected to worker %s", ws_url)

                    async for message in ws:
                        if not self._running:
                            break

                        try:
                            state = orjson.loads(message)
                            await self._registry.update_worker(url, state)
                        except orjson.JSONDecodeError:
                            logger.warning("Invalid JSON from worker %s: %s", url, message[:100])

            except ConnectionClosedError:
                logger.warning("Connection to %s closed", ws_url)
            except OSError as e:
                logger.warning("Error connecting to %s: %s", ws_url, e)

            # Mark worker as unhealthy on disconnect
            if self._running:
                if url in self._removed_urls:
                    logger.info("Worker %s was removed; stopping reconnection", url)
                    return
                await self._registry.mark_unhealthy(url)
                logger.info("Reconnecting to %s in %.1fs", ws_url, self._reconnect_delay_s)
                await asyncio.sleep(self._reconnect_delay_s)

    async def _discovery_loop(self) -> None:
        """Periodically re-discover workers (fallback for non-watch discovery)."""
        while self._running:
            await asyncio.sleep(self._discovery_interval_s)

            try:
                urls = await self._discovery.get_worker_urls()

                # Start connections to new workers
                for url in urls:
                    if url not in self._tasks or self._tasks[url].done():
                        logger.info("New worker discovered: %s", url)
                        self._start_connection(url)

                # Note: We don't remove workers that are no longer in discovery
                # They will be marked unhealthy when their WebSocket disconnects

            except (AttributeError, ValueError, OSError) as e:
                logger.error("Discovery loop error: %s", e)

    async def _watch_loop(self) -> None:
        """Watch for worker discovery events (event-driven, no polling).

        Used when the discovery mechanism supports watch (e.g., Kubernetes).
        Workers are discovered immediately when they come online.
        Retries with backoff on transient failures.
        """
        backoff_s = 1.0
        while self._running:
            try:
                async for event in self._discovery.watch():
                    if not self._running:
                        break

                    if event.event_type == DiscoveryEventType.ADDED:
                        self._removed_urls.discard(event.url)
                        if event.url not in self._tasks or self._tasks[event.url].done():
                            logger.info("Watch: new worker discovered: %s", event.url)
                            self._start_connection(event.url)

                    elif event.event_type == DiscoveryEventType.REMOVED:
                        self._removed_urls.add(event.url)
                        # Cancel connection task and remove worker
                        if event.url in self._tasks:
                            logger.info("Watch: worker removed: %s", event.url)
                            self._tasks[event.url].cancel()
                            try:
                                await self._tasks[event.url]
                            except asyncio.CancelledError:
                                pass
                            del self._tasks[event.url]
                        await self._registry.remove_worker(event.url)

                # Generator exited normally (e.g., watch timeout) - reset backoff
                backoff_s = 1.0

            except asyncio.CancelledError:
                logger.info("Watch loop cancelled")
                raise
            except Exception as e:  # noqa: BLE001 — K8s watch errors are varied
                jitter = random.uniform(0, backoff_s * 0.2)  # noqa: S311 — retry jitter, not security
                delay = backoff_s + jitter
                logger.error("Watch loop error: %s; retrying in %.1fs", e, delay)
                await asyncio.sleep(delay)
                backoff_s = min(backoff_s * 1.5, 60.0)

    async def _heartbeat_loop(self) -> None:
        """Periodically check for missed heartbeats."""
        while self._running:
            await asyncio.sleep(5.0)  # Check every 5 seconds
            await self._registry.check_heartbeats()
