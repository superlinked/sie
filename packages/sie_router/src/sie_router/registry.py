"""Worker registry for tracking worker state.

The registry maintains state for all workers in the cluster,
updated via WebSocket connections to each worker's /ws/stats endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from sie_sdk.types import ModelSummary, WorkerInfo, WorkerStatusMessage

from sie_router.types import ClusterStatus, WorkerHealth, WorkerState

logger = logging.getLogger(__name__)


class WorkerRegistry:
    """Registry of worker states, updated via WebSocket connections.

    Routing strategy:
    1. GPU match: Must match requested GPU type
    2. Model affinity: Prefer workers with model already loaded
    3. Load balance: Among matching workers, prefer lowest queue depth
    4. Health: Skip unhealthy workers
    """

    # Type alias for the callback
    WorkerCallback = Callable[["WorkerState"], Awaitable[None]]

    def __init__(
        self,
        *,
        heartbeat_timeout_s: float = 30.0,
        on_worker_healthy: WorkerCallback | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            heartbeat_timeout_s: Mark worker unhealthy after this many seconds
                without a heartbeat.
            on_worker_healthy: Async callback invoked when a worker becomes healthy.
                Used to trigger pool assignment on scale-from-zero.
        """
        self._workers: dict[str, WorkerState] = {}
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._lock = asyncio.Lock()
        self._on_worker_healthy = on_worker_healthy

        # Track request counts for QPS calculation
        self._request_counts: dict[str, int] = defaultdict(int)
        self._last_qps_calculation: float = 0.0
        self._current_qps: float = 0.0

    @property
    def workers(self) -> dict[str, WorkerState]:
        """Get all workers."""
        return self._workers

    @property
    def healthy_workers(self) -> list[WorkerState]:
        """Get list of healthy workers."""
        return [w for w in self._workers.values() if w.healthy]

    def get_worker(self, url: str) -> WorkerState | None:
        """Get worker by URL."""
        return self._workers.get(url)

    async def update_worker(
        self,
        url: str,
        state: WorkerStatusMessage,
    ) -> None:
        """Update worker state from WebSocket message.

        Args:
            url: Worker URL.
            state: Status message from worker's /ws/status endpoint.
        """
        became_healthy = False
        worker: WorkerState | None = None

        async with self._lock:
            now = time.time()

            is_new = url not in self._workers
            if is_new:
                self._workers[url] = WorkerState(url=url)
                logger.info("New worker discovered: %s", url)

            worker = self._workers[url]
            was_healthy = worker.healthy

            # Update state from message
            worker.name = state.get("name", url)
            worker.gpu_count = state.get("gpu_count", 1)
            worker.bundle = state.get("bundle", "default")
            # machine_profile from worker:
            # - In K8s: SIE_MACHINE_PROFILE env var (e.g., "l4-spot")
            # - Standalone: detected GPU type (e.g., "l4")
            worker.machine_profile = state.get("machine_profile", "")
            worker.models = state.get("loaded_models", [])

            # Compute aggregate queue depth from models
            models = state.get("models", [])
            worker.queue_depth = sum(m.get("queue_depth", 0) for m in models)

            # Extract GPU memory from gpus array (worker sends memory_*_bytes)
            # Aggregate across all GPUs, keep in bytes (no conversion)
            gpus = state.get("gpus", [])
            worker.memory_used_bytes = sum(g.get("memory_used_bytes", 0) for g in gpus)
            worker.memory_total_bytes = sum(g.get("memory_total_bytes", 0) for g in gpus)

            # Update heartbeat - worker is alive
            worker.last_heartbeat = now

            # Update health based on ready field
            # - ready=True: Worker is ready for traffic, mark HEALTHY
            # - ready=False: Worker is starting up, keep as UNKNOWN (not unhealthy)
            is_ready = state.get("ready", False)
            if is_ready:
                worker.health = WorkerHealth.HEALTHY
            elif worker.health != WorkerHealth.HEALTHY:
                # Worker is starting up - don't mark unhealthy, just not ready yet
                worker.health = WorkerHealth.UNKNOWN

            # Track if worker just became healthy (new or recovered)
            became_healthy = is_ready and (is_new or not was_healthy)

        # Call callback outside lock to avoid deadlocks
        if became_healthy and self._on_worker_healthy and worker:
            try:
                await self._on_worker_healthy(worker)
            except Exception as e:  # noqa: BLE001 — callback must not crash registry
                logger.warning("on_worker_healthy callback failed: %s", e)

    async def remove_worker(self, url: str) -> None:
        """Remove a worker from the registry.

        Args:
            url: Worker URL to remove.
        """
        async with self._lock:
            if url in self._workers:
                logger.info("Worker removed: %s", url)
                del self._workers[url]

    async def mark_unhealthy(self, url: str) -> None:
        """Mark a worker as unhealthy.

        Args:
            url: Worker URL.
        """
        async with self._lock:
            if url in self._workers:
                self._workers[url].health = WorkerHealth.UNHEALTHY
                logger.warning("Worker marked unhealthy: %s", url)

    async def check_heartbeats(self) -> list[str]:
        """Check for workers that have missed heartbeats.

        Returns:
            List of worker URLs that are now unhealthy.
        """
        now = time.time()
        unhealthy = []

        async with self._lock:
            for url, worker in self._workers.items():
                if worker.healthy:
                    elapsed = now - worker.last_heartbeat
                    if elapsed > self._heartbeat_timeout_s:
                        worker.health = WorkerHealth.UNHEALTHY
                        unhealthy.append(url)
                        logger.warning(
                            "Worker %s missed heartbeat (%.1fs since last)",
                            url,
                            elapsed,
                        )

        return unhealthy

    def select_worker(
        self,
        *,
        gpu: str | None = None,
        bundle: str | None = None,
        model: str | None = None,
        worker_urls: set[str] | None = None,
    ) -> WorkerState | None:
        """Select the best worker for a request.

        Routing strategy:
        1. Worker URL filter: If specified, only consider workers in the set
        2. GPU match: Must match requested GPU type
        3. Bundle match: Must match requested bundle (for multi-bundle clusters)
        4. Model affinity: Prefer workers with model already loaded
        5. Load balance: Among matching workers, prefer lowest queue depth
        6. Health: Skip unhealthy workers

        Args:
            gpu: Required GPU type (e.g., "l4", "a100-80gb").
            bundle: Required bundle (e.g., "default").
            model: Model name for affinity routing.
            worker_urls: If specified, only consider workers with these URLs
                (used for pool-based routing).

        Returns:
            Best matching worker, or None if no suitable worker found.
        """
        # Start with healthy workers
        candidates = [w for w in self._workers.values() if w.healthy]

        if not candidates:
            return None

        # Filter by worker URLs if specified (pool-based routing)
        if worker_urls is not None:
            candidates = [w for w in candidates if w.url in worker_urls]
            if not candidates:
                return None

        # Filter by machine_profile if specified
        if gpu:
            gpu_lower = gpu.lower()
            candidates = [w for w in candidates if w.machine_profile and w.machine_profile.lower() == gpu_lower]
            if not candidates:
                return None

        # Filter by bundle if specified (multi-bundle routing)
        if bundle:
            bundle_lower = bundle.lower()
            candidates = [w for w in candidates if w.bundle.lower() == bundle_lower]
            if not candidates:
                return None

        # Prefer workers with model already loaded
        if model:
            with_model = [w for w in candidates if model in w.models]
            if with_model:
                candidates = with_model

        # Pick worker with lowest queue depth
        return min(candidates, key=lambda w: w.queue_depth)

    def has_capacity(self, gpu: str, bundle: str | None = None) -> bool:
        """Check if any worker has capacity for the given machine_profile and bundle.

        Args:
            gpu: Machine profile to check (e.g., "l4", "l4-spot").
            bundle: Optional bundle to check. If None, any bundle matches.

        Returns:
            True if at least one healthy worker matches the machine_profile (and bundle).
        """
        gpu_lower = gpu.lower()
        bundle_lower = bundle.lower() if bundle else None
        for w in self._workers.values():
            if not w.healthy:
                continue
            if not w.machine_profile or w.machine_profile.lower() != gpu_lower:
                continue
            if bundle_lower and w.bundle.lower() != bundle_lower:
                continue
            return True
        return False

    def get_gpu_types(self) -> list[str]:
        """Get list of available machine_profile types across all healthy workers."""
        return list({w.machine_profile for w in self._workers.values() if w.healthy and w.machine_profile})

    def get_bundles(self) -> list[str]:
        """Get list of available bundles across all healthy workers."""
        return list({w.bundle for w in self._workers.values() if w.healthy and w.bundle})

    def get_models(self) -> dict[str, list[str]]:
        """Get models loaded on each worker.

        Returns:
            Dict mapping model name to list of worker URLs that have it loaded.
        """
        models: dict[str, list[str]] = defaultdict(list)
        for worker in self._workers.values():
            if worker.healthy:
                for model in worker.models:
                    models[model].append(worker.url)
        return dict(models)

    def record_request(self, worker_url: str) -> None:
        """Record a request to a worker for QPS tracking."""
        self._request_counts[worker_url] += 1

    def get_cluster_status(self) -> ClusterStatus:
        """Get aggregated cluster status for /ws/cluster-status.

        Returns:
            ClusterStatus with aggregated cluster information.

        Note: Field names match WorkerInfo/ClusterModelInfo from sie_server.types.status
        for consistency with sie-top consumption.
        """
        now = time.time()

        # Calculate QPS (requests per second) over last interval
        elapsed = now - self._last_qps_calculation
        if elapsed >= 1.0:
            total_requests = sum(self._request_counts.values())
            self._current_qps = total_requests / elapsed if elapsed > 0 else 0
            self._request_counts.clear()
            self._last_qps_calculation = now

        # Aggregate worker info
        workers_info: list[WorkerInfo] = []
        total_gpus = 0
        model_workers: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for worker in self._workers.values():
            # Use memory_*_bytes for consistency with GPUMetrics/WorkerInfo types
            # gpu field populated from machine_profile for display
            worker_info: WorkerInfo = {
                "name": worker.name,
                "url": worker.url,
                "gpu": worker.machine_profile,
                "gpu_count": worker.gpu_count,
                "loaded_models": worker.models,
                "queue_depth": worker.queue_depth,
                "memory_used_bytes": worker.memory_used_bytes,
                "memory_total_bytes": worker.memory_total_bytes,
                "healthy": worker.healthy,
            }
            workers_info.append(worker_info)

            if worker.healthy:
                total_gpus += worker.gpu_count
                for model in worker.models:
                    model_workers[model].append(
                        {
                            "worker": worker.name,
                            "gpu": worker.machine_profile,
                            "queue_depth": worker.queue_depth,
                        }
                    )

        # Aggregate model info - include state field for sie-top
        models_info: list[ModelSummary] = []
        for model, workers in model_workers.items():
            gpu_types = list({w["gpu"] for w in workers})
            total_queue = sum(w["queue_depth"] for w in workers)
            model_summary: ModelSummary = {
                "name": model,
                "state": "loaded",  # Models in model_workers are loaded by definition
                "worker_count": len(workers),
                "gpu_types": gpu_types,
                "total_queue_depth": total_queue,
            }
            models_info.append(model_summary)

        return ClusterStatus(
            timestamp=now,
            worker_count=len([w for w in self._workers.values() if w.healthy]),
            gpu_count=total_gpus,
            models_loaded=len(model_workers),
            total_qps=self._current_qps,
            workers=workers_info,
            models=models_info,
        )
