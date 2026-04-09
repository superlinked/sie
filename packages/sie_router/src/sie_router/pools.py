"""Resource pool management for SIE Router.

Pools provide capacity isolation by reserving workers for exclusive use.
Used for repeatable benchmarks, workload isolation, and capacity reservation.

Pool data is stored in K8s ConfigMaps with Leases for TTL management.
Multiple routers coordinate via K8s watch API.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC
from enum import Enum
from typing import TYPE_CHECKING, Any

import orjson

if TYPE_CHECKING:
    from sie_router.types import MachineProfile

logger = logging.getLogger(__name__)

# K8s retry settings
K8S_MAX_RETRIES = 10
K8S_INITIAL_BACKOFF_S = 0.1
K8S_MAX_BACKOFF_S = 5.0
K8S_CONFLICT_JITTER_S = (0.1, 0.3)  # (min, max) jitter for 409 conflicts (~2s total)


async def _k8s_retry[T](
    operation: Callable[[], Awaitable[T]],
    *,
    operation_name: str = "K8s operation",
    max_retries: int = K8S_MAX_RETRIES,
    initial_backoff_s: float = K8S_INITIAL_BACKOFF_S,
    max_backoff_s: float = K8S_MAX_BACKOFF_S,
) -> T:
    """Execute a K8s operation with retry and exponential backoff.

    Handles:
    - 409 Conflict: Optimistic concurrency conflict, retry immediately
    - 500/502/503/504: Transient server errors, retry with backoff
    - Other errors: Raise immediately

    Args:
        operation: Async callable that performs the K8s operation.
        operation_name: Name for logging.
        max_retries: Maximum number of retry attempts.
        initial_backoff_s: Initial backoff duration.
        max_backoff_s: Maximum backoff duration.

    Returns:
        Result of the operation.

    Raises:
        ApiException: If all retries exhausted or non-retryable error.
    """
    from kubernetes.client.rest import ApiException

    backoff_s = initial_backoff_s
    last_exception: ApiException | None = None

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except ApiException as e:
            last_exception = e

            # 409 Conflict: optimistic concurrency conflict
            # Retry with jitter (caller re-reads resource on next attempt)
            if e.status == 409:
                if attempt < max_retries:
                    jitter = random.uniform(*K8S_CONFLICT_JITTER_S)  # noqa: S311 — retry jitter, not security
                    logger.debug(
                        "%s conflict (409), retrying in %.2fs (attempt %d/%d)",
                        operation_name,
                        jitter,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(jitter)
                    continue
                logger.warning("%s failed after %d conflict retries", operation_name, max_retries)
                raise

            # 5xx: transient server errors, retry with backoff
            if 500 <= e.status < 600:
                if attempt < max_retries:
                    jitter = random.uniform(0, backoff_s * 0.2)  # noqa: S311 — retry jitter, not security
                    sleep_time = backoff_s + jitter
                    logger.debug(
                        "%s server error (%d), retrying in %.2fs (attempt %d/%d)",
                        operation_name,
                        e.status,
                        sleep_time,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(sleep_time)
                    backoff_s = min(backoff_s * 2, max_backoff_s)
                    continue
                logger.warning(
                    "%s failed after %d retries: %d %s",
                    operation_name,
                    max_retries,
                    e.status,
                    e.reason,
                )
                raise

            # Other errors (4xx except 409): non-retryable
            raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError(f"{operation_name} failed without exception")


# Default pool name - always available, protected from deletion
DEFAULT_POOL_NAME = "default"


class DefaultPoolProtectedError(Exception):
    """Raised when trying to delete the default pool."""

    def __init__(self) -> None:
        super().__init__(f"Cannot delete the default pool '{DEFAULT_POOL_NAME}'")


class InvalidMachineProfileError(Exception):
    """Raised when pool creation references an unknown machine profile."""

    def __init__(self, invalid_profiles: list[str], valid_profiles: list[str]) -> None:
        self.invalid_profiles = invalid_profiles
        self.valid_profiles = valid_profiles
        super().__init__(f"Unknown machine profiles: {invalid_profiles}. Valid profiles: {valid_profiles}")


class PoolState(Enum):
    """Pool lifecycle state."""

    PENDING = "pending"  # Waiting for worker assignment
    ACTIVE = "active"  # Workers assigned and ready
    EXPIRED = "expired"  # Lease expired, awaiting cleanup


@dataclass
class PoolSpec:
    """Pool specification (what the pool needs)."""

    name: str
    gpus: dict[str, int] = field(default_factory=dict)  # e.g., {"l4": 2, "a100-40gb": 1}
    bundle: str | None = None  # Optional bundle filter (e.g., "default")
    minimum_worker_count: int = 0  # Minimum warm workers (prevents scale-to-zero)


@dataclass
class AssignedWorker:
    """A worker assigned to a pool."""

    name: str
    url: str
    gpu: str  # Actually machine_profile (e.g., "l4-spot"), kept as "gpu" for JSON compat


@dataclass
class PoolStatus:
    """Pool status (current state)."""

    state: PoolState = PoolState.PENDING
    assigned_workers: list[AssignedWorker] = field(default_factory=list)
    created_at: float = 0.0
    last_renewed: float = 0.0


@dataclass
class Pool:
    """A resource pool combining spec and status."""

    spec: PoolSpec
    status: PoolStatus = field(default_factory=PoolStatus)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def is_active(self) -> bool:
        return self.status.state == PoolState.ACTIVE

    def has_worker(self, worker_url: str) -> bool:
        """Check if a worker is assigned to this pool."""
        return any(w.url == worker_url for w in self.status.assigned_workers)

    def get_workers_for_gpu(self, gpu: str) -> list[AssignedWorker]:
        """Get workers in this pool with the specified GPU type."""
        gpu_lower = gpu.lower()
        return [w for w in self.status.assigned_workers if w.gpu.lower() == gpu_lower]


class PoolManager:
    """Manages resource pools.

    In K8s mode, pools are stored as ConfigMaps with Leases.
    In local mode (for testing), pools are stored in-memory.
    """

    # ConfigMap label for pool discovery
    POOL_LABEL = "sie.superlinked.com/type"
    POOL_LABEL_VALUE = "pool"
    CONFIGMAP_PREFIX = "sie-pool-"

    # Lease settings
    DEFAULT_LEASE_DURATION_S = 1200  # 20 minutes — covers rolling upgrades up to 15 min

    def __init__(
        self,
        *,
        use_kubernetes: bool = False,
        k8s_namespace: str = "sie",
        lease_duration_s: float = DEFAULT_LEASE_DURATION_S,
        machine_profiles: dict[str, MachineProfile] | None = None,
    ) -> None:
        """Initialize the pool manager.

        Args:
            use_kubernetes: Use K8s ConfigMaps/Leases for storage.
            k8s_namespace: K8s namespace for pools.
            lease_duration_s: Lease duration in seconds.
            machine_profiles: Valid machine profiles for pool creation validation.
                If None, profile validation is skipped (for backward compat).
        """
        self._use_kubernetes = use_kubernetes
        self._k8s_namespace = k8s_namespace
        self._lease_duration_s = lease_duration_s
        self._machine_profiles = machine_profiles or {}

        # In-memory pool cache (synced from K8s in K8s mode)
        self._pools: dict[str, Pool] = {}
        self._lock = asyncio.Lock()

        # K8s client (lazy init)
        self._k8s_client: Any = None
        self._k8s_core_v1: Any = None
        self._k8s_coordination_v1: Any = None
        self._watch_task: asyncio.Task | None = None

    @property
    def use_kubernetes(self) -> bool:
        """Whether Kubernetes mode is active (read-only)."""
        return self._use_kubernetes

    @property
    def machine_profiles(self) -> dict[str, MachineProfile]:
        """Get configured machine profiles."""
        return self._machine_profiles

    @property
    def pools(self) -> dict[str, Pool]:
        """Get all pools."""
        return self._pools

    def get_pool(self, name: str) -> Pool | None:
        """Get a pool by name."""
        return self._pools.get(name)

    async def start(self) -> None:
        """Start the pool manager."""
        if self._use_kubernetes:
            await self._init_kubernetes()
            # Start watching for pool changes
            self._watch_task = asyncio.create_task(self._watch_pools())

    async def create_default_pool(self) -> None:
        """Create the default pool with access to all machine profiles.

        The default pool is used for requests that don't specify a pool.
        It has unlimited capacity per machine profile (999 per profile).
        """
        if not self._machine_profiles:
            logger.info("No machine profiles configured, skipping default pool creation")
            return

        # Create gpus dict with high limits for all profiles
        gpus = dict.fromkeys(self._machine_profiles, 999)

        await self.create_pool(DEFAULT_POOL_NAME, gpus)
        logger.info("Created default pool with profiles: %s", list(gpus.keys()))

    async def stop(self) -> None:
        """Stop the pool manager."""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

    async def _init_kubernetes(self) -> None:
        """Initialize Kubernetes client."""
        try:
            from kubernetes import client, config

            # Try in-cluster config first, fall back to kubeconfig
            try:
                config.load_incluster_config()
                logger.info("Using in-cluster K8s config")
            except config.ConfigException:
                config.load_kube_config()
                logger.info("Using kubeconfig for K8s")

            self._k8s_client = client
            self._k8s_core_v1 = client.CoreV1Api()
            self._k8s_coordination_v1 = client.CoordinationV1Api()

            # Load existing pools
            await self._sync_pools_from_k8s()

        except ImportError:
            logger.error("kubernetes package not installed, K8s mode unavailable")
            raise
        except Exception as e:
            logger.error("Failed to initialize K8s client: %s", e)
            raise

    async def _sync_pools_from_k8s(self) -> None:
        """Sync pools from K8s ConfigMaps."""
        if not self._k8s_core_v1:
            return

        try:
            # List all pool ConfigMaps
            configmaps = self._k8s_core_v1.list_namespaced_config_map(
                namespace=self._k8s_namespace,
                label_selector=f"{self.POOL_LABEL}={self.POOL_LABEL_VALUE}",
            )

            async with self._lock:
                for cm in configmaps.items:
                    pool = self._configmap_to_pool(cm)
                    if pool:
                        self._pools[pool.name] = pool
                        logger.info("Loaded pool from K8s: %s", pool.name)

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Failed to sync pools from K8s: %s", e)

    def _configmap_to_pool(self, cm: Any) -> Pool | None:
        """Convert a K8s ConfigMap to a Pool object."""
        try:
            name = cm.metadata.name
            name = name.removeprefix(self.CONFIGMAP_PREFIX)

            data = cm.data or {}

            # Parse spec
            spec_json = data.get("spec", "{}")
            spec_data = orjson.loads(spec_json)
            spec = PoolSpec(
                name=name,
                gpus=spec_data.get("gpus", {}),
                bundle=spec_data.get("bundle"),
                minimum_worker_count=spec_data.get("minimum_worker_count", 0),
            )

            # Parse status
            status_json = data.get("status", "{}")
            status_data = orjson.loads(status_json)
            status = PoolStatus(
                state=PoolState(status_data.get("state", "pending")),
                assigned_workers=[
                    AssignedWorker(
                        name=w.get("name", ""),
                        url=w.get("url", ""),
                        gpu=w.get("gpu", ""),
                    )
                    for w in status_data.get("assigned_workers", [])
                ],
                created_at=status_data.get("created_at", 0.0),
                last_renewed=status_data.get("last_renewed", 0.0),
            )

            return Pool(spec=spec, status=status)

        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Failed to parse ConfigMap %s: %s", cm.metadata.name, e)
            return None

    def _pool_to_configmap_data(self, pool: Pool) -> dict[str, str]:
        """Convert a Pool to ConfigMap data."""
        spec_data: dict[str, Any] = {
            "gpus": pool.spec.gpus,
        }
        if pool.spec.bundle:
            spec_data["bundle"] = pool.spec.bundle
        spec_data["minimum_worker_count"] = pool.spec.minimum_worker_count
        status_data = {
            "state": pool.status.state.value,
            "assigned_workers": [{"name": w.name, "url": w.url, "gpu": w.gpu} for w in pool.status.assigned_workers],
            "created_at": pool.status.created_at,
            "last_renewed": pool.status.last_renewed,
        }
        return {
            "spec": orjson.dumps(spec_data).decode(),
            "status": orjson.dumps(status_data).decode(),
        }

    async def _watch_pools(self) -> None:
        """Watch K8s for pool changes.

        Uses a short timeout and periodic resync to avoid blocking the event loop
        since the Kubernetes Python client's watch is synchronous.
        """
        if not self._k8s_core_v1:
            return

        from kubernetes import watch
        from kubernetes.client.rest import ApiException

        w = watch.Watch()
        resource_version: str | None = None
        backoff_s = 1.0

        while True:
            try:
                if resource_version is None:
                    pool_list = await asyncio.to_thread(
                        self._k8s_core_v1.list_namespaced_config_map,
                        namespace=self._k8s_namespace,
                        label_selector=f"{self.POOL_LABEL}={self.POOL_LABEL_VALUE}",
                    )
                    resource_version = pool_list.metadata.resource_version

                    for cm in pool_list.items:
                        pool = self._configmap_to_pool(cm)
                        if not pool:
                            continue
                        async with self._lock:
                            self._pools[pool.name] = pool

                logger.info("Starting pool watch")
                # Use timeout_seconds to prevent blocking too long
                # and resource_version to resume from where we left off
                stream = w.stream(
                    self._k8s_core_v1.list_namespaced_config_map,
                    namespace=self._k8s_namespace,
                    label_selector=f"{self.POOL_LABEL}={self.POOL_LABEL_VALUE}",
                    resource_version=resource_version,
                    timeout_seconds=30,  # Short timeout to yield control
                )

                # Run the synchronous watch in a thread pool to avoid blocking
                def process_events() -> list[tuple[str, Any]]:
                    """Process events in thread, return list of (event_type, pool)."""
                    events = []
                    for event in stream:
                        event_type = event["type"]
                        cm = event["object"]
                        events.append((event_type, cm))
                    return events

                events = await asyncio.to_thread(process_events)

                # Process events in the async context
                for event_type, cm in events:
                    if cm.metadata and cm.metadata.resource_version:
                        resource_version = cm.metadata.resource_version
                    pool = self._configmap_to_pool(cm)
                    if not pool:
                        continue

                    async with self._lock:
                        if event_type in ("ADDED", "MODIFIED"):
                            self._pools[pool.name] = pool
                            logger.debug("Pool %s: %s", event_type, pool.name)
                        elif event_type == "DELETED":
                            self._pools.pop(pool.name, None)
                            logger.debug("Pool DELETED: %s", pool.name)

                # Brief yield before restarting watch
                await asyncio.sleep(0.1)
                backoff_s = 1.0

            except asyncio.CancelledError:
                logger.info("Pool watch cancelled")
                break
            except ApiException as e:
                if e.status == 410:
                    logger.warning("Pool watch expired (410 Gone); relisting")
                    resource_version = None
                    continue
                logger.error("Pool watch API error: %s", e)
                jitter = random.uniform(0, backoff_s * 0.2)  # noqa: S311 — retry jitter, not security
                await asyncio.sleep(backoff_s + jitter)
                backoff_s = min(backoff_s * 2, 30.0)
            except (ValueError, KeyError, OSError) as e:
                logger.error("Pool watch error: %s", e)
                jitter = random.uniform(0, backoff_s * 0.2)  # noqa: S311 — retry jitter, not security
                await asyncio.sleep(backoff_s + jitter)
                backoff_s = min(backoff_s * 2, 30.0)

    def _validate_machine_profiles(self, gpus: dict[str, int]) -> None:
        """Validate that all requested GPU keys are valid machine profiles.

        Args:
            gpus: GPU requirements dict with machine profile keys.

        Raises:
            InvalidMachineProfileError: If any key is not a valid machine profile.
        """
        if not self._machine_profiles:
            # No profiles configured, skip validation (backward compat)
            return

        invalid = [key for key in gpus if key not in self._machine_profiles]
        if invalid:
            raise InvalidMachineProfileError(
                invalid_profiles=invalid,
                valid_profiles=list(self._machine_profiles.keys()),
            )

    async def create_pool(
        self,
        name: str,
        gpus: dict[str, int],
        bundle: str | None = None,
        minimum_worker_count: int = 0,
    ) -> Pool:
        """Create a pool or return existing one (idempotent).

        Args:
            name: Pool name.
            gpus: GPU/machine profile requirements, e.g., {"l4-spot": 2, "a100-40gb": 1}.
            bundle: Optional bundle filter. When set, only workers with this bundle
                will be assigned to the pool.
            minimum_worker_count: Desired minimum number of warm workers in the pool.
                Stored in pool spec; enforcement depends on cluster autoscaler config.

        Returns:
            The created or existing pool.

        Raises:
            InvalidMachineProfileError: If gpus contains unknown machine profile keys.
        """
        # Validate machine profiles before acquiring lock
        self._validate_machine_profiles(gpus)

        now = time.time()

        async with self._lock:
            # Check if pool already exists - return it (idempotent)
            existing = self._pools.get(name)
            if existing:
                existing.status.last_renewed = now
                if self._use_kubernetes:
                    await self._update_pool_in_k8s(existing)
                return existing

            # Create new pool
            pool = Pool(
                spec=PoolSpec(
                    name=name,
                    gpus=gpus,
                    bundle=bundle,
                    minimum_worker_count=minimum_worker_count,
                ),
                status=PoolStatus(
                    state=PoolState.PENDING,
                    created_at=now,
                    last_renewed=now,
                ),
            )

            self._pools[name] = pool

            if self._use_kubernetes:
                await self._create_pool_in_k8s(pool)

            logger.info("Created pool: %s (gpus=%s, minimum_worker_count=%d)", name, gpus, minimum_worker_count)
            return pool

    async def _create_pool_in_k8s(self, pool: Pool) -> None:
        """Create pool ConfigMap and Lease in K8s.

        Uses create-or-update pattern with optimistic concurrency.
        Retries automatically on transient server errors.
        """
        if not self._k8s_core_v1 or not self._k8s_coordination_v1:
            return

        from datetime import datetime

        from kubernetes.client.rest import ApiException

        cm_name = f"{self.CONFIGMAP_PREFIX}{pool.name}"

        async def create_or_update_configmap() -> None:
            """Create ConfigMap or update if it already exists."""
            cm_body = self._k8s_client.V1ConfigMap(
                metadata=self._k8s_client.V1ObjectMeta(
                    name=cm_name,
                    namespace=self._k8s_namespace,
                    labels={self.POOL_LABEL: self.POOL_LABEL_VALUE},
                ),
                data=self._pool_to_configmap_data(pool),
            )

            try:
                await asyncio.to_thread(
                    self._k8s_core_v1.create_namespaced_config_map,
                    namespace=self._k8s_namespace,
                    body=cm_body,
                )
            except ApiException as e:
                if e.status == 409:
                    # Already exists - read to get resourceVersion, then update
                    existing = await asyncio.to_thread(
                        self._k8s_core_v1.read_namespaced_config_map,
                        name=cm_name,
                        namespace=self._k8s_namespace,
                    )
                    existing.data = self._pool_to_configmap_data(pool)
                    existing.metadata.labels = {self.POOL_LABEL: self.POOL_LABEL_VALUE}
                    await asyncio.to_thread(
                        self._k8s_core_v1.replace_namespaced_config_map,
                        name=cm_name,
                        namespace=self._k8s_namespace,
                        body=existing,
                    )
                else:
                    raise

        async def create_or_update_lease() -> None:
            """Create Lease or update if it already exists."""
            lease_body = self._k8s_client.V1Lease(
                metadata=self._k8s_client.V1ObjectMeta(
                    name=cm_name,
                    namespace=self._k8s_namespace,
                ),
                spec=self._k8s_client.V1LeaseSpec(
                    holder_identity=pool.name,
                    lease_duration_seconds=int(self._lease_duration_s),
                    renew_time=datetime.now(UTC),
                ),
            )

            try:
                await asyncio.to_thread(
                    self._k8s_coordination_v1.create_namespaced_lease,
                    namespace=self._k8s_namespace,
                    body=lease_body,
                )
            except ApiException as e:
                if e.status == 409:
                    # Already exists - read to get resourceVersion, then update
                    existing = await asyncio.to_thread(
                        self._k8s_coordination_v1.read_namespaced_lease,
                        name=cm_name,
                        namespace=self._k8s_namespace,
                    )
                    existing.spec.renew_time = datetime.now(UTC)
                    existing.spec.lease_duration_seconds = int(self._lease_duration_s)
                    await asyncio.to_thread(
                        self._k8s_coordination_v1.replace_namespaced_lease,
                        name=cm_name,
                        namespace=self._k8s_namespace,
                        body=existing,
                    )
                else:
                    raise

        # Create both with retry for transient errors
        await _k8s_retry(
            create_or_update_configmap,
            operation_name=f"Create pool ConfigMap '{pool.name}'",
        )
        await _k8s_retry(
            create_or_update_lease,
            operation_name=f"Create pool Lease '{pool.name}'",
        )

    async def _update_pool_in_k8s(self, pool: Pool) -> None:
        """Update pool ConfigMap in K8s with optimistic concurrency.

        Uses read-modify-write pattern with resourceVersion for conflict detection.
        Retries automatically on 409 Conflict or transient server errors.
        """
        if not self._k8s_core_v1:
            return

        cm_name = f"{self.CONFIGMAP_PREFIX}{pool.name}"

        async def do_update() -> None:
            # Read current ConfigMap to get resourceVersion
            existing = await asyncio.to_thread(
                self._k8s_core_v1.read_namespaced_config_map,
                name=cm_name,
                namespace=self._k8s_namespace,
            )

            # Update with new data, preserving resourceVersion
            existing.data = self._pool_to_configmap_data(pool)
            existing.metadata.labels = {self.POOL_LABEL: self.POOL_LABEL_VALUE}

            # Replace with optimistic lock (resourceVersion in metadata)
            await asyncio.to_thread(
                self._k8s_core_v1.replace_namespaced_config_map,
                name=cm_name,
                namespace=self._k8s_namespace,
                body=existing,
            )

        await _k8s_retry(do_update, operation_name=f"Update pool ConfigMap '{pool.name}'")

    async def delete_pool(self, name: str) -> bool:
        """Delete a pool.

        Args:
            name: Pool name.

        Returns:
            True if pool was deleted, False if not found.

        Raises:
            DefaultPoolProtectedError: If trying to delete the default pool.
        """
        # Protect the default pool from deletion
        if name == DEFAULT_POOL_NAME:
            raise DefaultPoolProtectedError

        async with self._lock:
            if name not in self._pools:
                return False

            del self._pools[name]

            if self._use_kubernetes:
                await self._delete_pool_in_k8s(name)

            logger.info("Deleted pool: %s", name)
            return True

    async def _delete_pool_in_k8s(self, name: str) -> None:
        """Delete pool ConfigMap and Lease from K8s.

        Retries automatically on transient server errors. 404 (not found) is
        treated as success (idempotent delete).
        """
        if not self._k8s_core_v1 or not self._k8s_coordination_v1:
            return

        from kubernetes.client.rest import ApiException

        cm_name = f"{self.CONFIGMAP_PREFIX}{name}"

        async def delete_configmap() -> None:
            try:
                await asyncio.to_thread(
                    self._k8s_core_v1.delete_namespaced_config_map,
                    name=cm_name,
                    namespace=self._k8s_namespace,
                )
            except ApiException as e:
                if e.status == 404:
                    # Already deleted - success
                    return
                raise

        async def delete_lease() -> None:
            try:
                await asyncio.to_thread(
                    self._k8s_coordination_v1.delete_namespaced_lease,
                    name=cm_name,
                    namespace=self._k8s_namespace,
                )
            except ApiException as e:
                if e.status == 404:
                    # Already deleted - success
                    return
                raise

        # Delete both with retry for transient errors
        try:
            await _k8s_retry(delete_configmap, operation_name=f"Delete pool ConfigMap '{name}'")
        except Exception as e:  # noqa: BLE001 — K8s cleanup must not raise
            logger.warning("Failed to delete ConfigMap %s: %s", cm_name, e)

        try:
            await _k8s_retry(delete_lease, operation_name=f"Delete pool Lease '{name}'")
        except Exception as e:  # noqa: BLE001 — K8s cleanup must not raise
            logger.warning("Failed to delete Lease %s: %s", cm_name, e)

    async def renew_lease(self, name: str) -> bool:
        """Renew a pool's lease.

        Args:
            name: Pool name.

        Returns:
            True if renewed, False if pool not found.
        """
        async with self._lock:
            pool = self._pools.get(name)
            if not pool:
                return False

            pool.status.last_renewed = time.time()

            if self._use_kubernetes:
                await self._update_pool_in_k8s(pool)
                await self._renew_lease_in_k8s(pool)

            return True

    async def _renew_lease_in_k8s(self, pool: Pool) -> None:
        """Renew the K8s Lease for a pool with optimistic concurrency.

        Uses read-modify-write pattern with resourceVersion for conflict detection.
        Retries automatically on 409 Conflict or transient server errors.
        """
        if not self._k8s_coordination_v1:
            return

        from datetime import datetime

        cm_name = f"{self.CONFIGMAP_PREFIX}{pool.name}"

        async def do_renew() -> None:
            # Get the existing lease to preserve resourceVersion
            existing_lease = await asyncio.to_thread(
                self._k8s_coordination_v1.read_namespaced_lease,
                name=cm_name,
                namespace=self._k8s_namespace,
            )

            # Update the lease with new renew time
            existing_lease.spec.renew_time = datetime.now(UTC)

            # Replace with optimistic lock (resourceVersion in metadata)
            await asyncio.to_thread(
                self._k8s_coordination_v1.replace_namespaced_lease,
                name=cm_name,
                namespace=self._k8s_namespace,
                body=existing_lease,
            )

        try:
            await _k8s_retry(do_renew, operation_name=f"Renew pool Lease '{pool.name}'")
        except Exception as e:  # noqa: BLE001 — K8s lease renewal is best-effort
            # Log but don't fail - lease renewal is best-effort
            logger.warning("Failed to renew lease in K8s for pool '%s': %s", pool.name, e)

    async def check_expired_leases(self) -> list[str]:
        """Check for pools with expired leases and clean them up.

        Note: The default pool is never expired.

        Returns:
            List of pool names that were cleaned up.
        """
        now = time.time()
        expired = []

        async with self._lock:
            for name, pool in list(self._pools.items()):
                # Default pool never expires
                if name == DEFAULT_POOL_NAME:
                    continue
                if now - pool.status.last_renewed > self._lease_duration_s:
                    pool.status.state = PoolState.EXPIRED
                    expired.append(name)

        # Clean up expired pools
        for name in expired:
            await self.delete_pool(name)
            logger.info("Cleaned up expired pool: %s", name)

        return expired

    def assign_workers(
        self,
        pool: Pool,
        available_workers: list[tuple[str, str, str, str]],  # (name, url, gpu, bundle)
    ) -> bool:
        """Assign workers to a pool based on its GPU and bundle requirements.

        Args:
            pool: Pool to assign workers to.
            available_workers: List of (name, url, gpu, bundle) tuples for unassigned workers.

        Returns:
            True if all requirements met, False if partial/no assignment.
        """
        assigned = []

        # Filter by bundle if pool spec requires it
        if pool.spec.bundle:
            available_workers = [
                (name, url, gpu, bundle) for name, url, gpu, bundle in available_workers if bundle == pool.spec.bundle
            ]

        # Group available workers by GPU type
        workers_by_gpu: dict[str, list[tuple[str, str, str, str]]] = {}
        for name, url, gpu, bundle in available_workers:
            gpu_lower = gpu.lower()
            if gpu_lower not in workers_by_gpu:
                workers_by_gpu[gpu_lower] = []
            workers_by_gpu[gpu_lower].append((name, url, gpu, bundle))

        # Try to fulfill GPU requirements
        all_met = True
        for gpu_type, count_needed in pool.spec.gpus.items():
            gpu_lower = gpu_type.lower()
            available = workers_by_gpu.get(gpu_lower, [])

            if len(available) < count_needed:
                all_met = False
                count_needed = len(available)  # Take what we can

            for i in range(count_needed):
                name, url, gpu, _bundle = available[i]
                assigned.append(AssignedWorker(name=name, url=url, gpu=gpu))

            # Remove assigned workers from available pool
            if gpu_lower in workers_by_gpu:
                workers_by_gpu[gpu_lower] = available[count_needed:]

        pool.status.assigned_workers = assigned
        pool.status.state = PoolState.ACTIVE if all_met and assigned else PoolState.PENDING

        return all_met

    def get_assigned_worker_urls(self) -> set[str]:
        """Get URLs of all workers currently assigned to pools."""
        urls = set()
        for pool in self._pools.values():
            if pool.status.state == PoolState.ACTIVE:
                for worker in pool.status.assigned_workers:
                    urls.add(worker.url)
        return urls

    def get_pool_for_worker(self, worker_url: str) -> Pool | None:
        """Get the pool that a worker is assigned to, if any."""
        for pool in self._pools.values():
            if pool.has_worker(worker_url):
                return pool
        return None


def parse_gpu_param(gpu: str) -> tuple[str | None, str]:
    """Parse GPU parameter to extract pool name and GPU type.

    Args:
        gpu: GPU parameter, e.g., "pool_name/l4" or "l4".

    Returns:
        Tuple of (pool_name, gpu_type). pool_name is None if not specified.
    """
    if "/" in gpu:
        parts = gpu.split("/", 1)
        return parts[0], parts[1]
    return None, gpu
