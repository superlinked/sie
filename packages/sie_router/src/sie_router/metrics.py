"""Prometheus metrics for the router."""

import logging
import time

from fastapi import APIRouter, Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from sie_router.registry import WorkerRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])

# Request metrics
# Labels use machine_profile (e.g., "l4-spot") instead of raw gpu type
REQUEST_COUNT = Counter(
    "sie_router_requests_total",
    "Total number of requests proxied",
    ["endpoint", "status", "machine_profile"],
)

REQUEST_LATENCY = Histogram(
    "sie_router_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "machine_profile"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

PROVISIONING_RESPONSES = Counter(
    "sie_router_provisioning_responses_total",
    "Number of 202 provisioning responses",
    ["machine_profile"],
)

# Scale-from-zero metrics
# This gauge is used by KEDA to trigger scale-up when no workers exist
# Labels: machine_profile (e.g., "l4-spot"), bundle (dependency bundle)
# WorkerGroups are keyed by (machine_profile, bundle)
# Value = number of pools with pending demand (enables scaling to N workers for N pools)
PENDING_DEMAND = Gauge(
    "sie_router_pending_demand",
    "Number of pools with pending demand per WorkerGroup (for KEDA scaling)",
    ["machine_profile", "bundle"],
)

# Track pools with pending demand per WorkerGroup
# Key format: "machine_profile:bundle" -> set of pool names with pending demand
_pools_with_demand: dict[str, set[str]] = {}
# Track last demand timestamp per pool for decay
# Key format: "machine_profile:bundle:pool_name" -> timestamp
_pool_demand_time: dict[str, float] = {}
# Demand expires after 2 minutes of no requests
DEMAND_EXPIRY_SECONDS = 120.0


def _demand_key(machine_profile: str, bundle: str) -> str:
    """Create a composite key for demand tracking."""
    return f"{machine_profile}:{bundle}"


def _pool_key(machine_profile: str, bundle: str, pool_name: str) -> str:
    """Create a key for per-pool demand tracking."""
    return f"{machine_profile}:{bundle}:{pool_name}"


def record_pending_demand(machine_profile: str, bundle: str, pool_name: str = "default") -> None:
    """Record that there is unmet demand for a WorkerGroup from a specific pool.

    Called when router returns 202 (no capacity available).
    KEDA uses the aggregate count to scale up workers.

    Args:
        machine_profile: Machine profile name (e.g., "l4-spot", "a100-40gb").
        bundle: Dependency bundle (e.g., "default").
        pool_name: Name of the pool requesting capacity.
    """
    key = _demand_key(machine_profile, bundle)
    pool_key = _pool_key(machine_profile, bundle, pool_name)

    # Track this pool as having demand
    if key not in _pools_with_demand:
        _pools_with_demand[key] = set()
    _pools_with_demand[key].add(pool_name)
    _pool_demand_time[pool_key] = time.monotonic()

    # Update gauge to number of pools with demand
    PENDING_DEMAND.labels(machine_profile=machine_profile, bundle=bundle).set(len(_pools_with_demand[key]))


def clear_pending_demand(machine_profile: str, bundle: str) -> None:
    """Clear all pending demand for a WorkerGroup when capacity becomes available.

    Called when workers for this WorkerGroup come online.
    """
    key = _demand_key(machine_profile, bundle)

    # Clear all pool demands for this WorkerGroup
    if key in _pools_with_demand:
        for pool_name in list(_pools_with_demand[key]):
            pool_key = _pool_key(machine_profile, bundle, pool_name)
            _pool_demand_time.pop(pool_key, None)
        _pools_with_demand.pop(key, None)

    PENDING_DEMAND.labels(machine_profile=machine_profile, bundle=bundle).set(0)


def _update_pending_demand(active_machine_profiles: set[str] | None = None) -> None:
    """Decay pending demand after expiry period.

    If no requests for a pool in DEMAND_EXPIRY_SECONDS, remove it from demand tracking.
    However, if the machine_profile has an active pool using it, keep the demand.

    Args:
        active_machine_profiles: Set of machine_profile names with active pools.
            If provided, these profiles won't have their pending_demand decayed.
    """
    now = time.monotonic()
    active_profiles = active_machine_profiles or set()

    # Check each pool's demand timestamp
    for pool_key in list(_pool_demand_time.keys()):
        parts = pool_key.split(":", 2)
        if len(parts) != 3:
            continue

        machine_profile, bundle, pool_name = parts
        key = _demand_key(machine_profile, bundle)

        # Don't decay if machine_profile has an active pool - refresh the timestamp
        if machine_profile in active_profiles:
            _pool_demand_time[pool_key] = now
            continue

        # Check if this pool's demand has expired
        if now - _pool_demand_time[pool_key] > DEMAND_EXPIRY_SECONDS:
            _pool_demand_time.pop(pool_key, None)
            if key in _pools_with_demand:
                _pools_with_demand[key].discard(pool_name)
                if not _pools_with_demand[key]:
                    _pools_with_demand.pop(key, None)
                    PENDING_DEMAND.labels(machine_profile=machine_profile, bundle=bundle).set(0)
                else:
                    PENDING_DEMAND.labels(machine_profile=machine_profile, bundle=bundle).set(
                        len(_pools_with_demand[key])
                    )


# Active lease GPU count — KEDA 3rd trigger
# Reports the total GPU count from non-default pools with active leases.
# This prevents KEDA scale-down during model loading gaps when both
# pending_demand (already expired) and queue_depth (not yet reporting) are 0.
ACTIVE_LEASE_GPUS = Gauge(
    "sie_router_active_lease_gpus",
    "Total GPUs requested by active leases per WorkerGroup (for KEDA scaling)",
    ["machine_profile", "bundle"],
)

# Track previous lease GPU label combos for staleness clearing
_previous_lease_gpu_keys: set[tuple[str, str]] = set()


def _clear_stale_lease_gpus(current_keys: dict[tuple[str, str], int]) -> None:
    """Set ACTIVE_LEASE_GPUS to 0 for label combos that are no longer active."""
    global _previous_lease_gpu_keys
    current = set(current_keys.keys())
    stale = _previous_lease_gpu_keys - current
    for machine_profile, bundle in stale:
        ACTIVE_LEASE_GPUS.labels(machine_profile=machine_profile, bundle=bundle).set(0)
    _previous_lease_gpu_keys = current


# Worker metrics
WORKER_COUNT = Gauge(
    "sie_router_workers",
    "Number of workers",
    ["status"],
)

WORKER_QUEUE_DEPTH = Gauge(
    "sie_router_worker_queue_depth",
    "Queue depth per worker",
    ["worker", "machine_profile", "bundle"],
)

WORKER_MEMORY_USED = Gauge(
    "sie_router_worker_memory_used_bytes",
    "GPU memory used per worker (bytes)",
    ["worker", "machine_profile", "bundle"],
)

# Model metrics
MODEL_WORKERS = Gauge(
    "sie_router_model_workers",
    "Number of workers with model loaded",
    ["model"],
)


def update_worker_metrics(
    registry: WorkerRegistry,
    active_machine_profiles: set[str] | None = None,
) -> None:
    """Update Prometheus metrics from registry state.

    Called periodically to update worker/model gauges.

    Args:
        registry: Worker registry with current worker states.
        active_machine_profiles: Set of machine_profile names with active pools.
            These profiles won't have their pending_demand decayed.
    """
    # Decay old pending demand, but preserve demand for profiles with active pools
    _update_pending_demand(active_machine_profiles)

    # Worker counts
    healthy = sum(1 for w in registry.workers.values() if w.healthy)
    unhealthy = len(registry.workers) - healthy

    WORKER_COUNT.labels(status="healthy").set(healthy)
    WORKER_COUNT.labels(status="unhealthy").set(unhealthy)

    # Per-worker metrics and track WorkerGroups with capacity
    # Key: (machine_profile, bundle) tuple
    workergroups_with_capacity: set[tuple[str, str]] = set()
    for worker in registry.workers.values():
        profile = worker.machine_profile
        WORKER_QUEUE_DEPTH.labels(worker=worker.name, machine_profile=profile, bundle=worker.bundle).set(
            worker.queue_depth
        )
        WORKER_MEMORY_USED.labels(worker=worker.name, machine_profile=profile, bundle=worker.bundle).set(
            worker.memory_used_bytes
        )
        if worker.healthy:
            workergroups_with_capacity.add((profile, worker.bundle))

    # Clear pending demand for WorkerGroups that now have capacity
    for key in list(_pools_with_demand.keys()):
        parts = key.split(":", 1)
        if len(parts) == 2:
            machine_profile, bundle = parts
            if (machine_profile, bundle) in workergroups_with_capacity:
                clear_pending_demand(machine_profile, bundle)

    # Model metrics
    model_workers = registry.get_models()
    for model, workers in model_workers.items():
        MODEL_WORKERS.labels(model=model).set(len(workers))


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    registry: WorkerRegistry = request.app.state.registry

    # Get active machine profiles from PoolManager (if available)
    # A machine profile is "active" if any pool with an active lease uses it
    active_machine_profiles: set[str] = set()
    pool_manager = getattr(request.app.state, "pool_manager", None)
    # Accumulate GPU counts per (machine_profile, bundle) from active non-default leases
    lease_gpu_counts: dict[tuple[str, str], int] = {}
    if pool_manager:
        import time as time_module

        lease_duration = getattr(pool_manager, "_lease_duration_s", 300.0)
        now = time_module.time()
        for pool in pool_manager.pools.values():
            if now - pool.status.last_renewed <= lease_duration:
                # Add all machine profiles this pool uses
                active_machine_profiles.update(pool.spec.gpus.keys())
                # Accumulate GPU counts for KEDA, excluding default pool (gpus=999)
                if pool.name != "default":
                    bundle = pool.spec.bundle or ""
                    for machine_profile, gpu_count in pool.spec.gpus.items():
                        key = (machine_profile, bundle)
                        lease_gpu_counts[key] = lease_gpu_counts.get(key, 0) + gpu_count

    # Update active lease GPU gauge
    # Reset all label combinations to 0 first, then set active ones
    # (prometheus_client doesn't auto-clear stale labels, so we clear explicitly)
    _clear_stale_lease_gpus(lease_gpu_counts)
    for (machine_profile, bundle), gpu_count in lease_gpu_counts.items():
        ACTIVE_LEASE_GPUS.labels(machine_profile=machine_profile, bundle=bundle).set(gpu_count)

    # Update worker metrics before generating output
    update_worker_metrics(registry, active_machine_profiles=active_machine_profiles)

    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )
