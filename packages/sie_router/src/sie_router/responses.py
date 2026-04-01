"""Response builders using SDK types for type safety.

This module provides conversion functions from internal router types
to SDK TypedDict types. The SDK is the source of truth for interface types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sie_sdk.types import (
    AssignedWorkerInfo,
    ClusterSummary,
    HealthResponse,
    PoolListItem,
    PoolResponse,
    PoolSpecResponse,
    PoolStatusInfo,
    WorkerInfo,
)

if TYPE_CHECKING:
    from sie_router.pools import AssignedWorker, Pool
    from sie_router.types import ClusterStatus, WorkerState


def pool_to_response(pool: Pool) -> PoolResponse:
    """Convert internal Pool to SDK PoolResponse type.

    Args:
        pool: Internal Pool dataclass.

    Returns:
        SDK PoolResponse TypedDict.
    """
    return {
        "name": pool.name,
        "spec": pool_spec_to_response(pool),
        "status": pool_status_to_response(pool),
    }


def pool_spec_to_response(pool: Pool) -> PoolSpecResponse:
    """Convert pool spec to SDK PoolSpecResponse."""
    response: PoolSpecResponse = {
        "gpus": pool.spec.gpus,
        "minimum_worker_count": pool.spec.minimum_worker_count,
    }
    if pool.spec.bundle:
        response["bundle"] = pool.spec.bundle
    return response


def pool_status_to_response(pool: Pool) -> PoolStatusInfo:
    """Convert pool status to SDK PoolStatusInfo."""
    return {
        "state": pool.status.state.value,
        "assigned_workers": [assigned_worker_to_response(w) for w in pool.status.assigned_workers],
        "created_at": pool.status.created_at,
        "last_renewed": pool.status.last_renewed,
    }


def assigned_worker_to_response(worker: AssignedWorker) -> AssignedWorkerInfo:
    """Convert internal AssignedWorker to SDK AssignedWorkerInfo."""
    return {
        "name": worker.name,
        "url": worker.url,
        "gpu": worker.gpu,
    }


def pool_to_list_item(pool: Pool) -> PoolListItem:
    """Convert internal Pool to list item format.

    Args:
        pool: Internal Pool dataclass.

    Returns:
        SDK PoolListItem TypedDict.
    """
    return {
        "name": pool.name,
        "state": pool.status.state.value,
        "gpus": pool.spec.gpus,
        "worker_count": len(pool.status.assigned_workers),
    }


def worker_state_to_info(worker: WorkerState) -> WorkerInfo:
    """Convert internal WorkerState to SDK WorkerInfo.

    Args:
        worker: Internal WorkerState dataclass.

    Returns:
        SDK WorkerInfo TypedDict.
    """
    return {
        "url": worker.url,
        "gpu": worker.machine_profile,  # Display uses machine_profile
        "healthy": worker.healthy,
        "queue_depth": worker.queue_depth,
        "loaded_models": worker.models,
    }


def cluster_status_to_summary(status: ClusterStatus) -> ClusterSummary:
    """Convert internal ClusterStatus to SDK ClusterSummary.

    Args:
        status: Internal ClusterStatus dataclass.

    Returns:
        SDK ClusterSummary TypedDict.
    """
    return {
        "worker_count": status.worker_count,
        "gpu_count": status.gpu_count,
        "models_loaded": status.models_loaded,
        "total_qps": status.total_qps,
    }


def build_health_response(
    status_str: str,
    cluster_status: ClusterStatus,
    configured_gpu_types: list[str],
    live_gpu_types: list[str],
) -> HealthResponse:
    """Build SDK HealthResponse from cluster status.

    Args:
        status_str: Overall status ("healthy", "degraded", "no_workers").
        cluster_status: Internal ClusterStatus.
        configured_gpu_types: GPU types configured in the cluster.
        live_gpu_types: GPU types currently running.

    Returns:
        SDK HealthResponse TypedDict.
    """
    return {
        "status": status_str,
        "type": "router",
        "cluster": cluster_status_to_summary(cluster_status),
        "configured_gpu_types": configured_gpu_types,
        "live_gpu_types": live_gpu_types,
        "workers": cluster_status.workers,
        "models": cluster_status.models,
    }
