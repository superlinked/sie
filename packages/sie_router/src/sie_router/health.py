"""Health and status endpoints for the router."""

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from sie_router.registry import WorkerRegistry

logger = logging.getLogger(__name__)

# GPU types configured in Helm (survives scale-to-zero)
_configured_gpus_str = os.environ.get("SIE_ROUTER_CONFIGURED_GPUS", "")
CONFIGURED_GPU_TYPES: list[str] = [g.strip() for g in _configured_gpus_str.split(",") if g.strip()]

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Kubernetes liveness probe.

    Returns 200 if the router process is running.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Kubernetes readiness probe.

    Returns 200 if the router process is ready to receive traffic.
    The router is ready even with 0 workers - it can return 202 (provisioning)
    while KEDA scales up workers. This is essential for scale-to-zero.

    For worker availability, check /health instead.
    """
    registry: WorkerRegistry = request.app.state.registry
    healthy_count = len(registry.healthy_workers)

    # Router is always ready - it can accept requests and trigger scale-up
    # via pending_demand metric even when no workers exist
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "healthy_workers": healthy_count,
        },
    )


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Detailed health status including worker information.

    This endpoint returns detailed information about the cluster state.
    Used by sie-top --cluster for initial state.
    """
    registry: WorkerRegistry = request.app.state.registry
    status = registry.get_cluster_status()

    # Live GPU types from running workers
    live_gpu_types = registry.get_gpu_types()

    return {
        "status": "healthy" if status.worker_count > 0 else "degraded",
        "type": "router",  # Distinguishes from worker /health
        "cluster": {
            "worker_count": status.worker_count,
            "gpu_count": status.gpu_count,
            "models_loaded": status.models_loaded,
            "total_qps": status.total_qps,
        },
        # GPU types configured in Helm (available even when scaled to zero)
        "configured_gpu_types": CONFIGURED_GPU_TYPES,
        # GPU types currently running (subset of configured)
        "live_gpu_types": live_gpu_types,
        "workers": status.workers,
        "models": status.models,
    }


@router.websocket("/ws/cluster-status")
async def cluster_status_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time cluster status updates.

    Clients connect here to receive aggregated cluster state.
    This is the cluster-mode equivalent of worker's /ws/stats.
    Used by sie-top --cluster for real-time monitoring.
    """
    await websocket.accept()

    registry: WorkerRegistry = websocket.app.state.registry
    logger.info("Client connected to /ws/cluster-status")

    try:
        while True:
            # Get current cluster status
            status = registry.get_cluster_status()

            # Send as JSON
            await websocket.send_text(
                json.dumps(
                    {
                        "timestamp": status.timestamp,
                        "cluster": {
                            "worker_count": status.worker_count,
                            "gpu_count": status.gpu_count,
                            "models_loaded": status.models_loaded,
                            "total_qps": status.total_qps,
                        },
                        "workers": status.workers,
                        "models": status.models,
                    }
                )
            )

            # Send updates every second
            import asyncio

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/cluster-status")
    except OSError as e:
        logger.error("Error in cluster-status WebSocket: %s", e)
