from __future__ import annotations

import asyncio
import getpass
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sie_sdk.types import (
    GPUMetrics,
    ModelConfig,
    ModelState,
    ModelStatus,
    ServerInfo,
    WorkerStatusMessage,
)

from sie_server.core.readiness import is_ready
from sie_server.observability.gpu import get_gpu_metrics
from sie_server.observability.prometheus import collect_prometheus_metrics

if TYPE_CHECKING:
    from sie_server.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Server start time for uptime calculation
_server_start_time: float | None = None


def init_server_start_time() -> None:
    """Initialize server start time. Called once at startup."""
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()


def get_server_info() -> ServerInfo:
    """Get server metadata.

    Returns:
        ServerInfo with version, uptime, user, working_dir, pid.
    """
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()

    return ServerInfo(
        version="0.1.0",
        uptime_seconds=int(time.time() - _server_start_time),
        user=getpass.getuser(),
        working_dir=str(Path.cwd()),
        pid=os.getpid(),
    )


def get_model_status(registry: ModelRegistry) -> list[ModelStatus]:
    """Get status for all models.

    Args:
        registry: The model registry.

    Returns:
        List of ModelStatus dicts.
    """
    models: list[ModelStatus] = []
    for name in registry.model_names:
        config = registry.get_config(name)
        loaded = registry.is_loaded(name)
        loading = registry.is_loading(name)
        unloading = registry.is_unloading(name)

        # Determine state: loading/unloading take precedence
        state: ModelState
        if loading:
            state = "loading"
        elif unloading:
            state = "unloading"
        elif loaded:
            state = "loaded"
        else:
            state = "available"

        inputs_list = config.inputs.to_list()
        adapter_path = config.resolve_profile("default").adapter_path

        # Base model info
        model_info: ModelStatus = {
            "name": name,
            "state": state,
            "device": None,
            "memory_bytes": 0,
            "config": ModelConfig(
                hf_id=config.hf_id,
                adapter=adapter_path,
                inputs=inputs_list,
                outputs=config.outputs,
                dims=config.dims,
                max_sequence_length=config.max_sequence_length,
            ),
            "queue_depth": 0,
            "queue_pending_items": 0,
        }

        if loaded:
            # Get loaded model details
            loaded_model = registry._loaded.get(name)
            if loaded_model:
                model_info["device"] = loaded_model.device
                model_info["memory_bytes"] = loaded_model.memory_bytes

                # Get queue info from worker
                if loaded_model.worker:
                    model_info["queue_pending_items"] = loaded_model.worker.pending_count
                    # queue_depth is the same as pending_count for our design
                    model_info["queue_depth"] = loaded_model.worker.pending_count

        models.append(model_info)

    # Sort by memory usage (highest first) like `top`
    models.sort(key=lambda m: m.get("memory_bytes", 0), reverse=True)
    return models


async def build_status_message(registry: ModelRegistry) -> WorkerStatusMessage:
    """Build the complete status message.

    Args:
        registry: The model registry.

    Returns:
        WorkerStatusMessage ready for JSON serialization.

    The status message includes:
    - machine_profile: For routing (SIE_MACHINE_PROFILE env var or detected GPU type)
    - gpu_count: Number of GPUs on this worker
    - loaded_models: List of model names currently loaded
    - models: Detailed per-model status including queue_depth
    - gpus: Detailed GPU metrics (includes gpu_type per GPU)
    """
    # Collect all data
    server_info = get_server_info()
    gpu_metrics_raw = get_gpu_metrics()
    model_status = get_model_status(registry)
    prometheus_data = collect_prometheus_metrics()

    # Add memory threshold to GPU metrics for TUI display
    memory_threshold_pct = registry.memory_manager.pressure_threshold_pct
    gpu_metrics: list[GPUMetrics] = []
    for gpu in gpu_metrics_raw:
        gpu_metrics.append(
            GPUMetrics(
                device=gpu["device"],
                name=gpu["name"],
                gpu_type=gpu["gpu_type"],
                utilization_pct=gpu["utilization_pct"],
                memory_used_bytes=gpu["memory_used_bytes"],
                memory_total_bytes=gpu["memory_total_bytes"],
                memory_threshold_pct=memory_threshold_pct,
            )
        )

    # GPU type: use first GPU's type (most common case is single-GPU worker)
    gpu_type = gpu_metrics[0]["gpu_type"] if gpu_metrics else None
    gpu_count = len(gpu_metrics) if gpu_metrics else 0

    # Bundle: from environment variable (set by CLI --bundle flag)
    bundle = os.environ.get("SIE_BUNDLE", "default")

    # Machine profile: env var if set, otherwise detected GPU type (for standalone workers)
    # - In K8s: SIE_MACHINE_PROFILE is set via downward API (e.g., "l4-spot")
    # - Standalone: No env var, so use detected GPU type (e.g., "l4") for direct SDK routing
    machine_profile = os.environ.get("SIE_MACHINE_PROFILE") or gpu_type or ""

    # Worker name: use hostname or pod name if available
    worker_name = os.environ.get("HOSTNAME", os.environ.get("POD_NAME", ""))

    # Loaded models: list of model names with state="loaded"
    loaded_models = [m["name"] for m in model_status if m["state"] == "loaded"]

    return WorkerStatusMessage(
        timestamp=time.time(),
        ready=is_ready(),
        name=worker_name,
        # Router-friendly fields
        # machine_profile: used for routing (env var or detected GPU type for standalone)
        machine_profile=machine_profile,
        gpu_count=gpu_count,
        bundle=bundle,
        loaded_models=loaded_models,
        # Detailed fields (for TUI, router model selection, debugging)
        # Note: queue_depth is per-model in models array, not aggregated
        server=server_info,
        gpus=gpu_metrics,  # Individual GPU info still available here
        models=model_status,
        counters=prometheus_data.get("counters", {}),
        histograms=prometheus_data.get("histograms", {}),
    )


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time server status.

    Pushes status updates every 200ms to connected clients.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    # Get registry from app state
    registry: ModelRegistry = websocket.app.state.registry

    try:
        while True:
            # Build and send status
            status = await build_status_message(registry)
            await websocket.send_json(status)

            # Wait 200ms before next update
            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
