from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
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

from sie_server.core.batcher import BatchConfig
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
        failed = registry.is_failed(name)

        # Determine state: loading/unloading take precedence over loaded.
        # ``failed`` ranks below ``loaded`` (a recovered failure that has
        # since loaded successfully should report ``loaded``) but above
        # ``available`` so the diagnostic surface is preserved.
        state: ModelState
        if loading:
            state = "loading"
        elif unloading:
            state = "unloading"
        elif loaded:
            state = "loaded"
        elif failed:
            state = "failed"
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

                    # Adaptive batching state (via snapshot API)
                    adaptive_state = loaded_model.worker.get_adaptive_state()
                    if adaptive_state is not None:
                        model_info["adaptive_batching"] = {
                            "calibrated": adaptive_state.calibrated,
                            "target_p50_ms": adaptive_state.target_p50_ms,
                            "wait_ms": adaptive_state.current_wait_ms,
                            "batch_cost": adaptive_state.current_batch_cost,
                            "p50_ms": adaptive_state.observed_p50_ms,
                            "headroom_ms": adaptive_state.headroom_ms,
                            "fill_ratio": adaptive_state.fill_ratio,
                        }

        models.append(model_info)

    # Sort by memory usage (highest first) like `top`
    models.sort(key=lambda m: m.get("memory_bytes", 0), reverse=True)
    return models


def _compute_bundle_config_hash(registry: ModelRegistry, bundle_id: str) -> str:
    """Compute SHA-256 hash of model configs assigned to this worker's bundle.

    The hash covers serialized model configs (sie_id + profiles) for models
    routable to the given bundle. Bundle metadata is excluded (immutable at
    runtime).

    Args:
        registry: The model registry.
        bundle_id: The bundle identifier to scope configs to.

    Returns:
        Hex-encoded SHA-256 hash string, or empty string if no configs.
    """
    configs = registry.get_configs_snapshot(bundle_id)
    if not configs:
        return ""

    # Deterministic serialization matching gateway's compute_bundle_config_hash:
    # both sides hash [{"sie_id": name, "profiles": [{name, config}]}]
    # where config contains routable fields (adapter_path, max_batch_tokens, etc.).
    _hash_fields = ("adapter_path", "max_batch_tokens", "compute_precision", "adapter_options")
    items = []
    for config in sorted(configs.values(), key=lambda c: c.sie_id):
        profiles_for_hash = []
        for pname in sorted(config.profiles.keys()):
            profile = config.profiles[pname]
            # Normalize adapter_options to a plain dict or None.
            # Pydantic models serialize default AdapterOptions() as
            # {"loadtime": {}, "runtime": {}} which would mismatch
            # the gateway's raw dict (None when not set). Treat a
            # default/empty AdapterOptions the same as None.
            adapter_opts_raw = None
            if profile.adapter_options:
                dumped = profile.adapter_options.model_dump(mode="json")
                # Treat all-empty-sub-dicts as None (matches gateway's raw None)
                if any(v for v in dumped.values()):
                    adapter_opts_raw = dumped
            profile_dict = {
                "adapter_path": profile.adapter_path,
                "max_batch_tokens": profile.max_batch_tokens,
                "compute_precision": profile.compute_precision,
                "adapter_options": adapter_opts_raw,
            }
            profiles_for_hash.append({"name": pname, "config": profile_dict})
        items.append({"sie_id": config.sie_id, "profiles": profiles_for_hash})

    serialized = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


# Cache of bundle config hashes. Populated by _compute_bundle_config_hash
# and invalidated when the registry is mutated.
_bundle_config_hash_cache: dict[str, tuple[int, str]] = {}


def compute_bundle_config_hash_cached(registry: ModelRegistry, bundle_id: str) -> str:
    """Return cached bundle config hash, recomputing only when configs change.

    Uses the registry's config version (mutation counter) to detect staleness.
    """
    version = getattr(registry, "_config_version", 0)
    cached = _bundle_config_hash_cache.get(bundle_id)
    if cached is not None and cached[0] == version:
        return cached[1]
    result = _compute_bundle_config_hash(registry, bundle_id)
    _bundle_config_hash_cache[bundle_id] = (version, result)
    return result


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

    # Compute bundle_config_hash from loaded model configs
    bundle_config_hash = compute_bundle_config_hash_cached(registry, bundle)

    # Machine profile: env var if set, otherwise detected GPU type (for standalone workers)
    # - In K8s: SIE_MACHINE_PROFILE is set via downward API (e.g., "l4-spot")
    # - Standalone: No env var, so use detected GPU type (e.g., "l4") for direct SDK routing
    machine_profile = os.environ.get("SIE_MACHINE_PROFILE") or gpu_type or ""
    pool_name = os.environ.get("SIE_POOL", "")

    # Worker name: use hostname or pod name if available
    worker_name = os.environ.get("HOSTNAME", os.environ.get("POD_NAME", ""))

    # Loaded models: list of model names with state="loaded"
    loaded_models = [m["name"] for m in model_status if m["state"] == "loaded"]

    # Compute aggregate max_batch_requests across loaded models.
    # The gateway uses this for fill-first scoring to know worker batch capacity.
    # Use the minimum across loaded models (conservative: GPU batch is model-specific).
    # Snapshot _loaded to avoid RuntimeError from concurrent mutation during iteration.
    loaded_snapshot = list(registry._loaded.values())
    loaded_model_workers = [lm.worker for lm in loaded_snapshot if lm.worker is not None]
    if loaded_model_workers:
        max_batch_requests = min(w._batch_config.max_batch_requests for w in loaded_model_workers)
    else:
        max_batch_requests = BatchConfig().max_batch_requests

    return WorkerStatusMessage(
        timestamp=time.time(),
        ready=is_ready(),
        name=worker_name,
        # Gateway-friendly fields
        machine_profile=machine_profile,
        pool_name=pool_name,
        gpu_count=gpu_count,
        bundle=bundle,
        bundle_config_hash=bundle_config_hash,
        loaded_models=loaded_models,
        max_batch_requests=max_batch_requests,
        # Detailed fields (for TUI, gateway model selection, debugging)
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
