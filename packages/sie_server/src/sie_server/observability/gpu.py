"""GPU metrics collection via pynvml.

Provides GPU utilization and memory metrics for the Terminal UI.
Gracefully handles missing pynvml or no GPU.

See DESIGN.md Section 5.6.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Standard GPU type identifiers for routing
GPU_TYPE_MAP: dict[str, str] = {
    # L4 variants
    "nvidia l4": "l4",
    "l4": "l4",
    # T4 variants
    "nvidia t4": "t4",
    "tesla t4": "t4",
    "t4": "t4",
    # A10G variants
    "nvidia a10g": "a10g",
    "a10g": "a10g",
    # A100 40GB variants
    "nvidia a100-sxm4-40gb": "a100-40gb",
    "nvidia a100-pcie-40gb": "a100-40gb",
    "nvidia a100 40gb": "a100-40gb",
    "a100-40gb": "a100-40gb",
    # A100 80GB variants
    "nvidia a100-sxm4-80gb": "a100-80gb",
    "nvidia a100-pcie-80gb": "a100-80gb",
    "nvidia a100 80gb": "a100-80gb",
    "a100-80gb": "a100-80gb",
    # H100 variants
    "nvidia h100": "h100",
    "nvidia h100-sxm5-80gb": "h100",
    "nvidia h100-pcie-80gb": "h100",
    "h100": "h100",
    # RTX variants (for development)
    "nvidia geforce rtx 3090": "rtx-3090",
    "nvidia geforce rtx 4090": "rtx-4090",
}


def normalize_gpu_type(gpu_name: str) -> str:
    """Normalize GPU name to standard type identifier.

    Args:
        gpu_name: Full GPU name from nvidia-smi (e.g., "NVIDIA L4", "NVIDIA A100-SXM4-80GB").

    Returns:
        Normalized GPU type (e.g., "l4", "a100-80gb") or "unknown" if not recognized.
    """
    # Check environment variable override first
    env_override = os.environ.get("SIE_GPU_TYPE")
    if env_override:
        return env_override.lower()

    # Normalize input for lookup
    name_lower = gpu_name.lower().strip()

    # Try exact match first
    if name_lower in GPU_TYPE_MAP:
        return GPU_TYPE_MAP[name_lower]

    # Try partial matches
    for pattern, gpu_type in GPU_TYPE_MAP.items():
        if pattern in name_lower:
            return gpu_type

    # Special case: detect memory size from name for A100
    if "a100" in name_lower:
        if "80g" in name_lower:
            return "a100-80gb"
        if "40g" in name_lower:
            return "a100-40gb"
        return "a100-40gb"  # Default to 40GB if unspecified

    return "unknown"


# Track initialization state
_nvml_initialized = False
_nvml_available = False


def _init_nvml() -> bool:
    """Initialize NVML if available.

    Returns:
        True if NVML is available and initialized.
    """
    global _nvml_initialized, _nvml_available

    if _nvml_initialized:
        return _nvml_available

    _nvml_initialized = True

    try:
        import pynvml
        from pynvml import NVMLError_LibraryNotFound  # ty:ignore[unresolved-import]
    except ImportError:
        logger.debug("pynvml not installed, GPU metrics unavailable")
        return False

    try:
        pynvml.nvmlInit()
        _nvml_available = True
        logger.info("NVML initialized successfully")
        return True
    except (OSError, NVMLError_LibraryNotFound):
        logger.debug("NVML initialization failed, GPU metrics unavailable")
        return False


def get_gpu_metrics() -> list[dict[str, Any]]:
    """Get metrics for all NVIDIA GPUs.

    Returns:
        List of GPU metric dicts. Empty list if no GPUs or pynvml unavailable.
    """
    if not _init_nvml():
        return []

    try:
        import pynvml

        gpus = []
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Get device name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Get utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            # Get memory info
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpus.append(
                {
                    "device": f"cuda:{i}",
                    "name": name,
                    "gpu_type": normalize_gpu_type(name),
                    "utilization_pct": utilization.gpu,
                    "memory_used_bytes": memory.used,
                    "memory_total_bytes": memory.total,
                }
            )

        return gpus

    except OSError:
        logger.exception("Error collecting GPU metrics")
        return []


def get_worker_gpu_type() -> str | None:
    """Get the GPU type for this worker.

    Returns the normalized GPU type (e.g., "l4", "a100-80gb") or None if no GPU.
    Checks SIE_GPU_TYPE environment variable first, then auto-detects from NVML.
    """
    # Check environment variable override first
    env_override = os.environ.get("SIE_GPU_TYPE")
    if env_override:
        return env_override.lower()

    # Get from first GPU
    gpus = get_gpu_metrics()
    if gpus:
        return gpus[0]["gpu_type"]

    return None


def shutdown_nvml() -> None:
    """Shutdown NVML. Called at server shutdown."""
    global _nvml_initialized, _nvml_available

    if _nvml_available:
        try:
            import pynvml

            pynvml.nvmlShutdown()
            logger.debug("NVML shutdown")
        except (ImportError, OSError):
            pass

    _nvml_initialized = False
    _nvml_available = False
