"""Memory management for multi-model serving.

Device-agnostic memory monitoring and LRU tracking for CUDA, MPS, and CPU.

Per DESIGN.md Section 5.4:
- Reactive LRU eviction without static VRAM budgets
- Try to load model → If OOM, evict LRU model and retry
- After each batch, check memory usage; evict if above threshold
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryStats:
    """Memory statistics for a device."""

    used_bytes: int
    total_bytes: int
    device_type: str  # "cuda", "mps", or "cpu"

    @property
    def used_gb(self) -> float:
        """Used memory in GB."""
        return self.used_bytes / (1024**3)

    @property
    def total_gb(self) -> float:
        """Total memory in GB."""
        return self.total_bytes / (1024**3)

    @property
    def usage_ratio(self) -> float:
        """Memory usage as a ratio (0.0 to 1.0)."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes

    @property
    def available_bytes(self) -> int:
        """Available memory in bytes."""
        return max(0, self.total_bytes - self.used_bytes)

    @property
    def available_gb(self) -> float:
        """Available memory in GB."""
        return self.available_bytes / (1024**3)


@dataclass
class ModelMemoryInfo:
    """Memory information for a loaded model."""

    model_name: str
    device: str
    loaded_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    estimated_bytes: int | None = None

    def touch(self) -> None:
        """Update last_used_at to current time (for LRU tracking)."""
        self.last_used_at = time.monotonic()


class DeviceMemoryTracker(ABC):
    """Abstract interface for device-specific memory tracking."""

    @abstractmethod
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics for the device."""
        ...

    @abstractmethod
    def device_type(self) -> str:
        """Return the device type string."""
        ...


class CUDAMemoryTracker(DeviceMemoryTracker):
    """Memory tracker for NVIDIA CUDA devices."""

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id

    def device_type(self) -> str:
        return "cuda"

    def get_stats(self) -> MemoryStats:
        """Get CUDA memory statistics using torch.cuda APIs.

        Uses mem_get_info() which queries NVML for actual device memory usage,
        not just PyTorch tensor allocations. This accounts for CUDA context,
        cached allocations, and provides accurate eviction decisions.
        """
        import torch

        if not torch.cuda.is_available():
            return MemoryStats(used_bytes=0, total_bytes=0, device_type="cuda")

        # mem_get_info returns (free, total) via NVML - accurate device memory
        free, total = torch.cuda.mem_get_info(self._device_id)
        used = total - free

        return MemoryStats(used_bytes=used, total_bytes=total, device_type="cuda")


class MPSMemoryTracker(DeviceMemoryTracker):
    """Memory tracker for Apple Silicon MPS devices."""

    def device_type(self) -> str:
        return "mps"

    def get_stats(self) -> MemoryStats:
        """Get MPS memory statistics using torch.mps APIs."""
        import torch

        if not torch.backends.mps.is_available():
            return MemoryStats(used_bytes=0, total_bytes=0, device_type="mps")

        # Get current allocated memory on MPS
        used = torch.mps.current_allocated_memory()

        # MPS uses unified memory - get system memory as total
        # Note: This is the full system memory, not a dedicated GPU budget
        try:
            import psutil

            total = psutil.virtual_memory().total
        except ImportError:
            # Fallback: assume 32GB if psutil not available
            total = 32 * (1024**3)

        return MemoryStats(used_bytes=used, total_bytes=total, device_type="mps")


class CPUMemoryTracker(DeviceMemoryTracker):
    """Memory tracker for CPU (system RAM) using psutil."""

    def device_type(self) -> str:
        return "cpu"

    def get_stats(self) -> MemoryStats:
        """Get CPU/system memory statistics using psutil."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            return MemoryStats(used_bytes=mem.used, total_bytes=mem.total, device_type="cpu")
        except ImportError:
            logger.warning("psutil not available, cannot track CPU memory")
            return MemoryStats(used_bytes=0, total_bytes=0, device_type="cpu")


def create_memory_tracker(device: str) -> DeviceMemoryTracker:
    """Create the appropriate memory tracker for a device string.

    Args:
        device: Device string (e.g., "cuda:0", "cuda", "mps", "cpu").

    Returns:
        A DeviceMemoryTracker for the specified device.
    """
    device_lower = device.lower()

    if device_lower.startswith("cuda"):
        # Parse device ID from "cuda:0" format
        if ":" in device_lower:
            device_id = int(device_lower.split(":")[1])
        else:
            device_id = 0
        return CUDAMemoryTracker(device_id)
    if device_lower == "mps":
        return MPSMemoryTracker()
    # Default to CPU
    return CPUMemoryTracker()


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    # Memory pressure threshold (0.0 to 1.0)
    # Evict LRU model when usage exceeds this ratio
    pressure_threshold: float = 0.85

    # Minimum free memory to maintain (in bytes)
    # Alternative to ratio-based threshold
    min_free_bytes: int | None = None

    # Whether to enable proactive eviction
    proactive_eviction: bool = True

    # Background memory monitor check interval (seconds)
    memory_check_interval_s: float = 1.0


class MemoryManager:
    """Manages memory across multiple loaded models with LRU eviction.

    The MemoryManager:
    - Tracks which models are loaded and when they were last used
    - Monitors memory usage on the current device
    - Evicts least-recently-used models when memory pressure is high

    Usage:
        manager = MemoryManager(device="cuda:0")
        manager.register_model("bge-m3")
        manager.touch("bge-m3")  # Update LRU on each request
        if manager.check_pressure():
            lru_model = manager.get_lru_model()
            # Evict lru_model...
    """

    def __init__(
        self,
        device: str = "cpu",
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize the memory manager.

        Args:
            device: Device string (e.g., "cuda:0", "mps", "cpu").
            config: Memory configuration. Uses defaults if not provided.
        """
        self._device = device
        self._config = config or MemoryConfig()
        self._tracker = create_memory_tracker(device)
        # OrderedDict maintains insertion order; we use it for LRU tracking
        # Most recently used models are moved to the end
        self._models: OrderedDict[str, ModelMemoryInfo] = OrderedDict()

    @property
    def device(self) -> str:
        """The device this manager tracks."""
        return self._device

    @property
    def device_type(self) -> str:
        """The type of device (cuda, mps, cpu)."""
        return self._tracker.device_type()

    @property
    def loaded_model_count(self) -> int:
        """Number of models currently tracked."""
        return len(self._models)

    @property
    def loaded_models(self) -> list[str]:
        """List of loaded model names in LRU order (oldest first)."""
        return list(self._models.keys())

    @property
    def pressure_threshold_pct(self) -> float:
        """Memory pressure threshold as a percentage (0-100)."""
        return self._config.pressure_threshold * 100

    @property
    def check_interval_s(self) -> float:
        """Background monitor check interval in seconds."""
        return self._config.memory_check_interval_s

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics for the device."""
        return self._tracker.get_stats()

    def register_model(self, model_name: str, estimated_bytes: int | None = None) -> None:
        """Register a newly loaded model.

        Args:
            model_name: Name of the model being loaded.
            estimated_bytes: Optional estimated memory footprint.
        """
        if model_name in self._models:
            logger.warning("Model '%s' already registered, updating", model_name)

        info = ModelMemoryInfo(
            model_name=model_name,
            device=self._device,
            estimated_bytes=estimated_bytes,
        )
        self._models[model_name] = info
        # Move to end (most recently used)
        self._models.move_to_end(model_name)
        logger.debug("Registered model '%s' in memory manager", model_name)

    def unregister_model(self, model_name: str) -> None:
        """Unregister a model when it's unloaded.

        Args:
            model_name: Name of the model being unloaded.
        """
        if model_name in self._models:
            del self._models[model_name]
            logger.debug("Unregistered model '%s' from memory manager", model_name)

    def touch(self, model_name: str) -> None:
        """Update a model's last_used_at timestamp (for LRU tracking).

        Call this when a model handles a request.

        Args:
            model_name: Name of the model being used.
        """
        if model_name in self._models:
            self._models[model_name].touch()
            # Move to end (most recently used)
            self._models.move_to_end(model_name)

    def get_model_info(self, model_name: str) -> ModelMemoryInfo | None:
        """Get memory info for a model.

        Args:
            model_name: Name of the model.

        Returns:
            ModelMemoryInfo if found, None otherwise.
        """
        return self._models.get(model_name)

    def check_pressure(self) -> bool:
        """Check if memory pressure is above threshold.

        Returns:
            True if memory usage exceeds the configured threshold.
        """
        stats = self.get_stats()

        # Check ratio-based threshold
        if stats.usage_ratio > self._config.pressure_threshold:
            logger.debug(
                "Memory pressure high: %.1f%% > %.1f%% threshold",
                stats.usage_ratio * 100,
                self._config.pressure_threshold * 100,
            )
            return True

        # Check absolute free memory threshold
        if self._config.min_free_bytes is not None and stats.available_bytes < self._config.min_free_bytes:
            logger.debug(
                "Memory pressure high: %.2f GB free < %.2f GB min",
                stats.available_gb,
                self._config.min_free_bytes / (1024**3),
            )
            return True

        return False

    def get_lru_model(self) -> str | None:
        """Get the least-recently-used model name.

        Returns:
            Name of the LRU model, or None if no models are loaded.
        """
        if not self._models:
            return None
        # First item in OrderedDict is the least recently used
        return next(iter(self._models))

    def get_eviction_candidates(self, count: int = 1) -> list[str]:
        """Get a list of models to evict (in LRU order).

        Args:
            count: Number of candidates to return.

        Returns:
            List of model names to evict (oldest first).
        """
        return list(self._models.keys())[:count]

    def should_evict_for_load(self, required_bytes: int | None = None) -> bool:
        """Check if we need to evict models before loading a new one.

        Args:
            required_bytes: Optional estimate of memory needed for new model.

        Returns:
            True if eviction is recommended before loading.
        """
        stats = self.get_stats()

        # If we're already under pressure, evict
        if self.check_pressure():
            return True

        # If we know how much memory we need, check if we have enough
        if required_bytes is not None:
            if stats.available_bytes < required_bytes:
                logger.debug(
                    "Need %.2f GB but only %.2f GB available",
                    required_bytes / (1024**3),
                    stats.available_gb,
                )
                return True

        return False

    def get_eviction_candidates_if_needed(self) -> list[str]:
        """Check memory pressure and return models that should be evicted.

        Synchronous version for callers that don't need async.

        Returns:
            List of model names that should be evicted (caller must handle eviction).
        """
        if not self._config.proactive_eviction:
            return []

        if not self.check_pressure() or not self._models:
            return []

        lru = self.get_lru_model()
        if lru is None:
            return []

        logger.info("Memory pressure detected, recommending eviction of: %s", lru)
        return [lru]

    async def check_memory_pressure(self) -> list[str]:
        """Check memory pressure and return models that should be evicted.

        Async wrapper for use in async request handlers (per DESIGN.md).
        The underlying check is fast and non-blocking.

        Returns:
            List of model names that should be evicted (caller must handle eviction).
        """
        return self.get_eviction_candidates_if_needed()
