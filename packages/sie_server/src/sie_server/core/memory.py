"""Memory management for multi-model serving.

Device-agnostic memory monitoring and LRU tracking for CUDA, MPS, and CPU.

Memory management behavior:
- Proactive and reactive LRU eviction without static VRAM budgets
- Before load, evict when current pressure or requested load headroom requires it
- Try to load model → If OOM, evict LRU model and retry
- After each batch, check memory usage; evict if above threshold
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Container
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Fake Engine (#1836/#1848): when set, every MemoryManager uses a synthetic
# tracker instead of real device queries — total is the configured budget,
# used is the sum of registered models' declared ``estimated_bytes``. This
# makes pressure/eviction deterministic for the weightless fake adapters;
# the real decision logic (``check_pressure`` / ``should_evict_for_load`` /
# LRU) runs unmodified on top. Accepts plain bytes or a KiB/MiB/GiB/TiB
# (or KB/MB/GB/TB, decimal) suffix, e.g. ``SIE_FAKE_MEMORY_BUDGET=32GiB``.
SIE_FAKE_MEMORY_BUDGET_ENV = "SIE_FAKE_MEMORY_BUDGET"

_SIZE_SUFFIXES: dict[str, int] = {
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
}


def parse_memory_budget(value: str) -> int:
    """Parse a memory-budget string into bytes.

    Accepts a plain integer byte count or a number with a binary
    (``KiB``/``MiB``/``GiB``/``TiB``) or decimal (``KB``/``MB``/``GB``/``TB``)
    suffix. Raises ``ValueError`` on anything else — a malformed budget is a
    configuration error and must fail loudly at startup, not degrade to real
    device queries.
    """
    text = value.strip()
    upper = text.upper()
    for suffix, multiplier in _SIZE_SUFFIXES.items():
        if upper.endswith(suffix):
            number = text[: -len(suffix)].strip()
            try:
                return int(float(number) * multiplier)
            except ValueError:
                break
    try:
        return int(text)
    except ValueError:
        msg = f"invalid {SIE_FAKE_MEMORY_BUDGET_ENV} value {value!r}: expected bytes or a KiB/MiB/GiB/TiB suffix"
        raise ValueError(msg) from None


@dataclass(slots=True)
class MemoryStats:
    """Memory statistics for a device."""

    used_bytes: int
    total_bytes: int
    device_type: str  # "cuda", "mps", "cpu", or "synthetic" (#1848 fake mode)

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

        # MPS uses unified memory. Prefer Metal's recommended working-set size as the
        # budget so LRU eviction can actually fire (using full system RAM as `total`
        # means the 0.95 threshold is almost never crossed on unified memory). Fall
        # back to system RAM, then a fixed 32GB, if the API is unavailable.
        total = 0
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        if callable(recommended):
            try:
                total = int(recommended())
            except (RuntimeError, ValueError):
                total = 0
        if not total:
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


class SyntheticMemoryTracker(DeviceMemoryTracker):
    """Deterministic tracker for the Fake Engine (#1848).

    Reports ``total = configured budget`` and ``used = Σ declared
    estimated_bytes of registered models`` (supplied via ``used_bytes_fn``
    so the tracker stays stateless). No device or psutil query is ever
    made, so pressure and eviction decisions become exact functions of
    what the registry loaded — reproducible on any CI runner.
    """

    def __init__(self, total_bytes: int, used_bytes_fn: Callable[[], int]) -> None:
        if total_bytes <= 0:
            msg = f"synthetic memory budget must be positive, got {total_bytes}"
            raise ValueError(msg)
        self._total_bytes = total_bytes
        self._used_bytes_fn = used_bytes_fn

    def device_type(self) -> str:
        return "synthetic"

    def get_stats(self) -> MemoryStats:
        return MemoryStats(
            used_bytes=self._used_bytes_fn(),
            total_bytes=self._total_bytes,
            device_type="synthetic",
        )


def create_memory_tracker(
    device: str,
    *,
    declared_used_bytes_fn: Callable[[], int] | None = None,
) -> DeviceMemoryTracker:
    """Create the appropriate memory tracker for a device string.

    Tracker selection lives here — including the Fake Engine synthetic mode —
    so ``MemoryManager`` has a single construction path. When
    ``SIE_FAKE_MEMORY_BUDGET`` is set AND the caller supplies
    ``declared_used_bytes_fn`` (the MemoryManager's declared-footprint sum),
    a :class:`SyntheticMemoryTracker` is returned instead of a device tracker.
    A malformed budget raises here — configuration errors must fail loudly at
    startup, not degrade to real device queries.

    Args:
        device: Device string (e.g., "cuda:0", "cuda", "mps", "cpu").
        declared_used_bytes_fn: Source of "used bytes" for the synthetic mode.

    Returns:
        A DeviceMemoryTracker for the specified device.
    """
    budget_raw = os.environ.get(SIE_FAKE_MEMORY_BUDGET_ENV)
    if budget_raw is not None and declared_used_bytes_fn is not None:
        budget = parse_memory_budget(budget_raw)
        logger.info(
            "Synthetic memory tracker active for device '%s': budget=%.2f GB (%s set)",
            device,
            budget / (1024**3),
            SIE_FAKE_MEMORY_BUDGET_ENV,
        )
        return SyntheticMemoryTracker(budget, declared_used_bytes_fn)

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
    pressure_threshold: float = 0.95

    # Minimum free memory to maintain (in bytes)
    # Alternative to ratio-based threshold
    min_free_bytes: int | None = None

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
        # OrderedDict maintains insertion order; we use it for LRU tracking
        # Most recently used models are moved to the end
        self._models: OrderedDict[str, ModelMemoryInfo] = OrderedDict()
        # Single construction path: the factory owns tracker selection,
        # including the Fake Engine synthetic mode (#1848).
        self._tracker = create_memory_tracker(device, declared_used_bytes_fn=self._declared_used_bytes)

    def _declared_used_bytes(self) -> int:
        """Sum of registered models' declared footprints (synthetic mode)."""
        return sum(info.estimated_bytes or 0 for info in self._models.values())

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

    def get_lru_model(self, *, exclude: Container[str] = frozenset()) -> str | None:
        """Get the least-recently-used model name that is not in ``exclude``.

        Args:
            exclude: Lowercased model names to skip (e.g. the pinned set).
                     Matched case-insensitively against the loaded names.

        Returns:
            Name of the LRU non-excluded model, or None if all models are
            excluded or no models are loaded.
        """
        # First item in OrderedDict is the least recently used. Loaded names
        # preserve case (HF ids), so lowercase before testing the set.
        for name in self._models:
            if name.lower() not in exclude:
                return name
        return None

    def get_idle_models(
        self,
        *,
        idle_threshold_s: float,
        now: float | None = None,
        exclude: Container[str] = frozenset(),
    ) -> list[str]:
        """Return loaded models whose ``last_used_at`` is older than the threshold.

        Pure-function over the existing ``_models`` ``OrderedDict``: no
        additional state is kept. Used by the proactive idle-eviction
        background loop in ``ModelRegistry`` to unload cold models even
        when memory pressure is below the reactive threshold.

        Args:
            idle_threshold_s: Age in seconds beyond which a model is
                considered idle. Must be non-negative; ``0`` returns every
                tracked model (intended for tests, not production).
            now: Override the reference time (for tests). Defaults to
                ``time.monotonic()``.
            exclude: Lowercased model names to skip (e.g. the pinned set).
                     Matched case-insensitively against the loaded names.

        Returns:
            Model names sorted oldest-first (longest-idle leads), excluding
            any name in ``exclude``.
        """
        if idle_threshold_s < 0:
            msg = f"idle_threshold_s must be >= 0, got {idle_threshold_s}"
            raise ValueError(msg)
        ref = time.monotonic() if now is None else now
        idle: list[tuple[float, str]] = []
        for name, info in self._models.items():
            # Loaded names preserve case (HF ids); lowercase before the test.
            if name.lower() in exclude:
                continue
            age = ref - info.last_used_at
            if age >= idle_threshold_s:
                idle.append((info.last_used_at, name))
        # Sort by last_used_at ascending → longest-idle first.
        idle.sort(key=lambda pair: pair[0])
        return [name for _, name in idle]

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
