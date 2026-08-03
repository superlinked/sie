"""Model registry for managing loaded models.

The registry handles:
- Tracking which models are loaded
- Async-safe load/unload orchestration with locks
- Memory management with LRU eviction (proactive + reactive)
- On-demand config discovery
- Hot reload of model configurations (when enabled)

Model loading workflow is delegated to ModelLoader.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sie_sdk.storage import is_cloud_path

from sie_server.adapters.base import ModelAdapter
from sie_server.config.engine import EngineConfig
from sie_server.config.model import ModelConfig
from sie_server.core.disk_cache import DiskCacheConfig, ModelDiskCacheManager
from sie_server.core.hot_reload import HotReloader
from sie_server.core.load_errors import (
    LoadErrorClass,
    LoadFailure,
    ModelLoadTimeoutError,
    classify_load_error,
)
from sie_server.core.loader import expand_profile_variants, load_model_configs
from sie_server.core.memory import MemoryConfig, MemoryManager
from sie_server.core.model_loader import DEFAULT_MAX_LORAS, LoadedModel, ModelLoader
from sie_server.core.oom import is_oom_error
from sie_server.core.pool_isolation import (
    validate_no_legacy_scalar_lora_id,
    validate_pool_isolation,
)
from sie_server.core.postprocessor_registry import PostprocessorRegistry
from sie_server.core.preprocessor_registry import PreprocessorRegistry
from sie_server.core.residency import EvictionResult, select_eviction_candidate
from sie_server.core.worker import ModelWorker
from sie_server.core.worker.types import AdaptiveBatchingParams
from sie_server.observability.worker_telemetry import refresh_worker_metric_context, worker_telemetry

logger = logging.getLogger(__name__)

# Error messages
_ERR_MODEL_NOT_FOUND = "Model '{name}' not found in registry"
_ERR_MODEL_NOT_LOADED = "Model '{name}' is not loaded"
_ERR_MODEL_ALREADY_LOADED = "Model '{name}' is already loaded"


def _base_model_id(raw: str) -> str:
    """Bare ``sie_id`` with any ``:profile`` suffix removed, lowercased.

    HuggingFace org/model ids contain ``/`` but never ``:``, so the part before
    the first ``:`` is the base model and the rest is a profile name. Used where
    the base model is what matters (e.g. disk weights, which a profile variant
    shares with its base).
    """
    return raw.split(":", maxsplit=1)[0].strip().lower()


def _canonical_pinned_id(raw: str) -> str:
    """Canonicalise a pinned id, preserving non-default profile variants.

    The worker keys ``_loaded`` / configs / memory by the FULL profile-qualified
    id for non-default profiles (``loader._expand_profile_variants`` materialises
    each as its own ``{base}:{profile}`` config), so the pinned set must keep
    that granularity to load and protect the right unit. Mirrors the gateway's
    canonicalization (``pools.rs``): the literal ``:default`` folds to the bare
    base; any other ``:profile`` is preserved. Everything is lowercased for
    case-insensitive matching against the loaded keys.

    Splits on the first ``:`` (HF ids contain ``/`` but never ``:``, and profile
    names are single-segment), so a single-colon id is the only valid shape.
    """
    base, sep, profile = raw.strip().lower().partition(":")
    base = base.strip()
    profile = profile.strip()
    if not sep or not profile or profile == "default":
        return base
    return f"{base}:{profile}"


def _normalize_pinned(pinned: Iterable[str] | None) -> frozenset[str]:
    """Normalise a raw pinned-model list into a lowercase canonical-id frozenset.

    Profile-qualified ids are preserved (only ``:default`` folds to the base).
    """
    if not pinned:
        return frozenset()
    return frozenset(canonical for raw in pinned if (canonical := _canonical_pinned_id(raw)))


def _model_config_semantic_hash(config: ModelConfig) -> str:
    """Stable hash for config equality that ignores Pydantic private attrs."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _model_configs_semantically_equal(left: ModelConfig | None, right: ModelConfig) -> bool:
    return left is not None and _model_config_semantic_hash(left) == _model_config_semantic_hash(right)


def _is_python_runtime_adapter(adapter_path: str) -> bool:
    return not adapter_path.startswith("sie_server_rust.")


def _adapter_load_required_bytes(adapter: ModelAdapter, memory_manager: MemoryManager) -> int | None:
    """Return an adapter-provided load-headroom estimate, if any."""
    stats = memory_manager.get_stats()
    if stats.total_bytes <= 0:
        return None

    try:
        required = adapter.load_required_memory_bytes(
            device_type=stats.device_type,
            device_total_bytes=stats.total_bytes,
        )
    except Exception:
        logger.exception("adapter.load_required_memory_bytes() raised; falling back to pressure-only eviction")
        return None
    if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
        return None
    return min(required, stats.total_bytes)


def _normalize_devices(device: str, devices: list[str] | None) -> list[str]:
    """Return a non-empty, de-duplicated device list preserving order."""
    raw_devices = devices or [device]
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_devices:
        item = raw.strip()
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return normalized or [device]


def _device_family(device: str) -> str:
    """Return the root device family (``cuda`` for ``cuda:0``)."""
    return device.split(":", 1)[0].lower()


class ModelRegistry:
    """Registry for managing model configs and loaded models.

    The registry maintains:
    - All discovered model configs from the models/ directory (local or cloud)
    - Currently loaded model adapters and their devices
    - Paths to model directories for adapter loading

    Usage:
        registry = ModelRegistry(models_dir=Path("./models"))
        await registry.load_async("bge-m3", device="cuda:0")
        adapter = registry.get("bge-m3")
        results = adapter.encode(items, output_types=["dense"])
        await registry.unload_async("bge-m3")
    """

    def __init__(
        self,
        models_dir: Path | str | None = None,
        memory_config: MemoryConfig | None = None,
        drain_timeout_s: float = 30.0,
        model_filter: list[str] | None = None,
        device: str = "cpu",
        devices: list[str] | None = None,
        engine_config: EngineConfig | None = None,
        enable_hot_reload: bool = True,
        pool_name: str | None = None,
        pinned_models: Iterable[str] | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            models_dir: Path to models directory (local path, s3://, gs://, abfs://, or abfss://).
                       If None, registry starts empty and configs must be added manually.
            memory_config: Configuration for memory management. If None, uses defaults.
            drain_timeout_s: Timeout in seconds to wait for worker drain before unload.
            model_filter: Optional list of model names to include. If None, all models are
                available; an EMPTY list advertises zero models (a bundle whose adapters
                match no onboarded model must not fall back to the full catalog).
            device: Default device for memory tracking (e.g., "cuda:0", "mps", "cpu").
            devices: Optional ordered list of devices available to this worker.
                When set, lazy model loads place whole models across these
                devices; adapters still receive a single concrete device.
            enable_hot_reload: Whether to enable hot reload of model configs when models_dir is set.
            pool_name: The worker's pool identity (``SIE_POOL``).
                Used to gate the pool-isolation validator that rejects mixing
                generation and non-generation models on the same worker. When
                ``None`` the validator is skipped (e.g. unit tests without a
                pool context).
            pinned_models: Optional list of model ids to pin in memory. Pinned models are
                eager-loaded and never chosen as LRU/idle/OOM eviction victims. Ids may be
                bare (``org/name``) or profile-qualified (``org/name:profile``); non-default
                profiles are first-class variants and are preserved, while ``:default`` folds
                to the base. Matching is case-insensitive against the full canonical id.
        """
        # Store as string to handle cloud URLs
        self._models_dir: str | None = str(models_dir) if models_dir is not None else None
        # `is not None`, not truthiness: an empty filter (zero-match bundle)
        # must stay an empty set — collapsing it to None would advertise the
        # ENTIRE catalog to the gateway from a worker that can serve none of it.
        self._model_filter = set(model_filter) if model_filter is not None else None
        self._device = device
        self._devices = _normalize_devices(device, devices)
        self._device_order = {name: i for i, name in enumerate(self._devices)}
        self._enable_hot_reload = enable_hot_reload
        self._pool_name = pool_name.strip().lower() if pool_name else None
        # Normalised pinned set: bare lowercase sie_ids, never chosen for eviction.
        self._pinned_models: frozenset[str] = _normalize_pinned(pinned_models)
        self._configs: dict[str, ModelConfig] = {}
        self._model_dirs: dict[str, Path] = {}
        self._loaded: dict[str, LoadedModel] = {}
        preprocessor_workers = engine_config.preprocessor_workers if engine_config else None
        self._preprocessor_registry = PreprocessorRegistry(max_workers=preprocessor_workers)
        # Share CPU pool between preprocessor and postprocessor registries
        self._postprocessor_registry = PostprocessorRegistry(self._preprocessor_registry._executor)
        self._memory_config = memory_config or MemoryConfig()
        self._memory_managers = {
            device_name: MemoryManager(device=device_name, config=self._memory_config) for device_name in self._devices
        }
        # Default manager for callers that operate on the registry's primary
        # device. Multi-device code paths use ``_memory_manager_for_device``
        # with the loaded model's concrete device.
        self._memory_manager = self._memory_managers[self._devices[0]]
        self._drain_timeout_s = drain_timeout_s

        # Disk cache management
        self._disk_cache_manager: ModelDiskCacheManager | None = None
        if engine_config is None or engine_config.disk_cache_enabled:
            from sie_sdk.cache import get_cache_config

            cache_config = get_cache_config()
            threshold = engine_config.disk_pressure_threshold_percent / 100.0 if engine_config else 0.85
            self._disk_cache_manager = ModelDiskCacheManager(
                DiskCacheConfig(
                    cache_dir=cache_config.local_cache,
                    pressure_threshold=threshold,
                ),
                pinned_provider=self._pinned_disk_repo_ids,
            )
            logger.info(
                "Disk cache manager enabled (threshold=%.0f%%, cache_dir=%s)",
                threshold * 100,
                cache_config.local_cache,
            )

        # Concurrency-safe loading
        self._load_lock: asyncio.Lock | None = None  # Created lazily on first use
        self._config_update_lock: asyncio.Lock | None = None
        self._loading: set[str] = set()  # Models currently being loaded
        self._unloading: set[str] = set()  # Models currently being unloaded
        # Terminal-failed state. Populated by ``_load_model_background`` when a
        # load raises; surfaces non-retryable ``MODEL_LOAD_FAILED`` errors via
        # the API and short-circuits hot retry loops (see ``start_load_async``).
        # Cleared on successful load, explicit ``clear_failure``, or hot-reload
        # of the model config.
        self._failed: dict[str, LoadFailure] = {}

        # Background memory monitor
        self._monitor_task: asyncio.Task[None] | None = None
        self._monitor_running = False

        # Background idle-evictor (proactive cold-model unload). None unless
        # ``engine_config.idle_evict_s`` is set.
        self._idle_evict_task: asyncio.Task[None] | None = None
        self._idle_evict_running = False
        self._engine_config = engine_config

        # Background fire-and-forget tasks (prevents GC collection)
        self._background_tasks: set[asyncio.Task[None]] = set()

        # Hot reloader (created lazily when started)
        self._hot_reloader: HotReloader | None = None

        # Monotonic config mutation counter used by bundle_config_hash cache
        # in ws.py to detect when configs change and the hash must be recomputed.
        self._config_version: int = 0

        if self._models_dir is not None:
            self._load_configs_from_dir(self._models_dir)
            logger.info("Initializing model registry from %s", self._models_dir)
            logger.info("Found %d models", len(self._configs))
        else:
            logger.info("No models directory specified, starting with empty registry")

        # Model loader - handles the actual loading workflow
        # Convert engine adaptive_batching config to worker params
        adaptive_params = None
        if engine_config and engine_config.adaptive_batching.enabled:
            ab = engine_config.adaptive_batching
            adaptive_params = AdaptiveBatchingParams(
                enabled=True,
                target_p50_ms=ab.target_p50_ms,
                calibration_multiplier=ab.calibration_multiplier,
                min_target_p50_ms=ab.min_target_p50_ms,
                max_target_p50_ms=ab.max_target_p50_ms,
                min_wait_ms=ab.min_wait_ms,
                max_wait_ms=ab.max_wait_ms,
                gain=ab.gain,
                integral_gain=ab.integral_gain,
                window_size=ab.window_size,
                update_interval=ab.update_interval,
                # Starvation detector lives on the runtime params struct so
                # it is configurable per-deployment without touching code.
                # Engine-level defaults (see AdaptiveBatchingConfig) are
                # tuned for production; tests override via direct
                # AdaptiveBatchingParams construction.
                starvation_recovery_enabled=ab.starvation_recovery_enabled,
                starvation_window=ab.starvation_window,
                starvation_batch_size=ab.starvation_batch_size,
            )

        # Build OOM recovery config from the engine config (if present).
        # The registry passes itself as the RegistryCallbacks impl: the
        # ``evict_lru_excluding`` method satisfies the protocol structurally.
        oom_recovery_config = engine_config.oom_recovery.to_runtime() if engine_config is not None else None

        self._loader = ModelLoader(
            preprocessor_registry=self._preprocessor_registry,
            postprocessor_registry=self._postprocessor_registry,
            all_configs=self._configs,
            default_compute_precision=engine_config.default_compute_precision if engine_config else "float16",
            attention_backend=engine_config.attention_backend if engine_config else "auto",
            max_batch_requests=engine_config.max_batch_requests if engine_config else None,
            max_batch_wait_ms=engine_config.max_batch_wait_ms if engine_config else None,
            coalesce_ms=engine_config.coalesce_ms if engine_config else None,
            coalesce_ratio=engine_config.coalesce_ratio if engine_config else None,
            max_queue_size=engine_config.max_concurrent_requests if engine_config else None,
            instrumentation=engine_config.instrumentation if engine_config else False,
            max_loras_per_model=engine_config.max_loras_per_model if engine_config else DEFAULT_MAX_LORAS,
            disk_cache_manager=self._disk_cache_manager,
            adaptive_batching=adaptive_params,
            oom_recovery=oom_recovery_config,
            registry_callbacks=self,
        )

    def _is_pinned(self, name: str) -> bool:
        """Return True if ``name`` is in the pinned set (case-insensitive)."""
        return name.strip().lower() in self._pinned_models

    def _pinned_disk_repo_ids(self) -> set[str]:
        """HF repo ids of pinned models, for disk-cache eviction protection.

        Matched at BASE-model granularity: a profile variant shares its base
        model's ``hf_id`` (disk weights), so a pinned ``org/model:fp8`` protects
        the same repo as ``org/model``. Mapping pinned ids to the configs'
        ``hf_id`` keeps a long-tail download from evicting a pinned model's
        weights and reintroducing its cold start.

        The disk cache matches by exact ``hf_id`` (no case-folding, unlike the
        in-memory sie_id path), which is safe because the cache key also derives
        from ``config.hf_id`` (model_loader passes it to ``ensure_space_before_download``
        and ``touch``).
        """
        if not self._pinned_models:
            return set()
        pinned_bases = {_base_model_id(pinned_id) for pinned_id in self._pinned_models}
        return {
            config.hf_id
            for sie_id, config in self._configs.items()
            if config.hf_id and _base_model_id(sie_id) in pinned_bases
        }

    async def _eager_load_pinned(self) -> None:
        """Eager-load every pinned model that has a config and is not yet resident.

        Idempotent and best-effort (``start_load_async`` skips loaded/loading and
        failed-in-cooldown models). Called whenever the pinned set or the config
        set changes, so a pin recorded before its config existed (default Helm
        workers receive configs at runtime via the sidecar) becomes resident once
        the config is known, and a pinned model unloaded by a config replacement
        is reloaded.
        """
        for pinned_id in sorted(self._pinned_models):
            name = self.resolve_model_id(pinned_id)
            if name is None:
                logger.debug("Pinned model '%s' has no local config yet; skipping eager-load", pinned_id)
                continue
            if self.is_loaded(name):
                continue
            try:
                await self.start_load_async(name, self._device)
            except Exception:
                logger.exception("Failed to start eager-load of pinned model '%s'", name)

    def _maybe_schedule_pinned_load(self, name: str) -> None:
        """Schedule a background pinned-load reconcile if ``name`` is pinned (sync callers).

        Used by the synchronous ``add_config`` to bridge into the async loader
        when a pinned model's config arrives at runtime, mirroring the
        ``start_load_async`` / ``_load_lora`` create-task pattern. Schedules the
        guarded ``_eager_load_pinned`` (so a load-validation error can't surface
        as an unretrieved task exception). A no-op when ``name`` is not pinned or
        when there is no running loop (e.g. sync tests / CLI), where the async
        reconcile paths cover it.
        """
        if not self._is_pinned(name):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._eager_load_pinned())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def set_pinned_models(self, raw_ids: Iterable[str]) -> frozenset[str]:
        """Replace the pinned set at runtime from the gateway's authoritative pool spec.

        ``raw_ids`` are gateway-canonical (bare ``sie_id`` or profile-qualified
        ``sie_id:profile``); they are normalized to lowercased canonical ids
        (profile variants preserved, ``:default`` folded to base). The new set
        fully REPLACES the current one (an empty set un-pins everything), so
        removing a model from the pool API actually un-pins it on the worker.

        On change: pinned models with a local config are eager-loaded in the
        background (non-blocking, best-effort) so the first request pays no cold
        start; un-pinned models are simply demoted to LRU/idle-evictable again
        (no forced unload). The frozenset is reassigned atomically, so every
        eviction path and the disk-cache provider see the new protection
        immediately. Idempotent: an unchanged set is a no-op.

        Returns the normalized pinned set now in effect.
        """
        new_pinned = _normalize_pinned(raw_ids)
        if new_pinned == self._pinned_models:
            return new_pinned

        newly_pinned = new_pinned - self._pinned_models
        # Reassign before loading so newly-pinned models are eviction-protected
        # the instant they are pinned, even before their weights are resident.
        self._pinned_models = new_pinned
        logger.info(
            "Pinned set updated: %d model(s) pinned (%d newly pinned)",
            len(new_pinned),
            len(newly_pinned),
        )
        await self._eager_load_pinned()
        return new_pinned

    def _load_configs_from_dir(self, models_dir: str) -> None:
        """Load all model configs from a directory (local or cloud)."""
        from sie_sdk.storage import is_cloud_path

        all_configs = load_model_configs(models_dir)

        # Apply model filter if specified
        if self._model_filter is not None:
            self._configs = {name: config for name, config in all_configs.items() if name in self._model_filter}
            logger.info(
                "Model filter applied: %d/%d models available",
                len(self._configs),
                len(all_configs),
            )
        else:
            self._configs = all_configs

        before_count = len(self._configs)
        self._configs = {name: config for name, config in self._configs.items() if self.accepts_config_pool(config)}
        filtered_count = before_count - len(self._configs)
        if filtered_count:
            logger.info(
                "Model config filter applied for SIE_POOL=%s: %d/%d models available",
                self._pool_name or "<unset>",
                len(self._configs),
                before_count,
            )

        # Pool isolation. With the post-filter set in hand,
        # reject mixed generation/non-generation pools loudly. The
        # check is best-effort when ``pool_name`` is None (tests).
        if self._pool_name is not None:
            self._validate_pool_isolation_of_loaded()

        # Legacy scalar ``lora_id`` exclusion fires regardless of pool_name —
        # it's a hard invariant, not a pool-fairness concern. Multi-LoRA
        # generation via ``adapter_options.loadtime.lora_paths`` is shipped
        # and is not affected by this check.
        for name, config in self._configs.items():
            validate_no_legacy_scalar_lora_id(name=name, config=config)

        self._config_version += 1

        # Track model directories for adapter loading (local only)
        # For cloud models, we use the cached config directory
        if not is_cloud_path(models_dir):
            self._populate_local_model_dirs(Path(models_dir))
        else:
            self._populate_cloud_model_dirs()

    def _populate_local_model_dirs(self, models_path: Path) -> None:
        """Populate model directories from a local models/ path.

        With flat YAML configs, all configs live directly in models_path.
        We set the model_dir to models_path itself for all models.
        """
        # With flat YAML structure, all model configs are in models_path directly
        for name in self._configs:
            self._model_dirs[name] = models_path

    def _populate_cloud_model_dirs(self) -> None:
        """Populate model directories from cached configs for cloud models.

        With flat YAML structure, cached configs are stored directly in the cache dir.
        All models share the same model_dir (the cache directory).
        """
        from sie_server.core.loader import _get_config_cache_dir

        cache_dir = _get_config_cache_dir()

        # With flat YAML structure, all model configs are in the cache dir directly
        for name in self._configs:
            self._model_dirs[name] = cache_dir

    def rescan_configs(self) -> list[str]:
        """Rescan models_dir for new model configs.

        Used for on-demand discovery when an unknown model is requested.

        Returns:
            List of newly discovered model names.
        """
        if self._models_dir is None:
            return []

        old_names = set(self._configs.keys())
        self._load_configs_from_dir(self._models_dir)
        new_names = set(self._configs.keys()) - old_names

        # A rescan replaces the whole config map, so it can modify or delete
        # entries even when it discovers no new model names. Keep both the
        # loader and the bounded telemetry catalog on that authoritative map.
        self._loader.update_configs(self._configs)
        self._refresh_worker_metric_catalog()

        if new_names:
            logger.info("Discovered %d new models: %s", len(new_names), ", ".join(new_names))

        return list(new_names)

    @property
    def model_names(self) -> list[str]:
        """Return list of all known model names."""
        return list(self._configs.keys())

    @property
    def loaded_model_names(self) -> list[str]:
        """Return list of currently loaded model names."""
        return list(self._loaded.keys())

    @property
    def loaded_model_memory_bytes(self) -> dict[str, int]:
        """Snapshot resident model memory for post-snapshot metric replay."""
        return {name: loaded.memory_bytes for name, loaded in self._loaded.items()}

    @property
    def preprocessor_registry(self) -> PreprocessorRegistry:
        """Return the preprocessor registry for parallel image preprocessing."""
        return self._preprocessor_registry

    @property
    def postprocessor_registry(self) -> PostprocessorRegistry:
        """Return the postprocessor registry for output transforms."""
        return self._postprocessor_registry

    @property
    def memory_manager(self) -> MemoryManager:
        """Return the memory manager for device memory tracking."""
        active_managers = [manager for manager in self._memory_managers.values() if manager.loaded_model_count > 0]
        if len(active_managers) == 1:
            return active_managers[0]
        return self._memory_manager

    @property
    def memory_managers(self) -> dict[str, MemoryManager]:
        """Return memory managers keyed by concrete device."""
        return dict(self._memory_managers)

    @property
    def engine_config(self) -> EngineConfig | None:
        """Return the engine config (or None if running without one).

        Read-only accessor used by API route handlers that need to thread
        config-derived values into per-request helpers (e.g. the
        ``Retry-After`` value on ``RESOURCE_EXHAUSTED`` 503s).
        """
        return self._engine_config

    @property
    def device(self) -> str:
        """Return the default device for model loading."""
        return self._device

    @property
    def devices(self) -> list[str]:
        """Return concrete devices available for whole-model placement."""
        return list(self._devices)

    def _memory_manager_for_device(self, device: str) -> MemoryManager:
        manager = self._memory_managers.get(device)
        if manager is None:
            manager = MemoryManager(device=device, config=self._memory_config)
            self._memory_managers[device] = manager
            self._device_order[device] = len(self._device_order)
        return manager

    def _memory_manager_for_model(self, name: str) -> MemoryManager:
        loaded = self._loaded.get(name)
        if loaded is None:
            return self._memory_manager
        return self._memory_manager_for_device(loaded.device)

    def _select_device_for_model(self, requested_family: str | None = None) -> str:
        candidates = [
            device for device in self._devices if requested_family is None or _device_family(device) == requested_family
        ]

        def score(device: str) -> tuple[bool, int, int]:
            manager = self._memory_manager_for_device(device)
            try:
                under_pressure = manager.check_pressure()
            except Exception:
                logger.exception("Failed to check memory pressure for device %s; keeping it eligible", device)
                under_pressure = False
            return (under_pressure, manager.loaded_model_count, self._device_order[device])

        return min(candidates, key=score)

    def _resolve_load_device(self, requested_device: str) -> str:
        """Resolve a request/default device to one concrete load target."""
        if len(self._devices) == 1:
            only_device = self._devices[0]
            if (
                requested_device != only_device
                and _device_family(requested_device) == _device_family(only_device)
                and ":" in only_device
                and ":" not in requested_device
            ):
                return only_device
            return requested_device
        if requested_device in self._memory_managers and requested_device not in self._devices:
            return requested_device
        if requested_device in self._devices:
            return requested_device
        if ":" in requested_device:
            return requested_device
        requested_family = _device_family(requested_device)
        device_families = {_device_family(device) for device in self._devices}
        if requested_family in device_families:
            return self._select_device_for_model(requested_family)
        return requested_device

    def has_model(self, name: str) -> bool:
        """Check if a model config exists in the registry."""
        return name in self._configs

    def is_loaded(self, name: str) -> bool:
        """Check if a model is currently loaded."""
        return name in self._loaded

    def is_loading(self, name: str) -> bool:
        """Check if a model is currently being loaded."""
        return name in self._loading

    def is_unloading(self, name: str) -> bool:
        """Check if a model is currently being unloaded."""
        return name in self._unloading

    def is_failed(self, name: str) -> bool:
        """Check whether a model has a recorded load failure still in cooldown.

        A model is considered "failed" only while its recorded
        :class:`LoadFailure` is within the cooldown window for its error
        class. Once the cooldown elapses (transient classes only) the
        failure is treated as expired and ``is_failed`` returns False so
        the next request can trigger a fresh attempt.

        Permanent failure classes (``GATED``, ``NOT_FOUND``,
        ``DEPENDENCY``, ``UNKNOWN``) have ``cooldown_s=None`` and stay
        sticky until :meth:`clear_failure` is invoked (e.g. by hot
        reload).
        """
        failure = self._failed.get(name)
        if failure is None:
            return False
        return failure.in_cooldown(time.monotonic())

    def get_failure(self, name: str) -> LoadFailure | None:
        """Return the recorded :class:`LoadFailure` for ``name`` if any.

        Returns ``None`` when no failure has been recorded; the failure
        record is returned regardless of cooldown so that ``GET /v1/models``
        can include diagnostic detail even after the cooldown has elapsed.
        """
        return self._failed.get(name)

    def clear_failure(self, name: str) -> bool:
        """Drop any recorded failure for ``name``.

        Returns True if a record existed and was removed. Used by hot
        reload (a config change is operator intent that may have fixed
        the underlying issue) and by successful loads.
        """
        return self._failed.pop(name, None) is not None

    def _get_load_lock(self) -> asyncio.Lock:
        """Get or create the load lock (must be called from async context)."""
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        return self._load_lock

    def _get_config_update_lock(self) -> asyncio.Lock:
        """Serialize async config mutations across IPC/hot-reload entrypoints."""
        if self._config_update_lock is None:
            self._config_update_lock = asyncio.Lock()
        return self._config_update_lock

    def _check_model_loadable(self, name: str) -> tuple[ModelConfig, Path]:
        """Check if a model can be loaded (exists in registry).

        This method performs config discovery synchronously:
        rescans models_dir if model not found.

        Call this BEFORE starting any background loading to surface errors early.

        Args:
            name: Model name.

        Returns:
            Tuple of (config, model_dir) for use in loading.

        Raises:
            KeyError: If model not found after rescan.
        """
        # On-demand config discovery
        if name not in self._configs:
            logger.info("Model '%s' not in registry, rescanning models_dir", name)
            self.rescan_configs()
            if name not in self._configs:
                msg = _ERR_MODEL_NOT_FOUND.format(name=name)
                raise KeyError(msg)

        config = self._configs[name]
        model_dir = self._model_dirs.get(name, Path())

        return config, model_dir

    def get_config(self, name: str) -> ModelConfig:
        """Get the config for a model.

        Args:
            name: Model name.

        Returns:
            The model's configuration.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=name)
            raise KeyError(msg)
        return self._configs[name]

    def resolve_model_id(self, raw_id: str) -> str | None:
        """Resolve a raw model id to a known config's ``sie_id``.

        Accepts profile-qualified (``sie_id:profile``) and differently-cased ids,
        which the pinned set may carry per the deployment design. Non-default
        profiles are first-class config entries keyed by the full ``base:profile``
        id (``loader._expand_profile_variants``), so this matches the FULL
        canonical id case-insensitively and returns that variant's original-case
        ``sie_id`` (not the base). ``:default`` folds to the base. Returns
        ``None`` when no config matches.
        """
        canonical = _canonical_pinned_id(raw_id)
        if not canonical:
            return None
        for sie_id in self._configs:
            if sie_id.strip().lower() == canonical:
                return sie_id
        return None

    def get(self, name: str) -> ModelAdapter:
        """Get a loaded model's adapter.

        Args:
            name: Model name.

        Returns:
            The model's adapter (must be loaded first).

        Raises:
            KeyError: If model not found or not loaded.
        """
        if name not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=name)
            raise KeyError(msg)
        if name not in self._loaded:
            msg = _ERR_MODEL_NOT_LOADED.format(name=name)
            raise KeyError(msg)

        # ``get`` is a pure read: it does not update LRU. Request hot paths call
        # ``touch_lru`` explicitly after obtaining the adapter (see #1541).
        return self._loaded[name].adapter

    def touch_lru(self, name: str) -> None:
        """Mark a loaded model as recently used for LRU eviction.

        :meth:`get` is a pure read and does not update LRU; request hot paths
        call this explicitly after obtaining the adapter so the model is
        protected from eviction. No-op for a model that is not loaded.
        """
        if name in self._loaded:
            self._memory_manager_for_model(name).touch(name)

    def load(self, name: str, device: str) -> ModelAdapter:
        """Load a model onto a device.

        If loading fails due to OOM, attempts to evict LRU model and retry once.

        Args:
            name: Model name.
            device: Device string (e.g., "cuda:0", "cpu").

        Returns:
            The loaded adapter.

        Raises:
            KeyError: If model not found.
            ValueError: If model already loaded.
            ImportError: If adapter cannot be loaded.
            RuntimeError: If OOM persists after eviction.
        """
        # Pre-load validation: config existence + dependency checking
        config, model_dir = self._check_model_loadable(name)

        if name in self._loaded:
            msg = _ERR_MODEL_ALREADY_LOADED.format(name=name)
            raise ValueError(msg)

        load_device = self._resolve_load_device(device)
        memory_manager = self._memory_manager_for_device(load_device)
        load_start = time.monotonic()
        load_outcome = "error"
        load_stage = "total"
        try:
            # Ensure model weights are cached (download from HF / cluster cache
            # if needed). Kept as a distinct phase from instantiate so the
            # post-download timeout in ``ModelLoader`` does not cover the
            # potentially long download — slow networks remain supported via
            # ``HF_HUB_DOWNLOAD_TIMEOUT`` stall detection only.
            self._loader.ensure_weights_cached(config)

            # Instantiate the adapter on the resolved concrete device.
            adapter = self._loader.instantiate_adapter(name, config, model_dir, load_device)

            required_load_bytes = _adapter_load_required_bytes(adapter, memory_manager)
            while memory_manager.should_evict_for_load(required_load_bytes):
                lru_model = memory_manager.get_lru_model(exclude=self._pinned_models)
                if lru_model is None:
                    break

                stats = memory_manager.get_stats()
                logger.info(
                    "Pre-load eviction: %s %.2f GB free, %.2f GB required, %.1f%% used "
                    "(threshold %.1f%%); evicting '%s' before loading '%s'",
                    load_device,
                    stats.available_gb,
                    (required_load_bytes or 0) / (1024**3),
                    stats.usage_ratio * 100,
                    memory_manager.pressure_threshold_pct,
                    lru_model,
                    name,
                )
                self.unload(lru_model, reason="preload_pressure")

            # Try to load onto device, with OOM retry.
            try:
                loaded = self._loader.load_and_register(name, load_device, adapter, config)
            except RuntimeError as exc:
                if not self._is_oom_error(exc):
                    raise

                # OOM - try to evict LRU non-pinned model and retry.
                lru_model = memory_manager.get_lru_model(exclude=self._pinned_models)
                if lru_model is None:
                    logger.error("OOM loading '%s' but no non-pinned models to evict", name)
                    raise

                logger.warning(
                    "OOM loading '%s', evicting LRU model '%s' and retrying",
                    name,
                    lru_model,
                )
                self.unload(lru_model, reason="load_oom")

                # Retry once after eviction - weights still on disk so we
                # skip ``ensure_weights_cached`` here.
                adapter = self._loader.instantiate_adapter(name, config, model_dir, load_device)
                loaded = self._loader.load_and_register(name, load_device, adapter, config)

            # Track loaded state and register it for LRU accounting.
            self._loaded[name] = loaded
            self._memory_manager_for_device(loaded.device).register_model(
                name,
                estimated_bytes=loaded.memory_bytes,
            )

            # Clear any stale failure record from a prior attempt.
            self._failed.pop(name, None)
            load_outcome = "success"
            return loaded.adapter
        except ModelLoadTimeoutError as exc:
            load_outcome = "timeout"
            load_stage = exc.stage
            raise
        finally:
            worker_telemetry().model_load_completed(
                model=name,
                duration_s=time.monotonic() - load_start,
                outcome=load_outcome,
                stage=load_stage,
            )

    async def load_async(self, name: str, device: str) -> ModelAdapter:
        """Load a model onto a device (async, concurrency-safe).

        This method:
        - Serializes all load/unload operations via an async lock
        - Proactively evicts LRU models if memory pressure or load headroom requires it
        - Runs the blocking load in a thread pool to avoid blocking the event loop
        - Falls back to OOM-triggered eviction if load still fails (fragmentation)
        - On-demand config discovery: rescans models_dir if model not found
        - Dependency checking: verifies model deps match installed packages

        Args:
            name: Model name.
            device: Device string (e.g., "cuda:0", "cpu").

        Returns:
            The loaded adapter.

        Raises:
            KeyError: If model not found after rescan.
            RuntimeError: If OOM persists after eviction.
        """
        # Pre-load validation: config discovery
        # Done before acquiring lock to surface errors early
        self._check_model_loadable(name)

        lock = self._get_load_lock()
        async with lock:
            # Double-check after acquiring lock (another request may have loaded it)
            if name in self._loaded:
                self._memory_manager_for_model(name).touch(name)
                return self._loaded[name].adapter

            load_device = self._resolve_load_device(device)
            memory_manager = self._memory_manager_for_device(load_device)

            # Check if model is being unloaded - caller should retry
            if name in self._unloading:
                msg = f"Model '{name}' is currently being unloaded"
                raise RuntimeError(msg)

            config = self._configs[name]

            # Mark as loading before starting (visible to WebSocket status)
            self._loading.add(name)
            load_start = time.monotonic()
            load_outcome = "error"
            load_stage = "total"

            try:
                model_dir = self._model_dirs.get(name, Path())

                # Ensure weights are cached BEFORE instantiation. This
                # phase is intentionally unbounded by the post-download
                # timeout in ``ModelLoader`` — slow user networks are
                # supported via ``HF_HUB_DOWNLOAD_TIMEOUT`` stall
                # detection inside ``huggingface_hub`` only.
                await self._loader.ensure_weights_cached_async(name, config)

                # Instantiate adapter (in thread pool, post-download timeout applies)
                adapter = await self._loader.instantiate_adapter_async(name, config, model_dir, load_device)

                required_load_bytes = _adapter_load_required_bytes(adapter, memory_manager)

                # Pre-load eviction: evict LRU non-pinned models until current pressure
                # and any adapter-provided load headroom requirement are satisfied.
                while memory_manager.should_evict_for_load(required_load_bytes):
                    lru_model = memory_manager.get_lru_model(exclude=self._pinned_models)
                    if lru_model is None:
                        break  # No non-pinned models to evict, proceed with load attempt

                    if lru_model not in self._loaded:
                        # Belt-and-braces: a MemoryManager entry with no matching
                        # ``_loaded`` model is a ghost (see #1600). ``_do_unload``
                        # would no-op on it, so drop the stale accounting entry and
                        # re-evaluate rather than spin. The ``_do_unload`` finally
                        # normally prevents ghosts; this guards any other source.
                        logger.warning(
                            "Evictor found ghost '%s' (in memory manager, not loaded); dropping stale entry",
                            lru_model,
                        )
                        memory_manager.unregister_model(lru_model)
                        continue

                    stats = memory_manager.get_stats()
                    logger.info(
                        "Pre-load eviction: %s %.2f GB free, %.2f GB required, %.1f%% used "
                        "(threshold %.1f%%); evicting '%s' before loading '%s'",
                        load_device,
                        stats.available_gb,
                        (required_load_bytes or 0) / (1024**3),
                        stats.usage_ratio * 100,
                        memory_manager.pressure_threshold_pct,
                        lru_model,
                        name,
                    )
                    await self._do_unload(lru_model, reason="preload_pressure")

                try:
                    # Load onto device (loader handles main thread vs executor)
                    loaded = await self._loader.load_and_register_async(name, load_device, adapter, config)
                except RuntimeError as e:
                    if not self._is_oom_error(e):
                        raise

                    # OOM despite pre-load eviction: evict LRU non-pinned model and retry once
                    lru_model = memory_manager.get_lru_model(exclude=self._pinned_models)
                    if lru_model is None:
                        logger.error("OOM loading '%s' but no non-pinned models to evict", name)
                        raise

                    logger.warning(
                        "OOM loading '%s' despite pre-eviction, evicting '%s' and retrying",
                        name,
                        lru_model,
                    )
                    await self._do_unload(lru_model, reason="load_oom")

                    # Retry once after eviction. Weights are still cached
                    # on disk so we skip ``ensure_weights_cached_async``.
                    adapter = await self._loader.instantiate_adapter_async(name, config, model_dir, load_device)
                    loaded = await self._loader.load_and_register_async(name, load_device, adapter, config)

                # Track loaded state
                self._loaded[name] = loaded

                # Register with memory manager for LRU tracking
                self._memory_manager_for_device(loaded.device).register_model(
                    name,
                    estimated_bytes=loaded.memory_bytes,
                )

                load_duration = time.monotonic() - load_start
                if load_duration > 300:
                    logger.warning(
                        "Model '%s' took %.0fs to load (>300s) — may indicate a gated model "
                        "missing HF_TOKEN or network issues",
                        name,
                        load_duration,
                    )

                load_outcome = "success"
                return loaded.adapter
            except ModelLoadTimeoutError as exc:
                load_outcome = "timeout"
                load_stage = exc.stage
                raise
            finally:
                worker_telemetry().model_load_completed(
                    model=name,
                    duration_s=time.monotonic() - load_start,
                    outcome=load_outcome,
                    stage=load_stage,
                )
                # Always clear loading state when done (success or failure)
                self._loading.discard(name)

    async def start_load_async(self, name: str, device: str) -> bool:
        """Start loading a model in the background (non-blocking).

        This method triggers model loading in a background task and returns
        immediately. Use is_loading() to check if load is in progress.

        Pre-load validation (config existence) is performed
        synchronously before starting the background task, so errors like
        KeyError are raised immediately.

        Args:
            name: Model name.
            device: Device string (e.g., "cuda:0", "cpu").

        Returns:
            True if load was started, False if model is already loaded or loading.

        Raises:
            KeyError: If model not found.
        """
        # Already loaded - no action needed
        if name in self._loaded:
            return False

        # Already loading - no action needed
        if name in self._loading:
            return False

        # Recorded failure still in cooldown — don't start another doomed
        # background load. The API surface (``ensure_loaded``) checks
        # ``is_failed`` and returns ``MODEL_LOAD_FAILED`` instead of
        # ``MODEL_LOADING``, so the SDK won't keep retrying.
        if self.is_failed(name):
            return False

        # Check model exists and dependencies are compatible BEFORE starting background task.
        # This surfaces errors synchronously so the API can return 404/409 immediately.
        self._check_model_loadable(name)

        # Mark as loading before creating task to avoid race conditions
        self._loading.add(name)

        # Start background load task
        task = asyncio.create_task(
            self._load_model_background(name, device),
            name=f"model-load-{name}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return True

    async def _load_model_background(self, name: str, device: str) -> None:
        """Load a model in the background (called via asyncio.create_task).

        This method wraps load_async() and handles logging. The loading state
        is managed here with a finally block to ensure cleanup even if load_async
        returns early (e.g., model already loaded by another task).

        On success any previously-recorded :class:`LoadFailure` for ``name``
        is cleared. On failure the exception is classified via
        :func:`classify_load_error` and recorded into ``self._failed`` so
        the API surface can return a non-retryable
        ``MODEL_LOAD_FAILED`` and the SDK stops hammering the loader.

        Args:
            name: Model name.
            device: Device string (e.g., "cuda:0", "cpu").
        """
        try:
            await self.load_async(name, device)
            logger.info("Background model load completed: %s", name)
            # Successful load clears any prior failure record (e.g. a
            # transient OOM that has since been resolved by eviction).
            self._failed.pop(name, None)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            # Operator-initiated shutdown / task cancellation must NOT
            # be recorded as a load failure — that would leave the
            # model permanently in ``failed`` state across server
            # restarts. Let these propagate so asyncio's task lifecycle
            # handles them normally.
            raise
        except Exception as exc:  # noqa: BLE001 — classify_load_error buckets every exception type
            self._record_load_failure(name, exc)
        finally:
            # Always remove from _loading when background task completes.
            # This handles the case where load_async returned early (model already
            # loaded by another task) without entering its own finally block.
            # For normal loads, load_async already discarded, so this is a no-op.
            self._loading.discard(name)

    def _record_load_failure(self, name: str, exc: BaseException) -> None:
        """Classify ``exc`` and record a :class:`LoadFailure` for ``name``.

        Increments ``attempts`` if a failure is already on file. Emits a
        WARNING-level log with structured fields so operators see the
        actionable hint immediately rather than waiting for the SDK's
        retry budget to elapse.
        """
        classification = classify_load_error(exc)
        previous = self._failed.get(name)
        attempts = (previous.attempts + 1) if previous is not None else 1
        message = f"{type(exc).__name__}: {exc}"
        failure = LoadFailure(
            error_class=classification.error_class,
            message=message,
            attempts=attempts,
            last_attempt_ts=time.monotonic(),
            cooldown_s=classification.cooldown_s,
        )
        self._failed[name] = failure

        if classification.error_class is LoadErrorClass.GATED:
            # Gated models almost always indicate a missing/invalid HF_TOKEN.
            # Emit the actionable hint up-front instead of relying on the
            # 300s soft-warning that only fires on slow loads.
            logger.warning(
                "Model '%s' load failed: gated repository (attempts=%d). "
                "Ensure HF_TOKEN is set and the account has accepted the model "
                "license on HuggingFace. Underlying error: %s",
                name,
                attempts,
                message,
                extra={"gated_model": True, "model": name},
            )
        else:
            logger.exception(
                "Background model load failed: %s (class=%s, attempts=%d, cooldown=%s)",
                name,
                classification.error_class.value,
                attempts,
                "permanent" if classification.cooldown_s is None else f"{classification.cooldown_s:.0f}s",
            )

    def unload(self, name: str, *, reason: str = "manual") -> None:
        """Unload a model and free resources.

        Args:
            name: Model name.

        Raises:
            KeyError: If model not found or not loaded.
        """
        if name not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=name)
            raise KeyError(msg)

        if name not in self._loaded:
            msg = _ERR_MODEL_NOT_LOADED.format(name=name)
            raise KeyError(msg)

        logger.info("Unloading model '%s'", name)

        loaded = self._loaded.pop(name)
        device = loaded.device
        worker_telemetry().model_evicted(model=name, reason=reason)

        # Adapter.unload() handles gc.collect + empty_cache
        loaded.adapter.unload()

        # Unregister tokenizer, preprocessor, and clear metrics
        self._loader.unregister(name, device)

        # Unregister from memory manager
        self._memory_manager_for_device(device).unregister_model(name)

        logger.info("Model '%s' unloaded", name)

    def _clear_transient_failures(self, classes: tuple[LoadErrorClass, ...] = (LoadErrorClass.OOM,)) -> int:
        """Drop transient-class failure records for all models.

        Used after a successful unload (memory has been freed, so a prior
        ``OOM`` is no longer relevant) and after device-level recovery
        events. Returns the number of records cleared.

        Permanent classes (``GATED``, ``DEPENDENCY``, ``NOT_FOUND``,
        ``UNKNOWN``) are never cleared by this helper — those require
        operator intent (config update or explicit ``clear_failure``).
        """
        cleared = 0
        for name, failure in list(self._failed.items()):
            if failure.error_class in classes:
                self._failed.pop(name, None)
                cleared += 1
        return cleared

    async def _do_unload(self, name: str, *, reason: str = "other") -> None:
        """Unload a model safely (drains worker first).

        Must be called while holding _load_lock.

        Args:
            name: Model name.
        """
        loaded = self._loaded.get(name)
        if loaded is None:
            return

        # Mark as unloading so new requests get 503
        self._unloading.add(name)

        try:
            logger.info("Unloading model '%s' (draining worker first)", name)

            # Stop worker (waits for pending batches, up to drain_timeout_s)
            if loaded.worker is not None and loaded.worker.is_running:
                try:
                    await asyncio.wait_for(
                        loaded.worker.stop(),
                        timeout=self._drain_timeout_s,
                    )
                    logger.info("Worker drained for model '%s'", name)
                except TimeoutError:
                    logger.warning(
                        "Worker drain timeout for model '%s' after %.1fs, force stopping",
                        name,
                        self._drain_timeout_s,
                    )
                    # Force stop is handled by the worker.stop() method

            # Remove from loaded dict before unloading adapter
            del self._loaded[name]
            device = loaded.device
            worker_telemetry().model_evicted(model=name, reason=reason)

            # Once removed from ``_loaded`` above, the model is committed-gone
            # from the registry's view, so its MemoryManager entry MUST be
            # dropped even if adapter teardown below raises. Otherwise the
            # model becomes a "ghost" — absent from ``_loaded`` but still
            # counted in ``_models`` — that ``get_lru_model`` keeps
            # re-selecting while ``_do_unload`` no-ops on it: a no-yield spin
            # under ``_load_lock`` that starves the whole event loop. See #1600.
            try:
                # Some adapters (e.g. the SGLang generation adapter) hold an
                # async HTTP client to a subprocess. ``unload()`` is sync and
                # can only fire-and-forget the client close, which races the
                # subprocess termination on the next line — the close could be
                # cut off before its connections drain, leaking fds / wedging
                # on a half-open socket. When the adapter exposes an awaitable
                # ``aclose_client``, drive it to completion HERE (we're on the
                # event loop) so the client is fully closed against the still-
                # live subprocess before ``unload()`` terminates it.
                aclose_client = getattr(loaded.adapter, "aclose_client", None)
                if aclose_client is not None:
                    try:
                        await aclose_client()
                    except Exception:  # noqa: BLE001 - close is best-effort
                        logger.warning("aclose_client() failed during unload of '%s'", name, exc_info=True)

                # Adapter.unload() handles gc.collect + empty_cache
                loaded.adapter.unload()
            finally:
                # Always run, even if adapter teardown above raised, so a
                # failed unload can never leave a ghost behind (#1600).
                try:
                    # Unregister tokenizer, preprocessor, and clear metrics
                    self._loader.unregister(name, device)
                except Exception:  # noqa: BLE001 - best-effort; must not skip below
                    logger.warning("loader.unregister failed during unload of '%s'", name, exc_info=True)
                # Unregister from memory manager (a plain dict-pop; never raises)
                self._memory_manager_for_device(device).unregister_model(name)

            # Freeing memory makes any prior OOM-class load failure on a
            # *sibling* model retryable; clear those records so the next
            # request can re-attempt without waiting out the cooldown.
            cleared = self._clear_transient_failures()
            if cleared:
                logger.debug(
                    "Cleared %d transient failure record(s) after unloading '%s'",
                    cleared,
                    name,
                )

            logger.info("Model '%s' unloaded", name)
        finally:
            self._unloading.discard(name)

    async def unload_async(self, name: str) -> None:
        """Unload a model and free resources (async, concurrency-safe).

        This method acquires the load lock and safely drains the worker
        before unloading the model.

        Args:
            name: Model name.

        Raises:
            KeyError: If model not found or not loaded.
        """
        if name not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=name)
            raise KeyError(msg)

        lock = self._get_load_lock()
        async with lock:
            if name not in self._loaded:
                msg = _ERR_MODEL_NOT_LOADED.format(name=name)
                raise KeyError(msg)

            await self._do_unload(name, reason="manual")

    def unload_all(self) -> None:
        """Unload all loaded models (sync version, for non-async contexts)."""
        for name in list(self._loaded.keys()):
            self.unload(name, reason="shutdown")

    async def unload_all_async(self) -> None:
        """Unload all loaded models (async, concurrency-safe)."""
        lock = self._get_load_lock()
        async with lock:
            for name in list(self._loaded.keys()):
                await self._do_unload(name, reason="shutdown")

    def get_configs_snapshot(self, bundle_id: str | None = None) -> dict[str, ModelConfig]:
        """Return a shallow copy of the current model configs.

        Thread-safe snapshot for reading configs without holding internal locks.

        Args:
            bundle_id: If provided, only return configs that belong to this
                worker's model filter (the set of models matched to the bundle
                at startup). When *None*, all configs are returned.

        Returns:
            Copy of the configs dict, optionally filtered.
        """
        configs = dict(self._configs)
        if bundle_id is not None and self._model_filter is not None:
            configs = {k: v for k, v in configs.items() if k in self._model_filter}
        return configs

    def _refresh_worker_metric_catalog(self) -> None:
        """Publish the current registry catalog to the bounded metric facade.

        The registry owns config mutation, so refresh belongs here rather than
        in individual IPC, filesystem, or hot-reload callers. Telemetry is
        strictly fail-open: a catalog projection bug must never reject an
        otherwise valid serving configuration.
        """
        try:
            refresh_worker_metric_context(configs=self.get_configs_snapshot())
        except Exception:  # noqa: BLE001 - telemetry cannot break config apply
            logger.warning("Failed to refresh worker metric catalog; serving config remains active", exc_info=True)

    def _validate_pool_isolation_of_loaded(self) -> None:
        """Enforce pool isolation across currently-loaded configs.

        Buckets configs by task class (gen vs non-gen) in a single
        O(n) pass and asserts at most one bucket is non-empty. Raises
        :class:`PoolIsolationError` naming the first incompatible pair
        when both buckets are non-empty.

        Called from :meth:`_load_configs_from_dir` (also reached from
        :meth:`rescan_configs`, which fires on request-handler config
        misses); the linear shape matters there because hot-discovery
        storms can rescan repeatedly.
        """
        assert self._pool_name is not None  # caller-checked
        gen_names: list[str] = []
        non_gen_names: list[str] = []
        for name, config in self._configs.items():
            if config.tasks.generate is not None:
                gen_names.append(name)
            else:
                non_gen_names.append(name)
        if gen_names and non_gen_names:
            # Delegate to ``validate_pool_isolation`` so the error message
            # stays consistent with the single-config add path.
            validate_pool_isolation(
                candidate_name=non_gen_names[0],
                candidate_config=self._configs[non_gen_names[0]],
                existing_configs={gen_names[0]: self._configs[gen_names[0]]},
                pool_name=self._pool_name,
            )

    def add_config(self, config: ModelConfig, model_dir: Path | None = None) -> set[str]:
        """Add a model config to the registry.

        Useful for programmatic config creation without files.

        Args:
            config: The model configuration.
            model_dir: Optional directory for custom adapter resolution.

        Returns:
            Concrete config ids considered by this update, including generated
            profile variants.
        """
        expanded, updated_ids, removed_ids = self._prepare_config_update(config)
        self._preflight_config_update(expanded, updated_ids, removed_ids)
        if removed_ids:
            removed = ", ".join(sorted(removed_ids))
            msg = f"cannot synchronously remove model config(s): {removed}; use add_config_async"
            raise RuntimeError(msg)
        self._apply_config_entry(config, model_dir)
        self._loader.update_configs(self._configs)
        self._refresh_worker_metric_catalog()
        return updated_ids | removed_ids

    async def add_config_async(self, config: ModelConfig, model_dir: Path | None = None) -> set[str]:
        """Add a model config, draining removed loaded variants under the load lock."""
        update_lock = self._get_config_update_lock()
        async with update_lock:
            expanded, updated_ids, removed_ids = self._prepare_config_update(config)
            self._preflight_config_update(expanded, updated_ids, removed_ids)
            changed_ids = {
                model_id
                for model_id, candidate in expanded.items()
                if model_id in self._configs
                and not _model_configs_semantically_equal(self._configs.get(model_id), candidate)
            }
            invalidated_ids = removed_ids | changed_ids
            if invalidated_ids:
                lock = self._get_load_lock()
                async with lock:
                    for model_id in sorted(invalidated_ids):
                        if model_id in self._loaded:
                            await self._do_unload(model_id, reason="config_change")
                    for model_id in sorted(removed_ids):
                        self._drop_config_entry(model_id)
            self._apply_config_entry(config, model_dir)
            self._loader.update_configs(self._configs)
            self._refresh_worker_metric_catalog()
            return updated_ids | removed_ids

    async def remove_config_async(self, model_id: str) -> set[str]:
        """Drain and remove one base config plus its materialized variants.

        This is the authoritative delete seam for filesystem hot reload. It
        keeps loader state, failure state, and the telemetry catalog in lockstep
        and handles profile-only configs whose bare base id is not routable.
        """
        update_lock = self._get_config_update_lock()
        async with update_lock:
            removed_ids = ({model_id} | self._synthetic_profile_variant_ids_for_base(model_id)) & set(self._configs)
            if not removed_ids:
                return set()

            lock = self._get_load_lock()
            async with lock:
                for removed_id in sorted(removed_ids):
                    if removed_id in self._loaded:
                        await self._do_unload(removed_id, reason="config_change")
                    self._drop_config_entry(removed_id)
                self._loader.update_configs(self._configs)

            self._refresh_worker_metric_catalog()
            return removed_ids

    def _prepare_config_update(self, config: ModelConfig) -> tuple[dict[str, ModelConfig], set[str], set[str]]:
        expanded = expand_profile_variants([config])
        updated_ids = set(expanded)
        previously_materialized = self._synthetic_profile_variant_ids_for_base(config.sie_id)
        if config.sie_id in self._configs:
            previously_materialized.add(config.sie_id)
        removed_ids = previously_materialized - updated_ids
        return expanded, updated_ids, removed_ids

    def _preflight_config_update(
        self,
        expanded: dict[str, ModelConfig],
        updated_ids: set[str],
        removed_ids: set[str],
    ) -> None:
        post_update = {
            name: existing_config
            for name, existing_config in self._configs.items()
            if name not in updated_ids and name not in removed_ids
        }
        for expanded_config in expanded.values():
            if not self.accepts_config_pool(expanded_config):
                post_update.pop(expanded_config.sie_id, None)
                continue
            self._validate_config_entry(expanded_config, post_update)
            post_update[expanded_config.sie_id] = expanded_config

    def _synthetic_profile_variant_ids_for_base(self, base_model_id: str) -> set[str]:
        return {
            name
            for name, existing_config in self._configs.items()
            if existing_config.synthetic_profile_variant_source is not None
            and existing_config.synthetic_profile_variant_source[0] == base_model_id
        }

    def _drop_config_entry(self, model_id: str) -> None:
        self._configs.pop(model_id, None)
        if self._model_filter is not None:
            self._model_filter.discard(model_id)
        self._model_dirs.pop(model_id, None)
        self._failed.pop(model_id, None)
        self._config_version += 1

    def _add_config_entry(self, config: ModelConfig, model_dir: Path | None = None) -> None:
        for expanded_config in expand_profile_variants([config]).values():
            if self.accepts_config_pool(expanded_config):
                self._validate_config_entry(expanded_config, self._configs)
        self._apply_config_entry(config, model_dir)

    def _validate_config_entry(self, config: ModelConfig, existing_configs: dict[str, ModelConfig]) -> None:
        # Pool isolation. Validate before mutating
        # ``self._configs`` so a rejected add does not leave the
        # registry in a half-mutated state. ``None`` pool skips the
        # check (test paths and registries running without a
        # SIE_POOL context).
        if self._pool_name is not None:
            validate_pool_isolation(
                candidate_name=config.sie_id,
                candidate_config=config,
                existing_configs=existing_configs,
                pool_name=self._pool_name,
            )
        # Legacy scalar ``lora_id`` exclusion is *not* pool-scoped (it is a
        # hard invariant) so it fires regardless of ``self._pool_name``.
        # Multi-LoRA generation (``loadtime.lora_paths``) is unaffected.
        validate_no_legacy_scalar_lora_id(name=config.sie_id, config=config)

    def _apply_config_entry(self, config: ModelConfig, model_dir: Path | None = None) -> None:
        base_id = (
            config.synthetic_profile_variant_source[0]
            if config.synthetic_profile_variant_source is not None
            else config.sie_id
        )
        expanded_configs = {
            name: expanded_config
            for name, expanded_config in expand_profile_variants([config]).items()
            if self.accepts_config_pool(expanded_config)
        }
        if not expanded_configs:
            stale_ids = {base_id} | self._synthetic_profile_variant_ids_for_base(base_id)
            for stale_id in sorted(stale_ids):
                self._drop_config_entry(stale_id)
            return
        stale_ids = ({base_id} | self._synthetic_profile_variant_ids_for_base(base_id)) - set(expanded_configs)
        for stale_id in sorted(stale_ids):
            self._drop_config_entry(stale_id)
        self._configs.update(expanded_configs)
        if self._model_filter is not None:
            self._model_filter.update(expanded_configs)
        if model_dir is not None:
            for name in expanded_configs:
                self._model_dirs[name] = model_dir
        # A config update is operator intent that may have fixed the
        # underlying issue (e.g. pinned a working revision, dropped a
        # broken adapter option). Clear any sticky failure so the next
        # request retries with the new config.
        for name in expanded_configs:
            self._failed.pop(name, None)
        self._config_version += 1
        # The config for an already-pinned model may have just arrived at runtime
        # (default Helm workers receive configs via the sidecar after a pin was
        # recorded). Eager-load it now so the pin actually keeps it resident.
        for name in expanded_configs:
            self._maybe_schedule_pinned_load(name)

    async def replace_configs_async(
        self,
        configs: Iterable[ModelConfig],
        model_dir: Path | None = None,
    ) -> set[str]:
        """Replace the registry config set from an authoritative snapshot.

        Used by worker-sidecar export reconciliation. Removed or changed
        loaded models are unloaded under the registry load lock before the
        config map is swapped, so a worker cannot keep serving a model that
        disappeared from the authoritative bundle export.
        """
        new_configs: dict[str, ModelConfig] = {}
        for config in expand_profile_variants(configs).values():
            if not self.accepts_config_pool(config):
                continue
            if config.sie_id in new_configs:
                msg = f"duplicate model config in authoritative snapshot: {config.sie_id}"
                raise ValueError(msg)
            new_configs[config.sie_id] = config

        if self._pool_name is not None:
            accepted: dict[str, ModelConfig] = {}
            for name, config in new_configs.items():
                validate_pool_isolation(
                    candidate_name=name,
                    candidate_config=config,
                    existing_configs=accepted,
                    pool_name=self._pool_name,
                )
                accepted[name] = config
        for name, config in new_configs.items():
            validate_no_legacy_scalar_lora_id(name=name, config=config)

        update_lock = self._get_config_update_lock()
        async with update_lock:
            lock = self._get_load_lock()
            async with lock:
                removed = set(self._configs) - set(new_configs)
                changed = {
                    name
                    for name, config in new_configs.items()
                    if not _model_configs_semantically_equal(self._configs.get(name), config)
                }
                if model_dir is not None:
                    changed.update(name for name in new_configs if self._model_dirs.get(name) != model_dir)
                invalidated = removed | changed

                if not invalidated:
                    return set()

                for name in sorted(invalidated):
                    if name in self._loaded:
                        await self._do_unload(name, reason="config_change")

                self._configs = dict(new_configs)
                self._loader.update_configs(self._configs)
                if self._model_filter is not None:
                    self._model_filter = set(new_configs)
                self._model_dirs = {name: path for name, path in self._model_dirs.items() if name in new_configs}
                if model_dir is not None:
                    for name in new_configs:
                        self._model_dirs[name] = model_dir
                for name in invalidated:
                    self._failed.pop(name, None)
                self._config_version += 1

            self._refresh_worker_metric_catalog()

        # Outside the load lock (``start_load_async`` acquires it): reload pinned
        # models that were just unloaded because their config changed, and load
        # any pinned model whose config arrived in this snapshot.
        await self._eager_load_pinned()
        return invalidated

    def accepts_config_pool(self, config: ModelConfig) -> bool:
        if self._pool_name is None:
            pool_matches = True
        else:
            config_pool = (config.pool or "default").strip().lower()
            pool_matches = config_pool == self._pool_name
        if not pool_matches:
            return False
        return _is_python_runtime_adapter(config.resolve_profile("default").adapter_path)

    def get_model_info(self, name: str) -> dict[str, Any]:
        """Get information about a model.

        Args:
            name: Model name.

        Returns:
            Dictionary with model information.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=name)
            raise KeyError(msg)

        config = self._configs[name]
        loaded = self._loaded.get(name)

        return {
            "name": config.sie_id,
            "inputs": config.inputs.to_list(),
            "outputs": config.outputs,
            "dims": config.dims,
            "max_sequence_length": config.max_sequence_length,
            "loaded": loaded is not None,
            "device": loaded.device if loaded else None,
        }

    def get_worker(self, name: str) -> ModelWorker | None:
        """Get the worker for a loaded model.

        Args:
            name: Model name.

        Returns:
            The model's worker if loaded, None if model not loaded.

        Raises:
            KeyError: If model not found in registry.
        """
        if name not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=name)
            raise KeyError(msg)
        if name not in self._loaded:
            return None
        return self._loaded[name].worker

    async def start_worker(self, name: str) -> ModelWorker:
        """Start the worker for a loaded model.

        If already running, this is a no-op. Uses the load/unload lock so
        request-driven idle touches cannot race an idle-eviction unload
        decision.

        Args:
            name: Model name.

        Returns:
            The model's worker.

        Raises:
            KeyError: If model not found or not loaded.
        """
        if name not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=name)
            raise KeyError(msg)
        lock = self._get_load_lock()
        async with lock:
            if name not in self._loaded:
                msg = _ERR_MODEL_NOT_LOADED.format(name=name)
                raise KeyError(msg)
            if name in self._unloading:
                msg = f"Model '{name}' is currently being unloaded"
                raise RuntimeError(msg)

            loaded = self._loaded[name]
            worker = loaded.worker
            if worker is None:
                msg = f"Worker not available for model '{name}'"
                raise RuntimeError(msg)

            self._memory_manager_for_model(name).touch(name)

            if not worker.is_running:
                await worker.start()
                logger.info("Worker started for model '%s'", name)
            return worker

    async def stop_worker(self, name: str) -> None:
        """Stop the worker for a loaded model.

        If not running, this is a no-op.

        Args:
            name: Model name.

        Raises:
            KeyError: If model not found or not loaded.
        """
        if name not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=name)
            raise KeyError(msg)
        if name not in self._loaded:
            msg = _ERR_MODEL_NOT_LOADED.format(name=name)
            raise KeyError(msg)

        worker = self._loaded[name].worker
        if worker is not None and worker.is_running:
            await worker.stop()
            logger.info("Worker stopped for model '%s'", name)

    async def stop_all_workers(self) -> None:
        """Stop all running workers."""
        for name in self._loaded:
            worker = self._loaded[name].worker
            if worker is not None and worker.is_running:
                await worker.stop()
                logger.info("Worker stopped for model '%s'", name)

    def _is_oom_error(self, error: RuntimeError) -> bool:
        """Check if a RuntimeError is an out-of-memory error.

        Thin wrapper over :func:`sie_server.core.oom.is_oom_error` kept for
        backwards-compatibility with existing call sites in this module.
        New code should import from ``core.oom`` directly.
        """
        return is_oom_error(error)

    async def evict_lru_excluding(self, exclude_name: str, *, timeout_s: float = 5.0) -> EvictionResult:
        """Evict the LRU model that is not ``exclude_name``.

        Used by the per-worker OOM recovery executor to free GPU memory by
        unloading a cold sibling model when an inference attempt OOMs. The
        load-lock is acquired with a soft timeout so a deadlocked / busy
        registry doesn't stall the worker indefinitely.

        **Concurrency caveat — drain-under-lock:** this method holds the
        registry's load-lock for the duration of ``_do_unload``, which
        includes ``worker.stop()`` waiting up to ``_DRAIN_TIMEOUT_S`` (30 s)
        for in-flight requests to complete. During a memory-pressure
        incident multiple workers may invoke ``evict_lru_excluding``
        concurrently; the soft ``timeout_s`` here prevents permanent
        deadlock, but a single eviction can still block other registry
        operations (additional loads / unloads) for up to the drain
        timeout. Operators should expect a transient stall window during
        large multi-worker OOM episodes and not interpret it as a hang.
        Splitting the drain off the lock is tracked as a follow-up.

        Args:
            exclude_name: The calling worker's own model. Never evicted, even
                if it happens to be the LRU entry — the caller still needs
                its weights resident.
            timeout_s: How long to wait for the registry's load-lock before
                giving up.

        Returns:
            An :class:`EvictionResult` reporting which outcome occurred —
            ``EVICTED`` when a sibling was unloaded, or ``NO_CANDIDATE`` /
            ``LOCK_TIMEOUT`` / ``UNLOAD_FAILED`` for the three formerly
            conflated failure modes.
        """
        lock = self._get_load_lock()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout_s)
        except TimeoutError:
            logger.warning(
                "evict_lru_excluding: lock acquisition timed out after %.1fs (caller=%s)",
                timeout_s,
                exclude_name,
            )
            return EvictionResult.LOCK_TIMEOUT

        try:
            manager = self._memory_manager_for_model(exclude_name)
            # Residency policy owns the *decision* (which sibling to evict);
            # the registry owns the *action* (draining + unloading it). Use
            # the caller's device-local LRU order so multi-device OOM recovery
            # frees memory on the GPU that actually hit pressure.
            candidate = select_eviction_candidate(
                manager.loaded_models,
                exclude_name=exclude_name,
                is_pinned=self._is_pinned,
                loaded=self._loaded,
                unloading=self._unloading,
            )
            if candidate is None:
                return EvictionResult.NO_CANDIDATE
            logger.info(
                "OOM recovery: evicting LRU sibling '%s' on %s (caller='%s')",
                candidate,
                manager.device,
                exclude_name,
            )
            try:
                await self._do_unload(candidate, reason="oom_recovery")
            except Exception:
                logger.exception("OOM recovery: failed to evict '%s'", candidate)
                return EvictionResult.UNLOAD_FAILED
            return EvictionResult.EVICTED
        finally:
            lock.release()

    async def start_memory_monitor(self) -> None:
        """Start the background memory monitor task.

        The monitor periodically checks memory pressure and evicts LRU models
        if needed. This catches memory growth during inference.
        """
        if self._monitor_task is not None:
            return  # Already running

        self._monitor_running = True
        self._monitor_task = asyncio.create_task(self._memory_monitor_loop())
        logger.info(
            "Memory monitor started (interval=%.1fs, threshold=%.0f%%)",
            self._memory_config.memory_check_interval_s,
            self._memory_config.pressure_threshold * 100,
        )

    async def stop_memory_monitor(self) -> None:
        """Stop the background memory monitor task."""
        if self._monitor_task is None:
            return

        self._monitor_running = False
        self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass
        self._monitor_task = None
        logger.info("Memory monitor stopped")

    async def start_idle_evictor(self) -> None:
        """Start the proactive idle-eviction loop, if configured.

        No-op when ``engine_config.idle_evict_s`` is None. Idempotent: a
        second call while the loop is running returns immediately. Paired
        with :meth:`stop_idle_evictor`; both are typically wired into the
        FastAPI lifespan alongside ``start_memory_monitor``.
        """
        if self._idle_evict_task is not None:
            return  # already running
        if self._engine_config is None or self._engine_config.idle_evict_s is None:
            logger.debug("Idle evictor disabled (idle_evict_s is None)")
            return

        self._idle_evict_running = True
        self._idle_evict_task = asyncio.create_task(self._idle_evict_loop())
        logger.info(
            "Idle evictor started (idle_threshold=%ds, check_interval=%.1fs)",
            self._engine_config.idle_evict_s,
            self._memory_config.memory_check_interval_s,
        )

    async def stop_idle_evictor(self) -> None:
        """Stop the idle-eviction loop. No-op if not running."""
        if self._idle_evict_task is None:
            return

        self._idle_evict_running = False
        self._idle_evict_task.cancel()
        try:
            await self._idle_evict_task
        except asyncio.CancelledError:
            pass
        self._idle_evict_task = None
        logger.info("Idle evictor stopped")

    async def _idle_evict_loop(self) -> None:
        """Background loop: unload models idle longer than ``idle_evict_s``.

        Runs at the same cadence as the pressure monitor. On each tick:
        1. Snapshot stale models (no lock — read-only over the LRU dict).
        2. Take the load lock once, walk the snapshot, evict the first
           still-stale model, then release. Subsequent stale models wait
           for the next tick. This caps lock contention against in-flight
           ``load_async`` callers and matches the "one eviction per tick"
           cadence of the existing pressure monitor.

        The recheck inside the lock guards against the snapshot/unload race:
        a request that arrived after the snapshot may have bumped
        ``last_used_at``, in which case we skip and look at the next
        candidate.

        Idle eviction *can* unload the only loaded model. The pressure
        monitor explicitly skips that case (because the eviction would
        immediately be undone by a new request); for idle eviction the
        operator's TTL is the explicit signal that they want even sole
        models to be released after idle.

        Configuration is bound at task-start: ``idle_evict_s`` and
        ``memory_check_interval_s`` are read once. Hot-reloading these
        values requires restarting the loop via ``stop_idle_evictor`` /
        ``start_idle_evictor``; documented as a future improvement.
        """
        assert self._engine_config is not None  # checked at start_idle_evictor
        interval = self._memory_config.memory_check_interval_s
        threshold = self._engine_config.idle_evict_s
        assert threshold is not None  # checked at start_idle_evictor

        while self._idle_evict_running:
            try:
                await asyncio.sleep(interval)
                stale = [
                    name
                    for manager in self._memory_managers.values()
                    for name in manager.get_idle_models(idle_threshold_s=threshold, exclude=self._pinned_models)
                ]
                if not stale:
                    continue

                lock = self._get_load_lock()
                async with lock:
                    for name in stale:
                        # Re-check under the lock: model may have been
                        # unloaded already, or bumped by a request that
                        # arrived after the snapshot.
                        if name not in self._loaded:
                            continue
                        info = self._memory_manager_for_model(name).get_model_info(name)
                        if info is None:
                            continue
                        age = time.monotonic() - info.last_used_at
                        if age < threshold:
                            continue
                        logger.info(
                            "Idle eviction: model '%s' idle for %.0fs (threshold=%ds), unloading",
                            name,
                            age,
                            threshold,
                        )
                        try:
                            await self._do_unload(name, reason="idle")
                        except Exception:
                            logger.exception("Idle eviction failed for '%s'", name)
                        # One eviction per tick: yield the lock so other
                        # registry operations can make progress. The next
                        # tick handles any remaining stale models.
                        break

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in idle eviction loop")

    async def start_hot_reload(self) -> None:
        """Start the hot reloader for model config changes.

        The hot reloader watches the models_dir for changes and automatically
        reloads affected models. Only starts if:
        - models_dir is set (not an empty registry)
        - enable_hot_reload was True at init
        - models_dir is a local path (not cloud URLs)
        """
        if not self._enable_hot_reload:
            logger.debug("Hot reload disabled")
            return

        if self._models_dir is None:
            logger.debug("No models_dir, skipping hot reload")
            return

        # Don't watch cloud URLs (s3://, gs://, abfs(s)://)
        if is_cloud_path(self._models_dir):
            logger.debug("Cloud models_dir, skipping hot reload (not supported)")
            return

        if self._hot_reloader is not None:
            logger.warning("Hot reloader already running")
            return

        self._hot_reloader = HotReloader(
            registry=self,
            models_dir=self._models_dir,
            device=self._device,
        )
        await self._hot_reloader.start()

    async def stop_hot_reload(self) -> None:
        """Stop the hot reloader."""
        if self._hot_reloader is None:
            return

        await self._hot_reloader.stop()
        self._hot_reloader = None

    async def _memory_monitor_loop(self) -> None:
        """Background task that monitors memory pressure and evicts LRU models."""
        while self._monitor_running:
            try:
                await asyncio.sleep(self._memory_config.memory_check_interval_s)

                # Quick check without lock
                pressured_managers = [manager for manager in self._memory_managers.values() if manager.check_pressure()]
                if not pressured_managers:
                    continue

                # Pressure detected - acquire lock and evict
                lock = self._get_load_lock()
                async with lock:
                    for manager in pressured_managers:
                        # Re-check under lock (may have resolved)
                        while manager.check_pressure():
                            # Don't evict if only 1 model loaded - would cause immediate reload
                            if manager.loaded_model_count <= 1:
                                logger.debug(
                                    "Memory pressure detected on %s but only %d model loaded, skipping eviction",
                                    manager.device,
                                    manager.loaded_model_count,
                                )
                                break

                            lru_model = manager.get_lru_model(exclude=self._pinned_models)
                            if lru_model is None:
                                break

                            if lru_model not in self._loaded:
                                # Belt-and-braces ghost guard (see #1600): drop a
                                # stale MemoryManager entry with no ``_loaded``
                                # model instead of spinning on a no-op _do_unload.
                                logger.warning(
                                    "Memory monitor found ghost '%s' (not loaded); dropping stale entry",
                                    lru_model,
                                )
                                manager.unregister_model(lru_model)
                                continue

                            stats = manager.get_stats()
                            logger.info(
                                "Memory monitor: %s pressure %.1f%% > %.1f%% threshold, evicting LRU model '%s'",
                                manager.device,
                                stats.usage_ratio * 100,
                                manager.pressure_threshold_pct,
                                lru_model,
                            )
                            await self._do_unload(lru_model, reason="memory_pressure")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in memory monitor loop")

    # -------------------------------------------------------------------------
    # LoRA Management Methods
    # -------------------------------------------------------------------------

    def is_lora_loaded(self, model: str, lora: str) -> bool:
        """Check if a LoRA adapter is loaded for a model.

        Args:
            model: Model name.
            lora: LoRA adapter name/path.

        Returns:
            True if the LoRA is loaded and not currently loading.
        """
        loaded = self._loaded.get(model)
        if loaded is None:
            return False

        lora_state = loaded.loras.get(lora)
        if lora_state is None:
            return False

        # LoRA exists and is not in loading state
        return not lora_state.loading

    def is_lora_loading(self, model: str, lora: str) -> bool:
        """Check if a LoRA adapter is currently being loaded.

        Args:
            model: Model name.
            lora: LoRA adapter name/path.

        Returns:
            True if the LoRA is in the process of loading.
        """
        loaded = self._loaded.get(model)
        if loaded is None:
            return False

        lora_state = loaded.loras.get(lora)
        if lora_state is None:
            return False

        return lora_state.loading

    async def ensure_lora_loaded_async(self, model: str, lora: str) -> tuple[bool, bool]:
        """Ensure a LoRA adapter is loaded, triggering load if needed.

        This method is concurrency-safe - multiple requests for the same LoRA
        will not trigger duplicate loads.

        Args:
            model: Model name.
            lora: LoRA adapter name/path.

        Returns:
            Tuple of (is_ready, is_loading):
            - (True, False): LoRA is loaded and ready
            - (False, True): LoRA is loading, caller should retry
            - (False, False): LoRA load failed or not supported

        Raises:
            KeyError: If model not found or not loaded.
            ValueError: If model doesn't support LoRA.
        """
        from sie_server.core.model_loader import LoadedLora

        if model not in self._configs:
            msg = _ERR_MODEL_NOT_FOUND.format(name=model)
            raise KeyError(msg)

        if model not in self._loaded:
            msg = _ERR_MODEL_NOT_LOADED.format(name=model)
            raise KeyError(msg)

        loaded_model = self._loaded[model]
        adapter = loaded_model.adapter

        # One LoRA capability handle for the whole load path, in place of the
        # scattered supports_lora() / supports_hot_lora_reload() flag-checks.
        lora_cap = adapter.lora_capability()
        if lora_cap is None:
            msg = f"Model '{model}' does not support LoRA"
            raise ValueError(msg)

        # Get or create LoRA lock for this model
        lora_lock = loaded_model.get_lora_lock()

        async with lora_lock:
            # Check if already loaded
            lora_state = loaded_model.loras.get(lora)
            if lora_state is not None:
                if lora_state.loading:
                    # Still loading - return loading status
                    return (False, True)
                # Already loaded
                # Touch to update LRU (move to end of OrderedDict)
                loaded_model.loras.move_to_end(lora)
                return (True, False)

            # Not loaded - start loading
            logger.info("Starting LoRA load for model '%s': %s", model, lora)

            # Create placeholder with loading=True
            loaded_model.loras[lora] = LoadedLora(
                adapter_id=lora,
                loading=True,
            )

            # LRU eviction if needed (evict oldest LoRA)
            while len(loaded_model.loras) > loaded_model.max_loras:
                oldest_lora = next(iter(loaded_model.loras))
                if oldest_lora == lora:
                    # Don't evict the one we're loading
                    break
                logger.info(
                    "LoRA LRU eviction for model '%s': evicting '%s' (max=%d)",
                    model,
                    oldest_lora,
                    loaded_model.max_loras,
                )
                try:
                    lora_cap.unload_lora(oldest_lora)
                except Exception:
                    logger.exception("Error unloading LoRA '%s'", oldest_lora)
                del loaded_model.loras[oldest_lora]

        # Load outside the lock if adapter supports hot reload
        # (PEFT can load while other inference continues)
        if lora_cap.supports_hot_lora_reload():
            # Non-blocking load in background
            task = asyncio.create_task(
                self._load_lora_background(model, lora),
                name=f"lora-load-{model}-{lora}",
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return (False, True)

        # Blocking load (SGLang) - load synchronously
        await self._load_lora_blocking(model, lora)
        return (self.is_lora_loaded(model, lora), False)

    async def _load_lora_background(self, model: str, lora: str) -> None:
        """Load a LoRA adapter in the background (for hot-reload adapters).

        Args:
            model: Model name.
            lora: LoRA adapter name/path.
        """
        loop = asyncio.get_running_loop()

        try:
            loaded_model = self._loaded.get(model)
            if loaded_model is None:
                logger.warning("Model '%s' unloaded during LoRA load", model)
                return

            adapter = loaded_model.adapter
            # Config-pinned revision (#2113): an on-demand load of a
            # profile-declared LoRA (e.g. a reload after LRU eviction) must
            # honor the same pin the preload used, or eviction would silently
            # reintroduce the drift the pin exists to prevent.
            revision = loaded_model.config.lora_revisions().get(lora)

            # Run blocking load in thread pool
            def _do_load() -> int:
                return adapter.load_lora(lora, revision=revision)

            memory_bytes = await loop.run_in_executor(None, _do_load)

            # Update LoRA state
            lora_lock = loaded_model.get_lora_lock()
            async with lora_lock:
                lora_state = loaded_model.loras.get(lora)
                if lora_state is not None:
                    lora_state.loading = False
                    lora_state.memory_bytes = memory_bytes
                    logger.info(
                        "LoRA '%s' loaded for model '%s' (%.2f MB)",
                        lora,
                        model,
                        memory_bytes / 1024 / 1024,
                    )

        except Exception:
            logger.exception("Error loading LoRA '%s' for model '%s'", lora, model)
            # Remove failed LoRA from tracking
            loaded_model = self._loaded.get(model)
            if loaded_model:
                loaded_model.loras.pop(lora, None)

    async def _load_lora_blocking(self, model: str, lora: str) -> None:
        """Load a LoRA adapter with blocking semantics (for SGLang).

        For adapters that don't support hot reload, we block until loading
        completes. This ensures the LoRA is ready when we return.

        Args:
            model: Model name.
            lora: LoRA adapter name/path.
        """
        loop = asyncio.get_running_loop()

        try:
            loaded_model = self._loaded.get(model)
            if loaded_model is None:
                return

            adapter = loaded_model.adapter
            # Config-pinned revision (#2113): an on-demand load of a
            # profile-declared LoRA (e.g. a reload after LRU eviction) must
            # honor the same pin the preload used, or eviction would silently
            # reintroduce the drift the pin exists to prevent.
            revision = loaded_model.config.lora_revisions().get(lora)

            # Run blocking load in thread pool
            def _do_load() -> int:
                return adapter.load_lora(lora, revision=revision)

            memory_bytes = await loop.run_in_executor(None, _do_load)

            # Update LoRA state
            lora_lock = loaded_model.get_lora_lock()
            async with lora_lock:
                lora_state = loaded_model.loras.get(lora)
                if lora_state is not None:
                    lora_state.loading = False
                    lora_state.memory_bytes = memory_bytes
                    logger.info(
                        "LoRA '%s' loaded for model '%s' (%.2f MB)",
                        lora,
                        model,
                        memory_bytes / 1024 / 1024,
                    )

        except Exception:
            logger.exception("Error loading LoRA '%s' for model '%s'", lora, model)
            # Remove failed LoRA from tracking
            loaded_model = self._loaded.get(model)
            if loaded_model:
                loaded_model.loras.pop(lora, None)
