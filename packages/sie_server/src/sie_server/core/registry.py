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
import logging
import time
from pathlib import Path
from typing import Any

from sie_sdk.storage import is_cloud_path

from sie_server.adapters.base import ModelAdapter
from sie_server.config.engine import EngineConfig
from sie_server.config.model import ModelConfig
from sie_server.core.disk_cache import DiskCacheConfig, ModelDiskCacheManager
from sie_server.core.hot_reload import HotReloader
from sie_server.core.loader import load_model_configs
from sie_server.core.memory import MemoryConfig, MemoryManager
from sie_server.core.model_loader import DEFAULT_MAX_LORAS, LoadedModel, ModelLoader
from sie_server.core.postprocessor_registry import PostprocessorRegistry
from sie_server.core.preprocessor_registry import PreprocessorRegistry
from sie_server.core.worker import ModelWorker

logger = logging.getLogger(__name__)

# Error messages
_ERR_MODEL_NOT_FOUND = "Model '{name}' not found in registry"
_ERR_MODEL_NOT_LOADED = "Model '{name}' is not loaded"
_ERR_MODEL_ALREADY_LOADED = "Model '{name}' is already loaded"


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
        engine_config: EngineConfig | None = None,
        enable_hot_reload: bool = True,
    ) -> None:
        """Initialize the registry.

        Args:
            models_dir: Path to models directory (local path, s3://, or gs://).
                       If None, registry starts empty and configs must be added manually.
            memory_config: Configuration for memory management. If None, uses defaults.
            drain_timeout_s: Timeout in seconds to wait for worker drain before unload.
            model_filter: Optional list of model names to include. If None, all models are available.
            device: Default device for memory tracking (e.g., "cuda:0", "mps", "cpu").
            enable_hot_reload: Whether to enable hot reload of model configs when models_dir is set.
        """
        # Store as string to handle cloud URLs
        self._models_dir: str | None = str(models_dir) if models_dir is not None else None
        self._model_filter = set(model_filter) if model_filter else None
        self._device = device
        self._enable_hot_reload = enable_hot_reload
        self._configs: dict[str, ModelConfig] = {}
        self._model_dirs: dict[str, Path] = {}
        self._loaded: dict[str, LoadedModel] = {}
        preprocessor_workers = engine_config.preprocessor_workers if engine_config else None
        self._preprocessor_registry = PreprocessorRegistry(max_workers=preprocessor_workers)
        # Share CPU pool between preprocessor and postprocessor registries
        self._postprocessor_registry = PostprocessorRegistry(self._preprocessor_registry._executor)
        self._memory_config = memory_config or MemoryConfig()
        self._memory_manager = MemoryManager(device=device, config=self._memory_config)
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
                )
            )
            logger.info(
                "Disk cache manager enabled (threshold=%.0f%%, cache_dir=%s)",
                threshold * 100,
                cache_config.local_cache,
            )

        # Concurrency-safe loading
        self._load_lock: asyncio.Lock | None = None  # Created lazily on first use
        self._loading: set[str] = set()  # Models currently being loaded
        self._unloading: set[str] = set()  # Models currently being unloaded

        # Background memory monitor
        self._monitor_task: asyncio.Task[None] | None = None
        self._monitor_running = False

        # Background fire-and-forget tasks (prevents GC collection)
        self._background_tasks: set[asyncio.Task[None]] = set()

        # Hot reloader (created lazily when started)
        self._hot_reloader: HotReloader | None = None

        if self._models_dir is not None:
            self._load_configs_from_dir(self._models_dir)
            logger.info("Initializing model registry from %s", self._models_dir)
            logger.info("Found %d models", len(self._configs))
        else:
            logger.info("No models directory specified, starting with empty registry")

        # Model loader - handles the actual loading workflow
        self._loader = ModelLoader(
            preprocessor_registry=self._preprocessor_registry,
            postprocessor_registry=self._postprocessor_registry,
            all_configs=self._configs,
            default_compute_precision=engine_config.default_compute_precision if engine_config else "float16",
            attention_backend=engine_config.attention_backend if engine_config else "auto",
            max_batch_requests=engine_config.max_batch_requests if engine_config else None,
            max_batch_wait_ms=engine_config.max_batch_wait_ms if engine_config else None,
            max_queue_size=engine_config.max_concurrent_requests if engine_config else None,
            instrumentation=engine_config.instrumentation if engine_config else False,
            max_loras_per_model=engine_config.max_loras_per_model if engine_config else DEFAULT_MAX_LORAS,
            disk_cache_manager=self._disk_cache_manager,
        )

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

        if new_names:
            logger.info("Discovered %d new models: %s", len(new_names), ", ".join(new_names))
            # Update loader's config reference
            self._loader.update_configs(self._configs)

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
        return self._memory_manager

    @property
    def device(self) -> str:
        """Return the default device for model loading."""
        return self._device

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

    def _get_load_lock(self) -> asyncio.Lock:
        """Get or create the load lock (must be called from async context)."""
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        return self._load_lock

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

        # Update LRU tracking - this model was recently used
        self._memory_manager.touch(name)

        return self._loaded[name].adapter

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

        # Instantiate the adapter with device-aware fallback selection
        adapter = self._loader.instantiate_adapter(name, config, model_dir, device)

        # Try to load onto device, with OOM retry
        try:
            loaded = self._loader.load_and_register(name, device, adapter, config)
        except RuntimeError as e:
            if not self._is_oom_error(e):
                raise

            # OOM - try to evict LRU model and retry
            lru_model = self._memory_manager.get_lru_model()
            if lru_model is None:
                logger.error("OOM loading '%s' but no models to evict", name)
                raise

            logger.warning(
                "OOM loading '%s', evicting LRU model '%s' and retrying",
                name,
                lru_model,
            )
            self.unload(lru_model)

            # Retry once after eviction - need new adapter instance
            adapter = self._loader.instantiate_adapter(name, config, model_dir, device)
            loaded = self._loader.load_and_register(name, device, adapter, config)

        # Track loaded state
        self._loaded[name] = loaded

        # Register with memory manager for LRU tracking
        self._memory_manager.register_model(name, estimated_bytes=loaded.memory_bytes)

        return loaded.adapter

    async def load_async(self, name: str, device: str) -> ModelAdapter:
        """Load a model onto a device (async, concurrency-safe).

        This method:
        - Serializes all load/unload operations via an async lock
        - Proactively evicts LRU models if memory pressure is high before loading
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
                self._memory_manager.touch(name)
                return self._loaded[name].adapter

            # Check if model is being unloaded - caller should retry
            if name in self._unloading:
                msg = f"Model '{name}' is currently being unloaded"
                raise RuntimeError(msg)

            # Pre-load eviction: evict LRU models until below pressure threshold
            while self._memory_manager.check_pressure():
                lru_model = self._memory_manager.get_lru_model()
                if lru_model is None:
                    break  # No models to evict, proceed with load attempt

                stats = self._memory_manager.get_stats()
                logger.info(
                    "Pre-load eviction: memory %.1f%% > %.1f%% threshold, evicting '%s' before loading '%s'",
                    stats.usage_ratio * 100,
                    self._memory_manager.pressure_threshold_pct,
                    lru_model,
                    name,
                )
                await self._do_unload(lru_model)

            # Mark as loading before starting (visible to WebSocket status)
            self._loading.add(name)
            load_start = time.monotonic()

            try:
                config = self._configs[name]
                model_dir = self._model_dirs.get(name, Path())

                # Instantiate adapter (in thread pool) with device-aware fallback selection
                adapter = await self._loader.instantiate_adapter_async(name, config, model_dir, device)

                try:
                    # Load onto device (loader handles main thread vs executor)
                    loaded = await self._loader.load_and_register_async(name, device, adapter, config)
                except RuntimeError as e:
                    if not self._is_oom_error(e):
                        raise

                    # OOM despite pre-load eviction: evict LRU and retry once
                    lru_model = self._memory_manager.get_lru_model()
                    if lru_model is None:
                        logger.error("OOM loading '%s' but no models to evict", name)
                        raise

                    logger.warning(
                        "OOM loading '%s' despite pre-eviction, evicting '%s' and retrying",
                        name,
                        lru_model,
                    )
                    await self._do_unload(lru_model)

                    # Retry once after eviction - need new adapter instance
                    adapter = await self._loader.instantiate_adapter_async(name, config, model_dir, device)
                    loaded = await self._loader.load_and_register_async(name, device, adapter, config)

                # Track loaded state
                self._loaded[name] = loaded

                # Register with memory manager for LRU tracking
                self._memory_manager.register_model(name, estimated_bytes=loaded.memory_bytes)

                load_duration = time.monotonic() - load_start
                if load_duration > 300:
                    logger.warning(
                        "Model '%s' took %.0fs to load (>300s) — may indicate a gated model "
                        "missing HF_TOKEN or network issues",
                        name,
                        load_duration,
                    )

                return loaded.adapter
            finally:
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

        Args:
            name: Model name.
            device: Device string (e.g., "cuda:0", "cpu").
        """
        try:
            await self.load_async(name, device)
            logger.info("Background model load completed: %s", name)
        except Exception:
            logger.exception("Background model load failed: %s", name)
        finally:
            # Always remove from _loading when background task completes.
            # This handles the case where load_async returned early (model already
            # loaded by another task) without entering its own finally block.
            # For normal loads, load_async already discarded, so this is a no-op.
            self._loading.discard(name)

    def unload(self, name: str) -> None:
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

        # Adapter.unload() handles gc.collect + empty_cache
        loaded.adapter.unload()

        # Unregister tokenizer, preprocessor, and clear metrics
        self._loader.unregister(name, device)

        # Unregister from memory manager
        self._memory_manager.unregister_model(name)

        logger.info("Model '%s' unloaded", name)

    async def _do_unload(self, name: str) -> None:
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

            # Adapter.unload() handles gc.collect + empty_cache
            loaded.adapter.unload()

            # Unregister tokenizer, preprocessor, and clear metrics
            self._loader.unregister(name, device)

            # Unregister from memory manager
            self._memory_manager.unregister_model(name)

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

            await self._do_unload(name)

    def unload_all(self) -> None:
        """Unload all loaded models (sync version, for non-async contexts)."""
        for name in list(self._loaded.keys()):
            self.unload(name)

    async def unload_all_async(self) -> None:
        """Unload all loaded models (async, concurrency-safe)."""
        lock = self._get_load_lock()
        async with lock:
            for name in list(self._loaded.keys()):
                await self._do_unload(name)

    def add_config(self, config: ModelConfig, model_dir: Path | None = None) -> None:
        """Add a model config to the registry.

        Useful for programmatic config creation without files.

        Args:
            config: The model configuration.
            model_dir: Optional directory for custom adapter resolution.
        """
        self._configs[config.sie_id] = config
        if model_dir is not None:
            self._model_dirs[config.sie_id] = model_dir

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

        If already running, this is a no-op.

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
        if name not in self._loaded:
            msg = _ERR_MODEL_NOT_LOADED.format(name=name)
            raise KeyError(msg)

        worker = self._loaded[name].worker
        if worker is None:
            msg = f"Worker not available for model '{name}'"
            raise RuntimeError(msg)

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

        Detects OOM errors from PyTorch (CUDA/MPS) and generic memory errors.

        Args:
            error: The RuntimeError to check.

        Returns:
            True if this appears to be an OOM error.
        """
        error_str = str(error).lower()
        oom_indicators = [
            "out of memory",
            "cuda out of memory",
            "mps backend out of memory",
            "cannot allocate memory",
            "failed to allocate",
            "oom",
        ]
        return any(indicator in error_str for indicator in oom_indicators)

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

        # Don't watch cloud URLs (s3://, gs://)
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
                if not self._memory_manager.check_pressure():
                    continue

                # Pressure detected - acquire lock and evict
                lock = self._get_load_lock()
                async with lock:
                    # Re-check under lock (may have resolved)
                    while self._memory_manager.check_pressure():
                        # Don't evict if only 1 model loaded - would cause immediate reload
                        if self._memory_manager.loaded_model_count <= 1:
                            logger.debug(
                                "Memory pressure detected but only %d model loaded, skipping eviction",
                                self._memory_manager.loaded_model_count,
                            )
                            break

                        lru_model = self._memory_manager.get_lru_model()
                        if lru_model is None:
                            break

                        stats = self._memory_manager.get_stats()
                        logger.info(
                            "Memory monitor: pressure %.1f%% > %.1f%% threshold, evicting LRU model '%s'",
                            stats.usage_ratio * 100,
                            self._memory_manager.pressure_threshold_pct,
                            lru_model,
                        )
                        await self._do_unload(lru_model)

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

        # Check if adapter supports LoRA
        if not adapter.supports_lora():
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
                    adapter.unload_lora(oldest_lora)
                except Exception:
                    logger.exception("Error unloading LoRA '%s'", oldest_lora)
                del loaded_model.loras[oldest_lora]

        # Load outside the lock if adapter supports hot reload
        # (PEFT can load while other inference continues)
        if adapter.supports_hot_lora_reload():
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
        import asyncio

        loop = asyncio.get_running_loop()

        try:
            loaded_model = self._loaded.get(model)
            if loaded_model is None:
                logger.warning("Model '%s' unloaded during LoRA load", model)
                return

            adapter = loaded_model.adapter

            # Run blocking load in thread pool
            def _do_load() -> int:
                return adapter.load_lora(lora)

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
        import asyncio

        loop = asyncio.get_running_loop()

        try:
            loaded_model = self._loaded.get(model)
            if loaded_model is None:
                return

            adapter = loaded_model.adapter

            # Run blocking load in thread pool
            def _do_load() -> int:
                return adapter.load_lora(lora)

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
