from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sie_server.adapters.base import ModelAdapter
from sie_server.config.model import ModelConfig
from sie_server.core.inference import AttentionBackend, ComputePrecision
from sie_server.core.loader import load_adapter
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.observability.metrics import set_model_loaded, set_model_memory

if TYPE_CHECKING:
    from sie_server.core.disk_cache import ModelDiskCacheManager
    from sie_server.core.postprocessor_registry import PostprocessorRegistry
    from sie_server.core.preprocessor_registry import PreprocessorRegistry

logger = logging.getLogger(__name__)

# Default maximum LoRAs per model before LRU eviction
DEFAULT_MAX_LORAS = 10


@dataclass
class LoadedLora:
    """Container for a loaded LoRA adapter's state."""

    adapter_id: str  # HuggingFace path or local path
    memory_bytes: int = 0  # Memory footprint of this LoRA
    peft_model: Any | None = None  # PeftModel instance (PEFT adapters only)
    loading: bool = False  # True while async loading is in progress


@dataclass
class LoadedModel:
    """Container for a loaded model's state."""

    config: ModelConfig
    adapter: ModelAdapter
    device: str
    worker: ModelWorker | None = None
    memory_bytes: int = 0  # Base model memory (excludes LoRAs)

    # LoRA management - OrderedDict maintains LRU order (oldest first)
    loras: OrderedDict[str, LoadedLora] = field(default_factory=OrderedDict)
    max_loras: int = DEFAULT_MAX_LORAS

    # Lock for concurrent LoRA loading (created on first use)
    _lora_lock: asyncio.Lock | None = field(default=None, init=False, repr=False)

    @property
    def total_memory_bytes(self) -> int:
        """Total memory including base model and all loaded LoRAs."""
        lora_memory = sum(lora.memory_bytes for lora in self.loras.values())
        return self.memory_bytes + lora_memory

    def get_lora_lock(self) -> asyncio.Lock:
        """Get or create the LoRA loading lock."""
        if self._lora_lock is None:
            self._lora_lock = asyncio.Lock()
        return self._lora_lock


class ModelLoader:
    """Handles model loading workflow.

    Responsibilities:
    - Ensure weights are in local cache (from cluster cache or HF Hub)
    - Instantiate adapters (check dependencies, create adapter instance)
    - Load onto device (choosing main thread vs executor based on adapter)
    - Register tokenizers and preprocessors
    - Create workers

    This class does NOT handle:
    - Tracking loaded models (that's ModelRegistry)
    - Memory management (that's MemoryManager)
    - Async concurrency/locks (that's ModelRegistry)
    """

    def __init__(
        self,
        preprocessor_registry: PreprocessorRegistry,
        postprocessor_registry: PostprocessorRegistry,
        all_configs: dict[str, ModelConfig],
        *,
        default_compute_precision: ComputePrecision = "float16",
        attention_backend: AttentionBackend = "auto",
        max_batch_requests: int | None = None,
        max_batch_wait_ms: int | None = None,
        max_queue_size: int | None = None,
        instrumentation: bool = False,
        max_loras_per_model: int = DEFAULT_MAX_LORAS,
        disk_cache_manager: ModelDiskCacheManager | None = None,
    ) -> None:
        """Initialize the model loader.

        Args:
            preprocessor_registry: Registry for preprocessors (text and image).
            postprocessor_registry: Registry for postprocessors (output transforms).
            all_configs: All model configs (for adapter resolution).
            disk_cache_manager: Optional disk cache manager for LRU eviction.
        """
        self._preprocessor_registry = preprocessor_registry
        self._postprocessor_registry = postprocessor_registry
        self._all_configs = all_configs
        self._default_compute_precision = default_compute_precision
        self._attention_backend = attention_backend
        self._max_batch_requests = max_batch_requests
        self._max_batch_wait_ms = max_batch_wait_ms
        self._max_queue_size = max_queue_size
        self._instrumentation = instrumentation
        self._max_loras_per_model = max_loras_per_model
        self._disk_cache = disk_cache_manager
        self._load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-load")

    def update_configs(self, configs: dict[str, ModelConfig]) -> None:
        """Update the config reference (called after rescan)."""
        self._all_configs = configs

    def instantiate_adapter(
        self,
        name: str,
        config: ModelConfig,
        model_dir: Path,
        device: str,
    ) -> ModelAdapter:
        """Instantiate an adapter with device-aware fallback selection.

        This is separated from loading so we can check requires_main_thread
        before deciding how to run the load.

        Also ensures model weights are in local cache (from cluster cache if needed).

        Note: Dependency checking is done earlier by ModelRegistry._check_model_loadable()
        before this method is called. This ensures errors surface synchronously.

        Args:
            name: Model name (for logging).
            config: Model configuration.
            model_dir: Path to model directory.
            device: Device string for device-aware adapter selection (e.g., "cuda:0", "mps", "cpu").

        Returns:
            The unloaded adapter instance (may be fallback adapter for non-CUDA devices).

        Raises:
            RuntimeError: If model not cached and HF fallback is disabled.
        """
        logger.info("Instantiating adapter for model '%s' (device=%s)", name, device)

        # Ensure model weights are cached (cluster cache -> local cache)
        self._ensure_weights_cached(config)

        # Instantiate the adapter (does not load weights yet)
        return load_adapter(
            config,
            model_dir,
            device=device,
            default_compute_precision=self._default_compute_precision,
            attention_backend=self._attention_backend,
        )

    def _ensure_weights_cached(self, config: ModelConfig) -> None:
        """Ensure model weights are available in local cache.

        Implements the caching hierarchy:
        1. Check local cache - return if found
        2. Download from cluster cache if configured
        3. Download from HF Hub if fallback enabled
        4. Raise error if no cache and fallback disabled

        Also handles disk cache LRU eviction before downloading if needed.

        Args:
            config: Model configuration.

        Raises:
            GatedModelError: If model is gated and authentication fails.
            RuntimeError: If model not cached and HF fallback disabled.
        """
        from sie_sdk.cache import ensure_model_cached, get_cache_config

        # Only applies to HF models (not local weights_path)
        model_id = config.hf_id
        if model_id is None:
            return  # Local weights, no caching needed

        # Check disk pressure and evict LRU models if needed before download
        if self._disk_cache is not None:
            evicted = self._disk_cache.ensure_space_before_download(model_id)
            if evicted:
                logger.info(
                    "Pre-download disk eviction: freed %d model(s): %s",
                    len(evicted),
                    evicted,
                )

        cache_config = get_cache_config()

        # Ensure model is cached (downloads if needed)
        # This now handles the full 3-tier hierarchy internally
        cached_path = ensure_model_cached(model_id, cache_config)

        # Update access time for LRU tracking
        if self._disk_cache is not None:
            self._disk_cache.touch(model_id)

        logger.debug("Model %s available at %s", model_id, cached_path)

    async def instantiate_adapter_async(
        self,
        name: str,
        config: ModelConfig,
        model_dir: Path,
        device: str,
    ) -> ModelAdapter:
        """Instantiate adapter in thread pool (async version).

        Args:
            name: Model name.
            config: Model configuration.
            model_dir: Path to model directory.
            device: Device string for device-aware adapter selection.

        Returns:
            The unloaded adapter instance (may be fallback adapter for non-CUDA devices).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._load_executor,
            self.instantiate_adapter,
            name,
            config,
            model_dir,
            device,
        )

    def load_and_register(
        self,
        name: str,
        device: str,
        adapter: ModelAdapter,
        config: ModelConfig,
    ) -> LoadedModel:
        """Load adapter onto device and register tokenizer/preprocessor.

        This is the sync loading path - blocks the calling thread.

        Args:
            name: Model name.
            device: Device string (e.g., "cuda:0", "cpu").
            adapter: The adapter to load.
            config: Model configuration.

        Returns:
            LoadedModel containing the loaded state.
        """
        logger.info("Loading model '%s' onto %s", name, device)
        adapter.load(device)
        return self._finish_load(name, device, adapter, config)

    async def load_and_register_async(
        self,
        name: str,
        device: str,
        adapter: ModelAdapter,
        config: ModelConfig,
    ) -> LoadedModel:
        """Load adapter onto device (async, choosing main thread vs executor).

        Args:
            name: Model name.
            device: Device string.
            adapter: The adapter to load.
            config: Model configuration.

        Returns:
            LoadedModel containing the loaded state.
        """
        if adapter.requires_main_thread:
            # SGLang and similar adapters need main thread for signal handlers
            logger.info(
                "Loading '%s' in main thread (adapter.requires_main_thread=True)",
                name,
            )
            return self._load_main_thread(name, device, adapter, config)

        # Normal adapters can run in thread pool
        return await self._load_in_executor(name, device, adapter, config)

    async def _load_in_executor(
        self,
        name: str,
        device: str,
        adapter: ModelAdapter,
        config: ModelConfig,
    ) -> LoadedModel:
        """Load adapter in thread pool (for normal adapters).

        Args:
            name: Model name.
            device: Device string.
            adapter: The adapter to load.
            config: Model configuration.

        Returns:
            LoadedModel containing the loaded state.
        """
        loop = asyncio.get_running_loop()

        def _load_sync() -> LoadedModel:
            logger.info("Loading model '%s' onto %s", name, device)
            adapter.load(device)
            return self._finish_load(name, device, adapter, config)

        return await loop.run_in_executor(self._load_executor, _load_sync)

    def _load_main_thread(
        self,
        name: str,
        device: str,
        adapter: ModelAdapter,
        config: ModelConfig,
    ) -> LoadedModel:
        """Load adapter in main thread (for adapters with signal handlers).

        This blocks the event loop but is required for adapters like SGLang
        that use Python signal handlers which only work in the main thread.

        Args:
            name: Model name.
            device: Device string.
            adapter: The adapter to load.
            config: Model configuration.

        Returns:
            LoadedModel containing the loaded state.
        """
        logger.info("Loading model '%s' onto %s (main thread)", name, device)
        adapter.load(device)
        return self._finish_load(name, device, adapter, config)

    def _finish_load(
        self,
        name: str,
        device: str,
        adapter: ModelAdapter,
        config: ModelConfig,
    ) -> LoadedModel:
        """Complete loading - register preprocessor, create worker.

        Args:
            name: Model name.
            device: Device string.
            adapter: The loaded adapter.
            config: Model configuration.

        Returns:
            LoadedModel containing the loaded state.
        """
        # Get preprocessor from adapter - all adapters implement get_preprocessor()
        preprocessor = adapter.get_preprocessor()

        # Register the preprocessor based on its modality
        if preprocessor is not None:
            modality = getattr(preprocessor, "modality", None)
            if modality == "text":
                self._preprocessor_registry._register(name, preprocessor)
                logger.info("Registered text preprocessor for model '%s'", name)
            elif modality == "image":
                self._preprocessor_registry.register_image(name, preprocessor)
                logger.info("Registered image preprocessor for model '%s'", name)

        # Register postprocessors if adapter provides them (Phase 5)
        if hasattr(adapter, "get_postprocessors"):
            postprocessors = adapter.get_postprocessors()
            if postprocessors:
                self._postprocessor_registry.register(name, postprocessors)
                logger.info(
                    "Registered postprocessors for model '%s': %s",
                    name,
                    list(postprocessors.keys()),
                )

        # Get actual memory footprint from the adapter
        memory_bytes = adapter.memory_footprint()

        # Create worker for the adapter with postprocessor support
        resolved = config.resolve_profile("default")
        worker_config = WorkerConfig(
            max_batch_tokens=resolved.max_batch_tokens,
            max_batch_requests=self._max_batch_requests or WorkerConfig().max_batch_requests,
            max_batch_wait_ms=self._max_batch_wait_ms or WorkerConfig().max_batch_wait_ms,
            max_queue_size=self._max_queue_size or WorkerConfig().max_queue_size,
            instrumentation=self._instrumentation,
        )
        worker = ModelWorker(
            adapter,
            worker_config,
            model_name=name,
            postprocessor_registry=self._postprocessor_registry,
        )

        # Record Prometheus metrics
        set_model_loaded(name, device, loaded=True)
        set_model_memory(name, device, memory_bytes)

        # Create LoadedModel instance
        loaded_model = LoadedModel(
            config=config,
            adapter=adapter,
            device=device,
            worker=worker,
            memory_bytes=memory_bytes,
            max_loras=self._max_loras_per_model,
        )

        # Preload profile LoRAs if adapter supports LoRA
        if adapter.supports_lora():
            profile_loras = self._collect_profile_loras(config)
            if profile_loras:
                logger.info(
                    "Preloading %d profile LoRAs for model '%s': %s",
                    len(profile_loras),
                    name,
                    list(profile_loras),
                )
                for lora_path in profile_loras:
                    try:
                        lora_memory = adapter.load_lora(lora_path)
                        loaded_model.loras[lora_path] = LoadedLora(
                            adapter_id=lora_path,
                            memory_bytes=lora_memory,
                            loading=False,
                        )
                        logger.info(
                            "Preloaded LoRA '%s' for model '%s' (%.2f MB)",
                            lora_path,
                            name,
                            lora_memory / 1024 / 1024,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to preload LoRA '%s' for model '%s'",
                            lora_path,
                            name,
                        )

        logger.info("Model '%s' loaded successfully", name)
        return loaded_model

    def _collect_profile_loras(self, config: ModelConfig) -> set[str]:
        """Collect all LoRA paths from model profiles.

        Profiles can specify a `lora_id` in `adapter_options.runtime` that points
        to a HuggingFace path or local path for a LoRA adapter. This method
        collects all unique LoRA paths across all profiles for preloading.

        Args:
            config: Model configuration.

        Returns:
            Set of unique LoRA paths from profiles.
        """
        loras: set[str] = set()

        # Collect LoRAs from profiles (lora_id lives in adapter_options.runtime)
        for profile_name, profile_config in config.profiles.items():
            lora_id = profile_config.adapter_options.runtime.get("lora_id")
            if lora_id:
                loras.add(lora_id)
                logger.debug(
                    "Found LoRA '%s' in profile '%s'",
                    lora_id,
                    profile_name,
                )

        return loras

    def unregister(self, name: str, device: str) -> None:
        """Unregister preprocessors, postprocessors, and clear metrics.

        Args:
            name: Model name.
            device: Device string (for metrics).
        """
        # Clear Prometheus metrics
        set_model_loaded(name, device, loaded=False)
        set_model_memory(name, device, 0)

        # Unregister all preprocessors (text and image)
        self._preprocessor_registry.unregister(name)

        # Unregister postprocessors
        self._postprocessor_registry.unregister(name)

        logger.debug("Unregistered pre/postprocessors for model '%s'", name)

    def shutdown(self) -> None:
        """Shutdown the loader's thread pool."""
        self._load_executor.shutdown(wait=False)
