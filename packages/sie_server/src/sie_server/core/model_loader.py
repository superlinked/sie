from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sie_server.adapters.base import ModelAdapter
from sie_server.config.model import ModelConfig, ProfileAdaptiveBatching, lora_entry_ref
from sie_server.core.inference import AttentionBackend, ComputePrecision
from sie_server.core.load_errors import ModelLoadTimeoutError
from sie_server.core.loader import load_adapter
from sie_server.core.oom import OomRecoveryConfig
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.types import AdaptiveBatchingParams
from sie_server.observability.worker_telemetry import worker_telemetry

if TYPE_CHECKING:
    from sie_server.core.disk_cache import ModelDiskCacheManager
    from sie_server.core.postprocessor_registry import PostprocessorRegistry
    from sie_server.core.preprocessor_registry import PreprocessorRegistry
    from sie_server.core.worker.oom_recovery import RegistryCallbacks

logger = logging.getLogger(__name__)

# Default maximum LoRAs per model before LRU eviction
DEFAULT_MAX_LORAS = 10

# Default total-time budget (seconds) for the post-download portion of a
# model load: adapter instantiation + ``adapter.load(device)`` + warmup.
# The download phase is intentionally NOT bounded by this budget — it is
# bounded only by ``HF_HUB_DOWNLOAD_TIMEOUT`` socket-inactivity stalls so
# users on slow links can still complete legitimate multi-hour downloads.
#
# Override via the ``SIE_MODEL_LOAD_TIMEOUT_S`` env var or the
# ``model_load_timeout_s`` constructor kwarg. ``0`` or a negative value
# disables the timeout entirely.
DEFAULT_MODEL_LOAD_TIMEOUT_S = 600.0
_TIMEOUT_ENV_VAR = "SIE_MODEL_LOAD_TIMEOUT_S"


def _resolve_load_timeout(explicit: float | None) -> float:
    """Resolve effective post-download timeout in seconds.

    Precedence: explicit kwarg > env var > built-in default. Returns ``0.0``
    when disabled (``0`` or negative); callers treat that as "no
    ``wait_for`` wrapping".
    """
    if explicit is not None:
        return max(0.0, float(explicit))
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_MODEL_LOAD_TIMEOUT_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "Invalid %s=%r, falling back to default %ss",
            _TIMEOUT_ENV_VAR,
            raw,
            DEFAULT_MODEL_LOAD_TIMEOUT_S,
        )
        return DEFAULT_MODEL_LOAD_TIMEOUT_S


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
        max_batch_wait_ms: float | None = None,
        coalesce_ms: float | None = None,
        coalesce_ratio: float | None = None,
        max_queue_size: int | None = None,
        instrumentation: bool = False,
        max_loras_per_model: int = DEFAULT_MAX_LORAS,
        disk_cache_manager: ModelDiskCacheManager | None = None,
        adaptive_batching: AdaptiveBatchingParams | None = None,
        oom_recovery: OomRecoveryConfig | None = None,
        registry_callbacks: RegistryCallbacks | None = None,
        model_load_timeout_s: float | None = None,
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
        self._coalesce_ms = coalesce_ms
        self._coalesce_ratio = coalesce_ratio
        self._max_queue_size = max_queue_size
        self._instrumentation = instrumentation
        self._max_loras_per_model = max_loras_per_model
        self._disk_cache = disk_cache_manager
        self._adaptive_batching = adaptive_batching or AdaptiveBatchingParams()
        # Reactive OOM recovery config + sibling-eviction callbacks. Default
        # config has recovery enabled; passing ``None`` here keeps that.
        self._oom_recovery = oom_recovery or OomRecoveryConfig()
        self._registry_callbacks = registry_callbacks
        self._load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-load")
        # Post-download budget (instantiate + adapter.load + warmup). 0.0
        # disables the outer ``wait_for``. Download is bounded separately
        # by ``HF_HUB_DOWNLOAD_TIMEOUT``.
        self._model_load_timeout_s = _resolve_load_timeout(model_load_timeout_s)

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

        IMPORTANT: As of the post-download-timeout refactor, this method
        no longer ensures weights are cached — callers must invoke
        :meth:`ensure_weights_cached` (or :meth:`ensure_weights_cached_async`)
        first. The split exists so the download phase runs without an
        outer total-time budget (slow networks are allowed) while the
        local-only portion runs under ``SIE_MODEL_LOAD_TIMEOUT_S``.

        Note: Dependency checking is done earlier by ``ModelRegistry._check_model_loadable()``
        before this method is called. This ensures errors surface synchronously.

        Args:
            name: Model name (for logging).
            config: Model configuration.
            model_dir: Path to model directory.
            device: Device string for device-aware adapter selection (e.g., "cuda:0", "mps", "cpu").

        Returns:
            The unloaded adapter instance (may be fallback adapter for non-CUDA devices).
        """
        logger.info("Instantiating adapter for model '%s' (device=%s)", name, device)

        # Instantiate the adapter (does not load weights yet). Weight
        # caching must already have been performed by the caller via
        # ``ensure_weights_cached`` — see method docstring.
        return load_adapter(
            config,
            model_dir,
            device=device,
            default_compute_precision=self._default_compute_precision,
            attention_backend=self._attention_backend,
        )

    def ensure_weights_cached(self, config: ModelConfig) -> None:
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

    async def ensure_weights_cached_async(self, name: str, config: ModelConfig) -> None:
        """Async version of :meth:`ensure_weights_cached`.

        Runs the (potentially long) download on the load executor so it
        doesn't block the event loop. Intentionally NOT wrapped in
        ``asyncio.wait_for`` — slow networks are allowed; stalls are
        detected by ``HF_HUB_DOWNLOAD_TIMEOUT`` inside ``huggingface_hub``.

        Args:
            name: Model name (for logging context only).
            config: Model configuration.
        """
        logger.debug("Ensuring weights cached for '%s'", name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._load_executor, self.ensure_weights_cached, config)

    async def instantiate_adapter_async(
        self,
        name: str,
        config: ModelConfig,
        model_dir: Path,
        device: str,
    ) -> ModelAdapter:
        """Instantiate adapter in the load executor with a post-download timeout.

        Bounded by ``SIE_MODEL_LOAD_TIMEOUT_S`` (default 600 s). On timeout
        the underlying thread cannot be killed in Python, so the executor
        is replaced with a fresh single-worker pool — the wedged thread
        leaks until process exit but new loads can proceed.

        Callers MUST have already invoked :meth:`ensure_weights_cached_async`
        for this config; this method assumes weights are on local disk.

        Raises:
            ModelLoadTimeoutError: If instantiation exceeds the configured
                budget. Classified as ``LoadErrorClass.TIMEOUT`` by the
                registry's failure recorder, which applies a 30 s cooldown
                before the next client request can retry.
        """
        return await self._run_with_timeout(
            stage="instantiate",
            name=name,
            func=self.instantiate_adapter,
            args=(name, config, model_dir, device),
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
        _run_load_with_markers(name, device, adapter)
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
        """Load adapter in thread pool with post-download timeout.

        Bounded by ``SIE_MODEL_LOAD_TIMEOUT_S``. The executor thread runs
        ONLY ``_run_load_with_markers`` (weight deserialization + warmup);
        the registry-state side effects (``_finish_load``: pre/postprocessor
        registration, OTel residency event, worker creation, LoRA
        preloading) run on the awaiting coroutine AFTER the future
        resolves cleanly.

        This split is critical for correctness on timeout: when
        ``wait_for`` cancels, the orphaned thread keeps running but can
        only mutate the adapter object itself (which the registry will
        discard along with the LoadedModel that never gets created). It
        cannot register stale preprocessors or publish model residency
        for a model the registry has marked failed.

        Adapters that set ``manages_own_load_timeout`` are also run in this
        executor, but without the generic ``wait_for`` wrapper. That keeps the
        event loop responsive while preserving adapter-owned cleanup semantics
        for subprocess-backed loaders whose startup timeout can terminate the
        child process cleanly.
        """

        def _deserialize_and_warmup() -> None:
            logger.info("Loading model '%s' onto %s", name, device)
            _run_load_with_markers(name, device, adapter)

        if getattr(adapter, "manages_own_load_timeout", False) is True:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._load_executor, _deserialize_and_warmup)
        else:
            await self._run_with_timeout(
                stage="load",
                name=name,
                func=_deserialize_and_warmup,
                args=(),
            )
        # Registry-state side effects (preprocessor/postprocessor
        # registration, metrics, worker creation, LoRA preloading) run
        # OUTSIDE the executor so an orphan thread from a wait_for
        # cancellation cannot corrupt registry state. Safe to call from
        # the awaiting coroutine: only cheap Python work, no I/O.
        return self._finish_load(name, device, adapter, config)

    async def _run_with_timeout(
        self,
        *,
        stage: str,
        name: str,
        func: Any,
        args: tuple[Any, ...],
    ) -> Any:
        """Run ``func(*args)`` in the load executor under the post-download budget.

        Centralises the ``wait_for`` + executor-recreate dance shared by
        :meth:`instantiate_adapter_async` and :meth:`_load_in_executor`. A
        timeout of ``0`` (or negative, via env) disables the wrapper and
        the call runs unbounded — matches the PR convention.

        On timeout:
          1. The asyncio task awaiting the executor future is cancelled
             via ``wait_for``. The underlying Python thread continues to
             run; it cannot be interrupted from outside.
          2. The wedged executor is ``shutdown(wait=False)``'d and replaced
             with a fresh single-worker pool so subsequent loads do not
             queue behind the leaked thread (the registry holds
             ``_load_lock`` while awaiting us, so this happens with the
             registry quiesced).
          3. A structured error is logged.
          4. :class:`ModelLoadTimeoutError` is raised; the registry's
             ``_record_load_failure`` classifies it as ``TIMEOUT``, records
             ``sie.worker.model.load.duration{outcome="timeout"}``, and installs
             a 30 s cooldown.
        """
        loop = asyncio.get_running_loop()
        timeout = self._model_load_timeout_s
        started = time.monotonic()
        fut = loop.run_in_executor(self._load_executor, func, *args)

        # Disabled: no wait_for wrapper, behave exactly like the pre-PR code.
        if timeout <= 0:
            return await fut

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError as exc:
            elapsed = time.monotonic() - started
            logger.error(
                "Model load timeout: model=%s stage=%s elapsed_s=%.1f timeout_s=%.0f",
                name,
                stage,
                elapsed,
                timeout,
            )
            # The thread keeps running with a stale adapter/config; we
            # cannot kill it. Replace the executor so the next load is
            # not queued behind the leaked worker.
            self._recreate_executor()
            raise ModelLoadTimeoutError(model=name, stage=stage, elapsed_s=elapsed, timeout_s=timeout) from exc

    def _recreate_executor(self) -> None:
        """Discard the current load executor and create a fresh one.

        Called on timeout: the in-flight thread cannot be terminated, so
        we orphan it (``shutdown(wait=False)`` returns immediately without
        joining) and bind ``self._load_executor`` to a new single-worker
        pool. The orphan thread continues to consume RAM/GPU until process
        exit; the timeout outcome on `sie.worker.model.load.duration` lets ops
        observe the rate.
        """
        old = self._load_executor
        self._load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-load")
        old.shutdown(wait=False)

    def _load_main_thread(
        self,
        name: str,
        device: str,
        adapter: ModelAdapter,
        config: ModelConfig,
    ) -> LoadedModel:
        """Load adapter in main thread (for adapters with signal handlers).

        This blocks the event loop and is reserved for adapters that truly need
        parent-process main-thread execution.

        Adapter-internal startup timeouts are rewrapped into
        :class:`ModelLoadTimeoutError` so the
        registry's failure classifier buckets them as
        ``LoadErrorClass.TIMEOUT`` (30 s cooldown) rather than
        ``UNKNOWN`` (permanent) — consistent with the executor path.

        Args:
            name: Model name.
            device: Device string.
            adapter: The adapter to load.
            config: Model configuration.

        Returns:
            LoadedModel containing the loaded state.
        """
        logger.info("Loading model '%s' onto %s (main thread)", name, device)
        started = time.monotonic()
        try:
            _run_load_with_markers(name, device, adapter)
        except RuntimeError as exc:
            _raise_if_adapter_startup_timeout(name, started, exc)
            raise
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
        resolved = config.resolve_profile("default")
        postprocessors = adapter.get_postprocessors() or {}
        if resolved.runtime.get("muvera") is not None and "muvera" not in postprocessors:
            raise ValueError(
                f"model profile {name!r} requests MUVERA but adapter "
                f"{type(adapter).__name__!r} did not register a 'muvera' postprocessor"
            )

        # Get preprocessor(s) from adapter - all adapters implement get_preprocessor().
        # Most return a single preprocessor; multi-modal adapters (e.g. NemoColEmbed v1,
        # which needs a text preprocessor for queries AND an image preprocessor for
        # documents) may return a list. Register each by its modality.
        preprocessors = adapter.get_preprocessor()
        if not isinstance(preprocessors, list):
            preprocessors = [preprocessors]
        for preprocessor in preprocessors:
            if preprocessor is None:
                continue
            modality = getattr(preprocessor, "modality", None)
            if modality == "text":
                self._preprocessor_registry.register(name, preprocessor)
                logger.info("Registered text preprocessor for model '%s'", name)
            elif modality == "image":
                self._preprocessor_registry.register_image(name, preprocessor)
                logger.info("Registered image preprocessor for model '%s'", name)
            elif modality == "audio":
                self._preprocessor_registry.register(name, preprocessor)
                logger.info("Registered audio preprocessor for model '%s'", name)

        # Register postprocessors if adapter provides them.
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
        # Merge per-model adaptive batching overrides onto engine defaults
        adaptive_params = _merge_adaptive_params(self._adaptive_batching, resolved.adaptive_batching)

        worker_config = WorkerConfig(
            max_batch_tokens=resolved.max_batch_tokens,
            max_batch_requests=self._max_batch_requests or WorkerConfig().max_batch_requests,
            max_batch_wait_ms=self._max_batch_wait_ms or WorkerConfig().max_batch_wait_ms,
            coalesce_ms=self._coalesce_ms if self._coalesce_ms is not None else WorkerConfig().coalesce_ms,
            coalesce_ratio=self._coalesce_ratio if self._coalesce_ratio is not None else WorkerConfig().coalesce_ratio,
            max_queue_size=self._max_queue_size or WorkerConfig().max_queue_size,
            instrumentation=self._instrumentation,
            adaptive_batching=adaptive_params,
            oom_recovery=self._oom_recovery,
        )
        worker = ModelWorker(
            adapter,
            worker_config,
            model_name=name,
            postprocessor_registry=self._postprocessor_registry,
            registry_callbacks=self._registry_callbacks,
        )

        # One authoritative lifecycle event expands to loaded + memory gauges.
        worker_telemetry().model_residency_changed(
            model=name,
            loaded=True,
            memory_bytes=memory_bytes,
        )

        # Create LoadedModel instance
        loaded_model = LoadedModel(
            config=config,
            adapter=adapter,
            device=device,
            worker=worker,
            memory_bytes=memory_bytes,
            max_loras=self._max_loras_per_model,
        )

        # Preload profile LoRAs if the adapter exposes a LoRA capability.
        # Engine-owned adapters (sglang: ``supports_hot_lora_reload() == False``)
        # consume ``loadtime.lora_paths`` themselves at engine launch, so only
        # hot-reload (PEFT) adapters take the loadtime declarations from here.
        lora_cap = adapter.lora_capability()

        # A pinned LoRA revision (#2113) is only honored on the hot-reload path
        # (``load_lora(revision=...)``). An engine-owned adapter consumes bare
        # ids at engine launch and a LoRA-less adapter never loads the ref at
        # all — in both cases a declared pin would silently resolve at the
        # Hub's default branch, which is exactly the drift the pin exists to
        # prevent. Fail the load loudly instead.
        pinned = {ref: rev for ref, rev in config.lora_revisions().items() if rev is not None}
        if pinned and (lora_cap is None or not lora_cap.supports_hot_lora_reload()):
            msg = (
                f"Model '{name}' pins LoRA revision(s) {pinned!r}, but adapter "
                f"{type(adapter).__name__!r} cannot honor a pin: it loads LoRAs at engine "
                "launch (or not at all), where only bare ids exist. Remove the pin, or "
                "serve via a hot-reload (PEFT) adapter. In the managed path, staging "
                "resolves pins to local snapshots before the server sees the config."
            )
            raise ValueError(msg)

        if lora_cap is not None:
            profile_loras = self._collect_profile_loras(
                config,
                include_loadtime_paths=lora_cap.supports_hot_lora_reload(),
            )
            if profile_loras:
                logger.info(
                    "Preloading %d profile LoRAs for model '%s': %s",
                    len(profile_loras),
                    name,
                    list(profile_loras),
                )
                for lora_path, lora_revision in profile_loras.items():
                    try:
                        lora_memory = lora_cap.load_lora(lora_path, revision=lora_revision)
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

    def _collect_profile_loras(
        self,
        config: ModelConfig,
        *,
        include_loadtime_paths: bool = True,
    ) -> dict[str, str | None]:
        """Collect all LoRA paths from model profiles, with any pinned revision.

        The canonical spelling is
        ``adapter_options.loadtime["lora_paths"]`` — either a list of
        HuggingFace/local paths (each path doubles as the public LoRA id on
        the PEFT path) or a dict of served-name → ref (the sglang shape,
        whose paths are what a PEFT adapter would load). A dict-form ref may
        be a bare path or a pinned ``{id, revision}`` mapping (#2113). The
        scalar ``adapter_options.runtime["lora_id"]`` is the DEPRECATED legacy
        alias: it is still accepted here for existing embedding profiles, but
        new profiles should declare ``loadtime.lora_paths`` (the scalar form is
        already rejected outright on generation models — see
        ``core.pool_isolation.validate_no_legacy_scalar_lora_id``).

        Args:
            config: Model configuration.
            include_loadtime_paths: Collect the canonical
                ``loadtime.lora_paths`` declarations. Pass ``False`` for
                engine-owned adapters (sglang) that load those at engine
                launch themselves — preloading them again here would
                double-load.

        Returns:
            Mapping of unique LoRA path/id -> pinned 40-hex revision, or
            ``None`` when unpinned. Revisions come from
            :meth:`ModelConfig.lora_revisions` (pin-beats-bare across
            profiles; conflicting pins already rejected at config load).
        """
        revisions = config.lora_revisions()
        loras: dict[str, str | None] = {}

        for profile_name, profile_config in config.profiles.items():
            # Canonical key: adapter_options.loadtime["lora_paths"].
            if include_loadtime_paths:
                lora_paths = profile_config.adapter_options.loadtime.get("lora_paths")
                entries: list[str] = []
                if isinstance(lora_paths, dict):
                    entries = [lora_entry_ref(ref)[0] for ref in lora_paths.values() if ref]
                elif isinstance(lora_paths, (list, tuple)):
                    entries = [str(path) for path in lora_paths if path]
                elif lora_paths:
                    logger.warning(
                        "Profile '%s' declares loadtime.lora_paths with unexpected "
                        "shape %s (expected list of paths or dict name->path); ignoring",
                        profile_name,
                        type(lora_paths).__name__,
                    )
                for lora_path in entries:
                    loras[lora_path] = revisions.get(lora_path)
                    logger.debug(
                        "Found LoRA '%s' in profile '%s' (loadtime.lora_paths)",
                        lora_path,
                        profile_name,
                    )

            # DEPRECATED alias: the scalar runtime["lora_id"] predates the
            # canonical loadtime.lora_paths spelling and is kept only for
            # backward compatibility with existing embedding profiles.
            lora_id = profile_config.adapter_options.runtime.get("lora_id")
            if lora_id:
                loras[lora_id] = revisions.get(lora_id)
                logger.debug(
                    "Found LoRA '%s' in profile '%s' (legacy runtime.lora_id)",
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
        worker_telemetry().model_residency_changed(
            model=name,
            loaded=False,
            memory_bytes=0,
        )

        # Unregister all preprocessors (text and image)
        self._preprocessor_registry.unregister(name)

        # Unregister postprocessors
        self._postprocessor_registry.unregister(name)

        logger.debug("Unregistered pre/postprocessors for model '%s'", name)

    def shutdown(self) -> None:
        """Shutdown the loader's thread pool."""
        self._load_executor.shutdown(wait=False)


def _run_load_with_markers(name: str, device: str, adapter: ModelAdapter) -> None:
    """Drive ``adapter.load()`` then ``adapter.warmup()`` with cold-start log markers.

    The four markers (``Model deserialize start/end`` and ``Model warmup start/end``)
    let the multipod cold-start bench (schema v6) attribute deserialize and warmup
    time separately. They are emitted unconditionally — adapters whose ``warmup()``
    is a no-op still produce both warmup markers so the parser can attribute
    consistently across adapters (the resulting ``warmup_s`` is just ~0).
    """
    logger.info("Model deserialize start: '%s' on %s", name, device)
    started = time.monotonic()
    try:
        adapter.load(device)
    except RuntimeError as exc:
        _raise_if_adapter_startup_timeout(name, started, exc)
        raise
    logger.info("Model deserialize end: '%s' on %s", name, device)
    logger.info("Model warmup start: '%s' on %s", name, device)
    adapter.warmup()
    logger.info("Model warmup end: '%s' on %s", name, device)


def _raise_if_adapter_startup_timeout(name: str, started: float, exc: RuntimeError) -> None:
    """Reclassify adapter-owned startup timeout errors.

    SGLang raises ``RuntimeError("SGLang server failed to start within timeout")``
    from its subprocess health poll. Pattern-match on the message; narrow enough
    to not bucket genuine runtime failures as timeouts. Other adapters that grow
    their own startup timeouts should follow the same convention.
    """
    msg = str(exc).lower()
    if "failed to start within timeout" not in msg and "startup timeout" not in msg:
        return

    elapsed = time.monotonic() - started
    logger.error(
        "Adapter startup timeout: model=%s elapsed_s=%.1f msg=%s",
        name,
        elapsed,
        exc,
    )
    raise ModelLoadTimeoutError(model=name, stage="load", elapsed_s=elapsed, timeout_s=elapsed) from exc


def _merge_adaptive_params(
    engine: AdaptiveBatchingParams,
    profile: ProfileAdaptiveBatching | None,
) -> AdaptiveBatchingParams:
    """Merge per-model profile overrides onto engine-level adaptive params.

    None fields in the profile fall through to engine defaults.
    If profile is None, returns the engine params unchanged.
    """
    if profile is None:
        return engine
    return AdaptiveBatchingParams(
        enabled=engine.enabled,
        target_p50_ms=profile.target_p50_ms if profile.target_p50_ms is not None else engine.target_p50_ms,
        calibration_multiplier=profile.calibration_multiplier
        if profile.calibration_multiplier is not None
        else engine.calibration_multiplier,
        min_target_p50_ms=profile.min_target_p50_ms
        if profile.min_target_p50_ms is not None
        else engine.min_target_p50_ms,
        max_target_p50_ms=profile.max_target_p50_ms
        if profile.max_target_p50_ms is not None
        else engine.max_target_p50_ms,
        min_wait_ms=profile.min_wait_ms if profile.min_wait_ms is not None else engine.min_wait_ms,
        max_wait_ms=profile.max_wait_ms if profile.max_wait_ms is not None else engine.max_wait_ms,
        gain=profile.gain if profile.gain is not None else engine.gain,
        integral_gain=profile.integral_gain if profile.integral_gain is not None else engine.integral_gain,
        window_size=engine.window_size,
        update_interval=engine.update_interval,
    )
