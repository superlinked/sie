"""Hot reload support for model configurations.

Watches the models/ directory and automatically reloads models when their
configurations change. Handles the full lifecycle:
1. Detect file changes via FileWatcher
2. Drain in-flight requests to affected model
3. Unload model and clear module cache
4. Load new adapter
5. Resume serving

See DESIGN.md Section 7.4.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from sie_server.core.deps import model_name_to_folder
from sie_server.core.loader import load_model_config, load_model_configs
from sie_server.core.watcher import ChangeType, FileWatcher, ModelChange, WatcherConfig

if TYPE_CHECKING:
    from sie_server.core.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ReloadStatus(Enum):
    """Status of a reload operation."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ReloadResult:
    """Result of a model reload operation.

    Attributes:
        model_name: Name of the model that was reloaded.
        status: Whether the reload succeeded, failed, or was skipped.
        change_type: Type of change that triggered the reload.
        error: Error message if reload failed.
    """

    model_name: str
    status: ReloadStatus
    change_type: ChangeType
    error: str | None = None


@dataclass
class HotReloadConfig:
    """Configuration for hot reloading.

    Attributes:
        debounce_seconds: Time to wait after a change before reloading.
        drain_timeout_seconds: Maximum time to wait for in-flight requests to complete.
        enabled: Whether hot reload is enabled.
    """

    debounce_seconds: float = 1.0
    drain_timeout_seconds: float = 30.0
    enabled: bool = True


@dataclass
class _PendingReload:
    """Internal state for a pending reload operation."""

    model_name: str
    change_type: ChangeType
    config_path: Path


class HotReloader:
    """Manages hot reloading of model configurations.

    Watches the models/ directory for changes and coordinates with the
    ModelRegistry to reload affected models without server restart.

    Usage:
        reloader = HotReloader(registry, models_dir)
        await reloader.start()
        # ... server runs, models reload automatically on config changes
        await reloader.stop()

    The reloader handles three types of changes:
    - ADDED: Register new model config, lazy-load on first request
    - MODIFIED: Drain requests, unload, reload config, load if was loaded
    - DELETED: Drain requests, unload, remove from registry
    """

    def __init__(
        self,
        registry: ModelRegistry,
        models_dir: Path | str,
        device: str = "cpu",
        config: HotReloadConfig | None = None,
    ) -> None:
        """Initialize the hot reloader.

        Args:
            registry: The model registry to manage.
            models_dir: Path to the models directory to watch.
            device: Device to load models on.
            config: Hot reload configuration.
        """
        self._registry = registry
        self._models_dir = Path(models_dir).resolve()
        self._device = device
        self._config = config or HotReloadConfig()

        # File watcher with matching debounce
        watcher_config = WatcherConfig(debounce_seconds=self._config.debounce_seconds)
        self._watcher = FileWatcher(models_dir, config=watcher_config)

        # Async components
        self._reload_queue: asyncio.Queue[ModelChange] = asyncio.Queue()
        self._reload_task: asyncio.Task[None] | None = None
        self._running = False

        # Track models being reloaded (for request draining)
        self._reloading: set[str] = set()
        self._reload_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Check if the hot reloader is running."""
        return self._running

    @property
    def models_dir(self) -> Path:
        """Return the models directory being watched."""
        return self._models_dir

    def is_model_reloading(self, model_name: str) -> bool:
        """Check if a model is currently being reloaded.

        Can be used by request handlers to return 503 during reload.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if the model is currently being reloaded.
        """
        return model_name in self._reloading

    async def start(self) -> None:
        """Start the hot reloader.

        Begins watching for file changes and processing reloads.
        Does nothing if already running or if hot reload is disabled.
        """
        if self._running:
            logger.warning("Hot reloader is already running")
            return

        if not self._config.enabled:
            logger.info("Hot reload is disabled")
            return

        logger.info("Starting hot reloader on %s", self._models_dir)

        # Register callback and start watcher
        self._watcher.on_change(self._on_file_change)
        self._watcher.start()

        # Start reload processor
        self._running = True
        self._reload_task = asyncio.create_task(
            self._reload_loop(),
            name="hot-reload-processor",
        )

        logger.info("Hot reloader started")

    async def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the hot reloader.

        Stops watching for changes and waits for pending reloads to complete.

        Args:
            timeout_s: Maximum time to wait for pending reloads.
        """
        if not self._running:
            return

        logger.info("Stopping hot reloader")

        self._running = False

        # Stop the watcher
        self._watcher.stop()

        # Cancel reload task
        if self._reload_task is not None:
            self._reload_task.cancel()
            try:
                await asyncio.wait_for(self._reload_task, timeout=timeout_s)
            except (TimeoutError, asyncio.CancelledError):
                pass
            self._reload_task = None

        logger.info("Hot reloader stopped")

    def _on_file_change(self, change: ModelChange) -> None:
        """Callback invoked by FileWatcher on detected changes.

        Queues the change for async processing.

        Args:
            change: The detected model change.
        """
        logger.info(
            "File change detected: %s %s (%s)",
            change.change_type.value,
            change.model_name,
            change.path.name,
        )

        # Queue for async processing (thread-safe)
        try:
            self._reload_queue.put_nowait(change)
        except asyncio.QueueFull:
            logger.warning(
                "Reload queue full, dropping change for %s",
                change.model_name,
            )

    async def _reload_loop(self) -> None:
        """Background loop that processes reload requests."""
        logger.debug("Reload loop started")

        while self._running:
            try:
                # Wait for next change (with timeout for graceful shutdown)
                try:
                    change = await asyncio.wait_for(
                        self._reload_queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                # Process the reload
                result = await self._process_reload(change)
                self._log_reload_result(result)

            except asyncio.CancelledError:
                logger.debug("Reload loop cancelled")
                break
            except Exception:
                logger.exception("Error in reload loop")

        logger.debug("Reload loop stopped")

    async def _process_reload(self, change: ModelChange) -> ReloadResult:
        """Process a single reload request.

        Args:
            change: The model change to process.

        Returns:
            Result of the reload operation.
        """
        model_name = change.model_name

        async with self._reload_lock:
            try:
                if change.change_type == ChangeType.ADDED:
                    return await self._handle_model_added(change)
                if change.change_type == ChangeType.MODIFIED:
                    return await self._handle_model_modified(change)
                if change.change_type == ChangeType.DELETED:
                    return await self._handle_model_deleted(change)
                return ReloadResult(
                    model_name=model_name,
                    status=ReloadStatus.SKIPPED,
                    change_type=change.change_type,
                    error=f"Unknown change type: {change.change_type}",
                )
            except Exception as e:
                logger.exception("Failed to reload model %s", model_name)
                return ReloadResult(
                    model_name=model_name,
                    status=ReloadStatus.FAILED,
                    change_type=change.change_type,
                    error=str(e),
                )

    async def _handle_model_added(self, change: ModelChange) -> ReloadResult:
        """Handle a new model being added.

        Loads the config and registers it with the registry.
        The model will be lazy-loaded on first request.

        Args:
            change: The add change event.

        Returns:
            Result of the operation.
        """
        model_name = change.model_name
        # Flat YAML config: models/{model_name}.yaml
        config_path = change.path
        if not config_path.exists():
            # Also try the flat naming convention
            config_path = self._models_dir / f"{model_name_to_folder(model_name)}.yaml"

        if not config_path.exists():
            return ReloadResult(
                model_name=model_name,
                status=ReloadStatus.SKIPPED,
                change_type=change.change_type,
                error="config file not found",
            )

        # Load the new config
        try:
            config = load_model_config(config_path)
        except (OSError, ValueError, KeyError) as e:
            return ReloadResult(
                model_name=model_name,
                status=ReloadStatus.FAILED,
                change_type=change.change_type,
                error=f"Failed to load config: {e}",
            )

        # Register with registry (config_path for flat YAML)
        self._registry.add_config(config, config_path)

        logger.info("Added new model: %s", model_name)
        return ReloadResult(
            model_name=model_name,
            status=ReloadStatus.SUCCESS,
            change_type=change.change_type,
        )

    async def _handle_model_modified(self, change: ModelChange) -> ReloadResult:
        """Handle a model config/adapter being modified.

        If the model is loaded:
        1. Mark as reloading (blocks new requests)
        2. Drain in-flight requests
        3. Unload model
        4. Clear module cache
        5. Reload config
        6. Load model again

        If not loaded, just reload the config.

        Args:
            change: The modify change event.

        Returns:
            Result of the operation.
        """
        model_name = change.model_name
        # Flat YAML config: models/{model_name}.yaml
        config_path = change.path
        if not config_path.exists():
            # Also try the flat naming convention
            config_path = self._models_dir / f"{model_name_to_folder(model_name)}.yaml"

        if not config_path.exists():
            return ReloadResult(
                model_name=model_name,
                status=ReloadStatus.SKIPPED,
                change_type=change.change_type,
                error="config file not found",
            )

        was_loaded = self._registry.is_loaded(model_name)

        if was_loaded:
            # Mark as reloading
            self._reloading.add(model_name)

            try:
                # Drain in-flight requests
                await self._drain_requests(model_name)

                # Stop worker if running
                await self._registry.stop_worker(model_name)

                # Unload the model
                self._registry.unload(model_name)

                # Clear custom adapter from module cache
                self._clear_custom_adapter_cache(model_name)

            finally:
                self._reloading.discard(model_name)

        # Reload all configs to handle base_model dependencies
        try:
            new_configs = load_model_configs(self._models_dir)
        except (OSError, ValueError, KeyError) as e:
            return ReloadResult(
                model_name=model_name,
                status=ReloadStatus.FAILED,
                change_type=change.change_type,
                error=f"Failed to reload configs: {e}",
            )

        # Update registry with new config
        if model_name in new_configs:
            self._registry.add_config(new_configs[model_name], config_path)

        # Reload model if it was loaded before
        if was_loaded and model_name in new_configs:
            try:
                self._registry.load(model_name, self._device)
                await self._registry.start_worker(model_name)
                logger.info("Reloaded model: %s", model_name)
            except (RuntimeError, OSError, ValueError) as e:
                return ReloadResult(
                    model_name=model_name,
                    status=ReloadStatus.FAILED,
                    change_type=change.change_type,
                    error=f"Failed to reload model: {e}",
                )

        return ReloadResult(
            model_name=model_name,
            status=ReloadStatus.SUCCESS,
            change_type=change.change_type,
        )

    async def _handle_model_deleted(self, change: ModelChange) -> ReloadResult:
        """Handle a model being deleted.

        Unloads the model if loaded and removes from registry.

        Args:
            change: The delete change event.

        Returns:
            Result of the operation.
        """
        model_name = change.model_name

        if not self._registry.has_model(model_name):
            return ReloadResult(
                model_name=model_name,
                status=ReloadStatus.SKIPPED,
                change_type=change.change_type,
                error="Model not in registry",
            )

        if self._registry.is_loaded(model_name):
            # Mark as reloading
            self._reloading.add(model_name)

            try:
                # Drain in-flight requests
                await self._drain_requests(model_name)

                # Stop worker if running
                await self._registry.stop_worker(model_name)

                # Unload the model
                self._registry.unload(model_name)

            finally:
                self._reloading.discard(model_name)

        # Remove from registry configs
        self._registry._configs.pop(model_name, None)
        self._registry._model_dirs.pop(model_name, None)

        # Clear custom adapter from module cache
        self._clear_custom_adapter_cache(model_name)

        logger.info("Removed model: %s", model_name)
        return ReloadResult(
            model_name=model_name,
            status=ReloadStatus.SUCCESS,
            change_type=change.change_type,
        )

    async def _drain_requests(self, model_name: str) -> None:
        """Wait for in-flight requests to complete.

        Args:
            model_name: Name of the model to drain.
        """
        worker = self._registry.get_worker(model_name)
        if worker is None or not worker.is_running:
            return

        start_time = asyncio.get_event_loop().time()
        timeout = self._config.drain_timeout_seconds

        while worker.pending_count > 0:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(
                    "Drain timeout for %s: %d requests still pending",
                    model_name,
                    worker.pending_count,
                )
                break

            logger.debug(
                "Draining %s: %d requests pending",
                model_name,
                worker.pending_count,
            )
            await asyncio.sleep(0.1)

        logger.debug("Drain complete for %s", model_name)

    def _clear_custom_adapter_cache(self, model_name: str) -> None:
        """Clear custom adapter modules from sys.modules.

        This ensures that modified adapter.py files are reloaded.

        Args:
            model_name: Name of the model whose adapter to clear.
        """
        # Custom adapters are loaded with names like "sie_custom_adapters.adapter_<id>"
        # We clear any modules that match the model's adapter file
        model_folder = model_name_to_folder(model_name)
        model_dir = self._models_dir / model_folder
        adapter_file = model_dir / "adapter.py"

        if not adapter_file.exists():
            return

        # Find and remove matching modules
        modules_to_remove = []
        for module_name in sys.modules:
            if module_name.startswith("sie_custom_adapters."):
                module = sys.modules[module_name]
                if hasattr(module, "__file__") and module.__file__:
                    if Path(module.__file__).resolve() == adapter_file.resolve():
                        modules_to_remove.append(module_name)

        for module_name in modules_to_remove:
            logger.debug("Clearing module cache: %s", module_name)
            del sys.modules[module_name]

        # Also invalidate importlib caches
        importlib.invalidate_caches()

    def _log_reload_result(self, result: ReloadResult) -> None:
        """Log the result of a reload operation."""
        if result.status == ReloadStatus.SUCCESS:
            logger.info(
                "Reload %s: %s %s",
                result.status.value,
                result.change_type.value,
                result.model_name,
            )
        elif result.status == ReloadStatus.FAILED:
            logger.error(
                "Reload %s: %s %s - %s",
                result.status.value,
                result.change_type.value,
                result.model_name,
                result.error,
            )
        else:
            logger.debug(
                "Reload %s: %s %s - %s",
                result.status.value,
                result.change_type.value,
                result.model_name,
                result.error,
            )
