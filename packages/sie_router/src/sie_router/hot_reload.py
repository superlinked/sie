from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from watchdog.observers.polling import PollingObserver

if TYPE_CHECKING:
    from sie_router.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class WatcherConfig:
    """Configuration for the config watcher.

    Attributes:
        debounce_seconds: Minimum time between reloads.
        bundle_patterns: File patterns to watch in bundles dir.
        model_patterns: File patterns to watch in models dir.
        use_polling: Use polling observer (required for K8s ConfigMap mounts).
        polling_interval: Polling interval in seconds (if use_polling is True).
    """

    debounce_seconds: float = 1.0
    bundle_patterns: tuple[str, ...] = ("*.yaml",)
    model_patterns: tuple[str, ...] = ("*.yaml",)
    use_polling: bool = False
    polling_interval: float = 5.0


class _ConfigEventHandler(FileSystemEventHandler):
    """Internal event handler for watchdog.

    Filters events to config files and triggers reload.
    """

    def __init__(self, watcher: ConfigWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory:
            self._watcher._handle_event(str(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory:
            self._watcher._handle_event(str(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory:
            self._watcher._handle_event(str(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events."""
        if not event.is_directory:
            self._watcher._handle_event(str(event.src_path))
            if hasattr(event, "dest_path"):
                self._watcher._handle_event(str(event.dest_path))


class ConfigWatcher:
    """Watches bundle and model config directories for changes.

    When changes are detected, triggers a reload of the ModelRegistry.
    Debounces rapid changes to avoid reload storms.

    Attributes:
        registry: The ModelRegistry to reload on changes.
        bundles_dir: Path to bundle YAML files.
        models_dir: Path to model config directories.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        bundles_dir: Path | str | None = None,
        models_dir: Path | str | None = None,
        config: WatcherConfig | None = None,
    ) -> None:
        """Initialize the config watcher.

        Args:
            registry: The ModelRegistry to reload on changes.
            bundles_dir: Path to bundles directory (defaults to registry.bundles_dir).
            models_dir: Path to models directory (defaults to registry.models_dir).
            config: Watcher configuration.
        """
        self._registry = registry
        self._bundles_dir = Path(bundles_dir) if bundles_dir else registry.bundles_dir
        self._models_dir = Path(models_dir) if models_dir else registry.models_dir

        # Use config or defaults with auto-detection for polling mode
        if config is not None:
            self._config = config
        else:
            # Auto-detect polling mode for Kubernetes ConfigMap mounts
            # ConfigMaps use symlinks that inotify doesn't detect properly
            use_polling = os.environ.get("SIE_ROUTER_POLLING_WATCHER", "").lower() == "true"
            # Also enable polling if running in Kubernetes (configMap mode)
            if not use_polling and os.environ.get("SIE_ROUTER_KUBERNETES", "").lower() == "true":
                bundles_path = str(self._bundles_dir)
                if "/configs/" in bundles_path:
                    use_polling = True
                    logger.info("Auto-enabling polling watcher for Kubernetes ConfigMap mode")
            self._config = WatcherConfig(use_polling=use_polling)

        # Watchdog components
        self._observer: BaseObserver | None = None
        self._handler = _ConfigEventHandler(self)

        # Debounce state
        self._lock = threading.Lock()
        self._last_reload_time: float = 0.0
        self._pending_reload: bool = False
        self._reload_timer: threading.Timer | None = None

        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the watcher is running."""
        return self._running

    def start(self) -> None:
        """Start watching for config changes.

        Begins monitoring bundles and models directories.
        Does nothing if already running.
        """
        if self._running:
            logger.warning("ConfigWatcher is already running")
            return

        logger.info(
            "Starting ConfigWatcher: bundles=%s, models=%s, polling=%s",
            self._bundles_dir,
            self._models_dir,
            self._config.use_polling,
        )

        # Use polling observer for Kubernetes ConfigMap mounts (symlink-based)
        if self._config.use_polling:
            self._observer = PollingObserver(timeout=self._config.polling_interval)
        else:
            self._observer = Observer()

        # Watch bundles directory
        if self._bundles_dir.exists():
            self._observer.schedule(
                self._handler,
                str(self._bundles_dir),
                recursive=False,
            )
            logger.debug("Watching bundles: %s", self._bundles_dir)
        else:
            logger.warning("Bundles directory not found: %s", self._bundles_dir)

        # Watch models directory (recursive for subdirectories)
        if self._models_dir.exists():
            self._observer.schedule(
                self._handler,
                str(self._models_dir),
                recursive=True,
            )
            logger.debug("Watching models: %s", self._models_dir)
        else:
            logger.warning("Models directory not found: %s", self._models_dir)

        self._observer.start()
        self._running = True

        logger.info("ConfigWatcher started")

    def stop(self) -> None:
        """Stop watching for config changes.

        Stops the observer and cancels any pending reloads.
        """
        if not self._running:
            return

        logger.info("Stopping ConfigWatcher")

        self._running = False

        # Cancel pending reload
        with self._lock:
            if self._reload_timer is not None:
                self._reload_timer.cancel()
                self._reload_timer = None
            self._pending_reload = False

        # Stop observer
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        logger.info("ConfigWatcher stopped")

    def _handle_event(self, path: str) -> None:
        """Handle a file system event.

        Filters to relevant config files and schedules a debounced reload.

        Args:
            path: Path to the changed file.
        """
        file_path = Path(path)

        # Check if this is a relevant file
        if not self._is_config_file(file_path):
            return

        logger.debug("Config change detected: %s", file_path)

        # Schedule debounced reload
        self._schedule_reload()

    def _is_config_file(self, path: Path) -> bool:
        """Check if a path matches our config file patterns.

        Args:
            path: Path to check.

        Returns:
            True if this is a config file we care about.
        """
        name = path.name

        # Resolve symlinks for comparison (macOS has /var -> /private/var)
        try:
            resolved_path = path.resolve()
            resolved_bundles = self._bundles_dir.resolve()
            resolved_models = self._models_dir.resolve()
        except OSError:
            # Path may not exist (deleted file), use original paths
            resolved_path = path
            resolved_bundles = self._bundles_dir
            resolved_models = self._models_dir

        # Check bundle patterns
        if resolved_bundles in resolved_path.parents or resolved_path.parent == resolved_bundles:
            for pattern in self._config.bundle_patterns:
                if path.match(pattern):
                    return True

        # Check model patterns
        if resolved_models in resolved_path.parents or resolved_path.parent == resolved_models:
            for pattern in self._config.model_patterns:
                if path.match(pattern) or name == pattern:
                    return True

        return False

    def _schedule_reload(self) -> None:
        """Schedule a debounced reload.

        If a reload is already pending, this does nothing.
        The actual reload happens after debounce_seconds.
        """
        with self._lock:
            if self._pending_reload:
                return

            # Calculate delay
            now = time.monotonic()
            elapsed = now - self._last_reload_time
            delay = max(0.0, self._config.debounce_seconds - elapsed)

            self._pending_reload = True

            # Schedule reload
            self._reload_timer = threading.Timer(delay, self._do_reload)
            self._reload_timer.daemon = True
            self._reload_timer.start()

            logger.debug("Scheduled reload in %.2f seconds", delay)

    def _do_reload(self) -> None:
        """Execute the reload.

        Called by the debounce timer in a separate thread.
        """
        with self._lock:
            self._pending_reload = False
            self._reload_timer = None
            self._last_reload_time = time.monotonic()

        try:
            logger.info("Reloading ModelRegistry...")
            self._registry.reload()
            logger.info(
                "ModelRegistry reloaded: %d bundles, %d models",
                len(self._registry.list_bundles()),
                len(self._registry.list_models()),
            )
        except Exception:
            logger.exception("Failed to reload ModelRegistry")
