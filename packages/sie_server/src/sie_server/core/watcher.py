"""File watcher for hot-reload of model configurations.

Watches the models/ directory for changes and triggers callbacks when model
configs are added, modified, or deleted. Uses watchdog for cross-platform
file system monitoring.

Key features:
- Watches for *.yaml config files and adapter.py changes
- Debounces rapid changes to avoid reload storms
- Thread-safe callback invocation
- Graceful start/stop
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of file system change detected."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class ModelChange:
    """Represents a detected change to a model.

    Attributes:
        model_name: Name of the affected model (directory name).
        change_type: Type of change (added, modified, deleted).
        path: Path to the changed file.
        timestamp: Unix timestamp when the change was detected.
    """

    model_name: str
    change_type: ChangeType
    path: Path
    timestamp: float = field(default_factory=time.time)


# Type alias for change callbacks
ChangeCallback = Callable[[ModelChange], None]


@dataclass
class WatcherConfig:
    """Configuration for the file watcher.

    Attributes:
        debounce_seconds: Minimum time between callbacks for the same model.
                         Prevents reload storms from rapid file changes.
        watch_patterns: File patterns to watch (glob patterns for model configs).
    """

    debounce_seconds: float = 1.0
    watch_patterns: tuple[str, ...] = ("*.yaml", "adapter.py")


class _ModelEventHandler(FileSystemEventHandler):
    """Internal event handler for watchdog.

    Filters events to only model-relevant files and passes them to the
    parent watcher for debouncing and callback invocation.
    """

    def __init__(self, watcher: FileWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory:
            self._watcher._handle_event(str(event.src_path), ChangeType.ADDED)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory:
            self._watcher._handle_event(str(event.src_path), ChangeType.MODIFIED)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory:
            self._watcher._handle_event(str(event.src_path), ChangeType.DELETED)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events as delete + create."""
        if not event.is_directory:
            self._watcher._handle_event(str(event.src_path), ChangeType.DELETED)
            if hasattr(event, "dest_path"):
                self._watcher._handle_event(str(event.dest_path), ChangeType.ADDED)


class FileWatcher:
    """Watches the models/ directory for configuration changes.

    The watcher monitors for changes to *.yaml config files and adapter.py files
    in the models directory and invokes registered callbacks when changes
    are detected.

    Usage:
        watcher = FileWatcher(Path("./models"))
        watcher.on_change(lambda change: print(f"Model {change.model_name} changed"))
        watcher.start()
        # ... later
        watcher.stop()
    """

    def __init__(
        self,
        models_dir: Path | str,
        config: WatcherConfig | None = None,
    ) -> None:
        """Initialize the file watcher.

        Args:
            models_dir: Path to the models directory to watch.
            config: Watcher configuration. Uses defaults if not provided.
        """
        self._models_dir = Path(models_dir).resolve()
        self._config = config or WatcherConfig()
        self._callbacks: list[ChangeCallback] = []
        self._observer: BaseObserver | None = None
        self._lock = threading.Lock()

        # Debouncing state: model_name -> last callback timestamp
        self._last_callback: dict[str, float] = {}
        self._pending_changes: dict[str, ModelChange] = {}
        self._debounce_timer: threading.Timer | None = None

    @property
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._observer is not None and self._observer.is_alive()

    @property
    def models_dir(self) -> Path:
        """Return the models directory being watched."""
        return self._models_dir

    def on_change(self, callback: ChangeCallback) -> None:
        """Register a callback to be invoked when models change.

        Callbacks are invoked after debouncing, from a background thread.
        They receive a ModelChange object with details about the change.

        Args:
            callback: Function to call when a model changes.
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: ChangeCallback) -> bool:
        """Remove a previously registered callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if the callback was found and removed, False otherwise.
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def start(self) -> None:
        """Start watching the models directory.

        Creates a background thread to monitor file system events.
        Does nothing if already running.

        Raises:
            FileNotFoundError: If the models directory does not exist.
        """
        if self.is_running:
            logger.warning("File watcher is already running")
            return

        if not self._models_dir.exists():
            msg = f"Models directory does not exist: {self._models_dir}"
            raise FileNotFoundError(msg)

        logger.info("Starting file watcher on %s", self._models_dir)

        self._observer = Observer()
        handler = _ModelEventHandler(self)
        self._observer.schedule(handler, str(self._models_dir), recursive=True)
        self._observer.start()

        logger.info("File watcher started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop watching the models directory.

        Waits for the background thread to terminate.
        Does nothing if not running.

        Args:
            timeout: Maximum time to wait for shutdown in seconds.
        """
        if not self.is_running:
            return

        logger.info("Stopping file watcher")

        # Cancel any pending debounce timer
        with self._lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
                self._debounce_timer = None
            self._pending_changes.clear()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout)
            self._observer = None

        logger.info("File watcher stopped")

    def _handle_event(self, path_str: str, change_type: ChangeType) -> None:
        """Handle a file system event.

        Filters events to only relevant files and schedules debounced callbacks.

        Args:
            path_str: Path to the changed file.
            change_type: Type of change detected.
        """
        path = Path(path_str)

        # Check if this is a relevant file (using fnmatch for glob patterns)
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in self._config.watch_patterns):
            return

        # Extract model name from path
        model_name = self._extract_model_name(path)
        if model_name is None:
            logger.debug("Ignoring change outside model directory: %s", path)
            return

        logger.debug(
            "Detected %s for model '%s': %s",
            change_type.value,
            model_name,
            path.name,
        )

        change = ModelChange(
            model_name=model_name,
            change_type=change_type,
            path=path,
        )

        self._schedule_callback(change)

    def _extract_model_name(self, path: Path) -> str | None:
        """Extract the model name from a file path.

        For flat YAML files directly in models/, the model name is derived from
        the filename (e.g., 'baai-bge-m3.yaml' -> 'baai-bge-m3').
        For adapter.py files in subdirectories, use the directory name.

        Args:
            path: Path to a file within the models directory.

        Returns:
            The model name, or None if the path is not within the models directory.
        """
        try:
            # Get the path relative to models directory
            relative = path.relative_to(self._models_dir)
            parts = relative.parts

            if len(parts) == 1:
                # Flat YAML file directly in models/
                # Extract model name from filename (remove .yaml extension)
                filename = parts[0]
                if filename.endswith(".yaml"):
                    return filename[:-5]  # Remove .yaml
                return filename
            if len(parts) >= 2:
                # File in subdirectory (e.g., adapter.py in model folder)
                return parts[0]

            return None
        except ValueError:
            # Path is not under models_dir
            return None

    def _schedule_callback(self, change: ModelChange) -> None:
        """Schedule a debounced callback for a change.

        If another change for the same model arrives within the debounce
        window, the callback is delayed. Only the most recent change type
        is reported.

        Args:
            change: The model change to report.
        """
        with self._lock:
            model_name = change.model_name
            now = time.time()

            # Store the pending change (overwrites previous for same model)
            self._pending_changes[model_name] = change

            # Calculate time since last callback for this model
            last_time = self._last_callback.get(model_name, 0)
            time_since_last = now - last_time

            if time_since_last >= self._config.debounce_seconds:
                # Can fire immediately
                self._fire_callbacks()
            else:
                # Schedule a delayed callback
                delay = self._config.debounce_seconds - time_since_last
                self._schedule_debounce_timer(delay)

    def _schedule_debounce_timer(self, delay: float) -> None:
        """Schedule (or reschedule) the debounce timer.

        Must be called with _lock held.

        Args:
            delay: Seconds to wait before firing callbacks.
        """
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()

        self._debounce_timer = threading.Timer(delay, self._fire_callbacks_threadsafe)
        self._debounce_timer.daemon = True
        self._debounce_timer.start()

    def _fire_callbacks_threadsafe(self) -> None:
        """Thread-safe wrapper for firing callbacks."""
        with self._lock:
            self._fire_callbacks()

    def _fire_callbacks(self) -> None:
        """Fire callbacks for all pending changes.

        Must be called with _lock held.
        """
        self._debounce_timer = None

        if not self._pending_changes:
            return

        # Take a snapshot of pending changes and clear
        changes = list(self._pending_changes.values())
        self._pending_changes.clear()

        # Update last callback times
        now = time.time()
        for change in changes:
            self._last_callback[change.model_name] = now

        # Make a copy of callbacks to release the lock during invocation
        callbacks = list(self._callbacks)

        # Fire callbacks outside the lock
        for change in changes:
            for callback in callbacks:
                try:
                    callback(change)
                except Exception:
                    logger.exception(
                        "Error in file watcher callback for model '%s'",
                        change.model_name,
                    )
