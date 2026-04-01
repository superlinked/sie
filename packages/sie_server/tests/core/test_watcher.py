"""Tests for FileWatcher and model change detection.

These tests are designed to be fast by:
- Testing internal methods directly (no real file system watching)
- Using debounce_seconds=0 to fire callbacks immediately
- No time.sleep() calls
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sie_server.core.watcher import (
    ChangeType,
    FileWatcher,
    ModelChange,
    WatcherConfig,
)


class TestModelChange:
    """Tests for ModelChange dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        change = ModelChange(
            model_name="test-model",
            change_type=ChangeType.MODIFIED,
            path=Path("/models/test-model.yaml"),
        )

        assert change.model_name == "test-model"
        assert change.change_type == ChangeType.MODIFIED
        assert change.path == Path("/models/test-model.yaml")
        assert change.timestamp > 0

    def test_change_types(self) -> None:
        """Test all change types."""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"


class TestWatcherConfig:
    """Tests for WatcherConfig."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = WatcherConfig()

        assert config.debounce_seconds == 1.0
        assert "*.yaml" in config.watch_patterns
        assert "adapter.py" in config.watch_patterns

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = WatcherConfig(
            debounce_seconds=0.5,
            watch_patterns=("*.yaml",),
        )

        assert config.debounce_seconds == 0.5
        assert config.watch_patterns == ("*.yaml",)


class TestFileWatcher:
    """Tests for FileWatcher - unit tests using internal methods."""

    @pytest.fixture
    def temp_models_dir(self) -> Generator[Path, None, None]:
        """Create a temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            yield models_dir

    @pytest.fixture
    def watcher(self, temp_models_dir: Path) -> FileWatcher:
        """Create a watcher with no debounce for instant callbacks."""
        config = WatcherConfig(debounce_seconds=0)
        return FileWatcher(temp_models_dir, config=config)

    def test_init(self, temp_models_dir: Path) -> None:
        """Test initialization."""
        watcher = FileWatcher(temp_models_dir)

        assert watcher.models_dir.resolve() == temp_models_dir.resolve()
        assert not watcher.is_running

    def test_start_stop(self, temp_models_dir: Path) -> None:
        """Test starting and stopping the watcher."""
        watcher = FileWatcher(temp_models_dir)
        assert not watcher.is_running

        watcher.start()
        assert watcher.is_running

        watcher.stop()
        assert not watcher.is_running

    def test_start_nonexistent_dir(self) -> None:
        """Test starting with non-existent directory raises error."""
        watcher = FileWatcher(Path("/nonexistent/path"))

        with pytest.raises(FileNotFoundError, match="does not exist"):
            watcher.start()

    def test_start_idempotent(self, temp_models_dir: Path) -> None:
        """Test starting twice is a no-op."""
        watcher = FileWatcher(temp_models_dir)
        watcher.start()
        watcher.start()  # Should not raise

        assert watcher.is_running
        watcher.stop()

    def test_stop_idempotent(self, watcher: FileWatcher) -> None:
        """Test stopping when not running is a no-op."""
        watcher.stop()  # Should not raise
        assert not watcher.is_running

    def test_register_callback(self, watcher: FileWatcher) -> None:
        """Test registering callbacks."""
        callback = MagicMock()

        watcher.on_change(callback)

        # Callback should be stored
        assert callback in watcher._callbacks

    def test_remove_callback(self, watcher: FileWatcher) -> None:
        """Test removing callbacks."""
        callback = MagicMock()
        watcher.on_change(callback)

        result = watcher.remove_callback(callback)

        assert result is True
        assert callback not in watcher._callbacks

    def test_remove_nonexistent_callback(self, watcher: FileWatcher) -> None:
        """Test removing non-existent callback returns False."""
        callback = MagicMock()

        result = watcher.remove_callback(callback)

        assert result is False

    def test_extract_model_name(self, watcher: FileWatcher) -> None:
        """Test extracting model name from paths."""
        models_dir = watcher.models_dir

        # Valid flat YAML file
        path = models_dir / "my-model.yaml"
        assert watcher._extract_model_name(path) == "my-model"

        # Nested path for adapter.py still gets model name
        path = models_dir / "my-model" / "adapter.py"
        assert watcher._extract_model_name(path) == "my-model"

        # Path outside models dir
        path = Path("/other/path/model.yaml")
        assert watcher._extract_model_name(path) is None

    def test_filter_irrelevant_files(self, watcher: FileWatcher) -> None:
        """Test that non-config files are ignored."""
        callback = MagicMock()
        watcher.on_change(callback)

        # Create model directory path (no real files needed for unit test)
        model_path = watcher.models_dir / "test-model" / "random.txt"

        # Simulate file event for non-watched file
        watcher._handle_event(str(model_path), ChangeType.MODIFIED)

        # Should not trigger callback (file pattern not in watch_patterns)
        callback.assert_not_called()

    def test_detect_config_added(self, watcher: FileWatcher) -> None:
        """Test detecting new YAML config via direct method call."""
        changes: list[ModelChange] = []
        watcher.on_change(changes.append)

        # Simulate file creation event
        config_path = watcher.models_dir / "new-model.yaml"
        watcher._handle_event(str(config_path), ChangeType.ADDED)

        assert len(changes) == 1
        assert changes[0].model_name == "new-model"
        assert changes[0].change_type == ChangeType.ADDED

    def test_detect_config_modified(self, watcher: FileWatcher) -> None:
        """Test detecting modified YAML config via direct method call."""
        changes: list[ModelChange] = []
        watcher.on_change(changes.append)

        # Simulate file modification event
        config_path = watcher.models_dir / "existing-model.yaml"
        watcher._handle_event(str(config_path), ChangeType.MODIFIED)

        assert len(changes) == 1
        assert changes[0].model_name == "existing-model"
        assert changes[0].change_type == ChangeType.MODIFIED

    def test_detect_config_deleted(self, watcher: FileWatcher) -> None:
        """Test detecting deleted YAML config via direct method call."""
        changes: list[ModelChange] = []
        watcher.on_change(changes.append)

        # Simulate file deletion event
        config_path = watcher.models_dir / "delete-model.yaml"
        watcher._handle_event(str(config_path), ChangeType.DELETED)

        assert len(changes) == 1
        assert changes[0].model_name == "delete-model"
        assert changes[0].change_type == ChangeType.DELETED

    def test_detect_adapter_change(self, watcher: FileWatcher) -> None:
        """Test detecting adapter.py changes via direct method call."""
        changes: list[ModelChange] = []
        watcher.on_change(changes.append)

        # Simulate adapter file event
        adapter_path = watcher.models_dir / "custom-model" / "adapter.py"
        watcher._handle_event(str(adapter_path), ChangeType.ADDED)

        assert len(changes) == 1
        assert changes[0].model_name == "custom-model"
        assert changes[0].path.name == "adapter.py"

    def test_debounce_rapid_changes(self, temp_models_dir: Path) -> None:
        """Test that rapid changes are debounced."""
        # Use a non-zero debounce to test debouncing logic
        config = WatcherConfig(debounce_seconds=1.0)  # Long debounce
        watcher = FileWatcher(temp_models_dir, config=config)

        changes: list[ModelChange] = []
        watcher.on_change(changes.append)

        config_path = watcher.models_dir / "rapid-model.yaml"

        # Send multiple rapid events
        for _ in range(5):
            watcher._handle_event(str(config_path), ChangeType.MODIFIED)

        # First event fires immediately, rest are pending
        assert len(changes) == 1
        assert len(watcher._pending_changes) == 1

        # Manually fire pending callbacks (simulates debounce timer)
        with watcher._lock:
            watcher._fire_callbacks()

        # Now should have 2 total (first immediate + debounced batch)
        assert len(changes) == 2

    def test_multiple_models(self, watcher: FileWatcher) -> None:
        """Test watching multiple model directories via direct method calls."""
        changes: list[ModelChange] = []
        watcher.on_change(changes.append)

        # Simulate events for two models
        path_a = watcher.models_dir / "model-a.yaml"
        path_b = watcher.models_dir / "model-b.yaml"

        watcher._handle_event(str(path_a), ChangeType.ADDED)
        watcher._handle_event(str(path_b), ChangeType.ADDED)

        assert len(changes) == 2
        model_names = {c.model_name for c in changes}
        assert "model-a" in model_names
        assert "model-b" in model_names

    def test_callback_exception_handling(self, watcher: FileWatcher) -> None:
        """Test that callback exceptions don't break the watcher."""
        good_changes: list[ModelChange] = []

        def bad_callback(change: ModelChange) -> None:
            msg = "Callback error"
            raise RuntimeError(msg)

        def good_callback(change: ModelChange) -> None:
            good_changes.append(change)

        # Bad callback first, good callback second
        watcher.on_change(bad_callback)
        watcher.on_change(good_callback)

        # Simulate event
        config_path = watcher.models_dir / "error-model.yaml"
        watcher._handle_event(str(config_path), ChangeType.ADDED)

        # Good callback should still be called despite bad one failing
        assert len(good_changes) == 1


class TestFileWatcherIntegration:
    """Integration tests with mocked Observer - no real file watching."""

    @pytest.fixture
    def temp_models_dir(self) -> Generator[Path, None, None]:
        """Create a temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            yield models_dir

    def test_full_lifecycle(self, temp_models_dir: Path) -> None:
        """Test a complete add -> modify -> delete lifecycle via direct calls."""
        config = WatcherConfig(debounce_seconds=0)
        watcher = FileWatcher(temp_models_dir, config=config)

        changes: list[ModelChange] = []
        watcher.on_change(changes.append)

        # Use watcher.models_dir to ensure proper path resolution
        config_path = watcher.models_dir / "lifecycle-model.yaml"

        # Simulate: Add -> Modify -> Delete
        watcher._handle_event(str(config_path), ChangeType.ADDED)
        watcher._handle_event(str(config_path), ChangeType.MODIFIED)
        watcher._handle_event(str(config_path), ChangeType.DELETED)

        # Check we got all event types
        change_types = {c.change_type for c in changes}

        assert ChangeType.ADDED in change_types
        assert ChangeType.MODIFIED in change_types
        assert ChangeType.DELETED in change_types

    def test_start_creates_observer(self, temp_models_dir: Path) -> None:
        """Test that start() properly creates and starts the observer."""
        watcher = FileWatcher(temp_models_dir)

        assert watcher._observer is None

        watcher.start()

        # Should have created and started an observer
        assert watcher._observer is not None
        assert watcher.is_running

        watcher.stop()
        assert not watcher.is_running
