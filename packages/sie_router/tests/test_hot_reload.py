from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sie_router.hot_reload import ConfigWatcher, WatcherConfig
from sie_router.model_registry import ModelRegistry


@pytest.fixture
def temp_config_dirs():
    """Create temporary directories with test bundle and model configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        bundles_dir = tmppath / "bundles"
        models_dir = tmppath / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()

        # Create initial bundle config (YAML with adapters)
        (bundles_dir / "default.yaml").write_text(
            "name: default\npriority: 10\ndefault: true\nadapters:\n  - sie_server.adapters.test\n"
        )

        # Create initial model config with profiles
        (models_dir / "test-org-initial-model.yaml").write_text(
            "name: test-org/initial-model\n"
            "hf_id: test-org/initial-model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.test:TestAdapter\n"
        )

        yield bundles_dir, models_dir


class TestConfigWatcher:
    """Tests for ConfigWatcher class."""

    def test_watcher_starts_and_stops(self, temp_config_dirs) -> None:
        """ConfigWatcher can be started and stopped cleanly."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        watcher = ConfigWatcher(registry, bundles_dir, models_dir)

        assert not watcher.is_running

        watcher.start()
        assert watcher.is_running

        watcher.stop()
        assert not watcher.is_running

    def test_watcher_double_start_is_safe(self, temp_config_dirs) -> None:
        """Starting an already running watcher is a no-op."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        watcher = ConfigWatcher(registry, bundles_dir, models_dir)
        watcher.start()
        watcher.start()  # Should not raise
        assert watcher.is_running

        watcher.stop()

    def test_watcher_double_stop_is_safe(self, temp_config_dirs) -> None:
        """Stopping an already stopped watcher is a no-op."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        watcher = ConfigWatcher(registry, bundles_dir, models_dir)
        watcher.start()
        watcher.stop()
        watcher.stop()  # Should not raise
        assert not watcher.is_running

    def test_reload_on_bundle_change(self, temp_config_dirs) -> None:
        """Adding a new bundle file triggers reload."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        # Use very short debounce for testing
        config = WatcherConfig(debounce_seconds=0.005)
        watcher = ConfigWatcher(registry, bundles_dir, models_dir, config=config)

        # Initial state
        assert "default" in registry.list_bundles()
        assert "new-bundle" not in registry.list_bundles()

        watcher.start()
        try:
            # Add a new bundle file
            (bundles_dir / "new-bundle.yaml").write_text(
                "name: new-bundle\npriority: 5\nadapters:\n  - sie_server.adapters.new\n"
            )

            # Wait for debounce + processing
            time.sleep(0.05)

            # Verify reload happened
            assert "new-bundle" in registry.list_bundles()
            # New bundle should be first (priority 5 < priority 10)
            assert registry.list_bundles()[0] == "new-bundle"
        finally:
            watcher.stop()

    def test_reload_on_model_change(self, temp_config_dirs) -> None:
        """Adding a new model config triggers reload."""
        bundles_dir, models_dir = temp_config_dirs

        # Add adapter to bundle first
        (bundles_dir / "default.yaml").write_text(
            "name: default\npriority: 10\ndefault: true\nadapters:\n  - sie_server.adapters.test\n"
        )

        registry = ModelRegistry(bundles_dir, models_dir)

        config = WatcherConfig(debounce_seconds=0.005)
        watcher = ConfigWatcher(registry, bundles_dir, models_dir, config=config)

        watcher.start()
        try:
            # Add a new model config with matching adapter
            (models_dir / "new-org-new-model.yaml").write_text(
                "name: new-org/new-model\n"
                "hf_id: new-org/new-model\n"
                "profiles:\n"
                "  default:\n"
                "    adapter_path: sie_server.adapters.test:TestAdapter\n"
            )

            # Wait for debounce + processing
            time.sleep(0.02)

            # Verify reload happened and model is tracked
            models = registry.list_models()
            assert "new-org/new-model" in models
        finally:
            watcher.stop()

    def test_debounce_multiple_changes(self, temp_config_dirs) -> None:
        """Multiple rapid changes are debounced into single reload."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        # Track reload calls
        reload_count = 0
        original_reload = registry.reload

        def counting_reload():
            nonlocal reload_count
            reload_count += 1
            original_reload()

        registry.reload = counting_reload

        config = WatcherConfig(debounce_seconds=0.04)
        watcher = ConfigWatcher(registry, bundles_dir, models_dir, config=config)

        watcher.start()
        try:
            # Make multiple rapid changes
            for i in range(5):
                (bundles_dir / f"bundle-{i}.yaml").write_text(f"name: bundle-{i}\npriority: {20 + i}\nadapters: []\n")
            # Changes are intentionally back-to-back to exercise debounce.

            # Wait for at least one reload to occur
            deadline = time.monotonic() + 1
            while reload_count < 1 and time.monotonic() < deadline:
                time.sleep(0.02)

            # Give another debounce window for any trailing events
            time.sleep(config.debounce_seconds * 2)

            # Should have debounced to 1-3 reloads
            assert reload_count <= 3, f"Expected debounced reloads, got {reload_count}"
        finally:
            watcher.stop()

    def test_ignores_non_config_files(self, temp_config_dirs) -> None:
        """Watcher ignores files that don't match config patterns."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        reload_count = 0
        original_reload = registry.reload

        def counting_reload():
            nonlocal reload_count
            reload_count += 1
            original_reload()

        registry.reload = counting_reload

        config = WatcherConfig(debounce_seconds=0.01)
        watcher = ConfigWatcher(registry, bundles_dir, models_dir, config=config)

        watcher.start()
        try:
            # Wait for any initial events from fixture setup to settle
            # On macOS with fsevents, initial file creation events may be queued
            # and debounced reloads scheduled. Wait longer than debounce_seconds.
            time.sleep(0.05)
            reload_count = 0  # Reset counter after settling

            # Create non-config files (should be ignored)
            (bundles_dir / "README.md").write_text("# Bundles")
            (bundles_dir / "notes.txt").write_text("Some notes")
            (models_dir / "cache.json").write_text("{}")

            time.sleep(0.05)

            # No reloads should have been triggered
            assert reload_count == 0
        finally:
            watcher.stop()

    def test_handles_missing_directory(self) -> None:
        """Watcher handles missing directories gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "nonexistent"

            # Only create bundles dir
            bundles_dir.mkdir()
            (bundles_dir / "default.yaml").write_text("name: default\npriority: 10\nadapters: []\n")

            registry = ModelRegistry(bundles_dir, models_dir)
            watcher = ConfigWatcher(registry, bundles_dir, models_dir)

            # Should start without error (with warning logged)
            watcher.start()
            assert watcher.is_running
            watcher.stop()


class TestConfigWatcherPatterns:
    """Tests for config file pattern matching."""

    def test_bundle_pattern_yaml(self, temp_config_dirs) -> None:
        """Watches *.yaml files in bundles directory."""
        bundles_dir, models_dir = temp_config_dirs

        watcher = ConfigWatcher(
            MagicMock(),
            bundles_dir,
            models_dir,
            config=WatcherConfig(bundle_patterns=("*.yaml",)),
        )

        # Should match
        assert watcher._is_config_file(bundles_dir / "default.yaml")
        assert watcher._is_config_file(bundles_dir / "new.yaml")

        # Should not match
        assert not watcher._is_config_file(bundles_dir / "readme.md")
        assert not watcher._is_config_file(bundles_dir / "some.toml")

    def test_model_pattern_yaml(self, temp_config_dirs) -> None:
        """Watches *.yaml files in models directory."""
        bundles_dir, models_dir = temp_config_dirs

        watcher = ConfigWatcher(
            MagicMock(),
            bundles_dir,
            models_dir,
            config=WatcherConfig(model_patterns=("*.yaml",)),
        )

        # Should match - flat YAML files directly in models_dir
        assert watcher._is_config_file(models_dir / "org-model.yaml")

        # Should not match
        assert not watcher._is_config_file(models_dir / "model.safetensors")
        assert not watcher._is_config_file(models_dir / "readme.md")


class TestConfigWatcherWithMockedRegistry:
    """Tests using mocked ModelRegistry to verify reload behavior."""

    def test_reload_called_on_file_change(self, temp_config_dirs) -> None:
        """Verify reload() is called when config files change."""
        bundles_dir, models_dir = temp_config_dirs

        mock_registry = MagicMock()
        mock_registry.bundles_dir = bundles_dir
        mock_registry.models_dir = models_dir
        mock_registry.list_bundles.return_value = ["default"]
        mock_registry.list_models.return_value = ["test/model"]

        config = WatcherConfig(debounce_seconds=0.005)
        watcher = ConfigWatcher(mock_registry, bundles_dir, models_dir, config=config)

        watcher.start()
        try:
            # Modify a bundle file
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.updated\n"
            )

            time.sleep(0.02)

            # Verify reload was called
            mock_registry.reload.assert_called()
        finally:
            watcher.stop()

    def test_reload_failure_is_logged_not_raised(self, temp_config_dirs) -> None:
        """Reload failures are logged but don't crash the watcher."""
        bundles_dir, models_dir = temp_config_dirs

        mock_registry = MagicMock()
        mock_registry.bundles_dir = bundles_dir
        mock_registry.models_dir = models_dir
        mock_registry.reload.side_effect = ValueError("Config parse error")
        mock_registry.list_bundles.return_value = []
        mock_registry.list_models.return_value = []

        config = WatcherConfig(debounce_seconds=0.005)
        watcher = ConfigWatcher(mock_registry, bundles_dir, models_dir, config=config)

        watcher.start()
        try:
            # Modify a bundle file
            (bundles_dir / "default.yaml").write_text("invalid: content\n")

            time.sleep(0.05)

            # Watcher should still be running
            assert watcher.is_running

            # Reload should have been attempted
            mock_registry.reload.assert_called()
        finally:
            watcher.stop()
