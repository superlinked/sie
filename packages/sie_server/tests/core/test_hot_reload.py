"""Tests for HotReloader and hot reload logic.

These tests are designed to be fast by:
- Testing internal methods directly
- Mocking the FileWatcher to avoid real file system watching
- Using mock.patch for asyncio.sleep to make it instant
- No time.sleep() or asyncio.sleep() calls in actual test code
"""

import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_server.core.hot_reload import (
    HotReloadConfig,
    HotReloader,
    ReloadResult,
    ReloadStatus,
)
from sie_server.core.registry import ModelRegistry
from sie_server.core.watcher import ChangeType, ModelChange


class TestReloadResult:
    """Tests for ReloadResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful reload result."""
        result = ReloadResult(
            model_name="test-model",
            status=ReloadStatus.SUCCESS,
            change_type=ChangeType.MODIFIED,
        )

        assert result.model_name == "test-model"
        assert result.status == ReloadStatus.SUCCESS
        assert result.change_type == ChangeType.MODIFIED
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed reload result."""
        result = ReloadResult(
            model_name="test-model",
            status=ReloadStatus.FAILED,
            change_type=ChangeType.MODIFIED,
            error="Something went wrong",
        )

        assert result.status == ReloadStatus.FAILED
        assert result.error == "Something went wrong"


class TestHotReloadConfig:
    """Tests for HotReloadConfig."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = HotReloadConfig()

        assert config.debounce_seconds == 1.0
        assert config.drain_timeout_seconds == 30.0
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = HotReloadConfig(
            debounce_seconds=0.5,
            drain_timeout_seconds=10.0,
            enabled=False,
        )

        assert config.debounce_seconds == 0.5
        assert config.drain_timeout_seconds == 10.0
        assert config.enabled is False


class TestHotReloader:
    """Tests for HotReloader - unit tests using mocks."""

    @pytest.fixture
    def temp_models_dir(self) -> Generator[Path, None, None]:
        """Create a temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            yield models_dir

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """Create a mock ModelRegistry."""
        registry = MagicMock(spec=ModelRegistry)
        registry.is_loaded.return_value = False
        registry.has_model.return_value = False
        registry.get_worker.return_value = None
        registry._configs = {}
        registry._model_dirs = {}
        return registry

    @pytest.fixture
    def reloader(self, temp_models_dir: Path, mock_registry: MagicMock) -> HotReloader:
        """Create a HotReloader for testing."""
        config = HotReloadConfig(debounce_seconds=0.1)
        return HotReloader(
            registry=mock_registry,
            models_dir=temp_models_dir,
            device="cpu",
            config=config,
        )

    def test_init(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test initialization."""
        reloader = HotReloader(mock_registry, temp_models_dir)

        assert reloader.models_dir.resolve() == temp_models_dir.resolve()
        assert not reloader.is_running

    @pytest.mark.asyncio
    async def test_start_stop(self, reloader: HotReloader) -> None:
        """Test starting and stopping the reloader."""
        assert not reloader.is_running

        # Mock the watcher to avoid real file system watching
        with patch.object(reloader._watcher, "start"), patch.object(reloader._watcher, "stop"):
            await reloader.start()
            assert reloader.is_running

            await reloader.stop()
            assert not reloader.is_running

    @pytest.mark.asyncio
    async def test_start_disabled(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test that disabled reloader doesn't start."""
        config = HotReloadConfig(enabled=False)
        reloader = HotReloader(mock_registry, temp_models_dir, config=config)

        await reloader.start()

        # Should not be running when disabled
        assert not reloader.is_running

    @pytest.mark.asyncio
    async def test_start_idempotent(self, reloader: HotReloader) -> None:
        """Test starting twice is a no-op."""
        with patch.object(reloader._watcher, "start"), patch.object(reloader._watcher, "stop"):
            await reloader.start()
            await reloader.start()  # Should not raise

            assert reloader.is_running
            await reloader.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, reloader: HotReloader) -> None:
        """Test stopping when not running is a no-op."""
        await reloader.stop()  # Should not raise
        assert not reloader.is_running

    def test_is_model_reloading(self, reloader: HotReloader) -> None:
        """Test checking if model is being reloaded."""
        assert not reloader.is_model_reloading("test-model")

        reloader._reloading.add("test-model")
        assert reloader.is_model_reloading("test-model")

        reloader._reloading.discard("test-model")
        assert not reloader.is_model_reloading("test-model")


class TestHotReloaderHandlers:
    """Tests for HotReloader change handlers - direct method testing."""

    @pytest.fixture
    def temp_models_dir(self) -> Generator[Path, None, None]:
        """Create a temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            yield models_dir

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """Create a mock ModelRegistry."""
        registry = MagicMock(spec=ModelRegistry)
        registry.is_loaded.return_value = False
        registry.has_model.return_value = False
        registry.get_worker.return_value = None
        registry._configs = {}
        registry._model_dirs = {}
        return registry

    @pytest.mark.asyncio
    async def test_handle_model_added(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test _handle_added registers new model config."""
        # Create flat YAML config file
        config_file = temp_models_dir / "new-model.yaml"
        config_file.write_text(
            "sie_id: new-model\n"
            "hf_id: BAAI/bge-m3\n"
            "inputs:\n"
            "  text: true\n"
            "tasks:\n"
            "  encode:\n"
            "    dense:\n"
            "      dim: 1024\n"
            "max_sequence_length: 512\n"
            "profiles:\n"
            "  default:\n"
            "    max_batch_tokens: 16384\n"
            "    adapter_path: sie_server.adapters.bge_m3:BGEM3Adapter\n"
            "    adapter_options:\n"
            "      loadtime: {}\n"
            "      runtime: {}\n"
        )

        config = HotReloadConfig(debounce_seconds=0)
        reloader = HotReloader(mock_registry, temp_models_dir, config=config)

        change = ModelChange(
            model_name="new-model",
            change_type=ChangeType.ADDED,
            path=config_file,
        )

        # Call handler directly
        result = await reloader._handle_model_added(change)

        assert result.status == ReloadStatus.SUCCESS
        mock_registry.add_config.assert_called_once()
        call_args = mock_registry.add_config.call_args
        assert call_args[0][0].name == "new-model"

    @pytest.mark.asyncio
    async def test_handle_model_modified_not_loaded(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test _handle_modified when model is not loaded."""
        # Create flat YAML config file
        config_file = temp_models_dir / "existing-model.yaml"
        config_file.write_text(
            "sie_id: existing-model\n"
            "hf_id: BAAI/bge-m3\n"
            "inputs:\n"
            "  text: true\n"
            "tasks:\n"
            "  encode:\n"
            "    dense:\n"
            "      dim: 1024\n"
            "max_sequence_length: 512\n"
            "profiles:\n"
            "  default:\n"
            "    max_batch_tokens: 16384\n"
            "    adapter_path: sie_server.adapters.bge_m3:BGEM3Adapter\n"
            "    adapter_options:\n"
            "      loadtime: {}\n"
            "      runtime: {}\n"
        )

        mock_registry.is_loaded.return_value = False
        mock_registry.has_model.return_value = True

        config = HotReloadConfig(debounce_seconds=0)
        reloader = HotReloader(mock_registry, temp_models_dir, config=config)

        change = ModelChange(
            model_name="existing-model",
            change_type=ChangeType.MODIFIED,
            path=config_file,
        )

        result = await reloader._handle_model_modified(change)

        assert result.status == ReloadStatus.SUCCESS
        # Should not unload (wasn't loaded)
        mock_registry.unload.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_model_modified_was_loaded(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test _handle_modified when model is loaded - triggers reload."""
        # Create flat YAML config file
        config_file = temp_models_dir / "loaded-model.yaml"
        config_file.write_text(
            "sie_id: loaded-model\n"
            "hf_id: BAAI/bge-m3\n"
            "inputs:\n"
            "  text: true\n"
            "tasks:\n"
            "  encode:\n"
            "    dense:\n"
            "      dim: 1024\n"
            "max_sequence_length: 512\n"
            "profiles:\n"
            "  default:\n"
            "    max_batch_tokens: 16384\n"
            "    adapter_path: sie_server.adapters.bge_m3:BGEM3Adapter\n"
            "    adapter_options:\n"
            "      loadtime: {}\n"
            "      runtime: {}\n"
        )

        mock_registry.is_loaded.return_value = True
        mock_registry.has_model.return_value = True

        # Mock worker with no pending requests
        mock_worker = MagicMock()
        mock_worker.is_running = True
        mock_worker.pending_count = 0
        mock_registry.get_worker.return_value = mock_worker

        config = HotReloadConfig(debounce_seconds=0)
        reloader = HotReloader(mock_registry, temp_models_dir, device="cpu", config=config)

        change = ModelChange(
            model_name="loaded-model",
            change_type=ChangeType.MODIFIED,
            path=config_file,
        )

        result = await reloader._handle_model_modified(change)

        assert result.status == ReloadStatus.SUCCESS
        mock_registry.unload.assert_called_with("loaded-model")
        mock_registry.load.assert_called_with("loaded-model", "cpu")

    @pytest.mark.asyncio
    async def test_handle_model_deleted(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test _handle_deleted unloads and removes model."""
        # Mock that model is registered and loaded
        mock_registry.has_model.return_value = True
        mock_registry.is_loaded.return_value = True

        # Mock worker with no pending requests
        mock_worker = MagicMock()
        mock_worker.is_running = True
        mock_worker.pending_count = 0
        mock_registry.get_worker.return_value = mock_worker

        # Setup _configs dict for deletion
        mock_registry._configs = {"delete-model": MagicMock()}
        mock_registry._model_dirs = {"delete-model": temp_models_dir / "delete-model"}

        config = HotReloadConfig(debounce_seconds=0)
        reloader = HotReloader(mock_registry, temp_models_dir, config=config)

        change = ModelChange(
            model_name="delete-model",
            change_type=ChangeType.DELETED,
            path=temp_models_dir / "delete-model.yaml",
        )

        result = await reloader._handle_model_deleted(change)

        assert result.status == ReloadStatus.SUCCESS
        mock_registry.unload.assert_called_with("delete-model")

    @pytest.mark.asyncio
    async def test_drain_requests(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test _drain_requests waits for pending requests to complete."""
        # Mock worker with pending requests that drain over time
        pending_counts = [3, 2, 1, 0]

        mock_worker = MagicMock()
        mock_worker.is_running = True

        def get_pending() -> int:
            if pending_counts:
                return pending_counts.pop(0)
            return 0

        type(mock_worker).pending_count = property(lambda self: get_pending())
        mock_registry.get_worker.return_value = mock_worker

        config = HotReloadConfig(debounce_seconds=0, drain_timeout_seconds=1.0)
        reloader = HotReloader(mock_registry, temp_models_dir, config=config)

        # Patch asyncio.sleep to be instant
        with patch("sie_server.core.hot_reload.asyncio.sleep", new_callable=AsyncMock):
            await reloader._drain_requests("test-model")

        # All pending counts should have been consumed (drain polled until 0)
        assert len(pending_counts) == 0

    @pytest.mark.asyncio
    async def test_drain_requests_timeout(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test _drain_requests handles timeout when requests don't drain."""
        # Mock worker that never drains
        mock_worker = MagicMock()
        mock_worker.is_running = True
        mock_worker.pending_count = 5  # Always has pending

        mock_registry.get_worker.return_value = mock_worker

        config = HotReloadConfig(debounce_seconds=0, drain_timeout_seconds=0.01)
        reloader = HotReloader(mock_registry, temp_models_dir, config=config)

        # Mock sleep to be instant so test completes quickly
        with patch("sie_server.core.hot_reload.asyncio.sleep", new_callable=AsyncMock):
            await reloader._drain_requests("test-model")

        # Should complete without hanging (timeout reached)


class TestModuleCacheClearing:
    """Tests for module cache clearing."""

    @pytest.fixture
    def temp_models_dir(self) -> Generator[Path, None, None]:
        """Create a temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            yield models_dir

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """Create a mock ModelRegistry."""
        registry = MagicMock(spec=ModelRegistry)
        registry.is_loaded.return_value = False
        registry.has_model.return_value = False
        registry.get_worker.return_value = None
        registry._configs = {}
        registry._model_dirs = {}
        return registry

    def test_clear_custom_adapter_cache(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test that custom adapter modules are cleared from cache."""
        # Create model with adapter
        model_dir = temp_models_dir / "custom-model"
        model_dir.mkdir()
        adapter_file = model_dir / "adapter.py"
        adapter_file.write_text("class CustomAdapter: pass")

        reloader = HotReloader(mock_registry, temp_models_dir)

        # Manually add a fake module to sys.modules
        fake_module = MagicMock()
        fake_module.__file__ = str(adapter_file)
        module_name = f"sie_custom_adapters.adapter_{id(adapter_file)}"
        sys.modules[module_name] = fake_module

        try:
            # Clear the cache
            reloader._clear_custom_adapter_cache("custom-model")

            # Module should be removed
            assert module_name not in sys.modules
        finally:
            # Cleanup in case test fails
            sys.modules.pop(module_name, None)


class TestFileChangeCallback:
    """Tests for the file change callback integration."""

    @pytest.fixture
    def temp_models_dir(self) -> Generator[Path, None, None]:
        """Create a temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            yield models_dir

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """Create a mock ModelRegistry."""
        registry = MagicMock(spec=ModelRegistry)
        registry.is_loaded.return_value = False
        registry.has_model.return_value = False
        registry.get_worker.return_value = None
        registry._configs = {}
        registry._model_dirs = {}
        return registry

    def test_on_file_change_queues_change(self, temp_models_dir: Path, mock_registry: MagicMock) -> None:
        """Test that _on_file_change queues changes for processing."""
        config = HotReloadConfig(debounce_seconds=0)
        reloader = HotReloader(mock_registry, temp_models_dir, config=config)

        change = ModelChange(
            model_name="test-model",
            change_type=ChangeType.MODIFIED,
            path=temp_models_dir / "test-model.yaml",
        )

        # Call the callback directly
        reloader._on_file_change(change)

        # Should be queued
        assert not reloader._reload_queue.empty()
        queued_change = reloader._reload_queue.get_nowait()
        assert queued_change.model_name == "test-model"
