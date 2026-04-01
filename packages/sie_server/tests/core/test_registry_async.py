from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry


def _make_config(
    name: str = "test",
    hf_id: str | None = "org/test",
    dense_dim: int = 768,
    max_sequence_length: int | None = None,
) -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=hf_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=dense_dim))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
        max_sequence_length=max_sequence_length,
    )


@pytest.fixture(autouse=True)
def patch_ensure_model_cached():
    """Patch ensure_model_cached to avoid actual HF downloads in tests."""
    with patch("sie_sdk.cache.ensure_model_cached") as mock:
        mock.return_value = Path("/fake/cache/models--org--test")
        yield mock


class TestAsyncLoading:
    """Tests for async model loading (DESIGN.md Section 5.4)."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.capabilities.outputs = ["dense"]
        mock.memory_footprint.return_value = 1_000_000
        return mock

    @pytest.fixture
    def registry_with_model(self, mock_adapter: MagicMock) -> ModelRegistry:
        """Create registry with a model config ready to load."""
        registry = ModelRegistry()
        config = _make_config(name="test-model", hf_id="org/test")
        registry.add_config(config)
        return registry

    async def test_load_async_basic(self, registry_with_model: ModelRegistry) -> None:
        """Test basic async loading."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            adapter = await registry_with_model.load_async("test-model", "cpu")

            assert adapter is mock_adapter
            assert registry_with_model.is_loaded("test-model")
            mock_adapter.load.assert_called_once_with("cpu")

    async def test_load_async_returns_existing_if_loaded(self, registry_with_model: ModelRegistry) -> None:
        """Second call to load_async returns existing model without reloading."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            # First load
            adapter1 = await registry_with_model.load_async("test-model", "cpu")
            # Second load should return same adapter without reloading
            adapter2 = await registry_with_model.load_async("test-model", "cpu")

            assert adapter1 is adapter2
            # load_adapter should only be called once
            mock_load.assert_called_once()

    async def test_load_async_concurrent_same_model(self, registry_with_model: ModelRegistry) -> None:
        """Two concurrent loads for same model only load once."""
        import asyncio

        load_count = 0
        load_event = asyncio.Event()

        def slow_load(device: str) -> None:
            nonlocal load_count
            load_count += 1
            # Signal that load started
            load_event.set()

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_adapter.load = slow_load
            mock_load.return_value = mock_adapter

            # Start two concurrent loads
            task1 = asyncio.create_task(registry_with_model.load_async("test-model", "cpu"))
            task2 = asyncio.create_task(registry_with_model.load_async("test-model", "cpu"))

            adapter1, adapter2 = await asyncio.gather(task1, task2)

            # Both should return the same adapter
            assert adapter1 is adapter2
            # Load should only happen once
            assert load_count == 1

    async def test_is_unloading_flag(self, registry_with_model: ModelRegistry) -> None:
        """is_unloading returns correct state."""
        assert not registry_with_model.is_unloading("test-model")

    async def test_is_loading_flag_initial_state(self, registry_with_model: ModelRegistry) -> None:
        """is_loading returns False before load starts."""
        # Model is configured but not loading yet
        assert not registry_with_model.is_loading("test-model")

    async def test_is_loading_flag_cleared_after_load(self, registry_with_model: ModelRegistry) -> None:
        """is_loading returns False after load completes."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")

            # After load completes, is_loading should be False
            assert not registry_with_model.is_loading("test-model")
            assert registry_with_model.is_loaded("test-model")

    async def test_is_loading_flag_cleared_on_failure(self, registry_with_model: ModelRegistry) -> None:
        """is_loading is cleared even when load fails."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_load.side_effect = RuntimeError("Load failed")

            with pytest.raises(RuntimeError, match="Load failed"):
                await registry_with_model.load_async("test-model", "cpu")

            # After failure, is_loading should still be False
            assert not registry_with_model.is_loading("test-model")
            assert not registry_with_model.is_loaded("test-model")

    async def test_unload_async_drains_worker(self, registry_with_model: ModelRegistry) -> None:
        """unload_async stops worker before unloading adapter."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")

            # Get the worker
            worker = registry_with_model.get_worker("test-model")
            assert worker is not None

            # Start the worker
            await registry_with_model.start_worker("test-model")

            # Now unload
            await registry_with_model.unload_async("test-model")

            assert not registry_with_model.is_loaded("test-model")
            mock_adapter.unload.assert_called_once()

    async def test_unload_all_async(self, registry_with_model: ModelRegistry) -> None:
        """unload_all_async unloads all models."""
        # Add another model
        config2 = _make_config(name="model-2", hf_id="org/test2")
        registry_with_model.add_config(config2)

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")
            await registry_with_model.load_async("model-2", "cpu")

            assert len(registry_with_model.loaded_model_names) == 2

            await registry_with_model.unload_all_async()

            assert len(registry_with_model.loaded_model_names) == 0

    async def test_load_async_model_not_found_raises(self) -> None:
        """load_async raises KeyError for unknown model."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="not found"):
            await registry.load_async("unknown-model", "cpu")

    async def test_load_async_while_unloading_raises(self, registry_with_model: ModelRegistry) -> None:
        """load_async raises RuntimeError if model is being unloaded."""
        # Manually set unloading flag
        registry_with_model._unloading.add("test-model")

        with pytest.raises(RuntimeError, match="currently being unloaded"):
            await registry_with_model.load_async("test-model", "cpu")


class TestProactiveEviction:
    """Tests for proactive memory eviction (pre-load and background monitor)."""

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        """Create a factory that returns fresh mock adapters."""

        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            mock.memory_footprint.return_value = 1000
            return mock

        return make_mock

    @pytest.fixture
    def registry_with_models(self, mock_adapter_factory: MagicMock) -> ModelRegistry:
        """Create registry with 3 model configs."""
        from sie_server.core.memory import MemoryConfig

        registry = ModelRegistry(
            memory_config=MemoryConfig(pressure_threshold=0.85),
        )

        for name in ["model-a", "model-b", "model-c"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        return registry

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_pre_load_eviction_triggers_when_above_threshold(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Pre-load eviction evicts LRU when memory is above threshold."""
        adapters = [mock_adapter_factory() for _ in range(3)]
        mock_load_adapter.side_effect = adapters

        # Load first two models
        await registry_with_models.load_async("model-a", "cpu")
        await registry_with_models.load_async("model-b", "cpu")

        assert registry_with_models.is_loaded("model-a")
        assert registry_with_models.is_loaded("model-b")

        # Mock memory pressure (90% usage, threshold is 85%)
        with patch.object(registry_with_models._memory_manager, "check_pressure", side_effect=[True, False]):
            # Load third model - should trigger eviction of model-a (LRU)
            await registry_with_models.load_async("model-c", "cpu")

        # model-a should be evicted, model-b and model-c loaded
        assert not registry_with_models.is_loaded("model-a")
        assert registry_with_models.is_loaded("model-b")
        assert registry_with_models.is_loaded("model-c")

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_pre_load_eviction_loop_evicts_multiple(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Pre-load eviction can evict multiple models until below threshold."""
        adapters = [mock_adapter_factory() for _ in range(3)]
        mock_load_adapter.side_effect = adapters

        # Load first two models
        await registry_with_models.load_async("model-a", "cpu")
        await registry_with_models.load_async("model-b", "cpu")

        # Mock memory pressure that requires evicting both models
        # check_pressure: True, True, False (evict a, evict b, then ok)
        with patch.object(registry_with_models._memory_manager, "check_pressure", side_effect=[True, True, False]):
            await registry_with_models.load_async("model-c", "cpu")

        # Both model-a and model-b should be evicted
        assert not registry_with_models.is_loaded("model-a")
        assert not registry_with_models.is_loaded("model-b")
        assert registry_with_models.is_loaded("model-c")

    async def test_background_monitor_loop_runs(self) -> None:
        """Background monitor loop runs and checks pressure periodically."""
        import asyncio

        from sie_server.core.memory import MemoryConfig

        # Create registry with short check interval for testing
        registry = ModelRegistry(
            memory_config=MemoryConfig(
                pressure_threshold=0.85,
                memory_check_interval_s=0.005,  # 5ms for fast testing
            ),
        )

        # Track how many times check_pressure is called
        check_count = 0

        def counting_check() -> bool:
            nonlocal check_count
            check_count += 1
            return False  # No pressure, so no eviction needed

        registry._memory_manager.check_pressure = counting_check

        await registry.start_memory_monitor()
        try:
            # Wait for a few check cycles
            await asyncio.sleep(0.02)
            # Should have been called multiple times
            assert check_count >= 2
        finally:
            await registry.stop_memory_monitor()

    async def test_memory_monitor_starts_and_stops(self) -> None:
        """Memory monitor can be started and stopped cleanly."""
        from sie_server.core.memory import MemoryConfig

        registry = ModelRegistry(
            memory_config=MemoryConfig(memory_check_interval_s=0.1),
        )

        assert registry._monitor_task is None
        assert not registry._monitor_running

        await registry.start_memory_monitor()

        assert registry._monitor_task is not None
        assert registry._monitor_running

        await registry.stop_memory_monitor()

        assert registry._monitor_task is None
        assert not registry._monitor_running

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_adapter_unload_called_on_unload(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Adapter.unload() is called when model is unloaded.

        Memory cleanup (gc.collect + empty_cache) is the adapter's responsibility.
        See the memory management contract in ModelAdapter docstring (base.py).
        """
        mock_adapter = mock_adapter_factory()
        mock_load_adapter.return_value = mock_adapter

        await registry_with_models.load_async("model-a", "cuda:0")
        await registry_with_models.unload_async("model-a")

        mock_adapter.unload.assert_called_once()
