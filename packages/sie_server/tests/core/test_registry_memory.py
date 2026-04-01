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


class TestRegistryMemoryManagerIntegration:
    """Tests for ModelRegistry + MemoryManager integration (LRU eviction)."""

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        """Create a factory that returns fresh mock adapters."""

        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            return mock

        return make_mock

    @patch("sie_server.core.model_loader.load_adapter")
    def test_load_registers_with_memory_manager(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Loading a model registers it with the memory manager."""
        mock_load_adapter.return_value = mock_adapter_factory()

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")

        # Model should be registered in memory manager
        assert registry.memory_manager.loaded_model_count == 1
        assert "test" in registry.memory_manager.loaded_models

    @patch("sie_server.core.model_loader.load_adapter")
    def test_unload_unregisters_from_memory_manager(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Unloading a model unregisters it from the memory manager."""
        mock_load_adapter.return_value = mock_adapter_factory()

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")
        registry.unload("test")

        # Model should be unregistered from memory manager
        assert registry.memory_manager.loaded_model_count == 0
        assert "test" not in registry.memory_manager.loaded_models

    @patch("sie_server.core.model_loader.load_adapter")
    def test_get_touches_model_for_lru(self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock) -> None:
        """Getting a model's adapter updates LRU tracking."""
        mock_load_adapter.side_effect = [mock_adapter_factory(), mock_adapter_factory()]

        registry = ModelRegistry()

        # Add and load two models
        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Initially model-a is LRU (loaded first)
        assert registry.memory_manager.get_lru_model() == "model-a"

        # Access model-a, now model-b should be LRU
        registry.get("model-a")
        assert registry.memory_manager.get_lru_model() == "model-b"

        # Access model-b, now model-a should be LRU again
        registry.get("model-b")
        assert registry.memory_manager.get_lru_model() == "model-a"

    @patch("sie_server.core.model_loader.load_adapter")
    def test_oom_triggers_lru_eviction_and_retry(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """OOM during load triggers LRU eviction and retry."""
        # First two loads succeed
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()

        # Third adapter fails with OOM on first try
        adapter_c_fail = mock_adapter_factory()
        oom_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        adapter_c_fail.load.side_effect = oom_error

        # Fourth adapter (retry) succeeds
        adapter_c_success = mock_adapter_factory()

        # Side effect: a, b, c_fail, c_success (retry creates new adapter)
        mock_load_adapter.side_effect = [adapter_a, adapter_b, adapter_c_fail, adapter_c_success]

        registry = ModelRegistry()

        # Add three model configs
        for name in ["model-a", "model-b", "model-c"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        # Load first two models
        registry.load("model-a", device="cuda:0")
        registry.load("model-b", device="cuda:0")

        # model-a is LRU
        assert registry.memory_manager.get_lru_model() == "model-a"

        # Load third model - should trigger OOM, evict model-a, then succeed on retry
        registry.load("model-c", device="cuda:0")

        # model-a should be evicted
        assert not registry.is_loaded("model-a")
        adapter_a.unload.assert_called_once()

        # model-c should be loaded (via retry adapter)
        assert registry.is_loaded("model-c")
        # First adapter failed, retry adapter succeeded
        adapter_c_fail.load.assert_called_once()
        adapter_c_success.load.assert_called_once()

        # Now only model-b and model-c are loaded
        assert len(registry.loaded_model_names) == 2
        assert set(registry.loaded_model_names) == {"model-b", "model-c"}

    @patch("sie_server.core.model_loader.load_adapter")
    def test_oom_with_no_models_to_evict_raises(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """OOM with no models to evict raises the original error."""
        adapter = mock_adapter_factory()
        adapter.load.side_effect = RuntimeError("CUDA out of memory")

        mock_load_adapter.return_value = adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)

        # No models loaded, so no LRU to evict - should raise
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            registry.load("test", device="cuda:0")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_non_oom_error_propagates(self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock) -> None:
        """Non-OOM RuntimeError propagates without eviction attempt."""
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()
        adapter_b.load.side_effect = RuntimeError("Some other error")

        mock_load_adapter.side_effect = [adapter_a, adapter_b]

        registry = ModelRegistry()

        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        registry.load("model-a", device="cpu")

        # Non-OOM error should propagate without evicting model-a
        with pytest.raises(RuntimeError, match="Some other error"):
            registry.load("model-b", device="cpu")

        # model-a should still be loaded (no eviction attempted)
        assert registry.is_loaded("model-a")
        adapter_a.unload.assert_not_called()


class TestRegistryOOMDetection:
    """Tests for _is_oom_error detection."""

    def test_cuda_oom_detected(self) -> None:
        """CUDA OOM error is detected."""
        registry = ModelRegistry()

        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert registry._is_oom_error(error) is True

    def test_mps_oom_detected(self) -> None:
        """MPS OOM error is detected."""
        registry = ModelRegistry()

        error = RuntimeError("MPS backend out of memory")
        assert registry._is_oom_error(error) is True

    def test_generic_oom_detected(self) -> None:
        """Generic OOM error is detected."""
        registry = ModelRegistry()

        error = RuntimeError("Cannot allocate memory for tensor")
        assert registry._is_oom_error(error) is True

    def test_allocation_failed_detected(self) -> None:
        """Allocation failed error is detected."""
        registry = ModelRegistry()

        error = RuntimeError("Failed to allocate 8GB")
        assert registry._is_oom_error(error) is True

    def test_non_oom_not_detected(self) -> None:
        """Non-OOM error is not detected as OOM."""
        registry = ModelRegistry()

        error = RuntimeError("Some other error")
        assert registry._is_oom_error(error) is False

    def test_case_insensitive_detection(self) -> None:
        """OOM detection is case insensitive."""
        registry = ModelRegistry()

        error = RuntimeError("OUT OF MEMORY - CUDA")
        assert registry._is_oom_error(error) is True
