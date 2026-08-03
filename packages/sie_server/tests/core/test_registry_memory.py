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
    def test_touch_lru_updates_model_and_get_is_pure(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """touch_lru() marks a model recently used; get() is a pure read (#1541)."""
        mock_load_adapter.side_effect = [mock_adapter_factory(), mock_adapter_factory()]

        registry = ModelRegistry()

        # Add and load two models
        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Initially model-a is LRU (loaded first)
        assert registry.memory_manager.get_lru_model() == "model-a"

        # get() is a pure read — it must NOT change LRU order.
        registry.get("model-a")
        assert registry.memory_manager.get_lru_model() == "model-a"

        # touch_lru(model-a) makes model-b the LRU.
        registry.touch_lru("model-a")
        assert registry.memory_manager.get_lru_model() == "model-b"

        # touch_lru(model-b) makes model-a the LRU again.
        registry.touch_lru("model-b")
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
        telemetry = MagicMock()
        with patch("sie_server.core.registry.worker_telemetry", return_value=telemetry):
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
        telemetry.model_evicted.assert_called_once_with(model="model-a", reason="load_oom")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_sync_preload_pressure_eviction_uses_bounded_reason(
        self,
        mock_load_adapter: MagicMock,
        mock_adapter_factory: MagicMock,
    ) -> None:
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()
        mock_load_adapter.side_effect = [adapter_a, adapter_b]
        registry = ModelRegistry()
        registry.add_config(_make_config(name="model-a", hf_id="org/model-a"))
        registry.add_config(_make_config(name="model-b", hf_id="org/model-b"))

        with patch.object(registry._memory_manager, "should_evict_for_load", return_value=False):
            registry.load("model-a", device="cpu")

        telemetry = MagicMock()
        with (
            patch.object(registry._memory_manager, "should_evict_for_load", side_effect=[True, False]),
            patch("sie_server.core.registry.worker_telemetry", return_value=telemetry),
        ):
            registry.load("model-b", device="cpu")

        assert not registry.is_loaded("model-a")
        assert registry.is_loaded("model-b")
        telemetry.model_evicted.assert_called_once_with(model="model-a", reason="preload_pressure")

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


class TestRegistryPinnedModels:
    """Tests for pinned-model LRU/OOM protection."""

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            return mock

        return make_mock

    @patch("sie_server.core.model_loader.load_adapter")
    def test_empty_pinned_set_unchanged(self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock) -> None:
        """pinned_models=None preserves true LRU behaviour (regression guard)."""
        mock_load_adapter.side_effect = [mock_adapter_factory(), mock_adapter_factory()]

        registry = ModelRegistry(pinned_models=None)
        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # model-a loaded first, so it is the LRU
        assert registry.memory_manager.get_lru_model() == "model-a"

    @patch("sie_server.core.model_loader.load_adapter")
    def test_oom_evicts_non_pinned_keeps_pinned(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """OOM during load evicts the oldest non-pinned model, never the pinned one."""
        adapter_pinned = mock_adapter_factory()
        adapter_nonpinned = mock_adapter_factory()

        # Third adapter fails OOM on first try
        adapter_c_fail = mock_adapter_factory()
        adapter_c_fail.load.side_effect = RuntimeError("CUDA out of memory")

        # Retry adapter succeeds
        adapter_c_success = mock_adapter_factory()

        mock_load_adapter.side_effect = [adapter_pinned, adapter_nonpinned, adapter_c_fail, adapter_c_success]

        # pinned is "model-pinned", loaded first (oldest)
        registry = ModelRegistry(pinned_models=["model-pinned"])
        for name in ["model-pinned", "model-non-pinned", "model-c"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        registry.load("model-pinned", device="cuda:0")
        registry.load("model-non-pinned", device="cuda:0")

        # model-pinned is LRU but pinned; OOM must evict model-non-pinned instead
        assert registry.memory_manager.get_lru_model() == "model-pinned"

        registry.load("model-c", device="cuda:0")

        # Pinned model stays; non-pinned was evicted
        assert registry.is_loaded("model-pinned")
        assert not registry.is_loaded("model-non-pinned")
        assert registry.is_loaded("model-c")
        adapter_pinned.unload.assert_not_called()
        adapter_nonpinned.unload.assert_called_once()

    @patch("sie_server.core.model_loader.load_adapter")
    def test_oom_eviction_skips_pinned(self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock) -> None:
        """When only pinned models are loaded and OOM occurs, the error propagates."""
        adapter_pinned = mock_adapter_factory()
        adapter_fail = mock_adapter_factory()
        adapter_fail.load.side_effect = RuntimeError("CUDA out of memory")

        mock_load_adapter.side_effect = [adapter_pinned, adapter_fail]

        registry = ModelRegistry(pinned_models=["model-pinned"])
        for name in ["model-pinned", "model-other"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        registry.load("model-pinned", device="cuda:0")

        # Only pinned model loaded; OOM must propagate since there is nothing to evict
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            registry.load("model-other", device="cuda:0")

        # Pinned model must not have been touched
        assert registry.is_loaded("model-pinned")
        adapter_pinned.unload.assert_not_called()

    def test_resolve_model_id_preserves_profile_and_case(self) -> None:
        """resolve_model_id matches the FULL profile-qualified config key; :default folds to base."""
        registry = ModelRegistry()
        registry.add_config(_make_config(name="BAAI/bge-m3", hf_id="BAAI/bge-m3"))
        # Non-default profiles are first-class config entries (loader expands them).
        registry.add_config(_make_config(name="BAAI/bge-m3:fp8", hf_id="BAAI/bge-m3"))

        # Base id (exact + case-insensitive) -> base config.
        assert registry.resolve_model_id("BAAI/bge-m3") == "BAAI/bge-m3"
        assert registry.resolve_model_id("baai/bge-m3") == "BAAI/bge-m3"
        # ":default" folds to the base config (matches the gateway canonicalization).
        assert registry.resolve_model_id("baai/bge-m3:default") == "BAAI/bge-m3"
        # A non-default profile resolves to the VARIANT config, not the base.
        assert registry.resolve_model_id("baai/bge-m3:fp8") == "BAAI/bge-m3:fp8"
        # A profile with no variant config must NOT collapse to the base.
        assert registry.resolve_model_id("baai/bge-m3:missing") is None
        assert registry.resolve_model_id("org/missing") is None

    def test_pinned_disk_repo_ids_maps_sie_id_to_hf_id(self) -> None:
        """_pinned_disk_repo_ids resolves pinned sie_ids to their configs' hf_ids (case-insensitive)."""
        registry = ModelRegistry(pinned_models=["Org/Pinned"])
        registry.add_config(_make_config(name="Org/Pinned", hf_id="org/pinned-hf"))
        registry.add_config(_make_config(name="other/model", hf_id="other/hf"))

        # Only the pinned model's hf_id is returned (the disk cache keys by hf repo id).
        assert registry._pinned_disk_repo_ids() == {"org/pinned-hf"}

    def test_pinned_disk_repo_ids_empty_when_no_pinned_set(self) -> None:
        """No pinned set means no disk-eviction protection (regression guard)."""
        registry = ModelRegistry()
        registry.add_config(_make_config(name="org/model", hf_id="org/model-hf"))
        assert registry._pinned_disk_repo_ids() == set()

    def test_pinned_disk_repo_ids_profile_pin_protects_base_weights(self) -> None:
        """A profile-qualified pin protects the base model's disk weights (shared hf_id)."""
        registry = ModelRegistry(pinned_models=["Org/Model:fp8"])
        # Only the base config is present (the variant config may not have arrived yet);
        # the variant shares the base hf_id, so the base weights must be protected.
        registry.add_config(_make_config(name="Org/Model", hf_id="org/model-hf"))
        assert registry._pinned_disk_repo_ids() == {"org/model-hf"}

    def test_disk_cache_manager_wired_with_pinned_provider(self) -> None:
        """The registry feeds its pinned set into the disk cache manager's eviction guard."""
        registry = ModelRegistry(pinned_models=["org/pinned"])
        registry.add_config(_make_config(name="org/pinned", hf_id="org/pinned-hf"))

        assert registry._disk_cache_manager is not None
        # The callback wired at construction resolves to the pinned models' hf_ids.
        assert registry._disk_cache_manager._pinned_provider is not None
        assert registry._disk_cache_manager._pinned_provider() == {"org/pinned-hf"}


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
