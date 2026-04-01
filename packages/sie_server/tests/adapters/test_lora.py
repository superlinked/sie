"""Tests for LoRA support functionality.

Tests cover:
1. LoadedModel and LoadedLora dataclasses
2. LRU eviction for LoRAs
3. PEFTLoRAMixin methods
4. Registry LoRA management
5. Worker per-LoRA batching (covered in test_worker.py)
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.model_loader import DEFAULT_MAX_LORAS, LoadedLora, LoadedModel
from sie_server.types.responses import ErrorCode

# =============================================================================
# LoadedLora Tests
# =============================================================================


class TestLoadedLora:
    """Tests for LoadedLora dataclass."""

    def test_loaded_lora_basic_fields(self) -> None:
        """Test basic field initialization."""
        lora = LoadedLora(adapter_id="org/my-lora")
        assert lora.adapter_id == "org/my-lora"
        assert lora.memory_bytes == 0
        assert lora.peft_model is None
        assert lora.loading is False

    def test_loaded_lora_with_memory(self) -> None:
        """Test with memory specified."""
        lora = LoadedLora(
            adapter_id="org/my-lora",
            memory_bytes=1024 * 1024,  # 1MB
        )
        assert lora.memory_bytes == 1024 * 1024

    def test_loaded_lora_loading_state(self) -> None:
        """Test loading state tracking."""
        lora = LoadedLora(adapter_id="org/my-lora", loading=True)
        assert lora.loading is True

        # Simulate load completion
        lora.loading = False
        assert lora.loading is False


# =============================================================================
# LoadedModel LoRA Tracking Tests
# =============================================================================


class TestLoadedModelLora:
    """Tests for LoadedModel LoRA tracking."""

    def test_loras_ordered_dict(self) -> None:
        """Test that loras is an OrderedDict for LRU tracking."""
        loaded = LoadedModel(
            config=MagicMock(),
            adapter=MagicMock(),
            device="cuda:0",
        )
        assert isinstance(loaded.loras, OrderedDict)
        assert len(loaded.loras) == 0

    def test_max_loras_default(self) -> None:
        """Test default max_loras value."""
        loaded = LoadedModel(
            config=MagicMock(),
            adapter=MagicMock(),
            device="cuda:0",
        )
        assert loaded.max_loras == DEFAULT_MAX_LORAS

    def test_total_memory_includes_loras(self) -> None:
        """Test total_memory_bytes includes both base and LoRA memory."""
        loaded = LoadedModel(
            config=MagicMock(),
            adapter=MagicMock(),
            device="cuda:0",
            memory_bytes=100 * 1024 * 1024,  # 100MB base
        )

        # Add some LoRAs
        loaded.loras["lora1"] = LoadedLora(adapter_id="lora1", memory_bytes=5 * 1024 * 1024)
        loaded.loras["lora2"] = LoadedLora(adapter_id="lora2", memory_bytes=3 * 1024 * 1024)

        # Total should be base + all LoRAs
        expected = 100 * 1024 * 1024 + 5 * 1024 * 1024 + 3 * 1024 * 1024
        assert loaded.total_memory_bytes == expected

    def test_lora_lock_lazy_creation(self) -> None:
        """Test that LoRA lock is created lazily."""
        loaded = LoadedModel(
            config=MagicMock(),
            adapter=MagicMock(),
            device="cuda:0",
        )

        # Initially None
        assert loaded._lora_lock is None

        # Created on first access
        lock = loaded.get_lora_lock()
        assert lock is not None
        assert loaded._lora_lock is lock

        # Same lock returned on subsequent calls
        assert loaded.get_lora_lock() is lock

    def test_lora_lru_order_maintained(self) -> None:
        """Test that LoRA insertion order is maintained."""
        loaded = LoadedModel(
            config=MagicMock(),
            adapter=MagicMock(),
            device="cuda:0",
        )

        loaded.loras["first"] = LoadedLora(adapter_id="first")
        loaded.loras["second"] = LoadedLora(adapter_id="second")
        loaded.loras["third"] = LoadedLora(adapter_id="third")

        # Order should be maintained
        lora_names = list(loaded.loras.keys())
        assert lora_names == ["first", "second", "third"]

    def test_lora_lru_move_to_end_on_access(self) -> None:
        """Test that accessing a LoRA moves it to the end (most recent)."""
        loaded = LoadedModel(
            config=MagicMock(),
            adapter=MagicMock(),
            device="cuda:0",
        )

        loaded.loras["first"] = LoadedLora(adapter_id="first")
        loaded.loras["second"] = LoadedLora(adapter_id="second")
        loaded.loras["third"] = LoadedLora(adapter_id="third")

        # Access "first" - should move to end
        loaded.loras.move_to_end("first")

        lora_names = list(loaded.loras.keys())
        assert lora_names == ["second", "third", "first"]


# =============================================================================
# PEFTLoRAMixin Tests
# =============================================================================


class MockAdapterWithLoRA(PEFTLoRAMixin, ModelAdapter):
    """Mock adapter for testing PEFTLoRAMixin."""

    def __init__(self) -> None:
        self._model = None
        self._device = "cuda:0"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["dense"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims(dense=768)

    def load(self, device: str) -> None:
        self._device = device
        # Simulate loading a PyTorch model
        self._model = MagicMock()
        self._model.named_parameters = MagicMock(return_value=[])
        self._model.parameters = MagicMock(return_value=[])
        self._model.buffers = MagicMock(return_value=[])

    def unload(self) -> None:
        self._model = None


class TestPEFTLoRAMixin:
    """Tests for PEFTLoRAMixin."""

    def test_supports_lora_returns_true(self) -> None:
        """Test that PEFTLoRAMixin.supports_lora() returns True."""
        adapter = MockAdapterWithLoRA()
        assert adapter.supports_lora() is True

    def test_supports_hot_lora_reload_returns_true(self) -> None:
        """Test that PEFTLoRAMixin.supports_hot_lora_reload() returns True."""
        adapter = MockAdapterWithLoRA()
        assert adapter.supports_hot_lora_reload() is True

    def test_load_lora_requires_loaded_model(self) -> None:
        """Test that load_lora raises if model not loaded."""
        adapter = MockAdapterWithLoRA()
        # Don't call load() - model is None

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.load_lora("org/my-lora")

    def test_load_lora_first_time(self) -> None:
        """Test first LoRA load creates PeftModel."""
        with patch.dict("sys.modules", {"peft": MagicMock()}):
            adapter = MockAdapterWithLoRA()
            adapter.load("cuda:0")

            # Mock PeftModel class
            mock_peft_model = MagicMock()
            mock_peft_model.named_parameters.return_value = []

            with patch("peft.PeftModel") as mock_peft_model_cls:
                mock_peft_model_cls.from_pretrained.return_value = mock_peft_model

                result = adapter.load_lora("org/my-lora")

                # PeftModel.from_pretrained should be called
                mock_peft_model_cls.from_pretrained.assert_called_once()
                assert adapter._peft_model is mock_peft_model
                assert "org/my-lora" in adapter._loaded_loras
                assert isinstance(result, int)

    def test_load_lora_additional(self) -> None:
        """Test additional LoRA uses load_adapter."""
        with patch.dict("sys.modules", {"peft": MagicMock()}):
            adapter = MockAdapterWithLoRA()
            adapter.load("cuda:0")

            # Mock PeftModel class
            mock_peft_model = MagicMock()
            mock_peft_model.named_parameters.return_value = []

            with patch("peft.PeftModel") as mock_peft_model_cls:
                mock_peft_model_cls.from_pretrained.return_value = mock_peft_model

                # Load first LoRA
                adapter.load_lora("org/lora1")

                # Load second LoRA - should use load_adapter
                adapter.load_lora("org/lora2")

                mock_peft_model.load_adapter.assert_called_once_with("org/lora2", adapter_name="org/lora2")
                assert "org/lora1" in adapter._loaded_loras
                assert "org/lora2" in adapter._loaded_loras

    def test_load_lora_duplicate_skipped(self) -> None:
        """Test loading same LoRA twice is skipped."""
        adapter = MockAdapterWithLoRA()
        adapter.load("cuda:0")
        adapter._loaded_loras = {"org/my-lora"}

        # Should return 0 and not raise
        result = adapter.load_lora("org/my-lora")
        assert result == 0

    def test_set_active_lora_when_no_peft_model(self) -> None:
        """Test set_active_lora with no PeftModel is a no-op."""
        adapter = MockAdapterWithLoRA()
        adapter._peft_model = None

        # Should not raise
        adapter.set_active_lora("org/my-lora")

    def test_set_active_lora_to_none_disables_adapters(self) -> None:
        """Test set_active_lora(None) disables adapter layers."""
        adapter = MockAdapterWithLoRA()
        adapter._peft_model = MagicMock()
        adapter._loaded_loras = {"org/some-lora"}
        # Start with an active LoRA so switching to None will disable
        adapter._active_lora = "org/some-lora"

        adapter.set_active_lora(None)

        # Uses PEFT's disable_adapter_layers() instead of transformers' disable_adapters()
        adapter._peft_model.disable_adapter_layers.assert_called_once()
        assert adapter._active_lora is None

    def test_set_active_lora_enables_and_sets(self) -> None:
        """Test set_active_lora enables adapter layers and sets the adapter."""
        adapter = MockAdapterWithLoRA()
        adapter._peft_model = MagicMock()
        adapter._loaded_loras = {"org/my-lora"}

        adapter.set_active_lora("org/my-lora")

        # Uses PEFT's enable_adapter_layers() instead of transformers' enable_adapters()
        adapter._peft_model.enable_adapter_layers.assert_called_once()
        adapter._peft_model.set_adapter.assert_called_once_with("org/my-lora")
        assert adapter._active_lora == "org/my-lora"

    def test_set_active_lora_not_loaded_raises(self) -> None:
        """Test set_active_lora raises for unloaded LoRA."""
        adapter = MockAdapterWithLoRA()
        adapter._peft_model = MagicMock()
        adapter._loaded_loras = set()

        with pytest.raises(ValueError, match="not loaded"):
            adapter.set_active_lora("org/unknown")

    def test_unload_lora_not_loaded_raises(self) -> None:
        """Test unload_lora raises for unloaded LoRA."""
        adapter = MockAdapterWithLoRA()
        adapter._loaded_loras = set()

        with pytest.raises(ValueError, match="not loaded"):
            adapter.unload_lora("org/unknown")

    def test_unload_lora_deletes_adapter(self) -> None:
        """Test unload_lora deletes the adapter."""
        adapter = MockAdapterWithLoRA()
        mock_peft_model = MagicMock()
        adapter._peft_model = mock_peft_model
        adapter._loaded_loras = {"org/my-lora", "org/another-lora"}  # Keep one so model isn't unwrapped
        adapter._active_lora = None

        adapter.unload_lora("org/my-lora")

        mock_peft_model.delete_adapter.assert_called_once_with("org/my-lora")
        assert "org/my-lora" not in adapter._loaded_loras
        assert "org/another-lora" in adapter._loaded_loras  # Still present


# =============================================================================
# ErrorCode Tests
# =============================================================================


class TestLoraLoadingErrorCode:
    """Tests for LORA_LOADING error code."""

    def test_lora_loading_error_code_exists(self) -> None:
        """Test LORA_LOADING error code is defined."""
        assert hasattr(ErrorCode, "LORA_LOADING")
        assert ErrorCode.LORA_LOADING.value == "LORA_LOADING"


# =============================================================================
# Model Loader Profile LoRA Collection Tests
# =============================================================================


class TestModelLoaderProfileLoras:
    """Tests for ModelLoader._collect_profile_loras."""

    def test_collect_profile_loras_empty_profiles(self) -> None:
        """Test with no profiles."""
        from sie_server.core.model_loader import ModelLoader

        loader = ModelLoader(
            preprocessor_registry=MagicMock(),
            postprocessor_registry=MagicMock(),
            all_configs={},
        )

        config = MagicMock()
        config.profiles = {}

        result = loader._collect_profile_loras(config)
        assert result == set()

    def test_collect_profile_loras_with_loras(self) -> None:
        """Test collecting LoRAs from profiles."""
        from sie_server.core.model_loader import ModelLoader

        loader = ModelLoader(
            preprocessor_registry=MagicMock(),
            postprocessor_registry=MagicMock(),
            all_configs={},
        )

        # Create mock profile configs with adapter_options.runtime containing lora_id
        profile1 = MagicMock()
        profile1.adapter_options.runtime = {"lora_id": "org/lora1"}

        profile2 = MagicMock()
        profile2.adapter_options.runtime = {"lora_id": "org/lora2"}

        profile3 = MagicMock()
        profile3.adapter_options.runtime = {}  # No LoRA

        config = MagicMock()
        config.profiles = {
            "legal": profile1,
            "medical": profile2,
            "default": profile3,
        }

        result = loader._collect_profile_loras(config)
        assert result == {"org/lora1", "org/lora2"}

    def test_collect_profile_loras_deduplicates(self) -> None:
        """Test that duplicate LoRAs are deduplicated."""
        from sie_server.core.model_loader import ModelLoader

        loader = ModelLoader(
            preprocessor_registry=MagicMock(),
            postprocessor_registry=MagicMock(),
            all_configs={},
        )

        # Create mock profile configs with same LoRA
        profile1 = MagicMock()
        profile1.adapter_options.runtime = {"lora_id": "org/shared-lora"}

        profile2 = MagicMock()
        profile2.adapter_options.runtime = {"lora_id": "org/shared-lora"}  # Same LoRA

        config = MagicMock()
        config.profiles = {
            "legal": profile1,
            "medical": profile2,
        }

        result = loader._collect_profile_loras(config)
        assert result == {"org/shared-lora"}


# =============================================================================
# SDK LoraLoadingError Tests
# =============================================================================


class TestSDKLoraLoadingError:
    """Tests for SDK LoraLoadingError."""

    def test_lora_loading_error_importable(self) -> None:
        """Test LoraLoadingError can be imported."""
        from sie_sdk import LoraLoadingError

        error = LoraLoadingError("Test error", lora="org/my-lora", model="bge-m3")
        assert str(error) == "Test error"
        assert error.lora == "org/my-lora"
        assert error.model == "bge-m3"
