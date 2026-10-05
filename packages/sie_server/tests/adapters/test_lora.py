"""Tests for LoRA support functionality.

Tests cover:
1. LoadedModel and LoadedLora dataclasses
2. LRU eviction for LoRAs
3. PEFTLoRAMixin methods
4. Registry LoRA management
5. Worker per-LoRA batching (covered in test_worker.py)
6. Honest capability + typed rejection for adapters whose forward cannot
   honor a LoRA (bert_flash, nomic_flash — LoRA capability audit)
7. Staging-side target_modules validation (audit §3 negative control)
8. Refusal of LoRAs that train biases (per-request isolation)
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.adapters.bert_flash import BertFlashAdapter
from sie_server.adapters.nomic_flash import NomicFlashAdapter
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin, validate_lora_target_modules
from sie_server.config.model import AdapterOptions, EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.model_loader import DEFAULT_MAX_LORAS, LoadedLora, LoadedModel, ModelLoader
from sie_server.core.registry import ModelRegistry
from sie_server.types.responses import ErrorCode
from torch import nn

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


def _bias_free_peft() -> MagicMock:
    peft = MagicMock()
    peft.PeftConfig.from_pretrained.return_value = SimpleNamespace(bias="none", target_modules=None)
    return peft


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
        with patch.dict("sys.modules", {"peft": _bias_free_peft()}):
            adapter = MockAdapterWithLoRA()
            adapter.load("cuda:0")

            # Mock PeftModel class
            mock_peft_model = MagicMock()
            mock_peft_model.named_parameters.return_value = []

            with patch("peft.PeftModel") as mock_peft_model_cls:
                mock_peft_model_cls.from_pretrained.return_value = mock_peft_model

                base_model = adapter._model
                result = adapter.load_lora("org/my-lora")
                peft_adapter_name = adapter._lora_adapter_names["org/my-lora"]

                # PeftModel.from_pretrained should be called
                mock_peft_model_cls.from_pretrained.assert_called_once_with(
                    base_model,
                    "org/my-lora",
                    adapter_name=peft_adapter_name,
                    revision=None,
                )
                assert peft_adapter_name.startswith("sie_lora_")
                assert "/" not in peft_adapter_name
                assert "." not in peft_adapter_name
                assert adapter._peft_model is mock_peft_model
                assert "org/my-lora" in adapter._loaded_loras
                assert isinstance(result, int)

    def test_load_lora_additional(self) -> None:
        """Test additional LoRA uses load_adapter."""
        with patch.dict("sys.modules", {"peft": _bias_free_peft()}):
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

                second_adapter_name = adapter._lora_adapter_names["org/lora2"]
                mock_peft_model.load_adapter.assert_called_once_with(
                    "org/lora2", adapter_name=second_adapter_name, revision=None
                )
                assert "org/lora1" in adapter._loaded_loras
                assert "org/lora2" in adapter._loaded_loras
                assert adapter._lora_adapter_names["org/lora1"] != second_adapter_name

    def test_load_lora_aliases_dotted_hf_repo_id_for_peft(self) -> None:
        """Test LoRA ids with dots/slashes are not passed to PEFT as adapter names."""
        with patch.dict("sys.modules", {"peft": _bias_free_peft()}):
            adapter = MockAdapterWithLoRA()
            adapter.load("cuda:0")

            mock_peft_model = MagicMock()
            mock_peft_model.named_parameters.return_value = []

            with patch("peft.PeftModel") as mock_peft_model_cls:
                mock_peft_model_cls.from_pretrained.return_value = mock_peft_model

                lora_id = "gauravprasadgp/qwen3-embedding_0.6B_lora"
                adapter.load_lora(lora_id)

                peft_adapter_name = adapter._lora_adapter_names[lora_id]
                assert peft_adapter_name != lora_id
                assert "gauravprasadgp_qwen3_embedding_0_6B_lora" in peft_adapter_name
                assert "__" in peft_adapter_name
                assert "/" not in peft_adapter_name
                assert "." not in peft_adapter_name
                mock_peft_model_cls.from_pretrained.assert_called_once()
                assert mock_peft_model_cls.from_pretrained.call_args.kwargs["adapter_name"] == peft_adapter_name
                assert lora_id in adapter._loaded_loras

    def test_peft_adapter_name_uses_hash_suffix_to_avoid_sanitized_collisions(self) -> None:
        """Test readable aliases remain distinct when sanitized names collide."""
        dotted = PEFTLoRAMixin._peft_adapter_name("org/a.b-lora")
        underscored = PEFTLoRAMixin._peft_adapter_name("org/a_b-lora")

        assert dotted != underscored
        assert dotted.startswith("sie_lora_org_a_b_lora__")
        assert underscored.startswith("sie_lora_org_a_b_lora__")

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
        adapter._lora_adapter_names = {"org/my-lora": "sie_lora_test"}

        adapter.set_active_lora("org/my-lora")

        # Uses PEFT's enable_adapter_layers() instead of transformers' enable_adapters()
        adapter._peft_model.enable_adapter_layers.assert_called_once()
        adapter._peft_model.set_adapter.assert_called_once_with("sie_lora_test")
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
        adapter._lora_adapter_names = {
            "org/my-lora": "sie_lora_test",
            "org/another-lora": "sie_lora_other",
        }
        adapter._active_lora = None

        adapter.unload_lora("org/my-lora")

        mock_peft_model.delete_adapter.assert_called_once_with("sie_lora_test")
        assert "org/my-lora" not in adapter._loaded_loras
        assert "org/my-lora" not in adapter._lora_adapter_names
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
        config.lora_revisions.return_value = {}

        result = loader._collect_profile_loras(config)
        assert result == {}

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
        config.lora_revisions.return_value = {}

        result = loader._collect_profile_loras(config)
        assert result == {"org/lora1": None, "org/lora2": None}

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
        config.lora_revisions.return_value = {}

        result = loader._collect_profile_loras(config)
        assert result == {"org/shared-lora": None}

    @staticmethod
    def _make_loader() -> Any:
        from sie_server.core.model_loader import ModelLoader

        return ModelLoader(
            preprocessor_registry=MagicMock(),
            postprocessor_registry=MagicMock(),
            all_configs={},
        )

    @staticmethod
    def _profile(runtime: dict | None = None, loadtime: dict | None = None) -> MagicMock:
        profile = MagicMock()
        profile.adapter_options.runtime = runtime or {}
        profile.adapter_options.loadtime = loadtime or {}
        return profile

    @staticmethod
    def _config(profiles: dict, revisions: dict | None = None) -> MagicMock:
        config = MagicMock()
        config.profiles = profiles
        config.lora_revisions.return_value = revisions or {}
        return config

    def test_collect_canonical_loadtime_lora_paths_list(self) -> None:
        """Canonical key: adapter_options.loadtime["lora_paths"] as a list of paths."""
        loader = self._make_loader()
        config = self._config({"byo": self._profile(loadtime={"lora_paths": ["org/lora-a", "org/lora-b"]})})

        assert loader._collect_profile_loras(config) == {"org/lora-a": None, "org/lora-b": None}

    def test_collect_canonical_loadtime_lora_paths_dict(self) -> None:
        """The sglang dict shape (served-name -> path) contributes its paths."""
        loader = self._make_loader()
        config = self._config({"byo": self._profile(loadtime={"lora_paths": {"banking": "org/lora-a"}})})

        assert loader._collect_profile_loras(config) == {"org/lora-a": None}

    def test_collect_pinned_dict_entry_surfaces_revision(self) -> None:
        """A pinned {id, revision} dict-form value yields (id -> revision) (#2113)."""
        loader = self._make_loader()
        sha = "a" * 40
        config = self._config(
            {"byo": self._profile(loadtime={"lora_paths": {"banking": {"id": "org/lora-a", "revision": sha}}})},
            revisions={"org/lora-a": sha},
        )

        assert loader._collect_profile_loras(config) == {"org/lora-a": sha}

    def test_collect_canonical_and_legacy_alias_union(self) -> None:
        """Canonical loadtime.lora_paths and the deprecated runtime.lora_id merge."""
        loader = self._make_loader()
        config = self._config(
            {
                "new-style": self._profile(loadtime={"lora_paths": ["org/lora-new"]}),
                "legacy": self._profile(runtime={"lora_id": "org/lora-legacy"}),
            }
        )

        assert loader._collect_profile_loras(config) == {"org/lora-new": None, "org/lora-legacy": None}

    def test_include_loadtime_paths_false_skips_canonical_only(self) -> None:
        """Engine-owned adapters (sglang) must not double-load loadtime.lora_paths.

        The loader passes include_loadtime_paths=adapter.supports_hot_lora_reload();
        sglang consumes its loadtime declarations at engine launch itself, so
        only the legacy runtime.lora_id survives collection for it.
        """
        loader = self._make_loader()
        config = self._config(
            {
                "engine-owned": self._profile(
                    runtime={"lora_id": "org/lora-legacy"},
                    loadtime={"lora_paths": ["org/lora-engine"]},
                ),
            }
        )

        result = loader._collect_profile_loras(config, include_loadtime_paths=False)
        assert result == {"org/lora-legacy": None}

    def test_collect_ignores_empty_lora_paths_entries(self) -> None:
        """Empty entries and non-list/dict shapes are skipped, not crashed on."""
        loader = self._make_loader()
        config = self._config(
            {
                "empties": self._profile(loadtime={"lora_paths": ["", None]}),
                "scalar-shape": self._profile(loadtime={"lora_paths": "org/not-a-list"}),
            }
        )

        assert loader._collect_profile_loras(config) == {}


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


# =============================================================================
# Honest LoRA capability for forwards that cannot honor a LoRA (audit fix)
# =============================================================================


class TestBrokenFlashAdaptersHonestLoraCapability:
    """bert_flash and nomic_flash must NOT advertise LoRA support.

    LoRA capability audit: ``bert_flash`` fuses Q/K/V weights at load and runs
    attention via ``functional.linear`` on the cached tensors; ``nomic_flash``
    hand-loads raw safetensors into tensor dicts with no wrappable
    ``self._model``. In both cases a PEFT LoRA would be silently dropped
    (served without the customer's adapter, no error) — the worst failure
    class. Honest capability = base-default ``supports_lora() == False`` so
    the request path rejects with the typed 400 instead.

    Constructors only — no model weights are loaded (no heavy local compute).
    """

    @pytest.mark.parametrize("adapter_cls", [BertFlashAdapter, NomicFlashAdapter])
    def test_supports_lora_is_false(self, adapter_cls: type) -> None:
        adapter = adapter_cls(model_name_or_path="stub/model")
        assert adapter.supports_lora() is False
        assert adapter.supports_hot_lora_reload() is False

    @pytest.mark.parametrize("adapter_cls", [BertFlashAdapter, NomicFlashAdapter])
    def test_not_a_peft_mixin(self, adapter_cls: type) -> None:
        """The PEFT mixin must not be re-added without fixing the forward."""
        assert not issubclass(adapter_cls, PEFTLoRAMixin)

    @pytest.mark.parametrize("adapter_cls", [BertFlashAdapter, NomicFlashAdapter])
    def test_load_lora_raises_not_implemented(self, adapter_cls: type) -> None:
        adapter = adapter_cls(model_name_or_path="stub/model")
        with pytest.raises(NotImplementedError, match="does not support LoRA"):
            adapter.load_lora("org/customer-lora")
        with pytest.raises(NotImplementedError, match="does not support LoRA"):
            adapter.unload_lora("org/customer-lora")

    @pytest.mark.parametrize("adapter_cls", [BertFlashAdapter, NomicFlashAdapter])
    def test_set_active_lora_is_noop(self, adapter_cls: type) -> None:
        """Base-class no-op — the worker may call this with lora=None freely."""
        adapter = adapter_cls(model_name_or_path="stub/model")
        assert adapter.set_active_lora(None) is None


class TestLoraRejectionOnNonSupportingAdapters:
    """The registry must raise the typed ValueError for these adapters.

    This is the real rejection seam: ``api/encode.py`` translates this exact
    ``ValueError`` into HTTP 400 INVALID_INPUT (locked by
    ``tests/api/test_encode_endpoint.py::TestEncodeLoraRouting::``
    ``test_lora_on_non_supporting_model_returns_typed_400``). Together they
    prove a lora_id aimed at bert_flash/nomic_flash is rejected loudly instead
    of silently served from base weights.
    """

    @pytest.mark.parametrize("adapter_cls", [BertFlashAdapter, NomicFlashAdapter])
    async def test_ensure_lora_loaded_async_raises_value_error(self, adapter_cls: type) -> None:
        registry = ModelRegistry()
        adapter = adapter_cls(model_name_or_path="stub/model")
        loaded = MagicMock()
        loaded.adapter = adapter
        registry._configs["stub-model"] = MagicMock()
        registry._loaded["stub-model"] = loaded

        with pytest.raises(ValueError, match="does not support LoRA"):
            await registry.ensure_lora_loaded_async("stub-model", "org/customer-lora")


# =============================================================================
# Staging-side target_modules validation (audit §3 negative control)
# =============================================================================


class TestValidateLoraTargetModules:
    """Direct tests for validate_lora_target_modules."""

    CALLED = frozenset({"query", "key", "value", "dense"})

    def test_no_intersection_rejects(self) -> None:
        """The audit's negative control: LoRA on modules the forward never calls."""
        with pytest.raises(ValueError, match="silently ignored"):
            validate_lora_target_modules("org/lora", ["pooler.dense2", "classifier"], self.CALLED)

    def test_intersection_passes(self) -> None:
        validate_lora_target_modules("org/lora", ["query", "value"], self.CALLED)

    def test_partial_intersection_passes(self) -> None:
        """One honored target is enough to have an effect (not zero-effect)."""
        validate_lora_target_modules("org/lora", ["classifier", "query"], self.CALLED)

    def test_dotted_target_matches_by_leaf_name(self) -> None:
        validate_lora_target_modules(
            "org/lora",
            ["encoder.layer.0.attention.self.query"],
            self.CALLED,
        )

    def test_set_shape_accepted(self) -> None:
        validate_lora_target_modules("org/lora", {"value"}, self.CALLED)

    def test_regex_string_matching_a_called_module_passes(self) -> None:
        validate_lora_target_modules("org/lora", ".*query", self.CALLED)

    def test_regex_string_matching_nothing_warns_but_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        """Regexes may target full dotted paths we cannot resolve — warn only."""
        with caplog.at_level("WARNING"):
            validate_lora_target_modules("org/lora", r"pooler\.dense2", self.CALLED)
        assert "matches none" in caplog.text

    def test_empty_target_modules_skips_check(self) -> None:
        validate_lora_target_modules("org/lora", None, self.CALLED)
        validate_lora_target_modules("org/lora", [], self.CALLED)

    def test_empty_called_set_skips_check(self) -> None:
        validate_lora_target_modules("org/lora", ["anything"], frozenset())


class MockAdapterWithCalledModules(MockAdapterWithLoRA):
    """PEFT mixin adapter declaring the modules its forward actually calls."""

    lora_called_module_names = frozenset({"query", "value"})


class TestMixinTargetModulesValidation:
    """load_lora must consult the LoRA's PEFT config against the called set."""

    @staticmethod
    def _peft_module(target_modules: object) -> MagicMock:
        peft = MagicMock()
        peft.PeftConfig.from_pretrained.return_value = SimpleNamespace(bias="none", target_modules=target_modules)
        mock_peft_model = MagicMock()
        mock_peft_model.named_parameters.return_value = []
        peft.PeftModel.from_pretrained.return_value = mock_peft_model
        return peft

    def test_mismatched_lora_rejected_before_wrapping(self) -> None:
        peft = self._peft_module(["pooler.dense2"])
        with patch.dict("sys.modules", {"peft": peft}):
            adapter = MockAdapterWithCalledModules()
            adapter.load("cuda:0")

            with pytest.raises(ValueError, match="silently ignored"):
                adapter.load_lora("org/mismatched-lora")

            # Rejected before any PEFT wrapping or tracking.
            peft.PeftModel.from_pretrained.assert_not_called()
            assert "org/mismatched-lora" not in adapter._loaded_loras
            assert adapter._peft_model is None

    def test_matching_lora_loads(self) -> None:
        peft = self._peft_module(["query"])
        with patch.dict("sys.modules", {"peft": peft}):
            adapter = MockAdapterWithCalledModules()
            adapter.load("cuda:0")

            adapter.load_lora("org/matching-lora")

            peft.PeftConfig.from_pretrained.assert_called_once_with("org/matching-lora", revision=None)
            peft.PeftModel.from_pretrained.assert_called_once()
            assert "org/matching-lora" in adapter._loaded_loras

    def test_config_fetch_failure_fails_the_load_before_wrapping(self) -> None:
        """Without the config the bias cannot be checked, so the load fails."""
        peft = self._peft_module(["query"])
        peft.PeftConfig.from_pretrained.side_effect = RuntimeError("offline")
        with patch.dict("sys.modules", {"peft": peft}):
            adapter = MockAdapterWithCalledModules()
            adapter.load("cuda:0")

            with pytest.raises(RuntimeError, match="offline"):
                adapter.load_lora("org/unfetchable-lora")

            peft.PeftModel.from_pretrained.assert_not_called()
            assert "org/unfetchable-lora" not in adapter._loaded_loras
            assert adapter._peft_model is None

    def test_default_none_declaration_skips_check(self) -> None:
        """Standard-forward adapters (no declaration) skip the target_modules check."""
        peft = self._peft_module(["anything"])
        with patch.dict("sys.modules", {"peft": peft}):
            adapter = MockAdapterWithLoRA()
            adapter.load("cuda:0")

            adapter.load_lora("org/any-lora")

            peft.PeftConfig.from_pretrained.assert_called_once_with("org/any-lora", revision=None)
            peft.PeftModel.from_pretrained.assert_called_once()
            assert "org/any-lora" in adapter._loaded_loras


# =============================================================================
# LoRAs that train biases (per-request isolation)
# =============================================================================


class TestMixinBiasRefusal:
    """load_lora refuses a LoRA that trains biases before PEFT sees the model."""

    @staticmethod
    def _peft_module(biases: dict[str, str]) -> MagicMock:
        peft = MagicMock()
        peft.PeftConfig.from_pretrained.side_effect = lambda path, revision=None: SimpleNamespace(
            bias=biases[path], target_modules=None
        )
        mock_peft_model = MagicMock()
        mock_peft_model.named_parameters.return_value = []
        peft.PeftModel.from_pretrained.return_value = mock_peft_model
        return peft

    @pytest.mark.parametrize("bias", ["all", "lora_only"])
    def test_first_lora_refused_before_wrapping(self, bias: str) -> None:
        revision = "b" * 40
        peft = self._peft_module({"org/bias-lora": bias})
        with patch.dict("sys.modules", {"peft": peft}):
            adapter = MockAdapterWithLoRA()
            adapter.load("cuda:0")
            base_model = adapter._model

            with pytest.raises(ValueError, match=f"bias={bias!r}"):
                adapter.load_lora("org/bias-lora", revision=revision)

            peft.PeftConfig.from_pretrained.assert_called_once_with("org/bias-lora", revision=revision)
            peft.PeftModel.from_pretrained.assert_not_called()
            assert adapter._peft_model is None
            assert adapter._model is base_model
            assert "org/bias-lora" not in adapter._loaded_loras

    def test_additional_lora_refused_before_load_adapter(self) -> None:
        peft = self._peft_module({"org/plain-lora": "none", "org/bias-lora": "all"})
        with patch.dict("sys.modules", {"peft": peft}):
            adapter = MockAdapterWithLoRA()
            adapter.load("cuda:0")
            adapter.load_lora("org/plain-lora")

            with pytest.raises(ValueError, match="bias='all'"):
                adapter.load_lora("org/bias-lora")

            peft.PeftModel.from_pretrained.return_value.load_adapter.assert_not_called()
            assert adapter._loaded_loras == {"org/plain-lora"}


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.query(x)))


def _tiny_encoder() -> _TinyEncoder:
    model = _TinyEncoder()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return model


def _probe() -> torch.Tensor:
    return torch.randn(3, 4, generator=torch.Generator().manual_seed(1))


def _save_tiny_lora(path: Path, *, bias: str) -> str:
    trained = get_peft_model(
        _tiny_encoder(),
        LoraConfig(r=2, target_modules=["query"], bias=bias, init_lora_weights=False),
    )
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if parameter.requires_grad and name.endswith(".bias"):
                parameter.add_(1.0)
    trained.save_pretrained(str(path))
    return str(path)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _unchanged(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> bool:
    state = model.state_dict()
    return state.keys() == snapshot.keys() and all(torch.equal(state[name], snapshot[name]) for name in snapshot)


def _tiny_config(lora_paths: list[str] | None = None) -> ModelConfig:
    loadtime = {"lora_paths": lora_paths} if lora_paths else {}
    return ModelConfig(
        sie_id="t/tiny",
        hf_id="org/tiny",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=2))),
        profiles={
            "default": ProfileConfig(
                adapter_path="mod:Cls",
                max_batch_tokens=8,
                adapter_options=AdapterOptions(loadtime=loadtime),
            )
        },
    )


class TinyLoRAAdapter(MockAdapterWithLoRA):
    """PEFT mixin adapter over a real two-layer torch model."""

    def load(self, device: str) -> None:
        self._device = device
        self._model = _tiny_encoder()


class TestBiasLoraIsolation:
    """Real PEFT: a LoRA that trains biases never reaches the shared base model."""

    @pytest.mark.filterwarnings("ignore:Careful, disabling adapter layers:UserWarning")
    def test_peft_keeps_trained_biases_when_the_adapter_is_disabled(self, tmp_path: Path) -> None:
        """The reason for the refusal: disabling such a LoRA does not restore the base outputs."""
        lora = _save_tiny_lora(tmp_path, bias="all")
        model = _tiny_encoder()
        expected = model(_probe())

        peft_model = PeftModel.from_pretrained(model, lora)
        peft_model.disable_adapter_layers()

        assert not torch.equal(peft_model(_probe()), expected)

    @pytest.mark.parametrize("bias", ["all", "lora_only"])
    def test_bias_lora_refused_with_the_base_model_unchanged(self, tmp_path: Path, bias: str) -> None:
        lora = _save_tiny_lora(tmp_path, bias=bias)
        adapter = TinyLoRAAdapter()
        adapter.load("cpu")
        base_model = adapter._model
        snapshot = _snapshot(base_model)

        with pytest.raises(ValueError, match=f"bias={bias!r}"):
            adapter.load_lora(lora)

        assert adapter._model is base_model
        assert adapter._peft_model is None
        assert lora not in adapter._loaded_loras
        assert _unchanged(base_model, snapshot)

    def test_bias_free_lora_leaves_base_outputs_unchanged(self, tmp_path: Path) -> None:
        lora = _save_tiny_lora(tmp_path, bias="none")
        adapter = TinyLoRAAdapter()
        adapter.load("cpu")
        expected = adapter._model(_probe())

        adapter.load_lora(lora)
        assert torch.equal(adapter._model(_probe()), expected)

        adapter.set_active_lora(lora)
        assert not torch.equal(adapter._model(_probe()), expected)

        adapter.set_active_lora(None)
        assert torch.equal(adapter._model(_probe()), expected)

    def test_declared_bias_lora_is_skipped_and_the_model_loads(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        lora = _save_tiny_lora(tmp_path, bias="all")
        adapter = TinyLoRAAdapter()
        adapter.load("cpu")
        snapshot = _snapshot(adapter._model)
        loader = ModelLoader(
            preprocessor_registry=MagicMock(),
            postprocessor_registry=MagicMock(),
            all_configs={},
        )

        loaded = loader._finish_load("t/tiny", "cpu", adapter, _tiny_config([lora]))

        assert lora not in loaded.loras
        assert adapter._peft_model is None
        assert _unchanged(adapter._model, snapshot)
        assert "bias='all'" in caplog.text

    async def test_request_time_bias_lora_is_refused(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        lora = _save_tiny_lora(tmp_path, bias="all")
        adapter = TinyLoRAAdapter()
        adapter.load("cpu")
        snapshot = _snapshot(adapter._model)
        config = _tiny_config()
        loaded = LoadedModel(config=config, adapter=adapter, device="cpu")
        registry = ModelRegistry()
        registry._configs["t/tiny"] = config
        registry._loaded["t/tiny"] = loaded

        assert await registry.ensure_lora_loaded_async("t/tiny", lora) == (False, True)
        await asyncio.gather(*registry._background_tasks)

        assert lora not in loaded.loras
        assert adapter._peft_model is None
        assert _unchanged(adapter._model, snapshot)
        assert "bias='all'" in caplog.text


# =============================================================================
# Pinned-revision serving tests (#2113)
# =============================================================================


class TestPinnedLoraServing:
    """Pins are honored on the hot-reload preload path and rejected elsewhere."""

    SHA = "a" * 40

    def _pinned_config(self) -> Any:
        from sie_server.config.model import (
            AdapterOptions,
            EmbeddingDim,
            EncodeTask,
            ModelConfig,
            ProfileConfig,
            Tasks,
        )

        return ModelConfig(
            sie_id="t/pinned",
            hf_id="org/base",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=8))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="mod:Cls",
                    max_batch_tokens=8,
                    adapter_options=AdapterOptions(
                        loadtime={"lora_paths": {"bank": {"id": "acme/l", "revision": self.SHA}}}
                    ),
                )
            },
        )

    @staticmethod
    def _finish_load_adapter(lora_cap: Any) -> MagicMock:
        adapter = MagicMock()
        adapter.get_postprocessors.return_value = None
        adapter.get_preprocessor.return_value = None
        adapter.memory_footprint.return_value = 0
        adapter.lora_capability.return_value = lora_cap
        return adapter

    @staticmethod
    def _loader() -> Any:
        from sie_server.core.model_loader import ModelLoader

        return ModelLoader(
            preprocessor_registry=MagicMock(),
            postprocessor_registry=MagicMock(),
            all_configs={},
        )

    def test_hot_reload_preload_passes_the_pinned_revision(self) -> None:
        lora_cap = MagicMock()
        lora_cap.supports_hot_lora_reload.return_value = True
        lora_cap.load_lora.return_value = 0
        adapter = self._finish_load_adapter(lora_cap)

        loaded = self._loader()._finish_load("t/pinned", "cpu", adapter, self._pinned_config())

        lora_cap.load_lora.assert_called_once_with("acme/l", revision=self.SHA)
        assert "acme/l" in loaded.loras

    def test_engine_owned_adapter_rejects_a_pin_loudly(self) -> None:
        """Sglang consumes bare ids at engine launch — a pin it cannot apply must
        fail the load, not silently resolve the default branch (#2113).
        """
        lora_cap = MagicMock()
        lora_cap.supports_hot_lora_reload.return_value = False
        adapter = self._finish_load_adapter(lora_cap)

        with pytest.raises(ValueError, match="cannot honor a pin"):
            self._loader()._finish_load("t/pinned", "cpu", adapter, self._pinned_config())

    def test_lora_less_adapter_rejects_a_pin_loudly(self) -> None:
        adapter = self._finish_load_adapter(None)

        with pytest.raises(ValueError, match="cannot honor a pin"):
            self._loader()._finish_load("t/pinned", "cpu", adapter, self._pinned_config())
