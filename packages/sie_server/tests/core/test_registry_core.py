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


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.capabilities.outputs = ["dense"]
        return mock

    def test_empty_registry(self) -> None:
        """Can create empty registry."""
        registry = ModelRegistry()

        assert registry.model_names == []
        assert registry.loaded_model_names == []

    def test_add_config(self) -> None:
        """Can add config programmatically."""
        registry = ModelRegistry()

        config = _make_config(name="test-model", hf_id="org/test")

        registry.add_config(config)

        assert registry.has_model("test-model")
        assert registry.get_config("test-model") == config

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Can load configs from directory."""
        # Create flat YAML config file
        (tmp_path / "my-model.yaml").write_text("""
sie_id: my-model
hf_id: org/my-model
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
    max_batch_tokens: 8192
""")

        registry = ModelRegistry(models_dir=tmp_path)

        assert registry.has_model("my-model")
        assert registry.get_config("my-model").tasks.encode.dense.dim == 384

    def test_cloud_models_dir_maps_cached_configs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cloud models_dir maps each model to the cache directory (flat YAML structure)."""
        cache_root = tmp_path / "cache"
        monkeypatch.setenv("SIE_LOCAL_CACHE", str(cache_root))
        cache_dir = cache_root / "sie_configs"
        cache_dir.mkdir(parents=True)

        def write_cached_config(model_name: str, dense_dim: int) -> Path:
            # Flat YAML structure: configs are directly in cache_dir
            yaml_filename = model_name.lower().replace("/", "-") + ".yaml"
            config_path = cache_dir / yaml_filename
            config_path.write_text(
                "\n".join(
                    [
                        f"sie_id: {model_name}",
                        f"hf_id: {model_name}",
                        "tasks:",
                        "  encode:",
                        "    dense:",
                        f"      dim: {dense_dim}",
                        "profiles:",
                        "  default:",
                        "    adapter_path: sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                        "    max_batch_tokens: 8192",
                        "",
                    ]
                )
            )
            return config_path

        model_a = "org/model-a"
        model_b = "org/model-b"
        write_cached_config(model_a, 384)
        write_cached_config(model_b, 768)

        configs = {
            model_a: _make_config(name=model_a, hf_id=model_a, dense_dim=384),
            model_b: _make_config(name=model_b, hf_id=model_b),
        }

        monkeypatch.setattr("sie_server.core.registry.load_model_configs", lambda _: configs)
        monkeypatch.setattr("sie_sdk.storage.is_cloud_path", lambda _: True)

        registry = ModelRegistry(models_dir="s3://bucket/models")

        # With flat YAML structure, all models share the same cache dir as their model_dir
        assert registry._model_dirs[model_a] == cache_dir
        assert registry._model_dirs[model_b] == cache_dir

    def test_has_model(self) -> None:
        """has_model returns correct values."""
        registry = ModelRegistry()

        config = _make_config(name="exists")
        registry.add_config(config)

        assert registry.has_model("exists") is True
        assert registry.has_model("nonexistent") is False

    def test_is_loaded(self) -> None:
        """is_loaded returns correct values."""
        registry = ModelRegistry()

        config = _make_config(name="test")
        registry.add_config(config)

        assert registry.is_loaded("test") is False

    @patch("sie_server.core.model_loader.load_adapter")
    def test_load_model(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Can load a model."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)

        adapter = registry.load("test", device="cpu")

        assert adapter is mock_adapter
        mock_adapter.load.assert_called_once_with("cpu")
        assert registry.is_loaded("test")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_get_loaded_model(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Can get a loaded model's adapter."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")

        adapter = registry.get("test")

        assert adapter is mock_adapter

    def test_get_unloaded_model_raises(self) -> None:
        """Get raises for unloaded model."""
        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)

        with pytest.raises(KeyError, match="is not loaded"):
            registry.get("test")

    def test_get_unknown_model_raises(self) -> None:
        """Get raises for unknown model."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_load_already_loaded_raises(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Load raises if model already loaded."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")

        with pytest.raises(ValueError, match="already loaded"):
            registry.load("test", device="cpu")

    def test_load_unknown_model_raises(self) -> None:
        """Load raises for unknown model."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.load("nonexistent", device="cpu")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_unload_model(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Can unload a model."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")

        registry.unload("test")

        mock_adapter.unload.assert_called_once()
        assert not registry.is_loaded("test")

    def test_unload_not_loaded_raises(self) -> None:
        """Unload raises if model not loaded."""
        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)

        with pytest.raises(KeyError, match="is not loaded"):
            registry.unload("test")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_unload_all(self, mock_load_adapter: MagicMock) -> None:
        """Can unload all models."""
        mock_adapter_a = MagicMock()
        mock_adapter_b = MagicMock()
        mock_load_adapter.side_effect = [mock_adapter_a, mock_adapter_b]

        registry = ModelRegistry()

        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        assert len(registry.loaded_model_names) == 2

        registry.unload_all()

        assert len(registry.loaded_model_names) == 0
        mock_adapter_a.unload.assert_called_once()
        mock_adapter_b.unload.assert_called_once()

    @patch("sie_server.core.model_loader.load_adapter")
    def test_get_model_info(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Can get model info."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test", max_sequence_length=512)
        registry.add_config(config)

        # Before loading
        info = registry.get_model_info("test")
        assert info["name"] == "test"
        assert info["loaded"] is False
        assert info["device"] is None
        assert info["dims"]["dense"] == 768

        # After loading
        registry.load("test", device="cuda:0")
        info = registry.get_model_info("test")
        assert info["loaded"] is True
        assert info["device"] == "cuda:0"
