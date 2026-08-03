from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.load_errors import ModelLoadTimeoutError
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

    def test_empty_model_filter_advertises_zero_models(self, tmp_path: Path) -> None:
        """model_filter=[] means ZERO models, not "no filter".

        A bundle whose adapters match no onboarded model (first case:
        transformers514 before its pilot lands) produces an empty filter; a
        truthiness check would collapse it to None and advertise the entire
        catalog from a worker that can serve none of it.
        """
        (tmp_path / "model-a.yaml").write_text("""
sie_id: model-a
hf_id: org/model-a
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
    max_batch_tokens: 8192
""")

        filtered = ModelRegistry(models_dir=tmp_path, model_filter=[])
        assert filtered.model_names == []

        unfiltered = ModelRegistry(models_dir=tmp_path, model_filter=None)
        assert unfiltered.model_names == ["model-a"]

    def test_add_config(self) -> None:
        """Can add config programmatically."""
        registry = ModelRegistry()

        config = _make_config(name="test-model", hf_id="org/test")

        registry.add_config(config)

        assert registry.has_model("test-model")
        assert registry.get_config("test-model") == config

    @pytest.mark.asyncio
    async def test_add_modify_delete_refresh_the_authoritative_metric_catalog(self) -> None:
        registry = ModelRegistry()
        original = _make_config(name="test-model", dense_dim=384)
        modified = _make_config(name="test-model", dense_dim=1024)

        with patch("sie_server.core.registry.refresh_worker_metric_context") as refresh:
            registry.add_config(original)
            await registry.add_config_async(modified)
            removed = await registry.remove_config_async("test-model")

        assert removed == {"test-model"}
        assert refresh.call_count == 3
        snapshots = [call.kwargs["configs"] for call in refresh.call_args_list]
        assert snapshots[0]["test-model"].tasks.encode.dense.dim == 384
        assert snapshots[1]["test-model"].tasks.encode.dense.dim == 1024
        assert snapshots[2] == {}

    def test_rescan_refreshes_modified_and_deleted_catalog_entries(self) -> None:
        original = _make_config(name="test-model", dense_dim=384)
        modified = _make_config(name="test-model", dense_dim=1024)
        with patch(
            "sie_server.core.registry.load_model_configs",
            side_effect=[{"test-model": original}, {"test-model": modified}, {}],
        ):
            registry = ModelRegistry(models_dir=Path("/fake/models"))
            with patch("sie_server.core.registry.refresh_worker_metric_context") as refresh:
                assert registry.rescan_configs() == []
                assert registry.get_config("test-model").tasks.encode.dense.dim == 1024
                assert registry.rescan_configs() == []

        snapshots = [call.kwargs["configs"] for call in refresh.call_args_list]
        assert snapshots[0]["test-model"].tasks.encode.dense.dim == 1024
        assert snapshots[1] == {}

    def test_add_config_expands_profile_variants(self) -> None:
        """Programmatic config updates expose non-default profile aliases."""
        registry = ModelRegistry()

        config = _make_config(name="test-model", hf_id="org/test")
        config.profiles["fast"] = ProfileConfig(
            adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
            max_batch_tokens=8192,
        )

        registry.add_config(config)

        assert registry.has_model("test-model:fast")
        variant = registry.get_config("test-model:fast")
        assert set(variant.profiles) == {"default"}
        assert variant.profiles["default"].adapter_path == (
            "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
        )

    def test_add_config_omits_bare_model_without_default_profile(self) -> None:
        """Profile-only configs expose only explicit variant aliases."""
        registry = ModelRegistry()

        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "fast": ProfileConfig(
                    adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                    max_batch_tokens=8192,
                )
            },
        )

        registry.add_config(config)

        assert not registry.has_model("test-model")
        assert registry.has_model("test-model:fast")
        variant = registry.get_config("test-model:fast")
        assert set(variant.profiles) == {"default"}
        assert variant.profiles["default"].adapter_path == (
            "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
        )

    def test_python_registry_filters_rust_only_profile_variants(self) -> None:
        """The Python worker must not advertise Rust-only adapter routes."""
        registry = ModelRegistry()

        config = _make_config(name="test-model", hf_id="org/test")
        config.profiles["candle"] = ProfileConfig(
            adapter_path="sie_server_rust.adapters.candle:CandleEmbeddingAdapter",
            max_batch_tokens=8192,
        )

        registry.add_config(config)

        assert registry.has_model("test-model")
        assert not registry.has_model("test-model:candle")

    @pytest.mark.asyncio
    async def test_add_config_async_drops_stale_bare_and_profile_variants(self) -> None:
        """Authoritative profile removal must stop serving stale route ids."""
        registry = ModelRegistry()

        config = _make_config(name="test-model", hf_id="org/test")
        config.profiles["fast"] = ProfileConfig(
            adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
            max_batch_tokens=8192,
        )
        registry.add_config(config)
        assert registry.has_model("test-model")
        assert registry.has_model("test-model:fast")

        profile_only = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "fast": ProfileConfig(
                    adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                    max_batch_tokens=8192,
                )
            },
        )

        changed = await registry.add_config_async(profile_only)

        assert changed == {"test-model", "test-model:fast"}
        assert not registry.has_model("test-model")
        assert registry.has_model("test-model:fast")

    @pytest.mark.asyncio
    async def test_add_config_async_unloads_loaded_model_when_revision_changes(self) -> None:
        """A new immutable revision must not keep serving already-loaded weights."""
        registry = ModelRegistry()
        first = _make_config(name="test-model", hf_id="org/test")
        first.hf_revision = "0123456789abcdef0123456789abcdef01234567"
        registry.add_config(first)
        registry._loaded["test-model"] = MagicMock()
        registry._do_unload = AsyncMock()

        second = _make_config(name="test-model", hf_id="org/test")
        second.hf_revision = "89abcdef0123456789abcdef0123456789abcdef"
        await registry.add_config_async(second)

        registry._do_unload.assert_awaited_once_with("test-model", reason="config_change")
        assert registry.get_config("test-model").hf_revision == second.hf_revision

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

    def test_load_from_directory_filters_rust_only_variants(self, tmp_path: Path) -> None:
        """Filesystem catalog loading keeps Rust-only profiles out of Python routes."""
        (tmp_path / "mixed-model.yaml").write_text("""
sie_id: mixed-model
hf_id: org/mixed-model
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
    max_batch_tokens: 8192
  candle:
    adapter_path: "sie_server_rust.adapters.candle:CandleEmbeddingAdapter"
    max_batch_tokens: 8192
""")
        (tmp_path / "candle-only.yaml").write_text("""
sie_id: candle-only
hf_id: org/candle-only
tasks:
  encode:
    dense:
      dim: 384
profiles:
  candle:
    adapter_path: "sie_server_rust.adapters.candle:CandleEmbeddingAdapter"
    max_batch_tokens: 8192
""")

        registry = ModelRegistry(models_dir=tmp_path)

        assert registry.has_model("mixed-model")
        assert not registry.has_model("mixed-model:candle")
        assert not registry.has_model("candle-only")
        assert not registry.has_model("candle-only:candle")

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
    def test_sync_load_records_managed_duration_outcome(
        self,
        mock_load_adapter: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        telemetry = MagicMock()
        mock_load_adapter.return_value = mock_adapter
        registry = ModelRegistry()
        registry.add_config(_make_config(name="test"))

        with patch("sie_server.core.registry.worker_telemetry", return_value=telemetry):
            registry.load("test", device="cpu")

        telemetry.model_load_completed.assert_called_once()
        kwargs = telemetry.model_load_completed.call_args.kwargs
        assert kwargs["model"] == "test"
        assert kwargs["duration_s"] >= 0
        assert kwargs["outcome"] == "success"
        assert kwargs["stage"] == "total"

    @pytest.mark.parametrize(
        ("error", "outcome", "stage"),
        [
            (RuntimeError("load failed"), "error", "total"),
            (
                ModelLoadTimeoutError(model="test", stage="instantiate", elapsed_s=11.0, timeout_s=10.0),
                "timeout",
                "instantiate",
            ),
        ],
    )
    def test_sync_load_records_managed_failure_outcome(
        self,
        error: BaseException,
        outcome: str,
        stage: str,
    ) -> None:
        telemetry = MagicMock()
        registry = ModelRegistry()
        registry.add_config(_make_config(name="test"))

        with (
            patch.object(registry._loader, "instantiate_adapter", side_effect=error),
            patch("sie_server.core.registry.worker_telemetry", return_value=telemetry),
            pytest.raises(type(error)),
        ):
            registry.load("test", device="cpu")

        telemetry.model_load_completed.assert_called_once()
        kwargs = telemetry.model_load_completed.call_args.kwargs
        assert kwargs["model"] == "test"
        assert kwargs["duration_s"] >= 0
        assert kwargs["outcome"] == outcome
        assert kwargs["stage"] == stage

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
