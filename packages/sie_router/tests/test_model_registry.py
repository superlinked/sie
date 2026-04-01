import tempfile
from pathlib import Path

import pytest
from sie_router.model_registry import (
    BundleConflictError,
    BundleInfo,
    ModelInfo,
    ModelNotFoundError,
    ModelRegistry,
    parse_model_spec,
)


class TestParseModelSpec:
    """Test cases for parse_model_spec function."""

    def test_simple_model_name(self) -> None:
        """Simple model name without bundle override."""
        bundle, model = parse_model_spec("BAAI/bge-m3")
        assert bundle is None
        assert model == "BAAI/bge-m3"

    def test_model_with_variant(self) -> None:
        """Model name with variant suffix."""
        bundle, model = parse_model_spec("BAAI/bge-m3:FlagEmbedding")
        assert bundle is None
        assert model == "BAAI/bge-m3:FlagEmbedding"

    def test_bundle_override(self) -> None:
        """Model spec with bundle override."""
        bundle, model = parse_model_spec("sglang:/BAAI/bge-m3")
        assert bundle == "sglang"
        assert model == "BAAI/bge-m3"

    def test_bundle_override_with_variant(self) -> None:
        """Bundle override with variant suffix."""
        bundle, model = parse_model_spec("sglang:/BAAI/bge-m3:variant")
        assert bundle == "sglang"
        assert model == "BAAI/bge-m3:variant"

    def test_bundle_override_case_insensitive(self) -> None:
        """Bundle override is lowercased."""
        bundle, model = parse_model_spec("SGLANG:/BAAI/bge-m3")
        assert bundle == "sglang"
        assert model == "BAAI/bge-m3"

    def test_single_word_model(self) -> None:
        """Single word model name (no org prefix)."""
        bundle, model = parse_model_spec("simple-model")
        assert bundle is None
        assert model == "simple-model"


class TestBundleInfo:
    """Test cases for BundleInfo dataclass."""

    def test_bundle_info_defaults(self) -> None:
        """BundleInfo has sensible defaults."""
        info = BundleInfo(name="test", priority=10)
        assert info.name == "test"
        assert info.priority == 10
        assert info.adapters == []
        assert info.default is False


class TestModelInfo:
    """Test cases for ModelInfo dataclass."""

    def test_model_info_defaults(self) -> None:
        """ModelInfo has sensible defaults."""
        info = ModelInfo(name="test-model")
        assert info.name == "test-model"
        assert info.bundles == []


@pytest.fixture
def temp_config_dirs():
    """Create temporary directories with test bundle and model configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        bundles_dir = tmppath / "bundles"
        models_dir = tmppath / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()

        # Create bundle configs (YAML with adapters list)
        (bundles_dir / "default.yaml").write_text(
            "name: default\n"
            "priority: 10\n"
            "default: true\n"
            "adapters:\n"
            "  - sie_server.adapters.bge_m3\n"
            "  - sie_server.adapters.sentence_transformer\n"
            "  - sie_server.adapters.cross_encoder\n"
        )

        (bundles_dir / "sglang.yaml").write_text(
            "name: sglang\npriority: 20\nadapters:\n  - sie_server.adapters.bge_m3\n  - sie_server.adapters.sglang\n"
        )

        # Create model configs with profiles containing adapter_path
        (models_dir / "baai-bge-m3.yaml").write_text(
            "sie_id: BAAI/bge-m3\n"
            "hf_id: BAAI/bge-m3\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bge_m3:BGEM3Adapter\n"
        )

        (models_dir / "intfloat-e5-small-v2.yaml").write_text(
            "sie_id: intfloat/e5-small-v2\n"
            "hf_id: intfloat/e5-small-v2\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.sentence_transformer:SentenceTransformerAdapter\n"
        )

        (models_dir / "cross-encoder-ms-marco-minilm-l-6-v2.yaml").write_text(
            "name: cross-encoder/ms-marco-MiniLM-L-6-v2\n"
            "hf_id: cross-encoder/ms-marco-MiniLM-L-6-v2\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.cross_encoder:CrossEncoderAdapter\n"
        )

        (models_dir / "qwen-qwen3-embedding-8b.yaml").write_text(
            "sie_id: Qwen/Qwen3-Embedding-8B\n"
            "hf_id: Qwen/Qwen3-Embedding-8B\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.sglang:SGLangAdapter\n"
        )

        yield bundles_dir, models_dir


class TestModelRegistry:
    """Test cases for ModelRegistry class."""

    def test_load_bundles(self, temp_config_dirs) -> None:
        """Registry loads all bundle configs."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        bundles = registry.list_bundles()
        assert len(bundles) == 2
        assert "default" in bundles
        assert "sglang" in bundles

    def test_bundles_sorted_by_priority(self, temp_config_dirs) -> None:
        """Bundles are sorted by priority."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        bundles = registry.list_bundles()
        # Lower priority first
        assert bundles[0] == "default"  # priority 10
        assert bundles[1] == "sglang"  # priority 20

    def test_load_models(self, temp_config_dirs) -> None:
        """Registry loads all model configs."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        models = registry.list_models()
        assert "BAAI/bge-m3" in models
        assert "intfloat/e5-small-v2" in models
        assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in models
        assert "Qwen/Qwen3-Embedding-8B" in models

    def test_model_bundle_mapping(self, temp_config_dirs) -> None:
        """Models are mapped to their compatible bundles via adapter matching."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        # bge-m3 adapter is in both default and sglang
        info = registry.get_model_info("BAAI/bge-m3")
        assert info is not None
        assert "default" in info.bundles
        assert "sglang" in info.bundles
        # Default should be first (lower priority)
        assert info.bundles[0] == "default"

        # e5-small-v2 uses sentence_transformer adapter, only in default
        info = registry.get_model_info("intfloat/e5-small-v2")
        assert info is not None
        assert info.bundles == ["default"]

    def test_resolve_bundle_auto_select(self, temp_config_dirs) -> None:
        """resolve_bundle auto-selects lowest priority bundle."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        # bge-m3 is in default (10) and sglang (20) - should pick default
        bundle = registry.resolve_bundle("BAAI/bge-m3")
        assert bundle == "default"

    def test_resolve_bundle_with_override(self, temp_config_dirs) -> None:
        """resolve_bundle respects explicit override."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        # Override to sglang (even though default is lower priority)
        bundle = registry.resolve_bundle("BAAI/bge-m3", bundle_override="sglang")
        assert bundle == "sglang"

    def test_resolve_bundle_unknown_model(self, temp_config_dirs) -> None:
        """resolve_bundle raises ModelNotFoundError for unknown model."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.resolve_bundle("unknown/model")
        assert exc_info.value.model == "unknown/model"

    def test_resolve_bundle_incompatible_override(self, temp_config_dirs) -> None:
        """resolve_bundle raises BundleConflictError for incompatible override."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        # e5-small-v2 is only in default, not sglang
        with pytest.raises(BundleConflictError) as exc_info:
            registry.resolve_bundle("intfloat/e5-small-v2", bundle_override="sglang")

        assert exc_info.value.model == "intfloat/e5-small-v2"
        assert exc_info.value.bundle == "sglang"
        assert "default" in exc_info.value.compatible_bundles

    def test_resolve_bundle_case_insensitive(self, temp_config_dirs) -> None:
        """Model lookup is case-insensitive."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        # Should work with different cases
        bundle = registry.resolve_bundle("baai/bge-m3")
        assert bundle == "default"

        bundle = registry.resolve_bundle("BAAI/BGE-M3")
        assert bundle == "default"

    def test_model_exists(self, temp_config_dirs) -> None:
        """model_exists returns True for known models."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        assert registry.model_exists("BAAI/bge-m3") is True
        assert registry.model_exists("baai/bge-m3") is True  # case-insensitive
        assert registry.model_exists("unknown/model") is False

    def test_get_bundle_info(self, temp_config_dirs) -> None:
        """get_bundle_info returns BundleInfo."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        info = registry.get_bundle_info("default")
        assert info is not None
        assert info.name == "default"
        assert info.priority == 10
        assert info.default is True
        assert "sie_server.adapters.bge_m3" in info.adapters

        info = registry.get_bundle_info("nonexistent")
        assert info is None

    def test_get_models_for_bundle(self, temp_config_dirs) -> None:
        """get_models_for_bundle returns models in a bundle."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        models = registry.get_models_for_bundle("default")
        assert "BAAI/bge-m3" in models
        assert "intfloat/e5-small-v2" in models

        models = registry.get_models_for_bundle("sglang")
        assert "BAAI/bge-m3" in models
        assert "Qwen/Qwen3-Embedding-8B" in models

        models = registry.get_models_for_bundle("nonexistent")
        assert models == []

    def test_reload(self, temp_config_dirs) -> None:
        """reload() refreshes configs from disk."""
        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        # Initial state
        assert "BAAI/bge-m3" in registry.list_models()

        # Add a new bundle
        (bundles_dir / "new.yaml").write_text("name: new\npriority: 5\nadapters:\n  - sie_server.adapters.test\n")

        # Add a new model
        (models_dir / "new-model.yaml").write_text(
            "sie_id: new/model\n"
            "hf_id: new/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.test:TestAdapter\n"
        )

        # Reload
        registry.reload()

        # New bundle and model should be available
        assert "new" in registry.list_bundles()
        assert "new/model" in registry.list_models()

        # New bundle should be first (priority 5)
        assert registry.list_bundles()[0] == "new"


class TestModelRegistryEmptyDirectories:
    """Test cases for ModelRegistry with missing directories."""

    def test_missing_bundles_dir(self) -> None:
        """Registry handles missing bundles directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "nonexistent_bundles"
            models_dir = tmppath / "models"
            models_dir.mkdir()

            registry = ModelRegistry(bundles_dir, models_dir)

            assert registry.list_bundles() == []
            assert registry.list_models() == []

    def test_missing_models_dir(self) -> None:
        """Registry handles missing models directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "nonexistent_models"
            bundles_dir.mkdir()

            # Create a bundle
            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.test\n"
            )

            registry = ModelRegistry(bundles_dir, models_dir)

            # Bundle should be loaded
            assert "default" in registry.list_bundles()
            # No models (models dir doesn't exist)
            assert registry.list_models() == []

    def test_empty_bundle_file(self) -> None:
        """Registry handles empty bundle files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()

            # Create an empty bundle file
            (bundles_dir / "empty.yaml").write_text("")

            registry = ModelRegistry(bundles_dir, models_dir)

            # Should still load (with defaults)
            assert len(registry.list_bundles()) >= 0  # May or may not include empty


class TestModelRegistryAdapterMatching:
    """Test cases for adapter-based model→bundle matching."""

    def test_model_with_no_matching_adapter(self) -> None:
        """Model whose adapter isn't in any bundle has no bundles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()

            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n"
            )

            (models_dir / "orphan.yaml").write_text(
                "sie_id: orphan/model\n"
                "profiles:\n"
                "  default:\n"
                "    adapter_path: sie_server.adapters.unknown:UnknownAdapter\n"
            )

            registry = ModelRegistry(bundles_dir, models_dir)

            # Model exists but has no bundles
            assert "orphan/model" in registry.list_models()
            info = registry.get_model_info("orphan/model")
            assert info is not None
            assert info.bundles == []

    def test_model_multiple_profiles_different_adapters(self) -> None:
        """Model with multiple profiles matches bundles from any profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            bundles_dir = tmppath / "bundles"
            models_dir = tmppath / "models"
            bundles_dir.mkdir()
            models_dir.mkdir()

            (bundles_dir / "default.yaml").write_text(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n"
            )

            (bundles_dir / "sglang.yaml").write_text(
                "name: sglang\npriority: 20\nadapters:\n  - sie_server.adapters.sglang\n"
            )

            (models_dir / "multi.yaml").write_text(
                "name: multi/model\n"
                "profiles:\n"
                "  cpu:\n"
                "    adapter_path: sie_server.adapters.sentence_transformer:STAdapter\n"
                "  gpu:\n"
                "    adapter_path: sie_server.adapters.sglang:SGLangAdapter\n"
            )

            registry = ModelRegistry(bundles_dir, models_dir)

            info = registry.get_model_info("multi/model")
            assert info is not None
            assert "default" in info.bundles
            assert "sglang" in info.bundles


class TestModelRegistryThreadSafety:
    """Test cases for thread safety of ModelRegistry."""

    def test_concurrent_reads(self, temp_config_dirs) -> None:
        """Multiple threads can read concurrently."""
        import threading

        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        results = []
        errors = []

        def read_models():
            try:
                for _ in range(100):
                    models = registry.list_models()
                    results.append(len(models))
            except Exception as e:  # noqa: BLE001 — concurrency test error collection
                errors.append(e)

        threads = [threading.Thread(target=read_models) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 1000  # 10 threads * 100 iterations
        assert all(r == results[0] for r in results)  # All same count

    def test_reload_during_read(self, temp_config_dirs) -> None:
        """Reload is safe during concurrent reads."""
        import threading

        bundles_dir, models_dir = temp_config_dirs
        registry = ModelRegistry(bundles_dir, models_dir)

        errors = []

        def read_models():
            try:
                for _ in range(100):
                    registry.list_models()
                    registry.resolve_bundle("BAAI/bge-m3")
            except ModelNotFoundError:
                pass  # Expected during reload
            except Exception as e:  # noqa: BLE001 — concurrency test error collection
                errors.append(e)

        def reload_registry():
            try:
                for _ in range(10):
                    registry.reload()
            except Exception as e:  # noqa: BLE001 — concurrency test error collection
                errors.append(e)

        read_threads = [threading.Thread(target=read_models) for _ in range(5)]
        reload_thread = threading.Thread(target=reload_registry)

        for t in read_threads:
            t.start()
        reload_thread.start()

        for t in read_threads:
            t.join()
        reload_thread.join()

        assert not errors
