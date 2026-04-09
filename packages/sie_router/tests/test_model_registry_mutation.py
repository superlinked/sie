from pathlib import Path

import pytest
import yaml
from sie_router.model_registry import ModelRegistry


def _setup_registry(
    root: Path,
    bundles: dict[str, list[str]] | None = None,
    models: dict[str, str] | None = None,
) -> tuple[ModelRegistry, Path]:
    """Create a ModelRegistry with test bundles and models."""
    bundles_dir = root / "bundles"
    models_dir = root / "models"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if bundles is None:
        bundles = {"default": ["sie_server.adapters.bert_flash", "sie_server.adapters.sentence_transformer"]}

    for name, adapters in bundles.items():
        data = {"name": name, "priority": 10, "adapters": adapters}
        (bundles_dir / f"{name}.yaml").write_text(yaml.dump(data))

    if models:
        for sie_id, adapter_path in models.items():
            config = {
                "sie_id": sie_id,
                "profiles": {"default": {"adapter_path": adapter_path, "max_batch_tokens": 8192}},
            }
            filename = sie_id.replace("/", "__") + ".yaml"
            (models_dir / filename).write_text(yaml.dump(config))

    registry = ModelRegistry(bundles_dir, models_dir)
    return registry, root


class TestAddModelConfig:
    """Tests for ModelRegistry.add_model_config()."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_add_new_model(self) -> None:
        registry, _ = _setup_registry(self._root / "add_new")
        config = {
            "sie_id": "new/model",
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                    "max_batch_tokens": 8192,
                },
            },
        }
        created, skipped, bundles = registry.add_model_config(config)
        assert created == ["default"]
        assert skipped == []
        assert "default" in bundles

    def test_model_becomes_routable(self) -> None:
        registry, _ = _setup_registry(self._root / "routable")
        registry.add_model_config(
            {
                "sie_id": "new/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        assert registry.model_exists("new/model")
        bundle = registry.resolve_bundle("new/model")
        assert bundle == "default"

    def test_add_profile_to_existing_model(self) -> None:
        registry, _ = _setup_registry(
            self._root / "add_prof",
            models={"existing/model": "sie_server.adapters.bert_flash:B"},
        )
        # "default" profile must match the existing config (same adapter_path and max_batch_tokens=8192)
        config = {
            "sie_id": "existing/model",
            "profiles": {
                "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 8192},
                "custom": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 2},
            },
        }
        created, skipped, _ = registry.add_model_config(config)
        assert "custom" in created
        assert "default" in skipped

    def test_add_conflicting_profile_raises(self) -> None:
        """Adding a profile with different config to existing model raises ValueError."""
        registry, _ = _setup_registry(
            self._root / "conflict",
            models={"existing/model": "sie_server.adapters.bert_flash:B"},
        )
        config = {
            "sie_id": "existing/model",
            "profiles": {
                "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1},
            },
        }
        with pytest.raises(ValueError, match="already exists with different config"):
            registry.add_model_config(config)

    def test_unroutable_adapter_raises(self) -> None:
        registry, _ = _setup_registry(self._root / "unroutable")
        config = {
            "sie_id": "bad/model",
            "profiles": {"default": {"adapter_path": "sie_server.adapters.unknown:X", "max_batch_tokens": 1}},
        }
        with pytest.raises(ValueError, match="not in any known bundle"):
            registry.add_model_config(config)

    def test_missing_sie_id_raises(self) -> None:
        registry, _ = _setup_registry(self._root / "no_id")
        with pytest.raises(ValueError, match="sie_id"):
            registry.add_model_config({"profiles": {"default": {}}})

    def test_missing_profiles_raises(self) -> None:
        registry, _ = _setup_registry(self._root / "no_prof")
        with pytest.raises(ValueError, match="profiles"):
            registry.add_model_config({"sie_id": "m"})

    def test_multi_bundle_routing(self) -> None:
        registry, _ = _setup_registry(
            self._root / "multi_bundle",
            bundles={
                "default": ["sie_server.adapters.bert_flash"],
                "sglang": ["sie_server.adapters.sglang"],
            },
        )
        registry.add_model_config(
            {
                "sie_id": "sglang/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.sglang:S", "max_batch_tokens": 1}},
            }
        )
        bundle = registry.resolve_bundle("sglang/model")
        # sglang adapter is only in the sglang bundle, so it must resolve to sglang
        assert bundle == "sglang"


class TestComputeBundleConfigHash:
    """Tests for ModelRegistry.compute_bundle_config_hash()."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_empty_bundle_returns_empty(self) -> None:
        registry, _ = _setup_registry(self._root / "empty")
        assert registry.compute_bundle_config_hash("default") == ""

    def test_hash_changes_when_model_added(self) -> None:
        registry, _ = _setup_registry(self._root / "hash_change")
        hash1 = registry.compute_bundle_config_hash("default")

        registry.add_model_config(
            {
                "sie_id": "new/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        hash2 = registry.compute_bundle_config_hash("default")

        assert hash1 != hash2
        assert len(hash2) == 64  # SHA-256 hex

    def test_hash_is_deterministic(self) -> None:
        registry, _ = _setup_registry(self._root / "deterministic")
        registry.add_model_config(
            {
                "sie_id": "m1",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        h1 = registry.compute_bundle_config_hash("default")
        h2 = registry.compute_bundle_config_hash("default")
        assert h1 == h2

    def test_hash_scoped_to_bundle(self) -> None:
        registry, _ = _setup_registry(
            self._root / "scoped",
            bundles={
                "default": ["sie_server.adapters.bert_flash"],
                "sglang": ["sie_server.adapters.sglang"],
            },
        )
        registry.add_model_config(
            {
                "sie_id": "m1",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        # Adding model to default bundle should not affect sglang hash
        assert registry.compute_bundle_config_hash("sglang") == ""
        assert registry.compute_bundle_config_hash("default") != ""

    def test_unknown_bundle_returns_empty(self) -> None:
        registry, _ = _setup_registry(self._root / "unknown")
        assert registry.compute_bundle_config_hash("nonexistent") == ""


class TestConcurrentAddModelConfig:
    """Tests for concurrent add_model_config under contention."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_concurrent_add_10_models(self) -> None:
        """10 concurrent add_model_config calls all succeed without exceptions."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        registry, _ = _setup_registry(self._root / "concurrent")

        exceptions: list[Exception] = []

        def add_model(i: int) -> None:
            config = {
                "sie_id": f"concurrent/model-{i}",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                        "max_batch_tokens": 8192,
                    },
                },
            }
            try:
                created, _, bundles = registry.add_model_config(config)
                assert created == ["default"]
                assert "default" in bundles
            except Exception as e:  # noqa: BLE001
                exceptions.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_model, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()  # Re-raises any exception from the thread

        # No exceptions should have been raised
        assert exceptions == [], f"Exceptions during concurrent add: {exceptions}"

        # All 10 models should be routable
        for i in range(10):
            model_id = f"concurrent/model-{i}"
            assert registry.model_exists(model_id), f"{model_id} should exist"
            bundle = registry.resolve_bundle(model_id)
            assert bundle == "default", f"{model_id} should resolve to 'default' bundle"
