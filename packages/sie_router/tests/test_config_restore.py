import tempfile
from pathlib import Path

import yaml
from sie_router.config_store import ConfigStore
from sie_router.model_registry import ModelRegistry


def _write_bundle(bundles_dir: Path, name: str, adapters: list[str], priority: int = 10) -> None:
    data = {"name": name, "priority": priority, "adapters": adapters}
    (bundles_dir / f"{name}.yaml").write_text(yaml.dump(data))


def _write_model(models_dir: Path, sie_id: str, adapter_path: str) -> None:
    config = {
        "sie_id": sie_id,
        "profiles": {"default": {"adapter_path": adapter_path, "max_batch_tokens": 8192}},
    }
    filename = sie_id.replace("/", "__") + ".yaml"
    (models_dir / filename).write_text(yaml.dump(config))


class TestConfigRestore:
    """Tests for config restore on startup — loading from config store into registry."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        self._bundles = root / "bundles"
        self._models = root / "models"
        self._store_dir = root / "store"
        self._bundles.mkdir()
        self._models.mkdir()
        _write_bundle(self._bundles, "default", ["sie_server.adapters.bert_flash"])

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_restore_loads_models_from_store(self) -> None:
        """Models in config store are loaded into registry on restore."""
        # Pre-populate store with a model
        store = ConfigStore(str(self._store_dir))
        model_yaml = yaml.dump(
            {
                "sie_id": "store/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        store.write_model("store/model", model_yaml)
        store.cas_epoch(0)

        # Create registry (no filesystem models)
        registry = ModelRegistry(self._bundles, self._models)
        assert not registry.model_exists("store/model")

        # Restore from store
        stored_models = store.load_all_models()
        for model_config in stored_models.values():
            registry.add_model_config(model_config)

        assert registry.model_exists("store/model")
        assert registry.resolve_bundle("store/model") == "default"

    def test_restore_merges_with_filesystem(self) -> None:
        """Restore adds store models alongside filesystem models."""
        # Filesystem has one model
        _write_model(self._models, "fs/model", "sie_server.adapters.bert_flash:B")

        # Store has a different model
        store = ConfigStore(str(self._store_dir))
        store_yaml = yaml.dump(
            {
                "sie_id": "api/model",
                "profiles": {"default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}},
            }
        )
        store.write_model("api/model", store_yaml)

        registry = ModelRegistry(self._bundles, self._models)
        assert registry.model_exists("fs/model")
        assert not registry.model_exists("api/model")

        # Restore
        for model_config in store.load_all_models().values():
            registry.add_model_config(model_config)

        assert registry.model_exists("fs/model")
        assert registry.model_exists("api/model")

    def test_restore_disabled_ignores_store(self) -> None:
        """When restore is not called, store models are not loaded."""
        store = ConfigStore(str(self._store_dir))
        store.write_model(
            "store/model",
            yaml.dump(
                {
                    "sie_id": "store/model",
                    "profiles": {
                        "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}
                    },
                }
            ),
        )

        registry = ModelRegistry(self._bundles, self._models)
        # No restore call
        assert not registry.model_exists("store/model")

    def test_restore_survives_registry_reload(self) -> None:
        """After restore + reload, filesystem models reload but store models are lost.

        This is expected: reload() reloads from disk only. Config restore must
        be re-applied after reload if needed.
        """
        _write_model(self._models, "fs/model", "sie_server.adapters.bert_flash:B")
        store = ConfigStore(str(self._store_dir))
        store.write_model(
            "api/model",
            yaml.dump(
                {
                    "sie_id": "api/model",
                    "profiles": {
                        "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}
                    },
                }
            ),
        )

        registry = ModelRegistry(self._bundles, self._models)
        for model_config in store.load_all_models().values():
            registry.add_model_config(model_config)
        assert registry.model_exists("api/model")

        # Reload clears runtime-added models
        registry.reload()
        assert registry.model_exists("fs/model")
        assert not registry.model_exists("api/model")  # Lost on reload

        # Re-restore brings it back
        for model_config in store.load_all_models().values():
            registry.add_model_config(model_config)
        assert registry.model_exists("api/model")

    def test_restore_with_corrupt_store_entry(self) -> None:
        """Corrupt entries in store are skipped during restore."""
        store = ConfigStore(str(self._store_dir))
        store.write_model(
            "good/model",
            yaml.dump(
                {
                    "sie_id": "good/model",
                    "profiles": {
                        "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}
                    },
                }
            ),
        )
        # Write corrupt entry directly via backend
        from sie_sdk.storage import join_path

        corrupt_path = join_path(store.base_dir, "models", "corrupt__model.yaml")
        store._backend.write_text(corrupt_path, "{{invalid")

        registry = ModelRegistry(self._bundles, self._models)
        stored = store.load_all_models()
        for model_config in stored.values():
            try:
                registry.add_model_config(model_config)
            except Exception:  # noqa: BLE001, S112 — test intentionally ignores corrupt entries
                continue

        assert registry.model_exists("good/model")

    def test_restore_model_added_by_another_router(self) -> None:
        """Simulates Router B restoring a model that Router A added."""
        # Router A adds model to shared store
        store = ConfigStore(str(self._store_dir))
        store.write_model(
            "router-a/model",
            yaml.dump(
                {
                    "sie_id": "router-a/model",
                    "profiles": {
                        "default": {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}
                    },
                }
            ),
        )
        store.cas_epoch(0)

        # Router B starts fresh, restores from same store
        registry_b = ModelRegistry(self._bundles, self._models)
        for model_config in store.load_all_models().values():
            registry_b.add_model_config(model_config)

        assert registry_b.model_exists("router-a/model")
        assert registry_b.resolve_bundle("router-a/model") == "default"
