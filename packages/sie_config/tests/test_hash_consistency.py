import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from sie_config.model_registry import ModelRegistry
from sie_server.api import ws as worker_ws
from sie_server.api.ws import compute_bundle_config_hash_cached
from sie_server.core.registry import ModelRegistry as WorkerModelRegistry


def _create_registry_with_model(
    root: Path,
    sie_id: str = "test/model",
    adapter_path: str = "sie_server.adapters.bert_flash:BertAdapter",
    profile_names: list[str] | None = None,
) -> ModelRegistry:
    bundles_dir = root / "bundles"
    models_dir = root / "models"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle_data = {"name": "default", "priority": 10, "adapters": ["sie_server.adapters.bert_flash"]}
    (bundles_dir / "default.yaml").write_text(yaml.dump(bundle_data))
    registry = ModelRegistry(bundles_dir, models_dir)
    if profile_names is None:
        profile_names = ["default"]
    profiles = {}
    for name in profile_names:
        profiles[name] = {"adapter_path": adapter_path, "max_batch_tokens": 8192}
    registry.add_model_config({"sie_id": sie_id, "profiles": profiles})
    return registry


def _compute_worker_hash_via_registry(model_configs: list[dict]) -> str:
    if not model_configs:
        return ""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        bundles_dir = root / "bundles"
        models_dir = root / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()
        all_adapters = set()
        for cfg in model_configs:
            for profile in cfg.get("profiles", {}).values():
                ap = profile.get("adapter_path", "")
                if ap:
                    all_adapters.add(ap.split(":", maxsplit=1)[0])
        if not all_adapters:
            all_adapters = {"sie_server.adapters.bert_flash"}
        bundle_data = {"name": "default", "priority": 10, "adapters": sorted(all_adapters)}
        (bundles_dir / "default.yaml").write_text(yaml.dump(bundle_data))
        registry = ModelRegistry(bundles_dir, models_dir)
        for cfg in model_configs:
            profiles = {}
            for pname, pdata in cfg.get("profiles", {}).items():
                if isinstance(pdata, dict) and "adapter_path" in pdata:
                    profiles[pname] = pdata
                else:
                    profiles[pname] = {
                        "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                        "max_batch_tokens": 8192,
                    }
            registry.add_model_config({"sie_id": cfg["sie_id"], "profiles": profiles})
        return registry.compute_bundle_config_hash("default")


class TestHashConsistency:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_single_model_single_profile(self) -> None:
        registry = _create_registry_with_model(self._root / "single", "test/model", profile_names=["default"])
        router_hash = registry.compute_bundle_config_hash("default")
        worker_hash = _compute_worker_hash_via_registry([{"sie_id": "test/model", "profiles": {"default": {}}}])
        assert router_hash == worker_hash
        assert len(router_hash) == 64

    def test_single_model_multiple_profiles(self) -> None:
        registry = _create_registry_with_model(
            self._root / "multi_prof", "test/model", profile_names=["default", "custom", "fast"]
        )
        router_hash = registry.compute_bundle_config_hash("default")
        worker_hash = _compute_worker_hash_via_registry(
            [{"sie_id": "test/model", "profiles": {"default": {}, "custom": {}, "fast": {}}}]
        )
        assert router_hash == worker_hash

    def test_multiple_models(self) -> None:
        aaa_profile = {"adapter_path": "sie_server.adapters.bert_flash:BertAdapter", "max_batch_tokens": 8192}
        zzz_profile = {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 1}
        registry = _create_registry_with_model(self._root / "multi_mod", "aaa/model", profile_names=["default"])
        registry.add_model_config({"sie_id": "zzz/model", "profiles": {"default": zzz_profile}})
        router_hash = registry.compute_bundle_config_hash("default")
        worker_hash = _compute_worker_hash_via_registry(
            [
                {"sie_id": "aaa/model", "profiles": {"default": aaa_profile}},
                {"sie_id": "zzz/model", "profiles": {"default": zzz_profile}},
            ]
        )
        assert router_hash == worker_hash

    def test_inherited_profiles_match_worker_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        root = self._root / "extends"
        bundles_dir = root / "bundles"
        models_dir = root / "models"
        bundles_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        bundle_id = "hash-test-bundle"
        adapter_module = "test.hash_adapter"
        (bundles_dir / f"{bundle_id}.yaml").write_text(
            yaml.dump(
                {
                    "name": bundle_id,
                    "priority": 10,
                    "adapters": [adapter_module],
                }
            )
        )
        model_config = {
            "sie_id": "test/model",
            "hf_id": "test/model",
            "inputs": {"text": True},
            "tasks": {"encode": {"dense": {"dim": 3}}},
            "profiles": {
                "default": {
                    "adapter_path": f"{adapter_module}:Adapter",
                    "max_batch_tokens": 8192,
                    "compute_precision": "float16",
                    "adapter_options": {
                        "loadtime": {},
                        "runtime": {"normalize": True},
                    },
                },
                "query": {
                    "extends": "default",
                    "adapter_options": {
                        "runtime": {
                            "normalize": False,
                            "instruction": "query",
                        }
                    },
                },
            },
        }
        (models_dir / "test-model.yaml").write_text(yaml.dump(model_config))

        config_registry = ModelRegistry(bundles_dir, models_dir)
        config_hash = config_registry.compute_bundle_config_hash(bundle_id)

        def resolve_worker_default_dir(name: str) -> Path:
            if name == "bundles":
                return bundles_dir
            return root / name

        worker_ws._bundle_adapter_modules.cache_clear()
        monkeypatch.setattr(worker_ws, "_resolve_default_dir", resolve_worker_default_dir)
        worker_registry = WorkerModelRegistry(
            models_dir=models_dir,
            model_filter=["test/model"],
            pool_name="default",
            enable_hot_reload=False,
        )
        try:
            worker_hash = compute_bundle_config_hash_cached(worker_registry, bundle_id)
        finally:
            worker_ws._bundle_adapter_modules.cache_clear()

        assert config_hash == worker_hash

    def test_package_artifact_manifest_digest_changes_worker_parity_hash(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = self._root / "package_artifacts"
        bundles_dir = root / "bundles"
        models_dir = root / "models"
        bundles_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        bundle_id = "package-bundle"
        adapter_module = "test.package_adapter"
        (bundles_dir / f"{bundle_id}.yaml").write_text(
            yaml.dump({"name": bundle_id, "priority": 10, "adapters": [adapter_module]})
        )

        def resolve_worker_default_dir(name: str) -> Path:
            if name == "bundles":
                return bundles_dir
            return root / name

        monkeypatch.setattr(worker_ws, "_resolve_default_dir", resolve_worker_default_dir)

        def hashes(manifest_sha256: str) -> tuple[str, str]:
            model_config = {
                "sie_id": "package/model",
                "package_backed": True,
                "tasks": {"extract": {}},
                "profiles": {
                    "default": {
                        "adapter_path": f"{adapter_module}:Adapter",
                        "max_batch_tokens": 1,
                        "adapter_options": {
                            "loadtime": {
                                "package_artifact_mode": "staged",
                                "package_artifact_manifest_path": "/models/package/model/manifest.json",
                                "package_artifact_manifest_sha256": manifest_sha256,
                            }
                        },
                    }
                },
            }
            (models_dir / "package__model.yaml").write_text(yaml.dump(model_config))
            config_hash = ModelRegistry(bundles_dir, models_dir).compute_bundle_config_hash(bundle_id)
            worker_ws._bundle_adapter_modules.cache_clear()
            try:
                worker_registry = WorkerModelRegistry(
                    models_dir=models_dir,
                    model_filter=["package/model"],
                    pool_name="default",
                    enable_hot_reload=False,
                )
                worker_hash = compute_bundle_config_hash_cached(worker_registry, bundle_id)
            finally:
                worker_ws._bundle_adapter_modules.cache_clear()
            return config_hash, worker_hash

        first_config, first_worker = hashes("a" * 64)
        second_config, second_worker = hashes("b" * 64)

        assert first_config == first_worker
        assert second_config == second_worker
        assert first_config != second_config

    def test_worker_hash_fails_closed_when_bundle_metadata_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        root = self._root / "missing_bundle"
        models_dir = root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "test-model.yaml").write_text(
            yaml.dump(
                {
                    "sie_id": "test/model",
                    "hf_id": "test/model",
                    "inputs": {"text": True},
                    "tasks": {"encode": {"dense": {"dim": 3}}},
                    "profiles": {
                        "default": {
                            "adapter_path": "test.hash_adapter:Adapter",
                            "max_batch_tokens": 8192,
                        }
                    },
                }
            )
        )

        def resolve_worker_default_dir(name: str) -> Path:
            return root / name

        worker_ws._bundle_adapter_modules.cache_clear()
        monkeypatch.setattr(worker_ws, "_resolve_default_dir", resolve_worker_default_dir)
        worker_registry = WorkerModelRegistry(
            models_dir=models_dir,
            model_filter=["test/model"],
            pool_name="default",
            enable_hot_reload=False,
        )
        try:
            assert compute_bundle_config_hash_cached(worker_registry, "missing-bundle") == ""
            bundles_dir = root / "bundles"
            bundles_dir.mkdir()
            (bundles_dir / "missing-bundle.yaml").write_text(
                yaml.dump(
                    {
                        "name": "missing-bundle",
                        "priority": 10,
                        "adapters": ["test.hash_adapter"],
                    }
                )
            )
            recovered_hash = compute_bundle_config_hash_cached(worker_registry, "missing-bundle")
            assert len(recovered_hash) == 64
        finally:
            worker_ws._bundle_adapter_modules.cache_clear()

    def test_worker_hash_reports_missing_profile_parent(self) -> None:
        config = SimpleNamespace(profiles={"default": SimpleNamespace(extends="missing-parent")})
        with pytest.raises(ValueError, match="Profile 'missing-parent' referenced via extends does not exist"):
            worker_ws._resolved_profile_for_hash(config, "default")

    def test_order_independence(self) -> None:
        aaa_profile = {"adapter_path": "sie_server.adapters.bert_flash:BertAdapter", "max_batch_tokens": 8192}
        zzz_profile = {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 4096}

        def _make(root, first_id, first_prof, second_id, second_prof):
            bundles_dir = root / "bundles"
            models_dir = root / "models"
            bundles_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            bundle_data = {"name": "default", "priority": 10, "adapters": ["sie_server.adapters.bert_flash"]}
            (bundles_dir / "default.yaml").write_text(yaml.dump(bundle_data))
            reg = ModelRegistry(bundles_dir, models_dir)
            reg.add_model_config({"sie_id": first_id, "profiles": {"default": first_prof}})
            reg.add_model_config({"sie_id": second_id, "profiles": {"default": second_prof}})
            return reg

        r1 = _make(self._root / "order1", "zzz/model", zzz_profile, "aaa/model", aaa_profile)
        r2 = _make(self._root / "order2", "aaa/model", aaa_profile, "zzz/model", zzz_profile)
        assert r1.compute_bundle_config_hash("default") == r2.compute_bundle_config_hash("default")

    def test_empty_registry_hash_is_empty(self) -> None:
        root = self._root / "empty"
        bundles_dir = root / "bundles"
        models_dir = root / "models"
        bundles_dir.mkdir(parents=True)
        models_dir.mkdir()
        (bundles_dir / "default.yaml").write_text(yaml.dump({"name": "default", "priority": 10, "adapters": ["x"]}))
        registry = ModelRegistry(bundles_dir, models_dir)
        assert registry.compute_bundle_config_hash("default") == ""
        assert _compute_worker_hash_via_registry([]) == ""

    def test_hash_algorithm_is_sha256(self) -> None:
        registry = _create_registry_with_model(self._root / "sha256", "test/model")
        h = registry.compute_bundle_config_hash("default")
        assert len(h) == 64
        int(h, 16)
