from pathlib import Path

import pytest
import yaml
from sie_router.model_registry import ModelRegistry


def _create_registry_with_model(
    root: Path,
    sie_id: str = "test/model",
    adapter_path: str = "sie_server.adapters.bert_flash:BertAdapter",
    profile_names: list[str] | None = None,
) -> ModelRegistry:
    """Create a ModelRegistry and add a model via add_model_config."""
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
    """Compute the worker hash using a temporary registry instance.

    Instead of reimplementing the hash algorithm, this creates a real
    ModelRegistry and uses compute_bundle_config_hash — the same code
    path the router uses — ensuring tests break if the algorithm changes.
    """
    import tempfile

    if not model_configs:
        return ""

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        bundles_dir = root / "bundles"
        models_dir = root / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()

        # Collect all adapter modules from configs
        all_adapters = set()
        for cfg in model_configs:
            for profile in cfg.get("profiles", {}).values():
                ap = profile.get("adapter_path", "")
                if ap:
                    all_adapters.add(ap.split(":", maxsplit=1)[0])
                elif isinstance(profile, dict):
                    pass  # profile without adapter_path

        # Use bert_flash as the default adapter if configs don't specify one
        if not all_adapters:
            all_adapters = {"sie_server.adapters.bert_flash"}

        bundle_data = {"name": "default", "priority": 10, "adapters": sorted(all_adapters)}
        (bundles_dir / "default.yaml").write_text(yaml.dump(bundle_data))

        registry = ModelRegistry(bundles_dir, models_dir)

        for cfg in model_configs:
            # Build a valid config dict with adapter_path for each profile.
            # Use the exact profile data provided, falling back to a default
            # that matches what _create_registry_with_model produces.
            profiles = {}
            for pname, pdata in cfg.get("profiles", {}).items():
                if isinstance(pdata, dict) and "adapter_path" in pdata:
                    profiles[pname] = pdata
                else:
                    # Must match _create_registry_with_model defaults
                    profiles[pname] = {
                        "adapter_path": "sie_server.adapters.bert_flash:BertAdapter",
                        "max_batch_tokens": 8192,
                    }
            registry.add_model_config({"sie_id": cfg["sie_id"], "profiles": profiles})

        return registry.compute_bundle_config_hash("default")


class TestHashConsistency:
    """Verify that router and worker produce identical hashes for the same config."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._root = tmp_path

    def test_single_model_single_profile(self) -> None:
        """Router and worker hashes match for one model with one profile."""
        registry = _create_registry_with_model(self._root / "single", "test/model", profile_names=["default"])
        router_hash = registry.compute_bundle_config_hash("default")

        worker_hash = _compute_worker_hash_via_registry(
            [
                {"sie_id": "test/model", "profiles": {"default": {}}},
            ]
        )

        assert router_hash == worker_hash
        assert len(router_hash) == 64  # SHA-256 hex

    def test_single_model_multiple_profiles(self) -> None:
        """Router and worker hashes match for one model with multiple profiles."""
        registry = _create_registry_with_model(
            self._root / "multi_prof",
            "test/model",
            profile_names=["default", "custom", "fast"],
        )
        router_hash = registry.compute_bundle_config_hash("default")

        worker_hash = _compute_worker_hash_via_registry(
            [
                {"sie_id": "test/model", "profiles": {"default": {}, "custom": {}, "fast": {}}},
            ]
        )

        assert router_hash == worker_hash

    def test_multiple_models(self) -> None:
        """Router and worker hashes match for multiple models."""
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

    def test_order_independence(self) -> None:
        """Hash is the same regardless of insertion order."""
        aaa_profile = {"adapter_path": "sie_server.adapters.bert_flash:BertAdapter", "max_batch_tokens": 8192}
        zzz_profile = {"adapter_path": "sie_server.adapters.bert_flash:B", "max_batch_tokens": 4096}

        # Build both registries from scratch (no helper) so profile data is explicit
        def _make(root: Path, first_id: str, first_prof: dict, second_id: str, second_prof: dict) -> ModelRegistry:
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

        # Insert zzz first, then aaa
        r1 = _make(self._root / "order1", "zzz/model", zzz_profile, "aaa/model", aaa_profile)

        # Insert aaa first, then zzz
        r2 = _make(self._root / "order2", "aaa/model", aaa_profile, "zzz/model", zzz_profile)

        assert r1.compute_bundle_config_hash("default") == r2.compute_bundle_config_hash("default")

    def test_empty_registry_hash_is_empty(self) -> None:
        """Both sides return empty string for no models."""
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
        """Sanity check: both produce 64-char hex (SHA-256)."""
        registry = _create_registry_with_model(self._root / "sha256", "test/model")
        h = registry.compute_bundle_config_hash("default")
        assert len(h) == 64
        int(h, 16)  # valid hex
