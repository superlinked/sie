from __future__ import annotations

from pathlib import Path

from sie_sdk.bundle_utils import (
    find_bundle_for_model_adapters,
    find_bundle_for_models,
    match_bundle_model_adapters,
    match_bundle_models,
)


def _write_model(models_dir: Path, name: str, *, pool: str | None = None) -> None:
    pool_line = f"pool: {pool}\n" if pool else ""
    (models_dir / f"{name.replace('/', '__')}.yaml").write_text(
        f"""
sie_id: {name}
{pool_line}profiles:
  default:
    adapter_path: pkg.adapters.sglang:Adapter
""".lstrip()
    )


def test_match_bundle_models_filters_by_pool(tmp_path: Path) -> None:
    bundle_path = tmp_path / "sglang.yaml"
    bundle_path.write_text("adapters:\n  - pkg.adapters.sglang\n")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    _write_model(models_dir, "org/generation")
    _write_model(models_dir, "org/embedding", pool="SGLang-Embedding")

    assert set(match_bundle_models(bundle_path, models_dir)) == {"org/generation", "org/embedding"}
    assert match_bundle_models(bundle_path, models_dir, pool_name="default") == ["org/generation"]
    assert match_bundle_models(bundle_path, models_dir, pool_name="sglang-embedding") == ["org/embedding"]
    assert match_bundle_models(bundle_path, models_dir, pool_name="SGLANG-EMBEDDING") == ["org/embedding"]


def test_match_bundle_models_routes_profile_variants_by_effective_adapter(tmp_path: Path) -> None:
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    default_bundle = bundles_dir / "default.yaml"
    default_bundle.write_text("adapters:\n  - pkg.adapters.pytorch\n")
    candle_bundle = bundles_dir / "candle.yaml"
    candle_bundle.write_text("adapters:\n  - pkg.adapters.candle\n")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "org__embedding.yaml").write_text(
        """
sie_id: org/embedding
profiles:
  default:
    adapter_path: pkg.adapters.pytorch:Adapter
  candle:
    extends: default
    adapter_path: pkg.adapters.candle:Adapter
""".lstrip()
    )

    assert match_bundle_models(default_bundle, models_dir) == ["org/embedding"]
    assert match_bundle_models(candle_bundle, models_dir) == ["org/embedding:candle"]
    assert find_bundle_for_models(["org/embedding"], bundles_dir, models_dir) == "default"
    assert find_bundle_for_models(["org/embedding:candle"], bundles_dir, models_dir) == "candle"


def test_match_bundle_models_omits_bare_model_without_default_profile(tmp_path: Path) -> None:
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    candle_bundle = bundles_dir / "candle.yaml"
    candle_bundle.write_text("adapters:\n  - pkg.adapters.candle\n")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "org__candle-only.yaml").write_text(
        """
sie_id: org/candle-only
profiles:
  candle:
    adapter_path: pkg.adapters.candle:Adapter
""".lstrip()
    )

    assert match_bundle_models(candle_bundle, models_dir) == ["org/candle-only:candle"]
    assert find_bundle_for_models(["org/candle-only"], bundles_dir, models_dir) is None
    assert find_bundle_for_models(["org/candle-only:candle"], bundles_dir, models_dir) == "candle"


def test_match_bundle_model_adapters_supports_in_memory_catalogs(tmp_path: Path) -> None:
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    bundle_path = bundles_dir / "transformers5.yaml"
    bundle_path.write_text("adapters:\n  - pkg.adapters.transformers5\n")
    inventory = {
        "org/default": ({"pkg.adapters.default"}, None),
        "org/lighton": ({"pkg.adapters.transformers5"}, None),
    }

    assert match_bundle_model_adapters(bundle_path, inventory) == ["org/lighton"]
    assert find_bundle_for_model_adapters(["org/lighton"], bundles_dir, inventory) == "transformers5"
