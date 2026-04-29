# Regression guard: every model.yaml's adapter_path must be declared in some bundle.
#
# If this ever fails again we get a clear CI error instead of a silent
# "Adapter(s) not in any known bundle" warning at gateway bootstrap time
# and a stuck sie-config epoch. See PR #702 (and the Qwen3-VL bundle
# omission surfaced during the 2025-04 eu-central-1 deployment) for context.

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SIE_SERVER_ROOT = Path(__file__).resolve().parents[2]
BUNDLES_DIR = SIE_SERVER_ROOT / "bundles"
MODELS_DIR = SIE_SERVER_ROOT / "models"


def _load_bundle_adapters() -> dict[str, set[str]]:
    # Fail loudly on duplicate bundle names rather than silently letting the
    # later file's adapter list clobber the earlier one: a collision would
    # otherwise corrupt the union `all_declared` below and mask real drift.
    out: dict[str, set[str]] = {}
    source_file: dict[str, Path] = {}
    for bf in sorted(BUNDLES_DIR.glob("*.yaml")):
        data = yaml.safe_load(bf.read_text()) or {}
        name = data.get("name", bf.stem)
        if name in out:
            raise AssertionError(
                f"Duplicate bundle name '{name}' in {source_file[name].name} and "
                f"{bf.name}. Bundle names must be unique across packages/sie_server/bundles/*.yaml."
            )
        out[name] = set(data.get("adapters", []) or [])
        source_file[name] = bf
    return out


def _load_model_adapter_refs() -> dict[str, dict[str, str]]:
    """Return {model_file: {profile_name: adapter_module}}."""
    out: dict[str, dict[str, str]] = {}
    for mf in sorted(MODELS_DIR.glob("*.yaml")):
        data = yaml.safe_load(mf.read_text()) or {}
        profs = data.get("profiles") or {}
        per_profile: dict[str, str] = {}
        for pname, pval in profs.items():
            ap = (pval or {}).get("adapter_path") or ""
            if not ap:
                continue
            per_profile[pname] = ap.split(":", 1)[0]
        if per_profile:
            out[mf.name] = per_profile
    return out


def test_every_model_adapter_is_declared_in_some_bundle() -> None:
    # Guard against vacuous success: if the bundle/model directories ever
    # move (packaging-layout drift, rename, wrong pytest rootdir) the globs
    # would silently return nothing and the regression guard would stop
    # guarding. Asserting non-empty discovery up-front turns a layout break
    # into an immediate, obvious CI failure.
    assert sorted(BUNDLES_DIR.glob("*.yaml")), f"No bundle YAML files found in {BUNDLES_DIR}"
    assert sorted(MODELS_DIR.glob("*.yaml")), f"No model YAML files found in {MODELS_DIR}"

    bundles = _load_bundle_adapters()
    all_declared = {a for adapters in bundles.values() for a in adapters}

    model_refs = _load_model_adapter_refs()
    assert model_refs, (
        f"No model YAMLs declared any adapter_path under {MODELS_DIR}; the coverage "
        "guard cannot validate anything and would pass vacuously."
    )
    missing: list[str] = []
    for mfile, profiles in model_refs.items():
        for pname, module in profiles.items():
            if module not in all_declared:
                missing.append(f"{mfile}::{pname} -> {module}")

    assert not missing, (
        "Model YAMLs reference adapter modules not declared in any bundle. "
        "Either add the module to the appropriate bundle (default/sglang/"
        "transformers5) or remove the model YAML. Without a matching bundle "
        "sie-config cannot route the model and sie-gateway bootstrap stays "
        "stuck.\n\nUnrouteable references:\n  " + "\n  ".join(missing)
    )


def test_load_bundle_adapters_raises_on_duplicate_bundle_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Two YAMLs declaring the same `name:` would previously have silently
    # overwritten each other in `out[name]`, corrupting the union check.
    # Now it must fail loudly with both filenames in the error message so
    # operators can see exactly which bundles collided.
    tmp_bundles = tmp_path / "bundles"
    tmp_bundles.mkdir()
    (tmp_bundles / "a.yaml").write_text("name: shared\npriority: 10\nadapters:\n  - pkg.adapters.one\n")
    (tmp_bundles / "b.yaml").write_text("name: shared\npriority: 20\nadapters:\n  - pkg.adapters.two\n")
    import sys

    this_module = sys.modules[__name__]
    monkeypatch.setattr(this_module, "BUNDLES_DIR", tmp_bundles)
    with pytest.raises(AssertionError, match=r"Duplicate bundle name 'shared'.*a\.yaml.*b\.yaml"):
        _load_bundle_adapters()


def test_bundle_and_model_dirs_are_non_empty() -> None:
    # Companion guard for the parametrized suite below: an empty bundles/
    # directory would cause `pytest.mark.parametrize` to collect zero
    # test cases, which is a silent pass. Asserting here turns that into
    # a loud failure regardless of what the parametrized test does.
    assert sorted(BUNDLES_DIR.glob("*.yaml")), f"No bundle YAML files found in {BUNDLES_DIR}"
    assert sorted(MODELS_DIR.glob("*.yaml")), f"No model YAML files found in {MODELS_DIR}"


@pytest.mark.parametrize("bundle_yaml", sorted(BUNDLES_DIR.glob("*.yaml")))
def test_bundle_yaml_has_required_fields(bundle_yaml: Path) -> None:
    data = yaml.safe_load(bundle_yaml.read_text()) or {}
    assert data.get("name"), f"{bundle_yaml.name}: missing 'name'"
    assert isinstance(data.get("priority"), int), f"{bundle_yaml.name}: 'priority' must be int"
    adapters = data.get("adapters")
    assert isinstance(adapters, list), f"{bundle_yaml.name}: 'adapters' must be a list"
    assert adapters, f"{bundle_yaml.name}: 'adapters' must be non-empty"


# NOTE: Intentionally no "adapter appears in exactly one bundle" test.
#
# sie-config (packages/sie_config/src/sie_config/model_registry.py) and
# sie-gateway (packages/sie_gateway/src/state/model_registry.rs) both
# treat multi-bundle adapter membership as a supported state: they
# collect every bundle whose `adapters` list overlaps the model and
# pick the default route by priority. Existing registry tests (e.g.
# test_model_multiple_profiles_different_adapters in test_model_registry.py)
# rely on the same model being compatible with both `default` and
# `sglang`.
#
# Forbidding duplicates here would block legitimate future setups where
# the same adapter is intentionally available in multiple dependency
# stacks (e.g. a CPU bundle and a CUDA bundle that both ship the same
# sentence-transformer adapter). If equal-priority resolution ever
# needs to be fully deterministic in sie-config too (the Rust gateway
# already tie-breaks by bundle name per PR #702, but the Python side
# only sorts by priority), the fix is a registry-level secondary sort
# key, not a CI lint on bundle contents.
