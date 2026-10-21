"""Composite bundle definitions and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_server.composite_bundles import (
    COMPONENT_ENV,
    CompositeBundleError,
    active_component,
    bundle_components,
    component_libs_dir,
    component_socket_path,
    is_composite,
    validate_composite_bundle,
)

BUNDLES_DIR = Path(__file__).resolve().parents[1] / "bundles"


def _composites() -> list[str]:
    return sorted(path.stem for path in BUNDLES_DIR.glob("*.yaml") if is_composite(BUNDLES_DIR, path.stem))


def test_repository_ships_the_default_sglang_composite() -> None:
    assert "default-sglang" in _composites()
    assert bundle_components(BUNDLES_DIR, "default-sglang") == ("default", "sglang")


@pytest.mark.parametrize("name", _composites())
def test_every_composite_is_valid(name: str) -> None:
    components = validate_composite_bundle(BUNDLES_DIR, name)
    data = yaml.safe_load((BUNDLES_DIR / f"{name}.yaml").read_text())
    # Never preferred by priority: the gateway reaches a composite only through
    # lane coverage, so it must rank below every component.
    component_priorities = [
        int(yaml.safe_load((BUNDLES_DIR / f"{c}.yaml").read_text()).get("priority", 100)) for c in components
    ]
    assert int(data["priority"]) > max(component_priorities)
    assert data.get("default") is False


def test_adapter_order_does_not_matter(tmp_path: Path) -> None:
    _write(tmp_path, "a", {"adapters": ["m.a"]})
    _write(tmp_path, "b", {"adapters": ["m.b"]})
    _write(tmp_path, "ab", {"components": ["a", "b"], "adapters": ["m.b", "m.a"]})
    assert validate_composite_bundle(tmp_path, "ab") == ("a", "b")


def test_ordinary_bundles_are_not_composite() -> None:
    assert bundle_components(BUNDLES_DIR, "default") == ()
    assert not is_composite(BUNDLES_DIR, "sglang")
    assert not is_composite(BUNDLES_DIR, "no-such-bundle")


def _write(tmp: Path, name: str, body: dict) -> None:
    (tmp / f"{name}.yaml").write_text(yaml.safe_dump({"name": name, **body}))


@pytest.mark.parametrize(
    ("composite", "match"),
    [
        ({"components": ["a"], "adapters": ["m.a"]}, "at least two"),
        ({"components": ["a", "a"], "adapters": ["m.a"]}, "distinct"),
        ({"components": ["a", "b"], "adapters": ["m.a", "m.b"], "deps": {"x": ""}}, "no deps"),
        ({"components": ["a", "missing"], "adapters": ["m.a"]}, "not found"),
        ({"components": ["a", "b"], "adapters": ["m.a", "m.a", "m.b"]}, "exactly once"),
        ({"components": ["a", "b"], "adapters": ["m.a"]}, r"missing: \['m.b'\]"),
        ({"components": ["a", "b"], "adapters": ["m.a", "m.b", "m.z"]}, r"not in any component: \['m.z'\]"),
        ({"components": ["a", "c"], "adapters": ["m.a", "m.c"]}, "platforms"),
        ({"components": ["a", "nested"], "adapters": ["m.a", "m.b"]}, "itself a composite"),
    ],
)
def test_invalid_composites_are_refused(tmp_path: Path, composite: dict, match: str) -> None:
    _write(tmp_path, "a", {"adapters": ["m.a"]})
    _write(tmp_path, "b", {"adapters": ["m.b"]})
    _write(tmp_path, "c", {"adapters": ["m.c"], "platform": "cuda13"})
    _write(tmp_path, "nested", {"components": ["a", "b"], "adapters": ["m.a", "m.b"]})
    _write(tmp_path, "bad", composite)
    with pytest.raises(CompositeBundleError, match=match):
        validate_composite_bundle(tmp_path, "bad")


def test_component_socket_and_overlay_paths() -> None:
    assert component_socket_path("/var/run/sie/ipc.sock", "default") == "/var/run/sie/ipc.default.sock"
    assert component_socket_path("/var/run/sie/ipc-3.sock", "sglang") == "/var/run/sie/ipc-3.sglang.sock"
    assert component_libs_dir("sglang") == "/app/bundle-libs.d/sglang"


def test_active_component_reads_the_supervisor_variable() -> None:
    assert active_component({}) is None
    assert active_component({COMPONENT_ENV: " "}) is None
    assert active_component({COMPONENT_ENV: "sglang"}) == "sglang"
