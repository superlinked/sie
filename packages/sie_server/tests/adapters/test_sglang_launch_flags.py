"""Every engine flag an SGLang launch carries must be one its bundle's engine declares.

The declared sets are snapshots of each pinned engine's own argument parser,
committed under ``conformance/engine-flags/<engine>-<version>.txt``. The image
check (``tools/ci/engine_launch_contract.py``) takes the snapshot from the
parser inside each built image and fails when it differs, so a snapshot cannot
drift from the engine a bundle installs. This module renders every routed
profile's argv with the real adapter, as the image check does, and holds each
flag to the snapshot of the engine its bundle pins. Flags must be declared
exactly: an argparse prefix match works only until upstream adds a flag that
shares the prefix.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml
from sie_server.adapters.sglang import _memory

_PACKAGE_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ADAPTER_DIR = _PACKAGE_DIR / "src/sie_server/adapters"
_SNAPSHOT_DIR = _REPO_ROOT / "conformance/engine-flags"
_PROBE_SPEC = importlib.util.spec_from_file_location(
    "engine_launch_probe", _REPO_ROOT / "tools/ci/engine_launch_probe.py"
)
assert _PROBE_SPEC is not None
assert _PROBE_SPEC.loader is not None
probe = importlib.util.module_from_spec(_PROBE_SPEC)
sys.modules[_PROBE_SPEC.name] = probe
_PROBE_SPEC.loader.exec_module(probe)

# SGLang 0.5.20 renamed these 0.5.10 launch flags with no alias (public #320).
_RENAMED_IN_0_5_20 = {
    "--mamba-radix-cache-strategy": "--mamba-scheduler-strategy",
    "--cuda-graph-max-bs-decode": "--cuda-graph-max-bs",
    "--disable-prefill-cuda-graph": "--disable-piecewise-cuda-graph",
}

_FLAG_LITERAL = re.compile(r'"(--[a-z0-9][a-z0-9-]*)"')


def _pinned_sglang(bundle: dict) -> str | None:
    spec = (bundle.get("deps") or {}).get("sglang")
    if isinstance(spec, dict):
        spec = spec.get("version")
    if spec is None:
        return None
    assert isinstance(spec, str), f"{bundle['name']} declares sglang as {spec!r}"
    assert spec.startswith("=="), f"{bundle['name']} must pin sglang exactly, not {spec!r}"
    return spec.removeprefix("==")


def _sglang_bundles() -> dict[str, str]:
    bundles = {}
    for path in sorted((_PACKAGE_DIR / "bundles").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        version = _pinned_sglang(data)
        if version is not None:
            bundles[data["name"]] = version
    return bundles


def _declared(version: str) -> frozenset[str]:
    path = _SNAPSHOT_DIR / f"sglang-{version}.txt"
    assert path.is_file(), f"no declared-flag snapshot for sglang {version}; the image check writes {path.name}"
    return frozenset(path.read_text(encoding="utf-8").split())


def _rendered(bundle: str) -> dict[str, list[str]]:
    models_dir = _PACKAGE_DIR / "models"
    routes = probe.routed_profiles(models_dir, probe.bundle_adapter_modules(_PACKAGE_DIR / "bundles", bundle))
    return {
        route: probe.render_sglang_argv(config, models_dir)
        for route, config in routes.items()
        if probe.is_sglang_route(config)
    }


@pytest.mark.parametrize("bundle", sorted(_sglang_bundles()))
def test_every_routed_profile_launches_only_declared_flags(bundle: str) -> None:
    """A flag the engine renamed or never declared fails here rather than when the model loads."""
    declared = _declared(_sglang_bundles()[bundle])
    rendered = _rendered(bundle)
    undeclared = {
        route: missing for route, argv in rendered.items() if (missing := probe.undeclared_flags(argv, declared))
    }

    assert rendered, f"no profile routes to the {bundle} bundle; the catalog walk found nothing"
    assert not undeclared, f"{bundle} profiles launch flags its pinned engine does not declare: {undeclared}"


def test_the_0_5_20_renames_fail_against_the_engine_that_made_them() -> None:
    """The #320 class: each 0.5.10 spelling on the 0.5.20 lane is caught."""
    declared = _declared(_sglang_bundles()["sglang-cu130"])
    old_spellings = {
        flag
        for argv in _rendered("sglang-cu130").values()
        for flag in probe.undeclared_flags([_RENAMED_IN_0_5_20.get(token, token) for token in argv], declared)
    }

    assert old_spellings == set(_RENAMED_IN_0_5_20.values())


def test_every_adapter_flag_literal_is_declared_by_a_pinned_engine() -> None:
    """Covers launch paths no shipped profile exercises, such as LoRA or a width above one."""
    declared = frozenset().union(*(_declared(version) for version in set(_sglang_bundles().values())))
    sources = [*(_ADAPTER_DIR / "sglang").rglob("*.py"), *(_ADAPTER_DIR / "sglang_vision_extract").rglob("*.py")]
    found: dict[str, set[str]] = {}
    for source in sorted(sources):
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            for flag in _FLAG_LITERAL.findall(line):
                found.setdefault(flag, set()).add(str(source.relative_to(_ADAPTER_DIR)))
    undeclared = {flag: sorted(paths) for flag, paths in found.items() if flag not in declared}

    assert "--tensor-parallel-size" in found
    assert not undeclared, f"adapter sources carry flags no pinned engine declares: {undeclared}"


@pytest.mark.parametrize("abbreviation", ["--tp", "--dp", "--pp", "--ep"])
def test_no_parallelism_abbreviation_is_emitted(abbreviation: str) -> None:
    """``--tp`` was emitted for years and only ever worked through argparse prefix matching."""
    for source in sorted((_ADAPTER_DIR / "sglang").rglob("*.py")):
        emitting = [
            line
            for line in source.read_text(encoding="utf-8").splitlines()
            if f'"{abbreviation}"' in line and not line.lstrip().startswith("#")
        ]
        assert not emitting, f"{source.name} emits the abbreviated parallelism flag {abbreviation!r}: {emitting}"


@pytest.fixture(autouse=True)
def mock_memory_hardware(monkeypatch):
    """Launch lifecycle tests use synthetic weight metadata and CUDA capacity."""
    monkeypatch.setattr(_memory, "device_memory", lambda _: (80 * _memory.GIB, 80 * _memory.GIB))
    monkeypatch.setattr(_memory.SGLangMemoryBudget, "reservation", _memory.SGLangReservation(8 * _memory.GIB))
