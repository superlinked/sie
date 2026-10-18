"""Declared deployments match the reviewed Transformers loading boundary."""

import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

_SERVER = Path(__file__).resolve().parents[2]
_ROOT = _SERVER.parents[1]


def _requirement(values: list[str]) -> Requirement:
    matches = [Requirement(value) for value in values if Requirement(value).name == "transformers"]
    assert len(matches) == 1
    return matches[0]


def test_default_local_and_resolved_tf4_are_the_exact_reviewed_version() -> None:
    project = tomllib.loads((_SERVER / "pyproject.toml").read_text())["project"]
    default = yaml.safe_load((_SERVER / "bundles/default.yaml").read_text())
    locked = tomllib.loads((_ROOT / "uv.lock").read_text())
    runtime = [package for package in locked["package"] if package["name"] == "transformers"]
    assert len(runtime) == 1
    assert runtime[0]["version"] == "4.57.6"
    assert str(_requirement(project["optional-dependencies"]["local"]).specifier) == "==4.57.6"
    assert default["deps"]["transformers"] == "==4.57.6"
    assert "4.57.6" in _requirement(project["dependencies"]).specifier


@pytest.mark.parametrize("version", ["4.57.0", "4.57.5", "5.0.0", "5.0.9", "5.1.9", "5.2.9", "6.0.0"])
def test_server_excludes_known_unreviewed_runtime_ranges(version: str) -> None:
    project = tomllib.loads((_SERVER / "pyproject.toml").read_text())["project"]
    assert version not in _requirement(project["dependencies"]).specifier


def test_server_keeps_fixed_tf5_overlays_and_documents_future_tf4_limit() -> None:
    text = (_SERVER / "pyproject.toml").read_text()
    project = tomllib.loads(text)["project"]
    requirement = _requirement(project["dependencies"])
    assert requirement.specifier == SpecifierSet(">=4.57.6,<6,!=5.0.*,!=5.1.*,!=5.2.*")
    for version in ("4.57.6", "5.3.0", "5.17.0"):
        assert version in requirement.specifier
    # Metadata cannot represent the disjoint support set. The permanent loader
    # guard, not this declaration, refuses future unreviewed 4.x versions.
    assert "4.58.0" in requirement.specifier
    assert "any future unreviewed 4.x version" in text
    tf5 = yaml.safe_load((_SERVER / "bundles/transformers5.yaml").read_text())
    assert tf5["deps"]["transformers"] == ">=5.17,<6"


def test_lock_metadata_preserves_the_declared_base_and_local_requirements() -> None:
    project = tomllib.loads((_SERVER / "pyproject.toml").read_text())["project"]
    locked = tomllib.loads((_ROOT / "uv.lock").read_text())
    server = next(package for package in locked["package"] if package["name"] == "sie-server")
    requirements = [item for item in server["metadata"]["requires-dist"] if item["name"] == "transformers"]
    base = [item for item in requirements if "marker" not in item]
    local = [item for item in requirements if item.get("marker") == "extra == 'local'"]
    assert len(base) == len(local) == 1
    assert SpecifierSet(base[0]["specifier"]) == _requirement(project["dependencies"]).specifier
    assert SpecifierSet(local[0]["specifier"]) == _requirement(project["optional-dependencies"]["local"]).specifier
