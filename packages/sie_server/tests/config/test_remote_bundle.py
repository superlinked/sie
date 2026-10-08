"""Remote profiles route to the remote bundle, and a single server still serves them.

The gateway picks the compatible bundle with the lowest priority number. A
cluster runs the remote bundle on the only worker lane that holds upstream
credentials, so every remote adapter must be in that bundle and the bundle
must outrank any other bundle that lists a remote adapter. A single server
serves the default bundle when no bundle is named, so the default bundle keeps
the remote adapters too. The chart runs the remote bundle on another bundle's
image; that image's dependencies are checked in tools/ci/tests/test_helm_render.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml

BUNDLES_DIR = Path(__file__).resolve().parents[2] / "bundles"
REMOTE_ADAPTER_PREFIX = "sie_server.adapters.remote."


def load_bundles() -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for path in sorted(BUNDLES_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        bundles[data.get("name", path.stem)] = data
    return bundles


def remote_adapters(bundle: dict[str, Any]) -> set[str]:
    return {adapter for adapter in bundle.get("adapters") or [] if adapter.startswith(REMOTE_ADAPTER_PREFIX)}


def test_the_remote_bundle_lists_only_existing_remote_adapters() -> None:
    remote = load_bundles()["remote"]

    assert remote["adapters"]
    assert remote_adapters(remote) == set(remote["adapters"])
    assert all(importlib.util.find_spec(adapter) is not None for adapter in remote["adapters"])


def test_every_remote_adapter_in_any_bundle_is_in_the_remote_bundle() -> None:
    bundles = load_bundles()

    listed = set().union(*(remote_adapters(bundle) for bundle in bundles.values()))

    assert listed == set(bundles["remote"]["adapters"])


def test_the_remote_bundle_outranks_every_other_bundle_with_a_remote_adapter() -> None:
    bundles = load_bundles()
    others = {
        name: bundle["priority"] for name, bundle in bundles.items() if name != "remote" and remote_adapters(bundle)
    }

    assert others
    assert all(bundles["remote"]["priority"] < priority for priority in others.values()), others


def test_the_default_bundle_keeps_the_remote_adapters_for_a_single_server() -> None:
    bundles = load_bundles()

    assert set(bundles["remote"]["adapters"]) <= set(bundles["default"]["adapters"])
