"""Composite runtime bundles.

A composite bundle lists ordinary bundles under ``components:``. It runs their
environments side by side on the same accelerator, behind one worker-sidecar
and one queue lane. The image installs each component's saved dependency
overlay under ``/app/bundle-libs.d/<component>``. ``sie-server serve`` then
supervises one worker process per component (see
:mod:`sie_server.composite_supervisor`), each with its own IPC socket and its own
environment overlay, selected through ``SIE_BUNDLE_LIBS``.

Every component process loads the composite's whole model catalog and hashes
the composite's whole adapter scope, so all of them advertise the same bundle
config hash. Each one serves only the adapters of its own component and reports
the rest as unsupported; the sidecar places a model only on a process that
serves it.

This module stays free of serving dependencies so the image build, the CLI and
tests can use it without Torch.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

COMPONENT_ENV = "SIE_BUNDLE_COMPONENT"
"""The component bundle a composite worker process serves (set by the supervisor)."""

BUNDLE_LIBS_ENV = "SIE_BUNDLE_LIBS"
"""Dependency overlay directory the interpreter puts ahead of the shared venv."""

COMPONENT_LIBS_ROOT = "/app/bundle-libs.d"
"""Where a composite image installs each component's overlay."""

ORDINARY_LIBS_DIR = "/app/bundle-libs"
"""Where an ordinary bundle's image installs its overlay."""

DEFAULT_PLATFORM = "cuda12"
_MIN_COMPONENTS = 2
_TOKEN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CompositeBundleError(ValueError):
    """A composite bundle definition is invalid."""


def load_bundle_definition(bundles_dir: Path, name: str) -> dict[str, Any]:
    """Read one bundle YAML from ``bundles_dir``."""
    path = bundles_dir / f"{name}.yaml"
    if not path.is_file():
        msg = f"bundle {name!r} not found in {bundles_dir}"
        raise CompositeBundleError(msg)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        msg = f"bundle {name!r} is not a mapping"
        raise CompositeBundleError(msg)
    return data


def declared_components(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ``components`` a bundle definition declares, or ``()``."""
    raw = data.get("components")
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        msg = f"bundle {data.get('name')!r}: components must be a list of bundle names"
        raise CompositeBundleError(msg)
    return tuple(raw)


def bundle_components(bundles_dir: Path, name: str) -> tuple[str, ...]:
    """Return the component bundles of ``name``; ``()`` for an ordinary bundle."""
    return declared_components(load_bundle_definition(bundles_dir, name))


def is_composite(bundles_dir: Path, name: str) -> bool:
    """Whether ``name`` names a composite bundle in ``bundles_dir``."""
    try:
        return bool(bundle_components(bundles_dir, name))
    except CompositeBundleError:
        return False


def active_component(environ: Mapping[str, str] | None = None) -> str | None:
    """The component this process serves, or ``None`` outside a composite."""
    value = (os.environ if environ is None else environ).get(COMPONENT_ENV, "").strip()
    return value or None


def component_socket_path(socket_path: str, component: str) -> str:
    """Return the IPC socket a component process serves.

    ``/var/run/sie/ipc-0.sock`` becomes ``/var/run/sie/ipc-0.<component>.sock``.
    The Helm chart renders the sidecar's ``SIE_IPC_SOCKET_PATHS`` with the same
    rule.
    """
    path = Path(socket_path)
    return str(path.with_name(f"{path.stem}.{component}{path.suffix}"))


def component_libs_dir(component: str, root: str = COMPONENT_LIBS_ROOT) -> str:
    """Return the overlay directory a composite image installs ``component`` into."""
    return f"{root.rstrip('/')}/{component}"


def validate_composite_bundle(bundles_dir: Path, name: str) -> tuple[str, ...]:
    """Validate a composite bundle and return its components.

    A composite names at least two distinct ordinary bundles on one platform
    and engine, declares no dependencies of its own, and lists exactly the
    union of its components' adapters.

    Raises:
        CompositeBundleError: The definition breaks one of those rules.
    """
    data = load_bundle_definition(bundles_dir, name)
    components = declared_components(data)
    errors: list[str] = []
    if len(components) < _MIN_COMPONENTS:
        errors.append("a composite needs at least two components")
    if len(set(components)) != len(components):
        errors.append("components must be distinct")
    if data.get("deps"):
        errors.append("a composite declares no deps; each component process uses its own saved lock")
    expected: list[str] = []
    platforms: set[str] = set()
    engines: set[str] = set()
    for component in components:
        if not _TOKEN.fullmatch(component):
            errors.append(f"component {component!r} is not a valid bundle name")
            continue
        try:
            component_data = load_bundle_definition(bundles_dir, component)
        except CompositeBundleError as exc:
            errors.append(str(exc))
            continue
        if declared_components(component_data):
            errors.append(f"component {component!r} is itself a composite")
        platforms.add(str(component_data.get("platform", DEFAULT_PLATFORM)))
        engines.add(str(component_data.get("engine", "pytorch")))
        for adapter in component_data.get("adapters") or []:
            if adapter not in expected:
                expected.append(adapter)
    if len(platforms) > 1:
        errors.append(f"components target different platforms: {sorted(platforms)}")
    if engines - {"pytorch"}:
        errors.append("every component must use the pytorch engine")
    if platforms and str(data.get("platform", DEFAULT_PLATFORM)) not in platforms:
        errors.append("the composite platform must match its components")
    declared = list(data.get("adapters") or [])
    missing = [adapter for adapter in expected if adapter not in declared]
    extra = [adapter for adapter in declared if adapter not in expected]
    if missing or extra or len(set(declared)) != len(declared):
        errors.append(
            "adapters must list each component adapter exactly once "
            f"(missing: {missing or 'none'}; not in any component: {extra or 'none'})"
        )
    if errors:
        msg = f"composite bundle {name!r}: " + "; ".join(errors)
        raise CompositeBundleError(msg)
    return components


def overlay_targets(bundles_dir: Path, name: str) -> list[tuple[str, str]]:
    """Return ``(bundle, directory)`` for each dependency overlay an image of ``name`` installs.

    An ordinary bundle installs its own saved lock at ``/app/bundle-libs``; a
    composite installs each component's saved lock under
    ``/app/bundle-libs.d/<component>``.
    """
    components = validate_composite_bundle(bundles_dir, name) if is_composite(bundles_dir, name) else ()
    if not components:
        return [(name, ORDINARY_LIBS_DIR)]
    return [(component, component_libs_dir(component)) for component in components]


def main(argv: list[str] | None = None) -> int:
    """Print the image build's overlay plan: one ``<bundle> <directory>`` line each."""
    import argparse  # noqa: PLC0415 - command-line entry point only

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=("overlays",))
    parser.add_argument("--bundles-dir", type=Path, required=True)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    for bundle, directory in overlay_targets(args.bundles_dir, args.bundle):
        print(f"{bundle} {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
