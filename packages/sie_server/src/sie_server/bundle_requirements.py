"""Dependency-only bundle requirement normalization.

This module deliberately stays free of the serving/runtime dependency graph so
release tooling can hash bundle requirements without installing Torch or CUDA.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

_CUDA_ONLY_PACKAGES = frozenset({"fla-core", "flash-attn", "xformers"})
_EXACT_OPERATORS = frozenset({"==", "==="})
_DOWNGRADE_EXAMPLES = ("sentence-transformers", "gliner")


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve_bundle_requirements(
    bundle_deps: Mapping[str, object],
    *,
    exclude_cuda: bool = False,
) -> list[str]:
    """Convert a bundle ``deps`` mapping into canonical PEP 508 strings."""
    requirements: list[str] = []
    for package, constraint in bundle_deps.items():
        normalized = _normalize_package_name(package)
        if exclude_cuda and normalized in _CUDA_ONLY_PACKAGES:
            continue

        if isinstance(constraint, Mapping):
            fields = cast("Mapping[str, object]", constraint)
            url = fields.get("url", "")
            marker = fields.get("marker", "")
            version = fields.get("version", "")
            if url:
                dependency = f"{package} @ {url}"
                if marker:
                    dependency += f" ; {marker}"
                requirements.append(dependency)
            elif version:
                dependency = f"{package}{version}"
                if marker:
                    dependency += f" ; {marker}"
                requirements.append(dependency)
            continue

        requirements.append(f"{package}{constraint}" if constraint else package)
    return requirements


def normalized_bundle_requirements(bundle_deps: Mapping[str, object]) -> list[str]:
    """Return the marker-free, sorted requirements used by release pins."""
    return sorted(
        requirement.split(";", maxsplit=1)[0].strip() for requirement in resolve_bundle_requirements(bundle_deps)
    )


def bundle_requirements_sha256(bundle_deps: Mapping[str, object]) -> str:
    """Hash the exact normalized requirement payload baked by worker images."""
    payload = "\n".join(normalized_bundle_requirements(bundle_deps)).encode()
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class LockEdge:
    """One uv.lock dependency edge, including any requested extras."""

    name: str
    extras: frozenset[str]


@dataclass(frozen=True)
class UvLock:
    """Package graph read from a uv.lock file.

    ``versions`` holds one constraint string per unambiguous package. Packages
    with two public versions are listed in ``ambiguous`` and omitted from
    ``versions`` so image builds skip them instead of aborting. ``dependencies``
    and ``optional_dependencies`` are the edges of the lock entries that the
    constraint installs, keyed by normalized name. A shared public version is
    emitted as ``2.9.1.*`` and takes edges from the public ``2.9.1`` entry
    only, not from a local build such as ``2.9.1+cu129``.
    """

    versions: Mapping[str, str]
    ambiguous: frozenset[str]
    dependencies: Mapping[str, tuple[LockEdge, ...]]
    optional_dependencies: Mapping[str, Mapping[str, tuple[LockEdge, ...]]]
    names: Mapping[str, str]


def parse_uv_lock(source: str | Path) -> UvLock:
    """Parse package tables from a uv.lock file into a dependency graph."""
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"invalid uv.lock: {exc}"
        raise ValueError(msg) from exc
    packages = data.get("package", [])
    if not isinstance(packages, list):
        msg = "uv.lock package table is not a list"
        raise ValueError(msg)

    grouped: dict[str, list[dict[str, object]]] = {}
    names: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            continue
        raw_name = package.get("name")
        raw_version = package.get("version")
        if not isinstance(raw_name, str) or not isinstance(raw_version, str):
            continue
        normalized = _normalize_package_name(raw_name)
        names.setdefault(normalized, raw_name)
        grouped.setdefault(normalized, []).append(package)

    versions: dict[str, str] = {}
    ambiguous: set[str] = set()
    dependencies: dict[str, tuple[LockEdge, ...]] = {}
    optional_dependencies: dict[str, dict[str, tuple[LockEdge, ...]]] = {}
    for normalized, entries in grouped.items():
        found = {version for entry in entries if isinstance(version := entry.get("version"), str)}
        resolved = _constraint_version(normalized, found)
        if resolved is None:
            ambiguous.add(normalized)
        else:
            versions[normalized] = resolved
        base: set[LockEdge] = set()
        optional: dict[str, set[LockEdge]] = {}
        for entry in _entries_for_emitted_constraint(entries, resolved):
            base.update(_lock_edges(entry.get("dependencies")))
            raw_optional = entry.get("optional-dependencies")
            if not isinstance(raw_optional, dict):
                continue
            for extra, items in raw_optional.items():
                if isinstance(extra, str):
                    optional.setdefault(extra.lower(), set()).update(_lock_edges(items))
        dependencies[normalized] = _freeze_edges(base)
        if optional:
            optional_dependencies[normalized] = {extra: _freeze_edges(edges) for extra, edges in optional.items()}

    return UvLock(
        versions=versions,
        ambiguous=frozenset(ambiguous),
        dependencies=dependencies,
        optional_dependencies=optional_dependencies,
        names=names,
    )


def locked_versions_from_uv_lock(source: str | Path) -> dict[str, str]:
    """Return locked constraint versions keyed by normalized package name.

    One lock entry becomes that exact version string. Several entries that
    share a public version (``torch`` ``2.9.1`` and ``2.9.1+cu129``) become a
    prefix match, ``2.9.1.*``, so each locked build matches and a newer
    release does not. Distinct public versions are omitted; the constraints
    file records that skip instead of failing the image build.
    """
    return dict(parse_uv_lock(source).versions)


def merge_locked_requirements(requirements: Sequence[str], locked_versions: Mapping[str, str]) -> list[str]:
    """Pin ranged requirements to the lock unless the bundle spec overrides it.

    Exact ``==`` / ``===`` pins and URL or VCS specs are returned unchanged.
    A range is rewritten only when a locked version still satisfies it, so a
    bundle can deliberately move off the lock (transformers 5 versus a 4.x
    pin) without an unsatisfiable constraint. Packages missing from the lock
    stay as written.
    """
    merged: list[str] = []
    for requirement in requirements:
        parsed = Requirement(requirement)
        if _is_bundle_override(parsed):
            merged.append(requirement)
            continue
        pinned = locked_versions.get(_normalize_package_name(parsed.name))
        if pinned is None or not _locked_constraint_satisfies(parsed, pinned):
            merged.append(requirement)
            continue
        merged.append(_format_locked_requirement(requirement, parsed, pinned))
    return merged


def lock_constraint_lines(requirements: Sequence[str], lock: UvLock) -> list[str]:
    """Return constraints for the locked transitive closure of ``requirements``.

    Each pin is ``name==version`` with no extras and no markers. Pip rejects
    extras in a constraints file, and a marker would leave the pin inactive on
    other platforms. The file starts with a comment that ranged dependencies
    follow ``uv.lock`` and can therefore be older than an unconstrained install.

    Exact ``==`` pins and URL or VCS specs stay out of the file. An exact pin
    contributes lock neighbors only when it matches the locked version. A pin
    that differs (``torch==2.11.0`` against a ``2.9.1`` lock) is an untrusted
    root, same as a range that does not contain the lock: those edges describe
    a distribution the bundle will not install. A neighbor an untrusted root
    can reach is not pinned, even when a constrained root reaches it too.
    Pinning that shared package to the lock can contradict the untrusted
    distribution (``huggingface-hub`` 0.36 from ``sentence-transformers``
    versus ``huggingface-hub>=1`` from transformers 5). The constrained
    requirement itself stays pinned. A package with two public versions is
    skipped with a comment instead of aborting.
    """
    grouped = _requirement_groups(requirements)
    constrained: list[tuple[str, frozenset[str]]] = []
    exact: list[tuple[str, frozenset[str]]] = []
    untrusted: list[tuple[str, frozenset[str]]] = []
    for name, reqs in grouped.items():
        kind = _root_kind(name, reqs, lock)
        if kind is None:
            continue
        root = (name, _root_extras(reqs))
        if kind == "constrained":
            constrained.append(root)
        elif kind == "exact":
            exact.append(root)
        else:
            untrusted.append(root)

    constrained_reach = _reachable(constrained, lock)
    exact_reach = _reachable(exact, lock)
    untrusted_reach = _reachable(untrusted, lock)
    # A transitive package an untrusted root can reach stays unpinned, even
    # when a constrained root reaches it too. The image installs the untrusted
    # distribution, and the lock pin can make that solve fail. Direct
    # constrained requirements stay pinned: dropping ``torch`` because
    # transformers 5 also depends on it would float the bundle's own torch.
    constrained_names = {name for name, _extras in constrained}
    trusted = ((constrained_reach | exact_reach) - untrusted_reach) | constrained_names

    pins: list[str] = []
    skips: list[str] = []
    pinned_names: set[str] = set()
    for name in sorted(trusted):
        reqs = grouped.get(name, [])
        if _pin_blocked(name, reqs, lock):
            continue
        display = lock.names.get(name, name)
        if name in lock.ambiguous:
            skips.append(f"# {display} skipped: uv.lock has multiple public versions")
            continue
        version = lock.versions.get(name)
        if version is None:
            continue
        pins.append(f"{display}=={version}")
        pinned_names.add(name)
    return [*_constraint_header(pinned_names), *skips, *pins]


def _constraint_version(name: str, versions: set[str]) -> str | None:
    if len(versions) == 1:
        return next(iter(versions))
    try:
        public_versions = {Version(version).public for version in versions}
    except InvalidVersion as exc:
        msg = f"uv.lock pins {name} to an invalid version"
        raise ValueError(msg) from exc
    if len(public_versions) != 1:
        return None
    return f"{next(iter(public_versions))}.*"


def _entries_for_emitted_constraint(
    entries: Sequence[Mapping[str, object]],
    constraint: str | None,
) -> list[Mapping[str, object]]:
    """Edges for the build ``constraint`` installs, not every lock entry.

    A shared public version is emitted as ``2.9.1.*``. That installs the public
    wheel (``2.9.1``), so a local build's edges (``2.9.1+cu129``) do not apply.
    An exact version uses that build only. Ambiguous names keep every entry.
    """
    if constraint is None:
        return list(entries)
    if constraint.endswith(".*"):
        public = constraint.removesuffix(".*")
        return [entry for entry in entries if entry.get("version") == public]
    return [entry for entry in entries if entry.get("version") == constraint]


def _lock_edges(items: object) -> tuple[LockEdge, ...]:
    if not isinstance(items, list):
        return ()
    edges: list[LockEdge] = []
    for item in items:
        if isinstance(item, str):
            edges.append(LockEdge(_normalize_package_name(item), frozenset()))
            continue
        if not isinstance(item, dict):
            continue
        fields = cast("Mapping[str, object]", item)
        raw_name = fields.get("name")
        if not isinstance(raw_name, str):
            continue
        extras = _lock_extras(fields.get("extra", fields.get("extras")))
        edges.append(LockEdge(_normalize_package_name(raw_name), extras))
    return tuple(edges)


def _lock_extras(value: object) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value.lower()})
    if isinstance(value, list):
        return frozenset(item.lower() for item in value if isinstance(item, str))
    return frozenset()


def _freeze_edges(edges: set[LockEdge]) -> tuple[LockEdge, ...]:
    return tuple(sorted(edges, key=lambda edge: (edge.name, sorted(edge.extras))))


def _requirement_groups(requirements: Sequence[str]) -> dict[str, list[Requirement]]:
    grouped: dict[str, list[Requirement]] = {}
    for requirement in requirements:
        parsed = Requirement(requirement)
        grouped.setdefault(_normalize_package_name(parsed.name), []).append(parsed)
    return grouped


def _root_kind(name: str, reqs: Sequence[Requirement], lock: UvLock) -> str | None:
    """Classify a direct requirement as constrained, exact override, or untrusted.

    Untrusted roots are URL/VCS specs, ranges that do not contain the locked
    version, and exact pins of a different version. Their lock edges belong to
    a distribution the bundle will not install, so those edges must not become
    constraints. An exact pin that matches the locked version still contributes
    its edges.
    """
    if name not in lock.versions and name not in lock.ambiguous:
        return None
    if any(req.url is not None for req in reqs):
        return "untrusted"
    version = lock.versions.get(name)
    overrides = [req for req in reqs if _is_bundle_override(req)]
    if overrides:
        if version is not None and all(_locked_constraint_satisfies(req, version) for req in overrides):
            return "exact"
        return "untrusted"
    if version is not None and any(not _locked_constraint_satisfies(req, version) for req in reqs):
        return "untrusted"
    return "constrained"


def _root_extras(reqs: Sequence[Requirement]) -> frozenset[str]:
    extras: set[str] = set()
    for req in reqs:
        extras.update(extra.lower() for extra in req.extras)
    return frozenset(extras)


def _reachable(roots: Sequence[tuple[str, frozenset[str]]], lock: UvLock) -> set[str]:
    seen_packages: set[str] = set()
    seen_extras: set[tuple[str, str]] = set()
    stack = list(roots)
    while stack:
        name, extras = stack.pop()
        if name not in lock.versions and name not in lock.ambiguous:
            continue
        if name not in seen_packages:
            seen_packages.add(name)
            for edge in lock.dependencies.get(name, ()):
                stack.append((edge.name, edge.extras))
        for extra in extras:
            key = (name, extra)
            if key in seen_extras:
                continue
            seen_extras.add(key)
            for edge in lock.optional_dependencies.get(name, {}).get(extra, ()):
                stack.append((edge.name, edge.extras))
    return seen_packages


def _pin_blocked(name: str, reqs: Sequence[Requirement], lock: UvLock) -> bool:
    if not reqs:
        return False
    if any(_is_bundle_override(req) for req in reqs):
        return True
    version = lock.versions.get(name)
    if version is None:
        return False
    return any(not _locked_constraint_satisfies(req, version) for req in reqs)


def _constraint_header(pinned_names: set[str]) -> list[str]:
    lines = [
        "# Ranged dependencies are pinned to uv.lock. This can install an older",
        "# release than an unconstrained build of the same commit.",
    ]
    present = [name for name in _DOWNGRADE_EXAMPLES if name in pinned_names]
    if not present:
        return lines
    shown = " and ".join(present)
    verb = "follow" if len(present) > 1 else "follows"
    lines.append(
        f"# For example, {shown} {verb} uv.lock rather than the newest release that still matches the bundle range."
    )
    return lines


def _locked_constraint_satisfies(requirement: Requirement, constraint: str) -> bool:
    """A prefix pin is usable when its public version is inside the bundle spec."""
    candidate = constraint.removesuffix(".*") if constraint.endswith(".*") else constraint
    try:
        version = Version(candidate)
    except InvalidVersion:
        return False
    return requirement.specifier.contains(version, prereleases=True)


def _is_bundle_override(requirement: Requirement) -> bool:
    """Exact pins and direct URL/VCS references are deliberate bundle overrides."""
    if requirement.url is not None:
        return True
    specs = list(requirement.specifier)
    if len(specs) != 1:
        return False
    spec = specs[0]
    return spec.operator in _EXACT_OPERATORS and not spec.version.endswith(".*")


def _format_locked_requirement(raw: str, requirement: Requirement, version: str) -> str:
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    pinned = f"{requirement.name}{extras}=={version}"
    if requirement.marker is None:
        return pinned
    marker = raw.split(";", maxsplit=1)[1].strip()
    return f"{pinned} ; {marker}"
