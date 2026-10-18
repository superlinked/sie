"""Check saved bundle closures and the inventory installed from those locks.

This is build tooling: it never imports a model runtime. The scope is the
``bundle-libs`` overlay, not the separately resolved base virtual environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform as host_platform
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import yaml
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from sie_server.bundle_requirements import resolve_bundle_requirements

SCHEMA_VERSION = 1
UV_VERSION = "0.9.28"
PYTHON_VERSION = "3.12"
_MACHINES = {"amd64": "x86_64", "arm64": "aarch64"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def digest(path: Path) -> str:
    """Hash a build input without interpreting it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: object) -> str:
    """Hash JSON data independently of indentation or dictionary order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def policy_digest(compiler: Path | None = None) -> str:
    """Bind the exact verifier, declaration resolver and compilation policy."""
    if compiler is None:
        compiler = Path(__file__).resolve().parents[4] / "tools/ci/bundle_lock.py"
    return canonical_digest(
        {
            "verifier": digest(Path(__file__)),
            "requirements": digest(Path(__file__).with_name("bundle_requirements.py")),
            "compiler": digest(compiler),
        }
    )


@dataclass(frozen=True)
class Target:
    """One image overlay, with an explicit architecture and Python ABI."""

    platform: str
    bundle: str
    architecture: str

    def __post_init__(self) -> None:
        if self.platform not in {"cpu", "cuda12", "cuda13"} or self.architecture not in _MACHINES:
            msg = "unsupported bundle lock platform or architecture"
            raise ValueError(msg)
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.bundle):
            msg = "invalid bundle name"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        return f"{self.platform}-{self.bundle}-{self.architecture}"

    @property
    def python_platform(self) -> str:
        return f"{_MACHINES[self.architecture]}-manylinux_2_35"

    def as_dict(self) -> dict[str, str]:
        return {
            "platform": self.platform,
            "bundle": self.bundle,
            "architecture": self.architecture,
            "python_version": PYTHON_VERSION,
            "python_platform": self.python_platform,
        }

    def marker_environment(self) -> dict[str, str]:
        return {
            "implementation_name": "cpython",
            "implementation_version": "3.12.0",
            "os_name": "posix",
            "platform_machine": _MACHINES[self.architecture],
            "platform_python_implementation": "CPython",
            "platform_release": "",
            "platform_system": "Linux",
            "platform_version": "",
            "python_full_version": "3.12.0",
            "python_version": PYTHON_VERSION,
            "sys_platform": "linux",
            "extra": "",
        }


def bundle_requirements(bundles: Path, target: Target) -> list[str]:
    """Use the same declaration and CPU exclusions as resolve-deps."""
    data = yaml.safe_load((bundles / f"{target.bundle}.yaml").read_text(encoding="utf-8"))
    return resolve_bundle_requirements(data["deps"], exclude_cuda=target.platform == "cpu")


def root_packages(root_lock: Path) -> dict[str, list[dict[str, Any]]]:
    """Keep exact versions and artifact sources, including local Torch builds."""
    result: dict[str, list[dict[str, Any]]] = {}
    for row in tomllib.loads(root_lock.read_text(encoding="utf-8"))["package"]:
        source = row.get("source", {})
        if "editable" in source or "virtual" in source:
            continue  # Workspace projects are installed separately from bundle-libs.
        if "version" not in row or not ({"registry", "url", "git"} & source.keys()):
            msg = f"unsupported root package source: {row.get('name')}"
            raise ValueError(msg)
        result.setdefault(canonicalize_name(row["name"]), []).append(row)
    return result


def root_constraints(root_lock: Path, exceptions: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Pin every root package, except an explicitly recorded exact alternative.

    ``==2.9.1`` admits a local build such as ``+cpu`` but excludes ``.post1``.
    The artifact and exact local version are checked separately against the
    exception manifest. This never drops an entire untrusted transitive graph.
    """
    result = []
    for name, rows in sorted(root_packages(root_lock).items()):
        if name in exceptions:
            version = exceptions[name]["version"]
        else:
            versions = {Version(row["version"]).public for row in rows}
            if len(versions) != 1:
                msg = f"root has multiple public versions of {name}; record an exact target exception"
                raise ValueError(msg)
            version = versions.pop()
        Version(version)
        result.append(f"{name}=={version}")
    return result


def _public_url(value: object) -> str:
    if not isinstance(value, str):
        msg = "lock source URL must be a string"
        raise ValueError(msg)
    url = urlsplit(value)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        msg = "lock sources must use credential-free HTTPS"
        raise ValueError(msg)
    return value


def artifact_sources(package: Mapping[str, Any]) -> list[dict[str, str]]:
    """Canonical immutable source identities, excluding upload-time metadata."""
    if "vcs" in package:
        vcs = package["vcs"]
        commit = vcs.get("commit-id", "")
        if vcs.get("type") != "git" or not isinstance(commit, str) or not _GIT_SHA.fullmatch(commit):
            msg = "VCS lock source requires an exact Git commit"
            raise ValueError(msg)
        return [{"url": _public_url(vcs.get("url")), "git_commit": commit}]
    artifacts = list(package.get("wheels", []))
    artifacts.extend(package[key] for key in ("sdist", "archive") if key in package)
    if not artifacts:
        msg = f"no immutable artifact for {package.get('name')}"
        raise ValueError(msg)
    result = []
    for artifact in artifacts:
        sha = artifact.get("hashes", {}).get("sha256", "")
        if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
            msg = "every locked artifact requires SHA256"
            raise ValueError(msg)
        result.append({"url": _public_url(artifact.get("url")), "sha256": sha})
    return sorted(result, key=lambda row: row["url"])


def read_packages(path: Path, target: Target) -> dict[str, dict[str, Any]]:
    """Read a fixed-target PEP751 lock, refusing unbound or duplicate entries."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if data.get("lock-version") != "1.0":
        msg = "unsupported bundle lock format"
        raise ValueError(msg)
    result = {}
    for row in data.get("packages", []):
        if row.get("marker") and not Marker(row["marker"]).evaluate(target.marker_environment()):
            continue
        name = canonicalize_name(row["name"])
        Version(row["version"])
        artifact_sources(row)
        if name in result:
            msg = f"duplicate locked distribution: {name}"
            raise ValueError(msg)
        wheels = row.get("wheels", [])
        if wheels:
            compatible = False
            for wheel in wheels:
                filename = unquote(urlsplit(wheel["url"]).path).rsplit("/", maxsplit=1)[-1]
                wheel_name, version, _build, tags = parse_wheel_filename(filename)
                if wheel_name != name or version != Version(row["version"]):
                    msg = f"locked wheel identity differs for {name}"
                    raise ValueError(msg)
                compatible |= any(
                    tag.platform == "any" or tag.platform.endswith(_MACHINES[target.architecture]) for tag in tags
                )
            if not compatible:
                msg = f"locked wheel architecture differs for {name}"
                raise ValueError(msg)
        result[name] = row
    if not result:
        msg = "bundle lock has no packages for the selected target"
        raise ValueError(msg)
    return result


def _root_artifacts(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    result = set()
    for row in rows:
        if "git" in row["source"]:
            url = urlsplit(row["source"]["git"])
            if not _GIT_SHA.fullmatch(url.fragment):
                msg = "root VCS source requires an exact Git commit"
                raise ValueError(msg)
            result.add((_public_url(urlunsplit(url._replace(query="", fragment=""))), url.fragment))
        for artifact in [*row.get("wheels", []), *([row["sdist"]] if "sdist" in row else [])]:
            result.add((artifact["url"], artifact["hash"].removeprefix("sha256:")))
    return result


def required_exceptions(
    packages: Mapping[str, Mapping[str, Any]],
    root_lock: Path,
) -> dict[str, dict[str, str]]:
    """Report exact version/source deviations; this does not approve them."""
    root = root_packages(root_lock)
    result = {}
    for name, package in packages.items():
        if name not in root:
            continue
        same_version = [row for row in root[name] if Version(row["version"]) == Version(package["version"])]
        sources = artifact_sources(package)
        allowed = _root_artifacts(same_version)
        if not same_version or any(
            (row["url"], row.get("sha256", row.get("git_commit", ""))) not in allowed for row in sources
        ):
            result[name] = {"version": package["version"], "artifacts_sha256": canonical_digest(sources)}
    return result


def validate_exceptions(
    packages: Mapping[str, Mapping[str, Any]],
    root_lock: Path,
    exceptions: Mapping[str, Mapping[str, str]],
) -> None:
    """Fail on an unapproved deviation, stale exception, or changed source."""
    required = required_exceptions(packages, root_lock)
    if set(required) != set(exceptions):
        msg = f"bundle exceptions differ: missing={sorted(set(required) - set(exceptions))}, stale={sorted(set(exceptions) - set(required))}"
        raise ValueError(msg)
    for name, expected in required.items():
        exception = exceptions[name]
        if (
            any(exception.get(key) != value for key, value in expected.items())
            or not exception.get("reason", "").strip()
        ):
            msg = f"version/source exception is not exact or documented: {name}"
            raise ValueError(msg)


def _check_requirement_source(requirement: Requirement, package: Mapping[str, Any]) -> None:
    if requirement.url is None:
        return
    if requirement.url.startswith("git+"):
        url, separator, commit = requirement.url.removeprefix("git+").rpartition("@")
        vcs = package.get("vcs", {})
        if not separator or not _GIT_SHA.fullmatch(commit) or vcs.get("url") != url or vcs.get("commit-id") != commit:
            msg = f"bundle declared VCS source differs: {requirement.name}"
            raise ValueError(msg)
    elif requirement.url != package.get("archive", {}).get("url"):
        msg = f"bundle declared archive source differs: {requirement.name}"
        raise ValueError(msg)


def check_lock(
    lock_dir: Path, bundles: Path, root_lock: Path, target: Target, *, compiler: Path | None = None
) -> dict[str, Any]:
    """Bind a saved closure to current inputs, target, and reviewed exceptions."""
    folder = lock_dir / target.name
    if not folder.is_dir():
        msg = f"unsupported bundle lock target: {target.name}"
        raise ValueError(msg)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    exception_file = json.loads((lock_dir / "exceptions.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or exception_file.get("schema_version") != SCHEMA_VERSION:
        msg = "unsupported bundle manifest schema"
        raise ValueError(msg)
    if manifest.get("target") != target.as_dict() or manifest.get("uv_version") != UV_VERSION:
        msg = "bundle manifest target or installer differs"
        raise ValueError(msg)
    if manifest.get("policy_sha256") != policy_digest(compiler):
        msg = "bundle compilation or verification policy changed; recompile and review"
        raise ValueError(msg)
    requirements = bundle_requirements(bundles, target)
    expected_inputs = {"uv_lock_sha256": digest(root_lock), "requirements_sha256": canonical_digest(requirements)}
    exceptions = exception_file["targets"].get(target.name, {})
    if manifest.get("inputs") != expected_inputs or manifest.get("exceptions_sha256") != canonical_digest(exceptions):
        msg = "bundle lock inputs changed; recompile and review the complete closure"
        raise ValueError(msg)
    lock = folder / "pylock.toml"
    if manifest.get("lock_sha256") != digest(lock):
        msg = "bundle lock hash differs"
        raise ValueError(msg)
    packages = read_packages(lock, target)
    validate_exceptions(packages, root_lock, exceptions)
    for line in requirements:
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate(target.marker_environment()):
            continue
        name = canonicalize_name(requirement.name)
        if name not in packages or not requirement.specifier.contains(packages[name]["version"], prereleases=True):
            msg = f"bundle declaration is not satisfied: {name}"
            raise ValueError(msg)
        _check_requirement_source(requirement, packages[name])
    return {"manifest": manifest, "packages": packages, "requirements": requirements}


def _check_direct_source(package: Mapping[str, Any], direct: Mapping[str, Any] | None) -> None:
    name = package["name"]
    if "vcs" in package:
        vcs = package["vcs"]
        if (
            not direct
            or direct.get("url") != vcs["url"]
            or direct.get("vcs_info", {}).get("commit_id") != vcs["commit-id"]
        ):
            msg = f"installed VCS source differs: {name}"
            raise ValueError(msg)
    elif "archive" in package:
        archive = package["archive"]
        if not direct or direct.get("url") != archive["url"]:
            msg = f"installed archive source differs: {name}"
            raise ValueError(msg)
        _check_recorded_hash(direct, archive["hashes"]["sha256"], name)
    elif direct is not None:
        # uv may preserve a direct wheel URL when installing PEP751 locks.
        allowed = {row["url"]: row.get("sha256") for row in artifact_sources(package)}
        if direct.get("url") not in allowed:
            msg = f"installed artifact source differs: {name}"
            raise ValueError(msg)
        _check_recorded_hash(direct, allowed[direct["url"]], name)


def _check_recorded_hash(direct: Mapping[str, Any], expected: str | None, name: str) -> None:
    # uv0.9.28 checks the PEP751 hash during installation but writes an empty
    # archive_info in PEP610 metadata. Check a hash when recorded; do not claim
    # installed metadata independently proves the downloaded artifact's bytes.
    info = direct.get("archive_info", {})
    hashes = info.get("hashes", {})
    if (hashes and hashes.get("sha256") != expected) or (info.get("hash") and info["hash"] != f"sha256={expected}"):
        msg = f"installed recorded artifact hash differs: {name}"
        raise ValueError(msg)


def _validate_dependencies(
    distributions: Mapping[str, importlib.metadata.Distribution],
    packages: Mapping[str, Mapping[str, Any]],
    requirements: Sequence[str],
    target: Target,
) -> None:
    extras: dict[str, set[str]] = {name: {""} for name in distributions}
    for line in requirements:
        req = Requirement(line)
        if req.marker is None or req.marker.evaluate(target.marker_environment()):
            extras[canonicalize_name(req.name)].update(req.extras)
    visited = set()
    while pending := [
        (name, extra) for name, values in extras.items() for extra in values if (name, extra) not in visited
    ]:
        for name, extra in pending:
            visited.add((name, extra))
            # The saved target fixes the ABI. A patch-specific installed
            # requirement must still be checked against this actual Python.
            marker = target.marker_environment() | {
                "extra": extra,
                "python_full_version": host_platform.python_version(),
                "implementation_version": host_platform.python_version(),
            }
            for line in distributions[name].requires or []:
                req = Requirement(line)
                if req.marker is not None and not req.marker.evaluate(marker):
                    continue
                dependency = canonicalize_name(req.name)
                if dependency not in distributions or not req.specifier.contains(
                    distributions[dependency].version, prereleases=True
                ):
                    msg = f"installed dependency is not satisfied: {name} requires {dependency}"
                    raise ValueError(msg)
                extras[dependency].update(req.extras)
                _check_requirement_source(req, packages[dependency])


def verify_inventory(
    checked: Mapping[str, Any],
    installed: Path,
    target: Target,
) -> dict[str, Any]:
    """Inspect installed metadata, never import the distribution's code."""
    packages = checked["packages"]
    distributions = {}
    rows = []
    for dist in importlib.metadata.distributions(path=[str(installed)]):
        name = canonicalize_name(dist.metadata["Name"])
        if name in distributions:
            msg = f"duplicate installed distribution: {name}"
            raise ValueError(msg)
        distributions[name] = dist
    if set(distributions) != set(packages):
        msg = f"installed inventory differs: missing={sorted(set(packages) - set(distributions))}, unexpected={sorted(set(distributions) - set(packages))}"
        raise ValueError(msg)
    for name, dist in sorted(distributions.items()):
        if Version(dist.version) != Version(packages[name]["version"]):
            msg = f"installed version differs: {name}"
            raise ValueError(msg)
        text = dist.read_text("direct_url.json")
        direct = json.loads(text) if text else None
        _check_direct_source(packages[name], direct)
        rows.append({"name": name, "version": dist.version, "direct_url": direct})
    _validate_dependencies(distributions, packages, checked["requirements"], target)
    return {
        "schema_version": SCHEMA_VERSION,
        "target": target.as_dict(),
        "lock_sha256": checked["manifest"]["lock_sha256"],
        "inputs": checked["manifest"]["inputs"],
        "exceptions_sha256": checked["manifest"]["exceptions_sha256"],
        "policy_sha256": checked["manifest"]["policy_sha256"],
        "uv_version": checked["manifest"]["uv_version"],
        "runtime_python_version": host_platform.python_version(),
        "source_check": "installer verifies locked artifact hashes; inventory checks versions and recorded direct sources",
        "packages": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Docker pre-install validation and post-install inventory receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "verify"))
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("--bundles-dir", type=Path, required=True)
    parser.add_argument("--root-lock", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--installed", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    target = Target(args.platform, args.bundle, args.architecture)
    checked = check_lock(args.lock_dir, args.bundles_dir, args.root_lock, target, compiler=args.compiler)
    if args.command == "verify":
        if (
            sys.platform != "linux"
            or host_platform.machine() != _MACHINES[target.architecture]
            or sys.version_info[:2] != (3, 12)
        ):
            parser.error("installed inventory verification must run on the locked Linux/Python/architecture target")
        if args.installed is None or args.receipt is None:
            parser.error("verify requires --installed and --receipt")
        receipt = verify_inventory(checked, args.installed, target)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
