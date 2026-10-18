"""Compile and check the complete, reviewed image-overlay dependency locks.

Run with the Docker-pinned uv version; compilation uses public indexes and an
explicit cutoff. Ordinary image builds consume the saved artifacts, without a
fresh dependency solve. No package is installed by this command.
"""
# ruff: noqa: INP001 - repository command modules use namespace packages

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from sie_server.bundle_lock import (
    SCHEMA_VERSION,
    UV_VERSION,
    Target,
    bundle_requirements,
    canonical_digest,
    check_lock,
    digest,
    policy_digest,
    read_packages,
    root_constraints,
    source_constraints,
    validate_exceptions,
    validate_source_pins,
)

ROOT = Path(__file__).resolve().parents[2]
LOCKS = Path("packages/sie_server/bundle-locks")
BUNDLES = Path("packages/sie_server/bundles")


def targets(root: Path) -> list[Target]:
    matrix = json.loads((root / ".github/release-matrix.json").read_text())
    result = [Target(platform, bundle, "amd64") for platform in matrix["platforms"] for bundle in matrix["bundles"]]
    result += [Target(item["platform"], item["bundle"], "amd64") for item in matrix["include"]]
    # Dockerfile.cpu documents arm64 as well as the amd64 release matrix.
    result += [Target("cpu", bundle, "arm64") for bundle in matrix["bundles"]]
    if len({target.name for target in result}) != len(result):
        raise ValueError("duplicate bundle lock target")
    return result


def index_arguments(target: Target) -> list[str]:
    arguments = ["--default-index", "https://pypi.org/simple"]
    if target.platform == "cpu":
        arguments += ["--index", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match"]
    elif target.platform == "cuda13":
        arguments += [
            "--index",
            "https://download.pytorch.org/whl/cu130",
            "--index",
            "https://docs.sglang.ai/whl/cu130/",
            "--index-strategy",
            "unsafe-best-match",
            "--prerelease",
            "allow",
        ]
    elif target.bundle in {"sglang", "sglang-embedding", "sglang-vision-extract"}:
        arguments += ["--index", "https://docs.sglang.ai/whl/cu129/", "--index-strategy", "unsafe-best-match"]
    return arguments


def resolver_environment() -> dict[str, str]:
    # Do not let an ambient private index, dependency override, or disabled hash
    # check change a public lock. The explicit cache argument remains local.
    return {key: value for key, value in os.environ.items() if not key.startswith(("UV_", "PIP_"))}


def check_all(root: Path) -> None:
    expected = targets(root)
    actual = {path.name for path in (root / LOCKS).iterdir() if path.is_dir()}
    if actual != {target.name for target in expected}:
        raise ValueError("saved bundle target set differs from release matrix plus CPU arm64")
    for target in expected:
        checked = check_lock(
            root / LOCKS, root / BUNDLES, root / "uv.lock", target, compiler=root / "tools/ci/bundle_lock.py"
        )
        if checked["manifest"].get("indexes") != index_arguments(target):
            raise ValueError(f"bundle index policy differs: {target.name}")


def complete_artifact_hashes(lock: Path) -> list[dict[str, object]]:
    """Hash small public artifacts omitted by uv0.9.28's PEP751 writer.

    Some PyTorch mirror entries lack index hashes even with --generate-hashes.
    Never accept those as unhashed installs: retrieve at most 64MiB per missing
    artifact during explicit lock compilation, or fail without saving a lock.
    """
    text = lock.read_text()
    completed = []
    cache: dict[str, tuple[str, int]] = {}
    for package in tomllib.loads(text)["packages"]:
        artifacts = package.get("wheels", []) + [package[key] for key in ("sdist", "archive") if key in package]
        for artifact in artifacts:
            if artifact.get("hashes", {}).get("sha256"):
                continue
            url = artifact["url"]
            parts = urlsplit(url)
            if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
                raise ValueError("missing-hash artifact must use credential-free HTTPS")
            if url not in cache:
                digestor = hashlib.sha256()
                size = 0
                with urlopen(url, timeout=60) as response:  # noqa: S310 - HTTPS validated above
                    if urlsplit(response.geturl()).scheme != "https":
                        raise ValueError("artifact redirect is not HTTPS")
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > 64 * 1024 * 1024:
                            raise ValueError("missing-hash artifact exceeds 64MiB compilation bound")
                        digestor.update(block)
                cache[url] = (digestor.hexdigest(), size)
            sha256, size = cache[url]
            # The Docker-pinned uv writer emits each artifact on one TOML line.
            candidates = [
                line for line in text.splitlines() if f"url = {json.dumps(url)}" in line and "hashes = {}" in line
            ]
            if len(candidates) != 1:
                raise ValueError("unexpected uv artifact serialization; refusing an ambiguous hash edit")
            line = candidates[0]
            text = text.replace(line, line.replace("hashes = {}", f'hashes = {{ sha256 = "{sha256}" }}'), 1)
            completed.append({"url": url, "sha256": sha256, "bytes": size})
    lock.write_text(text)
    return completed


def compile_target(root: Path, target: Target, uv: str, cutoff: str, cache: Path | None) -> None:
    policy = json.loads((root / LOCKS / "exceptions.json").read_text())
    exceptions = policy["targets"].get(target.name, {})
    source_pins = policy.get("source_pins", {}).get(target.name, {})
    requirements = bundle_requirements(root / BUNDLES, target)
    constraints = root_constraints(root / "uv.lock", exceptions) + source_constraints(source_pins)
    version = subprocess.run(  # noqa: S603 - explicit compiler executable, no shell
        [uv, "--version"], check=True, capture_output=True, text=True
    ).stdout.split()
    if version[:2] != ["uv", UV_VERSION]:
        raise ValueError(f"compile requires uv {UV_VERSION}, matching the Docker installer")
    with tempfile.TemporaryDirectory(prefix="sie-bundle-lock-") as temp:
        directory = Path(temp)
        (directory / "requirements.in").write_text("\n".join(requirements) + "\n")
        (directory / "constraints.txt").write_text("\n".join(constraints) + "\n")
        command = [
            uv,
            "pip",
            "compile",
            "requirements.in",
            "--constraint",
            "constraints.txt",
            "--no-config",
            "--python-platform",
            target.python_platform,
            "--python-version",
            "3.12",
            "--python",
            sys.executable,
            "--no-python-downloads",
            "--format",
            "pylock.toml",
            "--generate-hashes",
            "--output-file",
            "pylock.toml",
            "--no-header",
            "--exclude-newer",
            cutoff,
            *index_arguments(target),
        ]
        if cache is not None:
            command += ["--cache-dir", str(cache.resolve())]
        subprocess.run(  # noqa: S603 - fixed compile arguments and explicit public indexes
            command, check=True, cwd=directory, env=resolver_environment(), timeout=900
        )
        completed_hashes = complete_artifact_hashes(directory / "pylock.toml")
        packages = read_packages(directory / "pylock.toml", target)
        validate_exceptions(packages, root / "uv.lock", exceptions)
        validate_source_pins(packages, source_pins)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "target": target.as_dict(),
            "uv_version": UV_VERSION,
            "policy_sha256": policy_digest(root / "tools/ci/bundle_lock.py"),
            "exclude_newer": cutoff,
            "indexes": index_arguments(target),
            "completed_artifact_hashes": completed_hashes,
            "inputs": {
                "uv_lock_sha256": digest(root / "uv.lock"),
                "requirements_sha256": canonical_digest(requirements),
            },
            "exceptions_sha256": canonical_digest(exceptions),
            "source_pins_sha256": canonical_digest(source_pins),
            "lock_sha256": digest(directory / "pylock.toml"),
        }
        destination = root / LOCKS / target.name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / "pylock.toml", destination / "pylock.toml")
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    check_lock(root / LOCKS, root / BUNDLES, root / "uv.lock", target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "compile"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--target", action="append")
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--exclude-newer")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        check_all(args.root)
        return 0
    if not args.exclude_newer:
        parser.error("compile requires an explicit --exclude-newer cutoff")
    selected = targets(args.root)
    if args.target:
        unknown = set(args.target) - {target.name for target in selected}
        if unknown:
            parser.error(f"unsupported targets: {sorted(unknown)}")
        selected = [target for target in selected if target.name in args.target]
    for target in selected:
        compile_target(args.root, target, args.uv, args.exclude_newer, args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
