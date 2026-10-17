"""Archive immutable B/N/R sources without installing or changing dependencies."""

# Fixed git subcommands, argv-only execution; refs resolve before archive reads.
# ruff: noqa: S603, S607

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest
from tools.benchmarks.cross_encoder_lora.protocol import BASE_REVISION, require

SOURCE_PATHS = (
    "packages/sie_server/src",
    "packages/sie_server/models",
    "packages/sie_sdk/src",
    "packages/sie_config/src",
    "packages/sie_bundles",
    "bundles",
    "uv.lock",
    "pyproject.toml",
)


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def prepare(root: Path, output: Path, candidate: str, baseline: str = BASE_REVISION) -> dict:
    # Resolve before collecting any bytes: moving symbolic refs are never
    # reused when archiving, hashing the lock, or writing the manifest.
    commits = {
        "B": git(root, "rev-parse", "--verify", "--end-of-options", f"{baseline}^{{commit}}").decode().strip(),
        "candidate": git(root, "rev-parse", "--verify", "--end-of-options", f"{candidate}^{{commit}}").decode().strip(),
    }
    require(commits["B"] == BASE_REVISION, "baseline differs from the frozen protocol")
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    locks = []
    for name, commit in commits.items():
        existing = [path for path in SOURCE_PATHS if git(root, "ls-tree", "--name-only", commit, path).strip()]
        archive = output / f"{name}.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "-C", str(root), "archive", "--format=tar", commit, *existing], stdout=stream, check=True
            )
        lock = git(root, "show", f"{commit}:uv.lock")
        locks.append(lock)
        records[name] = {"commit": commit, "archive": archive.name, "sha256": digest(archive), "paths": existing}
    require(locks[0] == locks[1], "baseline and candidate dependency locks differ")
    manifest = {"schema": 1, "sources": records, "dependency_lock_identical": True}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo, args.output, args.candidate), indent=2))


if __name__ == "__main__":
    main()
