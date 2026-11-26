"""Snapshot two Git revisions for a same-machine idle-worker comparison."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUTS = Path(__file__).parent / "_inputs"
PATHS = ("packages/sie_server/src", "packages/sie_server/models", "packages/sie_server/bundles")
GIT = shutil.which("git")


def git(*args: str) -> bytes:
    if GIT is None:
        raise RuntimeError("git is required")
    return subprocess.check_output([GIT, *args], cwd=ROOT)  # noqa: S603 — local Git revisions, no shell.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    INPUTS.mkdir(exist_ok=True)
    manifest = {}
    locks = []
    for arm, revision in (("baseline", args.baseline), ("candidate", args.candidate)):
        commit = git("rev-parse", f"{revision}^{{commit}}").decode().strip()
        archive = git("archive", "--format=tar", commit, *PATHS)
        (INPUTS / f"{arm}.tar").write_bytes(archive)
        lock = git("show", f"{commit}:uv.lock")
        locks.append(lock)
        manifest[arm] = {
            "commit": commit,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "uv_lock_sha256": hashlib.sha256(lock).hexdigest(),
        }
    if locks[0] != locks[1] or locks[1] != (ROOT / "uv.lock").read_bytes():
        raise SystemExit("Both revisions and the execution workspace must use the same uv.lock")
    manifest["source_diff"] = git(
        "diff", "--stat", manifest["baseline"]["commit"], manifest["candidate"]["commit"], "--", *PATHS
    ).decode()
    (INPUTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
