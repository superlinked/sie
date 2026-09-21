#!/usr/bin/env python3
"""Download the recorded classify evidence from a pinned dataset revision.

    python3 fetch.py [--dest data]

Standard library only. No API key, no Hugging Face token, no inference spend.
The dataset is public and anonymous-readable.

REVISION is a commit SHA, never a branch. `main` moves; a SHA does not.

Every downloaded file is checked against a digest before the scorer sees it.
The chain is anchored in this file: MANIFEST_SHA256 pins manifest.json,
manifest.json pins calls.json and every input file. A file that is missing or
that fails its digest is a FAILURE, never a skip, and the script exits
nonzero without writing a partial tree the scorer could mistake for complete.

This script replaces `--dest` wholesale, so it refuses to touch anything it did
not write. A directory qualifies only when it holds a `.sie-evidence` marker naming
this dataset and task, and never when it is the working directory, an ancestor
of it, your home directory or the filesystem root. Anything else stops the run
with an explanation rather than being deleted.

The replacement itself is two renames, not a delete and a copy: the old
directory is renamed aside, the verified download is renamed into place, and
only then is the old one removed. If the second rename fails the first is
undone, so a reader is never left with neither.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DATASET = "superlinked/sie-task-evidence"
TASK = "classify"
REVISION = "ca886a38ae7f2ba0356355ee012c8d73938cd334"
MANIFEST_SHA256 = "f7c6545a83dd6a12d0fad1cd30a68d6878bfa721e4d368be89dcd09542af9d1d"

BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{TASK}"
HTTP_OK = 200
MARKER_NAME = ".sie-evidence"


def download(relative_path: str) -> bytes:
    url = f"{BASE}/{relative_path}"
    request = urllib.request.Request(url, headers={"Accept": "*/*"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            if response.status != HTTP_OK:
                raise RuntimeError(f"{url} returned HTTP {response.status}")
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{url} returned HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{url} could not be reached: {error.reason}") from error


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def marker_bytes() -> bytes:
    """What marks a directory as this script's to replace."""
    return (
        json.dumps(
            {"written_by": "fetch.py", "dataset": DATASET, "task": TASK, "revision": REVISION},
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def refuse_reason(dest: Path) -> str | None:
    """Why `dest` must not be replaced, or None when replacing it is safe.

    Checked before anything is downloaded and again before the swap.
    """
    resolved = dest.resolve()
    cwd = Path.cwd().resolve()
    # These hold whatever the marker says. A marker can be planted; the
    # working tree still must not be removable by a --dest typo.
    if resolved == resolved.parent:
        return f"{resolved} is the filesystem root"
    if resolved == cwd:
        return f"{resolved} is the current working directory"
    if resolved in cwd.parents:
        return f"{resolved} contains the current working directory"
    if resolved == Path.home().resolve():
        return f"{resolved} is your home directory"

    if dest.is_symlink():
        return f"{dest} is a symlink"
    if not dest.exists():
        return None
    if not dest.is_dir():
        return f"{dest} exists and is not a directory"

    marker = dest / MARKER_NAME
    if not marker.is_file():
        return (
            f"{dest} exists but holds no {MARKER_NAME}, so this script did not write it. "
            f"Move it aside, or pass --dest somewhere else."
        )
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"{dest}/{MARKER_NAME} could not be read: {error}"
    if recorded.get("dataset") != DATASET or recorded.get("task") != TASK:
        return f"{dest}/{MARKER_NAME} names {recorded.get('dataset')}/{recorded.get('task')}, not {DATASET}/{TASK}"
    return None


def swap_into_place(staging: Path, dest: Path) -> None:
    """Put `staging` at `dest` without deleting anything first.

    Both live in the same parent, so each rename is atomic and cannot half
    happen. The previous directory is only removed once the new one is in
    place; if that fails, the previous one goes back.
    """
    previous = None
    if dest.exists():
        previous = dest.with_name(f"{dest.name}.previous-{os.getpid()}")
        if previous.exists():
            shutil.rmtree(previous)
        dest.replace(previous)
    try:
        staging.replace(dest)
    except OSError:
        if previous is not None:
            previous.replace(dest)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="data", help="output directory (default: data)")
    args = parser.parse_args()

    dest = Path(args.dest)
    reason = refuse_reason(dest)
    if reason is not None:
        print(f"refusing to replace {dest}: {reason}", file=sys.stderr)
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)

    manifest_bytes = download("manifest.json")
    if digest(manifest_bytes) != MANIFEST_SHA256:
        print(
            f"manifest.json digest is {digest(manifest_bytes)}, expected {MANIFEST_SHA256}",
            file=sys.stderr,
        )
        return 1
    manifest = json.loads(manifest_bytes)

    files = manifest["files_sha256"]
    # Staged beside the destination, so the swap below is a rename on one
    # filesystem rather than a copy that can fail half way.
    staging = Path(tempfile.mkdtemp(prefix=f".{TASK}-evidence-", dir=dest.parent))
    try:
        (staging / "manifest.json").write_bytes(manifest_bytes)
        for relative_path, expected in sorted(files.items()):
            payload = download(relative_path)
            actual = digest(payload)
            if actual != expected:
                print(f"{relative_path} digest is {actual}, expected {expected}", file=sys.stderr)
                return 1
            target = staging / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        calls_sha = digest((staging / "calls.json").read_bytes())
        if calls_sha != manifest["calls_sha256"]:
            print("calls.json does not match the digest in manifest.json", file=sys.stderr)
            return 1

        (staging / MARKER_NAME).write_bytes(marker_bytes())

        # Re-checked here: the first check ran before the download, and the
        # destination could have changed since.
        reason = refuse_reason(dest)
        if reason is not None:
            print(f"refusing to replace {dest}: {reason}", file=sys.stderr)
            return 1
        swap_into_place(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"{TASK}: {len(files) + 1} files verified into {args.dest}/ at revision {REVISION}")
    print(f"{manifest['call_count']} recorded calls in {args.dest}/calls.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
