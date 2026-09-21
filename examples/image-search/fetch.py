#!/usr/bin/env python3
"""Download this task's recorded evidence from the pinned dataset revision.

    python3 fetch.py

Standard library only. No token, no account, no API key. The dataset is public
and this pulls it anonymously.

Everything lands in evidence/: the catalogue, every photograph, the requests,
the recorded calls and the run manifest. score.py reads from there and never
touches the network.

REVISION is a dataset commit, deliberately not `main`, so a later upload cannot
change what this example scores. The digest chain is anchored in this file:
MANIFEST_SHA256 pins manifest.json, and manifest.json pins calls.json and every
input file including each photograph.

A file that is missing or that fails its digest is a FAILURE, never a skip, and
the script exits nonzero rather than leaving a partial tree the scorer could
mistake for a complete one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DATASET = "superlinked/sie-task-evidence"
TASK = "image-search"
REVISION = "c28bc851802d1c8728d728c24a50e94a76299a04"
MANIFEST_SHA256 = "0b89072bd90eb1b903fd9730253b89083f2b3e0193eb5d7f309c8bfbf568d2f8"

BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{TASK}"
HTTP_OK = 200
MARKER_NAME = ".sie-evidence"


def download(relative_path: str) -> bytes:
    url = f"{BASE}/{relative_path}"
    # No Authorization header: the dataset is public and this must work for a
    # reader who has never signed in to HuggingFace.
    request = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": f"sie-examples/{TASK}"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
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
    return (
        json.dumps(
            {"written_by": "fetch.py", "dataset": DATASET, "task": TASK, "revision": REVISION},
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def backup_path(dest: Path) -> Path:
    """Where `dest` is moved aside to while the new tree is renamed into place."""
    return dest.with_name(f"{dest.name}.previous")


def refuse_reason(dest: Path) -> str | None:
    """Why `dest` must not be replaced, or None when replacing it is safe."""
    resolved = dest.resolve()
    cwd = Path.cwd().resolve()
    # These hold whatever the marker says. A marker can be planted; the working
    # tree still must not be removable by a --dest typo.
    if resolved == resolved.parent:
        return f"{resolved} is the filesystem root"
    if resolved == cwd:
        return f"{resolved} is the current working directory"
    if resolved in cwd.parents:
        return f"{resolved} contains the current working directory"
    if resolved == Path.home().resolve():
        return f"{resolved} is your home directory"

    # The backup path is one this script names, which is not the same as one it
    # owns. If something else already sits there, moving `dest` onto it destroys
    # that something. Checked here so the refusal is reported by the same path
    # as every other refusal, before anything is downloaded.
    backup = backup_path(dest)
    if backup.exists():
        return (
            f"{backup} already exists, and this script would move {dest} onto it. "
            f"Move {backup} aside, or pass --dest somewhere else."
        )

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
    """Put `staging` at `dest` without deleting anything first."""
    previous = None
    if dest.exists():
        previous = backup_path(dest)
        if previous.exists():
            # refuse_reason checks this twice before we get here. Reaching it
            # anyway means something appeared in between, and deleting it is
            # never the right answer.
            raise RuntimeError(f"refusing to replace {dest}: backup path {previous} already exists")
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
    parser.add_argument("--dest", default="evidence", help="output directory (default: evidence)")
    args = parser.parse_args()

    dest = Path(args.dest)
    reason = refuse_reason(dest)
    if reason is not None:
        print(f"refusing to replace {dest}: {reason}", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"{DATASET} at {REVISION}")
    manifest_bytes = download("manifest.json")
    if digest(manifest_bytes) != MANIFEST_SHA256:
        print(f"manifest.json digest is {digest(manifest_bytes)}, expected {MANIFEST_SHA256}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_bytes)

    files = manifest["files_sha256"]
    staging = Path(tempfile.mkdtemp(prefix=f".{TASK}-evidence-", dir=dest.parent))
    total = 0
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
            total += len(payload)

        if digest((staging / "calls.json").read_bytes()) != manifest["calls_sha256"]:
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

    print(f"{TASK}: {len(files) + 1} files verified into {args.dest}/ ({total} bytes)")
    print(f"{manifest['calls_recorded']} recorded calls over {manifest['images']} photographs")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
