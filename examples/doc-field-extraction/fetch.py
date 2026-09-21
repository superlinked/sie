#!/usr/bin/env python3
"""Download this task's recorded evidence from the pinned dataset revision.

    python3 fetch.py

Standard library only. No token, no account, no API key. The dataset is public
and this pulls it anonymously.

Everything lands in evidence/: the eight document images, the pre-registered
schemas and expected values, the recorded calls and the run manifest. score.py
reads from there and never touches the network.

REVISION is a dataset commit, deliberately not `main`, so a later upload cannot
change what this example scores.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

DATASET = "superlinked/sie-task-evidence"
REVISION = "2d733ecb8b270fbb154975776e7f5a4315330f36"
TASK = "doc-field-extraction"

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"

# score.py refuses to run without these. A missing one is a failure, not a skip.
REQUIRED = ("calls.json", "manifest.json", "inputs/inputs.json")


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes.

    SHA-1 is not chosen here for its strength; it is the identifier the dataset
    already exposes, so the value can be checked against HuggingFace by hand.
    """
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def get(url: str) -> bytes:
    # No Authorization header: the dataset is public and this must work for
    # a reader who has never signed in to HuggingFace.
    request = urllib.request.Request(url, headers={"User-Agent": f"sie-examples/{TASK}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def listing() -> list[dict]:
    entries = json.loads(get(f"{API}/{TASK}?recursive=true"))
    files = [entry for entry in entries if entry.get("type") == "file"]
    if not files:
        raise SystemExit(f"{DATASET} revision {REVISION} has no files under {TASK}/")
    return files


def main() -> int:
    print(f"{DATASET} at {REVISION}")
    total = 0
    written: set[str] = set()
    for entry in sorted(listing(), key=lambda item: item["path"]):
        remote = entry["path"]
        relative = remote[len(TASK) + 1 :]
        target = EVIDENCE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        body = get(f"{FILES}/{remote}")
        if entry.get("size") is not None and len(body) != entry["size"]:
            raise SystemExit(f"{remote}: downloaded {len(body)} bytes, the dataset lists {entry['size']}")
        # Every file is checked, by whichever id the dataset publishes for it.
        # Large files are stored with Git LFS and carry the SHA-256 of their
        # content; the rest are ordinary git blobs. Checking only the LFS ones
        # left calls.json, manifest.json and inputs.json verified by size alone.
        # A file the listing gives neither id for is a failure, not a skip.
        lfs_oid = (entry.get("lfs") or {}).get("oid")
        blob_oid = entry.get("oid")
        if lfs_oid:
            if hashlib.sha256(body).hexdigest() != lfs_oid:
                raise SystemExit(f"{remote}: the bytes do not hash to the LFS digest the dataset lists")
        elif blob_oid:
            if git_blob_oid(body) != blob_oid:
                raise SystemExit(f"{remote}: the bytes do not match the object id the dataset lists")
        else:
            raise SystemExit(f"{remote}: the dataset lists no id for this file, so it cannot be checked")
        target.write_bytes(body)
        written.add(relative)
        total += len(body)
        print(f"  {relative} ({len(body)} bytes)")

    missing = [name for name in REQUIRED if name not in written]
    if missing or not any(name.startswith("inputs/images/") for name in written):
        print("The download is incomplete; score.py would not be scoring the recorded run:", file=sys.stderr)
        for name in missing or ["inputs/images/*"]:
            print(f"  missing {name}", file=sys.stderr)
        return 1

    print(f"Wrote {total} bytes to {EVIDENCE}")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
