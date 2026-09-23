#!/usr/bin/env python3
"""Download this task's recorded evidence from the pinned dataset revision.

    python3 fetch.py

Standard library only. No token, no account, no API key. The dataset is public
and this pulls it anonymously.

Everything lands in evidence/: the pinned conversations, the sixty recorded
calls and the run manifest. score.py reads from there and never touches the
network.

REVISION is a dataset commit, deliberately not `main`, so a later upload cannot
change what this example scores.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

DATASET = "superlinked/sie-task-evidence"
REVISION = "56c2a2c57a78d4c18b6cb1d9ca48ae7879319de5"
TASK = "chat"

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"

# Every file score.py reads. A listing missing one of these is a failure, not a
# short download: score.py would otherwise report a clean pass over whatever
# happened to arrive.
REQUIRED = ("calls.json", "manifest.json", "inputs/conversations.json")


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes.

    SHA-1 is not chosen here for its strength; it is the identifier the dataset
    already exposes, so the value can be checked against HuggingFace by hand.
    """
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def get(url: str) -> bytes:
    # No Authorization header: the dataset is public and this must work for a
    # reader who has never signed in to HuggingFace.
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
    # Download into a staging directory and swap it in only once every required
    # file has arrived. Writing straight into evidence/ meant a listing missing
    # a file raised AFTER overwriting some of them, leaving a mixed set from two
    # revisions that score.py can accept whenever it happens to be internally
    # consistent.
    staging = Path(tempfile.mkdtemp(prefix="sie-evidence-", dir=HERE))
    try:
        total = 0
        written: set[str] = set()
        for entry in sorted(listing(), key=lambda item: item["path"]):
            remote = entry["path"]
            relative = remote[len(TASK) + 1 :]
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            body = get(f"{FILES}/{remote}")
            if entry.get("size") is not None and len(body) != entry["size"]:
                raise SystemExit(f"{remote}: downloaded {len(body)} bytes, the dataset lists {entry['size']}")
            # Every file is checked, by whichever id the dataset publishes for
            # it: the SHA-256 of the content for a Git LFS file, the git object
            # id for an ordinary blob. A file the listing gives neither id for
            # is a failure, not a skip.
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
        if missing:
            print(
                "The download is incomplete; score.py would not be scoring the recorded run:",
                file=sys.stderr,
            )
            for name in missing:
                print(f"  missing {name}", file=sys.stderr)
            return 1

        if EVIDENCE.exists():
            shutil.rmtree(EVIDENCE)
        staging.rename(EVIDENCE)
    finally:
        # A failure leaves the previous evidence/ untouched and removes the
        # half-downloaded staging directory rather than leaving it to be found.
        shutil.rmtree(staging, ignore_errors=True)

    print(f"Wrote {total} bytes to {EVIDENCE}")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
