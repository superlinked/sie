#!/usr/bin/env python3
"""Download this example's recorded run from the pinned dataset revision.

    python3 fetch.py              # results, per-image counts, serving costs (under 1 MB)
    python3 fetch.py --outputs    # also every arm's text for the 272 GNHK and CORD photos

Standard library only. No token, no account, no API key: the dataset is public and
this pulls it anonymously.

Everything lands in evidence/. score.py reads from there and never touches the network.
REVISION is a dataset commit, deliberately not `main`, so a later upload cannot change
what this example scores.
"""

from __future__ import annotations

import argparse
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
REVISION = "5258c09c87290a51730df44ff0b3aa3986296106"
TASK = "ocr-receipts-handwriting"

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"

# Every file score.py reads. A listing missing one is a failure, not a short download.
REQUIRED = ("manifest.json", "results.json", "per_image.json", "serving.json", "images.jsonl")


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes for a small file."""
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": f"sie-examples/{TASK}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def listing() -> list[dict]:
    entries = json.loads(get(f"{API}/{TASK}?recursive=true"))
    files = [entry for entry in entries if entry.get("type") == "file"]
    if not files:
        raise SystemExit(f"{DATASET} revision {REVISION} has no files under {TASK}/")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--outputs", action="store_true", help="also download every arm's text for the GNHK and CORD photos"
    )
    args = parser.parse_args()
    print(f"{DATASET} at {REVISION}")
    staging = Path(tempfile.mkdtemp(prefix="sie-evidence-", dir=HERE))
    try:
        written: set[str] = set()
        for entry in sorted(listing(), key=lambda item: item["path"]):
            remote = entry["path"]
            relative = remote[len(TASK) + 1 :]
            if relative.startswith("outputs/") and not args.outputs:
                continue
            body = get(f"{FILES}/{remote}")
            if entry.get("size") is not None and len(body) != entry["size"]:
                raise SystemExit(f"{remote}: downloaded {len(body)} bytes, the dataset lists {entry['size']}")
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
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            written.add(relative)
            print(f"  {relative} ({len(body)} bytes)")

        missing = [name for name in REQUIRED if name not in written]
        if missing:
            print("The download is incomplete; score.py would not be scoring the recorded run:", file=sys.stderr)
            for name in missing:
                print(f"  missing {name}", file=sys.stderr)
            return 1

        backup = EVIDENCE.with_name(EVIDENCE.name + ".old")
        shutil.rmtree(backup, ignore_errors=True)
        if EVIDENCE.exists():
            EVIDENCE.rename(backup)
        try:
            staging.rename(EVIDENCE)
        except OSError:
            if backup.exists():
                backup.rename(EVIDENCE)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"Wrote {EVIDENCE}")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
