#!/usr/bin/env python3
"""Download the recorded run and the two public sets, each at a pinned revision.

    python3 fetch.py

Standard library only. No token, no account, no API key: every source is public
and this pulls it anonymously.

- evidence/  the study's recorded rows, its report and the manifest, from the
             superlinked/sie-task-evidence dataset, folder guardrails/
- sets/      ToxicChat's test file and Aegis 2.0's test file, the labels
             score.py scores against and the prompts run.py sends

Every revision is a commit, deliberately not `main`, so a later upload cannot
change what this example scores. The two set files are also checked against the
SHA-256 in study.py.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import study

HERE = study.HERE
EVIDENCE = study.EVIDENCE

DATASET = "superlinked/sie-task-evidence"
REVISION = "b45775c419e37212cef67f485610eda88ee29491"
TASK = "guardrails"

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"

# Every evidence file score.py reads. A listing missing one of these is a failure, not a short download.
REQUIRED = (
    "manifest.json",
    "results.json",
    "gold.json",
    *(
        f"rows/{set_key}__{stem}.jsonl"
        for stem, sets in {s.stem: s.sets for s in study.SYSTEMS}.items()
        for set_key in sets
    ),
)


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes for a non-LFS file."""
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def get(url: str) -> bytes:
    # No Authorization header: every source is public and this must work for a reader with no account.
    request = urllib.request.Request(url, headers={"User-Agent": f"sie-examples/{TASK}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def fetch_evidence() -> int:
    print(f"{DATASET} at {REVISION}")
    entries = [e for e in json.loads(get(f"{API}/{TASK}?recursive=true")) if e.get("type") == "file"]
    if not entries:
        raise SystemExit(f"{DATASET} revision {REVISION} has no files under {TASK}/")
    # Download into a staging directory and swap it in only once every required file has arrived,
    # so a failure never leaves a mix of two revisions in evidence/.
    staging = Path(tempfile.mkdtemp(prefix="sie-evidence-", dir=HERE))
    total = 0
    try:
        written: set[str] = set()
        for entry in sorted(entries, key=lambda item: item["path"]):
            remote = entry["path"]
            relative = remote[len(TASK) + 1 :]
            body = get(f"{FILES}/{remote}")
            if entry.get("size") is not None and len(body) != entry["size"]:
                raise SystemExit(f"{remote}: downloaded {len(body)} bytes, the dataset lists {entry['size']}")
            lfs_oid = (entry.get("lfs") or {}).get("oid")
            if lfs_oid:
                if hashlib.sha256(body).hexdigest() != lfs_oid:
                    raise SystemExit(f"{remote}: the bytes do not hash to the LFS digest the dataset lists")
            elif entry.get("oid"):
                if git_blob_oid(body) != entry["oid"]:
                    raise SystemExit(f"{remote}: the bytes do not match the object id the dataset lists")
            else:
                raise SystemExit(f"{remote}: the dataset lists no id for this file, so it cannot be checked")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            written.add(relative)
            total += len(body)
            print(f"  evidence/{relative} ({len(body)} bytes)")
        missing = [name for name in REQUIRED if name not in written]
        if missing:
            for name in missing:
                print(f"  missing {name}", file=sys.stderr)
            raise SystemExit("The download is incomplete; score.py would not be scoring the recorded run.")
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
    return total


def fetch_sets() -> int:
    total = 0
    for data_set in study.SETS:
        print(f"{data_set.repo} at {data_set.revision} ({data_set.licence})")
        body = get(data_set.url)
        if hashlib.sha256(body).hexdigest() != data_set.sha256:
            raise SystemExit(f"{data_set.path}: the bytes do not match the SHA-256 study.py pins")
        data_set.local.parent.mkdir(parents=True, exist_ok=True)
        partial = data_set.local.with_suffix(data_set.local.suffix + ".part")
        partial.write_bytes(body)
        partial.replace(data_set.local)
        total += len(body)
        print(f"  {data_set.local.relative_to(HERE)} ({len(body)} bytes)")
    return total


def main() -> int:
    total = fetch_evidence() + fetch_sets()
    print(f"Wrote {total} bytes")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
