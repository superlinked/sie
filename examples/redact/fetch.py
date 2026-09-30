#!/usr/bin/env python3
"""Download this task's recorded evidence from the pinned dataset revision.

    python3 fetch.py

Standard library only. No token, no account, no API key. The dataset is public
and this pulls it anonymously.

Everything lands in evidence/: the 660 documents with their gold spans, every
arm's recorded response for every document, and the study's results files.
score.py reads from there and never touches the network.

REVISION is a dataset commit, deliberately not `main`, so a later upload cannot
change what this example scores.

Every file is checked twice. First against the id the dataset lists for it at
that revision (the git object id, or the SHA-256 of a Git LFS file). Then
against the SHA-256 in manifest.json, whose own SHA-256 is pinned below. A file
that is missing, extra or fails either check stops the run, and evidence/ is
left as it was.
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
REVISION = "06b6b5cb7f6593bcbd1badd07dee1a78508533d1"
TASK = "redact"
MANIFEST_SHA256 = "a0b4002b9f742cdb6971e8c15df1ba3c969f99af99c2bdad263a721fed033b42"

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"

# Every file score.py reads. A listing missing one of these is a failure, not a
# short download: score.py would otherwise report on whatever happened to arrive.
REQUIRED = (
    "manifest.json",
    "inputs/gretel-main.jsonl",
    "rows/sie__urchade__gliner_multi_pii-v1.jsonl",
    "rows/sie__numind__NuNER_Zero.jsonl",
    "rows/presidio.jsonl",
    "rows/privacy-filter.jsonl",
    "rows/comprehend.jsonl",
    "rows/llm__gpt-6-luna.jsonl",
    "rows/llm__claude-haiku-4-5.jsonl",
    "results/gretel-main_results.json",
    "results/gretel-main_tokens.json",
    "results/e2_results.json",
)


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes.

    SHA-1 is not chosen here for its strength; it is the identifier the dataset
    already exposes, so the value can be checked against Hugging Face by hand.
    """
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def get(url: str) -> bytes:
    # No Authorization header: the dataset is public and this must work for a
    # reader who has never signed in to Hugging Face.
    request = urllib.request.Request(url, headers={"User-Agent": f"sie-examples/{TASK}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def listing() -> list[dict]:
    entries = json.loads(get(f"{API}/{TASK}?recursive=true"))
    files = [entry for entry in entries if entry.get("type") == "file"]
    if not files:
        raise SystemExit(f"{DATASET} revision {REVISION} has no files under {TASK}/")
    return files


def check_listed_id(remote: str, body: bytes, entry: dict) -> None:
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


def check_manifest(staging: Path, written: set[str]) -> None:
    manifest_bytes = (staging / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise SystemExit("manifest.json does not match the digest this script pins")
    listed = {f["path"]: f for f in json.loads(manifest_bytes)["files"]}
    extra = written - set(listed) - {"manifest.json"}
    if extra:
        raise SystemExit(f"files the manifest does not list: {', '.join(sorted(extra))}")
    for path, entry in sorted(listed.items()):
        if path not in written:
            raise SystemExit(f"{path}: listed in manifest.json but not downloaded")
        body = (staging / path).read_bytes()
        if len(body) != entry["bytes"] or hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise SystemExit(f"{path}: does not match the SHA-256 in manifest.json")


def main() -> int:
    print(f"{DATASET} at {REVISION}")
    # Download into a staging directory and swap it in only once every file has
    # arrived and checked, so a failure never leaves a mixed set behind.
    staging = Path(tempfile.mkdtemp(prefix="sie-evidence-", dir=HERE))
    try:
        total = 0
        written: set[str] = set()
        for entry in sorted(listing(), key=lambda item: item["path"]):
            remote = entry["path"]
            relative = remote[len(TASK) + 1 :]
            body = get(f"{FILES}/{remote}")
            check_listed_id(remote, body, entry)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            written.add(relative)
            total += len(body)
            print(f"  {relative} ({len(body):,} bytes)")

        missing = [name for name in REQUIRED if name not in written]
        if missing:
            print("The download is incomplete; score.py would not be scoring the recorded run:", file=sys.stderr)
            for name in missing:
                print(f"  missing {name}", file=sys.stderr)
            return 1
        check_manifest(staging, written)

        # Move the old evidence aside and delete it only once the new set is in
        # place, so a failed rename never leaves no evidence at all.
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

    print(f"Wrote {total:,} bytes to {EVIDENCE}, every file checked against the dataset and manifest.json")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
