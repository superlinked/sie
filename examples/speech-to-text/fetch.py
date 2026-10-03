#!/usr/bin/env python3
"""Download this task's recorded evidence from the pinned dataset revision.

    python3 fetch.py

Standard library only. No token, no account, no API key. The dataset is public
and this pulls it anonymously.

Everything lands in evidence/: the twelve audio clips, the human transcripts and key
terms registered before the run, the Whisper spelling map both counts depend
on, the recorded calls and the run manifest. score.py reads from there and never touches the network.

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
REVISION = "66133a1a0bc97229e67891ab539ecb06da9af2cd"
TASK = "speech-to-text"

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"

# score.py refuses to run without these. A missing one is a failure, not a skip.
REQUIRED = (
    "calls.json",
    "manifest.json",
    "inputs/inputs.json",
    "inputs/normalizer.json",
    "inputs/scoring_amendment.json",
)


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
        # Large files are stored with Git LFS, and the listing then carries the
        # SHA-256 of their content. Check it here rather than only at score time.
        oid = (entry.get("lfs") or {}).get("oid")
        if oid and hashlib.sha256(body).hexdigest() != oid:
            raise SystemExit(f"{remote}: the bytes do not hash to the digest the dataset lists")
        target.write_bytes(body)
        written.add(relative)
        total += len(body)
        print(f"  {relative} ({len(body)} bytes)")

    missing = [name for name in REQUIRED if name not in written]
    if missing or not any(name.startswith("inputs/audio/") for name in written):
        print("The download is incomplete; score.py would not be scoring the recorded run:", file=sys.stderr)
        for name in missing or ["inputs/audio/*"]:
            print(f"  missing {name}", file=sys.stderr)
        return 1

    print(f"Wrote {total} bytes to {EVIDENCE}")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
