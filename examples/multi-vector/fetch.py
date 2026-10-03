#!/usr/bin/env python3
"""Download this task's recorded evidence from the pinned dataset revision.

    python3 fetch.py

Standard library only. No token, no account, no API key. The dataset is public
and this pulls it anonymously.

Everything lands in evidence/: the pinned inputs, the recorded calls and the
run manifest. score.py reads from there and never touches the network.

REVISION is a dataset commit, deliberately not `main`, so a later upload cannot
change what this example scores.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

DATASET = "superlinked/sie-task-evidence"
REVISION = "fc13484f89f9d03913a71ac3124195b98804d282"
TASK = "multi-vector"

API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
FILES = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"


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
    for entry in sorted(listing(), key=lambda item: item["path"]):
        remote = entry["path"]
        relative = remote[len(TASK) + 1 :]
        target = EVIDENCE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        body = get(f"{FILES}/{remote}")
        if entry.get("size") is not None and len(body) != entry["size"]:
            raise SystemExit(f"{remote}: downloaded {len(body)} bytes, the dataset lists {entry['size']}")
        target.write_bytes(body)
        total += len(body)
        print(f"  {relative} ({len(body)} bytes)")
    print(f"Wrote {total} bytes to {EVIDENCE}")
    print("Now run: python3 score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
