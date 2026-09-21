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
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

DATASET = "superlinked/sie-task-evidence"
REVISION = "1b6707ad110aaf2c8091c8585400e7b2a7153fa6"
TASK = "chat"

# Every file this example needs. A listing that is missing one of these is a
# failure, not a short download: score.py would otherwise report a clean pass
# over whatever happened to arrive.
REQUIRED = ("calls.json", "manifest.json", "inputs/cases.json")

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
            target.write_bytes(body)
            written.add(relative)
            total += len(body)
            print(f"  {relative} ({len(body)} bytes)")
        missing = [name for name in REQUIRED if name not in written]
        if missing:
            raise SystemExit(f"{DATASET} revision {REVISION} is missing {', '.join(missing)} under {TASK}/")
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
