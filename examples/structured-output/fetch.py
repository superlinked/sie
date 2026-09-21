#!/usr/bin/env python3
"""Download the recorded structured-output evidence from a pinned dataset revision.

    python3 fetch.py [--dest data]

Standard library only. No API key, no Hugging Face token, no inference spend.
The dataset is public and anonymous-readable.

REVISION is a commit SHA, never a branch. `main` moves; a SHA does not.

Every downloaded file is checked against a digest before the scorer sees it.
The chain is anchored in this file: MANIFEST_SHA256 pins manifest.json,
manifest.json pins calls.json and every input file. A file that is missing or
that fails its digest is a FAILURE, never a skip, and the script exits
nonzero without writing a partial tree the scorer could mistake for complete.
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
TASK = "structured-output"
REVISION = "eca4ac89206255642cadc9530f8faad562bc2829"
MANIFEST_SHA256 = "10ca0e46fdb80f9ed904584894841c10271fc9020279e2ffb4074893b5c65fa7"

BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{TASK}"
HTTP_OK = 200


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data", help="output directory (default: data)")
    args = parser.parse_args()

    manifest_bytes = download("manifest.json")
    if digest(manifest_bytes) != MANIFEST_SHA256:
        print(
            f"manifest.json digest is {digest(manifest_bytes)}, expected {MANIFEST_SHA256}",
            file=sys.stderr,
        )
        return 1
    manifest = json.loads(manifest_bytes)

    files = manifest["files_sha256"]
    staging = Path(tempfile.mkdtemp(prefix=f"{TASK}-evidence-"))
    try:
        (staging / "manifest.json").write_bytes(manifest_bytes)
        for relative_path, expected in sorted(files.items()):
            payload = download(relative_path)
            actual = digest(payload)
            if actual != expected:
                print(
                    f"{relative_path} digest is {actual}, expected {expected}",
                    file=sys.stderr,
                )
                return 1
            target = staging / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        calls_sha = digest((staging / "calls.json").read_bytes())
        if calls_sha != manifest["calls_sha256"]:
            print("calls.json does not match the digest in manifest.json", file=sys.stderr)
            return 1

        dest = Path(args.dest)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(dest))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"{TASK}: {len(files) + 1} files verified into {args.dest}/ at revision {REVISION}")
    print(f"{manifest['call_count']} recorded calls in {args.dest}/calls.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
