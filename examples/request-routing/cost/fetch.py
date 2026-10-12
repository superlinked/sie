"""Download the cost-routing study's evidence from its pinned Hugging Face revision, and verify it.

    uv run --frozen python fetch.py [--output DIR]

MANIFEST.json must match the SHA-256 pinned in study.py, and every file must
match the size and SHA-256 that MANIFEST.json lists for it. Files are staged in
a temporary directory beside the destination, and the directory is moved into
place only once every file has verified. An existing destination is verified,
never overwritten. About 9 MB; no key or Hugging Face account is needed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import study

TIMEOUT_SECONDS = 300
MANIFEST_MAX_BYTES = 1024 * 1024

Download = Callable[[str, int], bytes]


def url(name: str) -> str:
    return f"https://huggingface.co/datasets/{study.DATASET}/resolve/{study.REVISION}/{study.FOLDER}/{name}"


def get(address: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(address, headers={"User-Agent": "sie-examples/request-routing-cost"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise study.EvidenceError(f"{address} is larger than its manifest size")
    return body


def fetch(destination: Path, download: Download = get) -> bool:
    """Download and verify into a new destination. Returns False if a verified copy is already there."""
    if destination.exists():
        study.verify_evidence(destination)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "evidence"
        manifest = download(url(study.MANIFEST), MANIFEST_MAX_BYTES)
        entries = study.manifest_entries(manifest)
        for name, entry in entries.items():
            body = download(url(name), entry[1])
            study.check_file(name, body, entry)
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        (staging / study.MANIFEST).write_bytes(manifest)
        study.verify_evidence(staging)
        staging.rename(destination)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=study.EVIDENCE, help="evidence directory (default: evidence/)")
    args = parser.parse_args()
    print(f"{study.DATASET} at {study.REVISION}\n  {study.FOLDER}")
    try:
        fresh = fetch(args.output)
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"Evidence download failed: {error}", file=sys.stderr)
        return 1
    count = f"{len(study.FILES)} files and {study.MANIFEST}"
    print(f"Verified {count}; saved to {args.output}" if fresh else f"{args.output} already holds the verified {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
