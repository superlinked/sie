"""Fetch the immutable, public document-field count evidence; no credentials."""
# ruff: noqa: INP001 - Standalone example scripts are not a Python package.

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

REVISION = "61e751a5f7f374cacd84ccace621f71e8a498404"
PREFIX = "doc-field-extraction-omni/2026-10-01"
BASE = f"https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/{REVISION}/{PREFIX}"
MANIFEST_SHA256 = "bc4dbde435a57bb243e664ab15fd626e17a7179a177bdb448221bef257c96f95"
FILES = frozenset({"README.md", "summary.json", "counts.jsonl", "inputs.jsonl"})
HERE = Path(__file__).resolve().parent


def read_public_file(name: str, max_bytes: int) -> bytes:
    with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as response:  # noqa: S310 - Pinned HTTPS origin.
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Evidence exceeds its declared size: {name}")
    return data


def load_manifest(data: bytes) -> dict:
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Evidence manifest hash mismatch")
    manifest = json.loads(data)
    if set(manifest["files"]) != FILES:
        raise ValueError("Evidence manifest has an unexpected file set")
    return manifest


def verify_file(name: str, data: bytes, expected: dict) -> None:
    if name not in FILES:
        raise ValueError(f"Unexpected evidence file: {name}")
    if len(data) != expected["bytes"]:
        raise ValueError(f"Evidence size mismatch: {name}")
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError(f"Evidence hash mismatch: {name}")


def fetch(output: Path) -> None:
    manifest_bytes = read_public_file("manifest.json", 64 * 1024)
    manifest = load_manifest(manifest_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Verify the entire download before replacing any existing local evidence.
    with tempfile.TemporaryDirectory(prefix="dfe-evidence-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name, expected in manifest["files"].items():
            data = read_public_file(name, expected["bytes"])
            verify_file(name, data, expected)
            (staging / name).write_bytes(data)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        output.mkdir(parents=True, exist_ok=True)
        for name in sorted(FILES | {"manifest.json"}):
            (staging / name).replace(output / name)
    print(json.dumps({"revision": REVISION, "files": len(FILES) + 1, "verified": True}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "data")
    args = parser.parse_args()
    fetch(args.output)


if __name__ == "__main__":
    main()
