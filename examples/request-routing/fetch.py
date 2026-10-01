#!/usr/bin/env python3
"""Download recorded request-routing evidence anonymously at an explicit commit.

    python3 fetch.py --revision <40-hex-dataset-commit>

All ten files are downloaded and verified in same-parent staging first. A new
destination is created exclusively, verified files are linked exclusively, and
manifest.json is linked last as the completion marker. An interrupted transfer
can leave an incomplete destination without a manifest; it must not be scored.
Existing destinations are never overwritten. Standard library only; no key or
Hugging Face account is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DATASET = "superlinked/sie-task-evidence"
FOLDER = "request-routing/20261001-primary3000-v1"
EVIDENCE = Path(__file__).resolve().parent / "evidence"
FILES = (
    "assets.json",
    "inputs.jsonl",
    "front-clinc150.npz",
    "front-banking77.npz",
    "front-massive.npz",
    "vectors.npz",
    "records.jsonl",
    "results.json",
    "prices.json",
    "METHOD.md",
)
TIMEOUT_SECONDS = 300
MANIFEST_MAX_BYTES = 1024 * 1024


def revision_arg(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise argparse.ArgumentTypeError("revision must be a full 40-hex dataset commit")
    return value.lower()


def get(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "sie-examples/request-routing"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError("download exceeds its allowed size")
    return body


def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("manifest contains a duplicate key")
        result[key] = value
    return result


def manifest_files(body: bytes) -> dict[str, tuple[str, int]]:
    manifest = json.loads(body, object_pairs_hook=unique_keys)
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("manifest schema_version must be 1")
    if manifest["schema_version"] != 1:
        raise ValueError("manifest schema_version must be 1")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or set(entries) != set(FILES):
        raise ValueError("manifest must list exactly the ten request-routing evidence files")
    checked: dict[str, tuple[str, int]] = {}
    for name in FILES:
        entry = entries[name]
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size"}:
            raise ValueError(f"{name}: manifest must provide sha256 and size")
        digest, size = entry["sha256"], entry["size"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{name}: invalid SHA-256 digest")
        if type(size) is not int or size < 0:
            raise ValueError(f"{name}: size must be a nonnegative integer")
        checked[name] = (digest, size)
    return checked


def fetch(revision: str, destination: Path) -> None:
    revision = revision_arg(revision)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination already exists; choose a new directory")
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{FOLDER}"
    with tempfile.TemporaryDirectory(prefix=".sie-evidence-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        manifest = get(f"{base}/manifest.json", MANIFEST_MAX_BYTES)
        entries = manifest_files(manifest)
        for name in FILES:
            digest, size = entries[name]
            body = get(f"{base}/{name}", size)
            if len(body) != size:
                raise ValueError(f"{name}: downloaded size differs from manifest")
            if hashlib.sha256(body).hexdigest() != digest:
                raise ValueError(f"{name}: downloaded SHA-256 differs from manifest")
            (staging / name).write_bytes(body)
        (staging / "manifest.json").write_bytes(manifest)

        destination.mkdir(exist_ok=False)
        try:
            for name in (*FILES, "manifest.json"):
                os.link(staging / name, destination / name)
        except OSError:
            # Do not delete a directory whose contents may now belong to another writer.
            raise OSError("evidence transfer failed; destination is incomplete without a completion manifest") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, type=revision_arg)
    parser.add_argument("--output", type=Path, default=EVIDENCE, help="new evidence directory (default: evidence/)")
    args = parser.parse_args()
    try:
        fetch(args.revision, args.output)
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"Evidence download failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified {len(FILES)} files and manifest.json in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
