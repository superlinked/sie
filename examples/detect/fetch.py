#!/usr/bin/env python3
"""Fetch the immutable public RF100-VL evidence without sending model requests."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
DATASET = "superlinked/sie-task-evidence"
REVISION = "4dab11f3cd5933c4222d6f304a82221e2fc4a220"
MANIFEST_SHA256 = "4ea377268746f0b19cb29ab1407f4ccdeb120c77c9218db6a7a0ed7528a312b4"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/detect"


def download(name: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}/{urllib.parse.quote(name, safe='/')}", timeout=90) as response:
        return response.read()


def main() -> int:
    manifest_bytes = download("manifest.json")
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Manifest digest differs from the immutable example pin")
    manifest = json.loads(manifest_bytes)
    EVIDENCE.mkdir(exist_ok=True)
    for name, metadata in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {name}")
        target = EVIDENCE / relative
        data = target.read_bytes() if target.exists() else b""
        if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            data = download(name)
        if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Dataset file digest mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (EVIDENCE / "manifest.json").write_bytes(manifest_bytes)
    print(f"Verified {len(manifest['files'])} recorded files from {REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
