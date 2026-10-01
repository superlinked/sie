#!/usr/bin/env python3
"""Download and verify the pinned public comparison without an API key."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
DATASET = "superlinked/sie-task-evidence"
REVISION = "b30b600797ecf332d1c4c01c0bc9c0bb7dd349b2"
TASK = "ocr-glm-receipts-handwriting"
MANIFEST_SHA256 = "1a5f2b61217206da687807d0a7e6d26fb944fe834c35fc0359798f87922832e2"
REQUIRED = {
    "README.md",
    "PREREGISTRATION.md",
    "results.json",
    "per_image.json",
    "samples.json",
    "mini_usage.json",
    "serving.json",
}


def get(name: str) -> bytes:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{TASK}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "sie-ocr-glm-example"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ocr-evidence-", dir=HERE) as temporary:
        staging = Path(temporary)
        body = get("manifest.json")
        if hashlib.sha256(body).hexdigest() != MANIFEST_SHA256:
            raise SystemExit("The public manifest differs from the pinned comparison.")
        manifest = json.loads(body)
        if set(manifest["files"]) != REQUIRED:
            raise SystemExit("The public comparison file set differs from the pinned comparison.")
        (staging / "manifest.json").write_bytes(body)
        for name in sorted(REQUIRED):
            payload = get(name)
            if hashlib.sha256(payload).hexdigest() != manifest["files"][name]:
                raise SystemExit(f"{name}: evidence checksum failed")
            (staging / name).write_bytes(payload)
        EVIDENCE.mkdir(exist_ok=True)
        for file in staging.iterdir():
            shutil.copyfile(file, EVIDENCE / file.name)
    print(f"Verified {DATASET}@{REVISION}/{TASK}; run uv run score.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
