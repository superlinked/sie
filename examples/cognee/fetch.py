"""Fetch the pinned public workshop inputs anonymously; standard library only."""

import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parent
DATASET = "superlinked/sie-task-evidence"
REVISION = "487cdc3bd0480659afd7ee88329449e6a6c8655d"
PREFIX = "cognee/workshop-starter/2026-11-06/v1"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PREFIX}"
MANIFEST_SHA256 = "81dbd18690c085a6f12e4a1772a082d92e102b9cff81616f53965e2d398c0ac8"
REQUIRED = {
    "readiness-policy.txt",
    "projector-check.txt",
    "microphone-check.txt",
    "correction.txt",
    "access-note.txt",
    "access-note.png",
}
MANIFEST_LIMIT = 64 * 1024
FILE_LIMIT = 1024 * 1024
TOTAL_LIMIT = 4 * 1024 * 1024


def bounded_read(stream: BinaryIO, limit: int) -> bytes:
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Download or cached file exceeds its byte limit.")
    return payload


def download(name: str, limit: int) -> bytes:
    request = urllib.request.Request(f"{BASE_URL}/{name}", headers={"User-Agent": "sie-cognee-example"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return bounded_read(response, limit)


def read_cached(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        return bounded_read(stream, limit)


def manifest_files(body: bytes) -> dict[str, tuple[int, str]]:
    if hashlib.sha256(body).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Manifest checksum differs from the pinned workshop inputs.")
    manifest = json.loads(body)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("license") != "CC0-1.0"
    ):
        raise ValueError("Unexpected workshop manifest schema or license.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED:
        raise ValueError("Manifest must contain exactly the six workshop input filenames.")
    checked = {}
    for name, item in files.items():
        if not isinstance(item, dict):
            raise ValueError("Invalid manifest file entry.")
        size, digest = item.get("bytes"), item.get("sha256")
        if type(size) is not int or not 0 < size <= FILE_LIMIT:
            raise ValueError("Invalid manifest file size.")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid manifest file checksum.")
        checked[name] = (size, digest)
    if sum(size for size, _ in checked.values()) > TOTAL_LIMIT:
        raise ValueError("Manifest exceeds the total input byte limit.")
    return checked


def check_file(name: str, body: bytes, size: int, digest: str) -> None:
    if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
        raise ValueError(f"{name}: input size or checksum differs from the pinned dataset.")


def fetch(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    body = (
        read_cached(manifest_path, MANIFEST_LIMIT)
        if manifest_path.exists()
        else download("manifest.json", MANIFEST_LIMIT)
    )
    files = manifest_files(body)
    missing = []
    for name, (size, digest) in files.items():
        target = directory / name
        if target.exists():
            check_file(name, read_cached(target, size), size, digest)
        else:
            missing.append(name)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=directory) as temporary:
        staging = Path(temporary)
        if not manifest_path.exists():
            (staging / "manifest.json").write_bytes(body)
        for name in sorted(missing):
            size, digest = files[name]
            payload = download(name, size)
            check_file(name, payload, size, digest)
            (staging / name).write_bytes(payload)
        # Promote only after every downloaded input has passed verification.
        for path in staging.iterdir():
            path.replace(directory / path.name)


def main() -> None:
    try:
        fetch(ROOT / "data")
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"Input fetch failed ({type(error).__name__}). Check your connection and available disk space. "
            "For a cache mismatch, move edited files out of data/ before retrying."
        ) from None
    print(f"Verified six workshop inputs from {DATASET}@{REVISION}. No API key was used.")


if __name__ == "__main__":
    main()
