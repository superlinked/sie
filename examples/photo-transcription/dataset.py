"""Immutable public source packet and file-integrity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

REVISION = "fd29e7a2813e90deec6ac27e1fcdc22a5e237c44"
PREFIX = "ocr-photo-transcription/2026-10-08/source-provider-v2"
BASE_URL = f"https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/{REVISION}/{PREFIX}/"
CHECKSUM_SHA256 = "1a10fe5a68bd9ef8da5e699cdd71caae3b1f84dceafe413b0350030640b8eaf2"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def packet_path(directory: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Packet paths must be relative and stay inside the packet")
    resolved = (directory / path).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError("Packet symlinks must stay inside the packet")
    return resolved


def checksums(data: bytes) -> dict[str, str]:
    if digest(data) != CHECKSUM_SHA256:
        raise ValueError("Checksum list differs from the pinned source packet")
    entries: dict[str, str] = {}
    for line in data.decode().splitlines():
        checksum, separator, relative = line.partition("  ")
        if not separator or len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise ValueError("Invalid SHA256SUMS entry")
        packet_path(Path.cwd(), relative)
        if relative in entries:
            raise ValueError("Duplicate checksum path")
        entries[relative] = checksum
    return entries


def verify_packet(directory: Path) -> None:
    for relative, checksum in checksums((directory / "SHA256SUMS").read_bytes()).items():
        if digest(packet_path(directory, relative).read_bytes()) != checksum:
            raise ValueError(f"Packet file changed: {relative}")
