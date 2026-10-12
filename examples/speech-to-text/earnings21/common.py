"""Pinned inputs and durable, credential-free evidence for the standalone caller."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SOURCE = json.loads((Path(__file__).parent / "source.json").read_text())
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
OPTIONS = {"language": "en", "max_new_tokens": 8192}


class PersistenceFailure(RuntimeError):
    """Evidence did not reach the required file and directory fsync boundary."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_new(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(path.parent)
    except Exception as exc:
        raise PersistenceFailure("Exclusive evidence write failed") from exc


def save_atomic(path: Path, raw: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    except Exception as exc:
        raise PersistenceFailure("Atomic evidence write failed") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    return value


def normalize_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme in {"http", "https"}
            and parts.hostname
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
            and "?" not in value
            and "#" not in value
            and not any(char.isspace() or ord(char) < 32 for char in value)
        )
        _ = parts.port  # Validate the port before recording anything.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("SIE URL must be HTTP(S), with a host and no credentials, query, or fragment")
    return value.rstrip("/")


def verify_file(path: Path, pin: dict) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != pin["bytes"]:
        raise ValueError("Pinned file is missing or has the wrong identity or length")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != pin["sha256"]:
        raise ValueError("Pinned file hash differs")


def frame_from_calls(calls: list[dict], ids: list[str], *, call_count: int = 44, piece_count: int = 219) -> list[dict]:
    if len(calls) != call_count or ids != sorted(set(ids)) or [call["id"] for call in calls] != ids:
        raise ValueError("The exact sorted original call frame is required")
    pieces = []
    for call in calls:
        cid = call["id"]
        if not re.fullmatch(r"[0-9]+", cid) or call["mp3"]["file"] != f"{cid}.mp3":
            raise ValueError("Invalid original call path")
        offset = 0
        for index, piece in enumerate(call["sie_pieces"]):
            start, end = piece["start_sample"], piece["end_sample"]
            if (
                type(start) is not int
                or type(end) is not int
                or start != offset
                or not start < end <= start + 11_520_000
                or piece["wav_bytes"] != 44 + 2 * (end - start)
                or piece["start_s"] != start / 16000
                or piece["end_s"] != end / 16000
            ):
                raise ValueError("Original piece boundaries or PCM16 size changed")
            pieces.append({"id": f"{cid}-{index}", "call_id": cid, "piece_index": index, **piece})
            offset = end
        if not call["sie_pieces"] or offset != call["decoded"]["samples"]:
            raise ValueError("Original call has missing or noncontiguous pieces")
    if len(pieces) != piece_count:
        raise ValueError("The exact original piece frame is required")
    return pieces


def load_frame(evidence: Path, *, source: dict | None = None) -> tuple[list[dict], list[dict]]:
    pins = SOURCE if source is None else source  # Fixture injection is only used by offline controls.
    for name in ("manifest.json", "calls.json", "all_ids.txt", "references.jsonl"):
        verify_file(evidence / name, pins["files"][name])
    manifest = json.loads((evidence / "manifest.json").read_bytes())
    declared = {file["path"]: file for file in manifest["files"]}
    for name, pin in pins["files"].items():
        if name != "manifest.json" and declared.get(name) != {"path": name, **pin}:
            raise ValueError("Evidence manifest differs from the source pins")
    calls = json.loads((evidence / "calls.json").read_bytes())
    ids = (evidence / "all_ids.txt").read_text().splitlines()
    refs = [json.loads(line) for line in (evidence / "references.jsonl").read_text().splitlines()]
    if [ref["id"] for ref in refs] != ids:
        raise ValueError("Exact reference call identities and order are required")
    for call, ref in zip(calls, refs, strict=True):
        if sha256(ref["text"].encode()) != call["reference"]["text_sha256"]:
            raise ValueError("An original reference hash differs")
    return calls, frame_from_calls(calls, ids, call_count=pins["calls"], piece_count=pins["pieces"])


def piece_path(directory: Path, piece: dict) -> Path:
    return directory / f"{piece['call_id']}-{piece['piece_index']}.wav"


def qualify_pieces(directory: Path, pieces: list[dict]) -> None:
    for piece in pieces:
        verify_file(piece_path(directory, piece), {"bytes": piece["wav_bytes"], "sha256": piece["wav_sha256"]})


def read_piece(directory: Path, piece: dict) -> bytes:
    path = piece_path(directory, piece)
    if path.is_symlink():
        raise ValueError("Piece path changed after qualification")
    raw = path.read_bytes()
    if len(raw) != piece["wav_bytes"] or sha256(raw) != piece["wav_sha256"]:
        raise ValueError("Piece bytes changed after qualification")
    return raw
