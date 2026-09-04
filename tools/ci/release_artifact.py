#!/usr/bin/env python3
"""Bind retained release outputs to their original source and Actions run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

MANIFEST = "provenance.json"


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(version: str, tag_name: str, source_revision: str, run_id: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None or tag_name != f"v{version}":
        raise ValueError("artifact requires an exact stable version/tag")
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("artifact requires a full source revision")
    if re.fullmatch(r"[1-9][0-9]*", str(run_id)) is None:
        raise ValueError("artifact requires the original Actions run ID")
    return {
        "schema": 1,
        "repository": "superlinked/sie",
        "version": version,
        "tag_name": tag_name,
        "source_revision": source_revision,
        "run_id": str(run_id),
    }


def _files(directory: Path) -> list[dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("artifact directory must be a real directory")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("artifact must contain only regular files")
        name = path.relative_to(directory).as_posix()
        if name != MANIFEST:
            files.append({"name": name, "sha256": file_digest(path), "size": path.stat().st_size})
    if not files:
        raise ValueError("artifact contains no payload files")
    return files


def create_manifest(
    directory: Path,
    *,
    kind: str,
    version: str,
    tag_name: str,
    source_revision: str,
    run_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        **_identity(version, tag_name, source_revision, run_id),
        "kind": kind,
        "files": _files(directory),
        "metadata": metadata or {},
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path = directory / MANIFEST
    if path.is_symlink() or (path.exists() and path.read_text() != encoded):
        raise ValueError("refusing to replace different artifact provenance")
    path.write_text(encoded)
    return manifest


def validate_manifest(
    directory: Path,
    *,
    version: str,
    tag_name: str,
    source_revision: str,
    run_id: str,
    kind: str | None = None,
) -> dict[str, Any]:
    path = directory / MANIFEST
    if path.is_symlink():
        raise ValueError("artifact provenance must not be a symlink")
    manifest = json.loads(path.read_text())
    for key, value in _identity(version, tag_name, source_revision, run_id).items():
        if manifest.get(key) != value:
            raise ValueError(f"artifact provenance mismatch: {key}")
    if kind is not None and manifest.get("kind") != kind:
        raise ValueError("artifact provenance mismatch: kind")
    if not isinstance(manifest.get("metadata"), dict) or not isinstance(manifest.get("kind"), str):
        raise ValueError("artifact metadata/kind is malformed")
    if manifest.get("files") != _files(directory):
        raise ValueError("artifact file set, size, or SHA256 mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stamp", "check"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--kind")
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag-name", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    kwargs = vars(args).copy()
    command = kwargs.pop("command")
    if command == "stamp" and not args.kind:
        parser.error("stamp requires --kind")
    operation = create_manifest if command == "stamp" else validate_manifest
    operation(**kwargs)


if __name__ == "__main__":
    main()
