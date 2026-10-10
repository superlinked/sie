"""Fetch the fixed public source and unchanged scorer; no inference or scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path, PurePosixPath

SOURCE = json.loads((Path(__file__).parent / "source.json").read_text())


def checked_bytes(raw: bytes, pin: dict) -> bytes:
    if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("Public source length or checksum differs from the pinned revision")
    return raw


def fetch(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    root = directory.resolve()
    base = f"https://huggingface.co/datasets/{SOURCE['dataset']}/resolve/{SOURCE['revision']}/{SOURCE['prefix']}/"
    for name, pin in SOURCE["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name or "\\" in name:
            raise ValueError("Source file must be a canonical relative path")
        path = directory / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError("Source destination escapes the data directory")
        if path.exists():
            checked_bytes(path.read_bytes(), pin)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(base + name, timeout=60) as response:
            raw = checked_bytes(response.read(pin["bytes"] + 1), pin)
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    print(f"Verified {len(SOURCE['files'])} source files at {SOURCE['revision']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data"))
    fetch(parser.parse_args().directory)
