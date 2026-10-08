"""Download the pinned public photos, gold, protocol and saved-provider projection."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from dataset import BASE_URL, checksums, digest, packet_path, verify_packet


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE_URL + "SHA256SUMS", timeout=60) as response:
        checksum_bytes = response.read()
    entries = checksums(checksum_bytes)
    for relative, checksum in entries.items():
        target = packet_path(directory, relative)
        if target.exists() and digest(target.read_bytes()) == checksum:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE_URL + relative, timeout=60) as response:
            data = response.read()
        if digest(data) != checksum:
            raise ValueError(f"Downloaded file hash mismatch: {relative}")
        target.write_bytes(data)
    (directory / "SHA256SUMS").write_bytes(checksum_bytes)
    verify_packet(directory)
    print(f"Verified {len(entries)} files and the pinned checksum list in {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    download(parser.parse_args().directory)


if __name__ == "__main__":
    main()
