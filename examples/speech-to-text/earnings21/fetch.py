"""Explicit setup: fetch pinned small evidence, optionally the 44 original MP3s."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from urllib.request import urlopen

from common import SOURCE, load_frame, verify_file


def copy_pinned(source, destination, expected_bytes: int) -> None:
    copied = 0
    for block in iter(lambda: source.read(1024 * 1024), b""):
        copied += len(block)
        if copied > expected_bytes:
            raise ValueError("Fetched file exceeds its pinned byte size")
        destination.write(block)


def fetch_file(url: str, destination: Path, pin: dict, local: Path | None = None) -> None:
    if destination.exists():
        verify_file(destination, pin)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if local is not None:
                with local.open("rb") as source:
                    copy_pinned(source, stream, pin["bytes"])
            else:
                with urlopen(url, timeout=60) as source:
                    copy_pinned(source, stream, pin["bytes"])
            stream.flush()
            verify_file(temporary, pin)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("data/evidence"))
    parser.add_argument("--from-dir", type=Path, help="Copy the exact pinned small evidence from a local directory")
    parser.add_argument("--audio-dir", type=Path, help="Explicitly fetch all 44 frozen MP3s here (about 560 MB)")
    args = parser.parse_args()
    for name, pin in SOURCE["files"].items():
        fetch_file(
            f"{SOURCE['raw_base']}/{SOURCE['manifest_prefix']}/{name}",
            args.evidence / name,
            pin,
            args.from_dir / name if args.from_dir is not None else None,
        )
    calls, _ = load_frame(args.evidence)
    print("Pinned small evidence verified: 44 calls, 219 pieces. No decoding or evaluation performed.")
    if args.audio_dir is not None:
        for call in calls:
            mp3 = call["mp3"]
            fetch_file(
                f"{SOURCE['raw_base']}/{SOURCE['audio_prefix']}/{mp3['file']}",
                args.audio_dir / mp3["file"],
                {"bytes": mp3["bytes"], "sha256": mp3["sha256"]},
            )
        print("All 44 original MP3 identities verified. Run prepare.py explicitly to decode them.")


if __name__ == "__main__":
    main()
