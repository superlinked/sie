"""Wrap the unchanged pinned piece builder; qualify all inputs before decoding."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from common import SOURCE, canonical, load_frame, qualify_pieces, save_new, verify_file


def prepare(evidence: Path, mp3_dir: Path, pieces_dir: Path) -> None:
    calls, pieces = load_frame(evidence)
    builder = evidence / "make_pieces.py"
    verify_file(builder, SOURCE["files"]["make_pieces.py"])
    for call in calls:
        mp3 = call["mp3"]
        verify_file(mp3_dir / f"{call['id']}.mp3", {"bytes": mp3["bytes"], "sha256": mp3["sha256"]})
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()
    pieces_dir.mkdir(parents=True, exist_ok=False)
    save_new(
        pieces_dir / "prepare.json",
        canonical(
            {
                "source_revision": SOURCE["revision"],
                "builder_sha256": SOURCE["files"]["make_pieces.py"]["sha256"],
                "python_version": platform.python_version(),
                "ffmpeg_version": version,
                "qualified": False,
            }
        ),
    )
    result = subprocess.run(
        [sys.executable, str(builder.resolve()), str(mp3_dir.resolve()), str(pieces_dir.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        # Do not relay the pinned builder's obsolete "use the published pieces" hint.
        reason = (
            "Decoded PCM hash mismatch" if "decoded PCM sha256 mismatch" in result.stderr else "Piece builder failed"
        )
        raise ValueError(f"{reason}. No verified derived-WAV archive is identified; use a qualifying decoder build.")
    qualify_pieces(pieces_dir, pieces)
    save_new(pieces_dir / "qualified.json", canonical({"calls": 44, "pieces": 219, "qualified": True}))
    print("All 219 WAV sizes and hashes verified; no evaluation performed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("data/evidence"))
    parser.add_argument("--mp3-dir", type=Path, default=Path("data/mp3"))
    parser.add_argument("--pieces-dir", type=Path, default=Path("data/pieces"))
    args = parser.parse_args()
    prepare(args.evidence, args.mp3_dir, args.pieces_dir)


if __name__ == "__main__":
    main()
