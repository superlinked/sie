"""Render a local variant of the downloaded note without replacing the pinned image."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/access-note-rendered.png")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "data/access-note.png").resolve():
        parser.error("Choose an output outside data/access-note.png to preserve the pinned input.")
    try:
        text = (ROOT / "data/access-note.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        parser.error("Download the authored input first with uv run fetch.py.")
    image = Image.new("RGB", (1200, 480), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    draw.multiline_text((40, 35), text, fill="black", font=font, spacing=22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Rendered {args.output}")


if __name__ == "__main__":
    main()
