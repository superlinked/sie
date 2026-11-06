"""Render the authored synthetic note; no inference or external data."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def main() -> None:
    image = Image.new("RGB", (1200, 480), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    text = (ROOT / "data/access-note.txt").read_text()
    draw.multiline_text((40, 35), text, fill="black", font=font, spacing=22)
    image.save(ROOT / "data/access-note.png")


if __name__ == "__main__":
    main()
