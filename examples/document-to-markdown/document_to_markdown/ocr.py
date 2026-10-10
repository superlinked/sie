from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from sie_sdk import SIEClient
from sie_sdk.types import Item

from document_to_markdown.config import ROOT

DEFAULT_MODEL = "lightonai/LightOnOCR-3-4B"


def convert_image(image: Path, *, sie_url: str, model: str = DEFAULT_MODEL, timeout_s: float = 360) -> str:
    image_bytes = image.read_bytes()
    client = SIEClient(sie_url, api_key=os.environ.get("SIE_API_KEY") or None, timeout_s=timeout_s)
    try:
        result = client.extract(
            model,
            Item(id=image.stem, images=[image_bytes]),
            options={"max_new_tokens": 4096, "temperature": 0.1, "top_p": 1.0},
        )
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        markdown = result.get("data", {}).get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise RuntimeError("Model returned no Markdown")
        return markdown
    finally:
        client.close()


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Convert a page image to Markdown through a compatible SIE URL")
    parser.add_argument("image", type=Path)
    parser.add_argument("--sie-url", default=os.environ.get("SIE_URL", "http://localhost:8080"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    # Reserve the output before dispatch so an existing result is never replaced.
    output = args.out.open("x", encoding="utf-8")
    try:
        with output:
            markdown = convert_image(args.image, sie_url=args.sie_url, model=args.model, timeout_s=args.timeout)
            output.write(markdown)
    except BaseException:
        args.out.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
