#!/usr/bin/env python3
"""Read your photo, or the reconstructed comparison photos, through your SIE server."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps
from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
MODEL = "zai-org/GLM-OCR"


def photo_bytes(path: Path) -> bytes:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(image.size) > 2048:
        scale = 2048 / max(image.size)
        image = image.resize(tuple(round(side * scale) for side in image.size), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="your receipt or handwriting photo; one model call")
    source.add_argument(
        "--recorded", action="store_true", help="the 96 hash-verified inputs reconstructed by prepare.py"
    )
    parser.add_argument("--base-url", default=os.environ.get("SIE_BASE_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "outputs.jsonl")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.out.exists():
        parser.error("the output file exists; choose a new --out path to avoid overwriting a run")
    if args.image:
        rows = [{"id": args.image.name, "data": photo_bytes(args.image)}]
    else:
        references = HERE / "runs" / "references.jsonl"
        rows = [json.loads(line) for line in references.read_text().splitlines()]
        if len(rows) != 96 or len({row["id"] for row in rows}) != 96:
            parser.error("prepare.py must reconstruct all 96 comparison inputs first")
        for row in rows:
            row["data"] = Path(row["image"]).read_bytes()
            if hashlib.sha256(row["data"]).hexdigest() != row["image_sha256"]:
                parser.error(f"{row['id']}: input bytes changed after reconstruction")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with SIEClient(base_url=args.base_url, api_key=os.environ.get("SIE_API_KEY"), timeout_s=900) as client:

        def read(row: dict) -> dict:
            started = time.monotonic()
            result = client.extract(MODEL, {"images": [row["data"]]}, wait_for_capacity=True)
            text = "\n".join(entity["text"] for entity in result.get("entities", []))
            if not text.strip():
                raise ValueError("The model returned an empty transcription.")
            return {
                "id": row["id"],
                "text": text,
                "seconds": time.monotonic() - started,
                "finish_reason": "not_exposed",
            }

        with args.out.open("w") as output, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(read, row): row["id"] for row in rows}
            for future in as_completed(futures):
                ident = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    failed += 1
                    result = {"id": ident, "text": "", "error": type(error).__name__}
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
    print(f"Saved {len(rows)} responses to {args.out}; {failed} failed. No Cloud endpoint is selected by default.")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
