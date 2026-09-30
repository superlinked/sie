#!/usr/bin/env python3
"""Embed the catalogue and the questions with SIE yourself, then score your vectors.

    uv run python fetch.py --photos
    SIE_API_KEY=... uv run python run.py                          # SIE Cloud
    uv run python run.py --base-url http://localhost:8080         # your own server:
                                                                  #   sie-server serve -m google/siglip-so400m-patch14-384
    uv run python score.py --vectors run-output/vectors

SigLIP puts a photo and a shopper's words in one 1,152-dimensional space, so
ranking is a cosine between what the photo calls and the question calls
return. There is no caption step and no reranker. Photos go as the recorded
1,024-pixel JPEG bytes, 8 to a request; questions go 64 to a request. The key
comes from SIE_API_KEY and is never written out.

You do not need to run this: the recorded vectors are already published, and
score.py checks them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
MODEL = "google/siglip-so400m-patch14-384"


def dense(results: list, sent: int) -> list[list[float]]:
    """One vector per item sent, in order; a short or long batch is a failure, not a misaligned file."""
    if len(results) != sent:
        raise SystemExit(f"sent {sent} items and got {len(results)} results back")
    return [list(np.asarray(item["dense"], dtype=np.float32)) for item in results]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://api.superlinked.com")
    parser.add_argument("--out", type=Path, default=HERE / "run-output" / "vectors")
    args = parser.parse_args()
    photos = EVIDENCE / "photos"
    if not photos.exists():
        print("No photos yet: run `python3 fetch.py --photos` first.", file=sys.stderr)
        return 1
    catalogue = json.loads((EVIDENCE / "inputs" / "catalogue.json").read_text())
    questions = json.loads((EVIDENCE / "inputs" / "questions.json").read_text())
    client = SIEClient(api_key=os.environ.get("SIE_API_KEY"), base_url=args.base_url)

    images: list[list[float]] = []
    for start in range(0, len(catalogue), 8):
        rows = catalogue[start : start + 8]
        items = [{"images": [(photos / f"{r['image_id']}.jpg").read_bytes()]} for r in rows]
        images += dense(client.encode(MODEL, items), len(items))
        print(f"  photos {start + len(rows)}/{len(catalogue)}", end="\r")
    texts: list[list[float]] = []
    for start in range(0, len(questions), 64):
        items = [{"text": q["text"]} for q in questions[start : start + 64]]
        texts += dense(client.encode(MODEL, items), len(items))

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "images.npy", np.asarray(images, dtype=np.float32))
    np.save(args.out / "texts.npy", np.asarray(texts, dtype=np.float32))
    (args.out / "images.ids.json").write_text(json.dumps([r["image_id"] for r in catalogue]))
    (args.out / "texts.ids.json").write_text(json.dumps([q["id"] for q in questions]))
    print(f"\nWrote {len(images)} photo and {len(texts)} question vectors to {args.out}")
    print(f"Now run: uv run python score.py --vectors {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
