#!/usr/bin/env python3
"""Rank ViDoRe v3 pages yourself with SIE, then score the result beside the recorded run.

    uv run python run.py --smoke                         # computer_science, first 20 questions, 40 pages
    uv run python run.py --dataset computer_science      # one dataset, every page
    uv run python run.py                                 # all six datasets, 11,624 pages
    uv run python score.py --rankings run-output

It downloads each dataset from HuggingFace at the revision the recorded run used, renders every page the way the
recording did (JPEG quality 90, long side 1,650 pixels), encodes the pages and the English questions with
`TomoroAI/tomoro-colqwen3-embed-4b:compact`, ranks every page for every question by MaxSim and writes the top 100
to run-output/, in the recording's format.

By default it calls an SIE server on http://localhost:8080. Start one with the same model:

    pip install "sie-server[local]"
    sie-server serve --device cuda -m TomoroAI/tomoro-colqwen3-embed-4b:compact

Set SIE_BASE_URL and SIE_API_KEY to call SIE Cloud instead.

--smoke keeps the first 20 English questions of computer_science and only the pages graded for them: enough to check
the server, the model and the query flag in a minute. Its scores are not comparable with the full run, where every
page of the dataset is a candidate.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import Image as ImageFeature
from datasets import load_dataset
from PIL import Image
from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
MODEL = "TomoroAI/tomoro-colqwen3-embed-4b:compact"
ARM = "sie-tomoro-colqwen3-4b-768"
LONG_SIDE = 1650
BATCH = 8
TOP_K = 100


def render(original: bytes) -> bytes:
    """The recording's render: RGB, long side 1,650 pixels, JPEG quality 90."""
    with Image.open(io.BytesIO(original)) as im:
        im = im.convert("RGB")
        scale = LONG_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def maxsim(query: np.ndarray, pages: list[np.ndarray]) -> np.ndarray:
    """For each page, the sum over query vectors of the best dot product with any page vector."""
    return np.array([float((query @ page.T).max(axis=1).sum()) for page in pages])


def run(client: SIEClient, name: str, *, smoke: bool, out: Path) -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    source = manifest["datasets"][name]
    questions = json.loads((EVIDENCE / "questions" / f"{name}.json").read_text())
    qids = sorted(questions, key=int)
    if smoke:
        qids = qids[:20]
    wanted = {page for q in qids for page in questions[q]["qrels"]} if smoke else None
    corpus = load_dataset(source["dataset"], "corpus", split="test", revision=source["revision"])
    corpus = corpus.cast_column("image", ImageFeature(decode=False))
    page_ids: list[str] = []
    page_vectors: list[np.ndarray] = []
    batch_ids: list[str] = []
    batch_images: list[bytes] = []

    def flush() -> None:
        if not batch_images:
            return
        result = client.encode(MODEL, [{"images": [img]} for img in batch_images], output_types=["multivector"])
        page_vectors.extend(np.asarray(item["multivector"], dtype=np.float32) for item in result)
        page_ids.extend(batch_ids)
        batch_ids.clear()
        batch_images.clear()
        print(f"  {name}: {len(page_ids)} pages", file=sys.stderr)

    for row in corpus:
        cid = str(row["corpus_id"])
        if wanted is not None and cid not in wanted:
            continue
        batch_ids.append(cid)
        batch_images.append(render(row["image"]["bytes"]))
        if len(batch_images) == BATCH:
            flush()
    flush()
    ranking = {}
    for q in qids:
        vec = client.encode(MODEL, [{"text": questions[q]["text"]}], output_types=["multivector"], is_query=True)[0]
        scores = maxsim(np.asarray(vec["multivector"], dtype=np.float32), page_vectors)
        order = np.argsort(-scores)[:TOP_K]
        ranking[q] = [[page_ids[i], round(float(scores[i]), 6)] for i in order]
    target = out / ARM
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.json").write_text(json.dumps(ranking))
    print(f"wrote {target / f'{name}.json'} ({len(ranking)} questions, {len(page_ids)} pages)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "run-output")
    args = parser.parse_args()
    if not (EVIDENCE / "manifest.json").exists():
        print("No evidence/. Run: uv run python fetch.py", file=sys.stderr)
        return 1
    base_url = os.environ.get("SIE_BASE_URL", "http://localhost:8080")
    api_key = os.environ.get("SIE_API_KEY")
    client = SIEClient(base_url=base_url, api_key=api_key) if api_key else SIEClient(base_url=base_url)
    names = (
        ["computer_science"]
        if args.smoke
        else (args.dataset or list(json.loads((EVIDENCE / "manifest.json").read_text())["datasets"]))
    )
    for name in names:
        run(client, name, smoke=args.smoke, out=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
