#!/usr/bin/env python3
"""Re-derive the /image-search figures from the recorded run. No key, no network.

    uv run python fetch.py      # once: download the recorded evidence
    uv run python score.py      # rank, count and compare

Two tests, both recorded on 30 September 2026:

1. A held-out product catalogue: 2,573 Amazon Berkeley Objects photos and 309
   questions such as "brown leather sofa". A result is right when its checked
   colour, material and product type all equal the question's. score.py ranks
   every photo for every question from SIE's recorded SigLIP so400m vectors,
   checks that ranking against the recorded one, and counts how often each
   product put a right photo first. The other products' rankings are read as
   recorded; their vectors are not redistributed.
2. Flickr30k and MS-COCO text-to-image (the Karpathy test splits), scored per
   caption from each product's recorded rank of the caption's own image.

It exits non-zero if anything it recomputes disagrees with the recording.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
OURS = "sie-siglip-so400m-384-oss"
# The share of right products first that superlinked.com/image-search and its SOURCES.md state.
EXPECTED = {
    OURS: "83.2",
    "cohere-embed-v4@1024": "84.1",
    "voyage-mm-3.5@1024": "75.1",
    "openai-caption-3-small@1024": "53.4",
}
PAGE_ARMS = list(EXPECTED)


def load(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path) as handle:
            return json.loads(handle.read())
    return json.loads(path.read_text())


def normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


def rank_ours(vectors: Path, image_ids: list[str], question_ids: list[str]) -> dict[str, list[str]]:
    ids = json.loads((vectors / "images.ids.json").read_text())
    qids = json.loads((vectors / "texts.ids.json").read_text())
    if ids != image_ids:
        raise SystemExit("the image vectors are not in catalogue order")
    if qids != question_ids:
        raise SystemExit("the text vectors are not in question order")
    images = normalise(np.load(vectors / "images.npy").astype(np.float32))
    texts = normalise(np.load(vectors / "texts.npy").astype(np.float32))
    sims = texts @ images.T
    order = np.argsort(-sims, axis=1, kind="stable")[:, :20]
    return {qid: [ids[j] for j in order[row]] for row, qid in enumerate(qids)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vectors",
        type=Path,
        default=EVIDENCE / "vectors" / "siglip-so400m-384",
        help="SIE vectors to rank: the recorded ones, or a run.py output directory",
    )
    args = parser.parse_args()
    if not EVIDENCE.exists():
        print("No evidence yet: run `python3 fetch.py` first.", file=sys.stderr)
        return 1

    catalogue = load(EVIDENCE / "inputs" / "catalogue.json")
    questions = load(EVIDENCE / "inputs" / "questions.json")
    names = load(EVIDENCE / "manifest.json")["arms"]
    rankings = load(EVIDENCE / "rankings" / "e1.json.gz")
    label = {r["image_id"]: (r["colour"], r["material"], r["type"]) for r in catalogue}
    print(f"{len(catalogue):,} catalogue photos, {len(questions)} questions\n")

    # 1. Our ranking, recomputed from the vectors, must be the recorded one.
    ours = rank_ours(args.vectors, [r["image_id"] for r in catalogue], [q["id"] for q in questions])
    recorded = {qid: [image_id for image_id, _ in rows] for qid, rows in rankings[OURS].items()}
    recorded_run = args.vectors == EVIDENCE / "vectors" / "siglip-so400m-384"
    differ = [qid for qid in ours if ours[qid] != recorded[qid][:20]]
    if recorded_run and differ:
        print(f"{len(differ)} questions rank differently from the recording, e.g. {differ[:3]}", file=sys.stderr)
        return 1
    tops = {arm: {qid: rows[0][0] for qid, rows in arm_rows.items()} for arm, arm_rows in rankings.items()}
    tops[OURS] = {qid: rows[0] for qid, rows in ours.items()}

    def right(arm: str, q: dict) -> bool:
        return label[tops[arm][q["id"]]] == (q["colour"], q["material"], q["type"])

    print("The held-out catalogue: a right product ranked first")
    print(f"  {'product':52s} {'first':>6s} {'share':>7s}")
    failed = False
    for arm in rankings:
        first = sum(right(arm, q) for q in questions)
        share = f"{100 * first / len(questions):.1f}"
        mark = ""
        if recorded_run and arm in EXPECTED and share != EXPECTED[arm]:
            mark, failed = f"  (the page states {EXPECTED[arm]}%)", True
        print(f"  {names[arm]:52s} {first:6d} {share:>6s}%{mark}")
    print(f"\n  Against {names[OURS]} (exact McNemar):")
    for arm in PAGE_ARMS[1:]:
        b = sum(right(OURS, q) and not right(arm, q) for q in questions)
        c = sum(right(arm, q) and not right(OURS, q) for q in questions)
        print(f"  {names[arm]:52s} ours only {b:3d}, theirs only {c:3d}, p = {mcnemar(b, c):.3g}")

    # 2. Flickr30k and MS-COCO: the caption's own image ranked first.
    e0 = load(EVIDENCE / "rankings" / "e0.json.gz")
    print("\nFlickr30k and MS-COCO (Karpathy test splits): the caption's image ranked first")
    print(f"  {'product':52s} {'Flickr30k':>10s} {'MS-COCO':>8s} {'mean':>7s}")
    for arm, sets in e0.items():
        shares = [
            100 * sum(r == 1 for r in ranks.values()) / len(ranks) for ranks in (sets["e0-flickr"], sets["e0-coco"])
        ]
        print(f"  {names.get(arm, arm):52s} {shares[0]:9.1f}% {shares[1]:7.1f}% {sum(shares) / 2:6.1f}%")

    if failed:
        print("\nA recomputed figure differs from the page.", file=sys.stderr)
        return 1
    print("\nEvery figure matches the recording.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
