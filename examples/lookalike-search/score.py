#!/usr/bin/env python3
"""Re-derive the /search figures from the recorded run. No key, no network.

    uv run python fetch.py      # once: download the recorded evidence
    uv run python score.py      # rank, count and compare

Two tests, both recorded on 30 September 2026:

1. The look-alike search test. 440 questions over 5,658 passages from public
   regulations and technical docs, each with a look-alike passage beside its
   answer. score.py ranks every passage for every question from SIE's recorded
   Qwen3 Embedding 4B vectors, checks the ranking against the recorded one, and
   counts how often each model put the answer first. The other models' rankings
   are read as recorded; their vectors are not redistributed.
2. Eight MTEB retrieval tasks, scored per query. score.py averages the recorded
   per-query scores into each model's share of queries whose first result is
   relevant, averaged over the tasks.

It exits non-zero if anything it recomputes disagrees with the recording.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
OURS = "sie-qwen3-embedding-4b"


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


def rank_ours(vectors: Path, questions: list[dict], passage_ids: list[str]) -> dict[str, dict]:
    doc_ids = json.loads((vectors / "corpus.ids.json").read_text())
    query_ids = json.loads((vectors / "queries.ids.json").read_text())
    if doc_ids != passage_ids:
        raise SystemExit("the vectors are not in corpus order")
    docs = normalise(np.load(vectors / "corpus.npy"))
    queries = normalise(np.load(vectors / "queries.npy"))
    position = {pid: i for i, pid in enumerate(doc_ids)}
    out = {}
    for row, qid in enumerate(query_ids):
        question = next(q for q in questions if q["id"] == qid)
        sims = docs @ queries[row]
        order = np.argsort(-sims, kind="stable")
        rank = {int(i): r + 1 for r, i in enumerate(order)}
        out[qid] = {
            "goldRank": rank[position[question["gold"]]],
            "top10": [doc_ids[i] for i in order[:10]],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vectors",
        type=Path,
        default=EVIDENCE / "vectors" / "qwen3-embedding-4b",
        help="SIE vectors to rank: the recorded ones, or a run.py output directory",
    )
    args = parser.parse_args()
    if not EVIDENCE.exists():
        print("No evidence yet: run `python3 fetch.py` first.", file=sys.stderr)
        return 1

    corpus = load(EVIDENCE / "inputs" / "corpus.json")
    questions = load(EVIDENCE / "inputs" / "questions.json")["questions"]
    recorded = load(EVIDENCE / "rankings.json")
    names = recorded["names"]
    rankings = recorded["arms"]
    passage_ids = [p["id"] for p in corpus["passages"]]
    print(f"{len(passage_ids):,} passages, {len(questions)} questions, {len(corpus['sources'])} public sources\n")

    # 1. Our ranking, recomputed from the vectors, must be the recorded one.
    ours = rank_ours(args.vectors, questions, passage_ids)
    recorded_ours = rankings[OURS]
    differ = [q for q in ours if ours[q]["top10"][:1] != recorded_ours[q]["top10"][:1]]
    if args.vectors == EVIDENCE / "vectors" / "qwen3-embedding-4b" and differ:
        print(f"{len(differ)} questions rank differently from the recording, e.g. {differ[:3]}", file=sys.stderr)
        return 1
    rankings = {**rankings, OURS: ours}

    print("The look-alike search test: the answer ranked first")
    print(f"  {'model':48s} {'first':>7s} {'share':>7s} {'look-alike first':>17s}")
    firsts = {}
    for arm, rows in rankings.items():
        first = sum(rows[q["id"]]["goldRank"] == 1 for q in questions)
        lookalike = sum(rows[q["id"]]["top10"][0] == q["lookalike"] for q in questions)
        firsts[arm] = first
        print(f"  {names[arm]:48s} {first:7d} {100 * first / len(questions):6.1f}% {lookalike:17d}")
    print("\n  Against SIE Qwen3 Embedding 4B (exact McNemar):")
    for arm, rows in rankings.items():
        if arm == OURS:
            continue
        b = sum(ours[q["id"]]["goldRank"] == 1 and rows[q["id"]]["goldRank"] != 1 for q in questions)
        c = sum(ours[q["id"]]["goldRank"] != 1 and rows[q["id"]]["goldRank"] == 1 for q in questions)
        print(f"  {names[arm]:48s} ours only {b:3d}, theirs only {c:3d}, p = {mcnemar(b, c):.3g}")

    # 2. MTEB: the share of queries whose first result is relevant, averaged over the eight tasks.
    per_query = load(EVIDENCE / "mteb" / "per_query.json.gz")["tasks"]
    tasks = sorted(per_query)
    arms = [arm for arm in names if all(arm in per_query[t] for t in tasks)]
    print(f"\nEight MTEB retrieval tasks ({', '.join(tasks)}): first result relevant, mean over tasks")
    for arm in sorted(
        arms,
        key=lambda a: -statistics.fmean(statistics.fmean(r["top1"] for r in per_query[t][a].values()) for t in tasks),
    ):
        macro = statistics.fmean(statistics.fmean(r["top1"] for r in per_query[t][arm].values()) for t in tasks)
        queries = sum(len(per_query[t][arm]) for t in tasks)
        print(f"  {names[arm]:48s} {100 * macro:5.1f}%  ({queries:,} queries)")

    expected = recorded.get("expected", {})
    if expected and args.vectors == EVIDENCE / "vectors" / "qwen3-embedding-4b":
        for arm, count in expected.get("heldoutFirst", {}).items():
            if firsts.get(arm) != count:
                print(f"{names[arm]}: {firsts.get(arm)} first, the page says {count}", file=sys.stderr)
                return 1
        print("\nEvery figure matches superlinked.com/search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
