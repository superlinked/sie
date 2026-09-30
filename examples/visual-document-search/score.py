#!/usr/bin/env python3
"""Score the recorded visual document search run, or your own, from rankings and ViDoRe's relevance grades.

    uv run python score.py                                   # the recorded run, from evidence/
    uv run python score.py --rankings run-output             # add your own run from run.py

For every question it reads each arm's first ten pages and computes two figures:

- nDCG@10, with ViDoRe's graded relevance (1 or 2) as the gain: the benchmark's own metric;
- right page first: whether the first page has any positive grade.

Each figure is averaged per dataset, then over the six datasets, so every dataset counts equally. SIE's compact
profile is compared with every other arm by a paired bootstrap (questions resampled within each dataset, 10,000
draws, seed 20260930) and, on right page first, an exact McNemar test. The recorded figures in evidence/stats.json
are checked against what this script computes, and it exits non-zero if they differ.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
LEAD = "sie-tomoro-colqwen3-4b-768"
DATASETS = ("computer_science", "finance_en", "hr", "pharmaceuticals", "energy", "physics")
DRAWS = 10_000
SEED = 20260930


def ndcg10(ranked: list[str], grades: dict[str, int]) -> float:
    dcg = sum(grades.get(page, 0) / math.log2(i + 2) for i, page in enumerate(ranked[:10]))
    ideal = sorted(grades.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def per_question(rankings: Path, arm: str) -> dict[str, dict[str, np.ndarray]] | None:
    out = {}
    for name in DATASETS:
        path = rankings / arm / f"{name}.json"
        if not path.exists():
            return None
        ranking = json.loads(path.read_text())
        questions = json.loads((EVIDENCE / "questions" / f"{name}.json").read_text())
        ids = sorted(questions, key=int)
        nd, top = [], []
        for q in ids:
            grades = questions[q]["qrels"]
            pages = [page for page, _ in ranking[q]]
            nd.append(ndcg10(pages, grades))
            top.append(float(grades.get(pages[0], 0) > 0))
        out[name] = {"ndcg10": np.array(nd), "top1": np.array(top)}
    return out


def macro(scores: dict[str, dict[str, np.ndarray]], metric: str) -> float:
    return float(np.mean([scores[d][metric].mean() for d in DATASETS]))


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not (EVIDENCE / "manifest.json").exists():
        print("No evidence/. Run: uv run python fetch.py", file=sys.stderr)
        return 1
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    arms: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for arm in manifest["arms"]:
        scores = per_question(EVIDENCE / "rankings", arm)
        if scores is not None:
            arms[arm] = scores
    for extra in args.rankings:
        for arm_dir in sorted(p for p in extra.iterdir() if p.is_dir()):
            scores = per_question(extra, arm_dir.name)
            if scores is not None:
                arms[f"yours:{arm_dir.name}"] = scores
    n = sum(len(arms[LEAD][d]["ndcg10"]) for d in DATASETS)
    print(f"ViDoRe v3, {len(DATASETS)} datasets, {n} English questions, every page a candidate\n")
    print(f"{'arm':44s} {'nDCG@10':>8s} {'right first':>12s}")
    for arm, scores in arms.items():
        label = manifest["arms"].get(arm, {}).get("name", arm)
        print(f"{label[:44]:44s} {100 * macro(scores, 'ndcg10'):8.1f} {100 * macro(scores, 'top1'):11.1f}%")

    print(f"\nSIE compact minus each arm, 95% interval ({DRAWS:,} paired draws within each dataset)")
    rng = np.random.default_rng(SEED)
    samples = {
        d: [rng.integers(0, len(arms[LEAD][d]["ndcg10"]), len(arms[LEAD][d]["ndcg10"])) for _ in range(DRAWS)]
        for d in DATASETS
    }
    recorded = json.loads((EVIDENCE / "stats.json").read_text())["arms"]
    mismatches = 0
    for arm, scores in arms.items():
        if arm == LEAD:
            continue
        cells = []
        for metric in ("ndcg10", "top1"):
            diff = {d: arms[LEAD][d][metric] - scores[d][metric] for d in DATASETS}
            point = float(np.mean([diff[d].mean() for d in DATASETS]))
            boot = np.array([np.mean([diff[d][samples[d][i]].mean() for d in DATASETS]) for i in range(DRAWS)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cells.append(f"{100 * point:+5.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})")
        only_lead = sum(int((arms[LEAD][d]["top1"] > scores[d]["top1"]).sum()) for d in DATASETS)
        only_other = sum(int((scores[d]["top1"] > arms[LEAD][d]["top1"]).sum()) for d in DATASETS)
        label = manifest["arms"].get(arm, {}).get("name", arm)
        print(
            f"  {label[:42]:42s} nDCG@10 {cells[0]}  right first {cells[1]}  McNemar {only_lead}/{only_other} p={mcnemar(only_lead, only_other):.2g}"
        )
        if arm in recorded:
            for metric in ("ndcg10", "top1"):
                if abs(macro(scores, metric) - recorded[arm][f"{metric}_macro"]) > 1e-9:
                    mismatches += 1
                    print(f"    MISMATCH: {arm} {metric} differs from evidence/stats.json", file=sys.stderr)
    for metric in ("ndcg10", "top1"):
        if abs(macro(arms[LEAD], metric) - recorded[LEAD][f"{metric}_macro"]) > 1e-9:
            mismatches += 1
            print(f"MISMATCH: {LEAD} {metric} differs from evidence/stats.json", file=sys.stderr)

    print("\nPer dataset, nDCG@10:")
    for d in DATASETS:
        row = "  ".join(
            f"{manifest['arms'].get(a, {}).get('short', a)} {100 * s[d]['ndcg10'].mean():.1f}" for a, s in arms.items()
        )
        print(f"  {d:17s} {row}")
    if mismatches:
        return 1
    print("\nEvery recorded figure in evidence/stats.json matches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
