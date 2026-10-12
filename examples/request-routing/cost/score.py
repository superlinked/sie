"""Reproduce the cost-routing study's held-out table from its recorded vectors and outcomes.

    uv run --frozen python score.py [--evidence DIR]

No model calls and no key. For both pairs this recomputes every knn20 score
from the recorded bge-m3 vectors, re-chooses the threshold on the calibration
split, and recomputes the held-out cost per 1,000 requests (router included),
saving, quality ratio and cheap share from the recorded per-request quality
and list costs. The rules are those of the study's code/analyze.py: kNN in
`router_scores`, then `routed`, `choose_tau` and `summarize`. It exits non-zero
unless every figure matches results.json and the published headline.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import study

TARGET = 0.98  # routed calibration quality must keep 98% of always-strong quality

# The study README's held-out rows: always-strong cost, then the router's cost, saving, quality and cheap share.
PUBLISHED = {
    "openai": ("$5.19", "$3.19", "38.6%", "95.8%", "77.2%"),
    "anthropic": ("$10.36", "$9.15", "11.7%", "99.0%", "34.9%"),
}
FIELDS = ("cost_per_1k", "saving_vs_strong", "quality_vs_strong", "cheap_share")


def leave_one_out_scores(vectors: np.ndarray, y: np.ndarray, calibration: np.ndarray, k: int = study.K) -> np.ndarray:
    """Score every request against the calibration requests other than itself, as analyze.py does."""
    similarities = vectors @ vectors.T
    cal_idx = np.where(calibration)[0]
    scores = np.zeros(len(vectors))
    for i in range(len(vectors)):
        candidates = cal_idx[cal_idx != i]
        scores[i] = study.knn_score(similarities[i, candidates], y[candidates], k)
    return scores


def routed(
    o: study.Outcomes, score: np.ndarray, router_usd: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cheap = score >= tau
    quality = np.where(cheap, o.q_cheap, o.q_strong)
    cost = np.where(cheap, o.usd_cheap, o.usd_strong) + router_usd
    return quality, cost, cheap


def choose_tau(o: study.Outcomes, score: np.ndarray, router_usd: np.ndarray, mask: np.ndarray) -> float:
    """Cheapest calibration threshold that keeps 98% of always-strong quality; ties keep the higher one."""
    floor = TARGET * o.q_strong[mask].mean() - 1e-12
    best_cost, best_tau = np.inf, np.inf
    for tau in sorted(set(score[mask].tolist()) | {np.inf}, reverse=True):
        quality, cost, _ = routed(o, score, router_usd, tau)
        if quality[mask].mean() >= floor and cost[mask].mean() < best_cost - 1e-15:
            best_cost, best_tau = cost[mask].mean(), tau
    return best_tau


def summarize(
    o: study.Outcomes, quality: np.ndarray, cost: np.ndarray, cheap: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    return {
        "cost_per_1k": float(1000 * cost[mask].mean()),
        "saving_vs_strong": float(1 - cost[mask].mean() / o.usd_strong[mask].mean()),
        "quality_vs_strong": float(quality[mask].mean() / o.q_strong[mask].mean()),
        "cheap_share": float(np.mean(cheap[mask])),
    }


def strategies(o: study.Outcomes, score: np.ndarray, router_usd: np.ndarray, tau: float, held: np.ndarray) -> dict:
    """The held-out rows analyze.py reports for each pair, other than the LLM-as-router row."""
    n = len(score)
    rows = {
        "always_strong": summarize(o, o.q_strong, o.usd_strong, np.zeros(n, bool), held),
        "router": summarize(o, *routed(o, score, router_usd, tau), held),
        "always_cheap": summarize(o, o.q_cheap, o.usd_cheap, np.ones(n, bool), held),
    }
    # Random routing at the router's held-out cheap share, as an expected value over assignments.
    p = rows["router"]["cheap_share"]
    quality = p * o.q_cheap + (1 - p) * o.q_strong
    cost = p * o.usd_cheap + (1 - p) * o.usd_strong
    rows["random_same_share"] = summarize(o, quality, cost, np.zeros(n, bool), held) | {"cheap_share": p}
    return rows


def money(value: float) -> str:
    return f"${value:.2f}"


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def check_pair(s: study.Study, pair: str, failures: list[str]) -> dict[str, Any]:
    o = s.pairs[pair]
    recorded = s.results["pairs"][pair]
    calibration = s.results["calibration"][pair][study.ROUTER]
    held = ~s.calibration
    y = study.labels(o.q_cheap, o.q_strong)

    score = leave_one_out_scores(s.vectors, y, s.calibration)
    gap = float(np.max(np.abs(score - o.score)))
    if gap > 1e-12:
        failures.append(f"{pair}: knn20 scores differ from items-{pair}.jsonl by up to {gap:.3g}")
    router_usd = s.credits * study.SIE_USD_PER_CREDIT
    if not np.allclose(router_usd, o.usd_router, rtol=0, atol=1e-15):
        failures.append(f"{pair}: router cost from billed credits differs from items-{pair}.jsonl")

    tau = choose_tau(o, score, router_usd, s.calibration)
    recorded_tau = math.inf if calibration["tau"] is None else calibration["tau"]
    if not close(tau, recorded_tau):
        failures.append(f"{pair}: calibration threshold {tau!r}, results.json has {recorded_tau!r}")
    if [("cheap" if v else "strong") for v in score >= tau] != o.route:
        failures.append(f"{pair}: routes differ from items-{pair}.jsonl")

    # run.py's path: each held-out request against the calibration bank, one vector at a time.
    b = study.bank(s, pair)
    for i in np.where(held)[0]:
        decision = b.route(s.raw_vectors[i])
        if abs(decision["score"] - o.score[i]) > 1e-12 or decision["route"] != o.route[i]:
            failures.append(f"{pair}: run.py's bank routes {s.ids[i]} differently from the study")
            break

    rows = strategies(o, score, router_usd, tau, held)
    expected = {
        "always_strong": recorded["always_strong"],
        "router": recorded["routers"][study.ROUTER],
        "always_cheap": recorded["always_cheap"],
        "random_same_share": recorded["random_same_share"],
    }
    for name, row in rows.items():
        for field in FIELDS:
            if not close(row[field], expected[name][field]):
                failures.append(f"{pair} {name} {field}: {row[field]!r}, results.json has {expected[name][field]!r}")
    router_per_1k = float(1000 * router_usd[held].mean())
    if not close(router_per_1k, recorded["routers"][study.ROUTER]["router_cost_per_1k"]):
        failures.append(f"{pair}: router cost per 1,000 differs from results.json")

    r = rows["router"]
    shown = (
        money(rows["always_strong"]["cost_per_1k"]),
        money(r["cost_per_1k"]),
        percent(r["saving_vs_strong"]),
        percent(r["quality_vs_strong"]),
        percent(r["cheap_share"]),
    )
    if shown != PUBLISHED[pair]:
        failures.append(f"{pair}: headline {shown} differs from the published {PUBLISHED[pair]}")
    return {"rows": rows, "tau": tau, "router_per_1k": router_per_1k, "held": int(held.sum())}


def report(pair: str, result: dict[str, Any]) -> None:
    names = study.PAIRS[pair]
    rows = result["rows"]
    print(f"{names.cheap_name} (cheap) -> {names.strong_name} (strong), {result['held']} held-out requests")
    print(f"knn20 threshold {result['tau']:.4f}; router ${result['router_per_1k']:.3f} per 1,000, included\n")
    table = [
        (f"Always {names.strong_name}", rows["always_strong"]),
        ("SIE router (knn20)", rows["router"]),
        (f"Random at the same {percent(rows['router']['cheap_share'])}", rows["random_same_share"]),
        (f"Always {names.cheap_name}", rows["always_cheap"]),
    ]
    print(f"  {'Strategy':<26}{'per 1,000':>10}{'Saving':>9}{'Quality':>9}{'To cheap':>10}")
    for label, row in table:
        saving = "-" if label.startswith(f"Always {names.strong_name}") else percent(row["saving_vs_strong"])
        print(
            f"  {label:<26}{money(row['cost_per_1k']):>10}{saving:>9}"
            f"{percent(row['quality_vs_strong']):>9}{percent(row['cheap_share']):>10}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, default=study.EVIDENCE, help="directory written by fetch.py")
    args = parser.parse_args()
    try:
        s = study.load_study(args.evidence)
    except (OSError, ValueError) as error:
        print(f"Evidence check failed: {error}", file=sys.stderr)
        return 1
    failures: list[str] = []
    if s.results.get("selected_sie_router") != study.ROUTER:
        failures.append(f"results.json selected {s.results.get('selected_sie_router')!r}, not {study.ROUTER}")
    if s.results["n"] != {"calibration": int(s.calibration.sum()), "heldout": int((~s.calibration).sum())}:
        failures.append("calibration and held-out counts differ from results.json")
    results = {pair: check_pair(s, pair, failures) for pair in study.PAIRS}
    for pair, result in results.items():
        report(pair, result)
    print(
        "Quality is routed quality as a share of always-strong quality on the same requests.\n"
        "Cost is list price, with no caching or batch discount."
    )
    if failures:
        print("\nMISMATCH:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    n = len(s.ids)
    print(f"\nRecomputed {n:,} knn20 scores per pair and both thresholds. Every figure matches results.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
