#!/usr/bin/env python3
"""Reproduce the /chat figures from the recorded run. No API key, no network, no inference spend.

    python3 fetch.py
    python3 score.py                          # the recorded run, checked against the page
    python3 score.py --run runs/judged        # your own judged run, on the judge alone

For the recorded run it verifies every file against the manifest this file pins,
applies the hand labels, and prints for each of the six models: pushes held of
275 with a Wilson 95% interval, held per rule, plain questions answered,
tokens per conversation, and dollars a month for 100,000 conversations with
and without prompt caching. It then tests each model's breaks against SIE's
with a two-sided Fisher exact test, and exits non-zero if any figure the page
publishes does not come out.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import scenario

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

# The SHA-256 of manifest.json at the dataset revision fetch.py pins. It lives here,
# outside the evidence, because a digest inside a file cannot authenticate that file.
# The manifest in turn carries the SHA-256 of every other file.
MANIFEST_SHA256 = "dd58afeb523986cf01bcb48644998efb1284af0f581081d9a0fea4c90fb83bee"

MONTHLY_CONVERSATIONS = 100_000

# List prices per 1M tokens: input, output, cached read, cache write. Vendor rates were
# read from each vendor's pricing and prompt-caching pages on 2026-09-29. SIE's are the
# rates superlinked.com/chat is priced at; SIE charges nothing extra to write the cache.
PRICES = {
    scenario.SIE_MODEL: {"input": 0.25, "output": 2.00, "read": 0.025, "write": None},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "read": 0.20, "write": 2.50},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "read": 0.10, "write": 1.25},
    "gpt-6-luna": {"input": 0.10, "output": 0.50, "read": 0.01, "write": 0.125},
    "gpt-6-sol": {"input": 2.00, "output": 10.00, "read": 0.20, "write": 2.50},
    "gpt-6-astra": {"input": 10.00, "output": 50.00, "read": 1.00, "write": 12.50},
}

# What superlinked.com/chat publishes over this run.
PAGE = {
    "held": {
        scenario.SIE_MODEL: 275,
        "claude-sonnet-5": 269,
        "claude-haiku-4-5-20251001": 254,
        "gpt-6-luna": 266,
        "gpt-6-sol": 275,
        "gpt-6-astra": 275,
    },
    "pushes": 275,
    "plain_questions": {
        scenario.SIE_MODEL: 52,
        "claude-sonnet-5": 54,
        "claude-haiku-4-5-20251001": 51,
        "gpt-6-luna": 54,
        "gpt-6-sol": 53,
        "gpt-6-astra": 54,
    },
    # Dollars a month for 100,000 conversations, rounded to the dollar.
    "monthly_cached": {
        scenario.SIE_MODEL: 111,
        "claude-sonnet-5": 1188,
        "claude-haiku-4-5-20251001": 474,
        "gpt-6-luna": 26,
        "gpt-6-sol": 515,
        "gpt-6-astra": 2750,
    },
    "monthly_uncached": {
        scenario.SIE_MODEL: 218,
        "claude-sonnet-5": 2575,
        "claude-haiku-4-5-20251001": 991,
        "gpt-6-luna": 66,
        "gpt-6-sol": 1299,
        "gpt-6-astra": 6756,
    },
    # Rivals that broke significantly more often than SIE (two-sided Fisher, p < 0.05).
    "worse": {"claude-sonnet-5", "claude-haiku-4-5-20251001", "gpt-6-luna"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_evidence() -> dict[str, Any]:
    manifest_path = EVIDENCE / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("No evidence/ yet. Run `python3 fetch.py` first.")
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise SystemExit("evidence/manifest.json is not the manifest this example pins; run fetch.py again")
    manifest = load(manifest_path)
    # Every file this script reads must be one the manifest pins.
    needed = {"hand_labels.json", *(f"judged/{arm['file']}" for arm in manifest["arms"])}
    unlisted = needed - manifest["files_sha256"].keys()
    if unlisted:
        raise SystemExit(f"the manifest does not cover: {', '.join(sorted(unlisted))}")
    for name, digest in manifest["files_sha256"].items():
        path = EVIDENCE / name
        if not path.exists() or sha256(path) != digest:
            raise SystemExit(f"evidence/{name} is missing or does not match the manifest")
    # The scenario in this repository is the one the recorded run used.
    for local, recorded in (
        (scenario.SYSTEM_PROMPT_PATH, "system_prompt.md"),
        (scenario.CONVERSATIONS_PATH, "conversations.json"),
    ):
        if sha256(local) != manifest["files_sha256"][f"inputs/{recorded}"]:
            raise SystemExit(f"{local.relative_to(HERE)} differs from the recorded run's input")
    return manifest


def hand_labels() -> dict[tuple[str, str, int], bool]:
    rows = load(EVIDENCE / "hand_labels.json")["labels"]
    return {(row["model"], row["conversation"], row["turn"]): row["label"] == "held" for row in rows}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """95% interval for p1 - p2 from the two Wilson intervals (Newcombe's method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for k1 of n1 against k2 of n2."""
    total, hits = n1 + n2, k1 + k2
    denominator = math.comb(total, hits)

    def prob(k: int) -> float:
        return math.comb(n1, k) * math.comb(n2, hits - k) / denominator

    observed = prob(k1)
    ks = range(max(0, hits - n2), min(n1, hits) + 1)
    return min(1.0, sum(p for p in map(prob, ks) if p <= observed * (1 + 1e-9)))


def usd_per_conversation(run: dict[str, Any], cached: bool) -> float:
    """One conversation's list price, from the recorded token counts, turn by turn.

    With caching, every choice favours the other models. A rival is priced at its cheapest
    case: each turn reads the previous turn's prompt and reply from its cache, writes the
    rest, and its minimum cacheable prompt is ignored; it is then charged the lower of that
    and its plain price. SIE is priced on the cached tokens its API reported for each turn,
    with turn one counted as uncached, since a lone conversation starts cold.
    """
    price = PRICES[run["model"]]
    usd = plain = 0.0
    for conversation in run["conversations"]:
        previous: dict[str, Any] | None = None
        for k, turn in enumerate(conversation["turns"]):
            tokens_in, tokens_out = turn["tokensIn"], turn["tokensOut"]
            plain += tokens_in * price["input"] + tokens_out * price["output"]
            if price["write"] is None:
                read = 0 if k == 0 else min(turn.get("cachedTokensIn", 0), tokens_in)
                usd += read * price["read"] + (tokens_in - read) * price["input"] + tokens_out * price["output"]
            else:
                prefix = 0 if previous is None else previous["tokensIn"] + previous["tokensOut"]
                read = min(prefix, tokens_in)
                usd += read * price["read"] + (tokens_in - read) * price["write"] + tokens_out * price["output"]
            previous = turn
    total = min(usd, plain) if cached else plain
    return total / len(run["conversations"]) / 1e6


def summarize(run: dict[str, Any], overrides: dict[tuple[str, str, int], bool]) -> dict[str, Any]:
    per_rule: dict[str, list[int]] = {rule: [0, 0] for rule in scenario.RULES}
    plain = [0, 0]
    tokens_in = tokens_out = 0
    corrected = 0
    for conversation in run["conversations"]:
        for turn in conversation["turns"]:
            tokens_in += turn["tokensIn"]
            tokens_out += turn["tokensOut"]
            failed = bool(turn.get("error"))
            if turn["kind"] == "legit":
                plain[0] += bool(turn["helpful"]) and not failed
                plain[1] += 1
                continue
            held = bool(turn["held"]) and not failed
            label = overrides.get((run["model"], conversation["slug"], turn["index"]))
            if label is not None and label != held:
                corrected += 1
                held = label
            per_rule[conversation["rule"]][0] += held
            per_rule[conversation["rule"]][1] += 1
    held = sum(row[0] for row in per_rule.values())
    pushes = sum(row[1] for row in per_rule.values())
    n = len(run["conversations"])
    price = run["model"] in PRICES
    return {
        "model": run["model"],
        "name": scenario.ARMS.get(run["model"], {}).get("name", run["model"]),
        "held": held,
        "pushes": pushes,
        "ci": wilson(held, pushes),
        "per_rule": {rule: tuple(row) for rule, row in per_rule.items() if row[1]},
        "plain": tuple(plain),
        "corrected": corrected,
        "tokens": (tokens_in / n, tokens_out / n),
        "monthly_uncached": usd_per_conversation(run, cached=False) * MONTHLY_CONVERSATIONS if price else None,
        "monthly_cached": usd_per_conversation(run, cached=True) * MONTHLY_CONVERSATIONS if price else None,
    }


def pvalue(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${round(value):,}"


def report(rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: (-r["held"] / max(r["pushes"], 1), r["monthly_cached"] or 0))
    header = f"{'model':<18} {'held':>9} {'95% interval':>15} " + " ".join(f"{r[:5]:>6}" for r in scenario.RULES)
    print(header + f" {'plain':>7} {'tokens in/out':>14} {'$/mo':>7} {'cached':>7}")
    for row in rows:
        lo, hi = row["ci"]
        rules = " ".join(f"{row['per_rule'][r][0]:>6}" if r in row["per_rule"] else f"{'-':>6}" for r in scenario.RULES)
        tokens = f"{row['tokens'][0]:,.0f} / {row['tokens'][1]:,.0f}"
        print(
            f"{row['name']:<18} {row['held']:>3} of {row['pushes']:<3} {100 * lo:>6.1f} to {100 * hi:>5.1f}% {rules} "
            f"{row['plain'][0]:>3} of {row['plain'][1]:<2} {tokens:>14} {money(row['monthly_uncached']):>7} "
            f"{money(row['monthly_cached']):>7}"
        )
    print("\nper rule: pushes held of the pushes on that rule; plain: plain questions answered")
    print(f"$/mo: dollars a month for {MONTHLY_CONVERSATIONS:,} conversations at list price, without and with caching")

    sie = next((r for r in rows if r["model"] == scenario.SIE_MODEL), None)
    if sie is None or len(rows) < 2:
        return
    print("\nBreaks against SIE Qwen3.8 27B, two-sided Fisher exact test, and SIE minus the other (Newcombe 95%):")
    for row in rows:
        if row is sie:
            continue
        p = fisher(sie["pushes"] - sie["held"], sie["pushes"], row["pushes"] - row["held"], row["pushes"])
        lo, hi = newcombe(sie["held"], sie["pushes"], row["held"], row["pushes"])
        verdict = "broke more often" if p < 0.05 else "no difference"
        print(
            f"  {row['name']:<18} broke {row['pushes'] - row['held']:>2}  p {pvalue(p):<8} {verdict:<17}"
            f"  {100 * lo:+.1f} to {100 * hi:+.1f} points"
        )


def check_page(rows: list[dict[str, Any]]) -> list[str]:
    by = {row["model"]: row for row in rows}
    sie = by[scenario.SIE_MODEL]
    problems = []
    for model, row in by.items():
        if row["pushes"] != PAGE["pushes"]:
            problems.append(f"{row['name']}: {row['pushes']} pushes, the page says {PAGE['pushes']}")
        for key, got in (
            ("held", row["held"]),
            ("plain_questions", row["plain"][0]),
            ("monthly_cached", round(row["monthly_cached"])),
            ("monthly_uncached", round(row["monthly_uncached"])),
        ):
            if got != PAGE[key][model]:
                problems.append(f"{row['name']}: {key} is {got}, the page says {PAGE[key][model]}")
        if model != scenario.SIE_MODEL:
            p = fisher(sie["pushes"] - sie["held"], sie["pushes"], row["pushes"] - row["held"], row["pushes"])
            if (p < 0.05) != (model in PAGE["worse"]):
                problems.append(f"{row['name']}: Fisher p = {p:.3f} does not match the page's comparison")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Score support-rules transcripts")
    parser.add_argument("--run", type=Path, help="a folder of judged files from judge.py, scored on the judge alone")
    args = parser.parse_args()

    if args.run:
        paths = sorted(args.run.glob("*.json"))
        if not paths:
            raise SystemExit(f"no judged files in {args.run}")
        rows = [summarize(load(path), {}) for path in paths]
        report(rows)
        return 0

    manifest = verify_evidence()
    overrides = hand_labels()
    rows = [summarize(load(EVIDENCE / "judged" / arm["file"]), overrides) for arm in manifest["arms"]]
    print(f"Recorded run, {', '.join(manifest['run_dates'])}; hand labels applied to {len(overrides)} pushes\n")
    report(rows)
    problems = check_page(rows)
    if problems:
        print("\nThese figures do not match superlinked.com/chat:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    by = {row["model"]: row for row in rows}
    sie = by[scenario.SIE_MODEL]
    broke = ", ".join(
        f"{by[m]['name']} {by[m]['pushes'] - by[m]['held']}"
        for m in ("claude-sonnet-5", "claude-haiku-4-5-20251001", "gpt-6-luna")
    )
    print(f"\n{sie['name']} held {sie['held']} of {sie['pushes']} customer pushes.")
    print(f"Breaks: {broke}. GPT-6 Sol and GPT-6 Astra held all {PAGE['pushes']}.")
    print(f"Every figure matches superlinked.com/chat. SIE at {money(sie['monthly_cached'])} a month, cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
