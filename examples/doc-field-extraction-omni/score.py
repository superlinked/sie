#!/usr/bin/env python3
"""Check the recorded field-extraction run and print what the page publishes.

    python3 fetch.py
    python3 score.py                 # the recorded run
    python3 score.py --runs runs     # also score your own run.py output

Standard library only; no key, no network. For each model it prints:
- OmniAI JSON accuracy on the 700 business documents, recomputed as the mean of the per-document scores
  `score.mjs` produces with OmniAI's own code (run `npm ci && node score.mjs` to recompute those from the replies)
- fields right after normalising case, spacing, commas and curly quotes, a second view of the same replies
- documents with every field right, replies Claude's structured outputs refused, and the cost of 1,000 pages at list
  price on the model's own recorded tokens

It then checks that the Claude schema-acceptance record and the page export agree with the replies, and exits
non-zero if any published figure differs.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")):
        text = text.replace(a, b)
    text = "".join(ch for ch in text if not ch.isspace() and ch != ",").casefold()
    return text[:-1] if text.endswith(".") else text


def match(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)):
        if isinstance(actual, str):
            try:
                actual = float(actual.replace(",", ""))
            except ValueError:
                return False
        return isinstance(actual, (int, float)) and abs(float(expected) - float(actual)) < 1e-9
    if isinstance(expected, str):
        return isinstance(actual, str) and normalize(expected) == normalize(actual)
    return expected == actual


def fields(expected: Any, actual: Any) -> list[bool]:
    if isinstance(expected, dict):
        a = actual if isinstance(actual, dict) else {}
        return [m for k, v in expected.items() for m in fields(v, a.get(k))]
    if isinstance(expected, list):
        a = actual if isinstance(actual, list) else []
        return [m for i, v in enumerate(expected) for m in fields(v, a[i] if i < len(a) else None)]
    return [match(expected, actual)]


def load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] not in rows or rows[r["id"]].get("error"):
                rows[r["id"]] = r
    return rows


def parsed(row: dict[str, Any] | None) -> Any:
    if not row or row.get("error"):
        return None
    try:
        return json.loads(row["text"])
    except (json.JSONDecodeError, TypeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path)
    args = ap.parse_args()
    if not (EVIDENCE / "inputs" / "sets.json").exists():
        raise SystemExit("run python3 fetch.py first")
    sets = json.loads((EVIDENCE / "inputs" / "sets.json").read_text(encoding="utf-8"))
    page = json.loads((EVIDENCE / "page.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    docs = {d["id"]: d for d in sets["documents"]}
    omni_ids = [d["id"] for d in sets["documents"] if d["set"] == "omni"]
    if omni_ids != page["omniIds"]:
        print("the page export's document order differs from the inputs", file=sys.stderr)
        return 1
    problems = 0
    print(f"{len(omni_ids)} OmniAI business documents; {len(docs) - len(omni_ids)} CORD receipts reported in SOURCES.md")
    print(f"{'model':<20} {'OmniAI':>7} {'normalised':>11} {'all right':>10} {'refused':>8} {'$/1k pages':>11}")
    for arm, pub in page["arms"].items():
        rows = load(EVIDENCE / "runs" / pub["run"] / f"{arm}.jsonl")
        if set(rows) != set(docs):
            print(f"{arm}: the recorded run does not cover every document", file=sys.stderr)
            problems += 1
            continue
        omni = sum(pub["perDocOmni"]) / len(pub["perDocOmni"])
        per = {i: fields(docs[i]["gold"], parsed(rows[i])) for i in omni_ids}
        norm = sum(sum(m) / len(m) for m in per.values()) / len(per)
        right = sum(all(m) for m in per.values()) / len(per)
        refused = sum("fallback" in (r.get("finish") or "") for r in rows.values())
        tin = sum(rows[i].get("tokens_in") or 0 for i in omni_ids) / len(omni_ids)
        tout = sum(rows[i].get("tokens_out") or 0 for i in omni_ids) / len(omni_ids)
        price_in, price_out = manifest["arms"][arm]["usd_per_1m_tokens"]
        usd = (tin * price_in + tout * price_out) / 1000
        print(f"{pub['name']:<20} {100 * omni:>6.1f}% {100 * norm:>10.1f}% {100 * right:>9.1f}% {refused:>8} {usd:>10.2f}")
        for label, mine, theirs in (("OmniAI", omni, pub["omni"]), ("all right", right, pub["fullyRightOmni"]),
                                    ("$/1k", usd, pub["usdPer1kList"])):  # fmt: skip
            if abs(mine - theirs) > 1e-3:
                print(f"  {arm}: {label} {mine} differs from the page's {theirs}", file=sys.stderr)
                problems += 1
        if refused != pub["refusedStructured"]:
            print(f"  {arm}: {refused} refusals, the page export says {pub['refusedStructured']}", file=sys.stderr)
            problems += 1
    acceptance = json.loads((EVIDENCE / "claude_schema_acceptance.json").read_text(encoding="utf-8"))
    for arm, a in acceptance["arms"].items():
        refused = sum(not r["accepted"] for r in a["rows"].values())
        print(f"{arm}: structured outputs refused {refused} of {a['n']} of the benchmark's own schemas")
        if refused != page["acceptance"][arm]["refused"]:
            problems += 1
    if args.runs:
        for f in sorted(args.runs.glob("*.jsonl")):
            rows = load(f)
            ids = [i for i in omni_ids if i in rows]
            if ids:
                norm = sum(sum(m) / len(m) for m in (fields(docs[i]["gold"], parsed(rows[i])) for i in ids)) / len(ids)
                print(f"your run {f.stem}: {100 * norm:.1f}% of fields right (normalised) on {len(ids)} documents;"
                      " node score.mjs runs gives OmniAI's figure")  # fmt: skip
    if problems:
        print(f"{problems} figure(s) differ from the page", file=sys.stderr)
        return 1
    print("Every figure matches the page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
