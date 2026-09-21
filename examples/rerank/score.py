#!/usr/bin/env python3
"""Reproduce the /rerank page figures from the recorded calls.

    python3 fetch.py
    python3 score.py

The page publishes no percentage. What it publishes is a pair of scores per
case, rounded to three decimals, and the claim that the complete-evidence
passage is the one on top. This script re-derives both offline, with no API
key and no inference spend, and exits nonzero if any of them fails.

    4 of 4     cases rank the expected primary-source passage first
    1.000 / 0.992   the Salem alert over the Wayside Help Desk process
    1.000 / 0.439   the two-contract holding over the one-contract rule
    1.000 / 0.816   the missing-documentation denial over seven-month evidence
    1.000 / 0.107   the non-reliance statement over the amendment's scope

The two sides of every comparison come from different places. The expected top
candidate and every excerpt digest were registered in inputs/cases.json before
the run. The ranks and scores are read out of the recorded API responses in
calls.json. Neither is derived from the other, and the four score pairs in
EXPECTED below come from a third place again: the page.

The excerpt checks that used to run inside the runner run here too, so moving
the bytes to Hugging Face did not drop them. Every candidate's text is hashed
and compared with the digest recorded beside it and with the canonical digest
in inputs/sources.json, and a rewritten excerpt fails even when both copies of
its declared digest are rewritten with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

HTTP_OK = 200
MODEL = "Qwen/Qwen3-Reranker-4B"
# The page prints a pair per case, so a case with nothing to compare against
# cannot produce the figure. Refused rather than indexed into.
MIN_CANDIDATES = 2

# What the page prints, at the page's own three decimals: the winner and the
# closest other candidate for each case. Typed out from the rendered page, not
# computed from the recordings, so agreement means something.
EXPECTED = {
    "sec_filing_amendment": ("sec-non-reliance", "1.000", "sec-amendment-scope", "0.107"),
    "cms_lower_limb_orthosis": ("cms-missing-documentation", "1.000", "cms-seven-month-evidence", "0.816"),
    "ntsb_detector_alert": ("ntsb-salem-noncritical", "1.000", "ntsb-help-desk-process", "0.992"),
    "scotus_two_contracts": ("scotus-two-contract-holding", "1.000", "scotus-one-contract-rule", "0.439"),
}
EXPECTED_CANDIDATES = {
    "sec_filing_amendment": 4,
    "cms_lower_limb_orthosis": 5,
    "ntsb_detector_alert": 4,
    "scotus_two_contracts": 4,
}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_inputs(cases: dict[str, Any], sources: dict[str, Any]) -> None:
    """Every candidate is the verbatim excerpt its two manifests claim.

    Carried unchanged from the runner this example shipped before the evidence
    moved to Hugging Face. A candidate whose text was rewritten fails here even
    when the digest recorded beside it was rewritten to match, because the
    canonical digest in sources.json has to agree as well.
    """
    if cases.get("integrity_policy", {}).get("synthetic_or_paraphrased_evidence") is not False:
        raise SystemExit("inputs/cases.json must reject synthetic or paraphrased evidence")
    if sources.get("synthetic_or_paraphrased_evidence") is not False:
        raise SystemExit("inputs/sources.json must reject synthetic evidence")
    if cases.get("model") != MODEL:
        raise SystemExit(f"inputs/cases.json names {cases.get('model')}, expected {MODEL}")

    known = sources["sources"]
    for case_id, case in cases["cases"].items():
        seen: set[str] = set()
        for candidate in case["candidates"]:
            candidate_id = candidate["id"]
            if candidate_id in seen:
                raise SystemExit(f"{case_id}: candidate {candidate_id} appears twice")
            seen.add(candidate_id)
            source = known.get(candidate["source_id"])
            if source is None:
                raise SystemExit(f"{candidate_id}: no source {candidate['source_id']} in sources.json")
            actual = sha256_text(candidate["text"])
            if actual != candidate["sha256"]:
                raise SystemExit(f"{candidate_id}: excerpt does not match the digest recorded beside it")
            canonical = source.get("excerpts", {}).get(candidate_id)
            if canonical is None:
                raise SystemExit(f"{candidate_id}: no canonical excerpt in sources.json")
            if candidate["locator"] != canonical["locator"] or actual != canonical["sha256"]:
                raise SystemExit(f"{candidate_id}: excerpt does not match the canonical one in sources.json")
        if case["expected_top_candidate_id"] not in seen:
            raise SystemExit(f"{case_id}: expected top candidate is not one of its candidates")


def scored_calls(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The recorded calls, refusing anything that failed or repeats a case.

    A recorder that hit an error writes the call with status "error" and sets
    `complete` to false. Scoring such a file would turn a failed run into a
    published number, so it stops here instead.
    """
    if payload.get("complete") is False:
        failed = payload.get("failed_calls", "some")
        raise SystemExit(f"refusing to score: {failed} calls in this calls.json failed, so it is not a complete run")
    by_case: dict[str, dict[str, Any]] = {}
    for call in payload["calls"]:
        if call.get("status") != HTTP_OK:
            raise SystemExit(f"refusing to score {call['id']}: status {call.get('status')}")
        if call["case"] in by_case:
            raise SystemExit(f"two calls recorded for {call['case']}")
        by_case[call["case"]] = call
    return by_case


def ranked(case_id: str, case: dict[str, Any], body: dict[str, Any]) -> list[dict[str, Any]]:
    """The response rows in rank order, with the checks the runner applied.

    Fail-closed, in the same order the runner used: wrong model, wrong query
    id, missing or duplicated candidates, incomplete ranks, or a score that is
    not a finite number all stop the run rather than producing a figure.
    """
    if body.get("model") != MODEL:
        raise SystemExit(f"{case_id}: response names model {body.get('model')}")
    if body.get("query_id") != f"{case_id}-query":
        raise SystemExit(f"{case_id}: response names query {body.get('query_id')}")
    rows = body.get("scores")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"{case_id}: response carries no score list")

    expected_ids = {candidate["id"] for candidate in case["candidates"]}
    observed_ids = {row.get("item_id") for row in rows}
    if observed_ids != expected_ids or len(rows) != len(expected_ids):
        raise SystemExit(f"{case_id}: scored {len(rows)} rows for {len(expected_ids)} candidates")
    ranks = sorted(row.get("rank") for row in rows)
    if ranks != list(range(len(rows))) or any(type(row["rank"]) is not int for row in rows):
        raise SystemExit(f"{case_id}: ranks are not 0..{len(rows) - 1}")
    for row in rows:
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise SystemExit(f"{case_id}: {row.get('item_id')} has a non-numeric score")
    return sorted(rows, key=lambda row: row["rank"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    args = parser.parse_args()
    data_dir = Path(args.data)

    cases = load(data_dir / "inputs/cases.json")
    sources = load(data_dir / "inputs/sources.json")
    verify_inputs(cases, sources)
    calls = scored_calls(load(data_dir / "calls.json"))

    missing = [case_id for case_id in cases["cases"] if case_id not in calls]
    if missing:
        raise SystemExit("no call recorded for: " + ", ".join(missing))
    unexpected = [case_id for case_id in calls if case_id not in cases["cases"]]
    if unexpected:
        raise SystemExit("calls recorded for cases the inputs do not define: " + ", ".join(unexpected))

    first = 0
    candidates_scored = 0
    failures: list[str] = []
    print(f"{len(cases['cases'])} cases, model {MODEL}\n")
    for case_id, case in cases["cases"].items():
        rows = ranked(case_id, case, calls[case_id]["response"]["body"])
        candidates_scored += len(rows)
        if len(rows) < MIN_CANDIDATES:
            raise SystemExit(f"{case_id}: {len(rows)} candidates, so there is no closest other candidate to report")
        top, runner_up = rows[0], rows[1]
        expected_top = case["expected_top_candidate_id"]
        if top["item_id"] == expected_top:
            first += 1
        else:
            failures.append(f"{case_id}: {expected_top} was registered first, {top['item_id']} ranked first")
        print(
            f"  {case_id:<24} {len(rows)} candidates"
            f"   {top['item_id']:<28} {top['score']:.3f}"
            f"   next {runner_up['item_id']:<28} {runner_up['score']:.3f}"
        )
        want_top, want_top_score, want_next, want_next_score = EXPECTED[case_id]
        got = (top["item_id"], f"{top['score']:.3f}", runner_up["item_id"], f"{runner_up['score']:.3f}")
        if got != (want_top, want_top_score, want_next, want_next_score):
            failures.append(
                f"{case_id}: page prints {want_top} {want_top_score} over {want_next} {want_next_score},"
                f" recordings give {got[0]} {got[1]} over {got[2]} {got[3]}"
            )
        if len(rows) != EXPECTED_CANDIDATES[case_id]:
            failures.append(
                f"{case_id}: page says {EXPECTED_CANDIDATES[case_id]} candidates, recordings hold {len(rows)}"
            )

    print(f"\n{first} of {len(cases['cases'])} cases rank the expected primary-source passage first")
    print(f"{candidates_scored} candidates scored exactly once across the {len(cases['cases'])} cases")

    if failures:
        sys.stdout.flush()
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nReproduced the four score pairs the page prints: 1.000/0.992, 1.000/0.439, 1.000/0.816, 1.000/0.107")
    return 0


if __name__ == "__main__":
    sys.exit(main())
