#!/usr/bin/env python3
"""Reproduce the /classify page figures from the recorded calls.

    python3 fetch.py
    python3 score.py

Published on https://superlinked.com/classify:

    All 14 held-out requests land on the action they were written for

and, in the proof body:

    The same model put the reference label first for only 3 to 5 of 12 texts
    across 4 earlier runs on CFPB complaint products and CLINC150 assistant
    domains, where the labels overlapped. The 7 distinct action names above
    scored 14 of 14.

This script re-derives both offline, with no API key and no inference spend,
and exits nonzero if either fails to reproduce.

The rule, for every set: sort the returned classifications by score, take the
top one, and compare it with the reference label the case was written for. For
the two `-definitions` sets the label sent is the long intent-list string, so
the reference queue is mapped through `definition_labels` first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PAGE_SET = "snips"
EARLIER_SETS = ("cfpb", "cfpb-definitions", "clinc150", "clinc150-definitions")
FOLDER = {
    "snips": "snips",
    "clinc150": "clinc150",
    "clinc150-definitions": "clinc150",
    "cfpb": "cfpb",
    "cfpb-definitions": "cfpb",
}
EXPECTED_PAGE = (14, 14)
EXPECTED_EARLIER_RANGE = (3, 5)
EXPECTED_EARLIER_TOTAL = 12
EXPECTED_EARLIER_RUNS = 4


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def ranked(call: dict[str, Any]) -> list[dict[str, Any]]:
    classifications = call["response"]["body"]["items"][0]["classifications"]
    return sorted(classifications, key=lambda entry: entry["score"], reverse=True)


def score_set(set_name: str, calls: list[dict[str, Any]], data_dir: Path) -> tuple[int, int, list[str]]:
    case_file = load(data_dir / f"inputs/{FOLDER[set_name]}/cases.json")
    by_slug = {case["slug"]: case for case in case_file["cases"]}
    definition_label = {}
    if set_name.endswith("-definitions"):
        definition_label = {entry["queue"]: entry["label"] for entry in case_file["definition_labels"]}

    first = 0
    total = 0
    misses: list[str] = []
    for call in calls:
        if call["set"] != set_name:
            continue
        total += 1
        case = by_slug[call["case"]]
        reference = case["reference_label"]
        target = definition_label.get(reference, reference)
        order = ranked(call)
        if order[0]["label"] == target:
            first += 1
        else:
            rank = next((i for i, entry in enumerate(order, start=1) if entry["label"] == target), None)
            misses.append(f"{call['case']}: returned {order[0]['label']!r}, reference ranked {rank}")
    return first, total, misses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    args = parser.parse_args()
    data_dir = Path(args.data)
    calls = load(data_dir / "calls.json")["calls"]

    failures: list[str] = []

    first, total, misses = score_set(PAGE_SET, calls, data_dir)
    print("Page evidence, 7 distinct action names (SNIPS validation rows)")
    print(f"  reference label ranked first: {first} of {total}")
    for line in misses:
        print(f"  miss: {line}")
    if (first, total) != EXPECTED_PAGE:
        failures.append(f"snips: got {first} of {total}, page publishes {EXPECTED_PAGE[0]} of {EXPECTED_PAGE[1]}")

    print("\nFour earlier runs, where the labels overlapped")
    counts = []
    for set_name in EARLIER_SETS:
        first, total, _ = score_set(set_name, calls, data_dir)
        counts.append(first)
        print(f"  {set_name:<22} {first} of {total}")
        if total != EXPECTED_EARLIER_TOTAL:
            failures.append(f"{set_name}: {total} texts, expected {EXPECTED_EARLIER_TOTAL}")
    if len(counts) != EXPECTED_EARLIER_RUNS:
        failures.append(f"{len(counts)} earlier runs, expected {EXPECTED_EARLIER_RUNS}")
    if (min(counts), max(counts)) != EXPECTED_EARLIER_RANGE:
        failures.append(
            f"earlier runs span {min(counts)} to {max(counts)} of 12, "
            f"page publishes {EXPECTED_EARLIER_RANGE[0]} to {EXPECTED_EARLIER_RANGE[1]} of 12"
        )
    print(f"  range: {min(counts)} to {max(counts)} of {EXPECTED_EARLIER_TOTAL}")

    if failures:
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nReproduced: 14 of 14 on the page set, 3 to 5 of 12 across the 4 earlier runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
