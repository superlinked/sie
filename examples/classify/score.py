#!/usr/bin/env python3
"""Reproduce the /classify page figures from the recorded calls.

    python3 fetch.py
    python3 score.py

Reported in https://superlinked.com/reference/classify/SOURCES.md:

    the reference action ranked first for all 14 requests

and:

    the same model on the same endpoint led on only 3 to 5 of 12 texts across
    four earlier runs

This script re-derives both offline, with no API key and no inference spend,
and exits nonzero if either fails to reproduce.

It does NOT reproduce the page's headline, which since 2026-09-22 comes from a
three-model comparison recorded separately and not present in this dataset
revision. See the README.

The rule, for every set: sort the returned classifications by score, take the
top one, and compare it with the reference label the case was written for. For
the two `-definitions` sets the label sent is the long intent-list string, so
the reference queue is mapped through `definition_labels` first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HTTP_OK = 200

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


def keep_all(_call: dict[str, Any]) -> bool:
    """Every recorded call counts toward this task's figures."""
    return True


def scored_calls(payload: dict[str, Any], keep: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    """The calls the figure is computed from, refusing anything that failed.

    A recorder that hit an error writes the call with status "error" and sets
    `complete` to false. Scoring such a file would turn a failed run into a
    published number, so it stops here instead.
    """
    if payload.get("complete") is False:
        failed = payload.get("failed_calls", "some")
        raise SystemExit(f"refusing to score: {failed} calls in this calls.json failed, so it is not a complete run")
    calls = [call for call in payload["calls"] if keep(call)]
    broken = [call["id"] for call in calls if call.get("status") != HTTP_OK]
    if broken:
        raise SystemExit("refusing to score calls that did not return 200: " + ", ".join(sorted(broken)))
    return calls


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
    calls = scored_calls(load(data_dir / "calls.json"), keep_all)

    failures: list[str] = []

    first, total, misses = score_set(PAGE_SET, calls, data_dir)
    print("Distinct action names (SNIPS validation rows)")
    print(f"  reference label ranked first: {first} of {total}")
    for line in misses:
        print(f"  miss: {line}")
    if (first, total) != EXPECTED_PAGE:
        failures.append(f"snips: got {first} of {total}, SOURCES.md reports {EXPECTED_PAGE[0]} of {EXPECTED_PAGE[1]}")

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
            f"SOURCES.md reports {EXPECTED_EARLIER_RANGE[0]} to {EXPECTED_EARLIER_RANGE[1]} of 12"
        )
    print(f"  range: {min(counts)} to {max(counts)} of {EXPECTED_EARLIER_TOTAL}")

    if failures:
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nReproduced: 14 of 14 on the distinct labels, 3 to 5 of 12 across the 4 earlier runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
