#!/usr/bin/env python3
"""Reproduce the /redact page figures from the recorded calls.

    python3 fetch.py
    python3 score.py

The page has no single headline number. It publishes four concrete figures,
and this script re-derives all of them offline, with no API key and no
inference spend, exiting nonzero if any fails.

    23 of 25   masks confirmed by the published gold spans in six benchmark documents
    0 of 29    amounts on this page were masked, so the figures stay readable
    26 of 45   published PII spans masked by one request per document, before any splitting

and, on the request-splitting card:

    This 564-word chat masks 2 of 11 spans in one request and 8 of 11 when split.

All figures are for `urchade/gliner_multi_pii-v1`, which is the model the page
shows. The `numind/NuNER_Zero` calls over the same documents are in the dataset
and no published figure rests on them, so nothing here scores them.

The two sides of every comparison come from different places. The gold spans
are the benchmark publishers' own annotations, carried in inputs/cases.json.
The masks are read out of the recorded API responses in calls.json. Neither is
derived from the other.

Rules, exactly as sie-web's CI applies them:

  covered     a gold span counts as masked only when the UNION of returned
              spans covers every one of its characters, so two overlapping
              returned spans are never counted twice
  in schema   a returned span counts toward the 25 only when its label is one
              the gold schema can express, through GOLD_TO_REQUESTED below
  on gold     such a span counts toward the 23 when it overlaps any gold span
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HTTP_OK = 200

MODEL_SET = "urchade__gliner_multi_pii-v1"

# Benchmark gold labels mapped to the label names the requests used.
GOLD_TO_REQUESTED = {
    "first_name": "person",
    "last_name": "person",
    "name": "person",
    "email": "email",
    "ssn": "social security number",
    "phone_number": "phone number",
    "street_address": "address",
    "date_of_birth": "date of birth",
    "medical_record_number": "medical record number",
    "driver_license_number": "driver's license number",
    "password": "password",
    "bban": "account number",
    "iban": "iban",
    "swift_bic_code": "bank identifier code",
    "employee_id": "employee id",
    "credit_debit_card": "credit card number",
}

# The documents the page renders, across hero, proof cards, playground and the
# request-splitting card. The amounts figure is over exactly these, because it
# is a claim about what a reader sees, not about the whole recorded set.
DISPLAYED = [
    "cfpb_closing_disclosure_transaction",
    "cfpb_closing_disclosure_contacts",
    "cms_medicare_summary_notice_part_b",
    "gretel_german_health_claim",
    "gretel_policyholder_report",
    "cms_medicare_summary_notice_dme",
    "gretel_customer_support_log",
]

AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d[\d.,]*\s?EUR|(?<![\w-])\d+\.\d{2}(?![\d-])")

# GLiNER counts words with this rule, so the page's "564-word chat" and its
# "384 words or fewer" window are both derived here rather than taken on trust.
GLINER_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")

WINDOW_CASE = "gretel_customer_support_log"
WINDOW_TAIL = "gretel_customer_support_log_tail"

EXPECTED = {
    "returned_on_gold": 23,
    "returned_in_gold_schema": 25,
    "amounts_masked": 0,
    "amounts_found": 29,
    "gold_found": 26,
    "gold_spans": 45,
    "window_total": 11,
    "window_one_request": 2,
    "window_two_requests": 8,
    "window_past": 7,
    "window_recovered": 6,
    "window_words_total": 564,
    "window_words_in_window": 384,
}


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


def covered(entities: list[dict[str, Any]], gold: dict[str, Any]) -> bool:
    """True when the union of returned spans covers every character of gold."""
    overlaps = sorted(
        (max(entity["start"], gold["start"]), min(entity["end"], gold["end"]))
        for entity in entities
        if max(entity["start"], gold["start"]) < min(entity["end"], gold["end"])
    )
    reached = gold["start"]
    for start, end in overlaps:
        if start > reached:
            return False
        reached = max(reached, end)
        if reached >= gold["end"]:
            return True
    return False


def mapped_gold(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [span for span in case.get("gold_spans", []) if GOLD_TO_REQUESTED.get(span["label"], "") in case["labels"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    args = parser.parse_args()
    data_dir = Path(args.data)

    cases = {case["id"]: case for case in load(data_dir / "inputs/cases.json")["cases"]}
    entities: dict[str, list[dict[str, Any]]] = {}
    for call in scored_calls(load(data_dir / "calls.json"), lambda call: call["set"] == MODEL_SET):
        if call["set"] != MODEL_SET:
            continue
        if call["case"] in entities:
            print(f"two {MODEL_SET} calls recorded for {call['case']}", file=sys.stderr)
            return 1
        entities[call["case"]] = call["response"]["body"]["items"][0]["entities"]

    missing = [case_id for case_id in cases if case_id not in entities]
    if missing:
        print(f"no {MODEL_SET} call recorded for: {', '.join(missing)}", file=sys.stderr)
        return 1

    # --- benchmark agreement, one request per document ---------------------
    totals = {"gold_spans": 0, "gold_found": 0, "returned_in_gold_schema": 0, "returned_on_gold": 0}
    requested = set(GOLD_TO_REQUESTED.values())
    benchmark_documents = []
    for case_id, case in cases.items():
        if not case.get("gold_spans") or case.get("parent_case"):
            continue
        benchmark_documents.append(case_id)
        found = entities[case_id]
        gold = mapped_gold(case)
        totals["gold_spans"] += len(gold)
        totals["gold_found"] += sum(1 for span in gold if covered(found, span))
        for entity in found:
            if entity["label"] not in requested:
                continue
            totals["returned_in_gold_schema"] += 1
            if any(entity["start"] < span["end"] and span["start"] < entity["end"] for span in gold):
                totals["returned_on_gold"] += 1

    print(f"benchmark documents scored: {len(benchmark_documents)}")
    for case_id in benchmark_documents:
        gold = mapped_gold(cases[case_id])
        hit = sum(1 for span in gold if covered(entities[case_id], span))
        print(f"  {case_id:<36} {hit} of {len(gold)} published spans masked")
    print(
        f"\n{totals['returned_on_gold']} of {totals['returned_in_gold_schema']} masks confirmed by the published gold spans"
    )
    print(f"{totals['gold_found']} of {totals['gold_spans']} published PII spans masked by one request per document")
    unjudged = totals["returned_in_gold_schema"] - totals["returned_on_gold"]
    print(f"{unjudged} masks fall outside the gold spans, so neither benchmark can confirm or refute them")

    # --- amounts left readable, over the documents the page renders --------
    amounts = {"found": 0, "masked": 0}
    for case_id in DISPLAYED:
        text = cases[case_id]["text"]
        spans = entities[case_id]
        for match in AMOUNT.finditer(text):
            amounts["found"] += 1
            if any(span["start"] < match.end() and match.start() < span["end"] for span in spans):
                amounts["masked"] += 1
    print(
        f"\n{amounts['masked']} of {amounts['found']} amounts masked across the {len(DISPLAYED)} documents the page renders"
    )

    # --- request splitting -------------------------------------------------
    parent = cases[WINDOW_CASE]
    tail = cases[WINDOW_TAIL]
    offset = tail["parent_offset"]
    window_end = parent["gliner_window_end_char"]
    full = entities[WINDOW_CASE]
    second = entities[WINDOW_TAIL]

    gold = mapped_gold(parent)
    window = {"total": len(gold), "one_request": 0, "two_requests": 0, "past": 0, "recovered": 0}
    for span in gold:
        hit_full = covered(full, span)
        window["one_request"] += hit_full
        hit_second = False
        if span["start"] >= offset:
            shifted = {**span, "start": span["start"] - offset, "end": span["end"] - offset}
            hit_second = covered(second, shifted)
        window["two_requests"] += hit_full or hit_second
        if span["end"] > window_end:
            window["past"] += 1
            window["recovered"] += hit_second

    words = list(GLINER_WORD.finditer(parent["text"]))
    window["words_total"] = len(words)
    window["words_in_window"] = sum(1 for word in words if word.end() <= window_end)
    print(
        f"\n{window['words_total']}-word chat, {window['total']} published spans:"
        f" {window['one_request']} masked in one request, {window['two_requests']} when split"
    )
    print(f"  the model reads the first {window['words_in_window']} words, up to character {window_end}")
    print(f"  {window['past']} of those spans sit past word 384, and the second request recovers {window['recovered']}")

    got = {
        "returned_on_gold": totals["returned_on_gold"],
        "returned_in_gold_schema": totals["returned_in_gold_schema"],
        "amounts_masked": amounts["masked"],
        "amounts_found": amounts["found"],
        "gold_found": totals["gold_found"],
        "gold_spans": totals["gold_spans"],
        "window_total": window["total"],
        "window_one_request": window["one_request"],
        "window_two_requests": window["two_requests"],
        "window_past": window["past"],
        "window_recovered": window["recovered"],
        "window_words_total": window["words_total"],
        "window_words_in_window": window["words_in_window"],
    }
    failures = [f"{key}: got {got[key]}, page publishes {want}" for key, want in EXPECTED.items() if got[key] != want]
    if failures:
        sys.stdout.flush()
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nReproduced: 23 of 25, 0 of 29, 26 of 45, and 2 of 11 in one request against 8 of 11 when split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
