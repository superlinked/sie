#!/usr/bin/env python3
"""Reproduce the /redact composition from the recorded calls.

    python3 fetch.py
    python3 score.py

The page masks personal data with four steps, and this script re-derives each
one offline from the recorded responses, with no API key and no inference
spend, exiting nonzero if any figure fails.

    26 of 45   one call to urchade/gliner_multi_pii-v1, whole document
    33 of 45   splitting what runs past the model's input window
    41 of 45   unioning a second call to numind/NuNER_Zero
    45 of 45   masking every later mention of a name already found

    42 of 49   masks that land on a published gold span
    0 of 29    currency amounts masked, so the figures stay readable

Both models are in the same task's catalog and both were recorded on all twelve
documents in the same run; the dataset has held all 24 calls since the page was
first published.

The two sides of every comparison come from different places. The gold spans
are the benchmark publishers' own annotations, carried in inputs/cases.json.
The masks are read out of the recorded API responses in calls.json. Neither is
derived from the other.

Rules, exactly as sie-web's CI applies them:

  floor       a returned span counts only at score >= 0.6, a threshold the
              caller sets. Five of the run's 168 spans fall below it: `ID #`,
              `File #` and `MIC #`, each covering a field's printed label and
              no value, and the given name `Annibale` twice, which the first
              model also returned at the same offsets inside `Annibale Caboto`.
              --floor-0 re-runs every step with the floor removed; all four
              reach the same figure, so it costs no coverage in this run
  covered     a gold span counts as masked only when the UNION of returned
              spans covers every one of its characters, so two overlapping
              returned spans are never counted twice
  merged      overlapping spans become one mask covering all of them, which is
              what the caller removes and what the precision figures count
  propagate   every other whole-word, case-sensitive occurrence of a token of
              three letters or more from a returned `person` span is masked too
  in schema   a mask counts toward the 49 only when its label is one the gold
              schema can express, through GOLD_TO_REQUESTED below
  on gold     such a mask counts toward the 42 when it overlaps any gold span

What this script does NOT check: which of the recorded documents the page
displays, or in what order. That is the page's decision, it changes without the
run changing, and an example has no way to read the page.
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

FIRST_MODEL = "urchade/gliner_multi_pii-v1"
SECOND_MODEL = "numind/NuNER_Zero"
FIRST_SET = "urchade__gliner_multi_pii-v1"
SECOND_SET = "numind__NuNER_Zero"
MODEL_SETS = (FIRST_SET, SECOND_SET)

# The confidence floor, and the person label the propagation step reads.
SCORE_FLOOR = 0.6
PERSON_LABEL = "person"
NAME_TOKEN = re.compile(r"[^\W\d_]{3,}")

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

AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d[\d.,]*\s?EUR|(?<![\w-])\d+\.\d{2}(?![\d-])")

# GLiNER counts words with this rule, so the page's "564-word chat" and its
# "384 words or fewer" window are both derived here rather than taken on trust.
GLINER_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")

WINDOW_CASE = "gretel_customer_support_log"
WINDOW_TAIL = "gretel_customer_support_log_tail"

EXPECTED = {
    "gold_spans": 45,
    "step_one_call": 26,
    "step_chunked": 33,
    "step_two_models": 41,
    "step_propagated": 45,
    "masks_in_gold_schema": 49,
    "masks_on_gold": 42,
    "propagated_masks": 6,
    "amounts_masked": 0,
    "amounts_found": 29,
    "window_total": 11,
    "window_one_request": 2,
    "window_composed": 11,
    "window_past": 7,
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


def returned_spans(
    entities: dict[tuple[str, str], list[dict[str, Any]]],
    cases: dict[str, dict[str, Any]],
    case_id: str,
    *,
    sets: tuple[str, ...] = MODEL_SETS,
    chunked: bool = True,
    floor: float = SCORE_FLOOR,
) -> list[dict[str, Any]]:
    """Spans the named models returned, in the parent document's own offsets.

    `chunked` False reads only the whole-document request, which is the
    single-request regime the first step reports.
    """
    parts: list[tuple[str, int]] = [(case_id, 0)]
    if chunked:
        parts += sorted(
            ((case["id"], case["parent_offset"]) for case in cases.values() if case.get("parent_case") == case_id),
            key=lambda part: part[1],
        )
    merged: dict[tuple[int, int, str], dict[str, Any]] = {}
    for part_id, offset in parts:
        for model_set in sets:
            for entity in entities[(model_set, part_id)]:
                if entity["score"] < floor:
                    continue
                span = {
                    **entity,
                    "start": entity["start"] + offset,
                    "end": entity["end"] + offset,
                    "derived": False,
                }
                key = (span["start"], span["end"], span["label"])
                if key not in merged or merged[key]["score"] < span["score"]:
                    merged[key] = span
    return sorted(merged.values(), key=lambda span: (span["start"], span["end"]))


def propagated_spans(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every other whole-word, case-sensitive mention of a returned name."""
    tokens = sorted(
        {token for span in spans if span["label"] == PERSON_LABEL for token in NAME_TOKEN.findall(span["text"])}
    )
    added: list[dict[str, Any]] = []
    for token in tokens:
        pattern = re.compile(r"(?<![^\W\d_])" + re.escape(token) + r"(?![^\W\d_])")
        for match in pattern.finditer(text):
            if any(span["start"] <= match.start() and span["end"] >= match.end() for span in spans):
                continue
            added.append(
                {
                    "text": token,
                    "label": PERSON_LABEL,
                    "score": None,
                    "start": match.start(),
                    "end": match.end(),
                    "derived": True,
                }
            )
    return sorted(added, key=lambda span: span["start"])


def composed_spans(
    entities: dict[tuple[str, str], list[dict[str, Any]]],
    cases: dict[str, dict[str, Any]],
    case_id: str,
    floor: float = SCORE_FLOOR,
) -> list[dict[str, Any]]:
    spans = returned_spans(entities, cases, case_id, floor=floor)
    added = propagated_spans(cases[case_id]["text"], spans)
    return sorted(spans + added, key=lambda span: (span["start"], span["end"]))


def merged_masks(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlapping spans collapsed into one mask each.

    Two models rarely agree on where a value ends. Keeping whichever span
    sorted first masked the shorter of two that start together, which left
    ", M.D." readable after a masked name and left the last character of a
    password readable on the policyholder letter.
    """
    masks: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
        if masks and span["start"] < masks[-1]["end"]:
            masks[-1]["end"] = max(masks[-1]["end"], span["end"])
            masks[-1]["spans"].append(span)
            continue
        masks.append({"start": span["start"], "end": span["end"], "spans": [span]})
    for mask in masks:
        best = max(mask["spans"], key=lambda span: span["score"] or 0)
        mask["label"] = best["label"]
        mask["text"] = text[mask["start"] : mask["end"]]
    return masks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument(
        "--floor-0",
        action="store_true",
        help="score every step with the confidence floor removed, to show what it costs",
    )
    args = parser.parse_args()
    floor = 0.0 if args.floor_0 else SCORE_FLOOR
    if args.floor_0:
        print("confidence floor removed: every figure below is scored at 0.0\n")
    data_dir = Path(args.data)

    cases = {case["id"]: case for case in load(data_dir / "inputs/cases.json")["cases"]}
    entities: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in scored_calls(load(data_dir / "calls.json"), keep_all):
        key = (call["set"], call["case"])
        if key in entities:
            print(f"two {call['set']} calls recorded for {call['case']}", file=sys.stderr)
            return 1
        entities[key] = call["response"]["body"]["items"][0]["entities"]

    missing = [
        f"{model_set}/{case_id}"
        for model_set in MODEL_SETS
        for case_id in cases
        if (model_set, case_id) not in entities
    ]
    if missing:
        print(f"no call recorded for: {', '.join(missing)}", file=sys.stderr)
        return 1

    parents = [case_id for case_id, case in cases.items() if not case.get("parent_case")]
    benchmark = [case_id for case_id in parents if cases[case_id].get("gold_spans")]

    # --- the four steps ----------------------------------------------------
    steps = {"step_one_call": 0, "step_chunked": 0, "step_two_models": 0, "step_propagated": 0}
    totals = {"gold_spans": 0, "masks_in_gold_schema": 0, "masks_on_gold": 0}
    per_document: list[tuple[str, int, int, int]] = []
    for case_id in benchmark:
        case = cases[case_id]
        gold = mapped_gold(case)
        one_call = returned_spans(entities, cases, case_id, sets=(FIRST_SET,), chunked=False, floor=floor)
        chunked = returned_spans(entities, cases, case_id, sets=(FIRST_SET,), floor=floor)
        two_models = returned_spans(entities, cases, case_id, floor=floor)
        composed = composed_spans(entities, cases, case_id, floor=floor)

        totals["gold_spans"] += len(gold)
        steps["step_one_call"] += sum(1 for span in gold if covered(one_call, span))
        steps["step_chunked"] += sum(1 for span in gold if covered(chunked, span))
        steps["step_two_models"] += sum(1 for span in gold if covered(two_models, span))
        composed_hits = sum(1 for span in gold if covered(composed, span))
        steps["step_propagated"] += composed_hits
        per_document.append((case_id, len(gold), sum(1 for span in gold if covered(one_call, span)), composed_hits))

        requested = set(case["labels"])
        for mask in merged_masks(case["text"], composed):
            if mask["label"] not in requested:
                continue
            totals["masks_in_gold_schema"] += 1
            if any(mask["start"] < span["end"] and span["start"] < mask["end"] for span in gold):
                totals["masks_on_gold"] += 1

    print(f"benchmark documents scored: {len(benchmark)}")
    for case_id, gold_count, one_call, composed_hits in per_document:
        print(f"  {case_id:<36} {one_call} of {gold_count} in one call, {composed_hits} composed")

    print("\npublished PII spans masked, by step:")
    total = totals["gold_spans"]
    print(f"  {steps['step_one_call']:>2} of {total}  one call to {FIRST_MODEL}, whole document")
    print(f"  {steps['step_chunked']:>2} of {total}  splitting what runs past the model's input window")
    print(f"  {steps['step_two_models']:>2} of {total}  unioning a second call to {SECOND_MODEL}")
    print(f"  {steps['step_propagated']:>2} of {total}  masking every later mention of a name already found")

    unjudged = totals["masks_in_gold_schema"] - totals["masks_on_gold"]
    print(f"\n{totals['masks_on_gold']} of {totals['masks_in_gold_schema']} masks land on a published gold span")
    print(f"{unjudged} fall outside them, so neither benchmark can confirm or refute those")

    # --- what the propagation step added, over every recorded document -----
    propagated: list[str] = []
    for case_id in parents:
        spans = returned_spans(entities, cases, case_id, floor=floor)
        propagated += [span["text"] for span in propagated_spans(cases[case_id]["text"], spans)]
    print(f"\nthe propagation step added {len(propagated)} masks: {', '.join(sorted(set(propagated)))}")

    # --- amounts left readable, over every recorded document ---------------
    amounts = {"found": 0, "masked": 0}
    for case_id in parents:
        text = cases[case_id]["text"]
        spans = composed_spans(entities, cases, case_id, floor=floor)
        for match in AMOUNT.finditer(text):
            amounts["found"] += 1
            if any(span["start"] < match.end() and match.start() < span["end"] for span in spans):
                amounts["masked"] += 1
    print(
        f"\n{amounts['masked']} of {amounts['found']} currency amounts masked"
        f" across the {len(parents)} recorded documents"
    )

    # --- the input window --------------------------------------------------
    parent = cases[WINDOW_CASE]
    tail = cases[WINDOW_TAIL]
    window_end = parent["gliner_window_end_char"]
    gold = mapped_gold(parent)
    one_call = returned_spans(entities, cases, WINDOW_CASE, sets=(FIRST_SET,), chunked=False, floor=floor)
    composed = composed_spans(entities, cases, WINDOW_CASE, floor=floor)
    window = {
        "total": len(gold),
        "one_request": sum(1 for span in gold if covered(one_call, span)),
        "composed": sum(1 for span in gold if covered(composed, span)),
        "past": sum(1 for span in gold if span["end"] > window_end),
    }
    words = list(GLINER_WORD.finditer(parent["text"]))
    window["words_total"] = len(words)
    window["words_in_window"] = sum(1 for word in words if word.end() <= window_end)
    print(
        f"\n{window['words_total']}-word chat, {window['total']} published spans:"
        f" {window['one_request']} masked by one call, {window['composed']} by the composition"
    )
    print(f"  the model reads the first {window['words_in_window']} words, up to character {window_end}")
    print(
        f"  {window['past']} of those spans sit past that point,"
        f" and the second request starts at character {tail['parent_offset']}"
    )

    got = {
        "gold_spans": totals["gold_spans"],
        **steps,
        "masks_in_gold_schema": totals["masks_in_gold_schema"],
        "masks_on_gold": totals["masks_on_gold"],
        "propagated_masks": len(propagated),
        "amounts_masked": amounts["masked"],
        "amounts_found": amounts["found"],
        "window_total": window["total"],
        "window_one_request": window["one_request"],
        "window_composed": window["composed"],
        "window_past": window["past"],
        "window_words_total": window["words_total"],
        "window_words_in_window": window["words_in_window"],
    }
    failures = [f"{key}: got {got[key]}, page publishes {want}" for key, want in EXPECTED.items() if got[key] != want]
    # A step can only ever mask more than the one before it. Checked as an
    # ordering rather than only as four constants, because four constants that
    # all move together still pass.
    ladder = [steps["step_one_call"], steps["step_chunked"], steps["step_two_models"], steps["step_propagated"]]
    if ladder != sorted(ladder):
        failures.append(f"the steps do not increase: {ladder}")
    if ladder[-1] != totals["gold_spans"]:
        failures.append(f"the last step masks {ladder[-1]} of {totals['gold_spans']}, not all of them")
    if any(text.isalpha() is False for text in propagated):
        failures.append("the propagation step added something that is not a name")
    if failures:
        sys.stdout.flush()
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nReproduced: 26, 33, 41 and 45 of 45, 42 of 49 on gold, and 0 of 29 amounts masked.")
    if args.floor_0:
        print("Every figure holds with the floor removed, so it costs no coverage in this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
