#!/usr/bin/env python3
"""Reproduce the /named-entities page figures from the recorded calls.

    python3 fetch.py
    python3 score.py

The page has no single headline number. It publishes a span count per card, a
remainder counter derived from that count, and two labels it says came back
empty. This script re-derives all of them offline, with no API key and no
inference spend, exiting nonzero if any fails.

    24, 9, 13 and 7   spans returned for the four label sets, 53 in total
    28 of 28          required anchors matched at their registered offsets
    payment action    requested for the CMS claim, no span returned
    recipient         requested for the rail alerts, no span returned

The page shows 5, 6, 5 and 5 of those spans and renders the rest as a
"+N more spans returned" counter. That cap is a display decision in sie-web,
computed there from the same totals this script prints; the recordings hold
every span either way.

The two sides of every comparison come from different places. The required
anchors and every excerpt digest were registered in inputs/cases.json before the
run. The spans are read out of the recorded API responses in calls.json. Neither
is derived from the other.

The offset contract that used to run inside the runner runs here too, so moving
the bytes to Hugging Face did not drop it. Every returned span must use a
requested label, carry a finite score in [0, 1] and reproduce its own text from
input_text[start:end].
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
MODEL = "urchade/gliner_multi-v2.1"
ANCHOR_FIELDS = ("text", "label", "start", "end")

# What the page prints. Typed out from the rendered page, not computed from the
# recordings, so agreement means something.
EXPECTED_SPANS = {
    "sec_filing_amendment": 24,
    "cms_lower_limb_orthosis": 9,
    "ntsb_detector_alert": 13,
    "scotus_two_contracts": 7,
}
EXPECTED_UNRETURNED = {
    "cms_lower_limb_orthosis": ["payment action"],
    "ntsb_detector_alert": ["recipient"],
}
EXPECTED_TOTALS = {"spans": 53, "anchors": 28}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def anchor_key(value: dict[str, Any]) -> tuple[str, str, int, int]:
    return (value["text"], value["label"], value["start"], value["end"])


def verify_inputs(cases: dict[str, Any], sources: dict[str, Any]) -> None:
    """Every excerpt is the verbatim text its two manifests claim, and every
    registered anchor really sits at the offsets it names.

    Carried unchanged from the runner this example shipped before the evidence
    moved to Hugging Face. An excerpt whose text was rewritten fails here even
    when the digest recorded beside it was rewritten to match, because the
    canonical digest in sources.json has to agree as well.
    """
    if cases.get("integrity_policy", {}).get("synthetic_or_paraphrased_evidence") is not False:
        raise SystemExit("inputs/cases.json must reject synthetic or paraphrased evidence")
    if sources.get("synthetic_or_paraphrased_evidence") is not False:
        raise SystemExit("inputs/sources.json must reject synthetic evidence")
    if cases.get("model") != MODEL:
        raise SystemExit(f"inputs/cases.json names {cases.get('model')}, expected {MODEL}")

    canonical_excerpts = sources.get("excerpts")
    if not isinstance(canonical_excerpts, dict) or set(canonical_excerpts) != set(cases["cases"]):
        raise SystemExit("inputs/sources.json does not carry one canonical excerpt per case")
    known = set(sources["sources"])

    for case_id, case in cases["cases"].items():
        source = case["source"]
        canonical = canonical_excerpts[case_id]
        if source["source_id"] not in known:
            raise SystemExit(f"{case_id}: no source {source['source_id']} in sources.json")
        if source["source_id"] != canonical["source_id"] or source["locator"] != canonical["locator"]:
            raise SystemExit(f"{case_id}: the canonical excerpt names a different source or locator")
        if source["text"] != case["text"]:
            raise SystemExit(f"{case_id}: the case text and its source text differ")
        actual = sha256_text(case["text"])
        if actual != source["sha256"]:
            raise SystemExit(f"{case_id}: excerpt does not match the digest recorded beside it")
        if actual != canonical["sha256"]:
            raise SystemExit(f"{case_id}: excerpt does not match the canonical one in sources.json")

        labels = case["labels"]
        if len(set(labels)) != len(labels):
            raise SystemExit(f"{case_id}: a label is requested twice")
        seen: set[tuple[str, str, int, int]] = set()
        for index, anchor in enumerate(case["required_anchors"]):
            if set(anchor) != set(ANCHOR_FIELDS):
                raise SystemExit(f"{case_id}: anchor {index} does not carry exactly {ANCHOR_FIELDS}")
            if anchor["label"] not in labels:
                raise SystemExit(f"{case_id}: anchor {index} uses a label the request did not ask for")
            start, end = anchor["start"], anchor["end"]
            if type(start) is not int or type(end) is not int or start < 0 or end <= start or end > len(case["text"]):
                raise SystemExit(f"{case_id}: anchor {index} has offsets outside the text")
            if case["text"][start:end] != anchor["text"]:
                raise SystemExit(f"{case_id}: anchor {index} is not the text at its own offsets")
            key = anchor_key(anchor)
            if key in seen:
                raise SystemExit(f"{case_id}: anchor {index} is registered twice")
            seen.add(key)


def scored_calls(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The recorded calls, refusing anything that failed or repeats a case."""
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


def entities(case_id: str, case: dict[str, Any], body: dict[str, Any]) -> list[dict[str, Any]]:
    """The returned spans, with the offset contract the runner applied.

    Fail-closed: an unrequested label, a non-integer or out-of-range offset, a
    span whose text is not what its own offsets say, or a score outside [0, 1]
    all stop the run rather than producing a figure.
    """
    if body.get("model") != MODEL:
        raise SystemExit(f"{case_id}: response names model {body.get('model')}")
    if body.get("id") != f"{case_id}-source":
        raise SystemExit(f"{case_id}: response names item {body.get('id')}")
    rows = body.get("entities")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"{case_id}: response carries no entity list")

    allowed = set(case["labels"])
    text = case["text"]
    for index, entity in enumerate(rows):
        if entity.get("label") not in allowed:
            raise SystemExit(f"{case_id}: span {index} uses unrequested label {entity.get('label')!r}")
        start, end = entity.get("start"), entity.get("end")
        if type(start) is not int or type(end) is not int:
            raise SystemExit(f"{case_id}: span {index} has non-integer offsets")
        if start < 0 or end <= start or end > len(text):
            raise SystemExit(f"{case_id}: span {index} has offsets outside the text")
        if text[start:end] != entity.get("text"):
            raise SystemExit(f"{case_id}: span {index} is not the text at its own offsets")
        score = entity.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise SystemExit(f"{case_id}: span {index} has a non-numeric score")
        if score < 0 or score > 1:
            raise SystemExit(f"{case_id}: span {index} scores {score}, outside [0, 1]")
    return rows


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

    totals = {"spans": 0, "anchors": 0, "matched": 0}
    unreturned: dict[str, list[str]] = {}
    failures: list[str] = []
    print(f"{len(cases['cases'])} cases, model {MODEL}\n")
    for case_id, case in cases["cases"].items():
        spans = entities(case_id, case, calls[case_id]["response"]["body"])
        observed = {anchor_key(span) for span in spans}
        anchors = case["required_anchors"]
        matched = sum(1 for anchor in anchors if anchor_key(anchor) in observed)
        totals["spans"] += len(spans)
        totals["anchors"] += len(anchors)
        totals["matched"] += matched
        empty = sorted(set(case["labels"]) - {span["label"] for span in spans})
        if empty:
            unreturned[case_id] = empty
        print(
            f"  {case_id:<24} {len(case['labels'])} labels"
            f"   {len(spans):>2} spans"
            f"   {matched} of {len(anchors)} required anchors matched"
            + (f"   no span for: {', '.join(empty)}" if empty else "")
        )
        if len(spans) != EXPECTED_SPANS[case_id]:
            failures.append(f"{case_id}: page counts {EXPECTED_SPANS[case_id]} spans, recordings hold {len(spans)}")
        if matched != len(anchors):
            missed = [anchor for anchor in anchors if anchor_key(anchor) not in observed]
            failures.append(
                f"{case_id}: {len(missed)} required anchors missing: "
                + ", ".join(f"{a['text']!r} ({a['label']} at {a['start']}:{a['end']})" for a in missed)
            )
        if empty != EXPECTED_UNRETURNED.get(case_id, []):
            failures.append(
                f"{case_id}: page reports {EXPECTED_UNRETURNED.get(case_id, [])} unreturned, recordings give {empty}"
            )

    print(
        f"\n{totals['spans']} spans returned across the {len(cases['cases'])} cases, every one inside a requested label"
    )
    print(
        f"{totals['matched']} of {totals['anchors']} required anchors matched at the offsets registered before the run"
    )
    print("every one reproduces its own text from input_text[start:end], or this script would have stopped above")
    print(
        "the page displays 5, 6, 5 and 5 of them and counts the remaining 19, 3, 8 and 2."
        " That cap is a display decision in sie-web, computed there from these same totals."
    )

    if totals["spans"] != EXPECTED_TOTALS["spans"] or totals["anchors"] != EXPECTED_TOTALS["anchors"]:
        failures.append(
            f"totals: page publishes {EXPECTED_TOTALS['spans']} spans and {EXPECTED_TOTALS['anchors']} anchors,"
            f" recordings give {totals['spans']} and {totals['anchors']}"
        )
    if failures:
        sys.stdout.flush()
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nReproduced: 24, 9, 13 and 7 spans, 53 in total, and 28 of 28 required anchors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
