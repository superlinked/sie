#!/usr/bin/env python3
"""Reproduce the /rerank page's published figures from the recorded calls.

    python3 fetch.py
    python3 score.py

Standard library only. No API key, no network, no inference spend.

Each of the 24 questions was sent five times against the same four candidates:
once with no instruction, and once under each of four relevance rules. The query
names the subject of a rulemaking and never says whether the rule is in force,
so only the instruction can choose between a regulation already adopted and a
proposal on the same subject.

Two figures come out of that, per arm, over the 24 cases:

  exact       how often rank 1 is the one document the rule asks for: the right
              document type AND the docket the question is about.
  by type     how often rank 1 is a document of the type the rule asks for,
              whichever docket it came from.

The expected winner for each case and rule is pinned in inputs/cases.json, which
derives it mechanically from the Federal Register's own document type. Nothing
here decides which document is in force.

Absent data is a failure, never a skip. A case with no recorded call, a call
with no response, or a response that scored a different candidate set than the
inputs list, fails the run and is named.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
BASELINE = "none"

# Which candidate role each rule asks for, and which document type that role is.
ASKS_FOR = {
    "in-force": ("asked-about-final", "Rule"),
    "proposed": ("asked-about-proposal", "Proposed Rule"),
    "in-force-positive": ("asked-about-final", "Rule"),
    "proposed-positive": ("asked-about-proposal", "Proposed Rule"),
}

# The two arms https://superlinked.com/rerank displays. The other two are the
# pre-registered wording test recorded alongside them; calls.json holds all
# four, so a reader can recompute the pair that is not displayed.
PUBLISHED_ARMS = ("in-force-positive", "proposed-positive")

# Published figures, pinned in the committed source so the scorer compares what
# it derives from the recording against something the recording cannot move.
# Editing the fetched evidence alone will not satisfy this.
PUBLISHED = {
    "cases": 24,
    "calls": 120,
    "none/exact/asked-about-final": 23,
    "none/exact/asked-about-proposal": 0,
    "none/type/Rule": 24,
    "none/type/Proposed Rule": 0,
    "in-force-positive/exact": 21,
    "in-force-positive/type": 23,
    "proposed-positive/exact": 16,
    "proposed-positive/type": 17,
    "in-force/exact": 16,
    "proposed/exact": 20,
    "top1_differs_between_published_rules": 16,
}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def main_with(data_dir: Path, emit: str | None = None) -> int:
    """Score one fetched evidence directory. Returns the process exit status."""
    payload = load(data_dir / "inputs/cases.json")
    recording = load(data_dir / "calls.json")
    # Required, not optional. This used to read `if manifest_path.exists() else {}`,
    # which made the revision check below vanish when the manifest was absent, so
    # deleting one file was enough to have a recording from other weights scored.
    # Missing evidence is a failure, never a licence to skip the check over it.
    manifest = load(data_dir / "manifest.json")

    if manifest.get("model_revision") != MODEL_REVISION:
        raise SystemExit(
            f"This recording was made against model revision {manifest.get('model_revision')!r}, "
            f"and this example publishes figures for {MODEL_REVISION!r}. Different weights produce "
            "different scores, so the recording is not scored."
        )

    cases = {case["id"]: case for case in payload["cases"]}
    arms = [BASELINE, *payload["rules"]]
    calls = {call["id"]: call for call in recording["calls"]}

    failures: list[str] = []
    top: dict[tuple[str, str], dict[str, str]] = {}
    for case_id, case in cases.items():
        roles = {c["id"]: (c["role"], c["type"]) for c in case["candidates"]}
        for arm in arms:
            identifier = f"{case_id}/{arm}"
            call = calls.get(identifier)
            if call is None:
                failures.append(f"{identifier}: no recorded call")
                continue
            if call.get("status") != 200 or not call.get("response"):
                failures.append(f"{identifier}: recorded status {call.get('status')!r}, no result to score")
                continue
            if call["model"] != MODEL:
                failures.append(f"{identifier}: recorded against {call['model']}, not {MODEL}")
                continue
            # Per call, not just once per run: the manifest states one deployment
            # and every call has to have come from it. Compared against the
            # manifest's `deployment_revision` and NOT against MODEL_REVISION,
            # because the SDK's `last_model_revision` is the SIE deployment
            # digest that models served together share, not the HuggingFace
            # revision of these weights. Checking it against MODEL_REVISION
            # would reject all 120 of this example's own calls.
            served = (call.get("recorded") or {}).get("model_revision")
            if served != manifest.get("deployment_revision"):
                failures.append(
                    f"{identifier}: served by deployment {served!r}, and the manifest "
                    f"records {manifest.get('deployment_revision')!r}"
                )
                continue
            scored = {entry["item_id"] for entry in call["response"]["body"]["scores"]}
            if scored != set(roles):
                failures.append(f"{identifier}: scored {len(scored)} candidates, the inputs list {len(roles)}")
                continue
            winner = sorted(call["response"]["body"]["scores"], key=lambda s: s["rank"])[0]["item_id"]
            role, kind = roles[winner]
            top[(case_id, arm)] = {"id": winner, "role": role, "type": kind}

    if failures:
        print(f"{len(failures)} of {len(cases) * len(arms)} case/arm pairs could not be scored:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print("No figure is derived from an incomplete recording.", file=sys.stderr)
        return 1

    n = len(cases)
    derived: dict[str, Any] = {"cases": n, "calls": len(recording["calls"])}
    for role, kind in (("asked-about-final", "Rule"), ("asked-about-proposal", "Proposed Rule")):
        derived[f"{BASELINE}/exact/{role}"] = sum(top[(c, BASELINE)]["role"] == role for c in cases)
        derived[f"{BASELINE}/type/{kind}"] = sum(top[(c, BASELINE)]["type"] == kind for c in cases)
    for arm in payload["rules"]:
        role, kind = ASKS_FOR[arm]
        derived[f"{arm}/exact"] = sum(top[(c, arm)]["role"] == role for c in cases)
        derived[f"{arm}/type"] = sum(top[(c, arm)]["type"] == kind for c in cases)
    a, b = PUBLISHED_ARMS
    derived["top1_differs_between_published_rules"] = sum(top[(c, a)]["id"] != top[(c, b)]["id"] for c in cases)

    print(f"model   {MODEL} @ {MODEL_REVISION}")
    print(f"corpus  {n} questions, four Federal Register abstracts each, {len(recording['calls'])} calls recorded")
    print()
    print("With no relevance rule in the call, the document at rank 1 was")
    print(f"  a regulation already in force        {derived[f'{BASELINE}/type/Rule']}/{n}")
    print(f"  a proposal not yet adopted           {derived[f'{BASELINE}/type/Proposed Rule']}/{n}")
    print()
    print(f"{'rule stated in the call':<36} {'right document':>14} {'right kind':>12}")
    for arm in payload["rules"]:
        mark = "  (published)" if arm in PUBLISHED_ARMS else ""
        print(f"  {arm:<34} {derived[f'{arm}/exact']:>9}/{n}   {derived[f'{arm}/type']:>7}/{n}{mark}")
    print()
    print(f"the two published rules put a different document first in "
          f"{derived['top1_differs_between_published_rules']}/{n} questions")

    if emit:
        Path(emit).write_text(
            json.dumps({"derived": derived,
                        "per_case": {c: {a: top[(c, a)] for a in arms} for c in cases}},
                       indent=1) + "\n",
            encoding="utf-8")
        print(f"\nwrote {emit}")

    mismatched = [k for k, v in PUBLISHED.items() if derived.get(k) != v]
    if mismatched:
        for key in mismatched:
            print(f"MISMATCH {key}: recording gives {derived.get(key)}, the page publishes {PUBLISHED[key]}",
                  file=sys.stderr)
        return 1
    print("\nEvery published figure matches the recording.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--emit", help="write the derived figures as JSON to this path")
    args = parser.parse_args()
    return main_with(Path(args.data), args.emit)


if __name__ == "__main__":
    sys.exit(main())
