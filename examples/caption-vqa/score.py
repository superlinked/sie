#!/usr/bin/env python3
"""Reproduce the /caption-vqa figure from the recorded calls. No API key, no
network, no inference spend.

    python3 fetch.py
    python3 score.py

Prints "10 of 12 answers matched", the figure the page publishes under its
proof grid.

How a case is scored, exactly as the run recorded it: take the first non-empty
line of the returned text, then apply the regular expressions that inputs.json
registered before the first model call. A check with `must: false` passes when
its pattern does NOT match. A case passes only when every check passes and the
reply was not empty. Scoring is deterministic string matching; no model is
consulted.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
PAGE_FIGURE = (10, 12)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    """The digest the runner recorded: sorted keys, compact separators."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def load(name: str) -> Any:
    path = EVIDENCE / name
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def answer_line(text: str) -> str:
    """The first non-empty line: the one-sentence answer the prompt asks for."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def apply_checks(text: str, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for check in checks:
        matched = re.search(check["pattern"], text, re.IGNORECASE) is not None
        must = check.get("must", True)
        results.append({"label": check["label"], "matched": matched, "passed": matched == must})
    return results


def verify_evidence(inputs: dict[str, Any], manifest: dict[str, Any], calls: dict[str, Any]) -> list[str]:
    """Check the bytes before scoring them. Anything unreadable is a failure."""
    problems: list[str] = []

    inputs_sha = sha256_bytes((EVIDENCE / "inputs" / "inputs.json").read_bytes())
    if inputs_sha != manifest["inputs_sha256"]:
        problems.append(
            f"inputs.json hashes to {inputs_sha}, but the run was recorded against {manifest['inputs_sha256']}"
        )

    for entry in calls["calls"]:
        if canonical_sha256(entry["request"]) != entry["request_sha256"]:
            problems.append(f"{entry['slug']}: the request record does not match its recorded digest")
        if canonical_sha256(entry["response"]) != entry["response_sha256"]:
            problems.append(f"{entry['slug']}: the response record does not match its recorded digest")
        for image in entry["images"]:
            path = EVIDENCE / image["path"]
            if not path.is_file():
                problems.append(f"{entry['slug']}: {image['path']} was not downloaded")
            elif sha256_bytes(path.read_bytes()) != image["sha256"]:
                problems.append(f"{entry['slug']}: {image['path']} is not the image this call was sent")
    return problems


def main() -> int:
    inputs = load("inputs/inputs.json")
    manifest = load("manifest.json")
    calls = load("calls.json")

    problems = verify_evidence(inputs, manifest, calls)
    if problems:
        print("The evidence did not verify, so nothing was scored:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    recorded = {entry["slug"]: entry for entry in calls["calls"]}
    passed = 0
    missing: list[str] = []
    rows: list[str] = []

    for case in inputs["cases"]:
        entry = recorded.get(case["id"])
        if entry is None:
            # Counted and named, never passed over. A scorer that skips what it
            # cannot read prints a clean ratio over a set it did not score.
            missing.append(case["id"])
            continue
        body = entry["response"].get("body")
        text = body.get("text") or "" if isinstance(body, dict) else ""
        answer = answer_line(text)
        results = apply_checks(answer, case["checks"])
        ok = bool(answer) and all(result["passed"] for result in results)
        passed += ok
        labels = ", ".join(result["label"] for result in results if not result["passed"])
        rows.append(f"  {'match' if ok else 'MISS '}  {case['id']}: {answer}" + (f"   [{labels}]" if labels else ""))

    scored = len(inputs["cases"]) - len(missing)
    for row in rows:
        print(row)
    print(f"\n{passed} of {scored} answers matched")

    if missing:
        print(
            f"{len(missing)} case(s) NOT SCORED, so the figure above covers only {scored} of "
            f"{len(inputs['cases'])} cases:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    want_passed, want_total = PAGE_FIGURE
    if (passed, scored) != (want_passed, want_total):
        print(
            f"\nThis does NOT match the {want_passed} of {want_total} published on "
            f"{manifest['page']}. Report it rather than adjusting either number.",
            file=sys.stderr,
        )
        return 1
    print(f"Matches the {want_passed} of {want_total} published on {manifest['page']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
