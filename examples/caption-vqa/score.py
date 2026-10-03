#!/usr/bin/env python3
"""Reproduce the /caption-vqa figure from the recorded calls. No API key, no
network, no inference spend.

    python3 fetch.py
    python3 score.py

Prints "10 of 12 answers matched", the figure pinned in PAGE_FIGURE and
re-derived here from the recorded calls.

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
# The pass total over the twelve recorded cases, pinned in this file so that a
# rescore cannot move it quietly.
#
# The docstring above used to call it "the figure the page publishes under its
# proof grid". Where the task page reports a total, and which cases it shows,
# are the page's decisions. Nothing here reads the page, so this file could
# never have checked either, and the page's own SOURCES.md is where both are
# recorded.
PAGE_FIGURE = (10, 12)

# The object ids these files have at the dataset revision fetch.py pins. They
# live HERE, in the repository, and that is the whole point: a digest stored
# inside a file cannot authenticate that file. Every request_sha256 and
# response_sha256 this scorer checks travels inside calls.json, and
# inputs_sha256 travels inside manifest.json, so an editor who changes a
# response and recomputes the digest sitting beside it satisfies all of them.
# Only a value pinned outside the evidence catches that.
#
# These are the object ids HuggingFace publishes for the revision, so a reader
# can check them without running any of this code:
#   curl -s "https://huggingface.co/api/datasets/superlinked/sie-task-evidence/tree/<revision>/caption-vqa?recursive=true"
PINNED_OIDS = {
    "inputs/inputs.json": "46b6371eab6ca67cd7ec20bdf3de133b1a71d034",
    "calls.json": "5f091587ba4f945dbf7c975839683c922779297a",
    "manifest.json": "8166a6aed477c7c79cfdb5dd7335b055ee3a7015",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes.

    SHA-1 is not chosen here for its strength; it is the identifier the dataset
    already exposes, so the pin can be checked against HuggingFace by hand.
    """
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def canonical_sha256(value: Any) -> str:
    """The digest the runner recorded: sorted keys, compact separators."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def verify_files() -> list[str]:
    """Authenticate whole files against the pinned object ids, before parsing.

    This runs first because every other check reads a digest that travels with
    the evidence. A file that is absent is a failure here, not something a
    later check quietly skips.
    """
    problems: list[str] = []
    for name, pinned in PINNED_OIDS.items():
        path = EVIDENCE / name
        if not path.is_file():
            problems.append(f"{name} was not downloaded. Run: python3 fetch.py")
            continue
        oid = git_blob_oid(path.read_bytes())
        if oid != pinned:
            problems.append(f"{name} is {oid}, but this example pins {pinned} at the dataset revision fetch.py uses")
    return problems


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


def unique_by(rows: list[dict[str, Any]], key: str, what: str) -> dict[str, dict[str, Any]]:
    """Index rows by a key, refusing duplicates.

    Not a dict comprehension. A comprehension keeps the LAST row sharing a key
    and `next()` takes the FIRST, so two rows with one id let two checks agree
    with different data while every digest still verifies.
    """
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row[key] in indexed:
            raise SystemExit(f"{what} lists {row[key]!r} twice; refusing to score an ambiguous record")
        indexed[row[key]] = row
    return indexed


def verify_evidence(inputs: dict[str, Any], manifest: dict[str, Any], calls: dict[str, Any]) -> list[str]:
    """Check the bytes before scoring them. Anything unreadable is a failure.

    Four checks. `inputs.json` is hashed against the digest the run recorded.
    Every request and response record is re-digested the way the run digested
    it. Every stored file is hashed against the digest its own entry carries.
    And that entry digest is held against the `image_sha256` the case pinned in
    `inputs.json`, which is the authoritative side: without it an image that
    agrees with the call that sent it passes even when it is not the image the
    case registered, so a swap between two cases would go unnoticed.
    """
    problems: list[str] = []

    inputs_sha = sha256_bytes((EVIDENCE / "inputs" / "inputs.json").read_bytes())
    if inputs_sha != manifest["inputs_sha256"]:
        problems.append(
            f"inputs.json hashes to {inputs_sha}, but the run was recorded against {manifest['inputs_sha256']}"
        )

    pins = unique_by(inputs["cases"], "id", "inputs.json")

    for case in pins.values():
        # A case with no checks would pass on any non-empty answer, because
        # all() over an empty list is True. Absent scoring is a failure, not a
        # free pass.
        if not case.get("checks"):
            problems.append(f"{case['id']}: inputs.json registers no checks, so nothing would score this answer")

    for entry in calls["calls"]:
        # An entry is looked up below by slug and verified here by case. If the
        # two can disagree, an edit to one field alone gets the evidence
        # validated as one case and the answer scored against another's
        # patterns.
        if entry["slug"] != entry["case"]:
            problems.append(f"{entry['slug']}: recorded against case {entry['case']!r}; the two must name one case")
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

        case = pins.get(entry["slug"])
        if case is None:
            problems.append(f"{entry['slug']}: inputs.json registers no case with this id, so nothing pins its image")
            continue
        # One image per case is what inputs.json registers. Comparing only
        # entry["images"][0] would let a second image ride along unchecked.
        if len(entry["images"]) != 1:
            problems.append(f"{entry['slug']}: {len(entry['images'])} images recorded, but the case pins exactly one")
            continue
        image = entry["images"][0]
        want_name = Path(case["image"]).name
        if Path(image["path"]).name != want_name or image["sha256"] != case["image_sha256"]:
            problems.append(
                f"{entry['slug']}: this call sent {Path(image['path']).name} ({image['sha256'][:12]}), "
                f"but inputs.json pins {want_name} ({case['image_sha256'][:12]})"
            )
    return problems


def main() -> int:
    # Whole files first, against pins that do not travel with them. Nothing is
    # parsed until the bytes are the bytes this example was written against.
    problems = verify_files()
    if problems:
        print("The downloaded evidence is not what this example pins, so nothing was read:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    inputs = load("inputs/inputs.json")
    manifest = load("manifest.json")
    calls = load("calls.json")

    problems = verify_evidence(inputs, manifest, calls)
    if problems:
        print("The evidence did not verify, so nothing was scored:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    # verify_evidence walks the list; this walks the index. Two duplicate slugs
    # would let them read different rows, so the index refuses duplicates.
    recorded = unique_by(calls["calls"], "slug", "calls.json")
    passed = 0
    missing: list[str] = []
    rows: list[str] = []

    scored_slugs: set[str] = set()
    for case in inputs["cases"]:
        entry = recorded.get(case["id"])
        if entry is None:
            # Counted and named, never passed over. A scorer that skips what it
            # cannot read prints a clean ratio over a set it did not score.
            missing.append(case["id"])
            continue
        scored_slugs.add(entry["slug"])
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

    unscored = sorted(set(recorded) - scored_slugs)
    if unscored:
        # A recorded call no case reaches is evidence nothing looked at, and it
        # would sit in the file affecting no total. Say so rather than ignore it.
        print(f"{len(unscored)} recorded call(s) matched no case and were NOT scored:", file=sys.stderr)
        for slug in unscored:
            print(f"  {slug}", file=sys.stderr)
        return 1

    want_passed, want_total = PAGE_FIGURE
    if (passed, scored) != (want_passed, want_total):
        print(
            f"\nThis does NOT reproduce the {want_passed} of {want_total} pinned in "
            f"PAGE_FIGURE for {manifest['page']}. Report it rather than adjusting "
            f"either number.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Reproduces the {want_passed} of {want_total} pinned in PAGE_FIGURE for "
        f"{manifest['page']}. It does not check which of the twelve that page "
        f"shows, or where it reports a total; nothing here reads the page."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
