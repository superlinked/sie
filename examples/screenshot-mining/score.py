#!/usr/bin/env python3
"""Reproduce the /screenshot-mining figure from the recorded calls. No API key,
no network, no inference spend.

    python3 fetch.py
    python3 score.py

Prints "328 of 335 values matched the screen", the figure the page publishes,
and the three per-screen figures pinned in PAGE_PER_SCREEN.

`expected` in inputs.json was read off each screenshot at full resolution and
committed before any model run. This compares each recorded reply with it using
the rules inputs.json records under "scoring". It never edits an expected value.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"

# The object ids these files have at the dataset revision fetch.py pins. They
# live HERE, in the repository, and that is the whole point: a digest stored
# inside a file cannot authenticate that file. Every recorded_sha256 this
# scorer checks travels inside calls.json, and inputs_sha256 travels inside
# manifest.json, so an editor who changes a response and recomputes the digest
# sitting beside it satisfies all of them. Only a value pinned outside the
# evidence catches that.
#
# These are the object ids HuggingFace publishes for the revision, so a reader
# can check them without running any of this code:
#   curl -s "https://huggingface.co/api/datasets/superlinked/sie-task-evidence/tree/<revision>/screenshot-mining?recursive=true"
PINNED_OIDS = {
    "inputs/inputs.json": "07754353a0ae46b1d6d5431ed61af08feb9ff4ee",
    "calls.json": "e26085a47ab2d131eb5dbae51a92b665a1c9f6a2",
    "manifest.json": "1e09b5f87b2ed28fb3b4359fbf8d4420cebf9bed",
}

PAGE_FIGURE = (328, 335)
PAGE_SCREENS = 12
# Three per-screen figures the page publishes, checked against the recording so
# that a rescore cannot move a published number quietly.
#
# This block used to say which screens the proof grid draws, and in what order.
# That claim was wrong twice inside two days: once naming four screens, then
# naming three from a page change that had not shipped. Both times every run
# stayed green, because this script cannot see the page and never could. It is
# gone rather than corrected a third time. Which screens the page draws is the
# page's decision and the page's SOURCES.md records it.
#
# What survives is what the recording can settle: these figures, PAGE_FIGURE
# and PAGE_SCREENS. Nothing here was rescored in any of it, and every one of
# the twelve screens is scored and printed below whether the page draws it or
# not.
PAGE_PER_SCREEN = {
    "argocd-applicationsets": (32, 32),
    "kubernetes-dashboard-node": (26, 26),
    "airflow-dag-list": (31, 31),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes.

    SHA-1 is not chosen here for its strength; it is the identifier the dataset
    already exposes, so the pin can be checked against HuggingFace by hand.
    """
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


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


def canonical_sha256(value: Any) -> str:
    """The digest the runner recorded: sorted keys, compact separators."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def load(name: str) -> Any:
    path = EVIDENCE / name
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def norm_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("−", "-").replace("–", "-").replace("…", "...")
    return "".join(text.split()).casefold()


def leaf_equal(expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(expected), float(actual), rel_tol=0, abs_tol=1e-9)
        )
    if isinstance(expected, str):
        return isinstance(actual, str) and norm_text(expected) == norm_text(actual)
    raise TypeError(f"Unsupported expected leaf {expected!r}")


def compare(
    expected: Any,
    actual: Any,
    path: str,
    rules: dict[str, Any],
    out: list[dict[str, Any]],
    missing: bool = False,
) -> None:
    """Record one check per expected leaf. Lists compare by position unless a
    rule names them a set (matched by key) or an unordered list of strings.

    `missing` marks everything under a row the response never returned: per the
    registered rules a missing row fails every leaf in it, so an expected null
    is not allowed to "match" a field that is simply absent."""
    if isinstance(expected, dict):
        source = actual if isinstance(actual, dict) else {}
        absent = missing or not isinstance(actual, dict)
        for key, value in expected.items():
            compare(value, source.get(key), f"{path}.{key}", rules, out, absent)
        return
    if isinstance(expected, list):
        rule = rules.get(path, {})
        items = actual if isinstance(actual, list) else []
        absent = missing or not isinstance(actual, list)
        if rule.get("match") == "string-set":
            got = sorted(norm_text(v) for v in items if isinstance(v, str))
            want = sorted(norm_text(v) for v in expected)
            out.append({"path": path, "expected": expected, "actual": actual, "ok": not absent and got == want})
            return
        if rule.get("match") == "by-key":
            key = rule["key"]
            used: set[int] = set()
            for row in expected:
                found = None
                for candidate_index, candidate in enumerate(items):
                    if candidate_index in used or not isinstance(candidate, dict):
                        continue
                    if leaf_equal(row[key], candidate.get(key)):
                        found = candidate_index
                        break
                if found is not None:
                    used.add(found)
                compare(
                    row,
                    items[found] if found is not None else None,
                    f"{path}[{row[key]}]",
                    rules,
                    out,
                    absent or found is None,
                )
            extras = [items[i] for i in range(len(items)) if i not in used]
        else:
            for index, row in enumerate(expected):
                compare(
                    row,
                    items[index] if index < len(items) else None,
                    f"{path}[{index}]",
                    rules,
                    out,
                    absent or index >= len(items),
                )
            extras = items[len(expected) :]
        if extras:
            out.append(
                {"path": f"{path}+extra", "expected": None, "actual": extras, "ok": False, "extra_rows": len(extras)}
            )
        return
    out.append(
        {"path": path, "expected": expected, "actual": actual, "ok": not missing and leaf_equal(expected, actual)}
    )


def parse_generation(value: Any) -> tuple[Any, str | None]:
    """Parse the model's JSON out of a recorded response value."""
    if not isinstance(value, dict):
        return None, "no JSON body"
    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "empty text"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as error:
        return None, f"invalid JSON: {error}"


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

    Four checks, and one that is deliberately not run. `inputs.json` is hashed
    against the digest the run recorded. Every response record is re-digested
    the way the run digested it. Every stored image is hashed against the
    `$payload.sha256` of the request that sent it, which is the check the
    recorded `$payload` note asks a reader to perform. And that `$payload` is
    held against the `file_name` and `image_sha256` the case pinned in
    `inputs.json`, which is the authoritative side: without it an image that
    agrees with the call that sent it passes even when it is not the image the
    case registered, so a swap between two screens would go unnoticed.

    Not checked here: `entry_sha256`, the RFC 8785 canonical digest over each
    whole entry. That needs a JSON canonicalizer; the implementation lives in
    superlinked/sie-web beside the fixture and in
    examples/document-to-markdown in this repository, and is not duplicated
    into this example.
    """
    problems: list[str] = []

    inputs_sha = sha256_bytes((EVIDENCE / "inputs" / "inputs.json").read_bytes())
    if inputs_sha != manifest["inputs_sha256"]:
        problems.append(
            f"inputs.json hashes to {inputs_sha}, but the run was recorded against {manifest['inputs_sha256']}"
        )

    pins = unique_by(inputs["cases"], "id", "inputs.json")

    for case in pins.values():
        # The scoring loop selects on role and this scorer sends one body
        # shape. An unrecognised value in either field would drop a screen out
        # of the totals without anything objecting.
        if case.get("role") not in {"proof", "playground"}:
            problems.append(f"{case['id']}: role is {case.get('role')!r}, which this scorer does not count")
        for call in case["calls"]:
            if call.get("kind") != "qwen-schema":
                problems.append(
                    f"{case['id']}/{call['call']}: kind is {call.get('kind')!r}, which this scorer cannot score"
                )

    for entry in calls["calls"]:
        # An entry is looked up below by the slug built from a case id and a
        # call name, and verified here by its own case field. If the two can
        # disagree, an edit to slug alone gets the evidence validated as one
        # screen and the reply scored against another's expected values.
        if entry["slug"] != f"{entry['case']}__{entry['call']}":
            problems.append(
                f"{entry['slug']}: recorded as case {entry['case']!r} call {entry['call']!r}; "
                f"the slug must be exactly {entry['case']}__{entry['call']}"
            )
        response = entry["response"]
        # The digest the run recorded covers the response record as it was
        # written: status, headers and body.
        rebuilt = {"status": entry["http_status"], "headers": response["http_headers"], "body": response["value"]}
        if canonical_sha256(rebuilt) != response["recorded_sha256"]:
            problems.append(f"{entry['slug']}: the response record does not match its recorded digest")

        case = pins.get(entry["case"])
        if case is None:
            problems.append(
                f"{entry['slug']}: inputs.json registers no case {entry['case']!r}, so nothing pins its image"
            )
            continue

        payloads = [image["$payload"] for image in entry["request"]["body"].get("images") or [] if "$payload" in image]
        # One screenshot per call is what inputs.json registers. Checking only
        # the first would let a second image ride along unchecked, and a call
        # carrying none would otherwise pass this loop by doing nothing.
        if len(payloads) != 1:
            problems.append(f"{entry['slug']}: {len(payloads)} image payloads recorded, but the case pins exactly one")
            continue
        payload = payloads[0]

        if payload["file_name"] != case["file_name"] or payload["sha256"] != case["image_sha256"]:
            problems.append(
                f"{entry['slug']}: this call sent {payload['file_name']} ({payload['sha256'][:12]}), "
                f"but inputs.json pins {case['file_name']} ({case['image_sha256'][:12]})"
            )

        path = EVIDENCE / "inputs" / "images" / payload["file_name"]
        if not path.is_file():
            problems.append(f"{entry['slug']}: inputs/images/{payload['file_name']} was not downloaded")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != payload["sha256"] or len(data) != payload["bytes"]:
            problems.append(
                f"{entry['slug']}: inputs/images/{payload['file_name']} is not the image this call was sent"
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
    missing: list[str] = []
    per_screen: dict[str, tuple[int, int]] = {}
    proof_passed = proof_total = 0
    other_passed = other_total = 0
    screens = 0

    scored_slugs: set[str] = set()
    for case in inputs["cases"]:
        for call in case["calls"]:
            entry = recorded.get(f"{case['id']}__{call['call']}")
            if entry is None:
                # Counted and named, never passed over: a scorer that quietly
                # skips what it cannot read prints a clean ratio over a set it
                # did not score.
                missing.append(f"{case['id']}__{call['call']}")
                continue
            scored_slugs.add(entry["slug"])
            parsed, error = parse_generation(entry["response"]["value"])
            checks: list[dict[str, Any]] = []
            compare(call["expected"], parsed, "$", call.get("rules", {}), checks)
            ok = sum(1 for check in checks if check["ok"])
            if case.get("role") == "proof":
                proof_passed += ok
                proof_total += len(checks)
                per_screen[case["id"]] = (ok, len(checks))
                screens += 1
            else:
                other_passed += ok
                other_total += len(checks)
            note = f"  ({error})" if error else ""
            print(f"  {case['id']}/{call['call']}: {ok} of {len(checks)}{note}")

    if missing:
        print(f"{len(missing)} call(s) NOT SCORED, so no figure below covers the recorded run:", file=sys.stderr)
        for slug in missing:
            print(f"  missing {slug}", file=sys.stderr)
        return 1

    unscored = sorted(set(recorded) - scored_slugs)
    if unscored:
        # A recorded call no case reaches is evidence nothing looked at, and it
        # would sit in the file affecting no total. Say so rather than ignore it.
        print(f"{len(unscored)} recorded call(s) matched no case and were NOT scored:", file=sys.stderr)
        for slug in unscored:
            print(f"  {slug}", file=sys.stderr)
        return 1

    print(
        f"\nAcross all {screens} recorded screens, {proof_passed} of {proof_total} values matched the screen "
        f"and {proof_total - proof_passed} did not"
    )
    print(
        f"The playground call adds {other_passed} of {other_total}, for "
        f"{proof_passed + other_passed} of {proof_total + other_total} over every recorded call"
    )

    want_passed, want_total = PAGE_FIGURE
    failures: list[str] = []
    if (proof_passed, proof_total, screens) != (want_passed, want_total, PAGE_SCREENS):
        failures.append(
            f"headline: got {proof_passed} of {proof_total} over {screens} screens, "
            f"page publishes {want_passed} of {want_total} over {PAGE_SCREENS}"
        )
    for slug, figure in PAGE_PER_SCREEN.items():
        got = per_screen.get(slug)
        if got != figure:
            failures.append(f"{slug}: got {got}, page publishes {figure[0]} of {figure[1]}")
    if failures:
        print(
            f"\nThis does NOT reproduce what {manifest['page']} publishes. "
            "Report it rather than adjusting either number.",
            file=sys.stderr,
        )
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    # What this line may claim, found by tampering with it rather than by
    # reading it. Swapping one entry of PAGE_PER_SCREEN for a screen the page
    # does not draw, with its correct figures, leaves the run green: the loop
    # above checks the numbers of whatever slugs it is handed, not that those
    # slugs are the page's. That is the hole that let this file describe a grid
    # from an unmerged PR while every run passed. Nothing here can reach the
    # page, so the wording says what was checked instead of implying more.
    print(
        f"The {len(PAGE_PER_SCREEN)} per-screen figures below PAGE_PER_SCREEN match "
        f"the recording, as does the {want_passed} of {want_total} published on "
        f"{manifest['page']}."
    )
    print("Not checked here: that those are the screens the page currently draws. Its SOURCES.md lists them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
