#!/usr/bin/env python3
"""Reproduce the /detect figure from the recorded calls. No API key, no network,
no inference spend.

    python3 fetch.py
    python3 score.py

Prints "55 of 57 returned boxes sit on the object your agent asked for", the
figure the page publishes, and the per-photo boxes-per-label figures pinned in
PAGE_PER_PHOTO.

The page states no recall figure anywhere. The numbers it used to print are
still computed here and checked against the results table in the page's
SOURCES.md, which is where a reader finds them now.

Two sides that the same edit cannot move together:

- the recorded Grounding DINO responses, which say what boxes came back;
- box-review.json, one verdict per box, assigned by hand after the run.

score.py matches every reviewed box against a box actually in the response, by
label, score and rectangle, and fails if any box is reviewed twice, left
unreviewed, or is not in the response at all. Only then does it count anything.
The hand counts it scores against were registered before the run, in
inputs/inputs.json, and nothing here can change them.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"

# What https://superlinked.com/detect publishes. score.py exits non-zero if it
# computes anything else. If that happens, report it: it means the page or the
# evidence is wrong, and neither number should be quietly adjusted to agree.
PAGE_HEADLINE = (55, 57)
PAGE_DUPLICATES = 3
PAGE_WRONG = 2
PAGE_PHOTOS = 9
# Recall, which the page states nowhere. Its SOURCES.md prints it in the results
# table, so this pair is checked against that file rather than against the page.
SOURCES_COVERAGE = (52, 88)

# Four per-photo figures: the label sent, and the boxes that came back carrying
# it. Pinned so that a rescore cannot move one quietly.
#
# This block used to say which photographs the page displays, in which surface,
# and it carried a PAGE_DISPLAYED count and named the hero and playground
# photographs. Every one of those was a claim about a page this script cannot
# reach, and it was already wrong before this file was last touched: it named
# six proof cards at a time the page had shown three for weeks, and every run
# stayed green throughout, because nothing here can tell whether the slugs it
# was handed are the ones the page draws. Which photographs the page displays is
# the page's decision and its SOURCES.md records it.
#
# What survives is what the recording settles: these figures, PAGE_HEADLINE,
# PAGE_PHOTOS, PAGE_DUPLICATES, PAGE_WRONG and SOURCES_COVERAGE. All nine
# photographs are scored and printed below whether the page draws them or not.
PAGE_PER_PHOTO = {
    "sauce-shelf": {"sale sign": 10},
    "food-box-floor": {"safety vest": 9},
    "fulfillment-tour": {"safety vest": 7},
    "hangar-pallet-jacks": {"pallet jack": 4},
}

DETECTION_MODEL = "IDEA-Research/grounding-dino-base"
VERDICTS = ("hit", "duplicate", "wrong")


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


def box_key(item: dict[str, Any]) -> tuple[str, float, tuple[int, ...]]:
    """A box's identity: what it says, how sure it is, and where it sits.

    Compared as identities rather than counted, so a swapped or duplicated box
    cannot pass by keeping the total intact.
    """
    return (item["label"], item["score"], tuple(item["bbox"]))


def verify_evidence(manifest: dict[str, Any], calls: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    """Check the bytes before scoring them. Anything unreadable is a failure.

    Missing input fails the run; it is never passed over, because a scorer that
    skips what it cannot read prints a clean ratio over a set it did not score.
    """
    problems: list[str] = []

    # Checked for existence before it is opened, so an absent file is a
    # reported failure rather than a traceback, and does not depend on main()
    # happening to have loaded it first.
    inputs_path = EVIDENCE / "inputs" / "inputs.json"
    if not inputs_path.is_file():
        problems.append(
            "inputs/inputs.json was not downloaded. It holds the labels sent and the hand counts every figure "
            "below is scored against; run: python3 fetch.py"
        )
    else:
        inputs_sha = sha256_bytes(inputs_path.read_bytes())
        if inputs_sha != manifest["inputs_sha256"]:
            problems.append(
                f"inputs.json hashes to {inputs_sha}, but the run was recorded against {manifest['inputs_sha256']}"
            )

    pinned = {case["id"]: case for case in inputs["cases"]}
    for entry in calls["calls"]:
        if canonical_sha256(entry["request"]) != entry["request_sha256"]:
            problems.append(f"{entry['slug']}: the request record does not match its recorded digest")
        if canonical_sha256(entry["response"]) != entry["response_sha256"]:
            problems.append(f"{entry['slug']}: the response record does not match its recorded digest")
        if entry["http_status"] != 200:
            problems.append(f"{entry['slug']}: recorded HTTP {entry['http_status']}")

        case = pinned.get(entry["case"])
        if case is None:
            problems.append(f"{entry['slug']}: no case {entry['case']} in inputs.json")
            continue
        # The image this call was sent, checked against both the pin registered
        # before the run and the descriptor the request itself carries.
        sent = entry["request"]["body"]["items"][0]["images"][0]["data"]
        path = EVIDENCE / "inputs" / "images" / sent["file"]
        if not path.is_file():
            problems.append(f"{entry['slug']}: inputs/images/{sent['file']} was not downloaded")
            continue
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if digest != sent["sha256"] or len(data) != sent["bytes"]:
            problems.append(f"{entry['slug']}: inputs/images/{sent['file']} is not the image this call was sent")
        if digest != case["image_sha256"]:
            problems.append(f"{entry['slug']}: inputs/images/{sent['file']} is not the image inputs.json pins")

    recorded = {(entry["case"], entry["call"]) for entry in calls["calls"]}
    if len(recorded) != len(calls["calls"]):
        problems.append("calls.json holds two entries for the same case and call")
    for case in inputs["cases"]:
        for call in ("owlv2", "grounding-dino"):
            if (case["id"], call) not in recorded:
                problems.append(f"{case['id']}__{call}: registered in inputs.json but not recorded")
    return problems


def bind_review(review: dict[str, Any], calls: dict[str, Any]) -> list[str]:
    """Every reviewed box is a box the model actually returned, and every box
    the model returned was reviewed exactly once."""
    problems: list[str] = []
    responses = {
        entry["case"]: entry["response"]
        for entry in calls["calls"]
        if entry["call"] == "grounding-dino"
    }
    seen: set[str] = set()
    for case in review["cases"]:
        if case["id"] in seen:
            problems.append(f"{case['id']}: reviewed twice in box-review.json")
            continue
        seen.add(case["id"])
        response = responses.get(case["id"])
        if response is None:
            problems.append(f"{case['id']}: reviewed, but no grounding-dino call was recorded for it")
            continue
        returned = response["body"]["items"][0]["objects"]
        pool: dict[tuple[str, float, tuple[int, ...]], int] = {}
        for item in returned:
            pool[box_key(item)] = pool.get(box_key(item), 0) + 1
        for detection in case["detections"]:
            if detection["verdict"] not in VERDICTS:
                problems.append(f"{case['id']}: verdict {detection['verdict']!r} is not one this scorer knows")
            key = box_key(detection)
            if pool.get(key, 0) < 1:
                problems.append(
                    f"{case['id']}: a reviewed box ({detection['label']} {detection['score']} "
                    f"{list(detection['bbox'])}) is not in the recorded response"
                )
                continue
            pool[key] -= 1
        left = sum(pool.values())
        if left:
            problems.append(f"{case['id']}: {left} returned box(es) were never reviewed")
    for case_id in responses:
        if case_id not in seen:
            problems.append(f"{case_id}: recorded, but box-review.json never reviews it")
    return problems


def main() -> int:
    inputs = load("inputs/inputs.json")
    manifest = load("manifest.json")
    calls = load("calls.json")
    review = load("box-review.json")

    problems = verify_evidence(manifest, calls, inputs) + bind_review(review, calls)
    if problems:
        print("The evidence did not verify, so nothing was scored:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    counts = {case["id"]: case["expected_counts"] for case in inputs["cases"]}
    boxes = on_object = wrong = duplicates = found = counted = 0
    per_photo: dict[str, dict[str, tuple[int, int]]] = {}
    failures: list[str] = []

    for case in review["cases"]:
        registered = counts[case["id"]]
        # The label a box carries comes back lower-cased by the model, so it is
        # matched to the registered label without regard to case.
        by_label: dict[str, int] = {}
        for detection in case["detections"]:
            boxes += 1
            if detection["verdict"] == "wrong":
                wrong += 1
                continue
            on_object += 1
            if detection["verdict"] == "duplicate":
                duplicates += 1
                continue
            found += 1
            match = next((name for name in registered if name.casefold() == detection["label"].casefold()), None)
            if match is None:
                failures.append(f"{case['id']}: a box labelled {detection['label']!r} is not a registered label")
                continue
            by_label[match] = by_label.get(match, 0) + 1

        lines: dict[str, tuple[int, int]] = {}
        for label, number in registered.items():
            counted += number
            hits = by_label.get(label, 0)
            lines[label] = (hits, number)
            # `found` in the review is a hand figure. It has to equal the boxes
            # this scorer counted as hits, or the two layers disagree.
            stated = next((row["found"] for row in case["labels"] if row["label"] == label), None)
            if stated != hits:
                failures.append(
                    f"{case['id']}/{label}: box-review states {stated} found, the verdicts give {hits}"
                )
            registered_count = next((row["counted"] for row in case["labels"] if row["label"] == label), None)
            if registered_count != number:
                failures.append(
                    f"{case['id']}/{label}: box-review states {registered_count} counted, "
                    f"inputs.json registered {number}"
                )
        per_photo[case["id"]] = lines
        summary = ", ".join(f"{label} {hit} of {total}" for label, (hit, total) in lines.items())
        print(f"  {case['id']}: {summary}")

    photos = len(review["cases"])
    print(
        f"\nAcross all {photos} recorded photos, {on_object} of {boxes} returned boxes sit on the object "
        f"the request asked for and {wrong} do not"
    )
    print(
        f"Those {on_object} boxes cover {found} of {counted} hand-counted objects, "
        f"with {duplicates} more on objects already found"
    )

    want_on, want_boxes = PAGE_HEADLINE
    want_found, want_counted = SOURCES_COVERAGE
    if (on_object, boxes) != PAGE_HEADLINE:
        failures.append(f"headline: got {on_object} of {boxes}, page publishes {want_on} of {want_boxes}")
    if (found, counted) != SOURCES_COVERAGE:
        failures.append(
            f"coverage: got {found} of {counted}, the page's SOURCES.md prints {want_found} of {want_counted}"
        )
    if duplicates != PAGE_DUPLICATES:
        failures.append(f"duplicates: got {duplicates}, page publishes {PAGE_DUPLICATES}")
    if wrong != PAGE_WRONG:
        failures.append(f"wrong object: got {wrong}, page publishes {PAGE_WRONG}")
    if photos != PAGE_PHOTOS:
        failures.append(f"photos: got {photos}, page publishes {PAGE_PHOTOS}")

    # The four per-photo figures, and the property that makes a boxes count the
    # same number as a found count on each: every box these four returned is a
    # first box on the object its label names. Read from the hand verdicts, not
    # from the figures, so the two cannot agree by construction.
    for case_id, expected in PAGE_PER_PHOTO.items():
        got = {label: hits for label, (hits, _counted) in per_photo.get(case_id, {}).items()}
        if got != expected:
            failures.append(f"{case_id}: got {got}, this example pins {expected}")
        drawn = next((case for case in review["cases"] if case["id"] == case_id), None)
        if drawn is None:
            failures.append(f"{case_id}: pinned above and not in the recorded review")
            continue
        flawed = [d for d in drawn["detections"] if d["verdict"] != "hit"]
        if flawed:
            failures.append(
                f"{case_id}: {len(flawed)} box(es) are not a first box on the object named, "
                "so its boxes count and its found count are not the same number"
            )

    if failures:
        print(
            f"\nThis does NOT reproduce what {manifest['page']} publishes. "
            "Report it rather than adjusting either number.",
            file=sys.stderr,
        )
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    # What this line may claim, found by tampering with it rather than by reading
    # it. Swapping one entry of PAGE_PER_PHOTO for a photograph the page does
    # not display, with its correct figures, leaves the run green: the loop above
    # checks the numbers of whatever slugs it is handed. That is the hole that
    # let this file describe six proof cards for weeks after the page had three.
    # Nothing here can reach the page, so the wording says what was checked.
    print(
        f"Matches the {want_on} of {want_boxes} published on {manifest['page']}, the "
        f"{len(PAGE_PER_PHOTO)} per-photo figures below PAGE_PER_PHOTO, and the "
        f"{want_found} of {want_counted} in its SOURCES.md."
    )
    print(
        "Not checked here: which photographs the page displays, or on which surface. "
        "Its SOURCES.md lists them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
