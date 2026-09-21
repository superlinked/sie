#!/usr/bin/env python3
"""Reproduce the /image-classify figures from the recorded calls. No API key, no
network, no inference spend.

    python3 fetch.py
    python3 score.py

Prints the pass-or-reject result the page publishes: the reranker flagged 8 of
8 damaged pieces and passed 7 of 8 whole ones, and SigLIP 2 base sorted 12 of
16. It also prints the eight score pairs the page shows beside its photographs.

The SigLIP figure is re-derived here, not read back. The recorded response
holds the raw image and label vectors; this scorer L2-normalizes them and takes
the cosine itself, so the number comes from the embedding rather than from the
verdict the original run wrote down beside it.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"

RERANKER = "Qwen/Qwen3-VL-Reranker-2B"
SIGLIP = "google/siglip2-base-patch16-224"
PASS_REJECT = "02-pass-reject"
GRADES = "01-grades"

# What https://superlinked.com/image-classify publishes. score.py exits
# non-zero if it computes anything else. If that happens, report it: it means
# the page or the evidence is wrong, and neither should be quietly adjusted.
PAGE_PHOTOS = 16
PAGE_DAMAGED_FLAGGED = (8, 8)
PAGE_WHOLE_PASSED = (7, 8)
PAGE_TIES = 1
PAGE_TIE_CASE = "fryum-intact-003"
PAGE_TIE_SCORE = "0.562"
PAGE_SIGLIP_SORTED = (12, 16)
PAGE_SIGLIP_DAMAGED_PASSED = 2
PAGE_SIGLIP_WHOLE_FLAGGED = 2
# The first run, four grade labels per photo, cited on the page as the reason
# the two-label framing was chosen.
PAGE_GRADES = {"images": 16, "labelsPerImage": 4, "rerankerCorrect": 4, "siglipCorrect": 4}
# The eight reranker score pairs the page prints, as (whole, damaged) rounded
# the way the page rounds them. Four products, one whole and one damaged each.
# Read off the built page, not off the evidence these scores come from: the
# chewing-gum pair is in the hero and again in the grid, and the cashew damaged
# photo is also the playground's, so the eight cards are eight distinct photos
# of the sixteen recorded.
PAGE_PAIRS = {
    "chewinggum-intact-000": ("0.593", "0.484"),
    "chewinggum-damaged-016": ("0.547", "0.651"),
    "cashew-intact-002": ("0.577", "0.531"),
    "cashew-damaged-014": ("0.547", "0.637"),
    "fryum-intact-003": ("0.562", "0.562"),
    "fryum-damaged-012": ("0.593", "0.719"),
    "pipe-fryum-intact-000": ("0.500", "0.453"),
    "pipe-fryum-damaged-016": ("0.516", "0.547"),
}
# The photograph the playground shows, and the order its output panel lists.
PAGE_PLAYGROUND = ("cashew-damaged-014", "broken or damaged cashew", "0.637", "whole undamaged cashew", "0.547")


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


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors of different length")
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def verify_evidence(manifest: dict[str, Any], calls: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    """Check the bytes before scoring them. Anything unreadable is a failure.

    The two inputs files are the one thing here whose digests do NOT match what
    the run recorded, because a timestamp was corrected and a note explaining
    the correction was added afterwards. That is not waved through: the
    manifest records both digests, this checks the file against the one it says
    the file has now, and the run against the one it says the run used. An
    inputs file edited after packaging still fails.
    """
    problems: list[str] = []

    for run, meta in manifest["runs"].items():
        path = EVIDENCE / meta["inputs_file"]
        if not path.is_file():
            problems.append(f"{meta['inputs_file']} was not downloaded")
            continue
        digest = sha256_bytes(path.read_bytes())
        if digest != meta["inputs_sha256_now"]:
            problems.append(
                f"{meta['inputs_file']} hashes to {digest}, the manifest packaged {meta['inputs_sha256_now']}"
            )
        if meta["inputs_sha256_matches"]:
            problems.append(
                f"{run}: the manifest claims the inputs file still matches the run, which contradicts its "
                "own recorded note; this scorer was written for the mismatching case"
            )

    # The display renditions. They are not the bytes any call was sent and no
    # figure below moves if one is absent, but the manifest pins every one and
    # fetch.py requires them, so a missing or altered file means the bundle is
    # incomplete. Reported as a failure that says what it does and does not
    # affect, rather than passed over because the number survives it.
    per_case = manifest["input_sources"]["per_case"]
    if set(per_case) != {case["id"] for case in inputs[PASS_REJECT]}:
        problems.append(
            f"the manifest pins display renditions for a different set of photos than {PASS_REJECT} registers"
        )
    verified = 0
    for case_id, meta in sorted(per_case.items()):
        path = EVIDENCE / meta["display_file"]
        if not path.is_file():
            problems.append(
                f"{case_id}: {meta['display_file']} was not downloaded. It changes no figure below, being the "
                "page's rendition rather than the bytes scored, but the bundle is incomplete; "
                "run: python3 fetch.py"
            )
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != meta["display_sha256"] or len(data) != meta["display_bytes"]:
            problems.append(f"{case_id}: {meta['display_file']} is not the rendition the manifest pins")
            continue
        verified += 1
    if verified != manifest["images"]["display_count"]:
        problems.append(
            f"{verified} display renditions verified, the manifest counts {manifest['images']['display_count']}"
        )

    for entry in calls["calls"]:
        if canonical_sha256(entry["request"]) != entry["request_sha256"]:
            problems.append(f"{entry['slug']}: the request record does not match its recorded digest")
        if canonical_sha256(entry["response"]) != entry["response_sha256"]:
            problems.append(f"{entry['slug']}: the response record does not match its recorded digest")
        if entry["http_status"] != 200:
            problems.append(f"{entry['slug']}: recorded HTTP {entry['http_status']}")

    slugs = {entry["slug"] for entry in calls["calls"]}
    if len(slugs) != len(calls["calls"]):
        problems.append("calls.json holds two entries with the same slug")

    # The inputs digests moved, so the substance is checked instead of trusted:
    # every case still registered must have been run, with the photograph it
    # names and against the labels it names.
    for run, cases in inputs.items():
        recorded = {entry["case"]: entry for entry in calls["calls"] if entry["run"] == run}
        for case in cases:
            for call in ("reranker-score", "siglip-encode"):
                if f"{run}__{case['id']}__{call}" not in slugs:
                    problems.append(f"{run}/{case['id']}: registered in the inputs file but {call} was not recorded")
            entry = recorded.get(case["id"])
            if entry is None:
                continue
            if entry["source_image_sha256"] != case["sha256"]:
                problems.append(
                    f"{run}/{case['id']}: the photograph the call was sent is not the one the inputs file pins"
                )
        for case_id in recorded:
            if case_id not in {case["id"] for case in cases}:
                problems.append(f"{run}/{case_id}: recorded, but no longer registered in the inputs file")
    return problems


def label_set(document: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
    """The labels this photo's product registers, class key to label string."""
    labels = document["label_sets"][case["product"]]
    if sorted(labels) != sorted(document["classes"]):
        raise SystemExit(f"{case['id']}: its product registers labels for a different class set")
    return labels


def labels_of(entry: dict[str, Any]) -> list[str]:
    """The label strings this call sent, in the order it sent them."""
    body = entry["request"]["body"]
    if entry["call"] == "reranker-score":
        return [item["text"] for item in body["items"]]
    return [item["text"] for item in body["items"] if "text" in item]


def reranker_scores(entry: dict[str, Any]) -> dict[str, float]:
    """Label to score, resolved through the item ids the request assigned."""
    sent = {item["id"]: item["text"] for item in entry["request"]["body"]["items"]}
    out: dict[str, float] = {}
    for row in entry["response"]["body"]["scores"]:
        label = sent[row["item_id"]]
        if label in out:
            raise SystemExit(f"{entry['slug']}: two scores for the same label")
        out[label] = row["score"]
    return out


def siglip_scores(entry: dict[str, Any]) -> dict[str, float]:
    """Cosine of the image vector with each label vector, computed here."""
    items = entry["response"]["body"]["items"]
    labels = labels_of(entry)
    if len(items) != len(labels) + 1:
        raise SystemExit(f"{entry['slug']}: expected one image vector and {len(labels)} label vectors")
    image = items[0]["dense"]["values"]
    return {label: cosine(image, items[index + 1]["dense"]["values"]) for index, label in enumerate(labels)}


def verdict(scores: dict[str, float], whole: str, damaged: str) -> str:
    if scores[damaged] > scores[whole]:
        return "flagged"
    if scores[whole] > scores[damaged]:
        return "passed"
    return "tie"


def main() -> int:
    manifest = load("manifest.json")
    calls = load("calls.json")
    documents = {run: load(meta["inputs_file"]) for run, meta in manifest["runs"].items()}
    inputs = {run: document["cases"] for run, document in documents.items()}

    problems = verify_evidence(manifest, calls, inputs)
    if problems:
        print("The evidence did not verify, so nothing was scored:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(
        "note: both inputs files were edited after the run, to correct a hand-entered timestamp and to record\n"
        "      that correction. Their digests no longer match the run, the manifest carries both, and every\n"
        "      case and every recorded call is unchanged. Checked above, not skipped.\n"
    )

    by_slug = {entry["slug"]: entry for entry in calls["calls"]}
    failures: list[str] = []

    # --- the published run --------------------------------------------------
    tallies = {
        RERANKER: {"flagged": 0, "passed": 0, "tie": 0, "correct": 0, "damaged_passed": 0, "whole_flagged": 0},
        SIGLIP: {"flagged": 0, "passed": 0, "tie": 0, "correct": 0, "damaged_passed": 0, "whole_flagged": 0},
    }
    damaged_total = whole_total = 0
    ties: list[tuple[str, str]] = []
    pairs: dict[str, tuple[str, str]] = {}

    for case in inputs[PASS_REJECT]:
        labels = label_set(documents[PASS_REJECT], case)
        whole, damaged = labels["intact"], labels["damaged"]
        is_damaged = case["expected"] == "damaged"
        damaged_total += is_damaged
        whole_total += not is_damaged
        for model, call, reader in ((RERANKER, "reranker-score", reranker_scores), (SIGLIP, "siglip-encode", siglip_scores)):
            entry = by_slug[f"{PASS_REJECT}__{case['id']}__{call}"]
            if entry["model"] != model:
                failures.append(f"{entry['slug']}: recorded against {entry['model']}, expected {model}")
                continue
            scores = reader(entry)
            if set(scores) != {whole, damaged}:
                failures.append(f"{entry['slug']}: the labels scored are not the two the inputs file registers")
                continue
            result = verdict(scores, whole, damaged)
            tallies[model][result] += 1
            if result == "tie":
                if model == RERANKER:
                    ties.append((case["id"], f"{scores[whole]:.3f}"))
            elif (result == "flagged") == is_damaged:
                tallies[model]["correct"] += 1
            elif is_damaged:
                tallies[model]["damaged_passed"] += 1
            else:
                tallies[model]["whole_flagged"] += 1
            if model == RERANKER:
                pairs[case["id"]] = (f"{scores[whole]:.3f}", f"{scores[damaged]:.3f}")

    reranker = tallies[RERANKER]
    siglip = tallies[SIGLIP]
    damaged_flagged = sum(
        1
        for case in inputs[PASS_REJECT]
        if case["expected"] == "damaged"
        and verdict(
            reranker_scores(by_slug[f"{PASS_REJECT}__{case['id']}__reranker-score"]),
            label_set(documents[PASS_REJECT], case)["intact"],
            label_set(documents[PASS_REJECT], case)["damaged"],
        )
        == "flagged"
    )
    whole_passed = sum(
        1
        for case in inputs[PASS_REJECT]
        if case["expected"] == "intact"
        and verdict(
            reranker_scores(by_slug[f"{PASS_REJECT}__{case['id']}__reranker-score"]),
            label_set(documents[PASS_REJECT], case)["intact"],
            label_set(documents[PASS_REJECT], case)["damaged"],
        )
        == "passed"
    )

    for case_id, (whole_score, damaged_score) in sorted(pairs.items()):
        print(f"  {case_id}: whole {whole_score}, damaged {damaged_score}")

    photos = len(inputs[PASS_REJECT])
    print(
        f"\nAcross all {photos} recorded photos the reranker flagged {damaged_flagged} of {damaged_total} "
        f"damaged pieces and passed {whole_passed} of {whole_total} whole ones"
    )
    for case_id, score in ties:
        print(f"  it scored {score} on both labels for {case_id}, so that photo is neither flagged nor passed")
    siglip_sorted = siglip["correct"]
    print(
        f"SigLIP 2 base sorted {siglip_sorted} of {photos}: it passed {siglip['damaged_passed']} damaged "
        f"pieces and flagged {siglip['whole_flagged']} whole ones"
    )

    # --- the first run, four labels per photo -------------------------------
    grades = {"images": len(inputs[GRADES]), "rerankerCorrect": 0, "siglipCorrect": 0}
    labels_per_image = set()
    for case in inputs[GRADES]:
        for model, call, reader in ((RERANKER, "reranker-score", reranker_scores), (SIGLIP, "siglip-encode", siglip_scores)):
            entry = by_slug[f"{GRADES}__{case['id']}__{call}"]
            scores = reader(entry)
            labels_per_image.add(len(scores))
            top = max(scores, key=lambda label: scores[label])
            if top == label_set(documents[GRADES], case)[case["expected"]]:
                grades["rerankerCorrect" if model is RERANKER else "siglipCorrect"] += 1
    grades["labelsPerImage"] = labels_per_image.pop() if len(labels_per_image) == 1 else sorted(labels_per_image)
    print(
        f"The first run put {grades['labelsPerImage']} grade labels on each of {grades['images']} photos: "
        f"the reranker got {grades['rerankerCorrect']} and SigLIP 2 base {grades['siglipCorrect']}"
    )

    # --- compare with the page ----------------------------------------------
    if photos != PAGE_PHOTOS:
        failures.append(f"photos: got {photos}, page publishes {PAGE_PHOTOS}")
    if (damaged_flagged, damaged_total) != PAGE_DAMAGED_FLAGGED:
        failures.append(
            f"damaged: got {damaged_flagged} of {damaged_total}, page publishes "
            f"{PAGE_DAMAGED_FLAGGED[0]} of {PAGE_DAMAGED_FLAGGED[1]}"
        )
    if (whole_passed, whole_total) != PAGE_WHOLE_PASSED:
        failures.append(
            f"whole: got {whole_passed} of {whole_total}, page publishes "
            f"{PAGE_WHOLE_PASSED[0]} of {PAGE_WHOLE_PASSED[1]}"
        )
    if reranker["whole_flagged"]:
        failures.append(
            f"the page headline says no whole piece was flagged, and {reranker['whole_flagged']} was"
        )
    if reranker["damaged_passed"]:
        failures.append(
            f"the page headline says no broken piece got through, and {reranker['damaged_passed']} did"
        )
    if len(ties) != PAGE_TIES:
        failures.append(f"ties: got {len(ties)}, page publishes {PAGE_TIES}")
    for case_id, score in ties:
        if (case_id, score) != (PAGE_TIE_CASE, PAGE_TIE_SCORE):
            failures.append(f"tie: got {case_id} at {score}, page publishes {PAGE_TIE_CASE} at {PAGE_TIE_SCORE}")
    if (siglip_sorted, photos) != PAGE_SIGLIP_SORTED:
        failures.append(
            f"siglip: got {siglip_sorted} of {photos}, page publishes "
            f"{PAGE_SIGLIP_SORTED[0]} of {PAGE_SIGLIP_SORTED[1]}"
        )
    if siglip["damaged_passed"] != PAGE_SIGLIP_DAMAGED_PASSED:
        failures.append(
            f"siglip damaged passed: got {siglip['damaged_passed']}, page publishes {PAGE_SIGLIP_DAMAGED_PASSED}"
        )
    if siglip["whole_flagged"] != PAGE_SIGLIP_WHOLE_FLAGGED:
        failures.append(
            f"siglip whole flagged: got {siglip['whole_flagged']}, page publishes {PAGE_SIGLIP_WHOLE_FLAGGED}"
        )
    for key, want in PAGE_GRADES.items():
        if grades[key] != want:
            failures.append(f"first run {key}: got {grades[key]}, page publishes {want}")
    for case_id, want in PAGE_PAIRS.items():
        got = pairs.get(case_id)
        if got != want:
            failures.append(f"{case_id}: got {got}, page publishes whole {want[0]}, damaged {want[1]}")

    case_id, top_label, top_score, next_label, next_score = PAGE_PLAYGROUND
    played = reranker_scores(by_slug[f"{PASS_REJECT}__{case_id}__reranker-score"])
    ordered = sorted(played.items(), key=lambda row: row[1], reverse=True)
    if [(label, f"{score:.3f}") for label, score in ordered] != [(top_label, top_score), (next_label, next_score)]:
        failures.append(f"playground {case_id}: got {ordered}, page publishes {top_label} {top_score} first")

    if failures:
        print(
            f"\nThis does NOT reproduce what {manifest['page']} publishes. "
            "Report it rather than adjusting either number.",
            file=sys.stderr,
        )
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(
        f"\nMatches the {PAGE_DAMAGED_FLAGGED[0]} of {PAGE_DAMAGED_FLAGGED[1]} damaged, the "
        f"{PAGE_WHOLE_PASSED[0]} of {PAGE_WHOLE_PASSED[1]} whole, the SigLIP "
        f"{PAGE_SIGLIP_SORTED[0]} of {PAGE_SIGLIP_SORTED[1]}, and all "
        f"{len(PAGE_PAIRS)} displayed score pairs, published on {manifest['page']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
