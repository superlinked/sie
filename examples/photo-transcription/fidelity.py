"""Verify the photographed source and its recorded transcription decisions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dataset import digest, verify_packet


def load_packet(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_packet(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    projection = json.loads((directory / "provider-projection.json").read_text())
    records = manifest["records"]
    if manifest["n"] != 24 or len(records) != 24:
        raise ValueError("Expected the complete 24-photo cohort")
    cases = {row["id"]: row for row in records}
    if len(cases) != 24 or len({row["source_cluster"] for row in records}) != 24:
        raise ValueError("Source IDs and physical-document clusters must be unique")
    if Counter(row["domain"] for row in records) != {"receipt": 12, "handwriting": 12}:
        raise ValueError("Expected twelve receipts and twelve handwritten documents")
    for row in records:
        if digest(row["gold_text"].encode()) != row["gold_sha256"]:
            raise ValueError(f"Gold hash mismatch: {row['id']}")
        for field in ("source_image", "input_image"):
            info = row[field]
            relative = Path(info["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Source paths must stay inside the packet")
            if digest((directory / relative).read_bytes()) != info["sha256"]:
                raise ValueError(f"Image hash mismatch: {row['id']}")
    rows = projection["provider_records"]
    if len(rows) != 48:
        raise ValueError("Expected every recorded provider response")
    for arm in ("luna", "sol"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        if len(arm_rows) != 24 or {row["case_id"] for row in arm_rows} != set(cases):
            raise ValueError(f"Incomplete provider cohort: {arm}")
    for row in rows:
        source = cases[row["case_id"]]
        if row["image_sha256"] != source["input_image"]["sha256"]:
            raise ValueError("Provider response is bound to a different image")
        if digest(row["text"].encode()) != row["transcription_sha256"]:
            raise ValueError("Recorded transcription changed")
        if row["attempts"] != 1 or row["phase"] != "main":
            raise ValueError("Expected one main attempt per recorded provider")
    return manifest, projection


def recorded_report(projection: dict[str, Any]) -> dict[str, Any]:
    rows = projection["provider_records"]
    report: dict[str, Any] = {"metric": "Complete transcription and no critical source errors"}
    for arm in ("luna", "sol"):
        selected = [row for row in rows if row["arm"] == arm]
        report[arm] = {
            "n": len(selected),
            "passed": sum(row["source_grounded"]["source_grounded_primary_outcome"] == "pass" for row in selected),
        }
    report["native"] = None
    report["source"] = "Preserved source-adjacent-v2 decisions; not a new model run"
    return report


def native_report(
    manifest: dict[str, Any], calls: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply complete source-context reviews, never string-only value matching."""
    cases = {row["id"]: row for row in manifest["records"]}
    if len(calls) != len(cases) or {row["case_id"] for row in calls} != set(cases):
        raise ValueError("Retain every frozen case, including unattempted cases")
    reviewed = {row["case_id"]: row for row in decisions}
    if len(reviewed) != len(decisions) or set(reviewed) - set(cases):
        raise ValueError("Duplicate or foreign source decisions")
    outcomes: dict[str, str] = {}
    for call in calls:
        case_id = call["case_id"]
        source = cases[case_id]
        if call["image_sha256"] != source["input_image"]["sha256"]:
            raise ValueError("Native call is bound to a different image")
        if call["status"] == "unattempted":
            if call["attempts"] != 0:
                raise ValueError("Unattempted rows cannot contain physical attempts")
            outcomes[case_id] = "unattempted"
            continue
        if call["attempts"] != 1:
            raise ValueError("Expected exactly one physical native attempt")
        if call["status"] in ("error", "empty"):
            outcomes[case_id] = "fail"
            continue
        if call["status"] != "completed":
            raise ValueError("Unknown native call status")
        text = call["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Completed native calls must contain nonempty text")
        text_hash = digest(text.encode())
        if text_hash != call["transcription_sha256"]:
            raise ValueError("Native transcription changed")
        decision = reviewed.get(case_id)
        if decision is None:
            outcomes[case_id] = "unreviewed"
            continue
        if (
            not isinstance(decision.get("reviewer"), str)
            or not decision["reviewer"].strip()
            or decision["transcription_sha256"] != text_hash
        ):
            raise ValueError("Review identity and exact transcription binding are required")
        if decision["image_sha256"] != call["image_sha256"]:
            raise ValueError("Review is bound to a different source")
        critical = decision["critical_outcomes"]
        lines = decision["line_outcomes"]
        if len(critical) != len(source["critical_values"]) or len(lines) != len(source["reviewed_lines"]):
            raise ValueError("Every source occurrence and readable line needs a decision")
        if any(
            value not in {"preserved", "changed", "missing", "misassociated", "invented", "not_assessable"}
            for value in critical
        ):
            raise ValueError("Unknown critical-value decision")
        if any(value not in {"represented", "missing", "not_assessable"} for value in lines):
            raise ValueError("Unknown source-line decision")
        # The single declared publisher-blurred logo is the only excluded source.
        excluded_critical = {15} if case_id == "cord/train-0012" else set()
        excluded_lines = {0} if case_id == "cord/train-0012" else set()
        if any(critical[index] != "not_assessable" for index in excluded_critical) or any(
            lines[index] != "not_assessable" for index in excluded_lines
        ):
            raise ValueError("The publisher-blurred logo must be recorded as not assessable")
        if any(value == "not_assessable" and index not in excluded_critical for index, value in enumerate(critical)):
            raise ValueError("An undeclared critical occurrence cannot be excluded")
        if any(value == "not_assessable" and index not in excluded_lines for index, value in enumerate(lines)):
            raise ValueError("An undeclared readable line cannot be excluded")
        passed = all(value == "preserved" for index, value in enumerate(critical) if index not in excluded_critical)
        passed = passed and all(
            value == "represented" for index, value in enumerate(lines) if index not in excluded_lines
        )
        outcomes[case_id] = "pass" if passed else "fail"
    counts = Counter(outcomes.values())
    return {
        "n_frozen": len(cases),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "unattempted": counts["unattempted"],
        "unreviewed": counts["unreviewed"],
        "complete_reviewed_cohort": not counts["unattempted"] and not counts["unreviewed"],
        "outcomes": outcomes,
    }
