"""Replay document-level count evidence without making model calls."""
# ruff: noqa: INP001 - Standalone example scripts are not a Python package.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from fetch import HERE, load_manifest, verify_file

ARMS = ("sie-qwen3.8-27b", "claude-haiku-4-5", "claude-sonnet-5-5", "gpt-6-luna", "gpt-6-sol")
FULL_DOCUMENTS = 800
FRESH_DOCUMENTS = 770


def load_evidence(root: Path) -> tuple[dict, list[dict]]:
    """Verify every declared file before using the published counts."""
    manifest = load_manifest((root / "manifest.json").read_bytes())
    verified = {}
    for name, expected in manifest["files"].items():
        data = (root / name).read_bytes()
        verify_file(name, data, expected)
        verified[name] = data
    summary = json.loads(verified["summary.json"])
    rows = [json.loads(line) for line in verified["counts.jsonl"].splitlines()]
    identities = {(row["arm"], row["document_id"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("Duplicate arm/document pair")
    inputs = [json.loads(line) for line in verified["inputs.jsonl"].splitlines()]
    documents = {row["document_id"]: row for row in inputs}
    if len(documents) != FULL_DOCUMENTS or len(inputs) != FULL_DOCUMENTS:
        raise ValueError("Expected exactly 800 distinct input documents")
    expected_identities = {(arm, document_id) for arm in ARMS for document_id in documents}
    if identities != expected_identities:
        raise ValueError("Count evidence must cover every input once for each of the five arms")
    if set(summary["selected_claude_forms"]) != set(ARMS):
        raise ValueError("Summary must preserve all five model arms")
    for row in rows:
        document = documents[row["document_id"]]
        if (row["corpus"], row["fresh"]) != (document["corpus"], document["fresh"]):
            raise ValueError(f"Count/input identity mismatch: {row['document_id']}")
    return summary, rows


def aggregate(rows: list[dict], arm: str, fresh: bool) -> dict:
    """Average one complete model scope while retaining every failed reply."""
    selected = [row for row in rows if row["arm"] == arm and (not fresh or row["fresh"])]
    expected_n = FRESH_DOCUMENTS if fresh else FULL_DOCUMENTS
    if len(selected) != expected_n:
        raise ValueError(f"Incomplete scope for {arm}: {len(selected)} of {expected_n}")
    omni = [row for row in selected if row["corpus"] == "omni"]
    cord = [row for row in selected if row["corpus"] == "cord"]
    return {
        "documents": len(selected),
        "omni_documents": len(omni),
        "cord_documents": len(cord),
        "omni_json_accuracy": sum(row["omni_accuracy"] for row in omni) / len(omni),
        "cord_normalized_field_accuracy": sum(row["normalized_field_accuracy"] for row in cord) / len(cord),
        "unparsed": sum(not row["parsed"] for row in selected),
        "output_limits": sum(row["hit_limit"] for row in selected),
        "transport_errors": sum(row["error"] for row in selected),
        "fully_correct_omni": sum(row["fully_right"] for row in omni) / len(omni),
        "mean_omni_tokens_in": sum(row["tokens_in"] for row in omni) / len(omni),
        "mean_omni_tokens_out": sum(row["tokens_out"] for row in omni) / len(omni),
    }


def replay(root: Path) -> dict:
    """Recompute both scopes and require every aggregate to match the recording."""
    summary, rows = load_evidence(root)
    results = {scope: {arm: aggregate(rows, arm, scope == "fresh") for arm in ARMS} for scope in ("full", "fresh")}
    mapping = {
        "omni_documents": "omni_n",
        "cord_documents": "cord_n",
        "omni_json_accuracy": "omni_accuracy",
        "cord_normalized_field_accuracy": "cord_normalized_field_accuracy",
        "unparsed": "unparsed",
        "output_limits": "output_cap_rows",
        "transport_errors": "transport_errors",
        "fully_correct_omni": "omni_fully_right",
        "mean_omni_tokens_in": "tokens_in_mean_omni",
        "mean_omni_tokens_out": "tokens_out_mean_omni",
    }
    for scope, arms in results.items():
        for arm, result in arms.items():
            recorded = summary["scopes"][scope]["summary"][arm]
            for computed_key, recorded_key in mapping.items():
                if not math.isclose(result[computed_key], recorded[recorded_key], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Count replay mismatch: {scope}/{arm}/{computed_key}")
    return {
        "count_replay": results,
        "verified": True,
        "registered_bootstrap": summary["bootstrap"],
        "registered_decisions": summary["eligible_on_both_scopes"],
        "current_price_projection": summary["current_price_projection"],
    }


def main() -> None:
    """Print the verified count replay and its saved decision metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=HERE / "data")
    args = parser.parse_args()
    print(json.dumps(replay(args.evidence), indent=2))


if __name__ == "__main__":
    main()
