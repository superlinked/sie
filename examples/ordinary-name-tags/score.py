"""Score typed, source-literal name tags without contacting an inference service."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LABELS = ("person", "organization", "location")
MODELS = {
    "gliner-community/gliner_large-v2.5": {
        "threshold": 0.75,
        "checkpoint_revision": "3d6d1760be1c591069f85f207fced9214df8b15f",
    },
    "gliner-community/gliner_medium-v2.5": {
        "threshold": 0.55,
        "checkpoint_revision": "88c3b98b57ad5e7d66fb209ed61c53f4b1fd05da",
    },
}
Tag = tuple[str, str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tag_set(
    text: str, entities: list[dict[str, Any]], *, spans: bool = False, require_literal: bool = True
) -> set[Tag]:
    """Validate gold/native literals; retain invented comparison names as false positives."""
    tags: set[Tag] = set()
    for entity in entities:
        name, kind = entity.get("text"), entity.get("type", entity.get("label"))
        if not isinstance(name, str) or not name or kind not in LABELS or (require_literal and name not in text):
            raise ValueError("An entity is not a source-literal name with a requested type")
        if spans:
            start, end = entity.get("start"), entity.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError("An entity has invalid Unicode character offsets")
            if text[start:end] != name:
                raise ValueError("An entity does not match its Unicode character offsets")
            confidence = entity.get("score")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("An entity has an invalid confidence score")
        tags.add((name, kind))
    return tags


def load_frame(cases_path: Path, variants_path: Path) -> tuple[list[dict[str, Any]], dict[str, list[set[Tag]]]]:
    cases = read_jsonl(cases_path)
    ids = [case["id"] for case in cases]
    families = [case["source_family_id"] for case in cases]
    if not cases or len(set(ids)) != len(ids) or len(set(families)) != len(families):
        raise ValueError("The frame must have one distinct case per independent source family")
    for case in cases:
        if tuple(case["labels"]) != LABELS:
            raise ValueError("The frame must use the same three plain labels for every case")
        digest = hashlib.sha256(case["text"].encode()).hexdigest()
        if digest != case["text_sha256"]:
            raise ValueError(f"Input text digest mismatch for {case['id']}")
        gold = tag_set(case["text"], case["tags"])
        if tag_set(case["text"], case["entities"]) != gold:
            raise ValueError("The span and tag answer keys disagree")
        for entity in case["entities"]:
            if case["text"][entity["start"] : entity["end"]] != entity["text"]:
                raise ValueError("An answer-key span does not match its source")
    by_id = {case["id"]: case for case in cases}
    variants: dict[str, list[set[Tag]]] = {}
    for row in json.loads(variants_path.read_text(encoding="utf-8")):
        case_id = row["id"]
        if case_id not in by_id or case_id in variants:
            raise ValueError("An acceptable representation has an unknown or duplicate case")
        variants[case_id] = [tag_set(by_id[case_id]["text"], tags) for tags in row["acceptable_tag_sets"]]
        if not variants[case_id] or tag_set(by_id[case_id]["text"], by_id[case_id]["tags"]) not in variants[case_id]:
            raise ValueError("Acceptable representations must include the unchanged strict answer key")
    return cases, variants


def counts(predicted: set[Tag], gold: set[Tag]) -> tuple[int, int, int]:
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def f1(predicted: set[Tag], gold: set[Tag]) -> float:
    if not predicted and not gold:
        return 1.0
    tp, fp, fn = counts(predicted, gold)
    return 2 * tp / (2 * tp + fp + fn)


def evaluate(
    cases: list[dict[str, Any]], variants: dict[str, list[set[Tag]]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep failed and missing planned cases in the denominator with quality zero."""
    known = {case["id"] for case in cases}
    by_id: dict[str, dict[str, Any]] = {}
    models: set[str] = set()
    for row in rows:
        if row["id"] not in known or row["id"] in by_id:
            raise ValueError("Recordings contain an unknown or duplicate case")
        by_id[row["id"]] = row
        models.add(row["model"])
    if len(models) != 1:
        raise ValueError("Each recording file must contain exactly one model")
    model = next(iter(models))
    results = []
    for case in cases:
        row = by_id.get(case["id"], {"status": "unattempted"})
        gold = tag_set(case["text"], case["tags"])
        status = row["status"]
        predicted: set[Tag] = set()
        if status == "ok":
            if row.get("text_sha256") != case["text_sha256"]:
                raise ValueError("A successful recording is not bound to the same input")
            predicted = tag_set(case["text"], row["entities"], spans=model in MODELS, require_literal=model in MODELS)
        elif status not in {"failed", "unattempted"}:
            raise ValueError("Recordings must declare ok, failed or unattempted status")
        accepted = max(variants.get(case["id"], [gold]), key=lambda candidate: f1(predicted, candidate))
        tp, fp, fn = counts(predicted, accepted)
        results.append(
            {
                "id": case["id"],
                "domain": case["domain"],
                "status": status,
                "accepted_f1": f1(predicted, accepted) if status == "ok" else 0.0,
                "strict_f1": f1(predicted, gold) if status == "ok" else 0.0,
                "exact": status == "ok" and predicted == accepted,
                "entity_bearing": bool(gold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "empty_source_false_tags": len(predicted) if not gold else 0,
            }
        )
    tp, fp, fn = (sum(row[key] for row in results) for key in ("tp", "fp", "fn"))
    positive = [row for row in results if row["entity_bearing"]]
    precision = tp / (tp + fp) if tp + fp else 0.0
    positive_f1 = sum(row["accepted_f1"] for row in positive) / len(positive) if positive else 0.0
    complete = all(row["status"] == "ok" for row in results)
    return {
        "model": model,
        "planned_source_families": len(cases),
        "successful_calls": sum(row["status"] == "ok" for row in results),
        "macro_f1": sum(row["accepted_f1"] for row in results) / len(results),
        "strict_macro_f1": sum(row["strict_f1"] for row in results) / len(results),
        "entity_bearing_macro_f1": positive_f1,
        "pooled_precision": precision,
        "pooled_recall": tp / (tp + fn) if tp + fn else 0.0,
        "exact_documents": sum(row["exact"] for row in results),
        "empty_source_false_tags": sum(row["empty_source_false_tags"] for row in results),
        "useful_fit_gate": complete and len(positive) >= 12 and precision >= 0.9 and positive_f1 >= 0.8,
        "cases": results,
    }


def paired_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Resample paired independent source families, never individual mentions."""
    diffs = [a["accepted_f1"] - b["accepted_f1"] for a, b in zip(left["cases"], right["cases"], strict=True)]
    rng = random.Random(20261006)
    bootstrap = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10_000))
    return {
        "left": left["model"],
        "right": right["model"],
        "mean_f1_difference": sum(diffs) / len(diffs),
        "paired_source_bootstrap_95_ci": [bootstrap[249], bootstrap[9749]],
        "resamples": 10_000,
        "seed": 20261006,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "inputs/cases.jsonl")
    parser.add_argument("--variants", type=Path, default=ROOT / "inputs/acceptable-tag-representations.json")
    parser.add_argument(
        "--rows", type=Path, action="append", required=True, help="One JSONL file per model; repeat to compare"
    )
    args = parser.parse_args()
    cases, variants = load_frame(args.cases, args.variants)
    reports = [evaluate(cases, variants, read_jsonl(path)) for path in args.rows]
    comparisons = [paired_difference(reports[0], report) for report in reports[1:]]
    print(json.dumps({"models": reports, "comparisons": comparisons}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
