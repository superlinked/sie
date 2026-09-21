#!/usr/bin/env python3
"""Reproduce the /doc-field-extraction figures from the recorded calls. No API
key, no network, no inference spend.

    python3 fetch.py
    python3 score.py

Prints the three figures the page publishes:

    180 of 223 fields exact across all 8 recorded documents, in 9 calls
    8 of 8 ticked and empty boxes read correctly
    9 of 9 replies valid against the schema, first call

This is the website's evaluate.py with its inputs repointed at the fetched
evidence: it reads the pre-registered expected values and the recorded replies,
applies the comparison rules `inputs.json` records under "scoring", and never
edits an expected value.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
PAGE_FIGURES = {
    "fields_matched": 180,
    "fields_total": 223,
    "documents": 8,
    "calls": 9,
    "booleans": 8,
    "schema_valid": 9,
}

# Must stay identical to inputs_digest() in the runner that recorded the run.
INPUTS_METADATA_KEYS = ("note", "registered_utc")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    """The digest the runner recorded: sorted keys, compact separators."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def inputs_digest(inputs: dict[str, Any]) -> str:
    """Hash what decides the score, ignoring prose that only describes it."""
    scored = {key: value for key, value in inputs.items() if key not in INPUTS_METADATA_KEYS}
    return canonical_sha256(scored)


def load(name: str) -> Any:
    path = EVIDENCE / name
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_string(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    for source, target in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")):
        text = text.replace(source, target)
    text = "".join(ch for ch in text if not ch.isspace() and ch != ",")
    text = text.casefold()
    return text.removesuffix(".")


def values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)):
        if isinstance(actual, str):
            try:
                actual = float(actual.replace(",", "."))
            except ValueError:
                return False
        if not isinstance(actual, (int, float)):
            return False
        return abs(float(expected) - float(actual)) < 1e-9
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False
        return normalize_string(expected) == normalize_string(actual)
    return expected == actual


def compare(expected: Any, actual: Any, path: str, fields: list[dict[str, Any]], extra_rows: list[str]) -> None:
    if isinstance(expected, dict):
        actual_dict = actual if isinstance(actual, dict) else {}
        for key, value in expected.items():
            compare(value, actual_dict.get(key), f"{path}.{key}" if path else key, fields, extra_rows)
        return
    if isinstance(expected, list):
        actual_list = actual if isinstance(actual, list) else []
        for index, value in enumerate(expected):
            compare(
                value, actual_list[index] if index < len(actual_list) else None, f"{path}[{index}]", fields, extra_rows
            )
        for index in range(len(expected), len(actual_list)):
            extra_rows.append(f"{path}[{index}]")
        return
    fields.append({"field": path, "expected": expected, "returned": actual, "match": values_match(expected, actual)})


def schema_errors(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Check a returned object against the registered JSON schema.

    Covers what these schemas use: object properties and required keys, array
    items, scalar types and string enums. A reply only counts as schema-valid
    when this returns no errors.
    """
    where = path or "$"
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{where}: expected object"]
        errors: list[str] = []
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{where}.{key}: missing required key")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors += schema_errors(value[key], sub, f"{path}.{key}")
        return errors
    if kind == "array":
        if not isinstance(value, list):
            return [f"{where}: expected array"]
        items = schema.get("items")
        if not isinstance(items, dict):
            return []
        errors = []
        for index, entry in enumerate(value):
            errors += schema_errors(entry, items, f"{path}[{index}]")
        return errors
    if kind == "string":
        if not isinstance(value, str):
            return [f"{where}: expected string"]
        allowed = schema.get("enum")
        if allowed is not None and value not in allowed:
            return [f"{where}: {value!r} is outside the enum"]
        return []
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return [f"{where}: expected integer"]
        return []
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{where}: expected number"]
        return []
    if kind == "boolean":
        if not isinstance(value, bool):
            return [f"{where}: expected boolean"]
        return []
    return []


def evaluate_call(entry: dict[str, Any], expected: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    record = entry["response"]
    body = record["body"] if isinstance(record["body"], dict) else {}
    text = body.get("text")
    parsed: Any = None
    parse_error: str | None = None
    if isinstance(text, str):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            parse_error = str(error)
    parsed_json = parsed is not None and parse_error is None
    errors = schema_errors(parsed, schema) if parsed_json else ["reply did not parse as JSON"]
    fields: list[dict[str, Any]] = []
    extra_rows: list[str] = []
    compare(expected, parsed if isinstance(parsed, dict) else {}, "", fields, extra_rows)
    return {
        "call": entry["slug"],
        "status": entry["status"],
        "attempts": entry["attempts"],
        "finish_reason": body.get("finish_reason"),
        "parsed_json": parsed_json,
        "schema_valid_json": not errors,
        "schema_errors": errors,
        "parse_error": parse_error,
        "fields_total": len(fields),
        "fields_matched": sum(1 for field in fields if field["match"]),
        "extra_rows": extra_rows,
        "field_results": fields,
    }


def verify_evidence(inputs: dict[str, Any], manifest: dict[str, Any], calls: dict[str, Any]) -> list[str]:
    """Check the bytes before scoring them. Anything unreadable is a failure."""
    problems: list[str] = []

    digest = inputs_digest(inputs)
    if digest != manifest["inputs_sha256"]:
        problems.append(f"inputs.json scores to {digest}, but the run was recorded against {manifest['inputs_sha256']}")

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

    pinned = {case["id"]: case["image_sha256"] for case in inputs["cases"]}
    for entry in calls["calls"]:
        recorded = entry["images"][0]["sha256"]
        if pinned.get(entry["case"]) != recorded:
            problems.append(f"{entry['case']}: inputs.json pins a different image than the run sent")
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
    cases: list[dict[str, Any]] = []
    missing: list[str] = []

    for case in inputs["cases"]:
        entry = recorded.get(f"{case['id']}__proof")
        if entry is None:
            # Counted and named, never passed over.
            missing.append(f"{case['id']}__proof")
            continue
        result = evaluate_call(entry, case["expected"], inputs["schemas"][case["schema"]])
        result["id"] = case["id"]
        result["title"] = case["title"]
        cases.append(result)

    playground_case = inputs["playground"]
    playground_entry = recorded.get(f"{playground_case['case']}__playground")
    if playground_entry is None:
        missing.append(f"{playground_case['case']}__playground")
        playground = None
    else:
        playground = evaluate_call(
            playground_entry, playground_case["expected"], json.loads(playground_case["schema_text"])
        )
        playground["id"] = playground_case["case"]

    if missing:
        print("Call(s) NOT SCORED, so no figure below covers the recorded run:", file=sys.stderr)
        for name in missing:
            print(f"  missing {name}", file=sys.stderr)
        return 1

    total = sum(case["fields_total"] for case in cases)
    matched = sum(case["fields_matched"] for case in cases)
    scored_calls = cases + [playground]

    # The page's supporting claims, each derived from the same field results.
    booleans = [f for case in cases for f in case["field_results"] if isinstance(f["expected"], bool)]
    booleans_matched = sum(1 for f in booleans if f["match"])
    schema_valid = sum(1 for c in scored_calls if c["schema_valid_json"])
    first_call = sum(1 for c in scored_calls if c["attempts"] == 1)

    for case in cases:
        misses = [f["field"] for f in case["field_results"] if not f["match"]]
        line = f"  {case['id']}: {case['fields_matched']} of {case['fields_total']}"
        print(line + (f"   misses: {', '.join(misses)}" if misses else ""))
    print(f"  {playground['id']}__playground: {playground['fields_matched']} of {playground['fields_total']}")

    print(
        f"\n{matched} of {total} fields exact across all {len(cases)} recorded documents, in {len(scored_calls)} calls"
    )
    print(f"{booleans_matched} of {len(booleans)} ticked and empty boxes read correctly")
    print(f"{schema_valid} of {len(scored_calls)} replies valid against the schema, first call")

    want = PAGE_FIGURES
    checks = {
        "fields": (matched, total) == (want["fields_matched"], want["fields_total"]),
        "documents": len(cases) == want["documents"],
        "calls": len(scored_calls) == want["calls"],
        "booleans": (booleans_matched, len(booleans)) == (want["booleans"], want["booleans"]),
        "schema valid on the first call": schema_valid == want["schema_valid"] and first_call == want["schema_valid"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print(
            f"\nThis does NOT reproduce {', '.join(failed)} as published on {manifest['page']}. "
            "Report it rather than adjusting either number.",
            file=sys.stderr,
        )
        return 1
    print(f"Matches the figures published on {manifest['page']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
