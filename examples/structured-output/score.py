#!/usr/bin/env python3
"""Reproduce the /structured-output page figure from the recorded calls.

    python3 fetch.py
    python3 score.py

Published on https://superlinked.com/structured-output:

    All 21 yes-or-no fields came back right, and no document says true or false

and, under the proof grid:

    84 of the 85 checked fields came back right

This script re-derives all of those offline, with no API key and no inference
spend. It exits nonzero if any of them fails to reproduce.

A yes-or-no field is one the case schema declared `"type": "boolean"` or
`"type": ["boolean", "null"]`. That is read from the schema the call actually
sent, not from the value that came back, so a field that correctly returned
null because its listing gives no answer still counts as a question asked.

What it does:

  1. reads `data/calls.json` and keeps the 13 calls in the `page` set;
  2. drops the 3 cases listed in `data/inputs/excluded.json`, which are
     excluded from every published total with a recorded reason, leaving 10;
  3. parses the assistant message of each recorded response as JSON;
  4. validates it against that case's JSON Schema from `data/inputs/cases.json`;
  5. applies the acceptance checks in `data/inputs/checks.json`, which were
     written before each case's first run;
  6. groups those checks by what the schema asked for, and searches each source
     text for the words true and false, which is the page's second claim.

Schema validation runs under `jsonschema` when it is importable, and otherwise
under the small validator in this file, which covers the subset these schemas
use. Whichever runs, its verdict is also compared with the verdict the original
runner recorded, so two independent implementations have to agree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HTTP_OK = 200

EXPECTED = {
    "cases": 10,
    "parsed": 10,
    "schema_valid": 10,
    "checks_passed": 84,
    "checks_total": 85,
    "yes_no_passed": 21,
    "yes_no_total": 21,
    "enum_passed": 5,
    "enum_total": 5,
}

# The page heading says no document says true or false. Checked against the
# source texts here, not against that sentence.
LITERAL_BOOLEAN = re.compile(r"(?<![A-Za-z])(?:true|false)(?![A-Za-z])", re.IGNORECASE)


# A yes-or-no field is one declared "boolean" or ["boolean", "null"], and
# nothing wider. Membership alone would also catch a union like
# ["boolean", "string"], which is not a yes-or-no question and would inflate the
# published count, so the whole declared set has to be one of these.
YES_NO_TYPES = frozenset({"boolean", "null"})


def asked_for(spec: object) -> str:
    """What the schema declared a field to be: a yes-or-no question, a pick from
    a fixed list, or an ordinary value. Read from the schema that was sent."""
    if not isinstance(spec, dict):
        return "value"
    declared = spec.get("type")
    types = declared if isinstance(declared, list) else [declared]
    if "boolean" in types and set(types) <= YES_NO_TYPES:
        return "yes_no"
    items = spec.get("items")
    if isinstance(spec.get("enum"), list):
        return "enum"
    if isinstance(items, dict) and isinstance(items.get("enum"), list):
        return "enum"
    return "value"


JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def keep_all(_call: dict[str, Any]) -> bool:
    """Every recorded call counts toward this task's figures."""
    return True


def scored_calls(payload: dict[str, Any], keep: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    """The calls the figure is computed from, refusing anything that failed.

    A recorder that hit an error writes the call with status "error" and sets
    `complete` to false. Scoring such a file would turn a failed run into a
    published number, so it stops here instead.
    """
    if payload.get("complete") is False:
        failed = payload.get("failed_calls", "some")
        raise SystemExit(f"refusing to score: {failed} calls in this calls.json failed, so it is not a complete run")
    calls = [call for call in payload["calls"] if keep(call)]
    broken = [call["id"] for call in calls if call.get("status") != HTTP_OK]
    if broken:
        raise SystemExit("refusing to score calls that did not return 200: " + ", ".join(sorted(broken)))
    return calls


def same_value(actual: object, expected: object) -> bool:
    """Equality that never treats True as 1 or False as 0."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual is expected
    return actual == expected


def check_passes(op: str, actual: object, expected: object) -> bool:
    if op == "eq":
        return same_value(actual, expected)
    if op == "ieq":
        return isinstance(actual, str) and actual.casefold() == str(expected).casefold()
    if op == "icontains":
        return isinstance(actual, str) and str(expected).casefold() in actual.casefold()
    if op == "in":
        return any(same_value(actual, option) for option in expected)
    if op == "contains_all":
        return isinstance(actual, list) and all(item in actual for item in expected)
    if op == "contains_any":
        return isinstance(actual, list) and any(item in actual for item in expected)
    raise ValueError(f"unknown check op {op}")


def type_ok(value: object, spec: object) -> bool:
    names = spec if isinstance(spec, list) else [spec]
    for name in names:
        if name == "integer":
            if not isinstance(value, bool) and isinstance(value, int):
                return True
            continue
        if name == "boolean":
            if isinstance(value, bool):
                return True
            continue
        if name == "number":
            if not isinstance(value, bool) and isinstance(value, (int, float)):
                return True
            continue
        if isinstance(value, JSON_TYPES[name]):
            return True
    return False


def validate(schema: dict[str, Any], value: object, path: str = "$") -> list[str]:
    """Validate the subset of JSON Schema these cases use."""
    errors: list[str] = []
    if "type" in schema and not type_ok(value, schema["type"]):
        return [f"{path}: expected type {schema['type']}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not one of {schema['enum']}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        errors.append(f"{path}: below minimum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: required property missing")
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{name}: additional property" for name in value if name not in properties)
        for name, subschema in properties.items():
            if name in value:
                errors.extend(validate(subschema, value[name], f"{path}.{name}"))
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(validate(schema["items"], item, f"{path}[{index}]"))
    return errors


def validator() -> tuple[str, Any]:
    try:
        from importlib import metadata  # noqa: PLC0415

        import jsonschema  # noqa: PLC0415
    except ImportError:
        return "built in", None
    return f"jsonschema {metadata.version('jsonschema')}", jsonschema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    args = parser.parse_args()
    data_dir = Path(args.data)

    cases = {case["id"]: case for case in load(data_dir / "inputs/cases.json")["cases"]}
    checks = load(data_dir / "inputs/checks.json")["cases"]
    excluded = {entry["id"] for entry in load(data_dir / "inputs/excluded.json")["excluded"]}
    calls = scored_calls(load(data_dir / "calls.json"), lambda call: call["set"] == "page")

    name, module = validator()
    totals = {
        "cases": 0,
        "parsed": 0,
        "schema_valid": 0,
        "checks_passed": 0,
        "checks_total": 0,
        "yes_no_passed": 0,
        "yes_no_total": 0,
        "enum_passed": 0,
        "enum_total": 0,
    }
    disagreements: list[str] = []
    wrong_fields: list[str] = []
    spelled_out: list[str] = []

    for call in calls:
        case_id = call["case"]
        if case_id in excluded:
            continue
        totals["cases"] += 1
        text = call["response"]["body"]["choices"][0]["message"]["content"]
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            print(f"{case_id}: response is not JSON ({error})", file=sys.stderr)
            continue
        totals["parsed"] += 1

        schema = cases[case_id]["schema"]
        if module is None:
            valid = not validate(schema, value)
        else:
            validator_cls = module.validators.validator_for(schema)
            validator_cls.check_schema(schema)
            valid = not list(validator_cls(schema).iter_errors(value))
        if valid:
            totals["schema_valid"] += 1

        recorded = call["recorded"].get("schema_validation", {}).get("valid")
        if recorded is not None and recorded != valid:
            disagreements.append(f"{case_id}: recorded valid={recorded}, recomputed valid={valid}")

        if LITERAL_BOOLEAN.search(cases[case_id]["text"]):
            spelled_out.append(case_id)

        for check in checks[case_id]:
            totals["checks_total"] += 1
            group = asked_for(schema.get("properties", {}).get(check["field"]))
            if group != "value":
                totals[f"{group}_total"] += 1
            # An absent field is not a returned null. `.get()` cannot tell them
            # apart, so a check expecting null would score a field the model
            # never returned as correct. Schema validation happens to catch that
            # today, because every case schema lists every property as required,
            # but that is a different check shielding this one rather than this
            # one working.
            present = isinstance(value, dict) and check["field"] in value
            actual = value.get(check["field"]) if present else None
            if present and check_passes(check["op"], actual, check["expected"]):
                totals["checks_passed"] += 1
                if group != "value":
                    totals[f"{group}_passed"] += 1
            elif not present:
                wrong_fields.append(
                    f"{case_id}.{check['field']}: expected {check['expected']!r}, field absent from the reply"
                )
            else:
                wrong_fields.append(f"{case_id}.{check['field']}: expected {check['expected']!r}, got {actual!r}")

    print(f"schema validator: {name}")
    print(f"documents scored:  {totals['cases']}")
    print(f"parsed as JSON:    {totals['parsed']}")
    print(f"schema-valid:      {totals['schema_valid']}")
    print(f"fields right:      {totals['checks_passed']} of {totals['checks_total']}")
    print(f"  yes-or-no fields:  {totals['yes_no_passed']} of {totals['yes_no_total']}")
    print(f"  picks from a list: {totals['enum_passed']} of {totals['enum_total']}")
    print(
        f"  everything else:   {totals['checks_passed'] - totals['yes_no_passed'] - totals['enum_passed']}"
        f" of {totals['checks_total'] - totals['yes_no_total'] - totals['enum_total']}"
    )
    print(f"documents saying true or false: {len(spelled_out)}")
    print(f"excluded from every total: {len(excluded)} CPSC cases, listed with their reason in inputs/excluded.json")
    if wrong_fields:
        print("\nfields the model got wrong:")
        for line in wrong_fields:
            print(f"  {line}")

    failures = [
        f"{key}: got {totals[key]}, page publishes {want}" for key, want in EXPECTED.items() if totals[key] != want
    ]
    for line in disagreements:
        failures.append(f"validator disagreement, {line}")
    for case_id in spelled_out:
        failures.append(f"{case_id}: source text says true or false, so the page heading is wrong")
    if failures:
        print("\nFAILED to reproduce the published figure:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(
        f"\nReproduced: all {totals['yes_no_total']} yes-or-no fields right, no document says"
        f" true or false, {totals['checks_passed']} of {totals['checks_total']} checked fields right."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
