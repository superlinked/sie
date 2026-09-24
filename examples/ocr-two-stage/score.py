#!/usr/bin/env python3
"""Reproduce the /ocr page figures from the recorded two-stage run.

    python3 fetch.py
    python3 score.py

Reads inputs.json (registered before any call) together with the recorded
stage-1 and stage-2 responses, and writes evaluation.json beside them. Every
verdict is computed here; none is hand-authored, and this script never reads
the page.

The verdict turns on one question: did the field's registered `printed` token
survive into stage 1's Markdown? That is what separates an error stage 2
inherited from stage 1 from an error stage 2 made on text it could read.

This script re-derives the figures the /ocr page's evidence publishes, exiting
nonzero if any fails:

    81 of 86   registered printed tokens survive into stage 1's Markdown
    78 of 89   fields match for Qwen/Qwen3.8-27B-FP8, 7 of 7 replies schema-valid
    27 of 89   fields match for Qwen/Qwen3.5-4B, 2 of 7 replies schema-valid

Those are the whole recorded run, which is what
`superlinked.com/reference/ocr-review/SOURCES.md` reports. The page's proof
grid shows a selection of it: the schema calls that matched every field they
registered. What this script does NOT check is which recorded cases the page
displays, or in what order. That is the page's decision, it changes without the
run changing, and an example has no way to read the page.

Standard library only. No network, no API key, no inference spend.

This was `evaluate.py`, reading three committed files. It now reads the same
three files from the pinned Hugging Face revision that `fetch.py` downloads.
Every verdict rule, the normalization and the schema checks are unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

MISSING = object()

# Set by main() from --data. The three files this script reads and nothing else.
DATA_DIR = Path("data")
INPUTS_PATH = DATA_DIR / "inputs" / "inputs.json"
PREDICTIONS_PATH = DATA_DIR / "inputs" / "predictions.json"
CALLS_PATH = DATA_DIR / "calls.json"
EVALUATION_PATH = DATA_DIR / "evaluation.json"

# What the page's evidence publishes. Typed out from SOURCES.md, not computed
# from the recordings, so agreement means something.
#
# `documents_displayed` is the `displayed` flag in inputs.json, set before the
# run. It is a count of the pre-run display choice, not of what the proof grid
# renders today; the two differ by one document, and only sie-web can check the
# second.
EXPECTED = {
    "stage1_tokens_in_text": 81,
    "stage1_registered_tokens": 86,
    "primary_matched": 78,
    "primary_registered_fields": 89,
    "primary_schema_valid": 7,
    "primary_calls": 7,
    "secondary_matched": 27,
    "secondary_registered_fields": 89,
    "secondary_schema_valid": 2,
    "secondary_calls": 7,
    "documents_registered": 6,
    "documents_displayed": 3,
}

# Every response this example scores lives in one calls.json, keyed by the slug
# the runner recorded. sie-web keeps the same responses as one file per call;
# the verdict rules below are identical either way.
_CALLS: dict[str, Any] = {}


def use_data_dir(data_dir: Path) -> None:
    """Point the three reads at a fetched evidence directory."""
    global DATA_DIR, INPUTS_PATH, PREDICTIONS_PATH, CALLS_PATH, EVALUATION_PATH  # noqa: PLW0603
    DATA_DIR = data_dir
    INPUTS_PATH = data_dir / "inputs" / "inputs.json"
    PREDICTIONS_PATH = data_dir / "inputs" / "predictions.json"
    CALLS_PATH = data_dir / "calls.json"
    EVALUATION_PATH = data_dir / "evaluation.json"
    _CALLS.clear()


def norm(value: Any) -> str:
    """Unicode NFKC, collapse whitespace, trim. The registered normalization."""
    text = unicodedata.normalize("NFKC", str(value))
    # The OCR text uses curly quotes where the registered expectation uses
    # straight ones; fold them so a token search is not defeated by typography.
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def read_json(path: Path) -> Any:
    if not path.exists():
        # A missing input is a failure, never a skip. Scoring what is left
        # would report a smaller run as a complete one.
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_calls() -> dict[str, Any]:
    if not _CALLS:
        for call in read_json(CALLS_PATH)["calls"]:
            _CALLS[call["slug"]] = call
    return _CALLS


def response_body(slug: str) -> Any:
    call = load_calls().get(slug)
    if call is None:
        raise SystemExit(f"no recorded call for {slug!r} in {CALLS_PATH.name}")
    return call["response"]["body"]


def stage1_markdown(doc_id: str) -> str:
    body = response_body(f"{doc_id}__stage1")
    return body["items"][0]["entities"][0]["text"]


def stage2_object(doc_id: str, call_id: str, model_key: str) -> tuple[Any, str | None]:
    body = response_body(f"{doc_id}__{call_id}__stage2__{model_key}")
    content = body["choices"][0]["message"]["content"]
    try:
        return json.loads(content), None
    except json.JSONDecodeError as error:
        return None, f"{type(error).__name__}: {error}"


# --- path resolution -------------------------------------------------------

STEP = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(.+?)\])?$")


def resolve(obj: Any, path: str) -> Any:
    """Resolve a registered path such as rows[line_item=Taiwan].fy2026.

    A `[key=value]` selector picks the first list element whose `key` matches
    `value` after normalization. A `[*]` selector collects the field from every
    element, which is how list-recall fields are scored.
    """
    current: Any = obj
    for raw in split_path(path):
        match = STEP.match(raw)
        if not match:
            raise ValueError(f"Unparsable path segment {raw!r} in {path!r}")
        key, selector = match.group(1), match.group(2)
        if isinstance(current, list):
            return MISSING
        if not isinstance(current, dict) or key not in current:
            return MISSING
        current = current[key]
        if selector is None:
            continue
        if not isinstance(current, list):
            return MISSING
        if selector == "*":
            return current
        field, _, wanted = selector.partition("=")
        picked = MISSING
        for element in current:
            if isinstance(element, dict) and norm(element.get(field, "")) == norm(wanted):
                picked = element
                break
        if picked is MISSING:
            return MISSING
        current = picked
    return current


def split_path(path: str) -> list[str]:
    """Split on dots that sit outside a [...] selector."""
    parts, depth, buf = [], 0, ""
    for char in path:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "." and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += char
    parts.append(buf)
    return parts


def collect(values: Any, key: str) -> list[Any]:
    """Values of `key` across a list, keeping explicit nulls and skipping
    elements that do not carry the key at all. `.get()` would flatten the two
    into None and let an omitted key read as a returned null."""
    if not isinstance(values, list):
        return []
    return [v[key] for v in values if isinstance(v, dict) and key in v]


# --- rules -----------------------------------------------------------------


def compare_text(value: Any, compare: str | None) -> str:
    text = norm(value)
    if compare == "strip_spaces":
        text = text.replace(" ", "")
    return text


def rule_satisfied(field: dict[str, Any], returned: Any) -> bool:
    rule = field["rule"]
    expect = field.get("expect")
    compare = field.get("compare")

    if rule == "absent":
        return returned is None or returned is MISSING

    if returned is MISSING:
        return False

    if rule in {"integer", "signed_parenthesis"}:
        return isinstance(returned, int) and not isinstance(returned, bool) and returned == expect

    if rule in {"verbatim", "strip_footnote_marker", "gs1_element_string"}:
        return returned is not None and compare_text(returned, compare) == compare_text(expect, compare)

    if rule == "contains":
        return returned is not None and norm(expect).lower() in norm(returned).lower()

    if rule == "any_of":
        if returned is None:
            return False
        got = compare_text(returned, compare)
        return any(got == compare_text(option, compare) for option in expect)

    if rule == "any_of_in_list":
        # `returned` is the list produced by a [*] selector.
        if not isinstance(returned, list):
            return False
        wanted = {compare_text(option, compare) for option in expect}
        return any(v is not None and compare_text(v, compare) in wanted for v in returned)

    raise ValueError(f"Unknown rule {rule!r}")


def is_null(field: dict[str, Any], returned: Any) -> bool:
    """Did stage 2 return null for this field?

    MISSING is not null. A `[key=value]` selector that matches no element means
    stage 2 returned rows under different keys, which is a wrong non-null
    answer; the registered taxonomy reserves `stage2_dropped` for an explicit
    null. Conflating the two would report a mislabelled row as a drop.
    """
    if field["rule"] == "any_of_in_list":
        if not isinstance(returned, list):
            return False
        # all() is true of an empty list, and an empty array is stage 2
        # answering with no rows rather than answering null.
        return bool(returned) and all(v is None for v in returned)
    return returned is None


def verdict_for(field: dict[str, Any], returned: Any, markdown_norm: str) -> str:
    if rule_satisfied(field, returned):
        return "match"

    printed = field.get("printed")
    in_text = printed is not None and norm(printed) in markdown_norm

    if field["rule"] == "absent":
        # The value is not legible on the page and stage 1 rightly did not
        # produce it, so a non-null answer is stage 2's own invention. This
        # sub-case of the registered taxonomy is reported separately.
        return "stage2_invented"

    if is_null(field, returned):
        return "stage2_dropped" if in_text else "stage1_lost"
    return "stage2_error" if in_text else "stage1_error"


# --- minimal JSON Schema check --------------------------------------------


def schema_errors(schema: dict[str, Any], value: Any, where: str = "$") -> list[str]:
    """Checks the subset the registered schemas use: type, required, enum,
    additionalProperties and array items."""
    errors: list[str] = []
    types = schema.get("type")
    allowed = types if isinstance(types, list) else [types]

    def type_ok(name: str) -> bool:
        if name == "object":
            return isinstance(value, dict)
        if name == "array":
            return isinstance(value, list)
        if name == "string":
            return isinstance(value, str)
        if name == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if name == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if name == "boolean":
            return isinstance(value, bool)
        if name == "null":
            return value is None
        return True

    if types is not None and not any(type_ok(name) for name in allowed):
        return [f"{where}: expected {types}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {value!r} not in enum")

    if isinstance(value, dict) and "properties" in schema:
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{where}: missing required {name!r}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in schema["properties"]:
                    errors.append(f"{where}: unexpected property {name!r}")
        for name, sub in schema["properties"].items():
            if name in value:
                errors.extend(schema_errors(sub, value[name], f"{where}.{name}"))

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, element in enumerate(value):
            errors.extend(schema_errors(schema["items"], element, f"{where}[{index}]"))

    return errors


# --- pre-registered predictions ------------------------------------------


def primary_fields(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        field
        for document in documents
        for call in document["calls"]
        if call["model_key"] == "primary"
        for field in call["fields"]
    ]


def score_predictions(
    inputs: dict[str, Any],
    documents: list[dict[str, Any]],
    totals: dict[str, Any],
) -> list[dict[str, Any]]:
    """Decide each registered prediction from the recorded run.

    The prediction text lives in inputs.json, which was committed before the
    first call. predictions.json only says how each one is decided, and this
    function refuses to run if the two disagree, so a prediction cannot be
    restated on the page in softer words than the one that was registered.
    """
    spec = read_json(PREDICTIONS_PATH)
    registered = inputs["predictions"]["items"]
    fields = primary_fields(documents)
    primary = totals["primary"]
    scored: list[dict[str, Any]] = []

    for entry in spec["predictions"]:
        index = entry["index"]
        if index >= len(registered) or entry["text"] != registered[index]:
            raise SystemExit(f"prediction {index} in predictions.json does not match inputs.json verbatim")
        check = entry["check"]
        kind = check["kind"]
        numbers = {
            "matched": primary["matched"],
            "registered": primary["registered_fields"],
            "count": 0,
            "total": 0,
        }

        if kind == "any_field_missed_with_rule":
            pool = [f for f in fields if f["rule"] == check["rule"]]
            missed = [f for f in pool if f["verdict"] != "match"]
            numbers["count"] = len(missed)
            numbers["total"] = len(pool)
            held = bool(missed)
        elif kind == "field_missed":
            field = find_field(documents, check["document"], check["call"], check["path"])
            held = field["verdict"] != "match"
        elif kind == "field_returned_equals":
            field = find_field(documents, check["document"], check["call"], check["path"])
            held = field["path_resolved"] and norm(field["returned"]) == norm(check["value"])
        elif kind == "field_returned_in":
            # Several accepted values, so a prediction is scored the way that
            # most favours it. Used where the registered text names a date the
            # prediction's own reasoning does not produce.
            field = find_field(documents, check["document"], check["call"], check["path"])
            held = field["path_resolved"] and any(norm(field["returned"]) == norm(v) for v in check["values"])
        elif kind == "any_verdict":
            hits = [f for f in fields if f["verdict"] == check["verdict"]]
            numbers["count"] = len(hits)
            held = bool(hits)
        elif kind == "majority_match":
            held = primary["matched"] * 2 > primary["registered_fields"]
        else:
            raise SystemExit(f"Unknown prediction check {kind!r}")

        template = entry["evidence_held"] if held else entry["evidence_failed"]
        scored.append(
            {
                "index": index,
                "text": registered[index],
                "short": entry["short"],
                "held": held,
                "evidence": template.format(**numbers),
            }
        )

    if len(scored) != len(registered):
        raise SystemExit("every registered prediction must be scored")
    return scored


def find_field(documents: list[dict[str, Any]], doc_id: str, call_id: str, path: str) -> dict[str, Any]:
    for document in documents:
        if document["document"] != doc_id:
            continue
        for call in document["calls"]:
            if call["call"] != call_id or call["model_key"] != "primary":
                continue
            for field in call["fields"]:
                if field["path"] == path:
                    return field
    raise SystemExit(f"prediction references an unrecorded field: {doc_id}/{call_id}/{path}")


# --- main ------------------------------------------------------------------


def evaluate_call(
    inputs: dict[str, Any],
    doc: dict[str, Any],
    call: dict[str, Any],
    model_key: str,
    markdown: str,
) -> dict[str, Any]:
    markdown_norm = norm(markdown)
    returned_obj, parse_error = stage2_object(doc["id"], call["id"], model_key)

    # A reply that did not parse, or that parsed into something that is not an
    # object, cannot answer any field. Without this gate the `absent` rule is
    # satisfied by a missing path, so a reply like `[]` would earn `match` on
    # every field registered as not legible on the page.
    usable = isinstance(returned_obj, dict)

    fields: list[dict[str, Any]] = []
    for field in call["expected"]:
        if not usable:
            record = {
                "path": field["path"],
                "printed": field.get("printed"),
                "rule": field["rule"],
                "expected": field.get("expect"),
                # No reply to resolve against, so nothing resolved.
                "path_resolved": False,
                "returned": None,
                "printed_token_in_stage1_text": (
                    field.get("printed") is not None and norm(field["printed"]) in markdown_norm
                ),
                "verdict": "stage2_unusable",
            }
            fields.append(record)
            continue
        if parse_error is not None:
            returned: Any = MISSING
        else:
            resolved = resolve(returned_obj, field["path"])
            if field["rule"] == "any_of_in_list":
                leaf = split_path(field["path"])[-1]
                key = STEP.match(leaf).group(1) if STEP.match(leaf) else leaf
                parent = ".".join(split_path(field["path"])[:-1])
                container = resolve(returned_obj, parent) if parent else MISSING
                returned = collect(container, key) if isinstance(container, list) else resolved
            else:
                returned = resolved
        record = {
            "path": field["path"],
            "printed": field.get("printed"),
            "rule": field["rule"],
            "expected": field.get("expect"),
            # The verdict turns on this: an unresolved path is a wrong answer,
            # an explicit null is stage 2 declining to answer. Both serialise
            # as null, so the flag is what keeps them apart in the artifact.
            "path_resolved": returned is not MISSING,
            "returned": None if returned is MISSING else returned,
            "printed_token_in_stage1_text": (
                field.get("printed") is not None and norm(field["printed"]) in markdown_norm
            ),
            "verdict": verdict_for(field, returned, markdown_norm),
        }
        fields.append(record)

    errors = (
        schema_errors(inputs["schemas"][call["schema"]], returned_obj)
        if parse_error is None
        else [f"reply did not parse as JSON: {parse_error}"]
    )
    counts: dict[str, int] = {}
    for record in fields:
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1

    return {
        "document": doc["id"],
        "call": call["id"],
        "schema": call["schema"],
        "model_key": model_key,
        "model": inputs["stage2"]["models"][model_key],
        "json_parsed": parse_error is None,
        "parse_error": parse_error,
        "schema_valid": not errors,
        "schema_errors": errors,
        "fields": fields,
        "counts": counts,
        "registered_fields": len(fields),
        "matched": counts.get("match", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    args = parser.parse_args()
    use_data_dir(Path(args.data))

    inputs = read_json(INPUTS_PATH)
    documents: list[dict[str, Any]] = []
    model_keys = list(inputs["stage2"]["models"])

    for doc in inputs["documents"]:
        markdown = stage1_markdown(doc["id"])
        markdown_norm = norm(markdown)
        tokens = sorted(
            {
                field["printed"]
                for call in doc["calls"]
                for field in call["expected"]
                if field.get("printed") is not None
            }
        )
        found = [token for token in tokens if norm(token) in markdown_norm]
        entry = {
            "document": doc["id"],
            "kind": doc["kind"],
            "displayed": doc["displayed"],
            "image": doc["image"],
            "image_sha256": doc["image_sha256"],
            "stage1": {
                "markdown_chars": len(markdown),
                "registered_tokens": len(tokens),
                "tokens_in_text": len(found),
                "tokens_missing": [token for token in tokens if token not in found],
            },
            "calls": [
                evaluate_call(inputs, doc, call, model_key, markdown)
                for call in doc["calls"]
                for model_key in model_keys
            ],
        }
        documents.append(entry)

    totals: dict[str, dict[str, Any]] = {}
    for model_key in model_keys:
        counts: dict[str, int] = {}
        registered = matched = calls = parsed = valid = 0
        for entry in documents:
            for call in entry["calls"]:
                if call["model_key"] != model_key:
                    continue
                calls += 1
                parsed += 1 if call["json_parsed"] else 0
                valid += 1 if call["schema_valid"] else 0
                registered += call["registered_fields"]
                matched += call["matched"]
                for name, value in call["counts"].items():
                    counts[name] = counts.get(name, 0) + value
        totals[model_key] = {
            "model": inputs["stage2"]["models"][model_key],
            "calls": calls,
            "json_parsed": parsed,
            "schema_valid": valid,
            "registered_fields": registered,
            "matched": matched,
            "verdicts": counts,
        }

    predictions = score_predictions(inputs, documents, totals)

    stage1_tokens = sum(e["stage1"]["registered_tokens"] for e in documents)
    stage1_found = sum(e["stage1"]["tokens_in_text"] for e in documents)

    evaluation = {
        "inputs_sha256": __import__("hashlib").sha256(INPUTS_PATH.read_bytes()).hexdigest(),
        "documents_registered": len(documents),
        "documents_displayed": sum(1 for e in documents if e["displayed"]),
        "stage1": {
            "model": inputs["stage1"]["model"],
            "registered_tokens": stage1_tokens,
            "tokens_in_text": stage1_found,
        },
        "stage2_totals": totals,
        "predictions": predictions,
        "documents": documents,
    }
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent="\t", ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"stage 1: {stage1_found}/{stage1_tokens} registered printed tokens survive into the Markdown")
    for model_key, total in totals.items():
        print(
            f"stage 2 {model_key} ({total['model']}): {total['matched']}/{total['registered_fields']} fields,"
            f" {total['schema_valid']}/{total['calls']} schema-valid replies, {total['verdicts']}"
        )
    for entry in documents:
        for call in entry["calls"]:
            if call["model_key"] != "primary":
                continue
            bad = [f for f in call["fields"] if f["verdict"] != "match"]
            if bad:
                print(f"\n{entry['document']} / {call['call']}:")
                for field in bad:
                    print(
                        f"  {field['verdict']:16} {field['path']}"
                        f"  printed={field['printed']!r} returned={field['returned']!r}"
                    )

    got = {
        "stage1_tokens_in_text": stage1_found,
        "stage1_registered_tokens": stage1_tokens,
        "primary_matched": totals["primary"]["matched"],
        "primary_registered_fields": totals["primary"]["registered_fields"],
        "primary_schema_valid": totals["primary"]["schema_valid"],
        "primary_calls": totals["primary"]["calls"],
        "secondary_matched": totals["secondary"]["matched"],
        "secondary_registered_fields": totals["secondary"]["registered_fields"],
        "secondary_schema_valid": totals["secondary"]["schema_valid"],
        "secondary_calls": totals["secondary"]["calls"],
        "documents_registered": evaluation["documents_registered"],
        "documents_displayed": evaluation["documents_displayed"],
    }
    failures = [f"{key}: got {got[key]}, page publishes {want}" for key, want in EXPECTED.items() if got[key] != want]
    if failures:
        sys.stdout.flush()
        print("\nFAILED to reproduce the published figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nReproduced: 81 of 86, 78 of 89 with 7 of 7 schema-valid, and 27 of 89 with 2 of 7.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
