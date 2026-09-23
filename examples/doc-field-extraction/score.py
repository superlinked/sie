#!/usr/bin/env python3
"""Reproduce the /doc-field-extraction figures from the recorded calls. No API
key, no network, no inference spend.

    python3 fetch.py
    python3 score.py

Prints the figures the page publishes:

    180 of 223 fields exact across all 8 recorded documents, in 9 calls
    8 of 8 ticked and empty boxes read correctly
    a second call over the same eight answers: 204 of 223, 24 fixed, 0 broken
    17 of 17 replies valid against the schema, first try

This is the website's evaluate.py with its inputs repointed at the fetched
evidence: it reads the pre-registered expected values and the recorded replies,
applies the comparison rules `inputs.json` records under "scoring", and never
edits an expected value.

The second pass is scored by the same functions, over `second-pass/calls.json`,
which lives in this repository rather than in the dataset because it carries no
image bytes: A1 and A3 do send the page image, and their recorded requests store
a placeholder naming its path and SHA-256 instead. A2 sends no image at all. Its three arms and their prompts were fixed in
`second-pass/PRE-REGISTRATION.md` before any of those 24 calls, together with
the rule that published one of them and the rule that would have published none.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"

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
#   curl -s "https://huggingface.co/api/datasets/superlinked/sie-task-evidence/tree/<revision>/doc-field-extraction?recursive=true"
PINNED_OIDS = {
    "inputs/inputs.json": "07a806d337906ab97746306109786aabada866f5",
    "calls.json": "bdfcb482884c674ae54ff50ab59cec5a5089c6e8",
    "manifest.json": "5025ef7fbe9d5d58d57827d931ee4bb3a47de275",
}

PAGE_FIGURES = {
    "fields_matched": 180,
    "fields_total": 223,
    "documents": 8,
    "calls": 9,
    "booleans": 8,
    "schema_valid": 9,
    # The second call, arm a2 of the pre-registered three.
    "second_pass_arm": "a2",
    "second_pass_matched": 204,
    "second_pass_fixed": 24,
    "second_pass_broken": 0,
    # Both passes plus the playground call, every reply schema-valid first try.
    "schema_valid_both_passes": 17,
}

# The bar this experiment had to clear, copied from second-pass/PRE-REGISTRATION.md,
# which was committed before the first of those 24 calls.
SECOND_PASS_RULE = {"min_total": 195, "max_regressions": 5}

SECOND_PASS = ROOT / "second-pass"

# Must stay identical to inputs_digest() in the runner that recorded the run.
INPUTS_METADATA_KEYS = ("note", "registered_utc")


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
        # The object this call returned. The second pass sends it back as its
        # input, so it comes from the verified first pass rather than from a
        # copy stored beside the second.
        "returned": parsed,
    }


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
    And that entry digest is held against the `image` and `image_sha256` the
    case pinned in `inputs.json`, which is the authoritative side: without it an
    image that agrees with the call that sent it passes even when it is not the
    image the case registered, so a swap between two documents would go
    unnoticed.
    """
    problems: list[str] = []

    digest = inputs_digest(inputs)
    if digest != manifest["inputs_sha256"]:
        problems.append(f"inputs.json scores to {digest}, but the run was recorded against {manifest['inputs_sha256']}")

    pins = unique_by(inputs["cases"], "id", "inputs.json")

    for entry in calls["calls"]:
        # An entry is looked up below by the slug built from a case id and a
        # call name, and verified here by its own case field. If the two can
        # disagree, an edit to slug alone gets the evidence validated as one
        # document and the reply scored against another's schema and expected
        # values.
        if entry["slug"] != f"{entry['case']}__{entry['call']}":
            problems.append(
                f"{entry['slug']}: recorded as case {entry['case']!r} call {entry['call']!r}; "
                f"the slug must be exactly {entry['case']}__{entry['call']}"
            )
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

        case = pins.get(entry["case"])
        if case is None:
            problems.append(
                f"{entry['slug']}: inputs.json registers no case {entry['case']!r}, so nothing pins its image"
            )
            continue
        # One page image per call is what inputs.json registers. Comparing only
        # entry["images"][0] would let a second image ride along unchecked.
        if len(entry["images"]) != 1:
            problems.append(f"{entry['slug']}: {len(entry['images'])} images recorded, but the case pins exactly one")
            continue
        image = entry["images"][0]
        if Path(image["path"]).name != case["image"] or image["sha256"] != case["image_sha256"]:
            problems.append(
                f"{entry['slug']}: this call sent {Path(image['path']).name} ({image['sha256'][:12]}), "
                f"but inputs.json pins {case['image']} ({case['image_sha256'][:12]})"
            )
    return problems


def second_pass_path(arm_name: str, arm: dict[str, Any], model: str) -> str:
    """The path this arm posts to, rebuilt rather than compared as a string.

    The generate arms put the model in the path, url-quoted with `/` written as
    `__`, byte-identical to wireGenerateModel() in the site's codegen and to
    run_second_pass.py. Rebuilding it checks that the recorded path names the
    registered model, which a literal comparison against `arm["path"]` would
    not: that field holds `/v1/generate`, without the model.
    """
    if arm_name == "a2":
        return arm["path"]
    return f"/v1/generate/{urllib.parse.quote(model.replace('/', '__'), safe='')}"


def second_pass_reply(entry: dict[str, Any]) -> Any:
    """The JSON object one second-pass reply carries, or None.

    A1 and A3 answer on /v1/generate and A2 on /v1/chat/completions, so the text
    sits in a different place. Returning None rather than raising keeps an
    unusable reply scoreable as zero matches instead of stopping the run.
    """
    body = entry["response"]["body"]
    if not isinstance(body, dict):
        return None
    if entry["path"].endswith("/chat/completions"):
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        # Every hop is checked before it is indexed. `choices[0].get(...)` and
        # `(message or {}).get(...)` both raise AttributeError on a non-dict,
        # and a reply's shape is constrained by nothing but a digest that
        # travels with it, so the docstring's promise has to be enforced here.
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        text = message.get("content")
    else:
        text = body.get("text")
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def rebuild_second_pass_body(
    arm_name: str,
    arm: dict[str, Any],
    experiment: dict[str, Any],
    case: dict[str, Any],
    first_pass_reply: Any,
) -> dict[str, Any]:
    """The request body this arm sends for this case, built from the inputs.

    Byte-identical in structure to run_second_pass.py in superlinked/sie-web.
    Keeping it here is what makes the recorded requests checkable: the digests
    in calls.json travel with the calls, so only a body rebuilt from the
    pre-registered arm and the first pass's own reply can catch a request that
    was edited and re-digested.
    """
    schema = arm["schemas"][case["schema"]]
    max_new_tokens = experiment["max_new_tokens"]
    if arm_name == "a2":
        system = arm["prompt"] + "\n\nRequired JSON schema:\n" + json.dumps(schema, indent=2)
        return {
            "model": experiment["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(first_pass_reply, indent=2, ensure_ascii=False)},
            ],
            "max_completion_tokens": max_new_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "doc_field_extraction", "strict": True, "schema": schema},
            },
        }
    prompt = arm["prompt"]
    if arm["sends_stage_one"]:
        prompt += "\n\nFirst pass:\n" + json.dumps(first_pass_reply, indent=2, ensure_ascii=False)
    placeholder = (
        f"<base64 of apps/site/public/reference/doc-field-extraction/{case['image']}, "
        f"sha256 {case['image_sha256']}>"
    )
    return {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "images": [{"data": placeholder, "format": case["format"]}],
        "grammar": {"json_schema": schema, "strict": True},
    }


def score_second_pass(
    inputs: dict[str, Any], first_pass: list[dict[str, Any]], recorded: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Score all three arms, and rebuild every request before trusting one.

    `first_pass` is the scored first call, so the fixed and broken counts are
    paired field by field against the run this example already verified, never
    against a copy of it stored beside the second pass.
    """
    problems: list[str] = []
    experiment = json.loads((SECOND_PASS / "experiment.json").read_text(encoding="utf-8"))
    calls = json.loads((SECOND_PASS / "calls.json").read_text(encoding="utf-8"))
    entries = unique_by(calls["calls"], "slug", "second-pass/calls.json")

    cases = unique_by(inputs["cases"], "id", "inputs.json")
    first_by_case = {case["id"]: case for case in first_pass}
    arm_names = sorted(experiment["arms"])

    # A bijection: the expected slugs come from the arms and the registered
    # cases, so a call that is missing, duplicated or implied by no arm fails.
    expected_slugs = {f"{case_id}__{arm}" for arm in arm_names for case_id in cases}
    if set(entries) != expected_slugs:
        problems.append(
            f"second-pass/calls.json holds {sorted(set(entries) - expected_slugs)} that no arm implies "
            f"and is missing {sorted(expected_slugs - set(entries))}"
        )
        return {}, problems

    arms: dict[str, Any] = {}
    for arm_name in arm_names:
        arm = experiment["arms"][arm_name]
        scored: list[dict[str, Any]] = []
        for case_id, case in cases.items():
            entry = entries[f"{case_id}__{arm_name}"]
            if entry["status"] != 200:
                problems.append(f"{entry['slug']}: recorded status {entry['status']}")
            # Everything the scorer reads off an entry is held against the arm
            # and case it was looked up by, not just the body. `path` decides
            # which parser reads the reply, so an entry with the right body and
            # a wrong path would be parsed as the other endpoint and still
            # verify; `arm` and `case` are the same class as the first pass's
            # `slug` check; and the page claims one model and revision across
            # both passes, which is a claim only a comparison can hold up.
            wire_path = second_pass_path(arm_name, arm, experiment["model"])
            if entry["path"] != wire_path:
                problems.append(
                    f"{entry['slug']}: recorded on {entry['path']!r}, but arm {arm_name} posts to {wire_path!r}"
                )
            if (entry["arm"], entry["case"]) != (arm_name, case_id):
                problems.append(
                    f"{entry['slug']}: recorded as arm {entry['arm']!r} case {entry['case']!r}; "
                    f"the slug must be exactly {case_id}__{arm_name}"
                )
            if entry["model"] != experiment["model"]:
                problems.append(
                    f"{entry['slug']}: recorded on {entry['model']!r}, but the arms register "
                    f"{experiment['model']!r}"
                )
            # These two digests travel inside calls.json, so they catch a
            # corrupted file and nothing more; an editor who changes a record
            # and recomputes the digest beside it satisfies both. What pins this
            # file is that it lives in the repository rather than in the
            # dataset, so any change to it shows in the diff, and that the
            # request is rebuilt below from the pre-registered arm.
            if canonical_sha256(entry["request"]) != entry["request_sha256"]:
                problems.append(f"{entry['slug']}: the request record does not match its recorded digest")
            if canonical_sha256(entry["response"]) != entry["response_sha256"]:
                problems.append(f"{entry['slug']}: the response record does not match its recorded digest")
            first = first_by_case[case_id]
            rebuilt = rebuild_second_pass_body(
                arm_name, arm, experiment, case, first["returned"] if arm["sends_stage_one"] else None
            )
            if rebuilt != entry["request"]["body"]:
                problems.append(f"{entry['slug']}: the recorded request is not the one this arm builds")
            reply = second_pass_reply(entry)
            schema = inputs["schemas"][case["schema"]]
            errors = schema_errors(reply, schema) if reply is not None else ["reply did not parse as JSON"]
            fields: list[dict[str, Any]] = []
            extra_rows: list[str] = []
            compare(case["expected"], reply if isinstance(reply, dict) else {}, "", fields, extra_rows)
            before = {row["field"]: row["match"] for row in first["field_results"]}
            if set(before) != {row["field"] for row in fields}:
                problems.append(f"{entry['slug']}: a different field set from the first pass; nothing is comparable")
                continue
            scored.append(
                {
                    "id": case_id,
                    "fields_total": len(fields),
                    "fields_matched": sum(1 for row in fields if row["match"]),
                    "schema_valid_json": not errors,
                    "attempts": entry["attempts"],
                    "fixed": [row["field"] for row in fields if row["match"] and not before[row["field"]]],
                    "broken": [row["field"] for row in fields if not row["match"] and before[row["field"]]],
                    "field_results": fields,
                }
            )
        total = sum(case["fields_matched"] for case in scored)
        fixed = sum(len(case["fixed"]) for case in scored)
        broken = sum(len(case["broken"]) for case in scored)
        arms[arm_name] = {
            "label": arm["label"],
            "fields_total": sum(case["fields_total"] for case in scored),
            "fields_matched": total,
            "fixed": fixed,
            "broken": broken,
            "schema_valid": sum(1 for case in scored if case["schema_valid_json"]),
            "clears_bar": total >= SECOND_PASS_RULE["min_total"] and broken <= SECOND_PASS_RULE["max_regressions"],
            "cases": scored,
        }

    winners = [name for name in arm_names if arms[name]["clears_bar"]]
    winners.sort(key=lambda name: (-arms[name]["fields_matched"], arms[name]["broken"], name))
    return {"arms": arms, "winner": winners[0] if winners else None, "run": calls["run_started_utc"]}, problems


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
    cases: list[dict[str, Any]] = []
    missing: list[str] = []

    scored_slugs: set[str] = set()
    for case in inputs["cases"]:
        entry = recorded.get(f"{case['id']}__proof")
        if entry is None:
            # Counted and named, never passed over.
            missing.append(f"{case['id']}__proof")
            continue
        scored_slugs.add(entry["slug"])
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
        scored_slugs.add(playground_entry["slug"])
        playground = evaluate_call(
            playground_entry, playground_case["expected"], json.loads(playground_case["schema_text"])
        )
        playground["id"] = playground_case["case"]

    if missing:
        print("Call(s) NOT SCORED, so no figure below covers the recorded run:", file=sys.stderr)
        for name in missing:
            print(f"  missing {name}", file=sys.stderr)
        return 1

    unscored = sorted(set(recorded) - scored_slugs)
    if unscored:
        # A recorded call no case reaches is evidence nothing looked at, and it
        # would sit in the file affecting no total. Say so rather than ignore it.
        print(f"{len(unscored)} recorded call(s) matched no case and were NOT scored:", file=sys.stderr)
        for slug in unscored:
            print(f"  {slug}", file=sys.stderr)
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

    second, second_problems = score_second_pass(inputs, cases, recorded)
    if second_problems:
        print("\nThe second pass did not verify, so it was NOT scored:", file=sys.stderr)
        for line in second_problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"\nSecond call, {second['run'][:10]}. Three arms, one pre-registered rule:")
    print(
        f"  {'bar: ' + str(SECOND_PASS_RULE['min_total']) + ' of ' + str(total):<46}"
        f" and at most {SECOND_PASS_RULE['max_regressions']} broken"
    )
    for name in sorted(second["arms"]):
        arm = second["arms"][name]
        print(
            f"  {name}  {arm['label']:<32} {arm['fields_matched']} of {arm['fields_total']}"
            f"   fixed {arm['fixed']:>3}   broke {arm['broken']:>3}"
            f"   {'CLEARS' if arm['clears_bar'] else 'below the bar'}"
        )
    winner = second["winner"]
    if winner is None:
        print("\nNo arm cleared the bar, so the page publishes no second call.", file=sys.stderr)
        return 1
    won = second["arms"][winner]
    second_schema_valid = schema_valid + won["schema_valid"]
    # "First try" is a claim about attempts, so it is counted from attempts.
    # The first pass already required it; the second pass printed it and
    # checked only schema validity, so a retried reply satisfied the figure.
    second_first_try = first_call + sum(1 for case in won["cases"] if case["attempts"] == 1)
    second_responses = len(scored_calls) + len(won["cases"])
    print(
        f"\n{won['fields_matched']} of {won['fields_total']} fields exact after the second call, "
        f"up from {matched}: {won['fixed']} fixed, {won['broken']} broken"
    )
    print(f"{second_schema_valid} of {second_responses} replies valid against the schema, first try, across both passes")

    want = PAGE_FIGURES
    checks = {
        "fields": (matched, total) == (want["fields_matched"], want["fields_total"]),
        "documents": len(cases) == want["documents"],
        "calls": len(scored_calls) == want["calls"],
        "booleans": (booleans_matched, len(booleans)) == (want["booleans"], want["booleans"]),
        "schema valid on the first call": schema_valid == want["schema_valid"] and first_call == want["schema_valid"],
        "the arm the rule selected": winner == want["second_pass_arm"],
        "the second call's total": won["fields_matched"] == want["second_pass_matched"],
        "what the second call fixed and broke": (won["fixed"], won["broken"])
        == (want["second_pass_fixed"], want["second_pass_broken"]),
        "schema valid on the first try across both passes": second_schema_valid
        == want["schema_valid_both_passes"]
        and second_first_try == want["schema_valid_both_passes"]
        and second_responses == want["schema_valid_both_passes"],
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
