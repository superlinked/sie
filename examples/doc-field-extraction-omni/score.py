#!/usr/bin/env python3
"""Grade document-field-extraction replies with Omni's JSON accuracy. Python standard library only.

    python3 score.py --set confirm --published                     # regrade the published replies
    python3 score.py --set confirm --replies runs/confirm/replies.jsonl  # grade your own run

What it computes, per document, is the score of the grader the published study ran:

  1. Null fill, applied to the gold and to every reply alike: a key that the document's Omni JSON schema
     declares but the JSON leaves out is added with the value null. Strict-schema APIs return null for an
     absent field while others drop the key; nothing else is changed.
  2. Omni's calculateJsonAccuracy with ignoreCases false: 1 - (additions + deletions + modifications) / total
     gold fields, floored at 0 and rounded with JavaScript's toFixed(4). The diff is json-diff 1.0.6 with
     {sort: true}; its array alignment uses Python's difflib.SequenceMatcher, which json-diff's difflib port
     follows line for line.
  3. A reply that is not a JSON object or array, or a row that carries an "error", scores 0.

The mean is taken over every document in the set (100 pilot or 546 confirmation). A document with no row, an
error row, a capped reply or an unparsable reply counts as 0 in that denominator.

The original grader is JavaScript. This file reimplements it, reproducing JavaScript value semantics wherever
they change a score: number parsing (integers beyond 2**53 become doubles), object key order, strict equality
and the default Array.prototype.sort string order. On the published replies it gives every per-document record
of the published omni_scores.json exactly.
"""

# The diff and accuracy code below is a Python port of two MIT-licensed projects:
#
#   OmniAI OCR benchmark, src/evaluation/json.ts   https://github.com/getomni-ai/benchmark
#   json-diff 1.0.6, lib/index.js                  https://github.com/andreyvit/json-diff
#                                                  Copyright (c) 2015 Andrey Tarantsov
#
# Both are distributed under the MIT License:
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#   associated documentation files (the "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial
#   portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
#   LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
#   EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#   USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from fetch import DATA, PINS, SETS, fetch_published, load_omni_rows, study_ids

# ---- JavaScript value semantics --------------------------------------------------------------------------------

UNDEF = object()  # JavaScript undefined
SAFE = 2**53
# Names that the JavaScript `in` operator finds on Object.prototype. The original code tests membership with
# `in`, so these keys would behave differently there; they never occur in the Omni data and are refused here.
PROTO = {
    "constructor",
    "__defineGetter__",
    "__defineSetter__",
    "hasOwnProperty",
    "__lookupGetter__",
    "__lookupSetter__",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toString",
    "valueOf",
    "__proto__",
    "toLocaleString",
}
INDEX = re.compile(r"(0|[1-9][0-9]*)\Z")


class Unsupported(Exception):
    """An input whose JavaScript behaviour this port does not reproduce."""


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _int(text: str) -> int | float:
    value = int(text)
    return value if -SAFE <= value <= SAFE else float(text)


def js_parse(text: Any) -> Any:
    """JSON.parse: numbers become doubles, NaN and Infinity literals are rejected."""
    if text is None:
        text = "null"
    if not isinstance(text, str):
        raise TypeError("not a string")
    return json.loads(text, parse_int=_int, parse_constant=_no_constant)


def jtype(value: Any) -> str:
    if value is None:
        return "null"
    if value is UNDEF:
        return "undefined"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(type(value))


def truthy(value: Any) -> bool:
    if value is None or value is UNDEF or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0 and not math.isnan(value)
    if isinstance(value, str):
        return value != ""
    return True


def js_keys(obj: dict) -> list[str]:
    """Object.keys order: array-index keys ascending, then the rest in insertion order."""
    index = sorted((k for k in obj if INDEX.match(k) and int(k) < 2**32 - 1), key=int)
    seen = set(index)
    return index + [k for k in obj if k not in seen]


def js_in(key: str, obj: dict) -> bool:
    if key in obj:
        return True
    if key in PROTO:
        raise Unsupported(f"key {key!r} is inherited in JavaScript")
    return False


def number_to_string(x: float) -> str:
    """Number.prototype.toString() for a double (shortest round-trip digits, ECMAScript layout)."""
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "0"
    if x < 0:
        return "-" + number_to_string(-x)
    _, digit_tuple, exponent = Decimal(repr(float(x))).as_tuple()
    digits = "".join(map(str, digit_tuple))
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    digits = stripped.lstrip("0")
    k = len(digits)
    n = k + exponent  # x = 0.digits * 10**n
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    sign = "+" if e >= 0 else "-"
    return (digits if k == 1 else digits[0] + "." + digits[1:]) + "e" + sign + str(abs(e))


def to_string(value: Any) -> str:
    """String(value) for the scalars the default sort compares."""
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return number_to_string(float(value))


def sort_key(value: Any) -> bytes:
    """Array.prototype.sort() with no comparator compares String(x) by UTF-16 code units."""
    return to_string(value).encode("utf-16-be", "surrogatepass")


def same_key(value: Any) -> tuple:
    """An element key whose equality is JavaScript strict equality on these scalars."""
    t = jtype(value)
    return (t, float(value)) if t == "number" else (t, value)


# ---- json-diff 1.0.6, lib/index.js (MIT), the paths taken with options {sort: true} -----------------------------


def is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def object_diff(a: dict, b: dict) -> tuple[float, Any, bool]:
    result: dict[str, Any] = {}
    score: float = 0
    equal = True
    for key in js_keys(a):
        if not js_in(key, b):
            result[f"{key}__deleted"] = a[key]
            score -= 30
            equal = False
    for key in js_keys(b):
        if not js_in(key, a):
            result[f"{key}__added"] = b[key]
            score -= 30
            equal = False
    for key in js_keys(a):
        if key in b:
            score += 20
            c_score, c_result, c_equal = diff(a[key], b[key])
            if not c_equal:
                result[key] = c_result
                equal = False
            score += min(20, max(-10, c_score / 5))
    if equal:
        return 100 * max(len(a), 0.5), UNDEF, True
    return max(0, score), result, False


def find_matching_object(item: Any, index: int, fuzzy: dict[str, Any]) -> tuple[float, str] | None:
    best = None  # (score, key, index distance)
    for key, entry in fuzzy.items():
        if key == "__next":
            continue
        distance = abs(entry["index"] - index)
        if jtype(item) == jtype(entry["item"]):
            score = diff(item, entry["item"])[0]
            if best is None or score > best[0] or (score == best[0] and distance < best[2]):
                best = (score, key, distance)
    return None if best is None else (best[0], best[1])


def scalarize(array: list, originals: dict[str, Any], fuzzy: dict[str, Any] | None = None) -> list:
    matches: dict[int, str] = {}
    if fuzzy is not None:
        key_scores: dict[str, tuple[float, int]] = {}
        for index, item in enumerate(array):
            if is_scalar(item):
                continue
            best = find_matching_object(item, index, fuzzy)
            if best and (best[1] not in key_scores or best[0] > key_scores[best[1]][0]):
                key_scores[best[1]] = (best[0], index)
        for key, (_, index) in key_scores.items():
            matches[index] = key
    out = []
    for index, item in enumerate(array):
        if is_scalar(item):
            if isinstance(item, str) and (item in PROTO or item == "__next" or item.startswith("__$!SCALAR")):
                raise Unsupported(f"array string {item!r} collides with json-diff's internal keys")
            out.append(item)
        else:
            key = matches.get(index)
            if key is None:
                key = f"__$!SCALAR{originals['__next']}"
                originals["__next"] += 1
            originals[key] = {"item": item, "index": index}
            out.append(key)
    return out


def scalarized(item: Any, originals: dict[str, Any]) -> bool:
    return isinstance(item, str) and item in originals


def descalarize(item: Any, originals: dict[str, Any]) -> Any:
    return originals[item]["item"] if scalarized(item, originals) else item


def array_diff(a: list, b: list) -> tuple[float, Any, bool]:
    originals1: dict[str, Any] = {"__next": 1}
    seq1 = scalarize(a, originals1)
    originals2: dict[str, Any] = {"__next": originals1["__next"]}
    seq2 = scalarize(b, originals2, originals1)
    seq1.sort(key=sort_key)
    seq2.sort(key=sort_key)
    matcher = difflib.SequenceMatcher(None, [same_key(x) for x in seq1], [same_key(x) for x in seq2], autojunk=True)
    opcodes = matcher.get_opcodes()
    result: list = []
    score: float = 0
    equal = True
    for op, i1, i2, j1, j2 in opcodes:
        if op != "equal":
            equal = False
        if op == "equal":
            for i in range(i1, i2):
                item = seq1[i]
                if scalarized(item, originals1):
                    if not scalarized(item, originals2):
                        raise Unsupported("json-diff internal error path")
                    _, c_result, c_equal = diff(descalarize(item, originals1), descalarize(item, originals2))
                    if not c_equal:
                        result.append(["~", c_result])
                        equal = False
                    else:
                        result.append([" "])
                else:
                    result.append([" "])
                score += 10
        elif op == "delete":
            for i in range(i1, i2):
                result.append(["-", descalarize(seq1[i], originals1)])
                score -= 5
        elif op == "insert":
            for j in range(j1, j2):
                result.append(["+", descalarize(seq2[j], originals2)])
                score -= 5
        elif op == "replace":
            for i in range(i1, i2):
                result.append(["-", descalarize(seq1[i], originals1)])
                score -= 5
            for j in range(j1, j2):
                result.append(["+", descalarize(seq2[j], originals2)])
                score -= 5
    if equal or not opcodes:
        return 100, UNDEF, True
    return max(0, score), result, False


def diff(a: Any, b: Any) -> tuple[float, Any, bool]:
    ta, tb = jtype(a), jtype(b)
    if ta == tb == "object":
        return object_diff(a, b)
    if ta == tb == "array":
        return array_diff(a, b)
    if ta == tb and a == b:
        return 100, UNDEF, True
    return 0, {"__old": a, "__new": b}, False


# ---- OmniAI benchmark, src/evaluation/json.ts (MIT) ------------------------------------------------------------


def count_total_fields(obj: Any) -> int:
    count = 0

    def walk(current: Any) -> None:
        nonlocal count
        if not truthy(current) or not isinstance(current, (dict, list)):
            return
        if isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    walk(item)
                else:
                    count += 1
        else:
            for key in js_keys(current):
                if "__" in key:
                    continue
                value = current[key]
                if value is None or isinstance(value, (str, int, float)):  # bool is an int in Python
                    count += 1
                elif isinstance(value, (dict, list)):
                    walk(value)

    walk(obj)
    return count


def count_changes(diff_result: Any) -> dict[str, int]:
    changes = {"additions": 0, "deletions": 0, "modifications": 0, "total": 0}

    def walk(obj: Any) -> None:
        if not isinstance(obj, (dict, list)):
            return
        pairs = (
            ((k, obj[k]) for k in js_keys(obj)) if isinstance(obj, dict) else ((str(i), v) for i, v in enumerate(obj))
        )
        for key, value in pairs:
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, list) or len(item) != 2:
                        continue
                    operation, element = item
                    if not isinstance(element, (dict, list)):
                        if operation == "+":
                            changes["additions"] += 1
                        elif operation == "-":
                            changes["deletions"] += 1
                    elif operation == "+":
                        changes["additions"] += count_total_fields(element)
                    elif operation == "-":
                        changes["deletions"] += count_total_fields(element)
                    elif operation == "~":
                        walk(element)
            elif key.endswith("__deleted"):
                changes["deletions"] += count_total_fields(value) if isinstance(value, (dict, list)) else 1
            elif key.endswith("__added"):
                changes["additions"] += count_total_fields(value) if isinstance(value, (dict, list)) else 1
            elif isinstance(value, dict):
                if "__old" in value and "__new" in value:
                    if value["__old"] is None and value["__new"] is not None:
                        changes["modifications"] += count_total_fields(value["__new"]) or 1
                    else:
                        changes["modifications"] += count_total_fields(value["__old"]) or 1
                else:
                    walk(value)

    walk(diff_result)
    changes["total"] = changes["additions"] + changes["deletions"] + changes["modifications"]
    return changes


def to_fixed4(x: float) -> float:
    """Number(x.toFixed(4)): the nearest 4-decimal value to the exact double, ties to the larger."""
    if math.isnan(x):
        return x
    return float(Decimal(x).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def json_accuracy(actual: Any, predicted: Any) -> dict[str, Any]:
    result = diff(actual, predicted)[1]
    total_fields = count_total_fields(actual)
    if result is UNDEF:
        return {
            "score": 1,
            "stats": {"additions": 0, "deletions": 0, "modifications": 0, "total": 0},
            "totalFields": total_fields,
        }
    changes = count_changes(result)
    if total_fields == 0:
        raw = float("nan") if changes["total"] == 0 else 0.0
    else:
        raw = max(0, 1 - changes["total"] / total_fields)
    return {"score": to_fixed4(raw), "stats": changes, "totalFields": total_fields}


# ---- the study's null fill and row rules -----------------------------------------------------------------------


def schema_type(schema: Any) -> Any:
    t = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    return t


def fill_nulls(value: Any, schema: Any) -> Any:
    if not truthy(schema) or value is None or value is UNDEF:
        return value
    t = schema_type(schema)
    if t == "object" and isinstance(value, dict):
        out = dict(value)
        props = schema.get("properties")
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise Unsupported("schema properties is not an object")
        for key in js_keys(props):
            out[key] = fill_nulls(out[key], props[key]) if js_in(key, out) else None
        return out
    if t == "array" and isinstance(value, list) and truthy(schema.get("items", UNDEF)):
        return [fill_nulls(v, schema["items"]) for v in value]
    return value


def score_document(row: dict[str, Any] | None, omni_row: dict[str, Any]) -> dict[str, Any]:
    """One document's grader record. A missing row scores 0 like an error row."""
    schema = js_parse(omni_row["json_schema"])
    gold = fill_nulls(js_parse(omni_row["true_json_output"]), schema)
    if row is None:
        return {"score": 0, "parsed": False, "error": "no row", "totalFields": count_total_fields(gold)}
    parsed = None
    if not truthy(row.get("error")):
        try:
            parsed = js_parse(row.get("text"))
        except (ValueError, TypeError):
            parsed = None
    if not isinstance(parsed, (dict, list)):
        error = row.get("error")
        return {
            "score": 0,
            "parsed": False,
            "error": "not JSON" if error is None else error,
            "totalFields": count_total_fields(gold),
        }
    r = json_accuracy(gold, fill_nulls(parsed, schema))
    return {"score": r["score"], "parsed": True, "totalFields": r["totalFields"], "stats": r["stats"]}


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Grader rows {"id", "text", optional "error"}; a later row replaces an earlier error row for the same id."""
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):  # split on newline only, as the original did
        if not line.strip():
            continue
        row = json.loads(line)
        prev = rows.get(row["id"])
        if prev is None or truthy(prev.get("error")):
            rows[row["id"]] = row
    return rows


def score_set(set_name: str, rows: dict[str, dict[str, Any]], cache: Path) -> tuple[dict[str, dict], dict]:
    omni = load_omni_rows(cache)
    ids = study_ids(set_name)
    per = {i: score_document(rows.get(i), omni[i]) for i in ids}
    scores = [per[i]["score"] for i in ids]
    summary = {
        "set": set_name,
        "documents": len(ids),
        "mean": round(100 * sum(scores) / len(ids), 2),
        "mean_exact": 100 * sum(scores) / len(ids),
        "parsed": sum(1 for i in ids if per[i]["parsed"]),
        "rows_missing": sum(1 for i in ids if i not in rows),
        "rows_with_error": sum(1 for i in ids if i in rows and truthy(rows[i].get("error"))),
        "documents_scoring_1": sum(1 for s in scores if s == 1),
        "rows_outside_set_ignored": len(set(rows) - set(ids)),
    }
    return per, summary


def compare(per: dict[str, dict], expected: dict[str, dict]) -> dict[str, Any]:
    """Compare per-document records with a published omni_scores.json arm."""
    keys = ("score", "parsed", "totalFields", "stats")
    diffs = [i for i in per if i not in expected or any(per[i].get(k) != expected[i].get(k) for k in keys)]
    return {
        "documents_compared": len(per),
        "identical": len(per) - len(diffs),
        "different": diffs[:20],
        "expected_only": sorted(set(expected) - set(per))[:20],
    }


def main() -> None:
    """Grade a set of replies and, when given published scores, compare every document."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=SETS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replies", type=Path, help="grader rows, one JSON object per line")
    source.add_argument(
        "--published", action="store_true", help="regrade the published replies and compare with the published scores"
    )
    parser.add_argument("--expect", type=Path, help="an omni_scores.json to compare per-document records with")
    parser.add_argument("--expect-arm", default="A1")
    parser.add_argument("--arm", default="A1", help="arm name used in --out")
    parser.add_argument("--out", type=Path, help="write per-document records in omni_scores.json form")
    parser.add_argument("--cache", type=Path, default=DATA, help="download folder (default data/)")
    args = parser.parse_args()

    expected = None
    if args.published:
        published = fetch_published(args.set, args.cache)
        replies = published["replies"]
        expected = json.loads(published["scores"].read_text(encoding="utf-8"))["A1"]
    else:
        replies = args.replies
    if args.expect:
        expected = json.loads(args.expect.read_text(encoding="utf-8"))[args.expect_arm]
    try:
        per, summary = score_set(args.set, read_rows(replies), args.cache)
    except Unsupported as exc:
        raise SystemExit(f"unsupported input: {exc}") from exc
    report: dict[str, Any] = {"summary": summary}
    if expected is not None:
        report["per_document_check"] = compare(per, expected)
        report["published_mean"] = PINS["sets"][args.set]["published_mean"]
    if args.out:
        args.out.write_text(json.dumps({args.arm: per}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    check = report.get("per_document_check")
    if check and (check["different"] or check["expected_only"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
