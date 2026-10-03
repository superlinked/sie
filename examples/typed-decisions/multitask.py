#!/usr/bin/env python3
"""One GLiFormer call per CVE record: entities, relations and a typed record together.

    python3 multitask.py --show                                  # the requests, offline
    uv run python multitask.py --record --url http://localhost:8080 --split test
    python3 multitask.py --score --calls run-output/multitask--test.json

Standard library only, apart from `sie_sdk` for --record. Everything below was
fixed before the first call.

Records
    From the vulnerability-triage case file, one split (test for the published
    run). Records are ordered by sha256("sie-typed-decisions-multitask-v1:" +
    CVE id); the first 12 whose NVD entry lists exactly one vulnerable CPE
    vendor and product, both written in the description (after lowercasing
    and keeping only letters and digits), are taken. NVD's CPE names are often
    not in the text at all, and a name the text never gives cannot be found.

Calls, all to knowledgator/gliformer-large-v1 through SIE extract
    combined    one call: entity labels vendor, product and version; relation
                types "made by" and "affects version"; and an output_schema
                with `affected_component` (a string) and root enums `weakness`
                and `attack_vector` (the option names of those two questions),
                every task at threshold THRESHOLD. On dev, 0.1 found the most
                products (12 of 12, against 11 at 0.2 and 10 at 0.3), with
                either wording of the entity labels.
    separate    the same three tasks as three calls: entities alone, entities
                with relations, and the output_schema alone.
    encode      every record of the split, embedded with the same model.

What is scored
    product            found when an extracted product span matches the NVD CPE
                       product: after lowercasing and dropping everything but
                       letters and digits, the two are equal or one contains
                       the other and the shorter has at least 4 characters.
                       Rule: at least 10 of 12.
    weakness           the combined call's weakness enum against NVD gold, and
                       against the separate schema call. Rule: the two modes
                       agree on at least 11 of 12.
    reported, no rule  vendor, matched the same way, and the attack_vector enum.
                       On dev the combined call found the vendor for 3 of 12
                       (it folds the vendor into the product span; the entities
                       call alone finds most of them) and returned an
                       attack_vector for 1 of 12, so neither is claimed.
    neighbours         for every record of the split, whether its nearest other
                       record by cosine similarity has the same weakness class,
                       against the rate expected by chance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from questions import VULNERABILITY_QUESTIONS, VULNERABILITY_TRIAGE

MODEL = "knowledgator/gliformer-large-v1"
SALT = "sie-typed-decisions-multitask-v1:"
RECORDS = 12
ENTITY_LABELS = ["vendor", "product", "version"]
RELATIONS = ["made by", "affects version"]
# The option GLiFormer reads relation types from. Older adapter builds called it
# `relations`; `--relations-key` records against one of those on dev only.
RELATIONS_KEY = "relation_labels"
FOUND_AT_LEAST = 10
THRESHOLD = 0.1
AGREE_AT_LEAST = 11
MIN_CONTAINED = 4

ENUM_NAMES = {
    qid: {option["name"]: option["key"] for option in VULNERABILITY_QUESTIONS[qid]["options"]}
    for qid in ("weakness", "attack_vector")
}
SCHEMA = {
    "type": "object",
    "properties": {
        "affected_component": {"type": "string"},
        "weakness": {"type": "string", "enum": list(ENUM_NAMES["weakness"])},
        "attack_vector": {"type": "string", "enum": list(ENUM_NAMES["attack_vector"])},
    },
}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def eligible(case: dict[str, Any]) -> bool:
    pairs = case["source"]["cpe_vendor_product"]
    if len(pairs) != 1:
        return False
    text = normalise(case["state"])
    return all(normalise(name) in text for name in pairs[0])


def selected(cases: dict[str, Any], split: str) -> list[dict[str, Any]]:
    pool = [case for case in cases["cases"] if case["split"] == split]
    ordered = sorted(pool, key=lambda case: hashlib.sha256((SALT + case["slug"]).encode()).hexdigest())
    return [case for case in ordered if eligible(case)][:RECORDS]


def requests_for(
    text: str, threshold: float = THRESHOLD, relations_key: str = RELATIONS_KEY
) -> dict[str, dict[str, Any]]:
    item = {"text": text}
    options: dict[str, Any] = {"threshold": threshold}
    return {
        "combined": {
            "items": [item],
            "params": {
                "labels": ENTITY_LABELS,
                "output_schema": SCHEMA,
                "options": {**options, relations_key: RELATIONS},
            },
        },
        "entities": {"items": [item], "params": {"labels": ENTITY_LABELS, "options": options}},
        "relations": {
            "items": [item],
            "params": {"labels": ENTITY_LABELS, "options": {**options, relations_key: RELATIONS}},
        },
        "schema": {"items": [item], "params": {"output_schema": SCHEMA, "options": options}},
    }


def matches(span: str, cpe: str) -> bool:
    a, b = normalise(span), normalise(cpe)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= MIN_CONTAINED and shorter in longer


def record(args: argparse.Namespace, cases: dict[str, Any]) -> int:
    from sie_sdk import SIEClient  # noqa: PLC0415 - only --record needs the SDK

    client = SIEClient(args.url, timeout_s=900)
    calls = []
    for case in selected(cases, args.split):
        entry: dict[str, Any] = {
            "case": case["slug"],
            "requests": requests_for(case["state"], args.threshold, args.relations_key),
            "responses": {},
            "ms": {},
        }
        for name, body in entry["requests"].items():
            params = body["params"]
            started = time.perf_counter()
            result = client.extract(
                MODEL,
                body["items"][0],
                labels=params.get("labels"),
                output_schema=params.get("output_schema"),
                options=params.get("options"),
            )
            entry["ms"][name] = round((time.perf_counter() - started) * 1000, 2)
            entry["responses"][name] = {key: result.get(key) for key in ("entities", "relations", "data", "error")}
        calls.append(entry)
    pool = [case for case in cases["cases"] if case["split"] == args.split]
    started = time.perf_counter()
    vectors = client.encode(MODEL, [{"text": case["state"]} for case in pool])
    encode_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = {
        "model": MODEL,
        "split": args.split,
        "threshold": args.threshold,
        "relations_key": args.relations_key,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "calls": calls,
        "encode": {
            "cases": [case["slug"] for case in pool],
            "dense": [list(map(float, v["dense"])) for v in vectors],
            "ms": encode_ms,
        },
    }
    out = Path(args.out or f"run-output/multitask--{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(calls)} records, {len(pool)} embeddings")
    return 0


def spans(response: dict[str, Any], label: str) -> list[str]:
    return [entity["text"] for entity in response.get("entities") or [] if entity["label"] == label]


def enum_answer(response: dict[str, Any], qid: str) -> str | None:
    value = (response.get("data") or {}).get(qid)
    return ENUM_NAMES[qid].get(value) if isinstance(value, str) else None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def score(args: argparse.Namespace, cases: dict[str, Any]) -> int:
    payload = load(Path(args.calls))
    by_slug = {case["slug"]: case for case in cases["cases"]}
    expected = [case["slug"] for case in selected(cases, payload["split"])]
    if [call["case"] for call in payload["calls"]] != expected:
        raise SystemExit("the recorded records are not the pre-registered selection")
    found = {"vendor": 0, "product": 0}
    agree = {qid: 0 for qid in ENUM_NAMES}
    right = {qid: 0 for qid in ENUM_NAMES}
    relations = components = 0
    for call in payload["calls"]:
        case = by_slug[call["case"]]
        if call["requests"] != requests_for(
            case["state"], payload["threshold"], payload.get("relations_key", RELATIONS_KEY)
        ):
            raise SystemExit(f"{call['case']}: recorded requests differ from the rebuilt ones")
        ((vendor, product),) = case["source"]["cpe_vendor_product"]
        combined = call["responses"]["combined"]
        found["vendor"] += any(matches(span, vendor) for span in spans(combined, "vendor"))
        found["product"] += any(matches(span, product) for span in spans(combined, "product"))
        relations += bool(combined.get("relations"))
        components += bool((combined.get("data") or {}).get("affected_component"))
        for qid in ENUM_NAMES:
            answer = enum_answer(combined, qid)
            agree[qid] += answer == enum_answer(call["responses"]["schema"], qid)
            right[qid] += answer == case["gold"][qid]
        print(
            f"{call['case']}: cpe {vendor}/{product}; vendor {spans(combined, 'vendor')} product "
            f"{spans(combined, 'product')}; {len(combined.get('relations') or [])} relations; "
            f"weakness {enum_answer(combined, 'weakness')} (gold {case['gold']['weakness']}); "
            f"{call['ms']['combined']:.0f} ms"
        )
    n = len(payload["calls"])
    print(
        f"\nproduct found {found['product']} of {n} (rule: at least {FOUND_AT_LEAST}); vendor found {found['vendor']} of {n}"
    )
    for qid in ENUM_NAMES:
        print(
            f"{qid}: matches NVD {right[qid]} of {n}; combined and separate agree {agree[qid]} of {n}"
            + (f" (rule: at least {AGREE_AT_LEAST})" if qid == "weakness" else "")
        )
    print(f"records with at least one relation: {relations} of {n}; with an affected_component: {components} of {n}")

    encoded = payload["encode"]
    weakness = [by_slug[slug]["gold"]["weakness"] for slug in encoded["cases"]]
    same = 0
    for i, vector in enumerate(encoded["dense"]):
        nearest = max(
            (j for j in range(len(encoded["dense"])) if j != i), key=lambda j: cosine(vector, encoded["dense"][j])
        )
        same += weakness[i] == weakness[nearest]
    total = len(weakness)
    chance = sum(weakness.count(w) - 1 for w in weakness) / (total * (total - 1))
    print(f"nearest neighbour has the same weakness class: {same} of {total} (chance {chance:.3f})")
    passed = found["product"] >= FOUND_AT_LEAST and agree["weakness"] >= AGREE_AT_LEAST
    print("\nrules " + ("hold" if passed else "do NOT hold"))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--calls", help="recording --score reads")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="dev only; test uses THRESHOLD")
    parser.add_argument("--relations-key", default=RELATIONS_KEY, help="dev only; test uses RELATIONS_KEY")
    parser.add_argument("--out")
    args = parser.parse_args()
    cases = load(Path(args.data) / f"inputs/{VULNERABILITY_TRIAGE}/cases.json")
    if args.record:
        if args.split == "test" and (args.threshold, args.relations_key) != (THRESHOLD, RELATIONS_KEY):
            raise SystemExit(f"the test run uses threshold {THRESHOLD} and {RELATIONS_KEY}")
        return record(args, cases)
    if args.score:
        return score(args, cases)
    for case in selected(cases, args.split):
        print(case["slug"], case["source"]["cpe_vendor_product"])
    print(json.dumps(requests_for("<record text>")["combined"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
