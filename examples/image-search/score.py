#!/usr/bin/env python3
"""Re-derive the published figures from the recorded run, offline.

    python3 fetch.py
    python3 score.py

Standard library only. No API key, no network, no inference spend. Every number
below comes out of the recorded encode responses in evidence/calls.json.

What it measures, over every request in the catalogue:

  ranks first   how often the one photograph matching colour, material and
                category is the top result, under each of the four request
                forms. The catalogue is built so that for every request there
                is also a photograph matching each pair of the three, so first
                place has to be taken from a near miss rather than from noise.

  median rank   where that photograph lands when it is not first.

This fails rather than skipping. A missing file, an image whose bytes no longer
match their digest, a response that does not match its response_sha256, a
request the pinned inputs do not rebuild, an image or query with no recorded
vector, a vector returned twice, a declared width that disagrees with its own
values, or a non-finite value all exit non-zero. So does any published figure
that the recording no longer supports.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import catalogue

# Published on https://superlinked.com/image-search. Pinned in the committed
# source so the scorer compares what it derives from the recording against
# something the recording cannot move. Editing evidence/ alone will not satisfy
# this. Filled in from the run; `None` means "not published".
PUBLISHED: dict[str, int] = {
    "images": 50,
    "requests": 24,
    "first/category": 2,
    "first/colour-category": 7,
    "first/material-category": 6,
    "first/full": 13,
}

KIND_ORDER = ("full", "wrong-colour", "wrong-material", "wrong-category", "one-attribute", "unrelated")


def load_calls(expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = catalogue.EVIDENCE / "calls.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    calls = json.loads(path.read_text(encoding="utf-8"))["calls"]
    by_slug: dict[str, dict[str, Any]] = {}
    for call in calls:
        if call["slug"] in by_slug:
            raise SystemExit(f"calls.json records {call['slug']!r} twice")
        by_slug[call["slug"]] = call
    missing = sorted(set(expected) - set(by_slug))
    if missing:
        raise SystemExit(f"calls.json is missing {', '.join(missing)}")
    extra = sorted(set(by_slug) - set(expected))
    if extra:
        raise SystemExit(f"calls.json holds a call nothing scores: {', '.join(extra)}")
    return by_slug


def check_call(call: dict[str, Any], expected_body: dict[str, Any], manifest: dict[str, Any]) -> None:
    if call["status"] != 200:
        raise SystemExit(f"{call['slug']}: recorded status {call['status']}")
    if call["request"]["body"] != expected_body:
        raise SystemExit(
            f"{call['slug']}: the recorded request is not the one the pinned inputs rebuild. "
            "Either inputs/ changed or the recording is of something else."
        )
    expected_url = manifest["endpoint"].rstrip("/") + manifest["path"]
    if call["request"]["url"] != expected_url:
        raise SystemExit(f"{call['slug']}: recorded URL {call['request']['url']} is not {expected_url}")
    digest = catalogue.sha256_bytes(catalogue.compact_json(call["response"]))
    if digest != call["response_sha256"]:
        raise SystemExit(f"{call['slug']}: response digest {digest} is not the recorded {call['response_sha256']}")
    if call.get("deployment_revision") != manifest.get("deployment_revision"):
        raise SystemExit(
            f"{call['slug']}: served deployment revision {call.get('deployment_revision')!r} "
            f"is not the manifest's {manifest.get('deployment_revision')!r}"
        )


def vectors(call: dict[str, Any], expected_ids: list[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for item in call["response"]["items"]:
        if item["id"] in found:
            raise SystemExit(f"{call['slug']}: {item['id']!r} has two recorded vectors")
        values = item["dense"]["values"]
        if item["dense"]["dims"] != len(values):
            raise SystemExit(f"{item['id']}: declares {item['dense']['dims']} dimensions and carries {len(values)}")
        if len(values) != catalogue.DIMS:
            raise SystemExit(f"{item['id']}: {len(values)} dimensions, the model returns {catalogue.DIMS}")
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"{item['id']}: a returned value is not finite")
        found[item["id"]] = values
    absent = [name for name in expected_ids if name not in found]
    if absent:
        raise SystemExit(f"{call['slug']}: no recorded vector for {', '.join(absent)}")
    extra = [name for name in found if name not in expected_ids]
    if extra:
        raise SystemExit(f"{call['slug']}: a vector nothing asked for: {', '.join(extra)}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emit", help="write the derived figures as JSON to this path")
    args = parser.parse_args()

    manifest_path = catalogue.EVIDENCE / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is missing. Run: python3 fetch.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != catalogue.MODEL:
        raise SystemExit(f"manifest names model {manifest.get('model')!r}, this example scores {catalogue.MODEL!r}")
    if manifest.get("model_revision") != catalogue.MODEL_REVISION:
        raise SystemExit(
            f"manifest names model revision {manifest.get('model_revision')!r}, "
            f"this example scores {catalogue.MODEL_REVISION!r}. Different weights produce different scores."
        )

    records = catalogue.load_catalogue()
    queries = catalogue.load_queries()
    bodies = catalogue.bodies(records, queries)
    calls = load_calls(bodies)
    for slug, body in bodies.items():
        check_call(calls[slug], body, manifest)

    image_vectors: dict[str, list[float]] = {}
    image_slugs = [name for name in bodies if name != "queries"]
    batches = catalogue.image_batches(records)
    if len(image_slugs) != len(batches):
        raise SystemExit(f"{len(image_slugs)} recorded image calls for {len(batches)} batches of photographs")
    for slug, batch in zip(image_slugs, batches, strict=True):
        image_vectors.update(vectors(calls[slug], [record["id"] for record in batch]))
    if len(image_vectors) != len(records):
        raise SystemExit(f"{len(image_vectors)} recorded image vectors for {len(records)} photographs")

    items = catalogue.query_items(queries)
    query_vectors = vectors(calls["queries"], [item["id"] for item in items])

    by_id = {record["id"]: record for record in records}
    # Every request must have exactly one photograph matching all three
    # attributes, and at least one matching each pair. Checked before any
    # figure is derived, because a request without a near miss on some axis
    # makes "first place was taken from a near miss" false for that request.
    for query in queries:
        kinds: dict[str, int] = {}
        for record in records:
            kinds[catalogue.competitor_kind(record, query)] = (
                kinds.get(catalogue.competitor_kind(record, query), 0) + 1
            )
        if kinds.get("full", 0) != 1:
            raise SystemExit(f"{query['id']}: {kinds.get('full', 0)} photographs match all three attributes")
        for kind in ("wrong-colour", "wrong-material", "wrong-category"):
            if kinds.get(kind, 0) < 1:
                raise SystemExit(f"{query['id']}: the catalogue holds no {kind} near miss")
        if query["target"] not in by_id:
            raise SystemExit(f"{query['id']}: target {query['target']} is not in the catalogue")
        if catalogue.competitor_kind(by_id[query["target"]], query) != "full":
            raise SystemExit(f"{query['id']}: target {query['target']} does not match all three attributes")

    derived: dict[str, Any] = {"images": len(records), "requests": len(queries)}
    per_form: dict[str, dict[str, Any]] = {}
    beaten_by: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for form in catalogue.FORMS:
        positions = []
        for query in queries:
            ranked = catalogue.rank(query_vectors[f"{query['id']}/{form}"], image_vectors)
            order = [name for name, _ in ranked]
            position = order.index(query["target"]) + 1
            positions.append(position)
            if form == "full" and position != 1:
                beaten = catalogue.competitor_kind(by_id[order[0]], query)
                beaten_by[beaten] = beaten_by.get(beaten, 0) + 1
            rows.append(
                {
                    "query": query["id"],
                    "form": form,
                    "text": catalogue.phrase(form, query["colour"], query["material"], query["category"]),
                    "target": query["target"],
                    "rank": position,
                    "score": dict(ranked)[query["target"]],
                    "top": order[0],
                    "top_kind": catalogue.competitor_kind(by_id[order[0]], query),
                }
            )
        first = sum(1 for position in positions if position == 1)
        per_form[form] = {
            "first": first,
            "median_rank": statistics.median(positions),
            "mean_rank": round(statistics.fmean(positions), 2),
            "worst_rank": max(positions),
        }
        derived[f"first/{form}"] = first

    print(f"{manifest['model']} at revision {manifest['model_revision']}")
    print(f"{manifest['endpoint']}, recorded {manifest['run_date']}")
    print()
    print(f"{len(records)} photographs, {len(queries)} requests, {len(catalogue.FORMS)} forms each, "
          f"{len(calls)} calls recorded")
    print()
    print(f"{'request form':<34}{'ranks first':>12}{'median rank':>13}{'worst':>7}")
    labels = {
        "category": "the category alone",
        "colour-category": "colour and category",
        "material-category": "material and category",
        "full": "colour, material and category",
    }
    for form in catalogue.FORMS:
        stats = per_form[form]
        print(f"  {labels[form]:<32}{stats['first']:>7}/{len(queries):<4}"
              f"{stats['median_rank']:>13}{stats['worst_rank']:>7}")
    print()
    if beaten_by:
        print("When the full request did not win, what took first place:")
        for kind in KIND_ORDER:
            if kind in beaten_by:
                print(f"  {kind:<16}{beaten_by[kind]}")
    else:
        print("The full request took first place for every one of the requests.")

    if args.emit:
        Path(args.emit).write_text(
            json.dumps({"derived": derived, "per_form": per_form, "beaten_by": beaten_by, "rows": rows}, indent=1)
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.emit}")

    mismatched = [key for key, value in PUBLISHED.items() if derived.get(key) != value]
    if mismatched:
        for key in mismatched:
            print(f"MISMATCH {key}: recording gives {derived.get(key)}, the page publishes {PUBLISHED[key]}",
                  file=sys.stderr)
        return 1
    print("\nEvery published figure matches the recording.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
