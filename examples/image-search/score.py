#!/usr/bin/env python3
"""Re-derive the published ranking from the recorded run, offline.

    python3 fetch.py
    python3 score.py

Standard library only. No API key, no network, no inference spend. Every number
below comes out of the two recorded encode responses in evidence/calls.json.

This fails rather than skipping. A missing file, an image whose bytes no longer
match their digest, a response that does not match its response_sha256, a
request the pinned inputs do not rebuild, an image with no recorded vector, a
vector returned twice, a declared width that disagrees with its own values, a
non-finite value, or a ranking that no longer puts the handbag first all exit
non-zero.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import ranking

TARGET = "red-leather-handbag"


def load_calls() -> dict[str, dict[str, Any]]:
    path = ranking.EVIDENCE / "calls.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    calls = json.loads(path.read_text(encoding="utf-8"))["calls"]
    by_slug: dict[str, dict[str, Any]] = {}
    for call in calls:
        if call["slug"] in by_slug:
            raise SystemExit(f"calls.json records {call['slug']!r} twice")
        by_slug[call["slug"]] = call
    missing = {"images", "query"} - set(by_slug)
    if missing:
        raise SystemExit(f"calls.json is missing {', '.join(sorted(missing))}")
    if set(by_slug) != {"images", "query"}:
        raise SystemExit(
            f"calls.json holds a call nothing scores: {', '.join(sorted(set(by_slug) - {'images', 'query'}))}"
        )
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
    digest = ranking.sha256_bytes(ranking.compact_json(call["response"]))
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
        if len(values) != ranking.DIMS:
            raise SystemExit(f"{item['id']}: {len(values)} dimensions, the model returns {ranking.DIMS}")
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
    manifest_path = ranking.EVIDENCE / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is missing. Run: python3 fetch.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != ranking.MODEL:
        raise SystemExit(f"manifest names model {manifest.get('model')!r}, this example scores {ranking.MODEL!r}")
    if manifest.get("model_revision") != ranking.MODEL_REVISION:
        raise SystemExit(
            f"manifest names model revision {manifest.get('model_revision')!r}, "
            f"this example scores {ranking.MODEL_REVISION!r}. Different weights produce different scores."
        )

    records = ranking.load_images()
    query = ranking.load_query()
    bodies = ranking.bodies(records, query)
    calls = load_calls()

    for slug, body in bodies.items():
        check_call(calls[slug], body, manifest)

    image_vectors = vectors(calls["images"], [record["id"] for record in records])
    query_vector = next(iter(vectors(calls["query"], [query["id"]]).values()))

    labels = {record["id"]: record["label"] for record in records}
    ranked = ranking.rank(query_vector, image_vectors)

    print(f"{manifest['model']} at revision {manifest['model_revision']}")
    print(f"{manifest['endpoint']}, recorded {manifest['run_date']}")
    print(f"query: {query['text']!r}")
    print()
    print(f"{'rank':>4}  {'score':>8}  image")
    for position, (name, score) in enumerate(ranked, start=1):
        marker = " <- the query's subject" if name == TARGET else ""
        print(f"{position:>4}  {score:>8.3f}  {labels[name]}{marker}")
    print()

    derived_order = tuple(name for name, _ in ranked)
    if derived_order != ranking.EXPECTED_ORDER:
        raise SystemExit(
            "The ranking is not the published one.\n"
            f"  published: {', '.join(ranking.EXPECTED_ORDER)}\n"
            f"  derived:   {', '.join(derived_order)}"
        )

    top_name, top_score = ranked[0]
    if top_name != TARGET:
        raise SystemExit(f"The top result is {top_name!r}, not {TARGET!r}. The published claim no longer holds.")
    runner_up_name, runner_up_score = ranked[1]
    print(f"{len(records)} photographs, {ranking.DIMS}-dimensional vectors, one text query")
    print(f"the red leather handbag ranks first at {top_score:.3f}")
    print(f"the closest other photograph is the {labels[runner_up_name].lower()} at {runner_up_score:.3f}")
    gap = f"a gap of {top_score - runner_up_score:.3f}"
    # A cosine can legitimately be zero or negative, and neither divides into a
    # meaningful multiple. Only the ratio is conditional; the gap always prints.
    if runner_up_score > 0:
        print(f"{gap}, or {top_score / runner_up_score:.1f}x")
    else:
        print(gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
