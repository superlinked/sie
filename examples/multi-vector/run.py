#!/usr/bin/env python3
"""Encode questions and candidate passages as token vectors with SIE Cloud.

    uv run python run.py                          # every pinned search
    uv run python run.py --case mdn-cache-no-store
    python3 run.py --show mdn-cache-no-store      # print the requests, no network

Two calls per search, the pair the /multi-vector task page shows:

    POST https://api.superlinked.com/v1/encode/lightonai%2FGTE-ModernColBERT-v1
    1. the question alone, with options.is_query true
    2. its four candidate passages in one call, with options.is_query false

A ColBERT model returns one vector per token rather than one per text, so the
scoring happens client-side: every query token takes its best match in a
passage and they sum. score.py does that and nothing else.

Results go to --output as a manifest.json and a calls.json in the dataset's own
shape, one entry per call holding the request, the response, the HTTP status,
the served model revision and the round-trip time. The key comes from
SIE_API_KEY and is never written out.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
scores them offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sie_sdk import SIEClient

import maxsim

HERE = Path(__file__).resolve().parent


def multivector_items(results: list[dict[str, Any]], body: dict[str, Any]) -> list[dict[str, Any]]:
    """The returned token vectors, in the shape the recorded responses carry.

    The SDK hands back each multivector as a numpy array of shape
    [num_tokens, token_dims]. `tolist()` gives the same values the wire carried.
    """
    if len(results) != len(body["items"]):
        raise SystemExit(f"Sent {len(body['items'])} items and got {len(results)} back")
    items = []
    for sent, result in zip(body["items"], results):
        if result.get("id") != sent["id"]:
            raise SystemExit(f"Response item id {result.get('id')!r} does not match request item {sent['id']!r}")
        block = result["multivector"]
        values = block.tolist() if hasattr(block, "tolist") else [list(row) for row in block]
        if not values:
            raise SystemExit(f"{sent['id']}: no token vectors returned")
        items.append(
            {
                "id": sent["id"],
                "multivector": {
                    "dtype": str(getattr(block, "dtype", "float32")),
                    "num_tokens": len(values),
                    "token_dims": len(values[0]),
                    "values": values,
                },
            }
        )
    return items


def record(client: SIEClient, case: dict[str, Any], kind: str, body: dict[str, Any]) -> dict[str, Any]:
    """Send one encode call and return its calls.json entry."""
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    results = client.encode(
        maxsim.MODEL,
        list(body["items"]),
        output_types=list(body["params"]["output_types"]),
        is_query=body["params"]["options"]["is_query"],
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    if not isinstance(results, list):
        results = [results]
    response = {"items": multivector_items(results, body), "model": maxsim.MODEL}
    return {
        "slug": f"{case['slug']}.{kind}",
        "case": case["slug"],
        "kind": kind,
        "requested_at": requested_at,
        "request": {
            "method": "POST",
            # The URL the SDK actually used, not this module's default, so a run
            # against a regional endpoint records where it really went.
            "url": client.base_url.rstrip("/") + maxsim.ENCODE_PATH,
            "body": body,
        },
        "status": 200,
        "response": response,
        "response_sha256": maxsim.sha256_bytes(maxsim.compact_json(response)),
        "model_revision": client.last_model_revision,
        "timing": {"duration_ms": elapsed_ms},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode the pinned searches as token vectors")
    parser.add_argument("--case", action="append", default=[], help="slug to run; repeatable, default all")
    parser.add_argument("--show", metavar="SLUG", help="print both request bodies for one search and exit")
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="directory for manifest.json and calls.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_doc = maxsim.load_cases()
    cases = {case["slug"]: case for case in cases_doc["cases"]}

    if args.show:
        case = cases.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}. Known: {', '.join(cases)}")
        print(json.dumps(maxsim.query_body(case), indent=2, ensure_ascii=False))
        print("# call two, the four candidate passages:")
        print(json.dumps(maxsim.passages_body(case), indent=2, ensure_ascii=False))
        return 0

    selected = args.case or list(cases)
    unknown = [slug for slug in selected if slug not in cases]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
    duplicated = sorted({slug for slug in selected if selected.count(slug) > 1})
    if duplicated:
        raise SystemExit(f"Repeated --case: {', '.join(duplicated)}")

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY. To check the published figure without a key, run score.py instead.")

    client = SIEClient(os.environ.get("SIE_BASE_URL", maxsim.ENDPOINT), api_key=api_key, timeout_s=900)
    base_url = client.base_url.rstrip("/")
    print(f"endpoint {base_url}{maxsim.ENCODE_PATH}")
    print(f"model    {maxsim.MODEL}")

    entries = []
    for slug in selected:
        case = cases[slug]
        for kind, body in (("query", maxsim.query_body(case)), ("passages", maxsim.passages_body(case))):
            entries.append(record(client, case, kind, body))
        print(
            f"{slug:<32} {entries[-2]['timing']['duration_ms']:>7.0f} ms + {entries[-1]['timing']['duration_ms']:>7.0f} ms"
        )

    revisions = sorted({entry["model_revision"] for entry in entries if entry["model_revision"]})
    manifest = {
        "task": "multi-vector",
        "page": "https://superlinked.com/multi-vector",
        "endpoint": base_url,
        "path": maxsim.ENCODE_PATH,
        "model": maxsim.MODEL,
        "model_revision": revisions[0] if len(revisions) == 1 else revisions,
        "run_date": datetime.now(UTC).date().isoformat(),
        "recorded_by": "examples/multi-vector/run.py",
        "metric": (
            "MaxSim, computed client-side: the sum over query tokens of the largest dot product against the "
            "passage token vectors. The server returns L2-normalised vectors, so the dot product is the cosine."
        ),
        "cases_published": len(selected),
        "passages_per_case": len(cases[selected[0]]["passages"]),
        "calls_recorded": len(entries),
        "response_sha256": (
            "sha256 of json.dumps(response, ensure_ascii=False, separators=(',', ':')).encode('utf-8')"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Compact, because this file is almost entirely float arrays.
    (args.output / "calls.json").write_text(
        json.dumps({"calls": entries}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}/manifest.json and {args.output}/calls.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
