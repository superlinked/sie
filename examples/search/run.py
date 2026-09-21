#!/usr/bin/env python3
"""Embed a corpus and a set of questions with SIE Cloud.

    uv run python run.py                 # the whole corpus and every question
    python3 run.py --show corpus-000     # print a request, no network
    python3 run.py --show queries

Batched calls, the call the /search task page shows:

    POST https://api.superlinked.com/v1/encode/Snowflake/snowflake-arctic-embed-l-v2.0
    passages: {"items": [...24...], "params": {"output_types": ["dense"]}}
    questions: the same, plus {"options": {"is_query": true}}

The questions set is_query and the passages do not. That is the whole
asymmetry: the model encodes a question and a document differently, and the
ranking in score.py is a plain cosine over what comes back.

Results go to --output as a manifest.json and a calls.json in the dataset's own
shape, one entry per call holding the request, the response, the HTTP status,
the served model revision and the round-trip time. The key comes from
SIE_API_KEY and is never written out.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
ranks them offline.
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

import retrieval

HERE = Path(__file__).resolve().parent


def dense_items(results: list[dict[str, Any]], body: dict[str, Any]) -> list[dict[str, Any]]:
    """The returned vectors, in the shape the recorded responses carry.

    The SDK hands back each dense embedding as a numpy array. `tolist()` gives
    the same float32 values the wire carried, one row per item, in request order.
    """
    if len(results) != len(body["items"]):
        raise SystemExit(f"Sent {len(body['items'])} items and got {len(results)} back")
    items = []
    for sent, result in zip(body["items"], results):
        if result.get("id") != sent["id"]:
            raise SystemExit(f"Response item id {result.get('id')!r} does not match request item {sent['id']!r}")
        vector = result["dense"]
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        items.append(
            {
                "dense": {"dims": len(values), "dtype": str(getattr(vector, "dtype", "float32")), "values": values},
                "id": sent["id"],
            }
        )
    return items


def record(client: SIEClient, name: str, body: dict[str, Any]) -> dict[str, Any]:
    """Send one encode call and return its calls.json entry."""
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    results = client.encode(
        retrieval.MODEL,
        list(body["items"]),
        output_types=list(body["params"]["output_types"]),
        is_query=body["params"].get("options", {}).get("is_query"),
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    if not isinstance(results, list):
        results = [results]
    response = {"items": dense_items(results, body), "model": retrieval.MODEL}
    return {
        "slug": name,
        "kind": "queries" if name == "queries" else "corpus-batch",
        "requested_at": requested_at,
        "request": {
            "method": "POST",
            # The URL the SDK actually used, not this module's default, so a run
            # against a regional endpoint records where it really went.
            "url": client.base_url.rstrip("/") + retrieval.ENCODE_PATH,
            "body": body,
        },
        "status": 200,
        "response": response,
        "response_sha256": retrieval.sha256_bytes(retrieval.compact_json(response)),
        "model_revision": client.last_model_revision,
        "timing": {"duration_ms": elapsed_ms},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode the pinned corpus and questions")
    parser.add_argument("--show", metavar="NAME", help="print one request body and exit, without calling anything")
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="directory for manifest.json and calls.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    passages = retrieval.load_corpus()
    queries = retrieval.load_queries(passages)
    bodies = dict(retrieval.corpus_bodies(passages))
    bodies["queries"] = retrieval.query_body(queries)

    if args.show:
        body = bodies.get(args.show)
        if body is None:
            raise SystemExit(f"Unknown call: {args.show}. Known: {', '.join(bodies)}")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY. To check the published figure without a key, run score.py instead.")

    client = SIEClient(os.environ.get("SIE_BASE_URL", retrieval.ENDPOINT), api_key=api_key, timeout_s=900)
    base_url = client.base_url.rstrip("/")
    print(f"endpoint {base_url}{retrieval.ENCODE_PATH}")
    print(f"model    {retrieval.MODEL}")

    entries = []
    for name, body in bodies.items():
        entry = record(client, name, body)
        entries.append(entry)
        print(f"{name:<14} {len(body['items']):>3} items {entry['timing']['duration_ms']:>8.0f} ms")

    revisions = sorted({entry["model_revision"] for entry in entries if entry["model_revision"]})
    manifest = {
        "task": "search",
        "page": "https://superlinked.com/search",
        "endpoint": base_url,
        "path": retrieval.ENCODE_PATH,
        "model": retrieval.MODEL,
        "model_revision": revisions[0] if len(revisions) == 1 else revisions,
        "run_date": datetime.now(UTC).date().isoformat(),
        "recorded_by": "examples/search/run.py",
        "query_encoding": "options.is_query=true on the questions only",
        "metric": "cosine, computed client-side from the returned dense vectors",
        "passages": len(passages),
        "queries": len(queries),
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
