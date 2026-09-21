#!/usr/bin/env python3
"""Pull an entity and relation graph out of filing text with SIE Cloud.

    uv run python run.py                              # every pinned paragraph
    uv run python run.py --candidate flex-credit-facility
    python3 run.py --show flex-credit-facility        # print the requests, no network

Two calls per paragraph, the pair the /knowledge-graph task page shows:

    POST https://api.superlinked.com/v1/extract/fastino%2Fgliner2-large-v1
    1. the paragraph, with the entity types to look for
    2. the same paragraph, call one's entities as item metadata, and the
       relation types to look for

No threshold or other option is sent, so the server default applies and the
output is what the page's snippet produces. Results go to --output as a
manifest.json and a calls.json in the dataset's own shape, one entry per call
holding the request, the response, the status, the served model revision and
the round-trip time. The key comes from SIE_API_KEY and is never written out.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
counts them offline.
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

import graph

HERE = Path(__file__).resolve().parent


def envelope(result: dict[str, Any], model: str) -> dict[str, Any]:
    """The server's own extraction result for one item.

    The SDK hands back the item's result with the model and request-scoped
    metadata attached. Keep the item's own fields and the model; drop the rest.
    """
    missing = [key for key in graph.ITEM_KEYS if key not in result]
    if missing:
        raise SystemExit(f"Response is missing {', '.join(missing)}; refusing to record a partial result")
    if result.get("model") != model:
        raise SystemExit(f"Response came from {result.get('model')!r}, not {model!r}")
    return {"model": model, "item": {key: result[key] for key in graph.ITEM_KEYS}}


def call(
    client: SIEClient,
    model: str,
    candidate: dict[str, Any],
    kind: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Send one extract call and return its calls.json entry."""
    item = dict(body["items"][0])
    started = time.monotonic()
    result = client.extract(model, item, labels=list(body["params"]["labels"]))
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    if isinstance(result, list):
        raise SystemExit("Expected one result for one item")
    response = envelope(result, model)
    return {
        "slug": f"{candidate['id']}__{kind}",
        "candidate": candidate["id"],
        "kind": kind,
        "request": {
            "method": "POST",
            # The URL the SDK actually used, not this module's default, so a run
            # against a regional endpoint records where it really went.
            "url": client.base_url.rstrip("/") + graph.extract_path(model),
            "model": model,
            # The wire body these SDK arguments produce.
            "body": body,
        },
        "status": 200,
        "response": response,
        "response_sha256": graph.sha256_bytes(graph.compact_json(response)),
        "model_revision": client.last_model_revision,
        "timing": {"duration_ms": elapsed_ms},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an entity and relation graph for pinned filing paragraphs")
    parser.add_argument("--candidate", action="append", default=[], help="id to run; repeatable, default all")
    parser.add_argument("--show", metavar="ID", help="print both request bodies for one paragraph and exit")
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="directory for manifest.json and calls.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc = graph.load_candidates()
    model = doc["model"]
    candidates = {candidate["id"]: candidate for candidate in doc["candidates"]}

    if args.show:
        candidate = candidates.get(args.show)
        if candidate is None:
            raise SystemExit(f"Unknown candidate: {args.show}. Known: {', '.join(candidates)}")
        recorded = graph.read_json(graph.CALLS_PATH)["calls"]
        entities = next(
            entry["response"]["item"]["entities"]
            for entry in recorded
            if entry["candidate"] == candidate["id"] and entry["kind"] == "entities"
        )
        print(json.dumps(graph.entities_body(candidate), indent=2, ensure_ascii=False))
        print("# call two, using the entities the recorded call one returned:")
        print(json.dumps(graph.relations_body(candidate, entities), indent=2, ensure_ascii=False))
        return 0

    selected = args.candidate or list(candidates)
    unknown = [cid for cid in selected if cid not in candidates]
    if unknown:
        raise SystemExit(f"Unknown candidate(s): {', '.join(unknown)}")
    duplicated = sorted({cid for cid in selected if selected.count(cid) > 1})
    if duplicated:
        raise SystemExit(f"Repeated --candidate: {', '.join(duplicated)}")

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY. To check the published figures without a key, run score.py instead.")

    client = SIEClient(os.environ.get("SIE_BASE_URL", graph.ENDPOINT), api_key=api_key, timeout_s=900)
    base_url = client.base_url.rstrip("/")
    print(f"endpoint {base_url}{graph.extract_path(model)}")
    print(f"model    {model}")

    entries = []
    for cid in selected:
        candidate = candidates[cid]
        first = call(client, model, candidate, "entities", graph.entities_body(candidate))
        entities = first["response"]["item"]["entities"]
        second = call(client, model, candidate, "relations", graph.relations_body(candidate, entities))
        entries.extend([first, second])
        edges = len(second["response"]["item"]["relations"])
        revision = second["model_revision"] or "not reported"
        print(f"{cid:<24} {len(entities):>3} entities {edges:>3} edges  revision {revision}")

    revisions = sorted({entry["model_revision"] for entry in entries if entry["model_revision"]})
    manifest = {
        "task": "knowledge-graph",
        "page": "https://superlinked.com/knowledge-graph",
        "endpoint": base_url,
        "path": graph.extract_path(model),
        "model": model,
        "model_revision": revisions[0] if len(revisions) == 1 else revisions,
        "run_date": datetime.now(UTC).date().isoformat(),
        "recorded_by": "examples/knowledge-graph/run.py",
        "request_options": "none, so the server default threshold applies",
        "calls_recorded": len(entries),
        "response_sha256": (
            "sha256 of json.dumps(response, ensure_ascii=False, separators=(',', ':')).encode('utf-8')"
        ),
        "response_scope": (
            "the server's per-item extraction result. Request-scoped usage and credit metadata are not carried here."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("manifest.json", manifest), ("calls.json", {"calls": entries})):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}/manifest.json and {args.output}/calls.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
