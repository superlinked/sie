#!/usr/bin/env python3
"""Encode the catalogue and every request form with SIE Cloud.

    uv run python run.py              # every call, writes run-output/
    python3 run.py --show images-01   # print a request body, no network
    python3 run.py --show queries

    POST https://api.superlinked.com/v1/encode/google/siglip-so400m-patch14-384

SigLIP puts text and pixels in one 1152-dimensional space, so ranking is a
cosine between what the image calls and the query call return. There is no
reranker and no caption step in between.

Results go to --output as a manifest.json and a calls.json in the dataset's own
shape, one entry per call holding the request, the response, the HTTP status,
the served deployment revision and the round-trip time. The key comes from
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

import catalogue

HERE = Path(__file__).resolve().parent


def dense_items(results: list[Any], body: dict[str, Any]) -> list[dict[str, Any]]:
    """The returned vectors, in the shape the recorded responses carry.

    The SDK hands back each dense embedding as a numpy array. `tolist()` gives
    the same float32 values the wire carried, one row per item, in request order.
    """
    if len(results) != len(body["items"]):
        raise SystemExit(f"Sent {len(body['items'])} items and got {len(results)} back")
    items = []
    for sent, result in zip(body["items"], results, strict=True):
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


def send(client: Any, name: str, body: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Send one encode call and return its calls.json entry.

    The recorded request body names each image by file and digest. What goes on
    the wire is the file's bytes: the SDK transports already encoded JPEG bytes
    unchanged, so the digest in the record is the digest of what SIE read.
    """
    by_id = {record["id"]: record for record in records}
    items: list[dict[str, Any]] = []
    for item in body["items"]:
        if "images" in item:
            path = catalogue.INPUTS / by_id[item["id"]]["file"]
            items.append({"id": item["id"], "images": [path.read_bytes()]})
        else:
            items.append({"id": item["id"], "text": item["text"]})

    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    results = client.encode(catalogue.MODEL, items, output_types=list(body["params"]["output_types"]))
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    if not isinstance(results, list):
        results = [results]
    response = {"items": dense_items(results, body), "model": catalogue.MODEL}
    return {
        "slug": name,
        "kind": "queries" if name == "queries" else "images",
        "requested_at": requested_at,
        "request": {
            "method": "POST",
            # The URL the SDK actually used, not this module's default, so a run
            # against a regional endpoint records where it really went.
            "url": client.base_url.rstrip("/") + catalogue.ENCODE_PATH,
            "body": body,
        },
        "status": 200,
        "response": response,
        "response_sha256": catalogue.sha256_bytes(catalogue.compact_json(response)),
        "deployment_revision": client.last_model_revision,
        "timing": {"duration_ms": elapsed_ms},
    }


def served_revision(client: Any) -> str:
    """The model revision the endpoint reports for itself, via GET /v1/models.

    Raises rather than returning a placeholder. A run that cannot establish
    which weights answered has nothing to record.
    """
    # The gateway occasionally answers this with a body that is not JSON.
    # Three tries, then fail: a run that cannot establish which weights
    # answered has nothing to record, and a transient must not look like one.
    listed = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            listed = client.list_models()
            break
        except Exception as error:  # noqa: BLE001
            last_error = error
            print(f"/v1/models attempt {attempt + 1} failed: {error}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    if listed is None:
        raise SystemExit(f"Could not read the model revision from /v1/models: {last_error}")
    for model in listed if isinstance(listed, list) else listed.get("models", []):
        name = model.get("name") if isinstance(model, dict) else None
        if name == catalogue.MODEL:
            revision = model.get("revision") or ""
            if not revision:
                raise SystemExit(f"/v1/models lists {catalogue.MODEL} with no revision")
            return revision
    raise SystemExit(f"/v1/models does not list {catalogue.MODEL}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode the pinned catalogue and requests")
    parser.add_argument("--show", metavar="NAME", help="print one request body and exit, without calling anything")
    parser.add_argument(
        "--allow-revision-mismatch",
        action="store_true",
        help="record even if the endpoint serves a revision other than the published one",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="directory for manifest.json and calls.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = catalogue.load_catalogue()
    queries = catalogue.load_queries()
    bodies = catalogue.bodies(records, queries)

    if args.show:
        body = bodies.get(args.show)
        if body is None:
            raise SystemExit(f"Unknown call: {args.show}. Known: {', '.join(bodies)}")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    from sie_sdk import SIEClient

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY. To check the published figures without a key, run score.py instead.")

    client = SIEClient(os.environ.get("SIE_BASE_URL", catalogue.ENDPOINT), api_key=api_key, timeout_s=900)
    base_url = client.base_url.rstrip("/")
    print(f"endpoint {base_url}{catalogue.ENCODE_PATH}")
    print(f"model    {catalogue.MODEL}")
    print(f"catalogue {len(records)} photographs, {len(queries)} requests x {len(catalogue.FORMS)} forms")

    # Checked before anything is encoded, so a mismatch costs no credits. An
    # unreadable revision is a failure too: recording an empty one would produce
    # evidence score.py can never validate.
    model_revision = served_revision(client)
    print(f"revision {model_revision}")
    if model_revision != catalogue.MODEL_REVISION and not args.allow_revision_mismatch:
        raise SystemExit(
            f"This endpoint serves {model_revision!r} and this example publishes {catalogue.MODEL_REVISION!r}.\n"
            "Different weights produce different scores, and score.py rejects a recording made against "
            "another revision.\nRe-run with --allow-revision-mismatch to record anyway, for your own "
            "comparison rather than to reproduce the published figures."
        )

    entries = []
    for name, body in bodies.items():
        entry = send(client, name, body, records)
        entries.append(entry)
        print(f"{name:<11} {len(body['items']):>3} items {entry['timing']['duration_ms']:>9.0f} ms")

    # Re-read after the last call. The preflight check above proves the weights
    # were right when the run started; this proves they did not roll over while
    # it was in flight, which would leave the manifest attributing the vectors
    # to a checkpoint that did not produce all of them.
    final_revision = served_revision(client)
    if final_revision != model_revision:
        raise SystemExit(
            f"The endpoint served {model_revision!r} before these calls and {final_revision!r} after them. "
            "The recording spans two checkpoints, so it is not written out. Re-run it."
        )

    revisions = sorted({entry["deployment_revision"] for entry in entries if entry["deployment_revision"]})
    manifest = {
        "task": "image-search",
        "page": "https://superlinked.com/image-search",
        "endpoint": base_url,
        "path": catalogue.ENCODE_PATH,
        "model": catalogue.MODEL,
        # The HF revision of the checkpoint, read from GET /v1/models rather than
        # assumed. X-SIE-Model-Revision is a deployment digest that models served
        # together share, so it is recorded separately under deployment_revision
        # and never as the model's revision.
        "model_revision": model_revision,
        "deployment_revision": revisions[0] if len(revisions) == 1 else revisions,
        "run_date": datetime.now(UTC).date().isoformat(),
        "recorded_by": "examples/image-search/run.py",
        "metric": catalogue.METRIC,
        "dims": catalogue.DIMS,
        "images": len(records),
        "requests": len(queries),
        "forms": list(catalogue.FORMS),
        "query_items": len(catalogue.query_items(queries)),
        "calls_recorded": len(entries),
        "image_transport": (
            "raw file bytes; calls.json records each image's file name and SHA-256 "
            "in place of a second copy of the bytes in inputs/"
        ),
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
