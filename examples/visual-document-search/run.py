#!/usr/bin/env python3
"""Rank document pages by what they look like, with ColPali on your own SIE server.

    uv run python run.py                          # all four comparisons
    uv run python run.py --only aircraft-refueling-signals
    python3 run.py --show aircraft-refueling-signals   # print a request, no network

Start SIE first. The model downloads and loads on the first request, so that
call takes a few minutes and every later one is warm:

    docker run --gpus all -p 8080:8080 \
      -v sie-hf-cache:/app/.cache/huggingface \
      ghcr.io/superlinked/sie-server:latest-cuda12-default

    # macOS (Apple Silicon) or Linux, native, Python 3.12
    pip install "sie-server[local]" && sie-server serve

Point somewhere else with SIE_BASE_URL, and set SIE_API_KEY if your deployment
wants one:

    SIE_BASE_URL=https://sie.internal.example:8080 uv run python run.py

The call this sends, per batch of pages and once per query:

    POST http://localhost:8080/v1/encode/vidore/colpali-v1.3-hf
    {"items": [...], "params": {"output_types": ["multivector"], "output_dtype": "float16"}}

ColPali returns one vector per image patch rather than one per page, so a figure
covering a tenth of a page keeps its own vectors. Ranking is MaxSim: every query
vector takes its strongest match anywhere on the page, and those are summed.

Results go to --output as a manifest.json and a calls.json, one entry per call
holding the request, the response, the HTTP status and the round-trip time.
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

import retrieval

HERE = Path(__file__).resolve().parent
BATCH = 8


def multivector_items(results: list[Any], body: dict[str, Any]) -> list[dict[str, Any]]:
    """The returned multivectors, in the shape the recorded responses carry."""
    if len(results) != len(body["items"]):
        raise SystemExit(f"Sent {len(body['items'])} items and got {len(results)} back")
    items = []
    for sent, result in zip(body["items"], results):
        if result.get("id") != sent["id"]:
            raise SystemExit(f"Response item id {result.get('id')!r} does not match request item {sent['id']!r}")
        vector = result["multivector"]
        rows = vector.tolist() if hasattr(vector, "tolist") else [list(row) for row in vector]
        items.append(
            {
                "id": sent["id"],
                "multivector": {
                    "tokens": len(rows),
                    "dim": len(rows[0]) if rows else 0,
                    "dtype": str(getattr(vector, "dtype", "float16")),
                    "float16_base64": retrieval.encode_multivector(rows),
                },
            }
        )
    return items


def send(client: Any, slug: str, kind: str, body: dict[str, Any], payload: list[dict[str, Any]]) -> dict[str, Any]:
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    results = client.encode(
        retrieval.MODEL,
        payload,
        output_types=list(body["params"]["output_types"]),
        output_dtype=body["params"]["output_dtype"],
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    if not isinstance(results, list):
        results = [results]
    response = {"items": multivector_items(results, body), "model": retrieval.MODEL}
    return {
        "slug": slug,
        "kind": kind,
        "requested_at": requested_at,
        "request": {
            "method": "POST",
            "url": client.base_url.rstrip("/") + retrieval.ENCODE_PATH,
            "body": body,
        },
        "status": 200,
        "response": response,
        "response_sha256": retrieval.sha256_bytes(retrieval.compact_json(response)),
        "timing": {"duration_ms": elapsed_ms},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank ViDoRe pages with ColPali on a self-hosted SIE server")
    parser.add_argument("--show", metavar="ID", help="print one query request body and exit, without calling anything")
    parser.add_argument("--only", metavar="ID", help="run a single comparison")
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="directory for manifest.json and calls.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparisons = retrieval.load_comparisons()
    by_document = retrieval.load_pages()

    if args.show:
        match = next((c for c in comparisons if c["id"] == args.show), None)
        if match is None:
            raise SystemExit(f"Unknown comparison: {args.show}. Known: {', '.join(c['id'] for c in comparisons)}")
        print(json.dumps(retrieval.query_body(match), indent=2, ensure_ascii=False))
        return 0

    if args.only:
        comparisons = [c for c in comparisons if c["id"] == args.only]
        if not comparisons:
            raise SystemExit(f"Unknown comparison: {args.only}")

    from sie_sdk import SIEClient

    base_url = os.environ.get("SIE_BASE_URL", retrieval.ENDPOINT)
    client = SIEClient(base_url, api_key=os.environ.get("SIE_API_KEY", "local"), timeout_s=7200)
    resolved = client.base_url.rstrip("/")
    print(f"endpoint {resolved}{retrieval.ENCODE_PATH}")
    print(f"model    {retrieval.MODEL}")
    print("the first call loads the model, so it takes a few minutes\n")

    entries: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for comparison in comparisons:
        pages = by_document[comparison["doc_id"]]
        print(f"=== {comparison['label']}: {len(pages)} pages ===")

        query_request = retrieval.query_body(comparison)
        query_entry = send(
            client,
            f"{comparison['id']}/query",
            "query",
            query_request,
            [{"id": comparison["id"], "text": comparison["query"]}],
        )
        entries.append(query_entry)
        query_vectors = retrieval.decode_multivector(
            query_entry["response"]["items"][0]["multivector"]["float16_base64"]
        )

        page_vectors: dict[int, list[list[float]]] = {}
        for start in range(0, len(pages), BATCH):
            chunk = pages[start : start + BATCH]
            body = retrieval.page_body(chunk)
            payload = [{"id": str(page["corpus_id"]), "images": [retrieval.check_page_bytes(page)]} for page in chunk]
            entry = send(client, f"{comparison['id']}/pages-{start // BATCH:03d}", "pages", body, payload)
            entries.append(entry)
            for item in entry["response"]["items"]:
                page_vectors[int(item["id"])] = retrieval.decode_multivector(item["multivector"]["float16_base64"])
            print(f"  {min(start + BATCH, len(pages))}/{len(pages)} pages")

        visual = retrieval.visual_rank(query_vectors, page_vectors)
        text = retrieval.bm25_rank(comparison["query"], pages)
        relevant = comparison["relevant_corpus_id"]
        visual_rank = [cid for cid, _ in visual].index(relevant) + 1
        text_rank = [cid for cid, _ in text].index(relevant) + 1
        summary.append(
            {
                "id": comparison["id"],
                "label": comparison["label"],
                "candidate_pages": len(pages),
                "relevant_corpus_id": relevant,
                "text_rank": text_rank,
                "visual_rank": visual_rank,
                "visual_score": round(dict(visual)[relevant], 6),
                "text_top_1": text[0][0],
                "visual_top_1": visual[0][0],
            }
        )
        print(f"  text rank {text_rank}, visual rank {visual_rank}, of {len(pages)}\n")

    manifest = {
        "task": "visual-document-search",
        "page": "https://superlinked.com/visual-document-search",
        "endpoint": resolved,
        "path": retrieval.ENCODE_PATH,
        "model": retrieval.MODEL,
        "run_date": datetime.now(UTC).date().isoformat(),
        "recorded_by": "examples/visual-document-search/run.py",
        "scoring": "MaxSim over L2-normalized ColPali multivectors, as retrieval.maxsim",
        "baseline": {"name": "BM25 over the ViDoRe markdown field", "k1": retrieval.K1, "b": retrieval.B},
        "comparisons": len(comparisons),
        "pages_encoded": sum(row["candidate_pages"] for row in summary),
        "calls_recorded": len(entries),
        "results": summary,
        "response_sha256": (
            "sha256 of json.dumps(response, ensure_ascii=False, separators=(',', ':')).encode('utf-8')"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "calls.json").write_text(
        json.dumps({"calls": entries}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}/manifest.json and {args.output}/calls.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
