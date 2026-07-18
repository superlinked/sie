"""Run the bounded ABI-course retrieval ablation.

The default mode is an offline evaluator/output-contract smoke. Live mode
requires explicit catalog-frozen model IDs and reads credentials from the
environment only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "retrieval-course.json"
TOP_K = 3
DEFAULT_CANDIDATES = 4


def load_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_rank(query: str, documents: list[dict[str, Any]]) -> list[str]:
    query_tokens = _tokens(query)
    ranked = sorted(
        documents,
        key=lambda document: (
            len(query_tokens & _tokens(f"{document['title']} {document['text']}")),
            document["id"],
        ),
        reverse=True,
    )
    return [document["id"] for document in ranked]


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    numerator = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = TOP_K) -> float:
    dcg = sum(relevant.get(document_id, 0) / math.log2(index + 2) for index, document_id in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(score / math.log2(index + 2) for index, score in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _recall_at_k(ranked: list[str], relevant: dict[str, int], k: int = TOP_K) -> float:
    expected = {document_id for document_id, score in relevant.items() if score > 0}
    return len(expected & set(ranked[:k])) / len(expected) if expected else 0.0


def evaluate(rankings: dict[str, list[str]], queries: list[dict[str, Any]]) -> dict[str, float]:
    ndcg = [_ndcg_at_k(rankings[query["id"]], query["relevant"]) for query in queries]
    recall = [_recall_at_k(rankings[query["id"]], query["relevant"]) for query in queries]
    return {
        "ndcg_at_3": round(sum(ndcg) / len(ndcg), 4),
        "recall_at_3": round(sum(recall) / len(recall), 4),
    }


def _condition(
    name: str,
    metrics: dict[str, float],
    latency_ms: float,
    request_count: int,
    *,
    models: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "models": models or [],
        "metrics": metrics,
        "latency_ms": round(latency_ms, 3),
        "request_count": request_count,
        "request_ids": [],
        "usage": None,
        "settled_credits": None,
    }


def offline_contract(fixture: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    documents = fixture["documents"]
    queries = fixture["queries"]
    rankings = {query["id"]: lexical_rank(query["text"], documents) for query in queries}
    condition = _condition(
        "lexical-contract",
        evaluate(rankings, queries),
        (time.perf_counter() - started) * 1000,
        0,
    )
    return {
        "schema_version": "abi-course-ablation/v1",
        "mode": "offline-contract",
        "dataset": {"documents": len(documents), "queries": len(queries)},
        "conditions": [condition],
        "measurement_gate": "offline_only",
    }


def _validated_endpoint() -> str:
    endpoint = os.environ.get("SIE_BASE_URL", "").strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit("SIE_BASE_URL must be a bare http(s) endpoint with no credentials, query, or fragment")
    return endpoint


def _dense_vectors(client: Any, model: str, items: list[dict[str, str]], *, is_query: bool) -> list[list[float]]:
    results = client.encode(model, items, output_types=["dense"], is_query=is_query)
    return [[float(value) for value in result["dense"]] for result in results]


def live_ablation(
    fixture: dict[str, Any],
    *,
    embed_model: str,
    rerank_model: str,
    candidate_count: int,
) -> dict[str, Any]:
    from sie_sdk import SIEClient

    endpoint = _validated_endpoint()
    api_key = os.environ.get("SIE_API_KEY") or None
    documents = fixture["documents"]
    queries = fixture["queries"]
    document_by_id = {document["id"]: document for document in documents}
    request_count = 0

    with SIEClient(endpoint, api_key=api_key) as client:
        started = time.perf_counter()
        document_vectors = _dense_vectors(
            client,
            embed_model,
            [{"id": document["id"], "text": document["text"]} for document in documents],
            is_query=False,
        )
        request_count += 1
        query_vectors = _dense_vectors(
            client,
            embed_model,
            [{"id": query["id"], "text": query["text"]} for query in queries],
            is_query=True,
        )
        request_count += 1
        dense_rankings: dict[str, list[str]] = {}
        for query, query_vector in zip(queries, query_vectors, strict=True):
            scored = sorted(
                (
                    (_cosine(query_vector, document_vector), document["id"])
                    for document, document_vector in zip(documents, document_vectors, strict=True)
                ),
                reverse=True,
            )
            dense_rankings[query["id"]] = [document_id for _, document_id in scored]
        dense_elapsed_ms = (time.perf_counter() - started) * 1000
        dense_condition = _condition(
            "dense",
            evaluate(dense_rankings, queries),
            dense_elapsed_ms,
            request_count,
            models=[embed_model],
        )

        started = time.perf_counter()
        reranked: dict[str, list[str]] = {}
        rerank_requests = 0
        for query in queries:
            candidates = dense_rankings[query["id"]][:candidate_count]
            result = client.score(
                rerank_model,
                {"id": query["id"], "text": query["text"]},
                [{"id": document_id, "text": document_by_id[document_id]["text"]} for document_id in candidates],
            )
            rerank_requests += 1
            reranked_ids = [score["item_id"] for score in result["scores"] if score.get("item_id")]
            reranked[query["id"]] = reranked_ids + [
                document_id for document_id in dense_rankings[query["id"]] if document_id not in reranked_ids
            ]
        rerank_elapsed_ms = (time.perf_counter() - started) * 1000
        rerank_condition = _condition(
            "dense-plus-rerank",
            evaluate(reranked, queries),
            dense_elapsed_ms + rerank_elapsed_ms,
            request_count + rerank_requests,
            models=[embed_model, rerank_model],
        )

    return {
        "schema_version": "abi-course-ablation/v1",
        "mode": "live",
        "dataset": {"documents": len(documents), "queries": len(queries)},
        "conditions": [dense_condition, rerank_condition],
        "measurement_gate": "live_measurement_required",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call SIE using explicit catalog-frozen model IDs")
    parser.add_argument("--embed-model", help="exact catalog-frozen embedding model ID")
    parser.add_argument("--rerank-model", help="exact catalog-frozen reranker model ID")
    parser.add_argument("--candidates", type=int, choices=range(2, 5), default=DEFAULT_CANDIDATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = load_fixture()
    if args.live:
        if not args.embed_model or not args.rerank_model:
            raise SystemExit("--live requires --embed-model and --rerank-model from the frozen course catalog")
        output = live_ablation(
            fixture,
            embed_model=args.embed_model,
            rerank_model=args.rerank_model,
            candidate_count=args.candidates,
        )
    else:
        output = offline_contract(fixture)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
