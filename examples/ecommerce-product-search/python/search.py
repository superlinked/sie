"""Search module: encode query, find candidates, rerank with cross-encoder."""

import json
import os
from pathlib import Path

import numpy as np
import yaml

from sie_sdk import SIEClient
from sie_sdk.types import Item


def load_config():
    root = Path(__file__).resolve().parent.parent
    return yaml.safe_load((root / "config.yaml").read_text())


def load_index():
    """Load precomputed embeddings and metadata from disk."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    embeddings = np.load(data_dir / "embeddings.npy")
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    # Normalize embeddings for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    return embeddings, metadata


def search(client, config, embeddings, metadata, query, filters=None):
    """Full search pipeline: encode → retrieve → filter → rerank."""
    gpu = config["cluster"]["gpu"]
    timeout = config["cluster"]["provision_timeout_s"]
    top_k_candidates = config["search"]["top_k_candidates"]
    top_k_results = config["search"]["top_k_results"]

    # Step 1: Encode query
    # is_query=True so the adapter applies the query-side prefix for asymmetric
    # retrieval models like stella_en_400M_v5. Without it, the query is encoded
    # with the document prefix and retrieval quality drops.
    query_result = client.encode(
        config["models"]["embedding"],
        Item(text=query),
        is_query=True,
        gpu=gpu,
        wait_for_capacity=True,
        provision_timeout_s=timeout,
    )
    query_vec = query_result["dense"]
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    # Step 2: Cosine similarity (dot product on normalized vectors)
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k_candidates]

    candidates = []
    for idx in top_indices:
        candidates.append({**metadata[idx], "_vector_score": float(scores[idx])})

    # Step 3: Filter by extracted attributes (optional)
    if filters:
        filtered = []
        for c in candidates:
            attrs = c.get("attributes", {})
            match = True
            for key, value in filters.items():
                if key == "category":
                    if value.lower() not in c.get("category", "").lower():
                        match = False
                        break
                else:
                    attr_val = attrs.get(key, "")
                    if value.lower() not in attr_val.lower():
                        match = False
                        break
            if match:
                filtered.append(c)
        if filtered:
            candidates = filtered
        else:
            # No candidate matched the filter. Fall back to the unfiltered pool so the
            # user still gets results, but warn so the behavior is observable.
            print(f"  [filter] no matches for {filters}, falling back to unfiltered candidates")

    # Step 4: Rerank with cross-encoder
    rerank_items = [
        Item(
            id=c["id"],
            text=(
                f"{c['title']}. "
                f"Category: {c['category']}. "
                + (", ".join(f"{k}: {v}" for k, v in c.get("attributes", {}).items()) + ". " if c.get("attributes") else "")
                + c["description"]
            ),
        )
        for c in candidates
    ]
    rerank_result = client.score(
        config["models"]["reranker"],
        Item(text=query),
        rerank_items,
        gpu=gpu,
        wait_for_capacity=True,
        provision_timeout_s=timeout,
    )

    # Map scores back to candidates
    score_map = {s["item_id"]: s for s in rerank_result["scores"]}
    results = []
    for c in candidates:
        s = score_map.get(c["id"])
        if s:
            results.append(
                {
                    **c,
                    "_rerank_score": s["score"],
                    "_rerank_rank": s["rank"],
                }
            )

    results.sort(key=lambda x: x["_rerank_score"], reverse=True)
    return results[:top_k_results]


def print_results(results, query):
    print(f'\n  Query: "{query}"')
    print(f"  Results: {len(results)}\n")
    for i, r in enumerate(results):
        attrs = r.get("attributes", {})
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else "none"
        print(f"  {i + 1}. {r['title'][:70]}")
        print(f"     category={r['category']}  price=${r.get('price') or '?'}  rating={r.get('rating') or '?'}")
        print(f"     attrs: {attr_str}")
        print(f"     vector={r['_vector_score']:.3f}  rerank={r['_rerank_score']:.4f}")
        print()


def main():
    config = load_config()
    embeddings, metadata = load_index()
    print(f"Loaded index: {len(metadata)} products, {embeddings.shape[1]}d embeddings")

    cluster_url = os.environ.get("SIE_CLUSTER_URL", config["cluster"]["url"])
    api_key = os.environ.get("SIE_API_KEY", config["cluster"]["api_key"])

    queries = [
        ("lightweight waterproof hiking boots", None),
        ("gold jewelry for women", None),
        ("wireless bluetooth headphones", {"category": "All Electronics"}),
        ("ceramic coffee mug", {"category": "Amazon Home"}),
    ]

    with SIEClient(cluster_url, api_key=api_key) as client:
        for query, filters in queries:
            results = search(client, config, embeddings, metadata, query, filters)
            print_results(results, query)


if __name__ == "__main__":
    main()
