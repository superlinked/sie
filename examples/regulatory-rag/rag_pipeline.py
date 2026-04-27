#!/usr/bin/env python3
"""
Regulatory RAG Pipeline — SIE Example
======================================

A complete retrieval-augmented generation pipeline for US regulatory
intelligence, demonstrating SIE's encode, score, and extract primitives
with domain-specific custom models.

Custom Models:
  - Encoder LoRA: sugiv/modernbert-us-stablecoin-encoder
    (fine-tuned on answerdotai/ModernBERT-base via LoRA for US regulatory text)
  - Pruner:       sugiv/stablebridge-pruner-highlighter
    (PruningHead MLP on BAAI/bge-reranker-v2-m3 for reranking + highlighting)

Pipeline stages:
  1. Encode query + corpus with sugiv/modernbert-us-stablecoin-encoder LoRA
  2. Dense retrieval via cosine similarity
  3. Rerank candidates with sugiv/stablebridge-pruner-highlighter cross-encoder
  4. Extract semantic highlights from top results (pruning + highlighting)
  5. Build compressed context for LLM generation

Usage:
  python rag_pipeline.py                          # Run with default SIE server
  python rag_pipeline.py --url http://host:8080   # Custom server URL
  python rag_pipeline.py --query "your question"  # Custom query
"""

import argparse
import json
import math
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

# ── Configuration ──────────────────────────────────────────────────────

DEFAULT_SIE_URL = "http://localhost:8080"

# SIE serves the encoder as base + LoRA profile.
# Base model registered in SIE:
ENCODER_MODEL = "answerdotai/ModernBERT-base"
# LoRA profile that activates sugiv/modernbert-us-stablecoin-encoder weights:
ENCODER_PROFILE = "us-regulatory"
# Together: ENCODER_MODEL + ENCODER_PROFILE = sugiv/modernbert-us-stablecoin-encoder

PRUNER_MODEL = "sugiv/stablebridge-pruner-highlighter"

TOP_K_RETRIEVE = 5   # Candidates from dense retrieval
TOP_K_RERANK = 3     # Final results after reranking

SAMPLE_QUERIES = [
    "What reserve requirements apply to stablecoin issuers under the GENIUS Act?",
    "What licensing is required to issue payment stablecoins in the United States?",
    "How must stablecoin reserves be segregated from issuer operational funds?",
    "What penalties exist for non-compliant stablecoin issuers?",
    "Does the SEC consider payment stablecoins to be securities?",
]


# ── SIE Client ─────────────────────────────────────────────────────────

class SIEClient:
    """Minimal SIE client using urllib (no external dependencies)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, body: dict, timeout: int = 120) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()[:500]
            print(f"  ERROR {e.code}: {error_body}", file=sys.stderr)
            return {"error": e.code, "detail": error_body}

    def encode(
        self, model: str, texts: list[str], profile: str | None = None
    ) -> list[list[float]]:
        """Encode texts into dense embeddings."""
        body: dict[str, Any] = {"items": [{"text": t} for t in texts]}
        if profile:
            body["params"] = {"options": {"profile": profile}}
        r = self._post(f"/v1/encode/{model}", body)
        return [
            item.get("dense", {}).get("values", [])
            for item in r.get("items", [])
        ]

    def score(
        self, model: str, query: str, texts: list[str]
    ) -> list[dict]:
        """Score query-document pairs. Returns list of {score, rank}."""
        body = {
            "query": {"text": query},
            "items": [
                {"id": f"candidate-{idx}", "text": text}
                for idx, text in enumerate(texts)
            ],
        }
        r = self._post(f"/v1/score/{model}", body)
        return r.get("scores", [])

    def extract(
        self, model: str, texts: list[str], instruction: str | None = None
    ) -> list[list[dict]]:
        """Extract entities/highlights from texts. Returns entities per item."""
        body: dict[str, Any] = {"items": [{"text": t} for t in texts]}
        if instruction:
            body["params"] = {"instruction": instruction}
        r = self._post(f"/v1/extract/{model}", body)
        return [
            item.get("entities", [])
            for item in r.get("items", [])
        ]

    def health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/readyz")
            resp = urllib.request.urlopen(req, timeout=5)
            return resp.read().decode().strip() == "ok"
        except Exception:
            return False


# ── Vector Operations ──────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Display Helpers ────────────────────────────────────────────────────

def format_score(score: float) -> str:
    if score >= 0.9:
        return f"\033[92m{score:.4f}\033[0m"  # Green
    elif score >= 0.5:
        return f"\033[93m{score:.4f}\033[0m"  # Yellow
    else:
        return f"\033[91m{score:.4f}\033[0m"  # Red


def print_header(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_step(step: int, name: str, detail: str = ""):
    print(f"\n\033[1m[Step {step}]\033[0m {name}")
    if detail:
        print(f"  {detail}")


def print_entity(entity: dict, indent: int = 4):
    label = entity.get("label", "unknown")
    score = entity.get("score", 0)
    text = entity.get("text", "")
    prefix = " " * indent

    if label == "summary":
        print(f"{prefix}\033[90m[{label}] {text}\033[0m")
    elif label == "highlight":
        print(f"{prefix}\033[92m[{label}] score={score:.4f}\033[0m")
        print(f"{prefix}  \033[92m\"{text}\"\033[0m")
    elif label == "kept":
        print(f"{prefix}\033[93m[{label}] score={score:.4f}\033[0m")
        print(f"{prefix}  \"{text}\"")
    elif label == "pruned":
        print(f"{prefix}\033[91m[{label}] score={score:.4f}\033[0m")
        print(f"{prefix}  \033[90m\"{text[:80]}{'...' if len(text) > 80 else ''}\"\033[0m")


# ── RAG Pipeline ───────────────────────────────────────────────────────

def run_pipeline(
    client: SIEClient,
    query: str,
    corpus: list[dict],
    verbose: bool = True,
):
    """
    Execute the full RAG pipeline:
    1. Encode query + corpus with domain LoRA
    2. Dense retrieval via cosine similarity
    3. Rerank with cross-encoder
    4. Extract highlights from top results
    5. Build compressed context
    """
    timings = {}

    if verbose:
        print_header("REGULATORY RAG PIPELINE")
        print(f"\n  Query: \"{query}\"")
        print(f"  Corpus: {len(corpus)} documents")
        print(f"  Encoder: sugiv/modernbert-us-stablecoin-encoder")
        print(f"           (SIE: {ENCODER_MODEL} + LoRA profile '{ENCODER_PROFILE}')")
        print(f"  Pruner:  {PRUNER_MODEL}")

    # ── Stage 1: Encode ──────────────────────────────────────────────

    if verbose:
        print_step(1, "Encode", f"Embedding query + {len(corpus)} docs with sugiv/modernbert-us-stablecoin-encoder LoRA")

    t0 = time.perf_counter()

    # Encode query
    query_emb = client.encode(ENCODER_MODEL, [query], profile=ENCODER_PROFILE)[0]

    # Encode corpus (batch)
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_embs = client.encode(ENCODER_MODEL, corpus_texts, profile=ENCODER_PROFILE)

    timings["encode_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if verbose:
        print(f"  Embedding dim: {len(query_emb)}")
        print(f"  Time: {timings['encode_ms']:.0f}ms")

    # ── Stage 2: Dense Retrieval ─────────────────────────────────────

    if verbose:
        print_step(2, "Retrieve", f"Top-{TOP_K_RETRIEVE} by cosine similarity")

    t0 = time.perf_counter()

    similarities = [
        (i, cosine_similarity(query_emb, emb))
        for i, emb in enumerate(corpus_embs)
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:TOP_K_RETRIEVE]

    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if verbose:
        for rank, (idx, sim) in enumerate(top_k, 1):
            doc = corpus[idx]
            print(f"  #{rank} sim={sim:.4f} [{doc['id']}] {doc['title']}")

    # ── Stage 3: Rerank ──────────────────────────────────────────────

    if verbose:
        print_step(3, "Rerank", f"Cross-encoder scoring {len(top_k)} candidates")

    t0 = time.perf_counter()

    candidate_texts = [corpus[idx]["text"] for idx, _ in top_k]
    scores = client.score(PRUNER_MODEL, query, candidate_texts)

    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    score_by_candidate: dict[int, dict] = {}
    for score in scores:
        item_id = str(score.get("item_id", ""))
        if not item_id.startswith("candidate-"):
            continue
        try:
            candidate_idx = int(item_id.rsplit("-", 1)[1])
        except ValueError:
            continue
        score_by_candidate[candidate_idx] = score

    # Merge scores with retrieval results
    reranked = []
    for i, (orig_idx, sim) in enumerate(top_k):
        score_val = score_by_candidate.get(i, {}).get("score", 0)
        reranked.append({
            "corpus_idx": orig_idx,
            "doc": corpus[orig_idx],
            "similarity": sim,
            "rerank_score": score_val,
        })

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    if verbose:
        print(f"  Time: {timings['rerank_ms']:.0f}ms")
        for rank, r in enumerate(reranked, 1):
            marker = " <<<" if rank <= TOP_K_RERANK else ""
            print(
                f"  #{rank} rerank={format_score(r['rerank_score'])} "
                f"sim={r['similarity']:.4f} [{r['doc']['id']}]{marker}"
            )

    # ── Stage 4: Extract Highlights ──────────────────────────────────

    final = reranked[:TOP_K_RERANK]

    if verbose:
        print_step(4, "Extract", f"Highlighting top-{TOP_K_RERANK} results")

    t0 = time.perf_counter()

    final_texts = [r["doc"]["text"] for r in final]
    all_entities = client.extract(PRUNER_MODEL, final_texts, instruction=query)

    timings["extract_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    for i, entities in enumerate(all_entities):
        final[i]["entities"] = entities

    if verbose:
        print(f"  Time: {timings['extract_ms']:.0f}ms")
        for i, r in enumerate(final):
            print(f"\n  Document: [{r['doc']['id']}] {r['doc']['title']}")
            for entity in r["entities"]:
                print_entity(entity, indent=4)

    # ── Stage 5: Build Compressed Context ────────────────────────────

    if verbose:
        print_step(5, "Compress", "Building LLM context from highlights")

    context_parts = []
    for r in final:
        kept_texts = []
        for entity in r.get("entities", []):
            if entity.get("label") in ("highlight", "kept"):
                kept_texts.append(entity["text"])

        if kept_texts:
            source = f"[{r['doc']['id']}] {r['doc']['title']}"
            context_parts.append(f"Source: {source}\n" + " ".join(kept_texts))
        else:
            # If no highlights, include the full text (fallback)
            source = f"[{r['doc']['id']}] {r['doc']['title']}"
            context_parts.append(f"Source: {source}\n{r['doc']['text']}")

    compressed_context = "\n\n---\n\n".join(context_parts)

    original_chars = sum(len(r["doc"]["text"]) for r in final)
    compressed_chars = len(compressed_context)
    compression_pct = round(100 * (1 - compressed_chars / max(original_chars, 1)), 1)

    timings["total_ms"] = round(
        timings["encode_ms"] + timings["retrieve_ms"]
        + timings["rerank_ms"] + timings["extract_ms"],
        1,
    )

    if verbose:
        print(f"  Original:   {original_chars} chars")
        print(f"  Compressed: {compressed_chars} chars")
        print(f"  Reduction:  {compression_pct}%")

        print_header("COMPRESSED CONTEXT FOR LLM")
        print(compressed_context)

        print_header("PIPELINE SUMMARY")
        print(f"  Query: \"{query[:60]}{'...' if len(query) > 60 else ''}\"")
        print(f"  Corpus size:    {len(corpus)} documents")
        print(f"  Retrieved:      {TOP_K_RETRIEVE} candidates")
        print(f"  Reranked to:    {TOP_K_RERANK} results")
        print(f"  Compression:    {compression_pct}%")
        print()
        print(f"  Latency breakdown:")
        print(f"    Encode:     {timings['encode_ms']:>8.1f}ms")
        print(f"    Retrieve:   {timings['retrieve_ms']:>8.1f}ms  (in-memory cosine)")
        print(f"    Rerank:     {timings['rerank_ms']:>8.1f}ms  (SIE cross-encoder)")
        print(f"    Extract:    {timings['extract_ms']:>8.1f}ms  (SIE pruner)")
        print(f"    ─────────────────────────")
        print(f"    Total:      {timings['total_ms']:>8.1f}ms")
        print()

    return {
        "query": query,
        "results": [
            {
                "id": r["doc"]["id"],
                "title": r["doc"]["title"],
                "rerank_score": r["rerank_score"],
                "similarity": r["similarity"],
                "entities": r.get("entities", []),
            }
            for r in final
        ],
        "compressed_context": compressed_context,
        "compression_pct": compression_pct,
        "timings": timings,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Regulatory RAG Pipeline powered by SIE"
    )
    parser.add_argument(
        "--url", default=DEFAULT_SIE_URL,
        help=f"SIE server URL (default: {DEFAULT_SIE_URL})"
    )
    parser.add_argument(
        "--query", default=None,
        help="Query to run (default: runs all sample queries)"
    )
    parser.add_argument(
        "--corpus", default=None,
        help="Path to corpus JSON (default: sample_corpus.json)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output"
    )
    args = parser.parse_args()

    # Initialize client
    client = SIEClient(args.url)

    # Health check
    if not client.health():
        print(f"ERROR: SIE server not reachable at {args.url}")
        print("Start the server with: sie-server serve")
        sys.exit(1)

    # Load corpus
    if args.corpus:
        corpus_path = Path(args.corpus)
    else:
        corpus_path = Path(__file__).parent / "sample_corpus.json"

    if not corpus_path.exists():
        print(f"ERROR: Corpus file not found: {corpus_path}")
        sys.exit(1)

    with open(corpus_path) as f:
        corpus = json.load(f)

    print(f"SIE server: {args.url}")
    print(f"Corpus: {len(corpus)} documents from {corpus_path.name}")

    # Run queries
    queries = [args.query] if args.query else SAMPLE_QUERIES
    all_results = []

    for query in queries:
        result = run_pipeline(
            client, query, corpus, verbose=not args.quiet
        )
        all_results.append(result)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Summary
    if len(all_results) > 1:
        print_header("ALL QUERIES SUMMARY")
        avg_total = sum(r["timings"]["total_ms"] for r in all_results) / len(all_results)
        avg_compress = sum(r["compression_pct"] for r in all_results) / len(all_results)
        print(f"  Queries run: {len(all_results)}")
        print(f"  Avg latency: {avg_total:.1f}ms")
        print(f"  Avg compression: {avg_compress:.1f}%")


if __name__ == "__main__":
    main()
