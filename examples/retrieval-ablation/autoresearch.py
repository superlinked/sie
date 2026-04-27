"""Autoresearch: Scan SIE models to maximize NDCG@10.

20 min budget per experiment. Tests encoders, rerankers, and ColBERT models.
Results logged to autoresearch_results.tsv.

Usage:
    uv run python autoresearch.py --gpu l4-spot
    uv run python autoresearch.py --gpu l4-spot --type reranker
    uv run python autoresearch.py --gpu l4-spot --type colbert
    uv run python autoresearch.py --gpu l4-spot --type encoder
"""

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import maxsim_cpu
import numpy as np
import polars as pl
from dotenv import load_dotenv
from loguru import logger
from sie_sdk import SIEAsyncClient
from turbopuffer import AsyncTurbopuffer

load_dotenv(Path(__file__).parent / ".env")
logger.remove()
logger.add(sys.stderr, level="INFO")

CACHE_DIR = Path(__file__).parent / "cache" / "ablation"
TIMEOUT_S = 5400  # 90 min per experiment
PROVISION_TIMEOUT = 1200  # 20 min — models need time to load on cold GPU
TOP_K = 50
ENCODE_BATCH = 64
MV_BATCH = 16
RESULTS_FILE = Path(__file__).parent / "autoresearch_results.tsv"

# Pre-categorized model lists
RERANKERS = [
    "mixedbread-ai/mxbai-rerank-large-v2",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "BAAI/bge-reranker-v2-m3",
    "jinaai/jina-reranker-v2-base-multilingual",
    "jinaai/jina-reranker-v3",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-base",
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

COLBERT_MODELS = [
    "lightonai/Reason-ModernColBERT",
    "answerdotai/answerai-colbert-small-v1",
    "mixedbread-ai/mxbai-edge-colbert-v0-32m",
]

DENSE_ENCODERS = [
    "NovaSearch/stella_en_1.5B_v5",
    "NovaSearch/stella_en_400M_v5",
    "Qwen/Qwen3-Embedding-0.6B",
    "nomic-ai/nomic-embed-text-v2-moe",
    "Alibaba-NLP/gte-multilingual-base",
    "intfloat/multilingual-e5-large-instruct",
    "intfloat/e5-large-v2",
    "nvidia/NV-Embed-v2",
    "google/embeddinggemma-300m",
]


def _require_env(name):
    val = os.environ.get(name)
    if not val:
        logger.error(f"Missing: {name}")
        sys.exit(1)
    return val


def slugify(name):
    import re

    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")


def ndcg_at_k(ranked_ids, qrel_map, k=10):
    dcg = sum(qrel_map.get(cid, 0) / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]))
    ideal = sorted(qrel_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(ranked_results, query_items, qrel_map):
    ndcgs, recalls = [], []
    for ranked, qi in zip(ranked_results, query_items):
        qrels = qrel_map.get(qi["query_id"], {})
        if not qrels:
            continue
        ndcgs.append(ndcg_at_k(ranked, qrels))
        rel = {c for c, s in qrels.items() if s > 0}
        recalls.append(sum(1 for c in ranked[:10] if c in rel) / len(rel) if rel else 0.0)
    return round(float(np.mean(ndcgs)), 4), round(float(np.mean(recalls)), 4)


def normalize_mv(mv):
    mv = np.asarray(mv, dtype=np.float32)
    norms = np.linalg.norm(mv, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return mv / norms


def load_dataset():
    from datasets import load_dataset as hf_load

    corpus = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "corpus", split="test").data.table)
    queries = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "queries", split="test").data.table)
    qrels_df = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "qrels", split="test").data.table)

    qrel_map = {}
    for row in qrels_df.iter_rows(named=True):
        qrel_map.setdefault(row["query_id"], {})[row["corpus_id"]] = row["score"]

    corpus_items = []
    for row in corpus.iter_rows(named=True):
        text = row["markdown"] or ""
        if len(text.strip()) < 10:
            text = f"Page {row['page_number_in_doc']} of {row['doc_id']}"
        corpus_items.append({"corpus_id": row["corpus_id"], "text": text})

    query_items = [{"query_id": r["query_id"], "text": r["query"]} for r in queries.iter_rows(named=True)]
    return corpus_items, query_items, qrel_map


def log_result(model, model_type, ndcg, recall, elapsed, status, note=""):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a") as f:
        if not exists:
            f.write("model\ttype\tndcg10\trecall10\ttime_s\tstatus\tnote\n")
        f.write(f"{model}\t{model_type}\t{ndcg}\t{recall}\t{round(elapsed)}\t{status}\t{note}\n")


async def test_reranker(
    sie, model, query_texts, hybrid_pools, text_map, query_items, qrel_map, gpu, sem, cache_slug=None
):
    """Test a cross-encoder reranker on hybrid pool."""
    slug = cache_slug or slugify(model)
    cache_path = CACHE_DIR / f"autoresearch_ce_{slug}.json"
    if cache_path.exists():
        results = json.load(open(cache_path))
        ndcg, recall = evaluate(results, query_items, qrel_map)
        return ndcg, recall, 0, "cached"

    results = [None] * len(query_texts)
    done = 0

    import random

    async def score_one(idx):
        nonlocal done
        cands = hybrid_pools[idx]
        if not cands:
            return idx, []
        items = [{"text": text_map.get(cid, "")} for cid in cands]
        for attempt in range(5):
            try:
                async with sem:
                    result = await sie.score(
                        model,
                        query={"text": query_texts[idx]},
                        items=items,
                        gpu=gpu,
                        wait_for_capacity=True,
                        provision_timeout_s=PROVISION_TIMEOUT,
                    )
                    scored = sorted(
                        [(cands[int(s["item_id"].split("-")[1])], s["score"]) for s in result["scores"]],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    done += 1
                    if done % 200 == 0:
                        logger.info(f"  CE {model}: {done}/{len(query_texts)}")
                    return idx, [cid for cid, _ in scored]
            except Exception as e:
                if attempt < 4:
                    backoff = (2**attempt) + random.uniform(0, 2)
                    logger.warning(f"  CE retry {attempt + 1}/5 ({type(e).__name__}), backoff {backoff:.1f}s")
                    await asyncio.sleep(backoff)
                else:
                    raise

    # No batching — semaphore controls concurrency. Continuous flow keeps connections warm.
    for coro in asyncio.as_completed([score_one(i) for i in range(len(query_texts))]):
        idx, ranked = await coro
        results[idx] = ranked

    with open(cache_path, "w") as f:
        json.dump(results, f)
    ndcg, recall = evaluate(results, query_items, qrel_map)
    return ndcg, recall, 0, "ok"


async def test_colbert(sie, model, corpus_texts, query_texts, corpus_items, query_items, qrel_map, gpu, sem):
    """Test a ColBERT/MV model — encode + brute-force MaxSim."""
    slug = slugify(model)
    corpus_cache = CACHE_DIR / f"mv_corpus_{slug}.npz"
    query_cache = CACHE_DIR / f"mv_query_{slug}.npz"
    direct_cache = CACHE_DIR / f"autoresearch_mv_{slug}.json"

    if direct_cache.exists():
        results = json.load(open(direct_cache))
        ndcg, recall = evaluate(results, query_items, qrel_map)
        return ndcg, recall, 0, "cached"

    # Encode corpus
    if corpus_cache.exists():
        data = np.load(corpus_cache, allow_pickle=False)
        corpus_mvs = [data[f"mv_{i}"] for i in range(len(data.files))]
        logger.info(f"  Cache hit corpus: {corpus_cache}")
    else:
        corpus_mvs = []
        batches = [corpus_texts[i : i + MV_BATCH] for i in range(0, len(corpus_texts), MV_BATCH)]
        for bi, batch_texts in enumerate(batches):
            batch = [{"text": t} for t in batch_texts]
            for attempt in range(3):
                try:
                    async with sem:
                        result = await sie.encode(
                            model,
                            batch,
                            output_types=["multivector"],
                            is_query=False,
                            gpu=gpu,
                            wait_for_capacity=True,
                            provision_timeout_s=PROVISION_TIMEOUT,
                        )
                        corpus_mvs.extend(r["multivector"] for r in result)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"  Encode retry {attempt + 1}/3: {type(e).__name__}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise
            if (bi + 1) % 20 == 0:
                logger.info(f"  MV corpus {model}: {bi + 1}/{len(batches)}")
        corpus_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(corpus_cache, **{f"mv_{i}": mv for i, mv in enumerate(corpus_mvs)})

    # Encode queries
    if query_cache.exists():
        data = np.load(query_cache, allow_pickle=False)
        query_mvs = [data[f"mv_{i}"] for i in range(len(data.files))]
        logger.info(f"  Cache hit query: {query_cache}")
    else:
        query_mvs = []
        batches = [query_texts[i : i + MV_BATCH] for i in range(0, len(query_texts), MV_BATCH)]
        for bi, batch_texts in enumerate(batches):
            batch = [{"text": t} for t in batch_texts]
            for attempt in range(3):
                try:
                    async with sem:
                        result = await sie.encode(
                            model,
                            batch,
                            output_types=["multivector"],
                            is_query=True,
                            gpu=gpu,
                            wait_for_capacity=True,
                            provision_timeout_s=PROVISION_TIMEOUT,
                        )
                        query_mvs.extend(r["multivector"] for r in result)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"  Encode retry {attempt + 1}/3: {type(e).__name__}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise
            if (bi + 1) % 20 == 0:
                logger.info(f"  MV query {model}: {bi + 1}/{len(batches)}")
        np.savez(query_cache, **{f"mv_{i}": mv for i, mv in enumerate(query_mvs)})

    # Brute-force MaxSim
    corpus_ids = [item["corpus_id"] for item in corpus_items]
    normed_corpus = [normalize_mv(mv) for mv in corpus_mvs]
    normed_queries = [normalize_mv(q) for q in query_mvs]

    results = []
    for i, q_mv in enumerate(normed_queries):
        scores = maxsim_cpu.maxsim_scores_variable(q_mv, normed_corpus)
        scored = sorted(zip(corpus_ids, scores), key=lambda x: x[1], reverse=True)
        results.append([cid for cid, _ in scored])
        if (i + 1) % 200 == 0:
            logger.info(f"  MV direct {model}: {i + 1}/{len(query_mvs)}")

    with open(direct_cache, "w") as f:
        json.dump(results, f)
    ndcg, recall = evaluate(results, query_items, qrel_map)
    return ndcg, recall, 0, "ok"


TPUF_BATCH = 256


async def test_encoder(sie, tpuf, model, corpus_texts, query_texts, corpus_items, query_items, qrel_map, gpu, sem):
    """Test a dense encoder: encode → Turbopuffer index → ANN + BM25 → hybrid pool → evaluate."""
    slug = slugify(model)
    ns_name = f"ablation-{slug}"
    corpus_cache = CACHE_DIR / f"dense_corpus_{slug}.npz"
    query_cache = CACHE_DIR / f"dense_query_{slug}.npz"
    vec_cache = CACHE_DIR / f"vector_top{TOP_K}_{slug}.json"
    bm25_cache = CACHE_DIR / f"bm25_top{TOP_K}_{slug}.json"
    results_cache = CACHE_DIR / f"autoresearch_dense_{slug}.json"

    if results_cache.exists():
        results = json.load(open(results_cache))
        ndcg, recall = evaluate(results, query_items, qrel_map)
        return ndcg, recall, 0, "cached"

    # 1. Encode corpus
    if corpus_cache.exists():
        corpus_vecs = np.load(corpus_cache)["vectors"]
        logger.info(f"  Cache hit corpus: {corpus_cache}")
    else:
        all_vecs = []
        batches = [corpus_texts[i : i + ENCODE_BATCH] for i in range(0, len(corpus_texts), ENCODE_BATCH)]
        for bi, batch_texts in enumerate(batches):
            batch = [{"text": t} for t in batch_texts]
            for attempt in range(3):
                try:
                    async with sem:
                        result = await sie.encode(
                            model,
                            batch,
                            output_types=["dense"],
                            is_query=False,
                            gpu=gpu,
                            wait_for_capacity=True,
                            provision_timeout_s=PROVISION_TIMEOUT,
                        )
                        all_vecs.extend(r["dense"] for r in result)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"  Encode retry {attempt + 1}/3: {type(e).__name__}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise
            if (bi + 1) % 10 == 0:
                logger.info(f"  Dense corpus {model}: {bi + 1}/{len(batches)}")
        corpus_vecs = np.array(all_vecs, dtype=np.float32)
        corpus_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(corpus_cache, vectors=corpus_vecs)

    # 2. Encode queries
    if query_cache.exists():
        query_vecs = np.load(query_cache)["vectors"]
        logger.info(f"  Cache hit query: {query_cache}")
    else:
        all_vecs = []
        batches = [query_texts[i : i + ENCODE_BATCH] for i in range(0, len(query_texts), ENCODE_BATCH)]
        for bi, batch_texts in enumerate(batches):
            batch = [{"text": t} for t in batch_texts]
            for attempt in range(3):
                try:
                    async with sem:
                        result = await sie.encode(
                            model,
                            batch,
                            output_types=["dense"],
                            is_query=True,
                            gpu=gpu,
                            wait_for_capacity=True,
                            provision_timeout_s=PROVISION_TIMEOUT,
                        )
                        all_vecs.extend(r["dense"] for r in result)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"  Encode retry {attempt + 1}/3: {type(e).__name__}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise
            if (bi + 1) % 10 == 0:
                logger.info(f"  Dense query {model}: {bi + 1}/{len(batches)}")
        query_vecs = np.array(all_vecs, dtype=np.float32)
        np.savez(query_cache, vectors=query_vecs)

    # 3. Index in Turbopuffer (namespace = ablation-{model-slug})
    ns = tpuf.namespace(ns_name)
    logger.info(f"  Indexing {len(corpus_items)} docs in tpuf namespace '{ns_name}' (dim={corpus_vecs.shape[1]})")
    for i in range(0, len(corpus_items), TPUF_BATCH):
        end = min(i + TPUF_BATCH, len(corpus_items))
        await ns.write(
            distance_metric="cosine_distance",
            upsert_columns={
                "id": [str(corpus_items[j]["corpus_id"]) for j in range(i, end)],
                "vector": [corpus_vecs[j].tolist() for j in range(i, end)],
                "text": [corpus_texts[j] for j in range(i, end)],
            },
            schema={"text": {"type": "string", "full_text_search": {"tokenizer": "word_v2"}}},
        )
    logger.info(f"  Indexed {len(corpus_items)} docs in '{ns_name}'")

    # 4. ANN vector search
    if vec_cache.exists():
        vec_results = json.load(open(vec_cache))
        logger.info(f"  Cache hit vector search: {vec_cache}")
    else:

        async def vec_query(qv):
            r = await ns.query(rank_by=("vector", "ANN", qv.tolist()), top_k=TOP_K, include_attributes=False)
            return [int(row.id) for row in r.rows]

        vec_results = list(await asyncio.gather(*[vec_query(qv) for qv in query_vecs]))
        with open(vec_cache, "w") as f:
            json.dump(vec_results, f)
        logger.info(f"  Vector search: {len(vec_results)} queries, top-{TOP_K}")

    # 5. BM25 search (same namespace, same text)
    if bm25_cache.exists():
        bm25_results = json.load(open(bm25_cache))
        logger.info(f"  Cache hit BM25 search: {bm25_cache}")
    else:

        async def bm25_query(q):
            r = await ns.query(rank_by=("text", "BM25", q), top_k=TOP_K, include_attributes=False)
            return [int(row.id) for row in r.rows]

        bm25_results = list(await asyncio.gather(*[bm25_query(q) for q in query_texts]))
        with open(bm25_cache, "w") as f:
            json.dump(bm25_results, f)
        logger.info(f"  BM25 search: {len(bm25_results)} queries, top-{TOP_K}")

    # 6. Evaluate: vector-only, BM25, hybrid pool
    hybrid = [list(dict.fromkeys(v[:TOP_K] + b[:TOP_K])) for v, b in zip(vec_results, bm25_results)]

    vec_ndcg, vec_recall = evaluate(vec_results, query_items, qrel_map)
    bm25_ndcg, bm25_recall = evaluate(bm25_results, query_items, qrel_map)
    hyb_ndcg, hyb_recall = evaluate(hybrid, query_items, qrel_map)

    logger.info(f"  {model} vector-only: NDCG={vec_ndcg}, R@10={vec_recall}")
    logger.info(f"  {model} BM25:        NDCG={bm25_ndcg}, R@10={bm25_recall}")
    logger.info(f"  {model} hybrid:      NDCG={hyb_ndcg}, R@10={hyb_recall}")

    # Save hybrid as the main result (best of the three for comparison)
    best_results = hybrid if hyb_ndcg >= vec_ndcg else vec_results
    best_ndcg = max(hyb_ndcg, vec_ndcg)
    best_recall = hyb_recall if hyb_ndcg >= vec_ndcg else vec_recall

    with open(results_cache, "w") as f:
        json.dump(best_results, f)
    return best_ndcg, best_recall, 0, "ok"


async def main():
    parser = argparse.ArgumentParser(description="Autoresearch: scan SIE models")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--type", choices=["reranker", "colbert", "encoder", "ensemble", "pool", "all"], default="all")
    parser.add_argument("--models", help="Comma-separated model filter (substring match)")
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    def should_test(model_name):
        if not model_filter:
            return True
        return any(f.lower() in model_name.lower() for f in model_filter)

    sie_url = _require_env("SIE_BASE_URL")
    sie_key = os.getenv("SIE_API_KEY") or None

    sem = asyncio.Semaphore(3)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    corpus_items, query_items, qrel_map = load_dataset()
    corpus_texts = [item["text"] for item in corpus_items]
    query_texts = [qi["text"] for qi in query_items]
    text_map = {item["corpus_id"]: item["text"] for item in corpus_items}

    # Load cached search results for hybrid pool
    bm25_path = CACHE_DIR / f"bm25_top{TOP_K}.json"
    vec_path = CACHE_DIR / f"vector_top{TOP_K}.json"

    if not bm25_path.exists() or not vec_path.exists():
        logger.error("Run benchmark_ablation.py first to generate search caches")
        sys.exit(1)

    bm25 = json.load(open(bm25_path))
    vec = json.load(open(vec_path))
    hybrid_pools = [list(dict.fromkeys(b[:TOP_K] + v[:TOP_K])) for b, v in zip(bm25, vec)]

    # Load MV retrieval results for pool experiments
    mv_bge_path = CACHE_DIR / "retrieve_mv_baai-bge-m3.json"
    mv_jina_path = CACHE_DIR / "retrieve_mv_jinaai-jina-colbert-v2.json"
    vec100_path = CACHE_DIR / "vector_top100.json"

    logger.info(f"Hybrid pool: ~{np.mean([len(p) for p in hybrid_pools]):.0f} candidates/query")
    logger.info(
        f"Models to test: rerankers={len(RERANKERS)}, colbert={len(COLBERT_MODELS)}, encoders={len(DENSE_ENCODERS)}"
    )

    tpuf_key = os.environ.get("TURBOPUFFER_API_KEY")
    tpuf = AsyncTurbopuffer(api_key=tpuf_key, region="aws-us-east-1") if tpuf_key else None

    async with SIEAsyncClient(sie_url, api_key=sie_key, timeout_s=TIMEOUT_S, max_connections=5) as sie:
        # Test rerankers
        if args.type in ("reranker", "all"):
            logger.info("\n=== RERANKERS ===")
            for model in RERANKERS:
                if not should_test(model):
                    continue
                logger.info(f"\nTesting: {model}")
                t0 = time.perf_counter()
                try:
                    ndcg, recall, _, status = await asyncio.wait_for(
                        test_reranker(
                            sie, model, query_texts, hybrid_pools, text_map, query_items, qrel_map, args.gpu, sem
                        ),
                        timeout=TIMEOUT_S,
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info(f"  {model}: NDCG={ndcg}, Recall={recall} ({elapsed:.0f}s) [{status}]")
                    log_result(model, "reranker", ndcg, recall, elapsed, status)
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    logger.error(f"  {model}: FAILED ({elapsed:.0f}s) — {type(e).__name__}: {e}")
                    log_result(model, "reranker", 0, 0, elapsed, "failed", str(e)[:100])

        # Test ColBERT models
        if args.type in ("colbert", "all"):
            logger.info("\n=== COLBERT / MULTI-VECTOR ===")
            for model in COLBERT_MODELS:
                if not should_test(model):
                    continue
                logger.info(f"\nTesting: {model}")
                t0 = time.perf_counter()
                try:
                    ndcg, recall, _, status = await asyncio.wait_for(
                        test_colbert(
                            sie, model, corpus_texts, query_texts, corpus_items, query_items, qrel_map, args.gpu, sem
                        ),
                        timeout=TIMEOUT_S,
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info(f"  {model}: NDCG={ndcg}, Recall={recall} ({elapsed:.0f}s) [{status}]")
                    log_result(model, "colbert", ndcg, recall, elapsed, status)
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    logger.error(f"  {model}: FAILED ({elapsed:.0f}s) — {type(e).__name__}: {e}")
                    log_result(model, "colbert", 0, 0, elapsed, "failed", str(e)[:100])

        # Test dense encoders
        if args.type in ("encoder", "all"):
            logger.info("\n=== DENSE ENCODERS ===")
            for model in DENSE_ENCODERS:
                if not should_test(model):
                    continue
                logger.info(f"\nTesting: {model}")
                t0 = time.perf_counter()
                try:
                    if not tpuf:
                        logger.error("  TURBOPUFFER_API_KEY required for encoder tests")
                        break
                    ndcg, recall, _, status = await asyncio.wait_for(
                        test_encoder(
                            sie,
                            tpuf,
                            model,
                            corpus_texts,
                            query_texts,
                            corpus_items,
                            query_items,
                            qrel_map,
                            args.gpu,
                            sem,
                        ),
                        timeout=TIMEOUT_S,
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info(f"  {model}: NDCG={ndcg}, Recall={recall} ({elapsed:.0f}s) [{status}]")
                    log_result(model, "encoder", ndcg, recall, elapsed, status)
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    logger.error(f"  {model}: FAILED ({elapsed:.0f}s) — {type(e).__name__}: {e}")
                    log_result(model, "encoder", 0, 0, elapsed, "failed", str(e)[:100])

        # ── Ensemble experiments (CPU-only, uses cached CE results) ──────
        if args.type in ("ensemble", "all"):
            logger.info("\n=== ENSEMBLE EXPERIMENTS ===")
            ce_results = {}
            for path in CACHE_DIR.glob("autoresearch_ce_*.json"):
                model_slug = path.stem.replace("autoresearch_ce_", "")
                ce_results[model_slug] = json.load(open(path))
            logger.info(f"  Loaded {len(ce_results)} cached CE results: {list(ce_results.keys())}")

            def rrf_ensemble(result_lists, weights=None, k=60):
                if weights is None:
                    weights = [1.0] * len(result_lists)
                combined = []
                for qi in range(len(result_lists[0])):
                    scores = {}
                    for results, w in zip(result_lists, weights):
                        for rank, cid in enumerate(results[qi]):
                            scores[cid] = scores.get(cid, 0) + w / (k + rank + 1)
                    combined.append(sorted(scores, key=scores.get, reverse=True))
                return combined

            ensembles = []
            slugs = list(ce_results.keys())

            # All pairs with equal weight
            for i, s1 in enumerate(slugs):
                for s2 in slugs[i + 1 :]:
                    combo = rrf_ensemble([ce_results[s1], ce_results[s2]])
                    ensembles.append((f"RRF({s1} + {s2})", combo))

            # All triples with equal weight
            for i, s1 in enumerate(slugs):
                for j, s2 in enumerate(slugs[i + 1 :], i + 1):
                    for s3 in slugs[j + 1 :]:
                        combo = rrf_ensemble([ce_results[s1], ce_results[s2], ce_results[s3]])
                        ensembles.append((f"RRF({s1} + {s2} + {s3})", combo))

            # Weighted variants: best model gets 3x
            if len(slugs) >= 2:
                # Find best single model
                best_slug = None
                best_ndcg = 0
                for slug, results in ce_results.items():
                    ndcg, _ = evaluate(results, query_items, qrel_map)
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_slug = slug
                others = [s for s in slugs if s != best_slug]
                for other in others:
                    combo = rrf_ensemble([ce_results[best_slug], ce_results[other]], [3, 1])
                    ensembles.append((f"RRF(3:1 {best_slug}:{other})", combo))
                # Best + all others weighted 3:1:1:...
                if len(others) >= 2:
                    all_results = [ce_results[best_slug]] + [ce_results[s] for s in others]
                    weights = [3.0] + [1.0] * len(others)
                    combo = rrf_ensemble(all_results, weights)
                    ensembles.append((f"RRF(3:1x{len(others)} {best_slug}+rest)", combo))

            for name, combo in ensembles:
                ndcg, recall = evaluate(combo, query_items, qrel_map)
                logger.info(f"  {name}: NDCG={ndcg}, Recall={recall}")
                log_result(name, "ensemble", ndcg, recall, 0, "ok")

        # ── Pool experiments: MV-based pools + CE rerank ──────
        if args.type in ("pool", "all"):
            logger.info("\n=== POOL EXPERIMENTS ===")
            best_ce_model = "mixedbread-ai/mxbai-rerank-large-v2"

            pool_configs = []
            # A. MV-bge top-200 pool (0.89 recall vs 0.77 current)
            if mv_bge_path.exists():
                mv_bge = json.load(open(mv_bge_path))
                pool_configs.append(("mv-bge200", [m[:200] for m in mv_bge]))

            # B. Vec100 + MV-bge100 union (0.90 recall, ~211 candidates)
            if mv_bge_path.exists() and vec100_path.exists():
                vec100 = json.load(open(vec100_path))
                pool_configs.append(
                    ("vec100-mv-bge100", [list(dict.fromkeys(v[:100] + m[:100])) for v, m in zip(vec100, mv_bge)])
                )

            # C. MV-bge200 + MV-jina200 union (0.94 recall, ~293 candidates)
            if mv_bge_path.exists() and mv_jina_path.exists():
                mv_jina = json.load(open(mv_jina_path))
                pool_configs.append(
                    ("mv-bge200-jina200", [list(dict.fromkeys(m[:200] + j[:200])) for m, j in zip(mv_bge, mv_jina)])
                )

            for pool_name, pools in pool_configs:
                slug = f"pool-{pool_name}-mxbai-large"
                avg_size = np.mean([len(p) for p in pools])
                logger.info(f"\n  Pool '{pool_name}': ~{avg_size:.0f} candidates/query")
                logger.info(f"  CE-scoring {pool_name} with {best_ce_model}...")
                t0 = time.perf_counter()
                ndcg, recall, _, status = await test_reranker(
                    sie,
                    best_ce_model,
                    query_texts,
                    pools,
                    text_map,
                    query_items,
                    qrel_map,
                    args.gpu,
                    sem,
                    cache_slug=slug,
                )
                elapsed = time.perf_counter() - t0
                logger.info(f"  {pool_name} + CE: NDCG={ndcg}, Recall={recall} ({elapsed:.0f}s) [{status}]")
                log_result(f"pool:{pool_name}+mxbai-large", "pool", ndcg, recall, elapsed, status)

    # Print leaderboard
    if RESULTS_FILE.exists():
        logger.info("\n=== LEADERBOARD ===")
        with open(RESULTS_FILE) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = sorted(reader, key=lambda r: float(r.get("ndcg10") or 0), reverse=True)
        for r in rows[:15]:
            ndcg = r.get("ndcg10") or "?"
            recall = r.get("recall10") or "?"
            logger.info(f"  {r['model']:<50s} NDCG={ndcg:>6s}  R@10={recall:>6s}  {r['status']}")


if __name__ == "__main__":
    asyncio.run(main())
