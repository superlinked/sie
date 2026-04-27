"""Retrieval Ablation Benchmark — 6 conditions, async, incremental caching.

Conditions:
1. BM25-only (Turbopuffer FTS)
2. Vector-only (bge-m3 dense ANN)
3. RRF(BM25 + Vector)
4. Cross-Encoder Rerank over hybrid pool (union of BM25 + Vector top-25)
5. Multi-Vector Rerank over hybrid pool (MaxSim on candidates)
6. Multi-Vector Direct Retrieval (brute-force MaxSim over full corpus)

Usage:
    uv run python benchmark_ablation.py --gpu l4-spot
    uv run python benchmark_ablation.py --gpu l4-spot --skip-conditions 5,6
    uv run python benchmark_ablation.py --gpu rtx6000-spot --skip-conditions 1,2,3,4,5
"""

import argparse
import asyncio
import csv
import json
import math
import os
import re
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

# ── Config ──────────────────────────────────────────────────────────────────


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        logger.error(f"Missing required env var: {name}. Set it in .env or shell.")
        sys.exit(1)
    return val


PROVISION_TIMEOUT = 900
RANDOM_SEED = 42
MAX_TRANSPORT_RETRIES = 3

ENCODE_BATCH_SIZE = 64
MV_ENCODE_BATCH_SIZE = 16
TPUF_BATCH_SIZE = 500
TOP_K_RETRIEVE = 50
TOP_K_EVAL = 10
RRF_K = 60
CHECKPOINT_INTERVAL = 100
# SDK retries 503 internally; keep concurrent requests low to avoid thundering herd
GPU_CONCURRENCY = 10

ENCODER = "BAAI/bge-m3"

ALL_CE_RERANKERS = {
    "mxbai-rerank": "mixedbread-ai/mxbai-rerank-base-v2",
    "mxbai-rerank-large": "mixedbread-ai/mxbai-rerank-large-v2",
    "bge-reranker": "BAAI/bge-reranker-v2-m3",
}
ALL_MV_MODELS = {
    "bge-m3": {"model": "BAAI/bge-m3", "dim": 1024, "max_tokens": 8192},
    "colbert": {"model": "mixedbread-ai/mxbai-colbert-large-v1", "dim": 128, "max_tokens": 512},
    "jina-colbert": {"model": "jinaai/jina-colbert-v2", "dim": 128, "max_tokens": 8192},
    "modern-colbert": {"model": "lightonai/GTE-ModernColBERT-v1", "dim": 128, "max_tokens": 8192},
    "colbertv2": {"model": "colbert-ir/colbertv2.0", "dim": 128, "max_tokens": 512},
}

CACHE_DIR = Path(__file__).parent / "cache" / "ablation"
RESULTS_CSV = Path(__file__).parent / "ablation_results.csv"


# ── Caching ────────────────────────────────────────────────────────────────


class NpzCache:
    def __init__(self, path: Path | None):
        self.path = path

    def exists(self) -> bool:
        return self.path is not None and self.path.exists()

    def load_dense(self) -> list:
        return list(np.load(self.path)["vectors"])

    def save_dense(self, vectors: list):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, vectors=np.array(vectors))

    def load_multivector(self) -> list:
        data = np.load(self.path, allow_pickle=False)
        return [data[f"mv_{i}"] for i in range(len(data.files))]

    def save_multivector(self, mvs: list):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **{f"mv_{i}": mv for i, mv in enumerate(mvs)})


class JsonCache:
    def __init__(self, path: Path | None):
        self.path = path

    def exists(self) -> bool:
        return self.path is not None and self.path.exists()

    def load(self) -> list:
        with open(self.path) as f:
            return json.load(f)

    def save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f)


# ── Utilities ───────────────────────────────────────────────────────────────


def normalize_mv(mv):
    mv = np.asarray(mv, dtype=np.float32)
    norms = np.linalg.norm(mv, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return mv / norms


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")


def build_hybrid_pool(bm25_ranked, vec_ranked):
    """Union of BM25 top-K + Vector top-K, deduplicated, order preserved."""
    return list(dict.fromkeys(bm25_ranked[:TOP_K_RETRIEVE] + vec_ranked[:TOP_K_RETRIEVE]))


# ── Metrics ─────────────────────────────────────────────────────────────────


def ndcg_at_k(ranked_ids, qrel_map, k=10):
    dcg = sum(qrel_map.get(cid, 0) / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]))
    ideal_rels = sorted(qrel_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_ids, qrel_map, k=10):
    for i, cid in enumerate(ranked_ids[:k]):
        if qrel_map.get(cid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids, qrel_map, k=10):
    relevant = {cid for cid, s in qrel_map.items() if s > 0}
    if not relevant:
        return 0.0
    return sum(1 for cid in ranked_ids[:k] if cid in relevant) / len(relevant)


def evaluate(ranked_results, query_items, qrel_map, k=TOP_K_EVAL):
    ndcgs, mrrs, recalls = [], [], []
    for ranked, qi in zip(ranked_results, query_items):
        qrels = qrel_map.get(qi["query_id"], {})
        if not qrels:
            continue
        ndcgs.append(ndcg_at_k(ranked, qrels, k))
        mrrs.append(mrr_at_k(ranked, qrels, k))
        recalls.append(recall_at_k(ranked, qrels, k))
    return {
        "ndcg@10": round(float(np.mean(ndcgs)), 4),
        "mrr@10": round(float(np.mean(mrrs)), 4),
        "recall@10": round(float(np.mean(recalls)), 4),
        "n_queries": len(ndcgs),
    }


def rrf_fuse(ranked_lists, k=RRF_K):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ── Output ──────────────────────────────────────────────────────────────────


def write_results_csv(results_log, path):
    if not results_log:
        return
    all_keys = sorted({k for r in results_log for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results_log)


def append_result(results_log, result):
    results_log.append(result)
    write_results_csv(results_log, RESULTS_CSV)
    logger.info(
        f"  -> {result.get('condition')}: NDCG@10={result.get('ndcg@10')} [{len(results_log)} results in {RESULTS_CSV}]"
    )


def print_summary(results_log):
    logger.info(f"\n{'=' * 100}")
    logger.info("ABLATION RESULTS")
    logger.info(f"{'=' * 100}")
    hdr = f"{'#':<4} {'Condition':<25} {'Reranker':<42} {'NDCG@10':>8} {'MRR@10':>8} {'Recall@10':>10}"
    logger.info(hdr)
    logger.info("-" * 100)
    for r in sorted(results_log, key=lambda x: x.get("ndcg@10", 0), reverse=True):
        logger.info(
            f"{r.get('condition_num', ''):<4} "
            f"{r.get('condition', ''):<25} "
            f"{r.get('reranker', '-'):<42} "
            f"{r.get('ndcg@10', 0):>8.4f} "
            f"{r.get('mrr@10', 0):>8.4f} "
            f"{r.get('recall@10', 0):>10.4f}"
        )
    logger.info(f"{'=' * 100}")


# ── Data Loading ────────────────────────────────────────────────────────────


def load_dataset():
    from datasets import load_dataset as hf_load

    t0 = time.perf_counter()
    corpus = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "corpus", split="test").data.table)
    queries = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "queries", split="test").data.table)
    qrels_df = pl.from_arrow(hf_load("vidore/vidore_v3_finance_en", "qrels", split="test").data.table)
    logger.info(f"Dataset loaded in {time.perf_counter() - t0:.1f}s")

    qrel_map = {}
    for row in qrels_df.iter_rows(named=True):
        qrel_map.setdefault(row["query_id"], {})[row["corpus_id"]] = row["score"]

    corpus_items = []
    for row in corpus.iter_rows(named=True):
        text = row["markdown"] or ""
        if len(text.strip()) < 10:
            text = f"Page {row['page_number_in_doc']} of {row['doc_id']}"
        corpus_items.append(
            {
                "corpus_id": row["corpus_id"],
                "text": text,
                "doc_id": row["doc_id"],
                "page_number": row["page_number_in_doc"],
            }
        )

    query_items = [{"query_id": r["query_id"], "text": r["query"]} for r in queries.iter_rows(named=True)]
    logger.info(
        f"Corpus: {len(corpus_items)}, Queries: {len(query_items)}, Qrels: {sum(len(v) for v in qrel_map.values())}"
    )
    return corpus_items, query_items, qrel_map


# ── Async Encode (SDK handles 503 retries; we retry transport errors) ──────


async def _encode_with_retry(sie, semaphore, model, batch, output_types, is_query, gpu):
    for attempt in range(MAX_TRANSPORT_RETRIES):
        try:
            async with semaphore:
                return await sie.encode(
                    model,
                    batch,
                    output_types=output_types,
                    is_query=is_query,
                    gpu=gpu,
                    wait_for_capacity=True,
                    provision_timeout_s=PROVISION_TIMEOUT,
                )
        except Exception as e:
            logger.warning(f"  Encode error (attempt {attempt + 1}/{MAX_TRANSPORT_RETRIES}, {type(e).__name__}): {e}")
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Encode failed after {MAX_TRANSPORT_RETRIES} retries")


async def encode_dense(sie, model, texts, is_query, gpu, semaphore, cache_path=None):
    cache = NpzCache(cache_path)
    if cache.exists():
        logger.info(f"Cache hit: {cache_path}")
        return cache.load_dense()

    batches = [texts[i : i + ENCODE_BATCH_SIZE] for i in range(0, len(texts), ENCODE_BATCH_SIZE)]
    done = 0

    async def do_one(batch_texts):
        nonlocal done
        batch = [{"text": t} for t in batch_texts]
        result = await _encode_with_retry(sie, semaphore, model, batch, ["dense"], is_query, gpu)
        done += 1
        if done % 5 == 0 or done == len(batches):
            logger.info(f"  Dense encode: {done}/{len(batches)} batches")
        return result

    results_list = await asyncio.gather(*[do_one(b) for b in batches])
    all_vectors = []
    for results in results_list:
        all_vectors.extend(r["dense"] for r in results)

    if cache_path:
        cache.save_dense(all_vectors)
        logger.info(f"  Cached {len(all_vectors)} dense vectors")
    return all_vectors


async def encode_multivector(sie, model, texts, is_query, gpu, semaphore, cache_path=None):
    cache = NpzCache(cache_path)
    if cache.exists():
        logger.info(f"Cache hit: {cache_path}")
        return cache.load_multivector()

    batches = [texts[i : i + MV_ENCODE_BATCH_SIZE] for i in range(0, len(texts), MV_ENCODE_BATCH_SIZE)]
    done = 0

    async def do_one(batch_texts):
        nonlocal done
        batch = [{"text": t} for t in batch_texts]
        result = await _encode_with_retry(sie, semaphore, model, batch, ["multivector"], is_query, gpu)
        done += 1
        if done % 10 == 0 or done == len(batches):
            logger.info(f"  MV encode ({model}): {done}/{len(batches)} batches")
        return result

    results_list = await asyncio.gather(*[do_one(b) for b in batches])
    all_mvs = []
    for results in results_list:
        all_mvs.extend(r["multivector"] for r in results)

    if cache_path:
        cache.save_multivector(all_mvs)
        logger.info(f"  Cached {len(all_mvs)} multivectors ({model})")
    return all_mvs


# ── Turbopuffer (async) ───────────────────────────────────────────────────


async def index_corpus(tpuf, corpus_items, vectors, ns_name):
    ns = tpuf.namespace(ns_name)
    for i in range(0, len(corpus_items), TPUF_BATCH_SIZE):
        end = min(i + TPUF_BATCH_SIZE, len(corpus_items))
        batch_items, batch_vecs = corpus_items[i:end], vectors[i:end]
        await ns.write(
            distance_metric="cosine_distance",
            upsert_columns={
                "id": [str(item["corpus_id"]) for item in batch_items],
                "vector": [v.tolist() for v in batch_vecs],
                "text": [item["text"] for item in batch_items],
                "doc_id": [item["doc_id"] for item in batch_items],
                "page_number": [item["page_number"] for item in batch_items],
            },
            schema={"text": {"type": "string", "full_text_search": {"tokenizer": "word_v2"}}},
        )
    logger.info(f"Indexed {len(corpus_items)} docs in {ns_name}")


async def search_bm25(tpuf, ns_name, query_texts, cache_path=None):
    cache = JsonCache(cache_path)
    if cache.exists():
        logger.info(f"Cache hit: {cache_path}")
        return cache.load()

    ns = tpuf.namespace(ns_name)

    async def query_one(q):
        r = await ns.query(rank_by=("text", "BM25", q), top_k=TOP_K_RETRIEVE, include_attributes=False)
        return [int(row.id) for row in r.rows]

    results = list(await asyncio.gather(*[query_one(q) for q in query_texts]))
    if cache_path:
        cache.save(results)
    logger.info(f"BM25 search: {len(results)} queries, top-{TOP_K_RETRIEVE}")
    return results


async def search_vector(tpuf, ns_name, query_vectors, cache_path=None):
    cache = JsonCache(cache_path)
    if cache.exists():
        logger.info(f"Cache hit: {cache_path}")
        return cache.load()

    ns = tpuf.namespace(ns_name)

    async def query_one(qv):
        r = await ns.query(rank_by=("vector", "ANN", qv.tolist()), top_k=TOP_K_RETRIEVE, include_attributes=False)
        return [int(row.id) for row in r.rows]

    results = list(await asyncio.gather(*[query_one(qv) for qv in query_vectors]))
    if cache_path:
        cache.save(results)
    logger.info(f"Vector search: {len(results)} queries, top-{TOP_K_RETRIEVE}")
    return results


# ── Cross-Encoder Reranking with partial cache ─────────────────────────────


async def rerank_cross_encoder(sie, model, query_texts, candidate_ids_list, text_map, gpu, semaphore, cache_path=None):
    cache = JsonCache(cache_path)
    if cache.exists():
        logger.info(f"Cache hit: {cache_path}")
        return cache.load()

    n = len(query_texts)
    partial_path = cache_path.with_suffix(".partial.json") if cache_path else None

    results = [None] * n
    done = 0
    if partial_path and partial_path.exists():
        with open(partial_path) as f:
            partial = json.load(f)
        for i, r in enumerate(partial):
            if r is not None:
                results[i] = r
        done = sum(1 for r in results if r is not None)
        logger.info(f"  CE {model}: resumed {done}/{n} from checkpoint")

    remaining = [i for i in range(n) if results[i] is None]
    logger.info(f"  CE {model}: {len(remaining)} queries to score ({len(candidate_ids_list[0])} candidates/query)")

    async def score_one(idx):
        cands = candidate_ids_list[idx]
        if not cands:
            return idx, []
        items = [{"text": text_map.get(cid, "")} for cid in cands]
        for attempt in range(MAX_TRANSPORT_RETRIES):
            try:
                async with semaphore:
                    result = await sie.score(
                        model,
                        query={"text": query_texts[idx]},
                        items=items,
                        gpu=gpu,
                        wait_for_capacity=True,
                        provision_timeout_s=PROVISION_TIMEOUT,
                    )
                    # item_id = original index ("item-N" -> N), rank = output position
                    scored = sorted(
                        [(cands[int(s["item_id"].split("-")[1])], s["score"]) for s in result["scores"]],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    return idx, [cid for cid, _ in scored]
            except Exception as e:
                logger.warning(f"  CE error (attempt {attempt + 1}/{MAX_TRANSPORT_RETRIES}, {type(e).__name__}): {e}")
                await asyncio.sleep(5 * (attempt + 1))
        raise RuntimeError(f"CE score failed after {MAX_TRANSPORT_RETRIES} retries for query {idx}")

    completed = done
    batch_size = GPU_CONCURRENCY * 3  # Process in manageable batches to avoid connection pool exhaustion
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]
        for coro in asyncio.as_completed([score_one(i) for i in batch]):
            idx, ranked = await coro
            results[idx] = ranked
            completed += 1

            if completed % CHECKPOINT_INTERVAL == 0:
                if partial_path:
                    partial_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(partial_path, "w") as f:
                        json.dump(results, f)
                logger.info(f"  CE {model}: {completed}/{n} (checkpoint)")
            elif completed % 200 == 0:
                logger.info(f"  CE {model}: {completed}/{n}")

    if cache_path:
        cache.save(results)
    if partial_path and partial_path.exists():
        partial_path.unlink()
    logger.info(f"  CE {model}: done ({n} queries)")
    return results


# ── Multi-Vector Scoring (CPU, maxsim_cpu) ─────────────────────────────────


def rerank_multivector(query_mvs, normed_corpus_mv_map, candidate_ids_list, cache_path=None):
    """Rerank hybrid pool candidates using pre-normalized corpus multivectors."""
    cache = JsonCache(cache_path)
    if cache.exists():
        logger.info(f"Cache hit: {cache_path}")
        return cache.load()

    normed_queries = [normalize_mv(q) for q in query_mvs]
    all_reranked = []
    for i, (q_mv, cand_ids) in enumerate(zip(normed_queries, candidate_ids_list)):
        if not cand_ids:
            all_reranked.append([])
            continue
        doc_mvs = [normed_corpus_mv_map[cid] for cid in cand_ids]
        scores = maxsim_cpu.maxsim_scores_variable(q_mv, doc_mvs)
        scored = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
        all_reranked.append([cid for cid, _ in scored])
        if (i + 1) % 200 == 0:
            logger.info(f"  MV rerank: {i + 1}/{len(query_mvs)}")

    logger.info(f"  MV rerank done: {len(all_reranked)} queries")
    if cache_path:
        cache.save(all_reranked)
    return all_reranked


def retrieve_multivector(query_mvs, corpus_ids, normed_corpus_mvs, cache_path=None):
    """Brute-force MaxSim over pre-normalized corpus multivectors."""
    cache = JsonCache(cache_path)
    if cache.exists():
        logger.info(f"Cache hit: {cache_path}")
        return cache.load()

    normed_queries = [normalize_mv(q) for q in query_mvs]
    all_ranked = []
    for i, q_mv in enumerate(normed_queries):
        scores = maxsim_cpu.maxsim_scores_variable(q_mv, normed_corpus_mvs)
        scored = sorted(zip(corpus_ids, scores), key=lambda x: x[1], reverse=True)
        all_ranked.append([cid for cid, _ in scored])
        if (i + 1) % 100 == 0:
            logger.info(f"  MV direct: {i + 1}/{len(query_mvs)}")

    logger.info(f"  MV direct done: {len(all_ranked)} queries")
    if cache_path:
        cache.save(all_ranked)
    return all_ranked


# ── Main ────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(
        description="Retrieval Ablation Benchmark on vidore_v3_finance_en",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  uv run python benchmark_ablation.py --gpu l4-spot\n"
        "  uv run python benchmark_ablation.py --skip-conditions 5,6\n"
        "  uv run python benchmark_ablation.py --ce-models mxbai-rerank --mv-models colbert\n"
        "  uv run python benchmark_ablation.py --dry-run\n",
    )
    parser.add_argument("--gpu", default=None, help="GPU type (e.g. l4-spot, rtx6000-spot). Default: let router pick")
    parser.add_argument("--skip-conditions", default="", help="Comma-separated condition numbers to skip (1-6)")
    parser.add_argument("--namespace", default=None, help="Turbopuffer namespace. Default: ablation-{encoder-slug}")
    parser.add_argument(
        "--ce-models",
        default="all",
        help=f"CE reranker models: comma-separated from {list(ALL_CE_RERANKERS.keys())}, or 'all'",
    )
    parser.add_argument(
        "--mv-models", default="all", help=f"MV models: comma-separated from {list(ALL_MV_MODELS.keys())}, or 'all'"
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan without running")
    args = parser.parse_args()

    gpu = args.gpu
    skip = set(int(x) for x in args.skip_conditions.split(",") if x.strip())
    run_mv_rerank = 5 not in skip
    run_mv_direct = 6 not in skip
    ns_name = args.namespace or f"ablation-{slugify(ENCODER)}"

    # Resolve model selections
    if args.ce_models == "all":
        ce_rerankers = list(ALL_CE_RERANKERS.values())
    else:
        ce_rerankers = [ALL_CE_RERANKERS[k.strip()] for k in args.ce_models.split(",")]

    if args.mv_models == "all":
        mv_rerankers = list(ALL_MV_MODELS.values())
    else:
        mv_rerankers = [ALL_MV_MODELS[k.strip()] for k in args.mv_models.split(",")]

    sie_base_url = _require_env("SIE_BASE_URL")
    sie_api_key = os.getenv("SIE_API_KEY") or None

    semaphore = asyncio.Semaphore(GPU_CONCURRENCY)

    if args.dry_run:
        conditions = [c for c in range(1, 7) if c not in skip]
        logger.info("=== DRY RUN ===")
        logger.info(f"GPU: {gpu or 'router picks'}")
        logger.info(f"Namespace: {ns_name}")
        logger.info(f"Conditions: {conditions}")
        logger.info(f"CE models: {[slugify(r) for r in ce_rerankers]}")
        logger.info(f"MV models: {[m['model'] for m in mv_rerankers]}")
        logger.info(f"TOP_K: {TOP_K_RETRIEVE}, Hybrid pool: ~{TOP_K_RETRIEVE * 2} candidates")
        logger.info(f"Cache dir: {CACHE_DIR} (exists={CACHE_DIR.exists()})")
        logger.info(f"SIE endpoint: {sie_base_url}")
        logger.info(f"SIE auth: {'configured' if sie_api_key else 'not configured'}")
        logger.info("Config OK. Remove --dry-run to execute.")
        return

    tpuf_api_key = _require_env("TURBOPUFFER_API_KEY")

    np.random.seed(RANDOM_SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async with SIEAsyncClient(sie_base_url, api_key=sie_api_key, timeout_s=900, max_connections=10) as sie:
        tpuf = AsyncTurbopuffer(api_key=tpuf_api_key, region="aws-us-east-1")

        corpus_items, query_items, qrel_map = load_dataset()
        text_map = {item["corpus_id"]: item["text"] for item in corpus_items}
        corpus_texts = [item["text"] for item in corpus_items]
        query_texts = [qi["text"] for qi in query_items]
        corpus_ids = [item["corpus_id"] for item in corpus_items]

        results_log = []
        t_total = time.perf_counter()

        # ── Trial 1: Embed corpus + queries with bge-m3 ──────────────
        logger.info("\n=== TRIAL 1: Embed corpus + queries (bge-m3 dense) ===")
        t0 = time.perf_counter()

        corpus_vecs = await encode_dense(
            sie,
            ENCODER,
            corpus_texts,
            is_query=False,
            gpu=gpu,
            semaphore=semaphore,
            cache_path=CACHE_DIR / "dense_corpus.npz",
        )
        _, query_vecs = await asyncio.gather(
            index_corpus(tpuf, corpus_items, corpus_vecs, ns_name),
            encode_dense(
                sie,
                ENCODER,
                query_texts,
                is_query=True,
                gpu=gpu,
                semaphore=semaphore,
                cache_path=CACHE_DIR / "dense_query.npz",
            ),
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Trial 1: {elapsed:.1f}s ({len(corpus_texts) + len(query_texts)} items, "
            f"{(len(corpus_texts) + len(query_texts)) / max(elapsed, 0.1):.0f} items/s)"
        )

        # ── Trial 2: BM25 + Vector retrieval ──────────────────────
        logger.info(f"\n=== TRIAL 2: BM25 + Vector search (top-{TOP_K_RETRIEVE}) ===")
        t0 = time.perf_counter()

        bm25_results, vec_results = await asyncio.gather(
            search_bm25(tpuf, ns_name, query_texts, cache_path=CACHE_DIR / f"bm25_top{TOP_K_RETRIEVE}.json"),
            search_vector(tpuf, ns_name, query_vecs, cache_path=CACHE_DIR / f"vector_top{TOP_K_RETRIEVE}.json"),
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Trial 2: {elapsed:.1f}s ({len(query_texts) * 2} queries, "
            f"{len(query_texts) * 2 / max(elapsed, 0.1):.0f} queries/s)"
        )

        # ── Eval: BM25, Vector, RRF baselines ─────────────────────
        if 1 not in skip:
            m = evaluate(bm25_results, query_items, qrel_map)
            append_result(results_log, {"condition_num": 1, "condition": "BM25-only", "reranker": "-", **m})

        if 2 not in skip:
            m = evaluate(vec_results, query_items, qrel_map)
            append_result(results_log, {"condition_num": 2, "condition": "Vector-only", "reranker": "-", **m})

        if 3 not in skip:
            rrf_results = [rrf_fuse([b, v]) for b, v in zip(bm25_results, vec_results)]
            m = evaluate(rrf_results, query_items, qrel_map)
            append_result(results_log, {"condition_num": 3, "condition": "RRF(BM25+Vec)", "reranker": "-", **m})

        # ── Build hybrid pool ──────────────────────────────────────
        hybrid_pools = [build_hybrid_pool(b, v) for b, v in zip(bm25_results, vec_results)]
        pool_sizes = [len(p) for p in hybrid_pools]
        logger.info(
            f"Hybrid pool: mean={np.mean(pool_sizes):.0f}, min={min(pool_sizes)}, max={max(pool_sizes)} candidates"
        )

        # ── Trial 3: Cross-encoder rerank + Multi-vector scoring ──
        logger.info("\n=== TRIAL 3: Cross-encoder rerank + Multi-vector scoring ===")
        t0 = time.perf_counter()

        async def run_ce(reranker):
            slug = slugify(reranker)
            logger.info(f"  [CE] Start: {reranker}")
            reranked = await rerank_cross_encoder(
                sie,
                reranker,
                query_texts,
                hybrid_pools,
                text_map,
                gpu,
                semaphore,
                cache_path=CACHE_DIR / f"rerank_ce_{slug}_hybrid{TOP_K_RETRIEVE}.json",
            )
            m = evaluate(reranked, query_items, qrel_map)
            return {"condition_num": 4, "condition": f"CE-Rerank(hybrid-{TOP_K_RETRIEVE})", "reranker": reranker, **m}

        async def run_mv_pipeline(mv_config):
            model = mv_config["model"]
            slug = slugify(model)
            results = []

            logger.info(f"  [MV] Start encode: {model}")
            corpus_mvs = await encode_multivector(
                sie,
                model,
                corpus_texts,
                is_query=False,
                gpu=gpu,
                semaphore=semaphore,
                cache_path=CACHE_DIR / f"mv_corpus_{slug}.npz",
            )
            query_mvs = await encode_multivector(
                sie,
                model,
                query_texts,
                is_query=True,
                gpu=gpu,
                semaphore=semaphore,
                cache_path=CACHE_DIR / f"mv_query_{slug}.npz",
            )

            # Pre-normalize corpus once (not per-query)
            normed_corpus_mv_map = {
                corpus_items[i]["corpus_id"]: normalize_mv(corpus_mvs[i]) for i in range(len(corpus_items))
            }
            normed_corpus_mvs = [normed_corpus_mv_map[cid] for cid in corpus_ids]

            loop = asyncio.get_running_loop()

            if run_mv_rerank:
                reranked = await loop.run_in_executor(
                    None,
                    rerank_multivector,
                    query_mvs,
                    normed_corpus_mv_map,
                    hybrid_pools,
                    CACHE_DIR / f"rerank_mv_{slug}_hybrid{TOP_K_RETRIEVE}.json",
                )
                m = evaluate(reranked, query_items, qrel_map)
                results.append(
                    {
                        "condition_num": 5,
                        "condition": f"MV-Rerank(hybrid-{TOP_K_RETRIEVE})",
                        "reranker": model,
                        "mv_dim": mv_config["dim"],
                        **m,
                    }
                )

            if run_mv_direct:
                ranked = await loop.run_in_executor(
                    None,
                    retrieve_multivector,
                    query_mvs,
                    corpus_ids,
                    normed_corpus_mvs,
                    CACHE_DIR / f"retrieve_mv_{slug}.json",
                )
                m = evaluate(ranked, query_items, qrel_map)
                results.append(
                    {"condition_num": 6, "condition": "MV-Direct", "reranker": model, "mv_dim": mv_config["dim"], **m}
                )

            return results

        all_tasks = []
        if 4 not in skip:
            for r in ce_rerankers:
                all_tasks.append(run_ce(r))
        if run_mv_rerank or run_mv_direct:
            for cfg in mv_rerankers:
                all_tasks.append(run_mv_pipeline(cfg))

        for coro in asyncio.as_completed(all_tasks):
            r = await coro
            if isinstance(r, list):
                for item in r:
                    append_result(results_log, item)
            else:
                append_result(results_log, r)

        elapsed = time.perf_counter() - t0
        n_experiments = len([r for r in results_log if r.get("condition_num", 0) >= 4])
        logger.info(f"Trial 3: {elapsed:.1f}s ({n_experiments} experiments)")
        total = time.perf_counter() - t_total
        logger.info(f"Total: {total:.1f}s ({len(results_log)} conditions evaluated)")

    write_results_csv(results_log, RESULTS_CSV)
    print_summary(results_log)


if __name__ == "__main__":
    asyncio.run(main())
