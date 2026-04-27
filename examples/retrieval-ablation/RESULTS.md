# Retrieval Ablation Benchmark

## Intent

Answer one question with maximum rigor: **What is the best retrieval pipeline for page-level document search on financial 10-K filings?**

Specifically, isolate the contribution of each pipeline stage (keyword, semantic, hybrid fusion, cross-encoder reranking, multi-vector reranking) so we can make an informed cost/quality tradeoff.

**Non-goals:** OCR quality comparison (separate experiment), multi-modal retrieval, production latency optimization.

---

## Dataset

**vidore_v3_finance_en** — 6 bank 10-K filings from SEC EDGAR.

| Property | Value |
|----------|-------|
| Pages | 2,942 |
| Queries | 1,854 |
| Relevance judgments | 8,766 (1=relevant, 2=highly relevant) |
| Avg relevant docs/query | 4.7 |
| Median page text | 3,809 chars (~950 tokens) |
| Text source | Pre-extracted markdown (not OCR) |

---

## Ablation Design

| # | Condition | Retriever | Reranker | What it isolates |
|---|-----------|-----------|----------|------------------|
| 1 | BM25-only | Turbopuffer FTS (word_v2) | none | Keyword baseline |
| 2 | Vector-only | bge-m3 dense ANN | none | Semantic search baseline |
| 3 | RRF(BM25+Vector) | Fused (k=60) | none | Does hybrid beat vector-only? |
| 4 | Cross-Encoder Rerank | Hybrid pool (25 BM25 + 25 Vector) | mxbai-rerank, bge-reranker | Cross-encoder value over retrieval |
| 5 | Multi-Vector Rerank | Hybrid pool (25 BM25 + 25 Vector) | 5 ColBERT models | Multi-vector vs cross-encoder |
| 6 | Multi-Vector Direct | Brute-force MaxSim (full corpus) | 5 ColBERT models | MV as standalone retriever |

**Key controls:**
- Conditions 4 and 5 rerank the **same** hybrid pool (~46 candidates) — only the reranker changes
- Condition 6 uses the **same** MV encodings as condition 5, but searches the full corpus (no pre-filtering)
- bge-m3 appears as dense encoder (condition 2), MV reranker (condition 5), and MV retriever (condition 6) — isolates representation type AND retrieval strategy
- All metrics computed identically across conditions (same eval code, same qrels)

---

## Models

### Dense Encoder
| Model | Dim | Max tokens | Why |
|-------|-----|------------|-----|
| BAAI/bge-m3 | 1024 | 8192 | Best from Phase 1 (NDCG@10=0.3972) |

### Cross-Encoder Rerankers (Condition 4)
| Model | Scoring | Notes |
|-------|---------|-------|
| mixedbread-ai/mxbai-rerank-base-v2 | Server-side `sie.score()` | Strong general-purpose |
| BAAI/bge-reranker-v2-m3 | Server-side `sie.score()` | Matches our encoder family |

### Multi-Vector / Late Interaction Models (Conditions 5-6)
| Model | Dim | Max tokens | Scoring | Notes |
|-------|-----|------------|---------|-------|
| BAAI/bge-m3 | 1024 | 8192 | Client-side MaxSim | Same model as encoder, MV output |
| jinaai/jina-colbert-v2 | 128 | 8192 | Client-side MaxSim | Best cost/quality — 96% of bge-m3 at 12.5% storage |
| lightonai/GTE-ModernColBERT-v1 | 128 | 8192 | Client-side MaxSim | Modern architecture |
| mixedbread-ai/mxbai-colbert-large-v1 | 128 | 512 | Client-side MaxSim | Dedicated ColBERT |
| colbert-ir/colbertv2.0 | 128 | 512 | Client-side MaxSim | Original ColBERT baseline |

---

## Infrastructure

| Component | Details |
|-----------|---------|
| SIE cluster | Self-hosted, KEDA autoscaling, L4 + RTX6000 GPUs |
| Endpoint | From `SIE_BASE_URL` env var |
| Turbopuffer | aws-us-east-1, cosine_distance, word_v2 FTS tokenizer |
| MaxSim | [maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu) — optimized C library, CPU-only |

---

## Metrics

| Metric | Description | k |
|--------|-------------|---|
| NDCG@10 | Normalized Discounted Cumulative Gain | 10 |
| MRR@10 | Mean Reciprocal Rank | 10 |
| Recall@10 | Fraction of relevant docs in top 10 | 10 |

All computed against official qrels (score 1=relevant, 2=highly relevant).

---

## Results

**Sorted by NDCG@10 (hybrid pool = union of top-K BM25 + top-K Vector):**

### Primary Ablation (TOP_K=25, ~46 candidates)

| # | Condition | Model | Dim | Tokens | NDCG@10 | MRR@10 | R@10 |
|---|-----------|-------|-----|--------|---------|--------|------|
| 4 | CE Rerank | mxbai-rerank-base-v2 | — | — | 0.5098 | 0.6228 | 0.5587 |
| 4 | CE Rerank | bge-reranker-v2-m3 | — | — | 0.5069 | 0.6321 | 0.5558 |
| 6 | MV Direct | bge-m3 | 1024 | 8192 | 0.4354 | 0.581 | 0.4815 |
| 5 | MV Rerank | bge-m3 | 1024 | 8192 | 0.433 | 0.5808 | 0.4737 |
| 5 | MV Rerank | **jina-colbert-v2** | 128 | 8192 | **0.431** | 0.548 | 0.4937 |
| 6 | MV Direct | **jina-colbert-v2** | 128 | 8192 | **0.4187** | 0.5322 | 0.486 |
| 2 | Vector | bge-m3 dense | 1024 | — | 0.3962 | 0.5317 | 0.4377 |
| 3 | RRF | — | — | — | 0.3583 | 0.4505 | 0.4337 |
| 5 | MV Rerank | GTE-ModernColBERT | 128 | 8192 | 0.3439 | 0.4188 | 0.4241 |
| 6 | MV Direct | GTE-ModernColBERT | 128 | 8192 | 0.2853 | 0.355 | 0.3437 |
| 5 | MV Rerank | mxbai-colbert | 128 | 512 | 0.2415 | 0.3101 | 0.3034 |
| 5 | MV Rerank | colbertv2.0 | 128 | 512 | 0.2146 | 0.2626 | 0.2816 |
| 1 | BM25 | — | — | — | 0.1849 | 0.2115 | 0.2386 |
| 6 | MV Direct | mxbai-colbert | 128 | 512 | 0.1768 | 0.2392 | 0.2110 |
| 6 | MV Direct | colbertv2.0 | 128 | 512 | 0.1667 | 0.2139 | 0.206 |

### Reranker Sweep with Larger Pool (TOP_K=50, ~89 candidates)

Larger candidate pools improve CE reranking by giving the model more relevant documents to surface.

| # | Condition | Model | Pool | NDCG@10 | R@10 |
|---|-----------|-------|------|---------|------|
| 4 | **CE Rerank** | **mxbai-rerank-large-v2** | ~89 | **0.6004** | **0.6401** |
| 4 | CE Rerank | mxbai-rerank-base-v2 | ~89 | 0.5241 | 0.5876 |
| 4 | CE Rerank | bge-reranker-v2-m3 | ~89 | 0.5214 | 0.5778 |
| 4 | CE Rerank | bge-reranker-large (v1) | ~89 | 0.4463 | 0.4930 |
| 4 | CE Rerank | bge-reranker-base (v1) | ~89 | 0.3487 | 0.4100 |

### Pool Composition Experiments (TOP_K=50, CE = mxbai-rerank-large-v2)

Wider candidate pools improve CE reranking by increasing pool recall.

| Pool Strategy | Candidates | Pool Recall | NDCG@10 | R@10 |
|---------------|-----------|-------------|---------|------|
| **MV-bge100 + MV-jina100** | ~151 | 0.92 | **0.6208** | **0.6650** |
| MV-bge top-200 | ~200 | 0.89 | 0.6134 | 0.6560 |
| Vec100 + MV-bge100 | ~130 | 0.90 | 0.6086 | 0.6490 |
| Hybrid BM25+Vec (baseline) | ~89 | 0.77 | 0.6004 | 0.6401 |

*Additional rerankers tested: bge-reranker-large (0.4463), bge-reranker-base (0.3487), MiniLM-L-12 (0.2389). jina-reranker-v3 not available on SIE. Results in `autoresearch_results.tsv`.*

---

## Key Findings

1. **Cross-encoder reranking wins**: mxbai-large (0.6208 with dual-MV pool) >> mxbai-base (0.5241) — +57% over dense vector
2. **Pool recall is the bottleneck**: CE scores 0.69 within-pool. Dual-MV pool (0.92 recall) → 0.6208 vs hybrid-50 (0.77 recall) → 0.6004
3. **Two MV models > one**: bge-m3 + jina-colbert-v2 pool (0.6208) beats single bge-m3 pool (0.6134) — model diversity improves recall
4. **Model size matters for CE**: large reranker (+14.5% over base) is the single biggest quality lever
5. **Model generation matters**: bge-reranker-v2-m3 (0.5214) >> bge-reranker-large v1 (0.4463) — newer v2 beats larger v1
6. **jina-colbert-v2 = best cost/quality tradeoff**: 96% of bge-m3 MV quality at 12.5% storage (128d vs 1024d)
7. **Token limit matters more than architecture**: 8192-token models (jina, GTE) >> 512-token models (colbertv2, mxbai-colbert)
6. **Score fusion can't beat cross-encoder**: best MV+vector fusion = 0.4413, CE cross-attention gap is fundamental
7. **Pool optimization**: TOP_K=50, 50/50 hybrid union is optimal. Pool ceiling = 0.77 recall; reranker quality is the bottleneck
8. **RRF hurts** (0.3583 < 0.3962) — BM25 dilutes strong vector signal on this dataset

---

## Reproducibility

```bash
# Full run (all conditions, all models)
uv run python benchmark_ablation.py --gpu l4-spot

# Dry run (validate config)
uv run python benchmark_ablation.py --dry-run

# Selective models
uv run python benchmark_ablation.py --ce-models mxbai-rerank --mv-models bge-m3
```

- All intermediate artifacts cached in `cache/ablation/`
- Re-runs skip completed steps automatically
- Random seed: 42
- RRF k parameter: 60
- Retrieval depth: top-25 (primary), top-50 (reranker sweep)
- Evaluation depth: top-10
- MaxSim scoring: `maxsim-cpu` (mixedbread C library)
