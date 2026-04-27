# Production RAG Pipeline for Financial Document Search

**An evals-driven recipe.** Every model choice below is backed by a
head-to-head benchmark on 1,854 real queries against 2,942 pages from
six SEC 10-K filings, measured by NDCG@10. No guesswork, no defaults.
The recipe at the top is the one that won the evals; the
[Evidence](#evidence) section lists every other combination tried and
what each scored.

One SIE cluster, seven models, three API calls. No model serving to
manage.

![Pipeline](hero.png)

![Results](results_chart.png)

## The recipe

**Dual multi-vector retrieval, then cross-encoder rerank.**

1. Encode queries and pages with two multi-vector models,
   `BAAI/bge-m3` (1024d) and `jinaai/jina-colbert-v2` (128d), to get
   complementary retrieval pools.
2. Rerank the combined candidate pool with `mixedbread-ai/mxbai-rerank-large-v2`.

On 1,854 queries against 2,942 pages from six bank 10-K filings, this
pipeline hits **NDCG@10 = 0.621 and Recall@10 = 0.665**, 57% better
than a single dense model and 3× better than BM25 alone. Full
benchmarking numbers are in the [Evidence](#evidence) section below.

> *Built by [@NirantK](https://twitter.com/NirantK) for [Superlinked](https://superlinked.com).*

## The full pipeline in code

```python
from sie_sdk import SIEAsyncClient

async with SIEAsyncClient("http://your-sie-endpoint:8080", api_key="SL-...") as sie:
    # Multi-vector encode with two complementary models
    mv_bge = await sie.encode("BAAI/bge-m3", [{"text": "quarterly revenue"}],
                              output_types=["multivector"])

    mv_jina = await sie.encode("jinaai/jina-colbert-v2", [{"text": "quarterly revenue"}],
                               output_types=["multivector"])

    # Union the two pools, rerank with a cross-encoder
    result = await sie.score("mixedbread-ai/mxbai-rerank-large-v2",
                             query={"text": "quarterly revenue"},
                             items=[{"text": "Revenue was $50B..."},
                                    {"text": "The board met on Tuesday..."}])
```

Three model families. One endpoint. No container orchestration.

## Run the full pipeline

```bash
# Install
uv sync

# Validate config (no GPU needed)
uv run python benchmark_ablation.py --dry-run

# Run the production pipeline end-to-end (all 7 models, all 1,854 queries)
uv run python benchmark_ablation.py --gpu l4-spot

# Or run just the winning combination (skip baselines and alternatives)
uv run python benchmark_ablation.py --gpu l4-spot --skip-conditions 1,2,3
```

All expensive operations (encoding, search) cache to `cache/ablation/`. Re-runs skip completed steps. Cross-encoder reranking checkpoints every 100 queries for crash recovery.

## Evidence

We benchmarked six retrieval strategies against the same 1,854 queries
on 2,942 pages from six bank 10-K filings. The recipe at the top came
out first on every metric:

| Strategy | Model | NDCG@10 | Recall@10 | What it shows |
|----------|-------|---------|-----------|---------------|
| **Dual-MV → CE Rerank** | bge-m3+jina-colbert → mxbai-large | **0.621** | **0.665** | The recipe |
| MV pool → CE Rerank | MV-bge200 + mxbai-large | 0.613 | 0.656 | Single MV model + CE, nearly as good |
| CE Rerank | mxbai-rerank-large | 0.600 | 0.640 | +52% over vector; CE alone is strong |
| CE Rerank | mxbai-rerank-base | 0.524 | 0.588 | Smaller reranker, still strong |
| CE Rerank | bge-reranker | 0.521 | 0.578 | Near-identical to mxbai-base |
| MV Direct | bge-m3 (1024d) | 0.435 | 0.482 | No GPU at inference, +10% over vector |
| MV Rerank | jina-colbert-v2 (128d) | 0.431 | 0.494 | 96% of bge-m3 quality at 12.5% storage |
| Vector | bge-m3 dense | 0.396 | 0.438 | Single-model baseline |
| RRF | BM25+Vector | 0.358 | 0.434 | Hybrid hurts here; BM25 dilutes signal |
| BM25 | Turbopuffer FTS | 0.185 | 0.239 | Keyword search alone isn't enough |

See [RESULTS.md](RESULTS.md) for full methodology, all benchmarking conditions, and pool-composition experiments.

## What this shows about SIE

- **Model-agnostic**: swap `bge-m3` for `jina-colbert-v2` with one parameter change
- **Multi-model pipelines**: two encoders plus a cross-encoder rerank in one script, one cluster
- **85+ models available**: not locked into one vendor's embeddings
- **Async-native**: fire hundreds of concurrent requests, SIE handles batching and GPU scheduling

## API Keys Required

| Service | What for | Get one at |
|---------|---------|------------|
| **SIE** | Encoding, scoring, multi-vector | Self-hosted ([deploy guide](https://github.com/superlinked/sie)) or contact team |
| **Turbopuffer** | BM25 + vector search index (free tier covers this dataset) | [turbopuffer.com](https://turbopuffer.com) |
| **HuggingFace** | Dataset download (free, cached) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

`SIE_API_KEY` is optional. Leave it unset for a local or otherwise
unauthenticated SIE deployment; set it only when your managed/auth-enabled
cluster requires one.

Create `.env` in this directory:

```
SIE_BASE_URL=http://your-sie-endpoint:8080
# Optional: only needed for managed/auth-enabled SIE clusters.
SIE_API_KEY=
TURBOPUFFER_API_KEY=tpuf_...
```

## Recipes by constraint

- **Best quality overall**: the dual-MV then CE rerank recipe above (NDCG=0.621).
- **No GPU at inference time**: multi-vector direct with `bge-m3` (NDCG=0.44). Pre-encode offline, search with MaxSim on CPU.
- **Best cost/quality**: `jina-colbert-v2` multi-vector (NDCG=0.43). 128-dim vectors equal 8× less storage than `bge-m3` multi-vector, nearly identical quality.

## Dependencies

- [SIE SDK](https://github.com/superlinked/sie): async encode, score, extract
- [Turbopuffer](https://turbopuffer.com): BM25 + vector search
- [maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu): optimized ColBERT MaxSim scoring
- [datasets](https://huggingface.co/docs/datasets): HuggingFace dataset loading (cached locally)
