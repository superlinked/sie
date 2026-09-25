# What SIE does in rag-params-finder

rag-params-finder sweeps **embedding × chunking × retrieval** combinations and
ranks them by retrieval scores — before you build a RAG app. SIE is one embedding
(and optional rerank) **provider**, alongside Voyage AI and local sentence-transformers.

**Source of truth:**
[sie-setup.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md) ·
[configuration.md](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/configuration.md).

## SIE primitives

| Primitive | Role in this project |
|---|---|
| `encode` | Open-source embeddings for sweep experiments (`provider: sie`) |
| `score` | Optional SIE reranker (e.g. BGE reranker) when configured |

No LLM generation is required for core sweeps — evaluation stays embedding-centric.

## Models (SIE catalog used here)

| Model | Typical use |
|---|---|
| BGE-M3 | Dense 1024-dim default when SIE is enabled for Tier-1 sweep |
| Stella-v5 | Alternate dense encoder |
| SPLADE-v3 | Sparse retrieval experiments |
| BGE reranker | Optional rerank via SIE |

Exact IDs and YAML knobs live in the project
[`model_registry`](https://github.com/neomatrix369/rag-params-finder/blob/main/server/core/model_registry.py)
and example configs under `configs/mongodb/` and `configs/supabase/`.

## vs Voyage and local MiniLM

| Provider | Needs | Good for |
|---|---|---|
| **SIE** | Gateway or Docker + `SIE_ENABLED` | Open-source SOTA models under one HTTP API |
| **Voyage** | API key | Hosted Voyage embedding families |
| **Local** (`sentence-transformers`) | CPU/GPU on the app host | Offline demos (`all-MiniLM-L6-v2`, etc.) |

You can compare providers across sweeps; SIE does not replace the vector store
(MongoDB Atlas Vector Search or Postgres/pgvector).

## Architecture sketch

```text
Your corpus + questions
        │
        ▼
rag-params-finder server  ──encode/score──►  SIE (remote or :8720)
        │
        ▼
MongoDB / Postgres (vectors + scores)
        │
        ▼
Dashboard :5374
```

## Screenshots

SIE vs local experiment UIs are documented with images in the project
[README screenshots](https://github.com/neomatrix369/rag-params-finder/blob/main/README.md#-screenshots).

## Next

Wire it up: [SIE integration](./sie-integration.md). Problems:
[Troubleshooting](./troubleshooting.md).
