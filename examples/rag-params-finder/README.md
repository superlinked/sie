# Find the best RAG config before you build

> Sweep embeddings × chunking × retrieval on **your** data — then ship the winning config first.

This is an **external project guide**. The runnable app lives in
[neomatrix369/rag-params-finder](https://github.com/neomatrix369/rag-params-finder)
(MIT). This folder is the SIE-facing onboarding surface: short pages here, full
detail in that repo.

**SIE primitives used:** `encode` (embeddings via BGE-M3 / Stella-v5 / SPLADE-v3);
optional `score` (SIE rerank). SIE is **opt-in** — the default stack runs without it.

## Who this is for

| You are… | Start here |
|---|---|
| New to SIE, found this in the gallery | [Getting started](./getting-started.md) → [SIE integration](./sie-integration.md) |
| New to rag-params-finder, want SIE embeddings | Same path — then [What SIE does here](./what-sie-does.md) |

Both audiences share the same **local** starting path (MongoDB stack +
dashboard). SIE is optional afterward: enable a remote gateway, then run one
`example-sie.yaml` sweep.

## Start here

1. [Getting started](./getting-started.md) — clone, prereqs, local Mongo path, dashboard
2. [SIE integration](./sie-integration.md) — env vars, health checks, first SIE sweep
3. [What SIE does here](./what-sie-does.md) — models, encode/score, vs Voyage/local
4. [Troubleshooting](./troubleshooting.md) — short FAQ + deep-links

**Canonical docs in the project:**
[QUICKSTART](https://github.com/neomatrix369/rag-params-finder/blob/main/QUICKSTART.md) ·
[SIE setup](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/user-guide/sie-setup.md) ·
[docs index](https://github.com/neomatrix369/rag-params-finder/blob/main/docs/README.md)

## Ports cheat sheet

| Service | Port | Notes |
|---|---|---|
| API | `8001` | rag-params-finder server |
| Dashboard | `5374` | Experiments UI |
| SIE (self-hosted only) | `8720` | Not started by `./start-services.sh` |

## Attribution

Built and maintained in
[neomatrix369/rag-params-finder](https://github.com/neomatrix369/rag-params-finder)
under the [MIT License](https://github.com/neomatrix369/rag-params-finder/blob/main/LICENSE).
Screenshots and deeper guides live in that repository.
