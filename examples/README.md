# Examples

A project gallery of full end-to-end applications built with SIE. Each project lives in its own subdirectory. Clone it, run it, learn from it.

## Gallery

Use this table to pick the right starting point. "Runnable" means the
example has code, sample data or data-fetch instructions, and a documented
local path. "Advanced" examples may require a custom SIE image or third-party
service keys.

| Example | Best for | SIE primitives | Setup | Status |
|---|---|---|---|---|
| [E-Commerce Product Search](./ecommerce-product-search) | Showing the fastest local product-search path with extraction, embeddings, and reranking | `extract`, `encode`, `score` | Local SIE Docker image, Python or TypeScript app | Runnable |
| [Retrieval Ablation](./retrieval-ablation) | Demonstrating eval-driven model selection for production RAG retrieval | `encode`, `score` | SIE endpoint, Turbopuffer key, optional SIE API key for auth-enabled clusters | Runnable benchmark |
| [Hugging Face MTEB Semantic Search](./sie-hugging-face-mteb-semantic-search) | Building a semantic-search app over model metadata and benchmark results | `encode`, `score` | Backend seed script plus Vite frontend; falls back without a live SIE endpoint | Runnable |
| [Regulatory RAG](./regulatory-rag) | Hosting custom adapters and LoRA profiles in SIE | `encode`, `score`, `extract` | Custom SIE Docker image, GPU recommended | Advanced runnable example |
| [Wine Recommender](./wine-recommender) | Combining recommendation reranking with OCR-style extraction in a UI | `encode`, `score`, `extract` | Docker Compose app plus local SIE endpoint; API key optional for unauthenticated SIE | Runnable demo |
| [Taxonomy Classification](./taxonomy-classification) | Evaluating text, image, NLI, and reranking approaches for hierarchical product taxonomy classification | `extract`, `encode`, `score` | SIE endpoint, Shopify dataset prep via `uv run` scripts, standalone `uv` project | Runnable evaluation example |

For docs publishing, lead with the quickest runnable demos, then use the
benchmark and evaluation examples for deeper technical users.

## Submit your project

We welcome contributions. To add your project to the gallery:

1. **Create a subdirectory** with a short, descriptive name (e.g. `wikipedia-search/`, `pdf-rag/`)
2. **Include a README** that covers:
   - What the project does
   - How to run it (`docker compose up`, a script, etc.)
   - Which SIE features it uses (encode, score, extract, cluster, etc.)
3. **Keep it self-contained** - include a `requirements.txt` or `package.json`, a docker-compose if needed, and sample data or instructions to fetch it
4. **Open a PR** against `main`

Projects can be anything: a search engine, a RAG pipeline, a benchmark, a migration guide, a CLI tool. If it uses SIE, it belongs here.

## Links

- [SIE overview](../README.md)
- [API reference](https://superlinked.com/docs/reference/sdk)
- [Deployment guide](https://superlinked.com/docs/deployment/docker)
- [All models](https://superlinked.com/models)
