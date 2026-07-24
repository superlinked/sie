# Examples

A project gallery of full end-to-end applications built with SIE. Each project lives in its own subdirectory. Clone it, run it, learn from it.

New to SIE? Start with the **[quickstart notebook](./quickstart.ipynb)** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/superlinked/sie/blob/main/examples/quickstart.ipynb): encode, score, and extract in 5 minutes, then pick a project below.

## Gallery

Use this table to pick the right starting point. "Runnable" means the
example has code, sample data or data-fetch instructions, and a documented
local path. "Advanced" examples may require a custom SIE image or third-party
service keys.

| Example | Best for | SIE primitives | Setup | Status |
|---|---|---|---|---|
| [Self-hosted product search in 5 min](./ecommerce-product-search) | Showing the fastest local product-search path with extraction, embeddings, and reranking | `extract`, `encode`, `score` | Local SIE Docker image, Python or TypeScript app | Runnable |
| [Find the best retrieval strategy for your RAG](./retrieval-ablation) | Picking a production RAG retrieval pipeline by evals on real financial documents | `encode`, `score` | SIE endpoint, Turbopuffer key, optional SIE API key for auth-enabled clusters | Runnable benchmark |
| [Find SOTA embedding models by MTEB task](./sie-hugging-face-mteb-semantic-search) | Searching ~14K HF embedding models ranked by task-specific MTEB scores | `encode`, `score` | Backend seed script plus Vite frontend; falls back without a live SIE endpoint | Runnable |
| [Private fine-tuned compliance RAG](./regulatory-rag) | Hot-loading a domain LoRA encoder and a custom token-pruning adapter on SIE | `encode`, `score`, `extract` | Custom SIE Docker image, GPU recommended | Advanced runnable example |
| [Build a multimodal wine recommender with OCR](./wine-recommender) | Combining preference-based retrieval with OCR-driven label detection in one UI | `encode`, `score`, `extract` | Docker Compose app plus local SIE endpoint; API key optional for unauthenticated SIE | Runnable demo |
| [Build a multi-modal product classifier with embeddings](./taxonomy-classification) | Evaluating text, image, NLI, and reranking approaches for hierarchical product taxonomy classification | `extract`, `encode`, `score` | SIE endpoint, Shopify dataset prep via `uv run` scripts, standalone `uv` project | Runnable evaluation example |
| [Swap an OCR model with one identifier change](./document-ocr) | Driving recognition (VLM-OCR), structured extraction (Donut), and zero-shot NER (GLiNER) through the same `extract` call by swapping the model ID | `extract` | Docker Compose plus Node UI, no API key required, hosted version on [Hugging Face Spaces](https://huggingface.co/spaces/superlinked/document-ocr) | Runnable demo |
| [A Stripe Link checkout with an SIE fraud-risk gate](./stripe-link-fraud) | Wiring all three SIE primitives into a pre-authorization fraud-risk gate that runs in the same round-trip as the Stripe PaymentIntent | `extract`, `encode`, `score` | Docker Compose plus Node UI; Stripe test-mode keys optional (runs in mock mode without them) | Runnable demo |
| [Vision-first document RAG](./vision-doc-rag) | Retrieving and answering questions over a multi-tenant page corpus by looking at page images (including scanned drawings) with OCR kept out of the score path | `encode`, `chat/completions`, `score` (optional) | GPU SIE deployment required: ColQwen2.5 retriever + Qwen3.5-4B answer model (runs on the generation bundle) | Runnable demo |
| [Multi-model contract review with the OpenAI Agents SDK](./contract-review-agent) | Running an OpenAI Agents SDK agent whose every model call (triage, orchestration, vision, OCR, embeddings, rerank, entity extraction, text-to-SQL, reasoning, and a safety guardrail) is served by one SIE cluster, each step on the right catalog model, with per-model observability | `chat/completions`, `encode`, `score`, `extract` | GPU SIE deployment required; standalone `uv` project; real contracts fetched from CUAD (CC BY 4.0) | Runnable demo |
| [Turn difficult PDFs into Markdown](./document-to-markdown) | Preserving tables, reading order, headings, and form labels across real financial, academic, and government PDFs | `extract` | SIE endpoint with `docling`; standalone `uv` project; source PDFs fetched at run time | Runnable evaluation example |
| [Review a flood claim packet](./insurance-claims-agent) | Finding an unsigned form and a $400 evidence mismatch across a policy, estimate, inventory, and damage photograph | `extract`, `score`, `chat/completions` | GPU SIE deployment; standalone `uv` project; public FEMA documents and public-domain photograph fetched at run time | Runnable agent example |

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
