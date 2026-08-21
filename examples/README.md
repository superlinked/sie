# Examples

A project gallery of full end-to-end applications built with SIE. Each project lives in its own subdirectory. Clone it, run it, learn from it.

New to SIE? Start with the **[quickstart notebook](./quickstart.ipynb)** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/superlinked/sie/blob/main/examples/quickstart.ipynb): encode, score, and extract in 5 minutes, then pick a project below.

## Gallery

Use this table to pick the right starting point. "Runnable" means the
example has code, sample data or data-fetch instructions, and a documented
local path. "Advanced" examples may require a custom SIE image or third-party
service keys. "External project guide" means docs-only onboarding that deep-links
to a separately maintained repository (clone and run there).

| Example | Best for | SIE primitives | Setup | Status |
|---|---|---|---|---|
| [Self-hosted product search in 5 min](./ecommerce-product-search) | Showing the fastest local product-search path with extraction, embeddings, and reranking | `extract`, `encode`, `score` | Local SIE Docker image, Python or TypeScript app | Runnable |
| [Find the best retrieval strategy for your RAG](./retrieval-ablation) | Picking a production RAG retrieval pipeline by evals on real financial documents | `encode`, `score` | SIE endpoint, Turbopuffer key, optional SIE API key for auth-enabled clusters | Runnable benchmark |
| [Rank exact primary-source passages](./rerank) | Testing a reranker on verbatim SEC, CMS, NTSB, and Supreme Court excerpts | `score` | SIE endpoint with Qwen3 Reranker; standalone `uv` project | Runnable verified example |
| [Extract custom entities from primary sources](./named-entity-extraction) | Changing zero-shot labels across financial, healthcare, rail-safety, and legal text | `extract` | SIE endpoint with GLiNER; standalone `uv` project | Runnable verified example |
| [Search licensed images with text](./multimodal-search) | Recomputing a six-image hard-negative ranking from full SigLIP vectors | `encode` | SIE endpoint with SigLIP; standalone `uv` project | Runnable verified example |
| [Find SOTA embedding models by MTEB task](./sie-hugging-face-mteb-semantic-search) | Searching ~14K HF embedding models ranked by task-specific MTEB scores | `encode`, `score` | Backend seed script plus Vite frontend; falls back without a live SIE endpoint | Runnable |
| [Private fine-tuned compliance RAG](./regulatory-rag) | Hot-loading a domain LoRA encoder and a custom token-pruning adapter on SIE | `encode`, `score`, `extract` | Custom SIE Docker image, GPU recommended | Advanced runnable example |
| [Build a multimodal wine recommender with OCR](./wine-recommender) | Combining preference-based retrieval with OCR-driven label detection in one UI | `encode`, `score`, `extract` | Docker Compose app plus local SIE endpoint; API key optional for unauthenticated SIE | Runnable demo |
| [Build a multi-modal product classifier with embeddings](./taxonomy-classification) | Evaluating text, image, NLI, and reranking approaches for hierarchical product taxonomy classification | `encode`, `score`, `generate` | SIE endpoint, Shopify dataset prep via `uv run` scripts, standalone `uv` project | Runnable evaluation example |
| [Swap an OCR model with one identifier change](./document-ocr) | Driving recognition (VLM-OCR), structured extraction (Donut), and zero-shot NER (GLiNER) through the same `extract` call by swapping the model ID | `extract` | Docker Compose plus Node UI, no API key required, hosted version on [Hugging Face Spaces](https://huggingface.co/spaces/superlinked/document-ocr) | Runnable demo |
| [A Stripe Link checkout with an SIE fraud-risk gate](./stripe-link-fraud) | Wiring all three SIE primitives into a pre-authorization fraud-risk gate that runs in the same round-trip as the Stripe PaymentIntent | `extract`, `encode`, `score` | Docker Compose plus Node UI; Stripe test-mode keys optional (runs in mock mode without them) | Runnable demo |
| [Vision-first document RAG](./vision-doc-rag) | Retrieving and answering questions over a multi-tenant page corpus by looking at page images (including scanned drawings) with OCR kept out of the score path | `encode`, `chat/completions`, `score` (optional) | GPU SIE deployment required: ColQwen2.5 retriever + Qwen3.5-4B answer model (runs on the generation bundle) | Runnable demo |
| [Multi-model contract review with the OpenAI Agents SDK](./contract-review-agent) | Running an OpenAI Agents SDK agent whose every model call (triage, orchestration, vision, OCR, embeddings, rerank, entity extraction, text-to-SQL, reasoning, and a safety guardrail) is served by one SIE cluster, each step on the right catalog model, with per-model observability | `generate`, `encode`, `score`, `extract` | GPU SIE deployment required; standalone `uv` project; real contracts fetched from CUAD (CC BY 4.0) | Runnable demo |
| [Turn difficult PDFs into Markdown](./document-to-markdown) | Preserving tables, reading order, headings, and form labels across real financial, academic, and government PDFs | `extract` | SIE endpoint with `docling`; standalone `uv` project; source PDFs fetched at run time | Runnable evaluation example |
| [Review a published flood-insurance appeal](./insurance-claims-agent) | Separating FEMA's covered stone-removal scope from excluded barge, handling, disposal, and yard costs | `extract`, `score`, `generate` | GPU SIE deployment; standalone `uv` project; bundled public FEMA appeal and policy | Runnable agent example |
| [Trace a restated filing figure](./financial-filing-agent) | Following one reported figure through an original filing, corrective notice, and restatement while preserving source status | `extract`, `encode`, `score` | SIE endpoint; standalone `uv` project; public SEC facts and saved verified evidence | Runnable agent example |
| [Reproduce CMS's L1851 documentation finding](./prior-authorization-review-agent) | Tracing a published six-month requirement against a seven-month face-to-face encounter and CMS's recoupment result | `extract`, `encode`, `score` | SIE endpoint; standalone `uv` project; exact CMS published example | Runnable agent example |
| [Reconstruct a bearing failure](./maintenance-triage-agent) | Turning the NTSB's three East Palestine detector readings into a cited temperature and alert sequence without adding a new causal claim | `extract`, `encode`, `score` | SIE endpoint; standalone `uv` project; exact NTSB illustrated report spread | Runnable agent example |
| [Make a shelf gap auditable](./retail-shelf-audit) | Detecting one empty facing, deriving its notice and shelf-label crops by geometry, then preserving OCR evidence | `extract` | GPU SIE deployment; standalone `uv` project; CC0 supermarket shelf image and recorded direct-checkpoint evidence included | Runnable evaluation example |
| [A behavioural gate that catches hijacked AI agents by their actions, not their credentials](./agent-action-monitor) | Judging a proposed AI agent action against that agent's own learned baseline in real time, before it reaches a downstream system | `encode`, `score`, `extract` | Docker Compose (gate + self-hosted SIE + n8n + mock downstream), no API key required | Runnable demo |
| [Find the best RAG config before you build](./rag-params-finder) | Sweeping embeddings × chunking × retrieval on your data before building a RAG app | `encode`, `score` (optional rerank) | External repo; MongoDB local or Atlas/Postgres; SIE gateway or Docker | External project guide |

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

### Review workflow

Maintainers apply the `coderabbit-direct` label to eligible PRs that change content under `examples/**` or the root `README.md`. The label opts the PR into CodeRabbit review and allows CodeRabbit to formally approve it once review comments are resolved and required checks pass.

Projects can be anything: a search engine, a RAG pipeline, a benchmark, a migration guide, a CLI tool. If it uses SIE, it belongs here.

## Links

- [SIE overview](../README.md)
- [API reference](https://superlinked.com/docs/reference/sdk)
- [Deployment guide](https://superlinked.com/docs/deployment/docker)
- [All models](https://superlinked.com/models)
