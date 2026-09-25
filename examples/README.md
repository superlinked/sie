# Examples

A project gallery of full end-to-end applications built with SIE. Most entries
are self-contained under `examples/<name>/` — clone this repo, run them locally,
and learn from them. Rows marked **External project guide** are docs-only
landings that deep-link to a separately maintained repository (clone and run
there).

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
| [Rank exact primary-source passages](./rerank) | Testing a reranker on verbatim SEC, CMS, NTSB, and Supreme Court excerpts, where the closest wrong passage answers half the question | `score` | SIE endpoint with Qwen3 Reranker 4B; `fetch.py` pulls the recorded run, `score.py` reproduces the scores with no key | Runnable recorded example |
| [Extract custom entities from primary sources](./named-entity-extraction) | Changing zero-shot labels across financial, healthcare, rail-safety and legal text, with the two labels that came back empty left visible | `extract` | SIE endpoint with GLiNER multi v2.1; `fetch.py` pulls the recorded run, `score.py` reproduces the counts with no key | Runnable recorded example |
| [Answer a question from a passage, and quote the sentence you used](./chat) | Checking a generated answer for its citation, its length and whether it declines a question the passage cannot answer | `chat/completions` | SIE Cloud with Qwen3.8 27B; `fetch.py` pulls the recorded run, `score.py` reproduces the numbers with no key | Runnable recorded example |
| [Write public status updates from internal incident reports](./incident-status-updates) | Checking a generated status update for leaked hosts, tickets, staff names and the wrong outage window | `chat/completions` | SIE Cloud with Qwen3.8 27B; `fetch.py` pulls the recorded run, `score.py` reproduces the number with no key | Runnable recorded example |
| [Build a citable graph from filing text](./knowledge-graph) | Turning SEC, NHTSA and FDA paragraphs into entities and relations with offsets, and seeing where the relations go wrong | `extract` | SIE Cloud with GLiNER2; `fetch.py` pulls the recorded run, `score.py` reproduces the counts with no key | Runnable recorded example |
| [Find the passage that answers a question](./search) | Ranking 253 regulation and documentation passages full of near misses, with query-side encoding | `encode` | SIE Cloud with Arctic Embed L v2.0; `fetch.py` pulls the recorded vectors, `score.py` ranks them with no key | Runnable recorded example |
| [Pick the passage that meets every condition](./multi-vector) | Separating same-page near-duplicates with ColBERT late interaction, scored token by token | `encode` | SIE Cloud with GTE-ModernColBERT-v1; `fetch.py` pulls the recorded token vectors, `score.py` scores them with no key | Runnable recorded example |
| [Fill a JSON Schema from a messy document](./structured-output) | Reading NHTSA complaints, SEC officer filings and GSA listings into schema-valid JSON, with the two wrong fields left visible | `chat/completions` | SIE Cloud with Qwen3.8 27B; `fetch.py` pulls the recorded run, `score.py` reproduces the number with no key | Runnable recorded example |
| [Route a request to the action it asks for](./classify) | Scoring held-out SNIPS requests against seven action labels, and showing what happens when the labels overlap | `extract` | SIE Cloud with GLiClass; `fetch.py` pulls the recorded run, `score.py` reproduces the number with no key | Runnable recorded example |
| [Answer several triage questions about one record, each with a probability](./typed-decisions) | Asking CVE descriptions typed pick-one and yes-or-no questions against NVD analyst gold, with a fast, a smart and an LLM lane, and the questions no model passed reported rather than shown | `extract`, `chat/completions` | Self-hosted SIE with GLiFormer, GLiClass, Laya, GLiNER2 and Qwen3 4B; `fetch.py` pulls the recorded run, `score.py` reproduces the figures with no key | Runnable recorded example |
| [Check text for a planted instruction](./guardrails) | Gating an agent on one label over twelve inputs from BIPIA, InjecAgent, LLMail-Inject, AgentDojo and XSTest, misses included | `extract`, `chat_completions` | SIE Cloud with GLiGuard and Granite Guardian; `fetch.py` pulls the recorded run, `score.py` reproduces both models' numbers with no key | Runnable recorded example |
| [Expand a query into matchable terms](./sparse) | Comparing SPLADE, which adds terms the text never used, with bge-m3 sparse, which does not | `encode` | SIE Cloud with SPLADE++ and bge-m3; `fetch.py` pulls the recorded vectors, `score.py` reproduces the figures with no key | Runnable recorded example |
| [Get the exact character offsets to mask](./redact) | Naming your own ID types, measuring against published gold PII spans, and showing what a 384-word window costs | `extract` | SIE Cloud with GLiNER multi PII; `fetch.py` pulls the recorded run, `score.py` reproduces the figures with no key | Runnable recorded example |
| [Read the numbers off a dashboard that has no API](./screenshot-mining) | Reading twelve Superset, Argo CD, Airflow, Kubernetes, GitLab and Jaeger screens into schema-valid JSON | `generate` | SIE Cloud with Qwen3.8 27B; `fetch.py` pulls the recorded run and the screenshots, `score.py` reproduces 328 of 335 with no key | Runnable recorded example |
| [Read a scanned form into typed fields, then check the answer](./doc-field-extraction) | Extracting typed fields from FAA, OSHA, NIST, USPS and invoice pages under a strict JSON schema, then measuring whether a second call that never sees the page is worth making | `generate`, `chat/completions` | SIE Cloud with Qwen3.8 27B; `fetch.py` pulls the recorded run and the page images, `score.py` reproduces 180 of 223 and 204 after the second call with no key | Runnable recorded example |
| [Ask a question about an image](./caption-vqa) | Putting one specific question to equipment photographs, hazmat placards, wiring drawings and analogue dials | `generate` | SIE Cloud with Qwen3.8 27B; `fetch.py` pulls the recorded run and the images, `score.py` reproduces 10 of 12 with no key | Runnable recorded example |
| [Locate every object a prompt can name in a photo](./detect) | Grounding open-vocabulary phrases to boxes on nine photos, with the two boxes it puts on the wrong object kept visible | `extract` | SIE Cloud with Grounding DINO; `fetch.py` pulls the recorded run, `score.py` reproduces the number with no key | Runnable recorded example |
| [Sort product photos against labels you write yourself](./image-classify) | Scoring damaged and whole parts against your own label wording, and the cheaper model that gets four of sixteen wrong | `score`, `encode` | SIE Cloud with Qwen3 VL Reranker and SigLIP 2; `fetch.py` pulls the recorded run, `score.py` reproduces the number with no key | Runnable recorded example |
| [Find a spoken part number in a noisy recording](./speech-to-text) | Transcribing twelve clips and searching them for the doses, figures and statute numbers that matter, the five misses kept | `extract` | SIE Cloud with Whisper large v3 turbo; `fetch.py` pulls the recorded run, `score.py` reproduces the word error rate with no key | Runnable recorded example |
| [Find the image that matches a whole request](./image-search) | Ranking six product photos against one written query, where a colour-only and a category-only match sit just behind the right answer | `encode` | SIE Cloud with SigLIP so400m; `fetch.py` pulls the recorded run, `score.py` reproduces the ranking with no key | Runnable recorded example |
| [Search document pages without flattening them to text](./visual-document-search) | Ranking 712 report pages by what they look like, against a text baseline that buries the answer | `encode` | Self-hosted SIE with ColPali v1.3; `fetch.py` pulls the recorded run, `score.py` reproduces both rankings with no key | Runnable recorded example |
| [Search licensed images with text](./multimodal-search) | Recomputing a six-image hard-negative ranking from full SigLIP vectors | `encode` | SIE endpoint with SigLIP; standalone `uv` project | Runnable verified example |
| [Find SOTA embedding models by MTEB task](./sie-hugging-face-mteb-semantic-search) | Searching ~14K HF embedding models ranked by task-specific MTEB scores | `encode`, `score` | Backend seed script plus Vite frontend; falls back without a live SIE endpoint | Runnable |
| [Private fine-tuned compliance RAG](./regulatory-rag) | Hot-loading a domain LoRA encoder and a custom token-pruning adapter on SIE | `encode`, `score`, `extract` | Custom SIE Docker image, GPU recommended | Advanced runnable example |
| [Build a multimodal wine recommender with OCR](./wine-recommender) | Combining preference-based retrieval with OCR-driven label detection in one UI | `encode`, `score`, `extract` | Docker Compose app plus local SIE endpoint; API key optional for unauthenticated SIE | Runnable demo |
| [Build a multi-modal product classifier with embeddings](./taxonomy-classification) | Evaluating text, image, NLI, and reranking approaches for hierarchical product taxonomy classification | `encode`, `score`, `generate` | SIE endpoint, Shopify dataset prep via `uv run` scripts, standalone `uv` project | Runnable evaluation example |
| [Read a page image, then read the Markdown](./ocr-two-stage) | Measuring an OCR stage and a schema-filling stage separately, so a wrong field can be traced to the stage that lost it | `extract`, `chat/completions` | SIE Cloud with LightOnOCR-2-1B and two Qwen3 models; `fetch.py` pulls the recorded run, `score.py` reproduces the figures with no key | Runnable recorded example |
| [Swap an OCR model with one identifier change](./document-ocr) | Driving recognition (VLM-OCR), structured extraction (Donut), and zero-shot NER (GLiNER) through the same `extract` call by swapping the model ID | `extract` | Docker Compose plus Node UI, no API key required, hosted version on [Hugging Face Spaces](https://huggingface.co/spaces/superlinked/document-ocr) | Runnable demo |
| [A Stripe Link checkout with an SIE fraud-risk gate](./stripe-link-fraud) | Wiring all three SIE primitives into a pre-authorization fraud-risk gate that runs in the same round-trip as the Stripe PaymentIntent | `extract`, `encode`, `score` | Docker Compose plus Node UI; Stripe test-mode keys optional (runs in mock mode without them) | Runnable demo |
| [Vision-first document RAG](./vision-doc-rag) | Retrieving and answering questions over a multi-tenant page corpus by looking at page images (including scanned drawings) with OCR kept out of the score path | `encode`, `chat/completions`, `score` (optional) | GPU SIE deployment required: ColQwen2.5 retriever + Qwen3.5-4B answer model (runs on the generation bundle) | Runnable demo |
| [Multi-model contract review with the OpenAI Agents SDK](./contract-review-agent) | Running an OpenAI Agents SDK agent whose every model call (triage, orchestration, vision, OCR, embeddings, rerank, entity extraction, text-to-SQL, reasoning, and a safety guardrail) is served by one SIE cluster, each step on the right catalog model, with per-model observability | `generate`, `encode`, `score`, `extract` | GPU SIE deployment required; standalone `uv` project; real contracts fetched from CUAD (CC BY 4.0) | Runnable demo |
| [Turn difficult PDFs into Markdown](./document-to-markdown) | Preserving tables, reading order, headings, and form labels across real financial, academic, and government PDFs | `extract` | SIE endpoint with `docling`; `fetch.py` pulls the recorded run, `verify-run` checks 61 digests and relations offline; source PDFs fetched at run time | Runnable evaluation example |
| [Review a published flood-insurance appeal](./insurance-claims-agent) | Separating FEMA's covered stone-removal scope from excluded barge, handling, disposal, and yard costs | `extract`, `score`, `generate` | GPU SIE deployment; standalone `uv` project; bundled public FEMA appeal and policy | Runnable agent example |
| [Trace a restated filing figure](./financial-filing-agent) | Following one reported figure through an original filing, corrective notice, and restatement while preserving source status | `extract`, `encode`, `score` | SIE endpoint; standalone `uv` project; public SEC facts and saved verified evidence | Runnable agent example |
| [Reproduce CMS's L1851 documentation finding](./prior-authorization-review-agent) | Tracing a published six-month requirement against a seven-month face-to-face encounter and CMS's recoupment result | `extract`, `encode`, `score` | SIE endpoint; standalone `uv` project; exact CMS published example | Runnable agent example |
| [Reconstruct a bearing failure](./maintenance-triage-agent) | Turning the NTSB's three East Palestine detector readings into a cited temperature and alert sequence without adding a new causal claim | `extract`, `encode`, `score` | SIE endpoint; standalone `uv` project; exact NTSB illustrated report spread | Runnable agent example |
| [Make a shelf gap auditable](./retail-shelf-audit) | Detecting one empty facing, deriving its notice and shelf-label crops by geometry, then preserving OCR evidence | `extract` | GPU SIE deployment; standalone `uv` project; CC0 supermarket shelf image and recorded direct-checkpoint evidence included | Runnable evaluation example |
| [Turn threat reports into cited ATT&CK mapping suggestions](./threat-report-attck-mapper) | Mapping full reports against active ATT&CK 19.2, with a separate pinned AnnoCTR linking benchmark and analyst review for every suggestion | `generate`, `extract`, `encode`, `score` | GPU SIE deployment; standalone `uv` project; pinned MITRE ATT&CK and AnnoCTR sources | Runnable agent benchmark |
| [A behavioural gate that catches hijacked AI agents by their actions, not their credentials](./agent-action-monitor) | Judging a proposed AI agent action against that agent's own learned baseline in real time, before it reaches a downstream system | `encode`, `score`, `extract` | Docker Compose (gate + self-hosted SIE + n8n + mock downstream), no API key required | Runnable demo |
| [Find the best RAG config before you build](./rag-params-finder) | Sweeping embeddings × chunking × retrieval on your data before building a RAG app | `encode`, `score` (optional rerank) | External repo; MongoDB local or Atlas/Postgres; SIE gateway or Docker | External project guide |
| [Measure whether translation or paraphrase removes a text watermark](./watermark-robustness) | Watermarking text with a key you control, then measuring how much of the signal survives an Arabic round trip versus a paraphrase; all recorded runs readable offline | `generate`, `extract` | SIE endpoint for new transformations (hosted or local MADLAD path); standalone `uv` project; saved experiment data included | Runnable evaluation example |

For docs publishing, lead with the quickest runnable demos, then use the
benchmark and evaluation examples for deeper technical users.

## Submit your project

We welcome contributions. To add your project to the gallery:

### Runnable examples (default)

1. **Create a subdirectory** with a short, descriptive name (e.g. `wikipedia-search/`, `pdf-rag/`)
2. **Include a README** that covers:
   - What the project does
   - How to run it (`docker compose up`, a script, etc.)
   - Which SIE features it uses (encode, score, extract, cluster, etc.)
3. **Keep it self-contained** — include a `requirements.txt` or `package.json`, a docker-compose if needed, and sample data or instructions to fetch it
4. **Open a PR** against `main`

### External project guides

Use this path only when vendoring a runnable copy is impractical (large multi-service
apps). Ship a thin `examples/<name>/` landing (README + short sibling pages) that
deep-links to the external repo’s QUICKSTART/SIE setup, set Status to
**External project guide**, and do **not** require in-tree `requirements.txt` /
compose / sample data. See `examples/rag-params-finder/` for the shape.

### Recorded evidence

Examples that reproduce a number published on a task page keep their code here
and their evidence in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
one folder per task holding `inputs/`, `calls.json` and `manifest.json`. Each
example's `fetch.py` downloads that folder at a pinned dataset revision, never
`main`, so a later upload cannot change what the example scores.

The trade is deliberate: a reader cannot verify by cloning alone, and in return
nobody needs an API key or any inference spend to re-derive a published number.

### Review workflow

Maintainers apply the `coderabbit-direct` label to eligible PRs that change content under `examples/**` or the root `README.md`. The label opts the PR into CodeRabbit review and allows CodeRabbit to formally approve it once review comments are resolved and required checks pass.

Projects can be anything: a search engine, a RAG pipeline, a benchmark, a migration guide, a CLI tool. If it uses SIE, it belongs here.

## Links

- [SIE overview](../README.md)
- [API reference](https://superlinked.com/docs/reference/sdk)
- [Deployment guide](https://superlinked.com/docs/deployment/docker)
- [All models](https://superlinked.com/models)
