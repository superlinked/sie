# Vision-first document RAG

> ABI course: start with the three verified, redistributable fixtures and
> release gates in [`../abi-course-deep`](../abi-course-deep). The model IDs
> below are example configuration until the managed course catalog is frozen
> and live-validated.

Retrieve by image, answer by image. The whole pipeline is Qwen-family:
ColQwen2.5 reads each PDF page as a picture and ranks pages via late
interaction; Qwen3.5-4B (a vision LLM) reads the winning page image and
produces the textual answer. OCR never enters the score path, so schematics,
pinout diagrams, architecture slides, charts, and scanned engineering drawings
still drive ranking — and because the answer model reads the image too, a page
never needs a text layer at all. Everything runs on one SIE endpoint.

Each page also carries a `client` tag, so the same corpus serves multiple
tenants from one index. Queries scoped to `embedded-lab` cannot retrieve
`ops-eng` or `aerospace` pages.

## Corpus

The demo fetches a small public PDF batch on demand and renders selected pages
to PNGs. The page selections are deliberately capped so local ingest stays
fast while still indexing visually rich pages.

| Tenant | Sources | Visual signal |
|---|---|---|
| `embedded-lab` | Raspberry Pi Pico datasheet, Arduino UNO R3 datasheet, Arduino UNO R3 schematic | Pinout diagrams, board diagrams, circuit schematics |
| `ops-eng` | PostgreSQL 18 manual (config reference), CNCF Kubernetes / cloud-native architecture material | Architecture diagrams, operational flows, dense config tables |
| `aerospace` | NASA SP-8115 / SP-8120 rocket nozzle design reports | Scanned engineering drawings and nozzle cross-sections — pages with **no text layer**, so only visual retrieval finds them |

Generated files are ignored:

```text
data/pdfs/                # downloaded PDFs
data/pdfs_manifest.json   # source manifest from fetch_pdfs.py
data/pages/               # rendered PNG pages
data/pages_manifest.json  # page-level metadata from render_pages.py
data/metadata.json        # index metadata from ingest.py
data/multivectors.npz     # page multivectors from ingest.py
```

## SIE features used

- `encode` - `vidore/colqwen2.5-v0.2` on page images at ingest and on query
  text at search time. Output is a `[tokens, 128]` multivector. Late
  interaction (`sie_sdk.scoring.maxsim`) is the first-stage ranking signal.
- chat/completions - `Qwen/Qwen3.5-4B`, a vision+text LLM, reads the winning
  page image plus the question and returns a short answer. It runs over SIE's
  OpenAI-compatible `/v1/chat/completions` endpoint (the SIE SDK's `generate()`
  is text-only); see `python/search.py:_vqa_answer`. Because it reads the page
  as an image, scanned pages with no text layer still get answered.
- `score` optional - `Qwen/Qwen3-VL-Reranker-2B` second-stage rerank over
  `(query text, page image)`, keeping reranking in the same visual modality as
  retrieval. Off by default; flip `search.visual_rerank: true` in `config.yaml`
  to enable it on a deployment that serves the visual reranker (multi-image
  rerank also needs the gateway payload store configured). If the stack can't
  serve it, search falls back to the MaxSim ranking.

## Run it

You need Python 3.12 and a **GPU-backed SIE deployment**. The `Qwen/Qwen3.5-4B`
answer model runs on SIE's generation bundle, which is CUDA-only, so the
`latest-cpu-default` image can't serve it — use a local CUDA image or point
`SIE_CLUSTER_URL` / `SIE_API_KEY` at a managed GPU cluster.

```bash
# 1. SIE on a local NVIDIA GPU, or point SIE_CLUSTER_URL / SIE_API_KEY at a
#    managed GPU cluster.
docker run --gpus all -p 8080:8080 ghcr.io/superlinked/sie-server:latest-cuda12-default

# 2. Fetch public PDFs and render selected pages to PNG.
cd examples/vision-doc-rag
pip install -r python/requirements.txt
python data/fetch_pdfs.py
python data/render_pages.py

# 3. Encode every rendered page with ColQwen2.5 and save the multivectors.
python python/ingest.py

# 4a. CLI demo.
python python/search.py

# 4b. Or start the UI.
uvicorn --app-dir python server:app --port 8888
open http://localhost:8888
```

`render_pages.py` uses `pdf2image` when Poppler is available. If Poppler is
not installed, it falls back to PyMuPDF, which is installed from
`python/requirements.txt`.

First run on a cold cluster pays a one-time model load. ColQwen2.5 and the
Qwen3.5-4B answer model are several GB each, so expect a wait before the warm
path kicks in. The answer model runs on the generation bundle, which can scale
from zero — the first question may take a few minutes to provision while later
ones are fast; `_vqa_answer` retries the "still loading" responses under
`cluster.provision_timeout_s`. To skip that wait on a managed cluster, keep the
generation bundle warm — give it `minReplicas: 1` in your SIE deployment so
Qwen3.5-4B stays resident between queries.

### Managed cluster

```bash
export SIE_CLUSTER_URL="https://your-cluster-host:8080"
export SIE_API_KEY="SL-..."
```

The defaults in `config.yaml` point at `http://localhost:8080`. Set
`cluster.gpu` to a profile name like `l4-spot` if the cluster needs an
explicit GPU class.

## Try these queries

Each row names the page the demo should retrieve — and, for value lookups, the
answer the vision model reads off it — so you can spot regressions at a glance.
These are the runs that back the CLI demo in `python/search.py`.

### Visual signal — the ranking comes from the page image, not OCR

| Tenant | Query | Retrieves | Why it's interesting |
|---|---|---|---|
| `embedded-lab` | Raspberry Pi Pico pinout diagram | Pi Pico datasheet pinout page | Colored pinout diagram drives ranking, not prose. |
| `ops-eng` | Kubernetes architecture diagram | CNCF Kubernetes slides | Visual architecture slide instead of OCR text. |
| `aerospace` | solid rocket motor nozzle cross-section drawing | Solid nozzle report (NASA SP-8115) | Scanned drawing with **no text layer** — only visual retrieval finds it. |
| _(no filter)_ | Arduino UNO power tree diagram | Arduino UNO datasheet power-tree page | Routes to the right tenant unfiltered. |

### Value lookups — the vision model reads a value off the page

| Tenant | Query | Page | Answer |
|---|---|---|---|
| `ops-eng` | What is the default TCP port a PostgreSQL server listens on? | PG 18 manual, Connection Settings | `5432` |
| `ops-eng` | What is the default value of max_connections in PostgreSQL? | PG 18 manual, Connection Settings | `100` |
| `embedded-lab` | What is the GPIO (IO) voltage of the Raspberry Pi Pico? | Pi Pico datasheet | `3.3V` |

### Disambiguation — two reports in one tenant, the right one must win

| Tenant | Query | Picks | Beats |
|---|---|---|---|
| `aerospace` | regeneratively cooled nozzle | `liquid-rocket-engine-nozzles.pdf` (regenerative cooling is liquid-specific) | `solid-rocket-motor-nozzles.pdf` |
| `aerospace` | solid rocket motor nozzle cross-section drawing | `solid-rocket-motor-nozzles.pdf` | `liquid-rocket-engine-nozzles.pdf` |

### Tenant-leak negatives — the matching content lives in a different tenant

| Scoped to | Query | Pass condition |
|---|---|---|
| `ops-eng` | regeneratively cooled nozzle | No aerospace pages return. |
| `aerospace` | Kubernetes architecture diagram | No ops-eng pages return. |
| `embedded-lab` | PostgreSQL default port | No ops-eng pages return. |

## API

### `GET /api/search`

| Parameter | Required | Description |
|---|---|---|
| `q` | yes | Search query |
| `client` | no | Tenant filter, for example `embedded-lab`. Omitted means search all tenants. |

```bash
curl "http://localhost:8888/api/search?q=What+is+the+GPIO+voltage+of+the+Raspberry+Pi+Pico%3F&client=embedded-lab"
```

```json
{
  "query": "What is the GPIO voltage of the Raspberry Pi Pico?",
  "client": "embedded-lab",
  "answer": "1.8-3.3V",
  "results": [
    {
      "page_id": "embedded-lab__raspberry-pi-pico-datasheet__p005",
      "client": "embedded-lab",
      "title": "Raspberry Pi Pico Datasheet",
      "publisher": "Raspberry Pi Ltd",
      "source_pdf": "raspberry-pi-pico-datasheet.pdf",
      "page_number": 5,
      "citation": "raspberry-pi-pico-datasheet.pdf · p.5",
      "page_image": "/pages/embedded-lab/raspberry-pi-pico-datasheet_p005.png",
      "scores": { "maxsim": 19.63, "rerank": null }
    }
  ]
}
```

### `GET /api/clients`, `GET /api/stats`

Tenant list and runtime config: active models, rerank on/off, and page count.

## How it works

```text
        ingest.py  (once per corpus)
        fetch_pdfs.py -> data/pdfs/{tenant}/*.pdf
             -> render_pages.py -> data/pages/{tenant}/*.png
             -> data/pages_manifest.json
             -> SIE.encode(ColQwen2.5, images, multivector)
             -> data/multivectors.npz + data/metadata.json

        search.py / server.py  (per query)
        q -> SIE.encode(ColQwen2.5, text, is_query=True)
          -> filter metadata by tenant
          -> sie_sdk.scoring.maxsim -> top_k_candidates
          -> optional SIE.score(Qwen3-VL-Reranker, q, images)
          -> Qwen3.5-4B (vision) via /v1/chat/completions on [top_page] -> answer
```

OCR is never on the score path — retrieval and the answer both read the page as
an image. The visual reranker, when enabled, ranks over the same modality as
retrieval, so layout cues survive every stage.

The corpus is small enough that MaxSim runs in Python. For thousands of pages,
hand the multivectors to LanceDB, Vespa, or another multivector store; the SIE
calls stay the same.

## Customize

`data/fetch_pdfs.py` owns the curated source list. Add a source with:

```python
{
    "client": "my-tenant",
    "slug": "my-manual",
    "title": "My Manual",
    "publisher": "Example Publisher",
    "license": "CC BY 4.0",
    "url": "https://example.com/my-manual.pdf",
    "pages": [1, 2, 7, 8],
}
```

Then rerun:

```bash
python data/fetch_pdfs.py
python data/render_pages.py
python python/ingest.py
```

`config.yaml` is the model and rendering tuning surface:

```yaml
models:
  retriever: "vidore/colqwen2.5-v0.2"
  answer: "Qwen/Qwen3.5-4B"
  reranker: "Qwen/Qwen3-VL-Reranker-2B"
render:
  backend: "auto"
  dpi: 160
search:
  top_k_candidates: 5
  top_k_results: 3
  visual_rerank: false
  answer: true
```

## Project layout

```text
examples/vision-doc-rag/
├── config.yaml
├── data/
│   ├── fetch_pdfs.py          # curated public PDF source list + downloader
│   ├── render_pages.py        # PDFs -> PNG pages + pages_manifest.json
│   ├── pdfs/                  # generated
│   ├── pages/                 # generated PNGs
│   ├── metadata.json          # generated by ingest
│   └── multivectors.npz       # generated by ingest
├── python/
│   ├── ingest.py
│   ├── search.py
│   ├── server.py
│   └── requirements.txt
└── static/
    └── index.html
```
