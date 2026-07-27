# Turn difficult PDFs into Markdown with one SIE call

This example sends four real PDFs through SIE’s `docling` model and checks the
parts that usually break downstream agents: table structure, reading order,
headings, and form labels.

The source set includes an NVIDIA CFO commentary, a SiriusPoint investor deck,
the Docling paper, and a FEMA proof-of-loss form. The fetch command records the
publisher URL and exact checksum for each local copy. The federal government
form is bundled as a fallback because FEMA may block datacenter downloads.

## What the run proves

The conversion is one SDK call:

```python
result = client.extract(
    "docling",
    Item(document=Path("data/pdfs/nvidia-q4-fy2025-cfo-commentary.pdf")),
)
markdown = result["data"]["markdown"]
```

The saved run contains the raw SIE response, exported Markdown, endpoint, model,
latency, source URL, and deterministic checks. The checks look for exact facts,
section order where linear order applies, and Markdown tables. The form check
asserts labels, choices, amount sections, and certification language. It does
not pretend a form has one useful reading order.

## Verified result

We ran the example on an NVIDIA L4 through a local SIE endpoint on July 24,
2026. All 25 checks passed:

| Document | Checks | SIE latency |
|---|---:|---:|
| NVIDIA Q4 FY2025 CFO commentary | 6/6 | 50.3 s |
| SiriusPoint Q1 2025 investor presentation | 6/6 | 16.3 s |
| Docling technical report | 6/6 | 11.7 s |
| FEMA proof-of-loss form | 7/7 | 10.0 s |

## Run it

Use Python 3.12 and a SIE endpoint that serves `docling`.

```bash
cd examples/document-to-markdown
cp .env.example .env
uv sync

uv run fetch-documents
uv run convert-documents --run-id local
uv run eval-documents runs/local
```

The default `.env` points at a local SIE server:

```bash
pip install "sie-server[local]"
sie-server serve
```

For SIE Cloud, change only the endpoint and key:

```bash
SIE_CLUSTER_URL=https://api.superlinked.com
SIE_API_KEY=...
```

## Source and result layout

```text
config.yaml                       source URLs, model, and acceptance checks
fixtures/SOURCES.md               rights and attribution notes
data/pdfs/                        fetched PDFs, ignored by git
data/manifest.json                acquired checksums, ignored by git
runs/<run-id>/manifest.json       endpoint, model, latency, and output paths
runs/<run-id>/raw/*.json          complete SIE responses
runs/<run-id>/markdown/*.md       exported Markdown
runs/<run-id>/evaluation.json     exact pass and failure details
```

## Honest scope

This example measures whether the converted structure keeps the facts and order
needed by an application. It does not claim a universal PDF benchmark score.
Encrypted files, handwriting, and pages dominated by diagrams need separate
tests and may need an OCR or vision model instead of the default Docling profile.
