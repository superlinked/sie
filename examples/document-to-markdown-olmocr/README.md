# Turn PDF pages into Markdown, scored on Ai2's olmOCR-Bench

## What this shows

SIE's hosted `lightonai/LightOnOCR-2-1B` turned all 1,403 pages of Ai2's
[olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) into
Markdown, one `extract` call per page, on 30 September 2026. Ai2's own scorer,
unmodified, gives it an Overall of **75.3** (95% interval ±0.9).

Published runs of the same scorer version put Azure Document Intelligence
Layout at 48.7 and AWS Textract at 40.2. SIE charges $2.00 per 1,000 pages;
Azure's cheapest public tier for Layout is $6.00 and Textract's for Tables is
$10.00. `score.py` re-derives every figure from the recorded run and checks it
against [superlinked.com/doc-to-markdown](https://superlinked.com/doc-to-markdown),
whose sources are in its
[SOURCES.md](https://superlinked.com/reference/doc-to-markdown/SOURCES.md).

The call is one image in and Markdown out:

```python
from pathlib import Path
from sie_sdk import SIEClient

client = SIEClient(api_key="sk-sie-…", base_url="https://api.superlinked.com")
image = {"data": Path("page.png").read_bytes(), "format": "png"}
result = client.extract("lightonai/LightOnOCR-2-1B", {"images": [image]})
print(result["entities"][0]["text"])
```

## The benchmark and the score

olmOCR-Bench holds 1,403 single-page PDFs and 7,019 tests in seven files:
arXiv pages with math, old scans with math, tables, multi-column pages, long
runs of tiny text, old scans, and pages whose running headers and footers must
be left out. Each test checks one fact about the Markdown: a sentence is
present, a header is absent, two passages come in order, a table cell sits
next to the right neighbour, an equation renders. The scorer adds an eighth
file, `baseline`, with one test per page that the output is not blank or
looping.

Overall is the mean of the eight per-file pass rates, the figure on Ai2's
leaderboard. Pooling every test into one rate gives a different, higher number
(77.7 here) and must not be compared with a published Overall.

| System | Overall | arXiv math | Baseline | Headers and footers | Tiny text | Multi-column | Old scans | Old scans math | Tables |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIE LightOnOCR-2-1B | 75.3 | 89.8 | 99.7 | 19.9 | 91.6 | 84.5 | 42.0 | 85.8 | 88.8 |
| Azure Document Intelligence Layout, published | 48.7 | 0.0 | 99.1 | 21.7 | 85.5 | 67.4 | 28.3 | 0.0 | 87.6 |
| AWS Textract, published | 40.2 | 0.0 | 100.0 | 26.8 | 65.8 | 19.7 | 23.6 | 0.0 | 85.3 |

Without the two math files, where neither service emits LaTeX, the six other
files average 71.1 for SIE LightOnOCR-2-1B, 64.9 for Azure and 53.5 for
Textract; `score.py` prints both. Azure's Markdown wraps page headers and
footers in HTML comments, which the scorer reads as text, so its 21.7 on
headers and footers is likely an underestimate; with every one of those tests
passed its Overall would be 58.5.

The two rival rows are [Unsiloed's run](https://github.com/Unsiloed-AI/unsiloed-olmocr-benchmark/blob/main/reports/preliminary_2026-05-19.md)
on olmocr 0.4.27 (19 May 2026), not ours: Azure `prebuilt-layout` with Markdown
output, and Textract `AnalyzeDocument` with `TABLES`. Neither service emits
LaTeX by default, so both score 0 on the math files.

The recorded run has two more arms, reported but not claimed:

- **LightOnOCR-2-1B with headers and footers removed**: every Markdown line
  that matches a page header or footer found by SIE's `docling` on the same
  page is dropped. Headers and footers rise from 19.9 to 52.4 and Overall to
  79.3.
- **`docling` alone**: 46.9.

## Where the recorded run lives

The per-test results, the scorer's summaries and the Markdown for every page
are in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
folder `document-to-markdown-olmocr/`, pinned to one revision by `fetch.py`:

```
document-to-markdown-olmocr/
  scores/          Ai2's scorer's summary for each arm: per-file rates, Overall, interval
  per_test/        one line per test per arm: test id, file, type, page, passed
  outputs/         the Markdown for all 1,403 pages, one archive per arm, in the scorer's layout
  rivals.json      the published rival scores and the prices, with sources and dates
  hero/            the NVIDIA filing page shown on superlinked.com/doc-to-markdown
  manifest.json    endpoint, model, rate book, run date, and the SHA-256 of every file
```

`score.py` pins the manifest's SHA-256, and the manifest pins every other file,
so a missing or altered file stops the score instead of changing a number.

## Run it

Reproduce the published figures. Standard library only, no key:

```sh
python3 fetch.py
python3 score.py
```

Convert pages yourself (20 pages cost about $0.04), then score them with Ai2's
scorer:

```sh
export SIE_API_KEY=sk-sie-...
uv run run.py --pages 20
uv run --with 'olmocr[bench]==0.4.27' score.py --rescore runs
```

`--rescore` needs Playwright's Chromium for the math tests
(`uv run --with 'olmocr[bench]==0.4.27' playwright install chromium`).
`uv run run.py --all` converts every page for about $2.81. A score over a
sample is noisier than the full run: Ai2's interval is about ±1 point at 1,403
pages.

## Settings

| Setting | Value |
| --- | --- |
| Model | `lightonai/LightOnOCR-2-1B`, default profile, SIE Cloud |
| Input | each page rendered to PNG with pypdfium2, longest side 1,540 px |
| Benchmark | `allenai/olmOCR-bench` at `54a96a6fb6a2bd3b297e59869491db4d3625b711` |
| Scorer | `olmocr[bench]==0.4.27`, unmodified |
| Price | 200 credits a page, $2.00 per 1,000 pages |

olmOCR-Bench is Ai2's, released under ODC-BY 1.0.
