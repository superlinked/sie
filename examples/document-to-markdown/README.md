# Turn difficult PDFs into Markdown with one SIE call

This example sends four real PDFs through SIE’s `docling` model and checks the
parts that usually break downstream agents: table structure, reading order,
headings, and form labels.

The source set includes an NVIDIA CFO commentary, a SiriusPoint investor deck,
the Docling paper, and a FEMA proof-of-loss form. The fetch command records the
publisher URL and exact checksum for each local copy. The federal government
form travels with the dataset as a fallback because FEMA may block datacenter
downloads.

## What the run proves

The conversion is one SDK call:

```python
result = client.extract(
    "docling",
    Item(document=Path("pdfs/nvidia-q4-fy2025-cfo-commentary.pdf")),
)
markdown = result["data"]["markdown"]
```

The saved run contains the raw SIE response, exported Markdown, endpoint, model,
latency, source URL, and deterministic checks. The checks look for exact facts,
section order where linear order applies, and Markdown tables. The form check
asserts labels, choices, amount sections, and certification language. It does
not pretend a form has one useful reading order.

## Verified result

Recorded against SIE Cloud, `https://api.superlinked.com`, on September 16,
2026. All 25 checks passed. These conversions are published on
[superlinked.com/doc-to-markdown](https://superlinked.com/doc-to-markdown), and
that page's sources are in its
[SOURCES.md](https://superlinked.com/reference/doc-to-markdown/SOURCES.md), a
different file from the `SOURCES.md` in this directory. Every request,
response, model revision and timing is in the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so these numbers can be checked rather than taken on trust:

```bash
python3 fetch.py            # standard library only, no key, no token
uv run verify-run data
```

`fetch.py` pins a commit SHA rather than `main`, and checks every downloaded
file against a digest before the verifier sees it. The run bundle it downloads
is byte-for-byte the one that used to sit in this repository under
`runs/cloud-20260916/`, so the digest chain below is the same chain.

| Document | Checks | Latency, recorded run |
|---|---:|---:|
| NVIDIA Q4 FY2025 CFO commentary | 6/6 | 9.2 s |
| SiriusPoint Q1 2025 investor presentation | 6/6 | 12.2 s |
| Docling technical report | 6/6 | 6.7 s |
| FEMA proof-of-loss form | 7/7 | 1.3 s |

### All three runs

The session made three complete runs of the same four documents against the
same endpoint. All three passed, and all three produced **byte-identical
Markdown** — the conversion is deterministic here, so this is reproducible
rather than merely repeated. Every timing is below, because publishing one
run's numbers without the others invites you to read a single measurement as a
typical one.

| Document | Run 1 | Run 2 | Run 3 (recorded) |
|---|---:|---:|---:|
| NVIDIA Q4 FY2025 CFO commentary | 24.7 s | 9.8 s | 9.2 s |
| SiriusPoint Q1 2025 investor presentation | 12.9 s | 11.7 s | 12.2 s |
| Docling technical report | 7.3 s | 7.2 s | 6.7 s |
| FEMA proof-of-loss form | 1.4 s | 1.3 s | 1.3 s |
| **Total** | **46.3 s** | **30.1 s** | **29.3 s** |

Totals are summed from the unrounded measurements, so adding the rounded rows
above can land a tenth of a second away — run 3's rows come to 29.4 s against a
true total of 29.3 s. The exact milliseconds are in each run's manifest.

Run 3 is the recorded one because it is the only run produced by this version
of `convert.py`: runs 1 and 2 predate it recording a per-call model revision,
so they carry no `calls.json` and `verify-run` cannot check them.
That is the reason, and it is worth being blunt about what it does not explain.
**Run 3 is also the fastest of the three overall, and fastest on three of the
four documents.** It was not chosen for that.

`data/inputs/repeat-runs.json` keeps runs 1 and 2 — their timings and the
SHA-256 of every file they produced — but not their Markdown, so the corpus is
redistributed once rather than three times. To check the byte-identical claim
yourself, hash `data/markdown/<slug>.md` and compare.

Read any of these as single measurements on shared Cloud hardware we do not
control, not as a benchmark. The model was not resident when the session began,
and that first sequence paid about 15 seconds of provisioning on its first
document alone — 24.7 s against 9.8 s next time — while the other three moved
by under 1.3 s across all three runs.

The Markdown, though, did not move at all. `convert.py` changed between run 2
and run 3, to record the per-call model revision, and all four digests are the
same before and after. A change that adds provenance should leave the thing it
describes untouched, and this one did.

| Document | SHA-256 of the converted Markdown, identical in all three runs |
|---|---|
| NVIDIA Q4 FY2025 CFO commentary | `2c64f61d65417edf…` |
| SiriusPoint Q1 2025 investor presentation | `153064274f4e3bfc…` |
| Docling technical report | `47b058fe6676cbea…` |
| FEMA proof-of-loss form | `75520494bf872ebb…` |

### Verify without running

The whole bundle verifies offline, with no API key and no network:

```bash
python3 fetch.py
uv run verify-run data
```

It recomputes the 25 from the recorded check arrays rather than reading a
stored total, checks every digest, and asserts that each scored Markdown file is
exactly the Markdown its recorded API response returned. That last check exists
because `eval-documents` reads `<run>/markdown/<slug>.md` and never opens
the response beside it, so on its own it would be scoring a file this harness
wrote. Binding the two was added after running the example exposed the gap.

It prints `61 of 61 checks passed, 4 not checked`. The four not checked are the
source PDF digests, which need the PDFs: run `uv run fetch-documents` and they
become checks too. A missing file is reported, never skipped into the total.

## Run it

Use Python 3.12 and a SIE endpoint that serves `docling`.

```bash
cd examples/document-to-markdown
cp .env.example .env
uv sync

uv run fetch-documents                 # the four source PDFs, into pdfs/
uv run convert-documents --run-id local
uv run eval-documents runs/local       # writes runs/local/evaluation.json
uv run verify-run runs/local           # the same 61 checks, over your own run
```

`convert-documents`, `eval-documents` and `verify-run` share one run-bundle
layout, so a run you make yourself verifies exactly like the recorded one:
`run-manifest.json`, `calls.json`, `payloads/`, `markdown/` and
`evaluation.json`, with the source provenance in `pdfs/manifest.json`.

Checking the recorded run instead needs no key and no endpoint:

```bash
python3 fetch.py                       # the recorded run, into data/
uv run verify-run data                 # digests and relations
uv run eval-documents data             # the 25 checks, against the recorded Markdown
```

A fetched bundle carries the `.sie-evidence` marker `fetch.py` writes, and
`eval-documents` sends its output to `run-output/evaluation.json` when it sees
that marker. The recorded `data/evaluation.json` is pinned by a digest the
fetch checked, so overwriting it would leave the bytes disagreeing with the
manifest; keeping both lets you diff them. `eval-documents` prints how many
documents score the same as the recorded file.

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

In this repository, and nothing else:

```text
config.yaml                       source URLs, model, and acceptance checks
SOURCES.md                        rights and attribution notes
fetch.py                          downloads the recorded run at a pinned revision
document_to_markdown/             fetch, convert, evaluate, verify, canonical JSON
```

Downloaded by `python3 fetch.py` into `data/`, which it owns and replaces
wholesale:

```text
data/manifest.json                the dataset manifest: endpoint, models, run
                                  dates and a SHA-256 for every file below
data/run-manifest.json            endpoint, model revisions, timings, and digests
data/calls.json                   every call in one file: request, response,
                                  status, headers, timing, model revision
data/payloads/*.json              large unscored response members, by digest
data/markdown/*.md                exported Markdown, the text the checks score
data/evaluation.json              exact pass and failure details
data/inputs/sources.json          source URL, rights, byte length and SHA-256 per PDF
data/inputs/repeat-runs.json      earlier complete runs, kept for the
                                  byte-identical claim above
data/inputs/fema-proof-of-loss-form.pdf   the one PDF that is redistributed
```

Written by the commands you run, and all ignored by git:

```text
pdfs/                             the source PDFs, by `fetch-documents`
pdfs/manifest.json                their URLs, rights, byte lengths and SHA-256
runs/<run-id>/                    a run bundle, by `convert-documents`, in the
                                  same layout as the fetched one above
run-output/evaluation.json        by `eval-documents`, when the run bundle it
                                  scored was fetched rather than made locally
```

The source PDFs stay fetch-only: two of the four are investor documents whose
rights notes say not to redistribute the complete file. `data/inputs/sources.json`
pins each URL, byte length and SHA-256 instead, which is what a reader needs to
confirm they fetched the same bytes this run scored. The FEMA form is the
exception and travels with the dataset, because it is a work of the U.S. federal
government under 17 U.S.C. 105 and because `fema.gov` refuses datacenter
addresses.

One of the four may refuse you. `fema.gov` answers a request carrying a Chrome
user-agent with `403` and serves plain `curl` the PDF — backwards from what
anyone debugging that 403 would guess, and the reason `fetch.py` retries with
`curl` before falling back to the bundled copy. Tested from one network only,
so it says nothing about what a browser on a home connection sees. The
`retrieval` field in `data/inputs/sources.json` records which path a fetch took;
`publisher-after-403-via-curl` means the fallback was used. The recorded run
predates that finer-grained value and records `publisher` for all four,
which is true of every one of them — FEMA's arrived through the `curl` retry.

One call is one entry in `calls.json` rather than a file of its own. Members
that nothing scores move to `payloads/` and are referenced by digest — for
Docling that is `data.document`, between 85 and 98 percent of every response.
Those payloads are digested over their stored bytes rather than canonically,
because they carry `origin.binary_hash`, a 64-bit integer that JavaScript
cannot parse without silently changing it.

Two kinds of digest appear in a run, and which is which matters if you go to
check one:

| Digest | Taken over |
| --- | --- |
| `manifest_sha256`, `calls.sha256`, `entry_sha256`, `scored_markdown.response_markdown_sha256` | RFC 8785 canonical encoding of the parsed value, so reformatting a copy does not change it |
| `$payload.sha256`, `scored_markdown.sha256`, `document_sha256`, `source_sha256`, and `sha256` in `data/inputs/sources.json` | the recorded file bytes, reproducible with `shasum -a 256` |

The split is not arbitrary. Content that is compared against a reformatted copy
elsewhere is hashed canonically; content that is a file a reader will hash
themselves is hashed as bytes.

One consequence worth knowing before you edit anything. The note describing this
scheme lives *inside* `calls.json`, so correcting that note changes
`calls.sha256` and `manifest_sha256` — while every `entry_sha256` stays fixed,
because an entry holds only its own request, response, timing and model
revision. That is the intended shape: the entry digests are what pin the
evidence, and they cannot move without a request or a response moving. If you
change the prose and the two document-level digests move while the four entry
digests do not, nothing about the run has changed. If an entry digest moves, the
run has.

## What the conversion gets wrong

All 25 checks pass. The tables are still not structurally reliable, and those
two facts are both true because the checks test different things than you might
assume. Everything below is in `data/markdown/` to read, once you have run
`python3 fetch.py`.

**The tables do not survive as tables.** There is not one structurally clean
table in this corpus — I checked all fifteen. The behaviour is consistent:
Docling expands a spanning header across every cell it covers instead of
merging it, and it does not separate tables that are stacked on a page.

The NVIDIA output shows both at once. Lines 42 to 72 look like one 31-row
Markdown table. They are four separate PDF tables welded together, with each
section label expanded across all four cells acting as the seam:

Lines 42, 45, 50, 51 and 53 of that file, verbatim:

```text
| ($ in millions, except earnings per share)   | FY25                           | FY24                           | Y/Y                            |
| Gross margin                                 | 75.0 %                         | 72.7 %                         | Up 2.3 pts                     |
| Non-GAAP                                     | Non-GAAP                       | Non-GAAP                       | Non-GAAP                       |
| ($ in millions, except earnings per share)   | FY25                           | FY24                           | Y/Y                            |
| Gross margin                                 | 75.5 %                         | 73.8 %                         | Up 1.7 pts                     |
```

GAAP and non-GAAP figures for the same line item end up in one table under one
header, with nothing but a repeated word between them. The Q4 table above it
welds four PDF tables the same way. SiriusPoint's header repeats
`Financial Highlights` across all four columns and its `Q1'24` column comes
back twice. The Docling paper's own benchmark table goes further and collapses
multi-row cells into space-joined values — `4 16`, `177 s 167 s` — while a
second table flattens an entire per-class accuracy table into two cells, one
holding every label and one holding every value.

**A form loses its labels entirely.** On the FEMA proof-of-loss form, 26 of the
124 non-blank lines are a bare `$` and nothing else. Every amount field comes
back as a naked dollar sign with its label stripped, and no line in the file
carries a currency amount attached to a label:

```text
- [ ] Other:

$

$

$
```

This is the plainest thing in the corpus. A table with a repeated header still
reads as a table, so it is possible to talk yourself out of caring; a column of
26 dollar signs is not. And it is a form — the category where the relationship
between a label and its field is purely spatial, and so the hardest for any
converter.

**Three smaller behaviours.** Fenced code blocks are flattened onto one line,
so the four-statement Python example in the converted Docling paper would not
run as printed. Line-break hyphens are closed up, so `MIT-licensed` returns as
`MITlicensed`. And on the same FEMA form every section heading is emitted
together near the top while its fields appear much later — `TYPE OF PROOF OF
LOSS` at line 7, its six choices from line 57. The content survives; the
grouping does not.

**Why the checks pass anyway, and what that says about the checks.** They test
exact facts, section order, and that tables are present — not that a table is
the right shape. `_table_count` counts contiguous pipe blocks, so the four
welded NVIDIA tables count as one, and `7 found, 4 required` passes honestly
while saying nothing about structure. `contains:` checks have the same shape:
`contains:AMOUNTS CLAIMED` passes because that heading survives at line 9, and
says nothing about the 26 unlabelled `$` fields beneath it.

That is a limitation of this evaluation, not a detail about the model, and it is
worth knowing before you copy the approach: a conversion harness that checks
facts and order will not tell you your tables are wrong, and one that checks a
heading is present will not tell you the fields under it are gone. Each check
answers a narrower question than the claim it is used to back.

## Honest scope

This example measures whether the converted structure keeps the facts and order
needed by an application. It does not claim a universal PDF benchmark score.
Encrypted files, handwriting, and pages dominated by diagrams need separate
tests and may need an OCR or vision model instead of the default Docling profile.
