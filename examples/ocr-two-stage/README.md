# Read a page image, then read the Markdown

The runnable example behind [superlinked.com/ocr](https://superlinked.com/ocr).

**A recorded evaluation, not a demo.** There is no UI here and nothing to click.
If you want a browser demo that swaps OCR models live, that is
[`examples/document-ocr`](../document-ocr). This directory holds the code that
checks one recorded run's numbers.

## What this shows

Two stages, two different jobs, measured separately.

**Stage 1** sends a page image to `lightonai/LightOnOCR-2-1B` and gets Markdown
back. **Stage 2** sends *that Markdown*, never the image, to a chat model under a
fixed JSON schema, and gets typed fields back. Stage 2 runs twice, once on
`Qwen/Qwen3.8-27B-FP8` and once on `Qwen/Qwen3.5-4B`, so the same stage-1 text is
scored against two models.

Splitting them is the point. When a field comes back wrong you can ask which
stage lost it: the scorer checks whether the field's registered printed token
survived into stage 1's Markdown, which separates an error stage 2 inherited from
one it made on text it could read.

Six documents, 20 calls, all against `https://api.superlinked.com` on 2026-09-16.

## Run it

The code is in this repository. The registered inputs and the recorded responses
are in the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is no longer enough. Fetch, then score:

```sh
python3 fetch.py         # downloads the pinned revision into data/
python3 score.py         # reproduces every published figure offline
python3 run.py --check   # rebuilds all 20 recorded requests from the inputs
```

These need nothing installed. They are standard library only, and none of them
needs an API key, a Hugging Face token or any inference spend. `fetch.py` pins a
commit SHA rather than `main`, and checks every downloaded file against a digest
before the scorer sees it.

`run.py --check` is where the pipeline claim is tested. It rebuilds each stage-2
request body from the instruction template and the Markdown in the recorded
stage-1 response, then compares it with the stage-2 request that was actually
sent. All 14 match, so stage 2 saw that text and nothing else.

The images are **not** in the dataset. Several carry licences that do not permit
redistribution, one is a copyrighted NVIDIA investor page used as an attributed
excerpt, and another is CC BY-SA 3.0 whose share-alike would attach to a copy.
Each document in `inputs/inputs.json` carries the URL serving the exact bytes
that were sent, their length, their SHA-256, the upstream original and the
licence:

```sh
python3 run.py --verify-inputs   # fetches all six and checks length and digest
```

Re-recording the run needs a key. This is the only part that needs the SDK, and
the only command here that spends anything:

```sh
uv sync
SIE_API_KEY=... uv run python run.py --record --out run-output/calls.json
```

`run.py` sends through `sie_sdk.SIEClient`. The import is deferred into `main()`,
so `--check`, `--show` and `--verify-inputs` keep working on a bare `python3`.

## What result to expect

`score.py` prints the figures the
[/ocr page's evidence](https://superlinked.com/reference/ocr-review/SOURCES.md)
publishes for the whole recorded run:

| | |
|---|---|
| stage 1 | **81 of 86** registered printed tokens survive into the Markdown |
| stage 2, `Qwen/Qwen3.8-27B-FP8` | **78 of 89** fields match, 7 of 7 replies schema-valid |
| stage 2, `Qwen/Qwen3.5-4B` | **27 of 89** fields match, **2 of 7** replies schema-valid |

The gap between the two stage-2 models is the interesting part. The smaller one
mostly fails before it is even scored, because five of its seven replies do not
validate against the schema they were given.

`score.py` then lists every primary-model field that did not match, with its
verdict, and exits nonzero if any published figure fails to reproduce. It also
writes `data/evaluation.json`, the per-field artifact the page's data is built
from.

The page's proof grid shows a selection of this run: the schema calls that
matched every field they registered. Which cases it shows is the page's
decision, it changes without the run changing, and nothing here can read the
page, so `score.py` does not check it. `SOURCES.md` on the page names the
selection and the figures above in the same place.

## What this does NOT establish

- **It is not a benchmark.** Six documents chosen to be awkward in different
  ways, a born-digital financial page, a blurred thermal receipt, a barcode
  label, a container photo, a factory control panel and an invoice, are not a
  sample of anything. No score here generalises.
- **It is one run.** Recorded 2026-09-16. Nothing here shows variance across
  runs, and these models are not pinned to a revision by us; the revision each
  call actually got is recorded per call in `calls.json`.
- **The verdicts encode judgement.** What counts as a match was registered
  before the run in `inputs/inputs.json`, but somebody chose those rules. The
  scorer applies them mechanically; it does not make them right.
- **It says nothing about documents unlike these.** Handwriting, non-Latin
  scripts and multi-column layouts are absent.
- **No tamper resistance.** The digests catch a truncated or corrupted download.
  They are not a provenance chain.

## What is in the dataset

```
ocr-two-stage/
  inputs/inputs.json       what was registered before the run: the six documents,
                           how to fetch each image and check it, the JSON schemas,
                           the expected fields and the pre-registered predictions
  inputs/predictions.json  how each registered prediction is decided from the run
  calls.json               20 calls: request, response, status, model revision, timing
  manifest.json            endpoint, models, revisions, run date and a digest for every file
```

`score.py` reads exactly those three files and nothing else. It was
`evaluate.py`, reading the same three files from this repository; every verdict
rule, the normalization and the schema checks are unchanged, and the
`evaluation.json` it writes is byte-identical to the one that used to be
committed here.

## Relationship to sie-web

The same recorded responses back the `/ocr` page in the `sie-web` repository,
where CI asserts that the page cannot publish a number its recorded response does
not support. That copy is the one CI reads; this one is the one a reader reads.
They are not bound to each other by a digest, so they can in principle drift; if
they disagree, the sie-web fixture is the CI-bound record.
