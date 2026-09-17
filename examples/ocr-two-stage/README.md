# ocr-two-stage

**A recorded evaluation, not a demo.** There is no UI here and nothing to click.
If you want a browser demo that swaps OCR models live, that is
[`examples/document-ocr`](../document-ocr). This directory holds one recorded
run and the code to check its numbers.

## What this shows

Two stages, two different jobs, measured separately.

**Stage 1** sends a page image to `lightonai/LightOnOCR-2-1B` and gets Markdown
back. **Stage 2** sends *that Markdown* — never the image — to a chat model
under a fixed JSON schema, and gets typed fields back. Stage 2 runs twice, once
on `Qwen/Qwen3.8-27B-FP8` and once on `Qwen/Qwen3.5-4B`, so the same stage-1
text is scored against two models.

Splitting them is the point. When a field comes back wrong you can ask which
stage lost it: the evaluator checks whether the field's registered printed token
survived into stage 1's Markdown, which separates an error stage 2 inherited
from one it made on text it could read.

Six documents, 20 calls, all against `https://api.superlinked.com`.

## How to run it

Checking the numbers needs no API key and no network:

```
python3 evaluate.py
```

It reads exactly three committed files and nothing else:
`verified-run/calls.json` for what came back, `data/inputs.json` for what was
registered before the run, and `data/predictions.json` for the predictions it
scores. No key, no network, no fourth file.

Re-recording the run needs a key in `SIE_API_KEY`:

```
python3 run.py                 # all six documents, both stages
python3 run.py --verify-inputs # fetch the images and check their digests only
```

The images are **not committed**. Several carry licences that do not permit
redistribution here — one is a copyrighted NVIDIA investor page used as an
attributed excerpt, another is CC BY-SA 3.0 whose share-alike would attach to a
copy. Each document in `data/inputs.json` instead carries the URL serving the
exact bytes that were sent, their length, their SHA-256, the upstream original
and the licence. `run.py` fetches by that URL and refuses to proceed if the
length or digest differs.

## What result to expect

`evaluate.py` prints four figures, which are the ones the
[/ocr page](https://superlinked.com/ocr) publishes:

| | |
|---|---|
| stage 1 | **81 of 86** registered printed tokens survive into the Markdown |
| stage 2, `Qwen/Qwen3.8-27B-FP8` | **78 of 89** fields match, 7 of 7 replies schema-valid |
| stage 2, `Qwen/Qwen3.5-4B` | **27 of 89** fields match, **2 of 7** replies schema-valid |

The gap between the two stage-2 models is the interesting part: the smaller one
mostly fails before it is even scored, because five of its seven replies do not
validate against the schema they were given.

## What this does NOT establish

- **It is not a benchmark.** Six documents chosen to be awkward in different
  ways — a born-digital financial page, a blurred thermal receipt, a barcode
  label, a container photo, a factory control panel, an invoice — are not a
  sample of anything. No score here generalises.
- **It is one run.** Recorded 2026-09-16. Nothing here shows variance across
  runs, and these models are not pinned to a revision by us; the revision each
  call actually got is recorded per call in `calls.json`.
- **The verdicts encode judgement.** What counts as a match was registered
  before the run in `data/inputs.json`, but somebody chose those rules. The
  evaluator applies them mechanically; it does not make them right.
- **It says nothing about documents unlike these.** Handwriting, non-Latin
  scripts and multi-column layouts are absent.

## Layout

```
data/inputs.json        what was registered before the run, and how to fetch each image
data/predictions.json   the prediction spec the evaluator scores
verified-run/calls.json every call: request, response, status, model revision, timing
verified-run/evaluation.json  what evaluate.py produced from the above
run.py                  makes the calls
evaluate.py             scores the recorded calls, offline
```

## Relationship to sie-web

The same recorded responses back the `/ocr` page in the `sie-web` repository,
where CI asserts that the page cannot publish a number its recorded response
does not support. That copy is the one CI reads; this one is the one a reader
reads. They are not bound to each other by a digest, so they can in principle
drift; if they disagree, the sie-web fixture is the CI-bound record.

`evaluate.py` here is sie-web's evaluator with its input layer changed to read
`calls.json`. Every verdict rule, the normalization and the schema checks are
copied unchanged, and it reproduces sie-web's evaluation exactly: the same
stage-1 count, the same per-model totals, the same per-document verdicts.
