# Fill a JSON Schema from a messy document

The runnable example behind [superlinked.com/structured-output](https://superlinked.com/structured-output).

## What this shows

Ten public records, each with its own JSON Schema, sent to
`Qwen/Qwen3.8-27B-FP8` on `https://api.superlinked.com` through
`/v1/chat/completions` with `response_format: json_schema`. The documents are
NHTSA vehicle complaints, SEC officer appointment filings and GSA surplus
vehicle listings.

The page publishes one figure:

> All 10 documents came back as schema-valid JSON, 91 of 93 fields right

`score.py` re-derives that figure from the recorded responses, offline.

Acceptance checks were written on 2026-09-15 from reading each source text,
before that case's first model run, and are in `inputs/checks.json` with that
date recorded.

## Run it

The code is in this repository. The inputs and the recorded responses are in
the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is not enough. Fetch, then score:

```sh
python3 fetch.py         # downloads the pinned revision into data/
python3 score.py         # reproduces the figure offline
python3 run.py --check   # rebuilds every recorded request from the inputs
```

These three need nothing installed: they are standard library only, and none
of them needs an API key, a Hugging Face token or any inference spend.
`fetch.py` pins a commit SHA, not `main`, and checks every downloaded file
against a digest.

To call the API yourself. This is the only part that needs the SDK, and the
only command here that spends anything:

```sh
uv sync
SIE_API_KEY=... uv run python run.py --record --out run-output/calls.json
```

`run.py` sends through `sie_sdk.SIEClient`. The import is deferred into
`main()`, so `--check` and `--show` keep working on a bare `python3` with
nothing installed. `python3 run.py --show <id>` prints a request without
sending it.

## What to expect

```
schema validator: jsonschema 4.21.1
documents scored:  10
parsed as JSON:    10
schema-valid:      10
fields right:      91 of 93
excluded from every total: 3 CPSC cases, listed with their reason in inputs/excluded.json

fields the model got wrong:
  gsa-377053.condition: expected 'unknown', got 'repairable'
  nhtsa-11231274.components: expected ['AIR BAGS'], got ['FIRE RELATED']

Reproduced: all 10 documents schema-valid, 91 of 93 fields right.
```

`score.py` exits nonzero if any of those numbers fails to reproduce.
`run.py --check` prints `13 page requests rebuilt from inputs and matched the
recorded request`.

## What is in the dataset

```
structured-output/
  inputs/cases.json      13 cases: source excerpt, instruction, JSON Schema, provenance
  inputs/checks.json     the acceptance checks, with the date they were written
  inputs/excluded.json   3 CPSC cases excluded from every published total, with the reason
  calls.json             86 calls: request, response, status, timing, one file
  manifest.json          endpoint, model, served revision, run dates, file digests
```

`calls.json` holds 86 calls. 13 of them are the `page` set the figure is
computed from, after the 3 excluded cases are dropped, leaving 10. The other 73
are `diagnostics/*`: archival probes from earlier revisions of the sie-web
runner, kept so nothing was dropped in the move. They are in no total.

Those 86 calls were 172 separate JSON files in sie-web. Merging them changed no
byte of any request or response.

## What this does NOT establish

- **Nothing about accuracy on your documents.** Ten records from three
  publishers is a demonstration, not a benchmark.
- **Nothing about the two wrong fields being the only possible errors.** The
  93 checks cover the fields that could be read off each source text. Fields
  the source does not state are schema-validated and not scored.
- **Nothing about reproducibility of a new run.** The figure is re-derived from
  responses recorded on 2026-09-15 against server version 0.7.3 and model
  revision `8bd714204e67a1c6c81f84b0dc486b6a6e96e943c42ff488f6b3cbf936e07955`.
  A fresh `--record` run may differ.
- **Nothing about the `diagnostics/*` calls.** They were written by earlier
  runner revisions, `run.py --check` does not rebuild them, and it says so
  rather than counting them as checked.
- **A fresh `--record` run records less than the archive.** `sie_sdk` returns
  the per-item result rather than the server's envelope, and surfaces no
  response headers, so an entry written by `--record` carries a `shape` field
  saying so. `score.py` reads the published `calls.json`.
- **No tamper resistance.** The digests here catch a truncated or corrupted
  download. They are not a provenance chain and are not meant to survive
  someone who can write to the dataset.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/structured-output/`, which is what its CI
tests read. The two copies hold the same recorded responses. Nothing binds them
together, so they can drift.
