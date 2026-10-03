# Fill a JSON Schema from a messy document

The runnable example behind [superlinked.com/structured-output](https://superlinked.com/structured-output).
That page's sources are in its [SOURCES.md](https://superlinked.com/reference/structured-output/SOURCES.md).

## What this shows

Ten public records, each with its own JSON Schema, sent to
`Qwen/Qwen3.8-27B-FP8` on `https://api.superlinked.com` through
`/v1/chat/completions` with `response_format: json_schema`. The documents are
NHTSA vehicle complaints, SEC officer appointment filings and GSA surplus
vehicle listings.

The page's heading:

> All 21 yes-or-no fields came back right, and no document says true or false

and the line under its proof grid:

> 84 of the 85 checked fields came back right

`score.py` re-derives every one of those from the recorded responses, offline.

A yes-or-no field is one the case schema declared `"type": "boolean"` or
`"type": ["boolean", "null"]`, and nothing wider, read from the schema the call
sent rather than from the value that came back. A union like
`["boolean", "string"]` is not a yes-or-no question and is not counted as one. Sixteen are plain booleans. Five admit null, and
two of those correctly came back null because the GSA listing gives no answer,
which still counts as a question the schema asked.

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
python3 run.py --check   # rebuilds the 13 page requests from the inputs
```

These three need nothing installed: they are standard library only, and none
of them needs an API key, a Hugging Face token or any inference spend.
`fetch.py` pins a commit SHA, not `main`, and checks every downloaded file
against a digest. It replaces `--dest` wholesale, so it refuses to touch
anything without the `.sie-evidence` marker it writes, and it swaps the new
directory in by rename rather than deleting the old one first.

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
schema validator: built in
documents scored:  10
parsed as JSON:    10
schema-valid:      10
fields right:      84 of 85
  yes-or-no fields:  21 of 21
  picks from a list: 5 of 5
  everything else:   58 of 59
documents saying true or false: 0
excluded from every total: 3 CPSC cases, listed with their reason in inputs/excluded.json

fields the model got wrong:
  nhtsa-11443034.injured_people: expected 1, got 0
```

The one field the model got wrong is a count, which is why the other two groups
are perfect. Its narrative says the contact "had chest and neck pains due to
hitting the steering wheel and did go to urgent care but did not seek medical
attention", a sentence that contradicts itself; the registered 1 comes from
NHTSA's own coded record and the pains are stated, so 1 is the better reading
and 0 is a defensible reading of the second clause.

Two fields were dropped from the schemas on 2026-09-24 and the eight documents
carrying them were re-recorded, which is why the totals moved from 91 of 93.
`condition` on the GSA lots asked for a condition grade that lot 377053 does not
state; `components` on the NHTSA complaints asked for NHTSA's own taxonomy,
which the narrative cannot reach. Neither was answerable from the document.
`inputs/cases.json` records both, and the calls they replace are kept in the
`superseded/2026-09-15` set of `calls.json`.

`score.py` needs nothing installed. Schema validation runs under the small
validator built into `score.py`, which covers the subset of JSON Schema these
cases use, and it prints `schema validator: built in`. If `jsonschema` happens
to be importable, `score.py` uses that instead and prints its version, and the
two verdicts are compared against the one the original runner recorded, so all
three have to agree. `jsonschema` is optional and nothing here installs it.

`score.py` exits nonzero if any of those numbers fails to reproduce.
`run.py --check` prints `13 page requests rebuilt from inputs and matched the
recorded request`, and fails if a page call is missing, recorded twice or
implied by no case.

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
- **Nothing about the one wrong field being the only possible error.** The
  85 checks cover the fields that could be read off each source text. Fields
  the source does not state are schema-validated and not scored.
- **Nothing about why a yes-or-no field is right.** `score.py` establishes that
  the answer matches the check and that the words true and false appear in no
  source text. It cannot show the model reasoned rather than guessed, and with
  21 binary answers a run of luck is not ruled out by this set.
- **Nothing about reproducibility of a new run.** The figure is re-derived from
  responses recorded on 2026-09-15 and 2026-09-24 against server version 0.7.3
  and model
  revision `8bd714204e67a1c6c81f84b0dc486b6a6e96e943c42ff488f6b3cbf936e07955`.
  A fresh `--record` run may differ.
- **Nothing about the `diagnostics/*` calls.** They were written by earlier
  runner revisions, `run.py --check` does not rebuild them, and it says so
  rather than counting them as checked.
- **A fresh `--record` run records no response headers.**
  `client.chat_completions` returns the server's own envelope, so a fresh run's
  response body matches the archived one; what it cannot carry is the response
  headers, which the archived run recorded from raw HTTP. Each entry carries a
  `shape` field saying exactly that. `score.py` reads the published
  `calls.json`.
- **No tamper resistance.** The digests here catch a truncated or corrupted
  download. They are not a provenance chain and are not meant to survive
  someone who can write to the dataset.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/structured-output/`, which is what its CI
tests read. The two copies hold the same recorded responses. Nothing binds them
together, so they can drift.
