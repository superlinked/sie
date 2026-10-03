# Route a request to the action it asks for

The runnable example behind [superlinked.com/classify](https://superlinked.com/classify).

## What this shows

Fourteen held-out SNIPS validation requests scored against seven action labels
by `knowledgator/gliclass-large-v3.0` on `https://api.superlinked.com`, with
no training and no fine-tune. The labels are just strings in the request.

Two figures, both reported in the page's
[SOURCES.md](https://superlinked.com/reference/classify/SOURCES.md):

> the reference action ranked first for all 14 requests

and:

> the same model on the same endpoint led on only 3 to 5 of 12 texts across
> four earlier runs

`score.py` re-derives both from the recorded responses, offline. The second one
is the honest half: when the label set overlaps, the same model does much
worse, and the four runs that showed it are in the dataset rather than
discarded.

**These are not the page's headline.** Since 2026-09-22 the page leads with a
three-model comparison recorded separately: the same 14 requests answered by
this classifier and by `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.8-27B-FP8`, three
passes each. All three got every call right, and the classifier was charged 1
credit per request against the 27B's 6. That recording is not in this dataset
revision, so **this example does not reproduce the page's headline figure.**
It reproduces the two figures above, which is what it has always reproduced.
The comparison bundle, its pre-registration and its runner live in sie-web at
`apps/site/tests/fixtures/reference/classify/comparison/`.

Cases were selected by a published rule before any run. `inputs/snips/cases.json`
records the selection rule, the build bar (`at least 10 of 14`), the upstream
commit and a SHA-256 for every source file.

## Run it

The code is in this repository. The inputs and the recorded responses are in
the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
so a clone alone is not enough. Fetch, then score:

```sh
python3 fetch.py         # downloads the pinned revision into data/
python3 score.py         # reproduces both figures offline
python3 run.py --check   # rebuilds all 62 recorded requests from the inputs
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
SIE_API_KEY=... uv run python run.py --record --set snips --out run-output/calls.json
```

`--record` sends one set. The published `calls.json` holds all five, so
`--check` over a single-set recording correctly reports the other four as
missing.

`run.py` sends through `sie_sdk.SIEClient`. The import is deferred into
`main()`, so `--check` and `--show` keep working on a bare `python3` with
nothing installed. `python3 run.py --show <id>` prints a request without
sending it.

## What to expect

```
Distinct action names (SNIPS validation rows)
  reference label ranked first: 14 of 14

Four earlier runs, where the labels overlapped
  cfpb                   3 of 12
  cfpb-definitions       3 of 12
  clinc150               3 of 12
  clinc150-definitions   5 of 12
  range: 3 to 5 of 12

Reproduced: 14 of 14 on the distinct labels, 3 to 5 of 12 across the 4 earlier runs.
```

`score.py` exits nonzero if either figure fails to reproduce.
`run.py --check` prints `62 of 62 recorded requests rebuilt from the inputs and
matched`, and fails if a call is missing, recorded twice or implied by no case.

## What is in the dataset

```
classify/
  inputs/snips/cases.json     14 SNIPS rows, selection rule, source digests
  inputs/clinc150/cases.json  12 CLINC150 rows, queue map, intent-list labels
  inputs/cfpb/cases.json      12 CFPB narratives, product queue map
  calls.json                  62 calls: request, response, status, timing, one file
  manifest.json               endpoint, model, served revision, run dates, digests
```

The 62 calls were 124 separate JSON files in sie-web. Merging them changed no
byte of any request or response.

Sets: `snips` (14, the distinct action names), `clinc150` (12),
`clinc150-definitions` (12), `cfpb` (12), `cfpb-definitions` (12).

## What this does NOT establish

- **Nothing about your label set.** The whole point of the second figure is
  that the result depends on how distinct the labels are, and 14 of 14 was
  measured on seven labels that barely overlap. Run your own labels.
- **Nothing about the 3-to-5 range as a floor.** Those four runs used 12 texts
  each. Twelve texts cannot separate a 25% model from a 42% one.
- **Nothing about a served model revision mapping to a Hugging Face commit.**
  Every response carried
  `x-sie-model-revision: 10333b84de80b402376b626eb25366fb081d3faeb893eb4b01cf32e8c27e4aff`.
  SIE does not document how that maps to the upstream commit the model config
  pins, so this example claims no mapping.
- **The recorded path spells the model id differently from the SDK.** The
  2026-09-15 runner percent-encoded it into
  `/v1/extract/knowledgator%2Fgliclass-large-v3.0`; `sie_sdk.SIEClient` sends
  the slash unencoded. The two unquote to the same path and reach the same
  endpoint, the request bodies are byte-for-byte the same, and `run.py --check`
  asserts that equivalence rather than hiding it. The other four examples in
  this batch recorded the unencoded form.
- **Nothing about re-running today.** The responses were recorded on
  2026-09-15 against server version 0.7.3.
- **A fresh `--record` run records less than the archive.** `client.extract`
  returns the per-item result rather than the server's envelope, and surfaces
  no response headers, so `--record` rebuilds the envelope around that item and
  each entry carries a `shape` field saying so. `score.py` reads the published
  `calls.json`.
- **Nothing the page currently leads with.** The 2026-09-22 three-model
  comparison is not in this dataset revision, so nothing here reproduces the
  credit prices, the medians or the 126-of-126 figure the page publishes.
  Those come from a separate bundle held in sie-web.
- **No tamper resistance.** The digests catch a truncated or corrupted
  download. They are not a provenance chain.

sie-web keeps its own copy of these recordings under
`apps/site/tests/fixtures/reference/classify/`, which is what its CI tests
read. The two copies hold the same recorded responses. Nothing binds them
together, so they can drift.
