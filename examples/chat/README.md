# Write public status updates from internal incident reports

## What this shows

Twelve real incident reports from Wikimedia Foundation engineers go to SIE
Cloud, one `/v1/chat/completions` call each, and come back as a four-line
public status update. A report names hosts (`gerrit1003`), Phabricator tasks
(`T393034`) and the responders who worked the outage. A status page must carry
none of that, must state the outage window the report states, and must stay
short enough to read at a glance.

The run is already recorded. `calls.json` holds all twelve requests and the
exact responses they returned, so you can re-derive the published number with
no API key and no network. Those are the same bytes behind the figure on
[superlinked.com/chat](https://superlinked.com/chat).

- Model: `Qwen/Qwen3.8-27B-FP8`
- Endpoint: `https://api.superlinked.com/v1/chat/completions`
- Served model revision: `8bd714204e67a1c6c81f84b0dc486b6a6e96e943c42ff488f6b3cbf936e07955`
- SIE server version 0.7.3, recorded 2026-09-15

Each answer is scored against four checks, written before the run:

| Check | Passes when |
|---|---|
| `format` | exactly four lines, labelled Status, Impact, Window and Cause, and the first line reads `Status: Resolved` |
| `window` | the start and end in the Window line are times the report itself states |
| `length` | the Impact and Cause sentences are 25 words or fewer |
| `no_internal_identifiers` | no hostname, numbered instance, internal domain, Phabricator task or responder name appears |

## Run it

Score the recorded run. Nothing to install, no key, no network:

```sh
python3 score.py
```

Run the tests, which also cover the fail-closed paths:

```sh
python3 -m unittest discover -s tests -v
```

Look at a request without sending it:

```sh
python3 run.py --show sessionstore
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output/calls.json
```

## What result to expect

`score.py` prints `4/4` for each report and ends with:

```
12 of 12 incident reports passed all four checks
```

That is the figure the task page publishes. The page displays four of the
twelve reports, one in the hero and three in the proof grid; all twelve are in
`calls.json` here and all twelve are scored. The page's playground runs a
thirteenth call on a short excerpt of one report, which is not part of the
twelve and is not shipped here.

`score.py` fails rather than skipping. A source file whose bytes no longer
match `data/cases.json`, a case with no recorded call, a response that does not
match its `response_sha256`, or pinned text that no longer rebuilds the
recorded request body all exit non-zero. Nothing is scored around a missing
input.

## What this does NOT establish

- **Not that the updates are true.** The four checks are mechanical. An update
  whose Cause line confidently misdescribes the outage passes all four, as long
  as it is well formed, short and leaks no identifiers. Nobody checked the
  twelve summaries against the reports for accuracy.
- **Not that the model never leaks.** The leak check knows four regular
  expressions and the names each report itself lists. An internal identifier in
  a shape those patterns miss would pass.
- **Not a rate.** This is one run of twelve reports on one day. Sampling
  defaults were used and no case was repeated, so a rerun can return different
  wording and a different score. Treat 12 of 12 as what this recording shows,
  not as a measured pass rate.
- **Not a general result about incident reports.** All twelve come from one
  organisation, in one house format, with a `{{Incident scorecard}}` template
  the runner parses. Reports from a different tracker will need a different
  derivation and may score differently.
- **Not a benchmark.** The recorded `duration_ms` values are provenance. They
  include queueing and whatever the cluster was doing at the time.
- **Not bound to the website.** The same recordings back the fixtures in
  `superlinked/sie-web` under `apps/site/tests/fixtures/reference/chat/`, which
  is what that repository's CI checks. Nothing automatically ties the two
  copies together, so they could drift.

## Inputs

`data/cases.json` pins twelve Wikitech incident reports by revision id, with
the SHA-256 and byte length of the wikitext committed beside it in
`data/sources/`. Wikitech content is CC BY-SA 4.0. `score.py` verifies every
file against its digest before scoring, and separately rebuilds each recorded
request from that wikitext, so the committed text is provably the text that
produced the recorded answer.

`prompt.py` derives the report the model sees from the wikitext: the page
title, the scorecard fields, then the prose. `run.py` and `score.py` both read
it from there, so the request the scorer rebuilds is the request the runner
sends.
