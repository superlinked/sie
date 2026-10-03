# Pick the passage that meets every condition

## What this shows

Eight questions, each with four candidate passages taken from the same
documentation page, are encoded as token vectors. A ColBERT model returns
one vector per token rather than one per text, so the comparison happens
client-side: every query token takes its best match in a passage and the
matches sum. That is MaxSim, and `score.py` is mostly those four lines.

The four candidates in each search are deliberately near-duplicates. All four
SQLite passages are `ALTER TABLE` sections and two of them are about `DROP
COLUMN`; three of the four Kafka passages are about `acks`. A single vector per
passage has to average a whole paragraph into one point, which is how a passage
about the right topic beats the passage that answers the actual question. Token
vectors let one decisive word, `concurrently` or `no-store`, carry the
comparison.

The run is already recorded. The sixteen encode calls and the token vectors
they returned live in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision by `fetch.py`. Download it and you can re-derive the
published number with **no API key and no inference spend**. Those are the same
vectors behind the figure on
[superlinked.com/multi-vector](https://superlinked.com/multi-vector), whose sources
are in its [SOURCES.md](https://superlinked.com/reference/multivector/SOURCES.md).

You cannot verify this by cloning alone. The clone gives you the code; the
dataset gives you the evidence. Fetching it needs no account and no token.

- Model: `lightonai/GTE-ModernColBERT-v1`, 128 dimensions per token
- Endpoint: `https://api.superlinked.com/v1/encode/lightonai%2FGTE-ModernColBERT-v1`
- Served deployment revision: `10333b84de80b402376b626eb25366fb081d3faeb893eb4b01cf32e8c27e4aff`, the `X-SIE-Model-Revision` every recorded call carries. Models served together share this value, so it names the deployment rather than the weights.
- HuggingFace revision the weights came from: `cbbe53366e564450558f5e639dd499171f127538`
- SIE server version 0.7.3, recorded 2026-09-15
- 16 calls: per search, the question with `is_query` true, then its four passages with `is_query` false

## Run it

Download the recorded run, then score it. Both steps are standard library only,
so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

The download is about 4 MB of float arrays.

Look at both requests for one search without sending them:

```sh
python3 run.py --show mdn-cache-no-store
```

Encode it yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

## What result to expect

`score.py` prints a row per search and ends with:

```
8 searches, 128 dimensions per token
the answer passage beat the closest same-page passage in 7 of 8 searches
it lost in: mdn-cache-no-store
```

That is the figure the task page publishes. The loss is narrow and it is
printed with the rest: 11.718 for the `no-store` passage against 11.746 for the
`private` passage, a gap of 0.028 on a scale where the whole sum runs past 11.
All eight searches are scored here. Which of them the page features is the
page's decision and is recorded in its own `SOURCES.md`.

This paragraph used to say which searches the page draws, and that the search
the answer lost was one of them. That went stale as soon as the page was
reselected, and nothing here can check it, so the claim is removed rather than
restated the other way round. The number the page and this script share is
7 of 8, and that has not moved.

`score.py` fails rather than skipping. A response that does not match its
`response_sha256`, a request the pinned inputs do not rebuild, a missing call,
a call nothing pins, a passage with no recorded vector, a call whose served
model revision is not the one the manifest names, or a token vector whose width
disagrees with its own `token_dims` all exit non-zero. That last one matters
more than it looks: MaxSim zips a query token against a passage token, and
`zip` stops at the shorter of the two, so a short row would score on a prefix
and say nothing about it.

## What this does NOT establish

- **Not that MaxSim beats single-vector retrieval.** There is no single-vector
  run here to compare against. Eight searches picked to have same-page
  near-duplicates show that late interaction separates them; they do not
  measure how a dense model would have done on the same eight.
- **Not a benchmark score.** Eight searches of four passages each is 32
  passages in total. Nothing here supports a percentage, and MaxSim totals are
  not comparable between searches: the sum is over query tokens and is never
  divided by their count, so a longer question scores higher throughout. Only
  the ordering inside one search means anything.
- **Not free.** Storing a vector per token instead of per passage is the cost
  of this, and the recorded responses are the evidence: 4 MB of vectors for 32
  short passages and 8 questions.
- **Not a complete recording.** Two further searches were run,
  `nist-password-length-mfa` and `osha-scaffold-height`, and neither is
  published, because `/search` already uses those sources. Their raw responses
  were never committed upstream either, so they are not in the dataset. The
  third call per search that the page's playground shows is also not here; it
  encodes the question as a document and produces different numbers.
- **Not bound to the website.** The same run backs the fixtures in
  `superlinked/sie-web` under `apps/site/tests/fixtures/reference/multivector/`,
  which is what that repository's CI checks. Nothing automatically ties the two
  copies together, so they could drift.
- **Not a guarantee the dataset is unchanged.** `fetch.py` pins a dataset
  revision rather than `main`, so a later upload cannot silently change what
  you score. It does not prove the revision holds what it held yesterday.

## Inputs

`evidence/inputs/cases.json` holds the eight searches: a question, four
candidate passages with their location on the source page, and the id of the
one that answers it. `evidence/manifest.json` lists the sources with URL,
licence and the digest of the page snapshot each passage was checked against:
SQLite, PostgreSQL and Python documentation, the Kubernetes sidecar guide, two
MDN pages, Apache Kafka configuration docs and 14 CFR 121.629. Licences run
from public domain to CC BY-SA 2.5, and every one permits redistribution.

`evidence/inputs/cases.json` also records the server-side preprocessing the
model applies: the `[Q] ` and `[D] ` prefixes, the document punctuation
skiplist and L2 normalisation. That last one is why `maxsim.py` takes a plain
dot product and normalises nothing.

`maxsim.py` builds the request bodies and computes MaxSim. `run.py` and
`score.py` both read it from there, so the request the scorer rebuilds is the
request the runner sends, and the scorer refuses a recording it cannot rebuild.
