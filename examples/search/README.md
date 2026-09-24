# Find the passage that answers a question

## What this shows

253 passages from federal regulations, IRS instructions and software
documentation go to SIE Cloud as dense vectors. Twelve questions go through the
same model with `is_query` set, which is the asymmetry that makes this
retrieval rather than plain similarity. Then a cosine over what came back ranks
every passage for every question, and the scorer reports where each answer
landed.

The corpus is deliberately full of near misses. Thirty-one passages cover
general-industry fall protection and twenty-one cover construction, so the
fall-protection question has a close wrong answer sitting in the corpus. The
hours-of-service rules for trucks, buses and Alaska read almost identically.

The run is already recorded. The twelve encode calls and the vectors they
returned live in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision by `fetch.py`. Download it and you can re-derive the
published number with **no API key and no inference spend**. Those are the same
vectors behind the figure on
[superlinked.com/search](https://superlinked.com/search), whose sources are in its
[SOURCES.md](https://superlinked.com/reference/search/SOURCES.md).

You cannot verify this by cloning alone. The clone gives you the code; the
dataset gives you the evidence. Fetching it needs no account and no token.

- Model: `Snowflake/snowflake-arctic-embed-l-v2.0`, 1024-dimensional dense vectors
- Endpoint: `https://api.superlinked.com/v1/encode/Snowflake/snowflake-arctic-embed-l-v2.0`
- Model revision: `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`, as `GET /v1/models` reported on 2026-09-21
- Served deployment revision: `10333b84de80b402376b626eb25366fb081d3faeb893eb4b01cf32e8c27e4aff`, the `X-SIE-Model-Revision` every recorded call carries. Models served together share this value, so it names the deployment rather than the weights.
- SIE server version 0.7.3, recorded 2026-09-15
- 12 calls: the 253 passages in batches of 24, then all 12 questions in one call

## Run it

Download the recorded run, then rank it. Both steps are standard library only,
so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

The download is about 5 MB of float arrays and the ranking takes a few seconds,
because it is a plain Python loop over 253 by 12 cosines rather than a matrix
library.

Look at a request without sending it:

```sh
python3 run.py --show queries
python3 run.py --show corpus-000
```

Encode it yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --output run-output
```

## What result to expect

`score.py` prints a row per question and ends with:

```
253 passages, 12 questions, 1024-dimensional vectors
the answer is in the top 3 for every question
it ranks first for 9 of the 12
```

That is the figure the task page publishes. The three questions the model does
not lead are the construction fall-protection height, where the
general-industry rule outranks it, and the bus and Alaska driving limits, where
a neighbouring hours-of-service rule wins. All twelve are scored and printed;
the page displays seven of them.

`score.py` fails rather than skipping. A passage whose text no longer matches
its digest, a response that does not match its `response_sha256`, a request the
pinned corpus does not rebuild, a call nothing pins, a call whose served model
revision is not the one the manifest names, a passage with no recorded vector,
a vector returned twice, a vector whose declared width disagrees with its own
values, a non-finite value, or an answer that falls outside the top 3 all exit
non-zero.

## What this does NOT establish

- **Not a benchmark score.** Twelve questions is a demonstration, not a
  measurement. There is no comparison model here, no held-out set and no
  confidence interval, so nothing in this output supports "model A beats model
  B" or a percentage you could quote.
- **Not that the top hit is the answer.** The published claim is that the
  answer is in the top 3, and for three of the twelve questions it is not
  first. A system built on this needs to read more than the first result, or to
  rerank.
- **Not a test of scale.** 253 passages fit in memory and are scored by a
  Python loop. Nothing here says anything about latency, recall or index
  behaviour at a million passages.
- **Not evidence the questions are unbiased.** The questions were authored
  alongside the corpus. Each names its answer passage through a verbatim phrase
  recorded in `inputs/queries.json`, resolved before anything was encoded, so
  the answer key predates the run. It is still an authored set, not a sample of
  what users ask.
- **Not bound to the website.** The same run backs the fixtures in
  `superlinked/sie-web` under `apps/site/tests/fixtures/reference/search/`,
  which is what that repository's CI checks. Nothing automatically ties the two
  copies together, so they could drift.
- **Not a guarantee the dataset is unchanged.** `fetch.py` pins a dataset
  revision rather than `main`, so a later upload cannot silently change what
  you score. It does not prove the revision holds what it held yesterday.

## Inputs

`evidence/inputs/corpus.json` holds 253 passages, each with its own SHA-256 and
the source it was cut from. `score.py` checks every digest before ranking
anything. `evidence/manifest.json` lists the eight sources with publisher, URL,
licence and the digest of the file each was built from: FMCSA and DOT
regulations, OSHA general-industry and construction rules, IRS Publication 15,
Kubernetes probe docs, PostgreSQL vacuum docs and NIST SP 800-63B-4. All are US
government works, CC BY 4.0 or the PostgreSQL License.

`evidence/inputs/queries.json` holds the twelve questions. Each carries
`answer_ids` and, in `answers`, the verbatim phrase that made a passage the
answer.

`retrieval.py` builds the request bodies and does the ranking. `run.py` and
`score.py` both read it from there, so the request the scorer rebuilds is the
request the runner sends, and the scorer refuses a recording it cannot rebuild.
