# Find the page that answers a question, without reading its text

## What this shows

Four questions run against four real documents, 712 pages in all. Each question
is ranked two ways over the same candidate set, which is every page of the
document it belongs to:

- **Text.** BM25 over the page's extracted markdown.
- **Visual.** ColPali over the rendered page image, with no OCR step.

ColPali returns one vector per image patch rather than one vector per page, so a
chart covering a tenth of a page keeps its own vectors. Ranking is MaxSim: each
query vector takes its strongest match anywhere on the page, and those are
summed. A revenue figure that lives in a bar chart, a hand-signal diagram and a
boxed statistic are all findable this way. BM25 over the same page's text is not
close on any of them.

The pages come from [ViDoRe v3](https://huggingface.co/vidore), a public
benchmark that ships each query, its answer and a human relevance grade. The
page this example ranks for is the benchmark's own grade-2 page, fixed before
anything was encoded.

The run is already recorded. All 95 encode calls and the multivectors they
returned live in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision by `fetch.py`. Download it and you can re-derive both
rankings with **no API key and no inference spend**. These rankings are
published on
[superlinked.com/visual-document-search](https://superlinked.com/visual-document-search),
whose sources are in its
[SOURCES.md](https://superlinked.com/reference/visual-document-search/SOURCES.md).

You cannot verify this by cloning alone. The clone gives you the code; the
dataset gives you the evidence. Fetching it needs no account and no token.

- Model: `vidore/colpali-v1.3-hf`, 128 dimensions per token
- Model revision: `133a9eb02947310513f52f8f0d39d622e0eab8dc`
- Endpoint: `POST /v1/encode/vidore/colpali-v1.3-hf` on a SIE server you run
- Baseline: BM25 over the ViDoRe `markdown` field, `k1` 1.2, `b` 0.75
- Recorded 2026-09-21 on one NVIDIA L4, 95 calls over 712 pages

## Run it

Download the recorded run, then rank it. Both steps are standard library only,
so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

The download is 552 MB, most of it the 712 page images and the multivectors
they produced. Scoring takes about a minute, because MaxSim over 712 pages is a
plain Python loop rather than a matrix library.

See the text baseline on its own, straight from the inputs, with no recording
read at all:

```sh
python3 score.py --baseline
```

Encode it yourself. SIE Cloud does not serve a ColPali-family model, so this
example points at a SIE server you start. The model downloads and loads on the
first request, so that call takes a few minutes and every later one is warm:

```sh
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-default

uv sync
uv run python run.py
```

On macOS (Apple Silicon) or Linux you can run the server natively instead:

```sh
pip install "sie-server[local]" && sie-server serve
```

Point somewhere else with `SIE_BASE_URL`, and set `SIE_API_KEY` if your
deployment wants one:

```sh
SIE_BASE_URL=https://sie.example.internal:8080 uv run python run.py
```

Look at a request without sending it:

```sh
python3 run.py --show aircraft-refueling-signals
```

## What result to expect

`score.py` prints a row per comparison and ends with:

```
comparison                           pages  text  visual    maxsim
Aircraft refueling hand signals         68    10       1    19.871
Healthcare permanent-contract rates    202    13       1    27.883
Morgan Stanley revenue increase        268    49       2    20.906
Digital skills in Bulgaria             174     8       2    17.515

4 comparisons over 712 pages
the visual side puts the benchmark page first in 2 of 4
the text side never does, at best rank 8
```

Those are the figures the task page publishes. Read the two rank columns
together: the page a reader wants sits at 10, 13, 49 and 8 under text search,
and at 1, 1, 2 and 2 once the same page is ranked by what it looks like.

The Morgan Stanley row is the widest gap and the most instructive. Its answer
sits on the page in plain English: "revenues of $61.8 billion in 2024, which
increased by 14% compared with $54.1 billion". BM25 still ranks that page 49th
of 268, because it matches none of the query's three discriminating terms. The
question asks for `percentage`, `increase` and `revenue`. The page says
`increased` and `revenues`, and with no stemmer those are different tokens, so
all three score zero. What is left to match on is `morgan` and `stanley`, which
appear on 90% and 89% of the document's pages. Ranking the rendered image
instead puts the page second.

`score.py` fails rather than skipping. A missing file, a manifest naming a
different model or a different model revision, a page whose bytes no longer
match their digest, a response that does not match its `response_sha256`, a
request the pinned inputs do not rebuild, a response whose item ids are not the
ones its own request asked for in that order, a call nothing scores, a page with
no recorded multivector, a multivector recorded twice, a declared token count or
width that disagrees with its own values, a non-finite score, or a manifest
whose recorded results are missing, duplicated, extra or at a different rank
from the ones derived here all exit non-zero.

A response digest proves a response has not been edited. It does not tie a
response to the request beside it, which is why the id sequence is checked per
call: without that, two page batches could trade answers, every digest would
still verify, and every page would still appear exactly once.

That last check is the one covering the text side. Response digests span the
multivectors, so a changed vector is caught before any rank is computed. Nothing
digests the markdown in `pages.json`, so BM25 is where a silent edit would
otherwise land: the ranks `run.py` recorded in `manifest.json` and the ranks
recomputed here come from different files, and they have to agree.

## What this does NOT establish

- **Not a benchmark score.** Four questions is a demonstration. ViDoRe v3 ships
  thousands, and nothing here is an NDCG or a recall figure you could quote
  against another model.
- **Not a claim about attention.** MaxSim says which page matched and how
  strongly. It does not say which patch earned the match, and this example
  draws no boxes.
- **Not an OCR comparison.** The text baseline is BM25 over the markdown ViDoRe
  ships. A different extractor, a different tokenizer or a dense text model
  would all move the text ranks, and this example makes no claim about which
  text pipeline is best.
- **Not a test of scale.** The candidate set is one document at a time, at most
  268 pages, ranked by a Python loop. Nothing here speaks to a million-page
  index or to latency under load.
- **Not bound to the website.** Nothing automatically ties this run to the
  figures on superlinked.com, so the two could drift.

## Inputs

`evidence/inputs/pages.json` holds all 712 pages with the ViDoRe dataset,
`corpus_id`, `doc_id`, page number, dimensions, SHA-256 and the markdown the
text baseline reads. `score.py` checks every digest before ranking anything.

`evidence/inputs/comparisons.json` holds the four questions, each with its
ViDoRe query id and revision, the document it is asked against, and the grade-2
page the benchmark marks as the answer. The three source datasets are pinned by
revision: `vidore_v3_industrial` at `e26c8647`, `vidore_v3_hr` at `0cdf0979`
and `vidore_v3_finance_en` at `7f432c17`.

`retrieval.py` builds the request bodies and does both rankings. `run.py` and
`score.py` both read it from there, so the request the scorer rebuilds is the
request the runner sends, and the scorer refuses a recording it cannot rebuild.

The pages travel as raw file bytes. SIE's SDK keeps already encoded PNG
unchanged on the wire, so the digest recorded in `calls.json` is the digest of
what SIE read, and `inputs/` holds those exact bytes.

Multivectors are recorded as base64 float16, which is what the server returns
when the request asks for `output_dtype: "float16"`. `score.py` decodes them
with `base64` and `struct` from the standard library.
