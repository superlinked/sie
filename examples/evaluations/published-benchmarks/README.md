# Repeat the published benchmark numbers

This example repeats the method behind the six quality numbers published on
the SIE website, from one client process on your machine. Each
`references/<id>.json` is a projection of one published benchmark record: the
model and profile, the task key, the pinned dataset revision, the request shape
and the published metrics. `run.py quality` sends the same requests through
`sie_sdk.SIEClient`, ranks in the client, scores the ranking with `pytrec_eval`
and writes one record-shaped result with the delta against every published
metric. It prints deltas and never a verdict: no threshold, no pass, no fail.

## The six references

| Reference | Model | Profile | Task | Published main score |
| --- | --- | --- | --- | ---: |
| `prod-web-fw-20260905-splade-nfcorpus-quality-02` | `prithivida/Splade_PP_en_v2` | default | `mteb/NFCorpus` | ndcg_at_10 0.33056 |
| `prod-web-fw-20260905-bge-m3-sparse-nfcorpus-quality-04` | `BAAI/bge-m3` | sparse | `mteb/NFCorpus` | ndcg_at_10 0.28374 |
| `prod-web-fw-20260905-gte-moderncolbert-scifact-quality-06` | `lightonai/GTE-ModernColBERT-v1` | default | `mteb/SciFact` | ndcg_at_10 0.75995 |
| `prod-web-fw-20260905-bge-m3-multivector-scifact-quality-08` | `BAAI/bge-m3` | multivector | `mteb/SciFact` | ndcg_at_10 0.69583 |
| `prod-web-fw-20260905-qwen3-reranker-0-6b-askubuntu-quality-10` | `Qwen/Qwen3-Reranker-0.6B` | default | `mteb/AskUbuntuDupQuestions` | map_at_1000 0.59526 |
| `prod-web-fw-20260905-qwen3-reranker-4b-askubuntu-quality-12` | `Qwen/Qwen3-Reranker-4B` | default | `mteb/AskUbuntuDupQuestions` | map_at_1000 0.64484 |

## Run it

Install the locked dependencies, then set the key the endpoint requires:

```sh
uv sync --frozen
export SIE_API_KEY=your-key
```

A smoke run keeps the whole SciFact corpus and the first 50 queries that have a
relevance judgement. It still encodes every document, so it spends credits:

```sh
uv run python run.py quality \
  --reference references/prod-web-fw-20260905-gte-moderncolbert-scifact-quality-06.json \
  --task "mteb/SciFact?limit=50"
```

The full run repeats the published record. Drop `--task` and the reference's
own task key applies:

```sh
uv run python run.py quality \
  --reference references/prod-web-fw-20260905-gte-moderncolbert-scifact-quality-06.json
```

With `--reference`, model, profile, task, output contract, items per request
and request timeout come from the file. The only flags accepted beside it are
`--endpoint` (default `https://api.superlinked.com`) and `--task` with the
smoke key. Without a reference, `--model <id[:profile]>` and `--task` are
required; `--output sparse|multivector` and `--items-per-request` are required
for the encode tasks and refused for `mteb/AskUbuntuDupQuestions`, which scores
each query against its 20 task-native candidates.

A completed run writes two files under `results/`, which git ignores; a run
with a failed request writes the result record alone and names the failure
in `validity.reasons`:

- `results/<id>-<utc>.json`: the result record. It carries the reference id,
  `comparable` with its reasons, the model, profile and task, the dataset name
  and revision, the request shape, the observed `execution_revisions`, the
  run's request and failure counts, `validity` with the SDK retry count, every
  recomputed metric and a `comparison` row per published metric with the delta.
- `results/<id>-<utc>.predictions.json`: the ranked list per query, the top
  1000 documents with their scores, or the reranker's 20 candidates per query.
  Every metric is recomputed from this file.

## Recompute offline

```sh
uv run python score.py results/<id>-<utc>.json
```

`score.py` needs no API key and makes no inference call. It re-reads the
predictions file, loads the relevance judgements from the pinned dataset
revision named in the result with `datasets` (network or the local Hugging
Face cache), recomputes every metric and prints each one beside the published
value and the delta. It rewrites nothing unless you pass `--write`, which
updates `metrics` and `comparison` in the result in place.

## What is fixed by the method

These are not flags. They are the method behind the published numbers, and
the example applies them to every run:

- the whole `test` split of the pinned dataset revision; queries with no
  relevance judgement are dropped, as MTEB does;
- requests are sequential, one in flight, with no warmup;
- documents are encoded with `is_query=False` in batches of the reference's
  items per request, queries with `is_query=True`, and no instruction is sent;
- sparse vectors are scored by the raw dot product, multivector outputs by
  MaxSim over L2-normalised token rows, and the reranker's own scores rank its
  candidates;
- the top 1000 documents per query are kept, ties broken by corpus order, and
  scored at k in 1, 3, 5, 10, 20, 100 and 1000 with pytrec_eval's `ndcg_cut`,
  `map_cut`, `recall` and `P`, rounded to five decimals; the published method
  leaves the order of equal scores at rank 1000 unspecified, so an `_at_1000`
  value can differ in the fifth decimal when scores tie there;
- one attempt per request, `wait_for_capacity=False`, `max_oom_retries=0`,
  and the reference's request timeout (300 s; 600 s for the bge-m3 multivector
  record). After every call the SDK's retry count is read and recorded; any
  retry sets `valid: false`, as does any failed request.

## What the comparison means

`comparable: true` means the run's identity and parameters equal the
reference's: model, profile, task key, output contract, items per request,
endpoint, and one client process. The published quality records ran one worker
with one process, so a full run with no override is comparable. The smoke key,
an endpoint override or more than one execution revision observed in the run
sets `comparable: false` with the reason listed, and `comparison_basis` reads
`same method, different parameters: <reasons>` instead of `same parameters`.
A run without a reference is also `comparable: false` and has no comparison:
`comparison` is empty and `comparison_basis` is null.

Every response names the revision of the executed model bundle and
configuration. The run collects every distinct value into
`execution_revisions`; it is recorded, never compared with the reference's
checkpoint revision, and a change mid-run sets `comparable: false` with the
reason `execution revision changed`.

A delta is a number, not a result. The maintainers read a quality delta
against a 2% band when they decide whether a published number still holds;
that band is their reading policy, and this tool never prints it, nor any pass
threshold or quality floor.

## Verify

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The tests use a fake client and in-memory datasets; they need no key and no
network.

## Not in this example

- no publication path: nothing here writes to the website or to the published
  records, and no floor, target, threshold or gate;
- no `mteb` dependency, no nAUC or cv_recall metrics, and no match of the
  scorer's implementation digest;
- no multi-process client, load ladder or candidate generation;
- no cost, price or accounting fields;
- no recorded-evidence download: the inputs are the six reference files and
  the pinned public datasets;
- no PyPI distribution, and the flags are not a compatibility surface.
