# Repeat the published benchmark numbers

This example repeats the method behind the benchmark numbers published on the
SIE website, six quality records and six fixed-rate performance records, from
one client process on your machine. Each `references/<id>.json` is a
projection of one published benchmark record: the model and profile, the task
key, the pinned dataset revision, the request shape, the timing constants and
the published metrics. `run.py quality` sends the same requests through
`sie_sdk.SIEClient`, ranks in the client, scores the ranking with `pytrec_eval`
and writes one record-shaped result with the delta against every published
metric. `run.py performance` sends a record's requests at its published rate
and aggregates their timing the way the record did. Both print deltas and
never a verdict: no threshold, no pass, no fail.

## Quality

### The six references

| Reference | Model | Profile | Task | Published main score |
| --- | --- | --- | --- | ---: |
| `prod-web-fw-20260905-splade-nfcorpus-quality-02` | `prithivida/Splade_PP_en_v2` | default | `mteb/NFCorpus` | ndcg_at_10 0.33056 |
| `prod-web-fw-20260905-bge-m3-sparse-nfcorpus-quality-04` | `BAAI/bge-m3` | sparse | `mteb/NFCorpus` | ndcg_at_10 0.28374 |
| `prod-web-fw-20260905-gte-moderncolbert-scifact-quality-06` | `lightonai/GTE-ModernColBERT-v1` | default | `mteb/SciFact` | ndcg_at_10 0.75995 |
| `prod-web-fw-20260905-bge-m3-multivector-scifact-quality-08` | `BAAI/bge-m3` | multivector | `mteb/SciFact` | ndcg_at_10 0.69583 |
| `prod-web-fw-20260905-qwen3-reranker-0-6b-askubuntu-quality-10` | `Qwen/Qwen3-Reranker-0.6B` | default | `mteb/AskUbuntuDupQuestions` | map_at_1000 0.59526 |
| `prod-web-fw-20260905-qwen3-reranker-4b-askubuntu-quality-12` | `Qwen/Qwen3-Reranker-4B` | default | `mteb/AskUbuntuDupQuestions` | map_at_1000 0.64484 |

### Run it

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

### Recompute offline

```sh
uv run python score.py results/<id>-<utc>.json
```

`score.py` needs no API key and makes no inference call. It re-reads the
predictions file, loads the relevance judgements from the pinned dataset
revision named in the result with `datasets` (network or the local Hugging
Face cache), recomputes every metric and prints each one beside the published
value and the delta. It rewrites nothing unless you pass `--write`, which
updates `metrics` and `comparison` in the result in place.

### What is fixed by the method

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

### What the comparison means

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

## Performance

### The six performance references

| Reference | Model | Profile | Request | Level | p50 ms | p95 ms | p99 ms | Achieved rps |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `prod-web-fw-20260905-splade-nfcorpus-perf-01` | `prithivida/Splade_PP_en_v2` | default | one of 256 NFCorpus documents, sparse | `fixed-10-rps` | 144.4 | 241.6 | 491.5 | 9.98 |
| `prod-web-fw-20260905-bge-m3-sparse-nfcorpus-perf-03` | `BAAI/bge-m3` | sparse | one of 256 NFCorpus documents, sparse | `fixed-10-rps` | 172.7 | 217.8 | 387.0 | 9.98 |
| `prod-web-fw-20260905-gte-moderncolbert-scifact-perf-v2-r1` | `lightonai/GTE-ModernColBERT-v1` | default | one of 256 SciFact documents, multivector | `fixed-1-rps` | 178.8 | 198.8 | 285.0 | 1.0 |
| `prod-web-fw-20260905-bge-m3-multivector-scifact-perf-v2-r1` | `BAAI/bge-m3` | multivector | one of 256 SciFact documents, multivector | `fixed-1-rps` | 353.7 | 1164.3 | 1407.2 | 1.0 |
| `prod-web-fw-20260905-qwen3-reranker-0-6b-askubuntu-perf-09` | `Qwen/Qwen3-Reranker-0.6B` | default | one of 64 AskUbuntu queries with its 20 candidates, scores | `fixed-0-25-rps` | 210.5 | 230.2 | 230.2 | 0.25 |
| `prod-web-fw-20260905-qwen3-reranker-4b-askubuntu-perf-11` | `Qwen/Qwen3-Reranker-4B` | default | one of 64 AskUbuntu queries with its 20 candidates, scores | `fixed-0-25-rps` | 489.5 | 579.4 | 579.4 | 0.25 |

Published values of each record's first load level, latencies rounded to one
decimal, rates to two; the reference file carries every metric of that level.

### Run it

```sh
uv run python run.py performance \
  --reference references/prod-web-fw-20260905-gte-moderncolbert-scifact-perf-v2-r1.json
```

A run needs `SIE_API_KEY`, spends credits and takes about two minutes: one
readiness request, 30 s of warmup, 10 s of settle, 60 s of scheduled requests
and up to 10 s of drain. With `--reference`, model, profile, task, output
contract, items per request, timeout, in-flight cap, settle, drain and lag
tolerance come from the file; `--rate` defaults to the record's published
level, `--duration` to 60 and `--warmup` to 30, `--endpoint` is accepted as
for quality, and any value that differs from the reference adds a
`comparable_reasons` entry. Without a reference, `--model`, `--task` and
`--rate` are required, `--output` and `--items-per-request` are required for
the encode tasks and refused for the score task, and the timeout and cap are
30 s and 64 for encode, 300 s and 128 for score. No other flags exist.

### What is fixed by the method

- a readiness request on the first row, `wait_for_capacity=True` with a 300 s
  provisioning timeout, before the measurement; a failure writes
  `valid: false` with the reason and exits 1; then 30 s of unrecorded
  closed-loop warmup on that row (`--warmup 0` measures cold) and 10 s of
  idle settle;
- deterministic arrivals: request `i` is due at `i / rate` seconds while that
  is below the duration; a scheduler thread sleeps until each offset and hands
  the request to a pool of `max_in_flight_per_process` threads, and when all
  are busy the request waits and its scheduling lag grows; after the last
  offset, up to 10 s of drain, then a wait for any request still running to
  finish or time out, so every request has a final status;
- one attempt per request, no backoff, `wait_for_capacity=False`,
  `provision_timeout_s` equal to the request timeout, `max_oom_retries=0` and
  a connection pool twice the cap; after every call the SDK's retry count is
  read and written on the request's row;
- selection by the task key: `filter=nonempty` drops rows whose text is empty
  after stripping whitespace; `role=corpus` takes the documents, queries
  otherwise; the remaining rows are ordered by source id compared as strings
  and `limit=N` keeps the first N; for AskUbuntu, `limit=64` keeps the first
  64 queries in that order and `candidates-k=20` each query's 20 task-native
  candidates in source order; the chosen ids go to `dataset.selected_ids` in
  that order;
- rows cycle: encode request `s` takes documents `s × n` to `s × n + n − 1`
  of the selection, modulo its size, with `n` the items per request (1 in
  every record); score request `s` takes query `s` modulo the selection
  size (64 in the records);
- a document is sent as its title, a newline and its text, an empty title
  left out; encode items carry `is_query=False` and the reference's output
  type; a score request is the query text with its 20 candidates, without
  the `max_tokens_per_doc` the reranker references list, which bounded the
  published runs' inputs and was not part of their requests; responses are
  not scored, only their timing is kept.

### What the run writes

Beside the quality envelope, the result carries `load_level_name`, `schedule`
with the one level that ran, `timing` as in the reference, `run.requests_path`
and `validity`. The metrics use the record's names and aggregations:
`offered_requests_per_second` is the level rate; `achieved_requests_per_second`
counts successes completed inside the duration window (60 s in the records)
and `eventual_achieved_requests_per_second` all successes, each divided by
the duration, so a completion during drain counts only in the latter; the
four success rates are over the scheduled count, retry-free meaning one
attempt and a zero SDK retry count; the latency percentiles are nearest-rank
over successful requests of completed minus sent on the monotonic clock;
`scheduling_lag_p99_ms` is the nearest-rank p99 of actual send minus due
offset over every request, floored at zero. `validity.valid` is false after
any SDK retry, a lag p99 above the 20 ms tolerance, a failed readiness
request or an in-flight peak above the cap; `validity` also lists
`lag_max_ms`, `in_flight_peak`, `timeouts` and `sdk_retries`. A failed or
timed-out request does not invalidate the run; it lowers the success rates.

`results/<id>-<utc>.requests.jsonl` has one row per scheduled request:
`sequence`, `level_name`, `offset_s` (when it was due), `actual_send_offset_ns`
and `completed_offset_ns` (monotonic, from the schedule start),
`scheduling_lag_ms`, `status` (`success`, `timeout` when the SDK's request
timeout expired, or `error:<ExceptionType>`), `attempt` (always 1),
`client_retry_count` and `in_flight_at_send`. Every metric is recomputed from
this file:
`score.py results/<id>-<utc>.json`, as for quality, detects the mode, re-reads
the rows, re-aggregates every metric through the same function the run used,
`performance_metrics(rows, rate, duration_s)`, and prints each one beside the
published value and the delta, with no key, no network and no dataset;
`--write` updates `metrics` and `comparison` in place.

### What the comparison means

Same method, different client. The encode records ran two workers of two
processes and the reranker records one worker of two; this tool runs one
process, so every performance run is `comparable: false` with the reason
`client processes 1 != 4` or `1 != 2`, listed after any flag that differs,
and `comparison_basis` reads `same method, different parameters: <reasons>`.
The schedule is the union of the record's per-process partitions, so the
offered load is the same, but one process carries a 4× or 2× higher
per-process rate, and the in-flight cap and the 20 ms lag gate bind sooner.
At 0.25 rps for 60 s there are 15 requests, so p95 = p99 = max, in the
records and here; `run.request_count` shows it. The records' client ran in
the same cloud region as the endpoint; from anywhere else the latencies
include the network path and the response size (a multivector response is
hundreds of kilobytes), so read the rate and success metrics first and the
latency deltas as a property of where the run was made. One level runs per
invocation, the record's first level unless `--rate` is given, and
`load_level_name` is derived from the effective rate as `fixed-<rate>-rps`
with the decimal point written as a hyphen. A change of execution revision
mid-run sets `comparable: false` and `valid: false`.

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
- no multi-process client, load ladder, Poisson arrivals, clock calibration
  or candidate generation;
- no cost, price or accounting fields;
- no recorded-evidence download: the inputs are the six reference files and
  the pinned public datasets;
- no PyPI distribution, and the flags are not a compatibility surface.
