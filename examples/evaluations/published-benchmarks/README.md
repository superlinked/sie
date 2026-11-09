# Repeat the published benchmarks or test a new model

This example does two things from one client process on your machine. It
repeats the method behind the benchmark numbers published on the SIE website,
six quality records and six fixed-rate performance records. And it measures a
model you are adding the same way, against an official number it looks up in
the public results repository or one you supply, before you open a pull
request: see [Test a new model](#test-a-new-model). It also times one request
at a time against your own server, beside a per-request time the website
shows: see [Check a time on the website](#check-a-time-on-the-website).

Each `references/<id>.json` is a projection of one published benchmark record:
the model and profile, the task key, the pinned dataset revision, the request
shape, the timing constants and the published metrics. `run.py quality` sends
the same requests through `sie_sdk.SIEClient`, ranks in the client, scores the
ranking with `pytrec_eval` and writes one record-shaped result with the delta
against every published metric. `run.py performance` sends a record's requests
at its published rate and aggregates their timing the way the record did. Both
print deltas. A quality run also prints a [verdict](#the-verdict) against a
baseline; a performance run has none. The verdict never changes the exit
status, and the example has no floor, target or gate.

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

Install the locked dependencies, then set the key. The default endpoint
requires it; the key is sent only to the endpoint it is scoped to, so against
your own server see
[Run it against your own server](#run-it-against-your-own-server):

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
required; `--output sparse|dense|multivector` and `--items-per-request` are
required for the encode tasks and refused for the score tasks, which score each
query against its task-native candidates. `--baseline`, `--baseline-source` and
`--instruction` belong to a run without a reference: see
[Test a new model](#test-a-new-model).

When a result is not valid, the command prints each reason to stderr.

A completed run writes two files under `results/`, which git ignores; a run
with a failed request writes the result record alone and names the failure
in `validity.reasons`:

- `results/<id>-<utc>.json`: the result record. It carries the reference id,
  `comparable` with its reasons, the model, profile and task, the dataset name
  and revision, the request shape, the observed `execution_revisions`, the
  run's request and failure counts, `validity` with the SDK retry count, every
  recomputed metric, a `comparison` row per published metric with the delta,
  the `baseline` and `verdict`, and `run.command`, the command that repeats the
  run. It is rebuilt from the plan, so credentials in a URL never reach the
  result.
- `results/<id>-<utc>.predictions.json`: the ranked list per query, the top
  1000 documents with their scores, or the reranker's candidates per query.
  Every metric is recomputed from this file.

### Recompute offline

```sh
uv run python score.py results/<id>-<utc>.json
```

`score.py` needs no API key and makes no inference call. It re-reads the
predictions file, loads the relevance judgements from the pinned dataset
revision named in the result with `datasets` (network or the local Hugging
Face cache), recomputes every metric and prints each one beside the published
value and the delta, then recomputes the verdict from the result's `baseline`
and those metrics. A result with no `baseline` field, written before the
verdict existed, prints no verdict. It rewrites nothing unless you pass
`--write`, which updates `metrics`, `comparison` and `verdict` in the result in
place.

### What is fixed by the method

These are not flags. They are the method behind the published numbers, and
the example applies them to every run:

- the whole evaluation split of the pinned dataset revision, `test` unless
  [Task keys](#task-keys) names another; queries with no relevance judgement
  are dropped, as MTEB does;
- one readiness request on the first batch, not recorded, waits up to 300 s
  for the model to load; a failure writes `valid: false` with the reason and
  exits 1;
- requests are sequential, one in flight, with no warmup beyond that
  readiness request;
- documents are encoded with `is_query=False` in batches of the reference's
  items per request, queries with `is_query=True`, and no instruction is sent
  unless `--instruction` gives one, which goes with queries only;
- sparse vectors are scored by the raw dot product, dense vectors by cosine
  over L2-normalised vectors, multivector outputs by MaxSim over L2-normalised
  token rows, and the reranker's own scores rank its candidates, sent in
  requests of at most 100 items for one query and merged. A `muvera`
  profile is refused with `--output dense` on a quality run, because the
  published method scores those vectors by the similarity the profile declares,
  usually the raw dot product, and this example scores dense only by cosine;
- the top 1000 documents per query are kept, ties broken by corpus order, and
  scored at k in 1, 3, 5, 10, 20, 100 and 1000 with pytrec_eval's `ndcg_cut`,
  `map_cut`, `recall` and `P`, rounded to five decimals; the published method
  leaves the order of equal scores at rank 1000 unspecified, so an `_at_1000`
  value can differ when scores tie there (by about 0.001 in `recall_at_1000`
  and 0.0002 in `ndcg_at_1000` in the validation below);
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

Every response from the default endpoint names the revision of the executed
model bundle and configuration. The run collects every distinct value into
`execution_revisions`; it is recorded, never compared with the reference's
checkpoint revision, and a change mid-run sets `comparable: false` with the
reason `execution revision changed`. A self-hosted sie-server reached
directly sends no revision, so `execution_revisions` stays empty and that
check cannot fire there.

A delta is a number. On a `--reference` run the [verdict](#the-verdict) reads
one number, `main_score`, against the 2% band the maintainers use when they decide whether
a published number still holds. It does not need `comparable: true`: a run
against another endpoint keeps `comparable: false` with its reason and still
gets a verdict.

### The verdict

After the metrics, a quality run prints one `verdict:` line. It compares one
metric with a baseline: the record's `main_score` for a `--reference` run,
the number you supply with `--baseline`, or the official number the tool looks
up when you pass neither. With a margin of 0.98, the low bound is the baseline
× 0.98 and the high bound is the baseline ÷ 0.98. The first row that matches
decides:

| Case | Verdict | Reason |
| --- | --- | --- |
| The run wrote no predictions | `FAIL (no-output)` | `no_predictions` |
| The metric is exactly 0 | `FAIL (no-output)` | `current_zero` |
| No baseline | `MEASUREMENT` | `no_baseline` |
| A task key with `limit` | `MEASUREMENT` | `limited_run` |
| Below the low bound | `FAIL (regression)` | `below_margin` |
| Above the high bound | `WARN (suspicious-improvement)` | `above_margin` |
| Otherwise | `PASS` | `within_margin` |

The line names the metric, its value, the baseline and where it comes from, the
difference in percent and the band. A `FAIL` also prints `expected:`, `got:`
and `reproduce:` lines on stdout. A `WARN` means the run scored above the band:
a real improvement, or a run that does not measure what the baseline measured.

One published reference reads `WARN` today. A full run of the Qwen3-Reranker-0.6B
quality record scores about 3% above its 0.59526, because that record predates the
server change that scores this reranker in float32 so its top candidates stop
tying. The 4B record stays inside the band.

The verdict is a report. The exit status does not follow it: the command exits
0 on a `FAIL (regression)` and 1 only when the run wrote no predictions. Nothing
it writes reaches the website or a published record, and it is no floor, target
or gate.

The result stores `baseline`, null when there is none, with `kind` (`published`,
`supplied` or `official`), `metric`, `value` and `source`, and `verdict` with
`verdict`, `reason`, `metric`, `current`, `baseline`, `delta_pct`,
`threshold_low`, `threshold_high` and `margin`. An `official` baseline also has
`model_revision`, `dataset_revision` and `mteb_version`.

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

A run against the default endpoint needs `SIE_API_KEY` and spends credits;
against your own server the key is sent only when `SIE_BASE_URL` names that
server (see [Run it against your own server](#run-it-against-your-own-server)).
A run takes about two minutes: one readiness request, 30 s of warmup, 10 s of
settle, 60 s of scheduled requests and up to 10 s of drain. When a result is
not valid, the command prints each reason to stderr. With `--reference`,
model, profile, task, output contract, items per request, timeout, in-flight
cap, settle, drain and lag tolerance come from the file; `--rate` defaults to
the record's published level, `--duration` to 60 and `--warmup` to 30,
`--endpoint` is accepted as for quality, and any value that differs from the
reference adds a `comparable_reasons` entry. Without a reference, `--model`,
`--task` and `--rate` are required, `--output` and `--items-per-request` are
required for the encode tasks and refused for the score tasks, and the timeout
and cap are 30 s and 64 for encode, 300 s and 128 for score. `--output` takes
`sparse`, `dense` or `multivector`. No other flags exist.

### What is fixed by the method

- a readiness request on the first row, `wait_for_capacity=True` with a 300 s
  provisioning timeout, before the measurement; a failure writes
  `valid: false` with the reason and exits 1; then 30 s of unrecorded
  closed-loop warmup on that row (`--warmup 0` skips the closed-loop warmup;
  the readiness request and the 10 s settle still run, so it is not a cold
  start) and 10 s of idle settle;
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
  after stripping whitespace; `role=corpus` takes the documents, otherwise
  the queries with relevance judgments in the task's split; the remaining rows
  are ordered by source id compared as strings and `limit=N` keeps the first N; for AskUbuntu, `limit=64` keeps the first
  64 queries in that order and `candidates-k=20` each query's 20 task-native
  candidates in source order; one request sends at most 100 candidates, so a
  key without `candidates-k` sends the first 100 and `candidates-k` above 100
  exits 2; the chosen ids go to `dataset.selected_ids` in that order;
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
and `validity`, and it has no `baseline` or `verdict`. The metrics use the
record's names and aggregations:
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
latency deltas in the light of where the run was made. One level runs per
invocation, the record's first level unless `--rate` is given, and
`load_level_name` is derived from the effective rate as `fixed-<rate>-rps`
with the decimal point written as a hyphen. A change of execution revision
mid-run sets `comparable: false` and `valid: false`.

## Run it against your own server

Both commands run against the open-source sie-server as well. Start it with
the reference's model, here the Splade NFCorpus references:

```sh
pip install "sie-server[local]"
sie-server serve --host 127.0.0.1 -m prithivida/Splade_PP_en_v2
```

By default the server runs without a token, and `--host 127.0.0.1` keeps it
on this machine: leave `SIE_API_KEY` unset. A key in your shell is scoped to
the endpoint `SIE_BASE_URL` names, or to the default endpoint when it is
unset, which is the SDK's rule, so a key for the default endpoint is never
sent to your own server; the command says so on stderr when it holds one
back. With `SIE_AUTH_MODE=token` the server generates a token, saves it to a
file and prints where; export that value in `SIE_API_KEY` together with
`SIE_BASE_URL=http://localhost:8080`.

Then pass `--endpoint http://localhost:8080`. A quality smoke run:

```sh
uv run python run.py quality \
  --reference references/prod-web-fw-20260905-splade-nfcorpus-quality-02.json \
  --task "mteb/NFCorpus?limit=20" \
  --endpoint http://localhost:8080
```

A short performance run:

```sh
uv run python run.py performance \
  --reference references/prod-web-fw-20260905-splade-nfcorpus-perf-01.json \
  --rate 1 --duration 20 \
  --endpoint http://localhost:8080
```

What to expect:

- the first request loads the model; the readiness request waits for it up
  to 300 s, so a cold run is valid. A first download and load that takes
  longer than that fails the readiness request, the run says so and exits 1;
  send one request by hand, or run again once the server has loaded the model;
- every result is `comparable: false` with the reason
  `endpoint http://localhost:8080 != https://api.superlinked.com`, listed
  beside any other reason;
- a full quality run (the reference's task key, no `?limit`) against the
  same pinned model revision should land in the same band as the published
  numbers, within numeric precision;
- the smoke run above scores 20 queries, so its metrics are a check that the
  path works, not a comparison with the record, and its verdict is
  `MEASUREMENT`;
- performance numbers describe your hardware and your network path and are
  not comparable with the record;
- `execution_revisions` stays empty, because a sie-server reached directly
  sends no revision header;
- `score.py` works the same on these results.

On an Apple Silicon laptop with the Splade NFCorpus quality smoke, the server
was ready in about half a minute, the cold model load took about 80 s, and a
20-query smoke took about 30 s once warm.

## Test a new model

Before you open a pull request that adds a model, measure it against an
official number.

1. **Add the model config.** Add a YAML file to `packages/sie_server/models/`
   in your checkout, modelled on a neighbour. Its `sie_id` is the name you pass
   to `--model`.
2. **Serve it** with sie-server, from the root of the checkout:

   ```sh
   mise run serve -- --host 127.0.0.1 -m your-org/your-model
   ```

   The same rules as in
   [Run it against your own server](#run-it-against-your-own-server) apply: no
   key unless the server runs in token mode, and the first request loads the
   model.
3. **Run it** from this directory. The tool first looks the official number up
   in the public results repository,
   [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results),
   and prints an `official:` line before any inference request:

   ```sh
   uv run python run.py quality \
     --model your-org/your-model \
     --task mteb/NFCorpus \
     --output dense \
     --items-per-request 32 \
     --endpoint http://localhost:8080
   ```

4. **If it finds none**, the `official:` line says why and the run reads
   `MEASUREMENT`. Find the number on the model card, in the paper or on a
   leaderboard and pass it. It must be for the same benchmark and the same
   metric: the same dataset and split, and the same metric at the same cut. A
   number for another dataset, or `ndcg_at_100` where you compare `ndcg_at_10`,
   is not a baseline. This example ships no official numbers.

   ```sh
   uv run python run.py quality \
     --model your-org/your-model \
     --task mteb/NFCorpus \
     --output dense \
     --items-per-request 32 \
     --baseline ndcg_at_10=<official value> \
     --baseline-source "<link to the model card, paper or leaderboard>" \
     --endpoint http://localhost:8080
   ```

5. **Read the verdict**, the `verdict:` line after the metrics, in the table
   under [The verdict](#the-verdict). On a `FAIL`, `reproduce:` is the command
   that repeats the run.
6. **Attach the result**, `results/<id>-<utc>.json`, to the pull request. It
   holds the baseline with its source, the verdict and `run.command`. Keep the
   `.predictions.json` beside it: `score.py` needs that file to recompute the
   verdict.

### The baseline

A run with neither `--reference` nor `--baseline` looks the number up. The
lookup sends the model id and the task name to GitHub, with no key or token, so
do not use it for a model id you want to keep private: pass `--baseline`
instead. It asks for the current commit of the repository's `main`, the model's
revision directories, and `<TaskName>.json` in each. It reads the task's main
metric from the one `default` entry of the task's split. When several directories
have the file, it prefers the revision the endpoint serves, then the highest
`mteb_version`.

It prints one line first, so you know before you spend anything:

```text
official: <metric> <value> from <link to the file>
official: none, <reason>; the run reads MEASUREMENT unless you pass --baseline
```

A `note:` line follows when the file's model revision differs from the one the
endpoint serves, or its `dataset_revision` differs from the pinned one. These
are notes only and the run still compares, because mismatched revisions usually
give the same score.

The lookup is skipped, with the reason on that line, for a key with `limit`, for
a two-stage key, for a profile other than the default, and for an output other
than `dense` on a model with more than one of dense, sparse and multivector, or
on a model whose outputs the endpoint does not report, because the number in the
repository is for the dense output. A key with `limit` stays `MEASUREMENT` even
with `--baseline`, so its line leaves that part out.

A network error, a rate limit or a bad response never blocks the run: it goes on
as `MEASUREMENT`. Older result files of the reranking tasks name their number
`map` and not `map_at_1000`, and the lookup finds nothing in them. Read the file
and pass `--baseline` if it is the number you want to compare with.

A run with no `--baseline` records `official_lookup` with `status` (`found`,
`not_found`, `failed` or `skipped`) and `detail`, and a found number is stored as
the `baseline`. `score.py` recomputes the verdict from that stored value and
looks nothing up. `run.command` gains no `--baseline`, so repeating it looks the
number up again.

`--baseline <metric>=<value>` and `--baseline-source <text>` go together and win
over the lookup, which then does not run. The metric is one a quality run
writes: `main_score`, `accuracy`, or `<family>_at_<k>` with family `ndcg`, `map`,
`recall`, `precision` or `mrr` and k one of 1, 3, 5, 10, 20, 100, 1000; a bare
family name is refused. The value is
a ratio, finite, above 0 and at most 1, so a leaderboard that prints percent
needs dividing by 100. A bad form exits 2 before any request, and both flags are
refused beside `--reference`, which brings its own baseline.

Without a baseline the run still writes every metric, and the verdict is
`MEASUREMENT`, unless the run produced nothing to compare.

### The instruction trap

Some models expect a task instruction in the query, and their official number
was measured with it. The tool cannot detect this, because the models endpoint
does not expose a query template. Read `query_template` in the model's config:

- when it has an `{instruction}` placeholder, pass the instruction the official
  number used with `--instruction "<text>"`. Without the flag the query carries
  only what the config supplies by default, so a low score can be this mismatch
  and not the model;
- otherwise leave the flag off. A model with no query template can get the text
  prepended to every query, which pollutes the query.

`--instruction` is sent with queries only, never with documents, and is recorded
in `payload.parameters`. It is refused beside `--reference`, on the score tasks
and on a two-stage key. `run.py performance` has no such flag, because its encode
requests set `is_query` to false.

### The limited-run rule

A task key with `limit`, such as `mteb/NFCorpus?limit=50`, scores part of the
queries, so its number cannot be compared with a full-dataset number. The
verdict is `MEASUREMENT` whatever the baseline says, unless the run produced
nothing to compare: `limited run, not compared with a full-dataset number`. Use it to check that the
path works, then run the key without `limit` to get a verdict. It still encodes
every document.

### Task keys

Each key is pinned to the dataset revision mteb 2.7.5 names; the revisions are
in `score.py`. All of them load anonymously from Hugging Face.

| Task key | Dataset | Split | Operation | `main_score` |
| --- | --- | --- | --- | --- |
| `mteb/SciFact` | `mteb/scifact` | `test` | encode | `ndcg_at_10` |
| `mteb/NFCorpus` | `mteb/nfcorpus` | `test` | encode | `ndcg_at_10` |
| `mteb/AskUbuntuDupQuestions` | `mteb/AskUbuntuDupQuestions` | `test` | score | `map_at_1000` |
| `mteb/FiQA2018` | `mteb/fiqa` | `test` | encode | `ndcg_at_10` |
| `mteb/SCIDOCS` | `mteb/scidocs` | `test` | encode | `ndcg_at_10` |
| `mteb/CQADupstackPhysicsRetrieval` | `mteb/cqadupstack-physics` | `test` | encode | `ndcg_at_10` |
| `mteb/CosQA` | `CoIR-Retrieval/cosqa` | `test` | encode | `ndcg_at_10` |
| `mteb/StackOverflowQA` | `CoIR-Retrieval/stackoverflow-qa` | `test` | encode | `ndcg_at_10` |
| `mteb/LegalBenchConsumerContractsQA` | `mteb/legalbench_consumer_contracts_qa` | `test` | encode | `ndcg_at_10` |
| `mteb/MMarcoReranking` | `mteb/MMarcoReranking` | `dev` | score | `map_at_1000` |
| `mteb/T2Reranking` | `mteb/T2Reranking` | `dev` | score | `map_at_1000` |
| `mteb/CMedQAv1-reranking` | `mteb/CMedQAv1-reranking` | `test` | score | `map_at_1000` |
| `mteb/CMedQAv2-reranking` | `mteb/CMedQAv2-reranking` | `test` | score | `map_at_1000` |
| `mteb/NanoFiQA2018Retrieval` | `zeta-alpha-ai/NanoFiQA2018` | `train` | encode | `ndcg_at_10` |

`mteb/CMedQAv1Reranking` and `mteb/CMedQAv2Reranking` are accepted as other
spellings of the two CMedQA keys; the result keeps the key you typed in
`task_key` and names the dataset in `dataset.name`.

`mteb/FiQA2018` sets mteb's `ignore_identical_ids`: a document whose id equals
the query's id is removed from that query's ranking after the top-1000 cut and
before scoring, as mteb does. The predictions file keeps it, and `score.py`
applies the same rule.

`run.py quality` takes a name alone or with `limit`, and an encode task also
takes a [two-stage key](#two-stage-keys); any other parameter exits 2 before any
request. `run.py performance` takes the same names with the parameters its
selection rule reads, for example
`mteb/FiQA2018?split=test&role=corpus&limit=256&filter=nonempty`; any other
parameter exits 2 before any request.

### Two-stage keys

A two-stage key scores a reranker over a retriever's top candidates:

```sh
uv run python run.py quality \
  --model your-org/your-reranker \
  --task "mteb/NFCorpus?candidates_model=Alibaba-NLP/gte-multilingual-base&candidates_k=50" \
  --endpoint http://localhost:8080
```

`candidates_model`, a plain `<org>/<name>` id, and `candidates_k`, a positive
integer, go together on an encode task, optionally with `limit`. A bad form exits
2 before any request. The method is fixed, so `--output`, `--items-per-request`
and `--instruction` are refused:

- one readiness request for each model;
- the whole corpus goes to the encoder in batches of 64 documents with
  `is_query=False`, then the queries with `is_query=True`, as dense vectors with
  no instruction;
- cosine over L2-normalised vectors picks the top k documents per query, ties
  broken by corpus order;
- the reranker scores each query's k documents, in requests of at most 100 items,
  and its scores rank them. Only those k documents are in the predictions.

The result names the encoder and k in `payload.parameters`, and
`execution_revisions` holds the revisions of both models. There is no official
number for such a pair, so the verdict is `MEASUREMENT` unless you pass
`--baseline`. `score.py` recomputes the metrics from the predictions file as for
any other key.

## Check a number on the website

A row on a model page maps to the flags below. Start every command with
`uv run python run.py quality` and add `--endpoint` for your own server. Without
a baseline the run compares with the official number when there is one. To
compare with the number in the row, add `--baseline <metric>=<value in the row>
--baseline-source "<link to the page>"`. A profile row and a two-stage row have
no official number, so they need it to get a verdict.

| Row on a model page | Flags |
| --- | --- |
| A plain row | `--model <model> --task mteb/NFCorpus --output dense --items-per-request 32` |
| A profile row | `--model <model>:<profile> --task mteb/NFCorpus --output <the profile's output> --items-per-request 32` |
| A reranker row | `--model <reranker> --task mteb/T2Reranking` |
| A row that says "reranking the top 50 from \<model\>" | `--model <reranker> --task "mteb/NFCorpus?candidates_model=<model>&candidates_k=50"` |
| Image retrieval, and non-MTEB tasks such as extraction, classification, OCR, detection, speech and generation | This example does not run these |

Some rows on the model pages are measured on our own GPUs, and a row may differ
from your run for that reason. Read the `verdict:` line with that in mind.

## Check a time on the website

The per-request times the website shows for SIE come from public studies in the
Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence).
Each study timed a model one request at a time, with the client inside the
server's container. `run.py latency` repeats that timing against your own
server for ten pages:

| `--page` | On the website | Model | Bundle | GPU it was timed on | Inputs | One request |
| --- | --- | --- | --- | --- | --- | --- |
| `classify` | `/classify`: 13.1 ms | `fastino/GLiNER2.5-Decide` | `transformers5` | RTX PRO 6000 | 400 CLINC150 and BANKING77 messages | `extract` with the message's 8 labels, `options={"overflow_policy": "error"}` |
| `search` | home page and `/about`: 15.7 ms | `Qwen/Qwen3-Embedding-4B` | `sglang-embedding` | RTX PRO 6000 | 440 questions | `encode` with `is_query=True` and `output_types=["dense"]`, no instruction |
| `named-entities` | industry pages, extract entities: 24 ms | `numind/NuNER_Zero-span` | `default` | RTX PRO 6000 | 652 sentences in 7 sets | `extract` with the set's labels, `options={"threshold": 0.5}` |
| `redact` | `/redact`: 61 ms | `superlinked/pii-redact` | `default` | RTX PRO 6000 | 660 documents | `extract` with the study's 39 labels |
| `typed-decisions` | `/typed-decisions`: 13.8 ms | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | `default` | RTX PRO 6000 | the first 200 test comments of the comment-spam set | `extract` with one label statement, `options={"hypothesis_template": "{}", "multi_label": true}` |
| `rerank` | `/rerank`: 55 ms | `Qwen/Qwen3-Reranker-4B:fast` | `default` | H100 | 499 questions with 20 candidates each | `score` with the study's instruction |
| `request-routing` | `/request-routing`: 8.5 ms | `Qwen/Qwen3-Reranker-4B:fast` | `default` | H100 | 200 requests | `score`: one request against one description, with the study's instruction |
| `ocr` | industry pages, OCR a page: 287 ms | `zai-org/GLM-OCR` | `sglang-vision-extract` | H100 | 372 photos (115 MB) | `extract` with the photo, instruction `Text Recognition:` |
| `chat` | `/chat`: 0.74 s | `Qwen/Qwen3.8-27B-FP8` | `sglang-cu130` | H100 | 330 turns of 55 conversations | a chat completion, 400 new tokens at most, temperature 0 |
| `structured-output` | industry pages, structured output: 938 ms | `Qwen/Qwen3.8-27B-FP8` | `sglang-cu130` | H100 | 100 Structured Output Benchmark records | a chat completion with a strict JSON schema, 2,048 new tokens at most, temperature 0 |

On a Linux machine with an NVIDIA GPU, start the server for the page with its
bundle image. For `classify`:

```sh
docker run --gpus all -p 127.0.0.1:8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  -e SIE_PRELOAD_MODELS=fastino/GLiNER2.5-Decide \
  ghcr.io/superlinked/sie-server:v0.11.0-cuda12-transformers5
```

Then, from this directory, on the same machine:

```sh
uv run python run.py latency --page classify --endpoint http://localhost:8080
```

For another page, stop that server and start the page's bundle with its model
in `SIE_PRELOAD_MODELS`. The images are
`ghcr.io/superlinked/sie-server:v0.11.0-cuda12-<bundle>`, except `sglang-cu130`,
which is `v0.11.0-cuda13-sglang-cu130`. For `rerank` and `request-routing`,
preload `Qwen/Qwen3-Reranker-4B:fast`. `chat` and `structured-output` need an
80 GB GPU, and their CUDA 13 image needs an NVIDIA driver from the 580 series
or newer.

`--endpoint` is required. The key rule is the one in
[Run it against your own server](#run-it-against-your-own-server): a server
without a token needs no key.

The run:

1. downloads the study's inputs at their pinned revisions, rebuilds the
   request list the way the study's build scripts wrote it, and stops before
   any request when its SHA-256 differs from the one the study published. The
   `redact` and `typed-decisions` studies published no such hash, so the check
   uses the hash of the rebuild, pinned in `latency.py`. `ocr` also downloads
   its 372 photos;
2. sends the study's warm-up requests, built from the end of that list, and
   does not time them: 20 for `classify`, `search` and `named-entities`, 10 for
   `redact`, 5 for `typed-decisions`, `rerank`, `request-routing` and `ocr`, 3
   for `chat` and 2 for `structured-output`. Where the server keeps a cache,
   a warm-up carries a `[<level>-<n>] ` text prefix or one changed pixel, as
   the study's did, so it never equals a timed request. The first warm-up
   waits up to 15 minutes for the model to load;
3. sends every input once, in order, one request in flight, one attempt, and
   times each `SIEClient` call. A chat completion is timed to the complete
   reply;
4. prints your n, p50, p95 and mean beside the published values, with
   percentiles interpolated linearly as the studies computed them, says how
   the published figure was made, and writes
   `results/latency-<page>-<utc>.json` with the time of every request.

The published figure is one run per GPU, except `rerank` and
`request-routing`, which are the median of three runs on fresh servers, and
`typed-decisions`, which is the median of five containers of three runs each.
Your run is one run. For `chat`, the website shows a run of 2026-10-13; the
same requests from the 2026-10-11 study are shown beside it.

Every run also prints what differs from the study:

- the study's client ran inside the server's container and sent each request
  over the sidecar's local Unix socket, with no HTTP. This run sends HTTP
  through `sie_sdk.SIEClient`, so your time also includes the HTTP server and
  the path from the client to the server. Run the client on the GPU machine to
  keep that path short;
- your GPU, and the server you started. The studies ran v0.10.0 images with a
  newer server package; the commands above start v0.11.0;
- for `chat` and `structured-output`, a chat completion on a standalone server
  goes to the SGLang server it runs, while the study's client used the
  sidecar's generation stream.

A failed request, or one that succeeded only after an SDK retry, is counted,
listed by status, and left out of the numbers. The run exits 1 only when the
inputs fail the check or no request succeeds on its first attempt. There is no
pass or fail: the numbers are set side by side.

`/detect` (82.2 ms) is not covered: its 300 photos are not in the dataset, and
fetching them from their original archives means streaming about 11.5 GB.

## Validation

`validation/` holds the example's own output from 2026-11-03: three
consecutive runs of the Splade NFCorpus quality reference and three of the
GTE-ModernColBERT SciFact performance reference at `fixed-1-rps`, made from
one client outside the endpoint's region, as the result records the runs
wrote plus the performance runs' `requests.jsonl`. They are observed values,
not evidence of record, and no value in the tables below is a pass threshold
or quality floor. The delta range is the smallest to the largest `delta` the
three result files carry (observed minus published); the spread is the
largest minus the smallest of the three observed values, computed before
rounding. Quality keeps the tool's five decimals for ndcg, map, recall and
precision; mrr, which the tool and the record keep at full precision, is
rounded to five here. Milliseconds and rates are rounded to three here.

### `prod-web-fw-20260905-splade-nfcorpus-quality-02`

Each run was `comparable: true` and `valid: true`, 32 requests, one execution
revision. The table lists `main_score`, `accuracy` and the 1, 10, 100 and
1000 cuts of the five families; the 15 cuts left out, `_at_3`, `_at_5` and
`_at_20` of each family, behave like their neighbours: zero delta and zero
spread for ndcg, map, recall and precision, the same -0.00052 in the third
run for mrr.

| Metric | Published | Run 1 | Run 2 | Run 3 | Delta range | Spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `main_score` | 0.33056 | 0.33056 | 0.33056 | 0.33056 | 0.00000 | 0.00000 |
| `accuracy` | 0.05531 | 0.05531 | 0.05531 | 0.05531 | 0.00000 | 0.00000 |
| `ndcg_at_1` | 0.45046 | 0.45046 | 0.45046 | 0.45046 | 0.00000 | 0.00000 |
| `ndcg_at_10` | 0.33056 | 0.33056 | 0.33056 | 0.33056 | 0.00000 | 0.00000 |
| `ndcg_at_100` | 0.29223 | 0.29208 | 0.29223 | 0.29223 | -0.00015 to 0.00000 | 0.00015 |
| `ndcg_at_1000` | 0.37716 | 0.37697 | 0.37698 | 0.37698 | -0.00019 to -0.00018 | 0.00001 |
| `map_at_1` | 0.05531 | 0.05531 | 0.05531 | 0.05531 | 0.00000 | 0.00000 |
| `map_at_10` | 0.12606 | 0.12606 | 0.12606 | 0.12606 | 0.00000 | 0.00000 |
| `map_at_100` | 0.15290 | 0.15289 | 0.15290 | 0.15290 | -0.00001 to 0.00000 | 0.00001 |
| `map_at_1000` | 0.16569 | 0.16569 | 0.16569 | 0.16568 | -0.00001 to 0.00000 | 0.00001 |
| `recall_at_1` | 0.05531 | 0.05531 | 0.05531 | 0.05531 | 0.00000 | 0.00000 |
| `recall_at_10` | 0.15743 | 0.15743 | 0.15743 | 0.15743 | 0.00000 | 0.00000 |
| `recall_at_100` | 0.27037 | 0.26975 | 0.27037 | 0.27037 | -0.00062 to 0.00000 | 0.00062 |
| `recall_at_1000` | 0.58995 | 0.58880 | 0.58883 | 0.58883 | -0.00115 to -0.00112 | 0.00003 |
| `precision_at_1` | 0.46749 | 0.46749 | 0.46749 | 0.46749 | 0.00000 | 0.00000 |
| `precision_at_10` | 0.23994 | 0.23994 | 0.23994 | 0.23994 | 0.00000 | 0.00000 |
| `precision_at_100` | 0.07437 | 0.07433 | 0.07437 | 0.07437 | -0.00004 to 0.00000 | 0.00004 |
| `precision_at_1000` | 0.01957 | 0.01956 | 0.01956 | 0.01956 | -0.00001 | 0.00000 |
| `mrr_at_1` | 0.47368 | 0.47368 | 0.47368 | 0.47368 | 0.00000 | 0.00000 |
| `mrr_at_10` | 0.54839 | 0.54839 | 0.54839 | 0.54788 | -0.00052 to 0.00000 | 0.00052 |
| `mrr_at_100` | 0.55378 | 0.55378 | 0.55378 | 0.55326 | -0.00052 to 0.00000 | 0.00052 |
| `mrr_at_1000` | 0.55433 | 0.55433 | 0.55433 | 0.55381 | -0.00052 to 0.00000 | 0.00052 |

The main score and every ndcg, map, recall and precision cut at 1 to 20 equal
the published value in all three runs. The sparse weights the service returns
differ slightly from run to run: three quarters to four fifths of the document
scores are identical between runs and the rest differ by about 0.001, at most
0.03, on scores that run from 0 to about 31 (each query's top score lies
between 3 and 31), so neighbours up to about 0.02 apart swap places; that
moves the 100 and 1000 cuts and `mrr_at_3` to `mrr_at_1000` by up to 0.00062
between runs. `ndcg_at_1000` and `recall_at_1000` sit about 0.0002 and 0.0011
below the published value in every run, at the cut where the order of equal
scores is unspecified.

### `prod-web-fw-20260905-gte-moderncolbert-scifact-perf-v2-r1`, `fixed-1-rps`

Each run was `valid: true`: 60 requests, no failure, timeout or SDK retry, an
in-flight peak of 1 or 2, one execution revision, and `comparable: false` for
`client processes 1 != 4`, as for every performance run of this tool.

| Metric | Published | Run 1 | Run 2 | Run 3 | Delta range | Spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `offered_requests_per_second` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `achieved_requests_per_second` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `eventual_achieved_requests_per_second` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `in_window_success_rate` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `in_window_retry_free_success_rate` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `success_rate` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `retry_free_success_rate` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `scheduling_lag_p99_ms` | 5.886 | 10.294 | 9.626 | 14.316 | 3.740 to 8.430 | 4.690 |
| `latency_p50_ms` | 178.819 | 444.978 | 460.266 | 464.829 | 266.159 to 286.010 | 19.850 |
| `latency_p95_ms` | 198.758 | 642.293 | 654.308 | 637.308 | 438.550 to 455.550 | 17.000 |
| `latency_p99_ms` | 285.029 | 948.867 | 1121.944 | 890.516 | 605.487 to 836.915 | 231.428 |

The rate and success metrics equal the published values in every run. The
latency deltas are consistent with the client's location: one process instead
of four does not change a request's own latency at this rate (the in-flight
peak was 1 or 2), and the probe below measures the network share for one
document, not for the 60-request runs. The probe, from the same machine,
split three warm encode requests of a
2041-character document into 268 to 284 ms to the first response header, of
which one network round trip, the TLS handshake, took 163 ms, which leaves
an estimated 105 to 121 ms for the service's own answer, and 485 to 561 ms
of response transfer for the document's multivector (323 ms on the cold
connection). The probe
illustrates the transport share of one request; it does not decompose or
bound the runs' percentiles, since time to the first header and transfer time
vary from request to request. The p99, one request in 60, moved by 231 ms
between consecutive runs; the lag p99 of 9.6 to 14.3 ms stayed under the
20 ms tolerance. Read
`achieved_requests_per_second`, the four success rates and
`scheduling_lag_p99_ms` first.

`SIEClient.get_model` exposes a model's `max_sequence_length` (8192 for
GTE-ModernColBERT-v1, 512 for Splade_PP_en_v2), its weights `revision`, its
profiles and capabilities, and not the serving truncation or a default
instruction. Those two settings stay unverified here: the agreement of the
Splade NFCorpus main score and 1 to 20 cuts above is consistent with the
record's serving configuration for that model, and this directory checks no
quality metric of GTE-ModernColBERT.

`uv run python score.py validation/<id>-<utc>.json` re-aggregates a committed
performance result: `run.requests_path` names the `results/` file the run
wrote, and when that path does not exist `score.py` takes the file of the
same name beside the result. The quality results cannot be re-derived from
this directory: their predictions files, about 9.5 MB each, are not
committed, and `score.py` names the missing one; only repeating the quality
run re-derives them.

## Verify

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The tests use a fake client, in-memory datasets and the committed
`validation/` files; they need no key and no network.

## Not in this example

- no publication path: nothing here writes to the website or to the published
  records, and no floor, target or gate;
- no `mteb` dependency, no nAUC or cv_recall metrics, and no match of the
  scorer's implementation digest;
- no multi-process client, load ladder, Poisson arrivals or clock calibration;
- no cost, price or accounting fields;
- no recorded-evidence download beyond what `run.py latency` reads: the
  quality and performance inputs are the six reference files and the pinned
  public datasets, and `run.py latency` reads the per-request time studies'
  inputs and published results at pinned revisions;
- no PyPI distribution, and the flags are not a compatibility surface.
