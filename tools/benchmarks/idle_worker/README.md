# Idle-worker scheduling comparison

This benchmark compares two exact SIE source revisions on one NVIDIA GPU. It
uses `sie_sdk.SIEClient` against the local Python worker HTTP endpoint, records
the actual `X-Queue-Time` headers, and observes worker batch sizes. It targets
the Python worker scheduler changed in [PR #658](https://github.com/superlinked/sie/pull/658).
It does not measure the queue-only gateway/sidecar path, whose preformed batches
have a different scheduling owner.

The default workload uses the catalog-pinned `knowledgator/gliclass-large-v1.0`
checkpoint, four synthetic support requests, and five fixed classification
labels. Both arms use their checked-in default model profile and scheduling
configuration. The archive includes the server source, model catalog and bundle
definitions. Preparation refuses different dependency locks.

From an initialized checkout:

```sh
mise exec -- python tools/benchmarks/idle_worker/prepare.py \
  --baseline b0c26b02d21dbd146646c41b2b4f42627a0c9ebd \
  --candidate a7c024f6f53cce0f8e43825845c5ad6965c9c792
```

On the GPU machine, with the same locked workspace and prepared `_inputs/`:

```sh
mise exec -- python tools/benchmarks/idle_worker/run.py \
  --output /tmp/sie-idle-worker-results
```

The default order is baseline/candidate/candidate/baseline/baseline/candidate.
Each fresh server warms request concurrency 1, 8 and 32, then measures:

- 100 sequential requests, including every queue-time sample;
- four simultaneous HTTP bursts each at concurrency 8 and 32;
- 12 seconds of closed-loop traffic each at concurrency 8 and 32, including
  the final outstanding requests in elapsed time.

HTTP bursts can reach the worker at different times: distinguish an already
queued burst from the PR's intended first-request/tail split. The batch trace
records item IDs so the observed grouping is inspectable. A wrapper records a
batch only after normal dispatch completes; it does not replace scheduling or
inference. The existing worker unit tests cover deterministic queued-burst
formation.

Each completed phase is written before the next starts. If a later phase fails,
the trial is marked `failed` and its completed measurements remain available.
Trial JSON is replaced atomically so an incomplete write preserves the last
completed phase. Batch records are flushed as JSONL after each dispatch, so
SIGTERM during server shutdown cannot
discard them. If a trace is missing, request timings remain available and the
trial is marked `timing_complete_trace_missing`.
Reported p95 values use nearest rank: sort the samples and select rank
`ceil(0.95 * n)`, counting from one.

Use an otherwise idle GPU and reserve enough CPU for the client and server.
The server uses four Torch CPU threads. Keep the GPU allocation, dependencies,
input texts and warmup identical for both arms; compare all repeated trials
before interpreting a small throughput change. The output includes source and
archive hashes, runtime versions, GPU identity, raw sequential headers, HTTP
latencies, batch traces, and each server's log. Generated snapshots and results
are ignored by Git.
