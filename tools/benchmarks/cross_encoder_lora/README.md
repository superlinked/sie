# CrossEncoder runtime LoRA qualification

This probe measures the ordinary `CrossEncoderAdapter` and the explicit opt-in
`CrossEncoderLoraAdapter` using the real SIE loader, worker, loopback HTTP API
and `sie_sdk.SIEClient`. It does not provision compute,
install dependencies, change precision, or change the production catalog. The
catalog model is overridden locally to use the ordinary adapter rather than the
catalog's BertFlash implementation.

The fixed checkpoint is `cross-encoder/ms-marco-MiniLM-L-6-v2` at
`c5ee24cb16019beea0893ab7796b1df96625c6b8`; `model.safetensors` must hash to
`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`.
Fixtures A/B/C are deterministic, untrained rank-4, alpha-8, bias-free LoRAs on
every BERT encoder query/value projection. They establish reversible numerical
effects and lifecycle behavior, **not retrieval quality**. Source seeds, model
files and adapter files are hashed in the fixture manifest.

## Explicit opt-in adapter

Ordinary profiles keep `sie_server.adapters.cross_encoder:CrossEncoderAdapter`.
That module is restored exactly to the pinned baseline source, including its
constructor, load/unload and inherited LoRA hooks. It does not import the
runtime-LoRA module or construct a manager. The public catalog and fallback
paths are unchanged.

Experimental runtime profiles must select the separate class explicitly:

```yaml
profiles:
  default:
    adapter_path: sie_server.adapters.cross_encoder.runtime_lora:CrossEncoderLoraAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        lora_paths:
          task_adapter:
            id: example/bert-encoder-lora
            revision: 0123456789abcdef0123456789abcdef01234567
```

The repository and revision above are placeholders. Use declared, compatible
adapter weights. The explicit class requires nonempty declarations; revisions
alone do not enable it. Older experimental PR775 configurations that added
`lora_paths` to the ordinary class must change their `adapter_path`; the existing
loader rejects those options on the ordinary class. This explicit adapter admits
the exact standard, unquantized `BertForSequenceClassification` and
`XLMRobertaForSequenceClassification` classes, with targets restricted to their
encoder Linear modules. Native BertFlash/default catalog profiles, custom model
classes and adapters containing saved classifier heads remain unsupported.
XLM-R profiles must also select the explicit class. The fixed benchmark below
still measures only the named BERT checkpoint; its results do not qualify XLM-R
performance or trained-adapter quality.

The opt-in class inherits the ordinary scoring methods. With base selected, its
manager restores the original module objects and retains LoRA layers in a plain
off-tree dictionary, without copying the base weights. Repeated base selection
still checks manager readiness and active state; declared profiles retain their
existing admission, reservation and mutation controls. These structural facts
do not prove equal end-to-end latency or throughput for enabled profiles. All
previous failed measurements and performance gates remain in force; the
baseline-only pilot is not candidate qualification.

## Prepare immutable sources and fixtures

From a clean public checkout with the candidate committed:

```console
mise tasks
mise exec -- uv run --no-sync python -m tools.benchmarks.cross_encoder_lora.prepare --candidate HEAD --output /tmp/ce-sources
mise exec -- uv run --no-sync python -m tools.benchmarks.cross_encoder_lora.fixtures --output /tmp/ce-fixtures
```

The preparer resolves refs once, archives baseline
`d5e024ad3ba6127b1ee176e2c85c794b76077888` and the candidate by full commit ID,
and refuses differing dependency locks. Keep the archives with their manifest.
The same benchmark code observes every arm; `PYTHONPATH` selects the archived
server and SDK. No arm runs a dependency sync. Pass `--snapshot` to both fixture
preparation and the runner when using a retained public checkpoint directory;
the worker checks its full recorded file manifest before loading it.

Prepare one Linux L4 environment using the public lock. The qualification
preflight requires Torch `2.9.1+cu129`, Transformers `4.57.6`,
SentenceTransformers `5.4.1`, PEFT `0.19.1`, and **no installed `torchao`**.
Four Torch intra-op threads and one inter-op thread are fixed. Default CUDA
precision remains unchanged (fp16 for this adapter). Record any environment
preparation separately, including installed versions before and after; the
benchmark only inspects them and fails on a mismatch.

```console
mise exec -- uv run --no-sync python -m tools.benchmarks.cross_encoder_lora.runner \
  --inputs /tmp/ce-sources --fixtures /tmp/ce-fixtures \
  --output /tmp/ce-evidence --deadline-s 3900 --trial-deadline-s 270
```

Use a fresh output directory. Set the outer compute-session deadline before
launching, allowing bounded startup and cleanup. The runner owns and joins only
its subprocesses; it cannot terminate an external GPU session. The operator
must stop that exact session and record cleanup. A failed or incomplete trial
stops the experiment and retains its evidence. Each new trial requires strictly
more than its 270-second limit plus 90 seconds for wait, termination and kill
grace (45/35/10). The proposed outer limits are 4,500 seconds for the operator
and 4,800 seconds from provider start, with a 300-second startup cutoff and an
additional absolute ceiling of application creation plus 5,100 seconds. The
operator reserves 300 seconds for export and requires strictly more than 4,255
seconds remaining before starting the 3,900-second runner (plus 45 seconds wait
and 10 seconds cleanup). The public runner's 4,200-second ceiling remains
available, but is not this operator preset. These settings do not themselves
authorize or allocate compute. Never change thresholds or
drop failed trials to obtain a pass, or retry the same failed protocol unchanged.

## Frozen comparison

There are six fresh-process blocks, in order `BNR`, `NRB`, `RBN`, `BRN`, `RNB`,
`NBR` (18 fresh servers). B uses the pinned baseline with no declared LoRA.
N uses the candidate's ordinary class with no declared LoRA. R uses its explicit
runtime-LoRA class after A/B
activity, with base selected and two adapters resident but disabled.

Every arm uses the same password-reset query and four documents in `protocol.py`.
Each phase starts with 30 sequential warmup requests. C1 measures 2,000 sequential
requests; C4 and C16 each run a 60-second closed loop with one SDK client per
lane. Throughput includes draining requests admitted before the deadline.
Latencies use the nearest-rank p95. Neither warmup nor correctness calls enter
the primary statistics. Concurrent futures have a 120-second wait limit: the
60-second measurement, the existing 30-second request timeout, and 30 seconds
of scheduling margin. The worker and outer process limits still bound failures.

All six metrics are required for **both** N/B and R/B: C1 mean and p95 latency,
C4/C16 requests per second and p95 latency. The statistical unit is the six
paired blocks. Latency ratios are candidate/B; throughput ratios are B/candidate.
For log ratios, the one-sided 95% t upper bound (df=5, critical
`2.0150483733330233`) must be at most 1.03, the geometric point ratio at most
1.01, and every paired ratio at most 1.10. A wide interval is inconclusive.

Before selection, three SDK base results must be finite and stable at absolute
tolerance `1e-6`. The public fp32 checkpoint oracle has a separately fixed `0.1`
CUDA-fp16 compatibility bound. Every ordered batch shape of 4, 8, …, 64 pairs
has two preselection oracle runs, stable at `1e-6`. Actual batches must match
the corresponding ordered-workload fingerprint and shape; unexpected partial
requests or ordering fail qualification. All candidate base oracles and observed
untimed SDK replay outputs must match the corresponding B-block oracle at `1e-6`. This is
deliberately strict and is never relaxed after measurement.

R also checks cold `503 LORA_LOADING`, bounded fresh SDK retries, duplicate-load
single-flight, undeclared-source rejection before fetch, distinct A/base/B
effects, base restoration, real concurrent base/A/B isolation and same-selection
cross-request batching. It records actual PEFT selection at model forward
boundaries. Mutation cannot overlap a score forward. Base tensors and classifier,
tokenizer and outer-wrapper identities remain unchanged. Churn exceeds the
two-adapter capacity; final removal restores the original Linear objects,
module tree and PEFT flags; same-instance reload and full model unload follow.
Production fault-injection tests separately cover malformed artifacts and
partial-attachment rollback; this probe does not pretend to reproduce them on
every GPU trial.

The untimed observer wraps the model's bound `forward` method, because Sentence
Transformers calls that method directly and bypasses root module forward hooks.
Every observed outer score batch must have nonempty forward records, finite
logits and exact input-row coverage; selected/base tuner checks cannot pass on
an empty list. It records selection/logit details during correctness checks and
two observed SDK replay rounds at C1/C4/C16. During timed phases the original
forward attribute lookup is restored: an original instance override is restored
exactly, or the temporary instance attribute is deleted to restore descriptor
lookup. Production `score_pairs` and `set_active_lora` are bound directly.
The harness checks the original forward target and that no observer activity
occurred, including restoration after an exceptional exit. Successful response
status and timing headers are captured as small metadata snapshots in the HTTP
observer, without retaining the successful HTTPX response. Error responses stay
available for deferred body and retry evidence. Validation happens after the
phase clock stops; exactly one successful HTTP response is required per
warmup/measured SDK call, excluding hidden retries. The timed client retains
timer, result-list and snapshot overhead; this is an SDK workload measurement.

Actual timed C4/C16 batch shapes are **unknown**. Their raw four-score vectors
are compared diagnostically against complete preselection oracle vectors at
`1e-6`. Each response records one deterministic matching oracle reference, or
null if none matches. This witness does not identify the actual shape or imply
a unique match: the retained raw vector and complete oracle corpus allow every
match and its ambiguity to be recomputed. A diagnostic cannot combine
different oracle shapes per element or substitute for the separately observed
shape-qualified replays. Timed C1 has one outstanding four-document request and
must still match its preselection result at `1e-6`. This measurement is scoped to
the observed workload and environment; it does not establish zero overhead or
support for other CrossEncoder architectures.

After synchronization and GC, unadapted allocated GPU bytes must match B.
R's additional allocation is bounded by measured unique adapter storage bytes
rounded up to 512 bytes per storage. Final adapter removal must restore the
initial allocated bytes. Reserved allocator memory, RSS, load duration and
GPU clock/temperature/utilization are recorded separately, not conflated with
live parameter allocation. The memory reference is captured after finite
preselection base and shape-oracle warmup, before any LoRA attachment, to keep
one-time CUDA library initialization separate from adapter storage.

Removing timed forward observation was an earlier **preallocation protocol
amendment**. It preserved that version's trial order, workload and gates, and
withdrew its timed batch-attribution claim. Observed untimed replays supply that
evidence instead.

## Longer-window amendment

The original qualification used 200 C1 requests and 12-second C4/C16 phases.
Its failed full-run evidence and the failed baseline-only stability diagnostic
remain separate, with their original source and artifact hashes. Those outcomes
are not reinterpreted as passes. The diagnostic's longer warmup did not establish
stability, so qualification retains 30 sequential warmup requests.

The revised workload uses 2,000 C1 requests and 60-second concurrent phases to
observe longer intervals. The six balanced fresh-process triplets, checkpoint,
precision, dependencies, request payloads, numerical tolerances and all twelve
statistical gates stay fixed. Longer windows may reduce within-window sampling
variation; they do not guarantee comparability across fresh processes. There
are still only six independent paired units, not thousands of independent
latency samples for the t interval.

A model-free lifetime probe separately established that the old observer held
successful responses for the whole phase and kept its client alive through a
bound callback. Metadata snapshots and removal of the owned hook/history on
close address that benchmark-owned retention. Normal HTTPX response-stream and
SDK namespace cycles remain, with automatic GC enabled. This finding does not
identify the cause of earlier GPU timing variation. Snapshot capture adds timed
metadata work; its CPU cost must be recorded, without subtracting it from the
measurements or claiming zero overhead. That version shared one client/server process and included their normal GC and
GIL interactions; the subsequent isolation amendment below preserves its failed
result rather than reclassifying it.

Freeze the reviewed source and operator bounds before one complete amended
18-process run. A separate baseline-readiness allocation is not required. Every
N/B and R/B gate and every correctness/lifecycle check must pass. If any fails,
retain all evidence and investigate before another method change; do not retry
unchanged, select favorable windows or pairs, waive N failures, or loosen gates.

The optional `stability` diagnostic keeps its original 200-request/12-second
windows and 90-second future wait through scoped settings that are restored on
success and failure. It records the current observer mode and worker source
hash. A new invocation uses this checkout's snapshot observer and is **not** a
byte-identical replay of the earlier diagnostic. Exact historical replay uses
commit `482b599224920a97bb2fce4a740317efb89351ab` and its original protocol.

## SDK process isolation amendment

The complete longer-window run at `a2932d81ef0894d3270a26350a2cd0703fa66a82`
failed five of twelve upper-confidence gates. Its point estimates did not
establish a serving slowdown, but they also did not satisfy the fixed precision
requirement. No source defect or physical cause was established. The original
records and criteria remain unchanged.

Qualification now starts one fresh Python **exec** load generator per server
trial and reuses it across C1/C4/C16. It imports only the selected archive's SDK
and benchmark client/telemetry helpers, never Torch, PEFT or the serving model.
The server retains its original process, CPU allocation, affinity and model
state. The SDK, HTTP, serving work, normal GC and every fixed workload/gate
remain included. Separation removes direct client/server interpreter and GIL
coupling; it does not remove shared CPU competition or guarantee precise results.
The historical `stability` command retains its in-process diagnostic method.

Client startup, warmup, boundary handshakes, process shutdown and evidence serialization
are outside the request timers. The same lane loop releases a barrier after
recording its start and drains every admitted request before the end timestamp.
The parent waits asynchronously so the server event loop continues. A single
stdin reader monitors parent lifetime; unexpected EOF ends the owned child.
Normal shutdown wakes and joins that reader, writes final terminal evidence,
then reports completion; the parent reaps the child before closing stdin. No new process group or detached session is
created. Cancellation, malformed protocol and child errors fail the trial and
trigger bounded termination/reaping. SIGKILL or immediate parent-death exit can
lose the interrupted phase; prior completed atomic phase files remain evidence.

### Boundary-only observation amendment

The first isolated run at `250bd9a91c8f465aafe69de94aee98aad6ac3298` completed
the first baseline trial's three SDK phases, then failed its prospective
combined GC-callback cost limit. Its C16 estimate was **0.183546%** against
**0.1%**: measured inner callback self-time contributed **0.055258%** and the
fixed 10-microsecond allowance contributed **0.128287%**. That failed run remains
failed; no candidate trials or paired performance gates were qualified. The
allowance was not reduced after observing the result.

The qualification harness now adds **no per-event GC callbacks** in either
PID; pre-existing callbacks are preserved. Outside-clock snapshots retain normal GC enable state, thresholds,
collection counters and ordered callback identities. The original callback
objects stay referenced through both boundaries, and changes to their identities
or GC configuration fail the phase. These checks cannot detect an intervening
change undone before the next snapshot. No automatic collection is disabled,
and no forced collection is added during timing.

Actual GC pause intervals and callback self-time/cost are explicitly **unavailable**
(`null`), not measured zero; the absent observer's cost gate is `not_applicable`,
not passed. Boundary snapshots still have overhead outside the request clocks;
this is not a zero-overhead claim. The SDK process separation, workload,
statistical gates, numerical and memory limits, wire checks and process cleanup
remain unchanged. The historical recorder and calibration remain standalone
diagnostics and are not prerequisites for a run that does not install callbacks.
To run that optional diagnostic from a frozen checkout, retain its full JSON,
including failures and outliers:

```console
mise exec -- uv run --no-sync python -m tools.benchmarks.cross_encoder_lora.calibrate_gc --output /tmp/ce-gc-calibration.json
```

It estimates local mean callback cost, not worst-case event cost or another
host's cost. Exact replay of the instrumented method requires its recorded
`250bd9a9` checkout. Short CPU smoke checks evidence integrity but does not
qualify performance.

CPU/context-switch/cgroup snapshots bracket the actual shared monotonic interval;
they include boundary work and are not exact request-level or GPU-only timers.
Shared cgroup observations are labeled and must never be summed across PIDs.
Missing optional counters remain unavailable. Existing server correctness
histories are retained and counted; the pre-timing memory checkpoint keeps its
existing explicit GC and CUDA synchronization. No speculative history cleanup
or heap normalization is introduced. The child retains its request histories
until all phases end. The server ingests them only after the child is reaped.

Phase payloads, observed IPC clocks, PID/source identities, ordered terminal
records, lengths and hashes must agree. Raw requests, wire attempts and all
existing numerical checks remain available. Child observations and phase files
live under each trial's `loadgen/` directory, with a separate process receipt and
server boundary snapshots. Keep those files in the full durable archive.

## Evidence and CPU checks

### Whole-phase triplet feasibility mode

The separate `interleaved` entry point holds three fresh server/SDK pairs resident
for one triplet. It prepares every pair before timing, then runs C1 for all three
roles, C4 for all three, and C16 for all three, using the six existing balanced
role orders. Each arm keeps its contiguous 2,000-request C1 phase, 60-second
C4/C16 admission windows, 30 warmups, drain, and original numerical, memory and
12 statistical limits. The sequential `runner` entry point remains available.

The `baseline-pilot` mode maps **every** logical B/N/R role to the same archived
baseline source and actual B configuration, with no adapter declarations. It
retains the peer numerical and memory checks. Its schema-3 summary reports
`completed_valid` separately from `qualification_passed`, which is always false
for this descriptive mode. A successful diagnostic exit does not qualify a
candidate or authorize a subsequent run. The separately selected `qualification`
mode requires the original B/N/R configurations and source mapping; it cannot
silently substitute baseline-only R. Pilot and candidate observations must not
be pooled or appended until a gate passes.

From a clean frozen checkout, after the ordinary immutable source and fixture
preparation above, the prospective baseline-only invocation is:

```console
mise exec -- uv run --no-sync python -m tools.benchmarks.cross_encoder_lora.interleaved \
  --inputs /tmp/ce-sources --fixtures /tmp/ce-fixtures --output /tmp/ce-baseline-pilot \
  --mode baseline-pilot --deadline-s 3900 --group-work-s 600
```

These are bounds, not a promise that six groups fit. Strict group admission
reserves 600 seconds of work plus one shared 90-second cleanup interval. Each
worker also has a cumulative 270-second grant-to-ACK active budget, with a
120-second cap for each phase; explicit idle waits consume the group budget.
The smaller group cap can stop work before three full active budgets are spent.
The enclosing prospective operator keeps its 300-second archive reserve within
4,500 seconds, provider lifetime at most 4,800 seconds, and outer bound 5,100
seconds. This entry point does not allocate resources or extend those bounds.

A dedicated socket carries immutable mode, role, actual configuration, source,
group deadline and lease identities; stdout/stderr remain ordinary logs. One
lease covers warmup, measurement, complete drain, atomic phase persistence and
an outside-timer CUDA synchronization with no pending/in-flight model requests.
Inactive workers wait on events, without a benchmark polling loop. All nine
phase ACKs precede any finish grant, coordinator/server result materialization, lifecycle
churn, unload or archive work. Pre-existing server housekeeping and three idle
CUDA contexts remain part of this changed resource condition; single-active
inference does not establish their neutrality.

The coordinator owns each server process group; its SDK child inherits that
group. The dedicated EOF monitor starts before heavy server imports, explicitly
cancels the experiment even under uvicorn's signal handler, and has a bounded
group-kill fallback. Group mode also watches unexpected idle SDK exit. Shared
cleanup signals every owned group before waiting and joins teardown through
repeated cancellation. A hard kill may prevent the interrupted phase from
being finalized; prior atomic files remain evidence, not a reconstructed pass.

This prospective method follows the failed complete `4eb6a0da` sequential
experiment. A separately retained CPU mechanics experiment verified scheduling,
cleanup and sensitivity to an injected arm delay, but only 6/12 baseline gates
passed under either schedule and 0/12 passed with the planted common drift.
Those controls do not demonstrate improved precision, explain the earlier
hardware result, or qualify L4 performance. Six triplets remain the statistical
units; shared-device observations are not independent hardware replications.

`summary.json` schema 2 retains trial outcomes, aggregate metrics, statistical
gates and source hashes. Each entry links its complete `trial.json` by relative
path and SHA-256; raw request arrays and oracle corpora stay in those per-trial
files instead of being copied into the growing summary. Final cross-process
checks reopen the complete records and verify their hashes and identities.
The runner records its monotonic start/deadline and every admission timestamp,
full cleanup reserve and remaining margin, including a refused admission.
The summary also hashes the complete benchmark file manifest. Each trial
captures the full installed-distribution manifest; its digest must remain
identical across arms. An optional `--environment-receipt` attaches a separate
operator preparation receipt and its SHA-256 (never include credentials).
Each trial contains incremental `events.jsonl`, atomic `trial.json`, generated
model configuration and a separate process log. It preserves raw request
scores, ranks, IDs, usage, latencies, errors, batch membership, logits, tensor
hashes, library versions and shutdown status. Failure paths retain completed
phases and partial request samples. Hash evidence files when archiving them;
the runner records each terminal trial file hash.

The compact representation follows an incomplete run whose stdout archive was
truncated. Its old evidence stays separate and is not acceptance evidence. This
change affects only post-measurement diagnostics and storage, not the timed
path, workload, numerical tolerances, statistical gates, or raw observations.
Store the evidence directory on durable storage and verify the full archive
hash after retrieval; stdout is unsuitable for transferring large artifacts.

```console
mise run test -- tools/benchmarks/cross_encoder_lora/tests
mise run typecheck -- tools/benchmarks/cross_encoder_lora
```

The CPU suite serves a tiny real BERT model through the actual loader and SDK,
uses real PEFT tensors, and checks correctness plus positive and negative
statistical/provenance controls. Its deliberately shorter phases and CPU
precision are labeled `qualification: false`; it is a harness smoke test,
not acceptance evidence for the pinned MiniLM checkpoint or L4 performance.
