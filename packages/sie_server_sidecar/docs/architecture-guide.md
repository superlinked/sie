# SIE Server Sidecar Architecture

`packages/sie_server_sidecar` contains the Rust queue runtime used by
sidecar-enabled SIE worker pods.

The sidecar runs next to a SIE backend in the same worker pod. It consumes NATS
JetStream work, talks to the backend over Unix-domain-socket IPC, and publishes
results back to NATS Core.

```text
gateway -> NATS JetStream -> worker-sidecar -> UDS IPC -> worker backend
                                      |
                                      +-> NATS Core reply subject
```

References:

- [Gateway architecture guide](../../sie_gateway/docs/architecture-guide.md)
- [Sidecar README](../README.md)

## Component Boundaries

`sie_gateway` accepts client inference requests, resolves the model, bundle,
machine profile, and pool, publishes work to JetStream, collects worker results,
and owns DLQ publication from JetStream max-delivery advisories.

`worker-sidecar` owns JetStream consumption, subject validation, payload fetch,
batch formation, adaptive scheduling, Rust tokenization when enabled by adapter
descriptors, result framing, ACK/NAK decisions, reply publication, HTTP probes,
single-emission OTLP telemetry, NATS health publication, pool admission, and
bundle-scoped config apply.

`sie-server` owns model registry behavior, lazy model loading, GPU residency,
adapter execution, adapter-specific preprocessing and postprocessing, generation
execution, and the IPC server.

The sidecar does not load model weights and does not link GPU libraries.

Public names:

- Rust package and binary: `sie-server-sidecar`
- Kubernetes container: `worker-sidecar`
- Container image: `ghcr.io/superlinked/sie-server-sidecar`
- Metrics contract: dotted `sie.worker.*` OpenTelemetry instruments

## Request Flow

Encode, score, and extract queue traffic follows this path:

1. The gateway publishes a msgpack `WorkItem` to a lane-aware JetStream subject.
2. The sidecar consumes the message from the matching durable consumer.
3. The sidecar validates the NATS subject and reply subject.
4. The sidecar resolves payload references when the work item uses external
   payload storage.
5. The sidecar groups work by model.
6. The sidecar calls `EnsureModelReady` over IPC.
7. The scheduler batches compatible items by operation and LoRA key.
8. The sidecar sends scheduled batches over IPC with `RunBatch`.
9. The backend returns per-item outcomes.
10. The sidecar publishes a `WorkResult`, or a negotiated bounded sequence of
    `ResultChunkV1` envelopes, to the reply subject.
11. The sidecar ACKs the JetStream message only after the complete reply (all
    chunks plus one final NATS flush, when chunking is used) has published
    successfully.

Message settlement:

- Unsafe reply subjects are ACK-dropped.
- Malformed subjects and bad msgpack payloads are NAKed.
- Unknown operations publish per-item error outcomes when reply publication
  succeeds.
- A `load` work item carries no input. It goes through the same admission,
  config and readiness checks as other work for that model. Once
  `EnsureModelReady` reports ready, it is ACKed without inference or a result.
  This is the NATS worker-side warm-up primitive; the gateway publishes one
  before a `model_loading` bridge. Local-ingest `publish_work` rejects `load` before
  dispatch because that request/response lane requires a result.
- The gateway sets `fallback_reason` on a remote attempt that stands in for a
  local refusal it holds: every fallback bridge, and a threshold bridge while
  local capacity wakes. If the backend returns `nak_retry` for such an item,
  the sidecar publishes an error result with the outcome's `error_code`
  (`QUEUE_FULL` when absent) and `retry_after_s` (the NAK delay rounded up to
  seconds when absent), then ACKs after successful publication, so the gateway
  restores its local refusal at once. Ordinary items retain NAK behavior.
  Admission, config/readiness barriers and failed result publication retain
  their existing retry behavior.
- An outcome's optional `retry_after_s` is published on the `WorkResult`
  unchanged. For a `QUEUE_FULL`, `MODEL_LOADING` or `RESOURCE_EXHAUSTED`
  answer the gateway uses a hint of 1 to 60 seconds as `Retry-After`; without
  one, or with a hint outside that range, it uses five seconds.
- Active local model loads are held with JetStream progress ACKs until
  `EnsureModelReady` returns ready; this preserves the delivery budget during
  cold starts. The progress delay is clamped below the pool consumer `ack_wait`
  even if the retry NAK delay is configured higher.
- Transient backend, decode, stream, and drain failures are NAKed.
- Reply publication failure, including a partial chunk sequence, leaves the
  JetStream message unacked for redelivery.

Work-item deadlines:

- The gateway stamps non-streaming work items with an optional `deadline`, an
  absolute Unix time in seconds on the clock that stamps `timestamp`. An item
  without one, with a non-numeric one, or whose `deadline - timestamp` is
  negative or larger than `SIE_WORK_DEADLINE_MAX_BUDGET_S` (default 180) keeps
  the behaviour that predates deadlines, and a numeric deadline ignored this
  way is reported with a rate-limited warning naming its budget. Keep
  `SIE_WORK_DEADLINE_MAX_BUDGET_S` at or above the gateway's
  `SIE_GATEWAY_REQUEST_TIMEOUT`; otherwise every item's deadline is ignored.
  Generation items are never judged by this field.
- The comparison is between the gateway and worker wall clocks plus
  `SIE_WORK_DEADLINE_SKEW_TOLERANCE_MS` (default 5000, at most 60000). Keep
  gateway and worker hosts synchronised, for example with NTP. The sidecar
  logs a rate-limited warning when an item's publish timestamp is ahead of its
  own clock by more than the tolerance, and when a first delivery is already
  past its deadline, which means it waited in the stream longer than its budget
  or the worker clock is ahead.
- By default a NATS delivery past its deadline still executes. It is counted as
  `sie.worker.work_item.deadline_exceeded{action="executed"}` and logged with a
  rate-limited warning before backend IPC. With
  `SIE_WORK_DEADLINE_ENFORCE=true` it is instead ACK-dropped at the same
  checkpoints as `work_cancel`, before payload fetch and backend IPC, and
  counted as `sie.worker.work_item.deadline_exceeded{action="dropped"}` (and as
  `sie.worker.nats.operations{operation="ack",reason="deadline_exceeded"}`).
  Run with the default first and alert on a sustained
  `sie_worker_work_item_deadline_exceeded_total` rate, which points at a
  backlog or at clock skew, before enabling enforcement. Local-ingest callers
  bound their own calls and are never dropped.
- Encode, score, and extract deliveries with a deadline hold a progress lease
  from intake: they are progress-ACKed at most about 10 s apart until they are
  ACKed, NAKed, or dropped, so a slow scheduler queue or backend call does not
  trigger a redelivery of work that is still running. Settlement waits for a
  progress ACK already in flight, so none follows the ACK or NAK. Progress
  pauses once an established backend heartbeat goes stale or the sidecar is
  draining, so JetStream can move the work to another worker; the pause and
  the resume are logged. Before the first successful heartbeat the backend
  state is unknown and progress continues. With enforcement on, the lease
  ends at the deadline and a later redelivery is dropped as expired; with
  enforcement off it lasts one more maximum budget past the deadline.
- A `RunBatch` call waits for the longer of `SIE_IPC_REQUEST_TIMEOUT_S` and
  the time until the batch's latest deadline, at most
  `SIE_WORK_DEADLINE_MAX_BUDGET_S`, so a legitimate slow batch is not cut short
  and retried.

Source: [`dispatcher.rs`](../src/dispatcher.rs),
[`publisher.rs`](../src/publisher.rs), and [`work_types.rs`](../src/work_types.rs).

## NATS Subjects

Pool work uses this subject shape:

```text
sie.work.{pool}.{machine_profile}.{bundle}.{model}
```

Generation direct dispatch uses this subject shape:

```text
sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}
```

The pool stream subject is:

```text
sie.work.{pool}.*.*.*
```

The pool durable filter is:

```text
sie.work.{pool}.{machine_profile}.{bundle}.*
```

The sidecar extracts the model from the sixth subject token. Model IDs, bundle
IDs, machine profiles, pool names, and worker IDs are normalized to single NATS
subject tokens by the gateway and sidecar helpers.

The sidecar creates one long-lived async-nats pull stream for pool work. A slow
reconciler re-ensures stream and durable metadata outside the fetch loop.

The sidecar also creates and polls the worker-specific direct-dispatch stream.
Generation uses that stream for HRW placement; capped logical batch pools use it
when the gateway must target one assigned worker on a shared backing queue.

For non-generation abandonment, every active sidecar subscribes to
`work_cancel.>` and records subjects shaped as
`work_cancel.{router_id}.{request_id}` in a bounded, namespaced tombstone set.
Lookups are constant-time and occur before model readiness, during readiness
waits, around offloaded-payload fetch, at scheduler admission, and immediately
before backend IPC. Matching encode, score, and extract deliveries are
ACK-dropped through one settlement path; generation is excluded because its
streaming cancellation contract uses `cancel.*`. The gateway's
`cancel.{router_id}.{request_id}` records the same tombstone, so encode, score,
and extract work for a request cancelled that way is also dropped before
IPC.

The greater of the work stream age and retry window is the active worker's
tombstone expiry horizon. A 100,000-entry process cap evicts the oldest
tombstone sooner under sustained cancellation volume. `work_cancel` is NATS
Core and therefore best effort across disconnects and restarts. JetStream
remains at-least-once, so an ACK failure can redeliver work, but a retained
tombstone filters that redelivery. Static inference already sent over backend
IPC is not preempted; the gateway has removed its collector and drops the late
result.

### Connection and permissions

The sidecar connects with the credentials in `SIE_NATS_USER` and
`SIE_NATS_PASSWORD`, which are read from the environment only (there are no
CLI flags for them, and `--help` hides the value of `SIE_NATS_URL`). Setting
only one of the two fails startup. Credentials in `SIE_NATS_URL` are not used,
and logs redact any userinfo in it. In the Helm chart only the sidecar
container of a worker pod receives the worker credentials, not the container
that runs model code.

The client uses the inbox prefix `_INBOX_WORKER` for its JetStream API replies
and pull deliveries. The chart's `sie-worker` user may subscribe to that
prefix, to `sie.config.models.*`, and to the three cancel subject trees, and
may publish results into the gateway's `_INBOX` subjects, heartbeats on
`sie.health.>`, acknowledgements on `$JS.ACK.>`, and the JetStream API calls
above: stream info, create, and update (the sidecar creates its
direct-dispatch stream and reconciles the pool stream), and consumer list,
info, create, delete, and pull. It cannot publish work, config deltas, or
cancels, subscribe to the gateway's inboxes, or delete or purge streams. Its
stream and consumer management still reaches other pools' streams, durables,
and the gateway's inboxes; the full matrix and what the worker user can still
do are in the chart README ("NATS authentication"). The worker pod mounts no
Kubernetes API token.

Because stream management lets the server deliver stored messages on a
worker's behalf past its publish permissions, the sidecar drops cancel
signals that carry a reply subject or a `Nats-` header, and drops (ACKs) work
that carries a `Nats-` header other than the gateway's `Nats-Msg-Id`. A
subject transform on the pool stream still adds work without such a header.

On the generation path the sidecar publishes a backend `publish` event only
when its reply subject equals the work item's `reply_subject`. The backend
runs model code, so it must not choose where the sidecar's NATS user
publishes.

Source: [`nats_consumer.rs`](../src/nats_consumer.rs),
[`subject.rs`](../src/subject.rs), and the gateway
[queue publisher](../../sie_gateway/src/queue/publisher.rs).

## Pool Admission

When `SIE_GATEWAY_URL` is configured and pool admission is enabled, the sidecar
polls `GET /v1/pools` before pulling from the pool stream or the
worker-specific direct-dispatch stream. It may pull when its physical `SIE_POOL` is
admitted directly, or when at least one assigned logical pool is backed by that
same queue; each work item is then checked against its `admission_pool` before
backend IPC. The gateway normally direct-dispatches capped logical batch work to
an assigned worker, so unassigned peers do not spend the item's JetStream
delivery budget. Work items rejected by this per-item logical gate are NAKed and
logged as a defensive backstop for stale assignments, direct NATS publishes,
and rolling-version mismatches.

Capped named pools require the worker to appear in
`status.assigned_workers`. The default pool continues pulling during transient
gateway/status errors.

Source: [`pool_admission.rs`](../src/pool_admission.rs).

## Generation

Generation work uses `ProcessGenerate` over IPC. The sidecar passes the
msgpack-encoded `WorkItem` to the backend and handles streaming IPC events.

The gateway streaming path selects either a pool subject or a worker-specific
subject. Worker-specific routing uses HRW over eligible workers. The gateway
stores a pool fallback subject for streaming work and uses it for republish
paths such as NAK handling and first-chunk timeout handling.

The sidecar subscribes to `cancel.>`. Subjects shaped as
`cancel.{router_id}.{request_id}` are forwarded to the backend through
`SignalGenerateCancel`.

Source: [`lib.rs`](../src/lib.rs), [`dispatcher.rs`](../src/dispatcher.rs),
[`ipc_client.rs`](../src/ipc_client.rs), and
[`packages/sie_gateway/src/handlers/proxy.rs`](../../sie_gateway/src/handlers/proxy.rs).

## Local Ingest

Brokerless deployments can select local ingest and send length-prefixed
MessagePack requests over an owner-only Unix socket. Non-generation work
enters the same dispatcher, Rust preparation, scheduler, admission, and IPC
pipeline as queue work. The request digest detects route and payload assembly
drift between trusted colocated processes; socket permissions provide access
control.

Generation uses the canonical v0.2 `publish_generate_stream` operation. Rust
normalizes legacy base64 images to bounded MessagePack binary, validates
request/attempt identity, contiguous chunk sequence, and terminal ordering,
and awaits every write through a count- and byte-bounded connection writer.
The local-ingest reader also enforces fixed in-flight operation and retained
request-byte budgets per connection. It acquires the declared frame bytes
before reading the body and stops reading while either budget is exhausted, so
slow socket writes cannot accumulate unbounded request bodies or tasks. Active
envelope IDs are unique until their terminal write completes; a duplicate
closes the connection rather than producing ambiguous multiplexed responses.
Active generation request IDs are also claimed process-wide because backend
cancellation is keyed by request ID alone.
Writes and cancellation drain have finite transport deadlines; they do not
override generation's gateway-owned semantic deadline (`timeout_ms=0`). A
`cancel` control request or client disconnect is forwarded through
`SignalGenerateCancel`. Deployments must map their outer transport onto this
contract rather than bypassing the sidecar for media or stream handling.

Source: [`local_ingest.rs`](../src/local_ingest.rs) and
[`dispatcher.rs`](../src/dispatcher.rs).

## IPC

IPC framing is:

```text
[4-byte big-endian length][msgpack named-map body]
```

The Rust and Python protocol copies define the same method names:

- `Ping`
- `EnsureModelReady`
- `ProcessEncodeBatch`
- `ProcessScoreBatch`
- `ProcessExtractBatch`
- `ProcessGenerate`
- `WorkerCapabilities`
- `NumericalProfileSnapshot`
- `SignalGenerateCancel`
- `RunBatch`
- `RunBatchWithExecutionAuthorityV1`
- `RunBatchWithNumericalAdmissionV1`
- `ProcessGenerateWithExecutionAuthorityV1`
- `ApplyModelConfig`
- `ReplaceModelConfigs`
- `Drain`

`tools/check_ipc_types_parity.py` checks the Rust protocol schema against
`packages/sie_server/src/sie_server/ipc_types.py`.

### Numerical process observations (#415)

`NumericalProfileSnapshot` is an optional, on-demand diagnostic RPC. The Python
backend returns its process incarnation, registered model contract digests, and
available local numerical profile identities under one configuration execution
lease. It does not load model weights or run inference. Unsupported profiles
retain an absent identity; failures return an incomplete snapshot without error
details. A roster contains at most 1,024 models, with each model ID limited to
1,024 UTF-8 bytes. Truncation or an invalid configuration makes it incomplete.

A process that holds the upstream of a model's remote profile also reports, on
the bare model, that profile's `remote_contract_sha256` and the
`remote_execution_sha256` of its serving code. For a model with `encode` or
`score` tasks and a hybrid routing policy it adds the current `admission`: the
local execution identities that passing, fresh evidence covers (at most eight),
the model contract, the admitted outputs, the expiry and an admission digest.
An SIE upstream's identity is refreshed in the background, so a snapshot never
waits on the upstream. Empty fields are omitted, so a process without a remote
profile reports the same shape as before.

The adapter pool retains one observation for every configured child, including
unavailable and legacy backends. Missing fields are incomplete, malformed
digests or admissions are invalid, and a duplicated process incarnation invalidates both
children. Each call refreshes the observations so a replacement process does
not inherit the previous incarnation. The Candle backend does not implement
this optional method and is reported as unavailable.

These observations do not change readiness, execution authority, or routing.
Admitting a numerical bridge from them also requires the identity of every live
local process and an execution fence at the remote process.

### Execution authority protocol amendment (#415)

Verified execution uses the two `WithExecutionAuthorityV1` methods. They
require a nonempty configuration hash for every batch item or generation work
item and retain the live Python execution lease described above. An old backend
rejects these unknown methods before inference. The backend capability
`supports_execution_authority_v1` defaults to false when absent; the Python
backend advertises support, while Candle remains closed until it implements
the same contract. Capability discovery cannot replace the method fence: a
backend child can restart after its last positive capability response.

Authority work uses the worker-direct subject with an eighth token,
`execution-authority-v1`. The sidecar creates `WORK_AUTHORITY_V1_<worker>` and
`authority-v1-<worker>` separately from its ordinary pool/direct stream and
consumer. The stream retains the existing bounded age, message and delivery
limits. Older six-token pool and seven-token direct filters cannot receive this
work or change the authority consumer when reconciling their own streams.
Redelivery preserves the subject and rechecks the live configuration hash.
Verified subject and payload model ids must match exactly before any subject
rewrite, readiness check or offloaded payload retrieval. This refuses lossy
`__`/`_dot_` normalization collisions instead of retargeting another model
covered by the same configuration hash.

The sidecar derives the required contract from that subject. Before readiness
or fetching offloaded inputs, it requires a nonempty matching hash, live config
authority, and positively verified backend support. Numeric work also requires
the complete scheduler lifecycle; its scheduler batches are partitioned by
execution contract so older empty-hash work cannot invalidate verified work.
Batch and generation dispatch use the updated-only IPC methods, preserving the
final locked execution checks. The sidecar's health capability requires every
ready backend child to report support and the authority pull consumer to run.
Missing or unavailable children close that admission signal. The method fence
still applies independently if a backend changes after the last health probe.

Current gateway producers use the existing queue subjects. Gateway activation
must select positively capable workers and publish only on the versioned
subject for verified work.
Every publish and retry must retain both fences, with no legacy pool or IPC
fallback. A missing, stale, unsupported, or unavailable authority refuses work
before inputs execute. This extends the queue contract while preserving the
gateway's queue-only ownership and worker-owned inference.

### Numerical admission fence (#415)

A remote attempt for an `encode` or `score` item may carry
`numerical_admission_sha256`, the digest of the remote process's `admission`
that the producer relied on. Such admitted work uses the worker-direct subject
with the eighth token `numerical-admission-v1`. Only a sidecar with this fence
creates `WORK_ADMISSION_V1_<worker>` and `admission-v1-<worker>`, with the same
bounds as the authority stream. No consumer of a sidecar without the fence
matches that subject, including its authority consumer. A rollback therefore
cannot hand such a sidecar admitted work that is already queued, and until a
fenced sidecar has created the stream, admitted work cannot be published at
all. The contract comes from the subject, never from the payload field, which
an older sidecar would ignore. Every worker therefore runs a fourth stream,
durable consumer and pull loop, including lanes that never receive admitted
work.

Admitted work also runs under execution authority. Before readiness or
offloaded input retrieval, the sidecar answers an item with `INFERENCE_ERROR`
and ACKs it, whether or not the answer could be published, when the item and
its subject disagree: the field on any other subject, an item without the field
on the admission subject, an operation other than `encode` or `score`, a
payload model other than the subject's, or an admitted item without a
configuration hash. It does the same for an `encode` or `score` item that
carries `fallback_reason` without the field, because every numerical remote
attempt that stands in for a local refusal is admitted work. Such an item can
never become valid, and a NAK could hand an item on a pool subject to a worker
without the fence. An admitted item whose configuration hash this worker does
not hold is also answered with `INFERENCE_ERROR` at once, at receipt, at the
model group's check and at the scheduler's barrier, because its bridged caller
waits with a local refusal. Only an admitted item that arrives before every
backend child supports the method is NAKed on the worker's own admission
stream, like other authority work. An admitted item past its gateway deadline
is ACK-dropped before the backend call even when `SIE_WORK_DEADLINE_ENFORCE` is
off, because its caller has stopped waiting. The local-ingest lane rejects the
field, and `fallback_reason` on `encode` or `score`.

Admitted items keep their own scheduler partition and run only through
`RunBatchWithNumericalAdmissionV1`, which an older backend rejects. A sidecar
without the scheduler answers them instead of sending them through
`ProcessEncodeBatch` or `ProcessScoreBatch`, and the Python backend refuses the
field on every method other than the admission method. Under the same
execution lease as the authority checks, the Python backend derives its own
current admission for the bare model whose fallback profile the batch targets,
without contacting an SIE upstream. It runs an item only when the item names
that digest, requests only admitted outputs and keeps the measured runtime
options. Any other item gets `nak_retry` with `INFERENCE_ERROR` before the
upstream is called, as does every item when the check itself fails or when the
configuration barrier refuses an admitted batch. Redelivery to the pinned
worker cannot restore an admission, so the sidecar publishes a `nak_retry` for
an admitted item as its refusal at once and ACKs it, as it does for a fallback
attempt. A bridged caller then receives its local refusal with
`X-SIE-Fallback-Error: INFERENCE_ERROR`.

The check covers batches that start after a change. A replaced, expired or
removed record refuses every later batch without waiting for the next
heartbeat, while a batch that has already passed the check completes its
upstream calls. An SIE upstream's identity comes from a cache that is trusted
for up to 30 seconds.

The backend capability `supports_numerical_admission_v1` defaults to false. The
heartbeat reports `supports_numerical_admission_v1` when it reports
`supports_execution_authority_v1` and every backend child supports the method.
It reports `supports_numerical_admission_subject_v1` only when, in addition,
the admission pull consumer runs. Builds that predate the admission subject
never send that field, so the gateway selects on it alone. A sidecar rolled
back below this version does not run admitted work still queued for that
worker, and its callers time out. A fenced sidecar that later consumes such an
item drops it once its deadline has passed. The gateway stops selecting the
worker once its heartbeat no longer reports the capability. The gateway
produces these items only for an `encode` or `score` bridge or threshold route
whose numerical admission it has checked, and only toward a worker that
reports `supports_numerical_admission_subject_v1`. The gateway architecture
guide's "Numerical bridges" section describes that decision.

Non-streaming backend responses use one physical frame while the serialized
response is at most 32 MiB. For a larger response, the sidecar explicitly sets
the top-level `accepts_ipc_response_chunks_v1` capability on the request. A
supporting Python or native Rust backend serializes the ordinary response once,
hashes those exact bytes with SHA-256, and returns ordered
`IpcResponseChunkV1` frames with payloads of at most 4 MiB. The sidecar verifies
the version, request identity, stable layout, indexes, total length, and digest
before decoding the reconstructed response. Chunk payloads may not exceed 4 MiB,
the complete serialized chunk frame may not exceed that ceiling plus 4 KiB, the
declared total must exceed the 32 MiB legacy-frame ceiling, and one transfer is
bounded to 128 MiB and 64 chunks. Across all pool and mux clients in the
sidecar process, active reassemblies share one conservative memory budget; each
transfer reserves twice its declared size plus 4 KiB and holds that reservation
through typed decoding. The payload and minimum-total bounds ensure that this
reservation also covers the raw physical frame and decoded payload briefly held
during assembly.
Missing/false capability and over-limit responses retain a compact legacy error,
so backend and sidecar images can roll independently. Existing responses at or
below 32 MiB remain byte-for-byte on the legacy one-frame path, and generation
keeps its existing streaming IPC protocol.

The response assembler emits one bounded semantic transfer event through the
canonical facade. It projects to `sie.worker.ipc.response.chunks` (outcomes
`completed`, `protocol_error`, or `budget_rejected`),
`sie.worker.ipc.response.reconstructed.size`,
`sie.worker.ipc.response.chunk.count`, and
`sie.worker.ipc.response.chunk.reserved`. The sidecar does not emit a second
Prometheus-shaped copy.

`EnsureModelReady` returns readiness state, an optional `ModelDescriptor`, and,
once the model is ready, its `max_batch_tokens`: the batch cost budget from the
model's profile, in the model's cost units (tokens for text, milliseconds for
audio). The descriptor carries tokenizer path, tokenizer content hash, maximum
sequence length, output types, default text templates, and
`supports_run_batch`.
`loading_started` and `loading_in_progress` mean the same worker is actively
loading the model; the dispatcher progress-ACKs and rechecks instead of NAKing
those deliveries. `retry_later` remains a NAK path.

Source: [`protocol/ipc_types.rs`](../src/protocol/ipc_types.rs),
[`ipc_client.rs`](../src/ipc_client.rs), and
[`packages/sie_server/src/sie_server/ipc_types.py`](../../sie_server/src/sie_server/ipc_types.py).

## Scheduling

The sidecar scheduler handles queue-mode encode, score, and extract batches for
sidecar-enabled worker pools.

Batching keys include operation and LoRA key. Batch formation uses item cost,
request count, coalescing windows, adaptive pull timing, and model-specific
scheduler state. Oversize items flush alone. Per-item outcomes are published
without dropping the rest of the batch.

Each model's scheduler sizes its cost cap from the model's own
`max_batch_tokens`, as the Python worker does for its own batcher: the static
cost cap starts at that budget, and the adaptive cost range runs from
`min(max(256, budget / 4), budget)` to `4 * budget`. A speech model with a
720000 ms budget can therefore fill a clip batch up to the request count cap,
while a text model keeps its token budget. Precedence:

1. `SIE_BATCHER_MAX_BATCH_COST`, when set, pins the cap and the range for every
   model, as before per-model budgets.
2. Otherwise the budget `EnsureModelReady` reports for the model.
3. Otherwise, with a backend that does not report one, 16384.

`SIE_ADAPTIVE_BATCH_MIN_COST`, `SIE_ADAPTIVE_BATCH_MAX_COST`, and
`SIE_ADAPTIVE_BATCH_INITIAL_COST` still override the range for every model.
The request count cap and the hard audio duration cap per batch do not change.
When a ready model later reports a different budget, for example after the
backend restarts with a changed profile, its scheduler adopts the new budget
and restarts its adaptive controller.

Encode cost uses prepared-token length when available. Score tokenization stays
backend-owned, so its scheduler wrapper caches the same model-independent proxy
as Python: Unicode character count plus 1024 per media input, summed once per
query/document pair. The cached score estimate never enters the IPC schema,
never mutates or truncates inputs, and is not an authoritative token or billing
count. Extract uses the prepared audio duration in milliseconds when the
sidecar decoded the item's audio, and unit cost otherwise.

Prepared audio longer than 30 s (Whisper's feature window) never shares a
batch, whatever the model. Such recordings are served one per batch in arrival
order, and while other items are pending too the two kinds take turns: a
cost-sorted batch of the other items, then the oldest long recording. When no
long recording is pending, batch formation is unchanged. The Python batcher
applies the same rule, but there only the Whisper adapter flags audio as
long-form.

Routing remains gateway-owned. The sidecar does not keep a separate local
active-model routing list.

Source: [`scheduler/`](../src/scheduler/), [`latency.rs`](../src/latency.rs),
and [`dispatcher.rs`](../src/dispatcher.rs).

## Preparation And Framing

The sidecar performs GPU-independent preparation and framing:

- msgpack-native image/document shape validation and ingress-equivalent media
  count/byte limits before extract IPC;
- text template application for Rust tokenization;
- HF fast-tokenizer loading from adapter-declared paths;
- `PreparedTokens` attachment for encode IPC items;
- bounded payload-store fetch sized for the widest accepted native extract
  request plus serialization overhead, followed by modality-specific limits
  (16 MiB image/document, 24 MiB compressed audio) and inline replacement for
  offloaded work items;
- dense, sparse, multivector, score, and generated-output framing;
- `msgpack_numpy` sentinel-compatible payload handling.

Local filesystem payload resolution is a Linux worker capability: the sidecar
pins the configured directory and uses `openat2` without symlink traversal.
Non-Linux hosts, kernels older than 5.6, and sandboxes that block `openat2` fail
closed; cloud object-store payload resolution is unaffected.

Alibaba `oss://` payload references use a native region-scoped Signature V4
client. It accepts only a safe plain key or the exact full reference below the
configured bucket/prefix, derives the public or VPC-internal HTTPS endpoint from
`SIE_OSS_REGION`, and authenticates with explicit environment credentials or
ACK RRSA OIDC. ECS metadata and file/profile credential sources are absent.

The backend retokenizes when prepared tokens are absent or the tokenizer hash
does not match. Score pair construction stays backend-owned. Extract
tokenization stays backend-owned.

Large generation and vision payloads are fetched from payload storage and
inlined before `ProcessGenerate`, so the backend receives a self-contained msgpack
work item.

Non-streaming results normally remain one named-msgpack `WorkResult`. A gateway
that can reassemble bounded results sets `WorkItem.accepts_result_chunks=true`.
If that result exceeds the negotiated NATS `max_payload`, the sidecar hashes the
original serialized `WorkResult` with SHA-256 and publishes ordered
`ResultChunkV1` envelopes whose complete encoded size stays within the broker
limit. After the final fragment, one NATS flush provides a broker write barrier
before the source JetStream work item is ACKed. Chunk payloads target at most
768 KiB, one transfer is limited to 16 MiB and 64 chunks, and the gateway must
verify identity, indexes, total size, and digest before decoding the
reconstructed `WorkResult`. Missing/false capability, an over-limit transfer,
or a broker ceiling too small for a bounded transfer retains the compact typed
`PAYLOAD_TOO_LARGE` response. This negotiation keeps old gateways and old
sidecars safe during rolling deploys.

Both response-chunk hops retain the complete bounded logical result until its
length and SHA-256 have been verified, then decode the ordinary response.
Chunking removes physical UDS/NATS frame ceilings; it is not public streaming,
progressive decoding, or cross-process zero-copy transport. Producers use
borrowed payload views while framing chunks, and receiver assembly hashes in
canonical chunk order while copying into the required contiguous decode
buffer. These changes avoid unnecessary transient work without weakening that
verification boundary.

The publisher similarly emits one bounded terminal transport event and one
event per successfully queued result chunk through the canonical facade. These
project to `sie.worker.result.transport.attempts`,
`sie.worker.result.chunks.published`, and `sie.worker.result.chunk.size`;
Prometheus compatibility remains a collector/exporter concern.

The sidecar preserves compressed image and document bytes. It does not decode
pixels or rasterize PDFs: the current IPC protocol carries compressed media and
has no shared decoded-media representation, while Docling's PDFium-backed
pipeline needs structural PDF content for layout/text fidelity. Image
processors retain resize/color/normalization/tiling; Docling retains
document decode, rasterization, actual processed-page counting, and export.
Python's typed item decode remains a parity check for sidecar and non-sidecar
paths.

Source: [`prep/`](../src/prep/), [`tokenize/`](../src/tokenize/),
[`output/`](../src/output/), [`payload_store.rs`](../src/payload_store.rs), and
[`dispatcher.rs`](../src/dispatcher.rs).

## Readiness, Health, And Telemetry

The sidecar HTTP server exposes:

- `/healthz`: process liveness;
- `/readyz`: readiness based on successful IPC `Ping`, heartbeat freshness, and
  drain state.

The probe server listens on `SIE_WORKER_PROBE_PORT` (default `9095`). Metrics
have no scrape endpoint. Queue, batch, and sidecar-to-engine IPC observations
are emitted exactly once through the canonical OpenTelemetry facade and pushed
over OTLP when `SIE_METRICS_ENABLED` is true. The regional collector owns
Prometheus-compatible conversion and destination routing.

The deployment supplies `lane` as `<queue-pool>|<machine-profile>|<bundle>`.
The bounded telemetry catalog may be seeded at startup, grows only after a
trusted live config is successfully applied, and is replaced by every
successful full export response. It admits at most 256 exact semantic
`(model, profile)` pairs for the process lifetime; duplicates are free and
unknown, invalid, removed, or overflow pairs collapse together to
`(other, other)`. The budget affects telemetry only and emits at most one
warning; config application and serving continue unchanged. Explicit SDK views
cover the finite domain of every sidecar stream so valid labels do not enter
`otel.metric.overflow`.

Readiness state is shared with the NATS health publisher. The health publisher
emits worker identity, bundle, machine profile, readiness, the current
bundle config hash, and, when non-empty, the `unsupported_models` committed
with that hash. It also mirrors the latest `loaded_models` list reported by
the backend IPC heartbeat so gateway pool/model gauges reflect live residency in
NATS health mode.

Source: [`runtime_state.rs`](../src/runtime_state.rs),
[`readiness.rs`](../src/readiness.rs), and
[`health_publisher.rs`](../src/health_publisher.rs).

## Live Config

The sidecar subscribes to bundle-scoped config deltas:

```text
sie.config.models.{bundle}
```

A notification with a reply subject or any `Nats-` header is dropped first:
`sie-config` publishes plain core messages, and such a delivery can only come
from the NATS server acting for a user that manages JetStream streams or
consumers (for example a stream's republish setting), past that user's publish
permissions. Each remaining notification is checked for trusted producer,
bundle, epoch, and payload size. Accepted deltas are forwarded to the colocated backend through
`ApplyModelConfig`, together with the notification's `bundle_adapters` list for
this bundle when `sie-config` sends one. The backend returns the applied bundle
config hash and `unsupported_models`, the routable ids that hash covers but the
backend cannot serve. A delta whose config the Python backend rejects is not an
apply failure: the backend keeps the model's current config, hashes the received
config, and lists the model in `unsupported_models`, as it does for a rejected
export entry. The sidecar stores the hash and the list together in
`ConfigApplyState` only when the hash exactly matches a non-empty control-plane
hash. Before anything is committed, the sidecar adopts the backend's IPC `Ping`
hash, which never replaces a committed pair. `Ping` carries no list, so its hash
must imply full support: the Python worker scopes it by its image's bundle file,
and the Candle worker returns a hash it echoed only after applying every entry.
A mismatch or missing backend proof does not
advance the epoch, advertised hash, or loaded-model state, leaving the worker
quarantined from hash-bound work until reconciliation succeeds.

When `SIE_CONFIG_SERVICE_URL` is configured, the export reconciler fetches
`/v1/configs/epoch` and `/v1/configs/export` from `sie-config`. Bundle-relevant
exports are sent to the backend through `ReplaceModelConfigs`, with the
export's `bundle_adapters` list for this bundle.

The reconciler authenticates with `SIE_CONFIG_SERVICE_TOKEN`
(`--config-service-token`), `sie-config`'s read-scoped `SIE_CONFIG_READ_TOKEN`,
which cannot write configs. When the variable is set it decides, and a blank
value sends no token. When it is unset, the sidecar falls back to
`SIE_ADMIN_TOKEN` and logs a deprecation warning at startup.

Export reconciliation skips unchanged periodic exports. Exports older than the
local epoch are skipped unless the reconciler is handling an epoch-rewind
recovery. Partial or rejected replacements do not update the advertised bundle
hash. Neither does a control-plane/backend hash mismatch; the snapshot remains
eligible for retry rather than being recorded as reconciled.

Grouped encode, score, extract, and generate work carries the gateway's
expected bundle config hash for the resolved pool and bundle. Config mutation
takes an exclusive execution barrier and backend inference takes a shared
barrier. Immediately before execution the dispatcher requires an exact match
with the worker's current hash; old hashes are NAKed rather than executed
against newer weights. Work for a model in the current `unsupported_models`
list is NAKed at intake, again immediately before each readiness probe, and at
that barrier, before backend IPC, so the dispatcher never asks the backend to
load or run such a model. The backend's own eager load of pinned models does
not consult the list. These NAKs are counted with reason `model_unsupported`.
Successful non-streaming results echo that stable
execution hash so the gateway can bind response provenance to the exact worker
execution. Empty hashes remain accepted only for legacy, non-attested traffic.

A non-empty hash also requires an installed `ConfigApplyState`; custom
dispatchers without that authority refuse pinned work before backend execution.
The Python IPC receiver independently checks the live bundle configuration
view, including control-plane adapter scope and unsupported models. Its shared
registry execution lease permits concurrent inference while blocking
filesystem reload and asynchronous add/remove/replace until the batch or
generation stream finishes. Synchronous configuration mutation refuses while
the lease is held and must run on the registry's lifecycle loop once bound.
The lease is separate from model-load locks so a pinned request can load its
model while a configuration writer waits.

Configuration writers wait for existing executions within the registry's drain
timeout. If a long generation stream exceeds that wait, the update fails with
a retryable timeout and reopens reader admission; the old configuration remains
pinned until the stream ends. Identity and evidence validation run off the
lifecycle event loop during asynchronous configuration updates.

Hashed generation retains the gateway-selected model/profile. If the Python
grammar resolver would rewrite that target, the receiver NAKs before loading
or prewarming a model: grammar-routing declarations are not covered by the
bundle hash. An unavailable grammar profile is settled as a terminal
`unsupported_field` refusal by the streaming processor under the same lease.
Score IPC items carry the same optional hash as encode and
extract items, including scheduler batches. Deploy matching updated Python
workers and sidecars together to obtain this execution authority; legacy
empty-hash traffic and older IPC producers do not provide the guarantee.

Live config apply updates model configuration in the colocated backend registry/catalog.
It does not update adapter code or bundle definitions inside the running worker
image.

Source: [`config_subscriber.rs`](../src/config_subscriber.rs),
[`config_reconciler.rs`](../src/config_reconciler.rs),
[`dispatcher.rs`](../src/dispatcher.rs), and the gateway
[config guide](../../sie_gateway/docs/architecture-guide.md#43-live-deltas).

## Source Links

- Package summary: [README](../README.md)
- Helm values: [`deploy/helm/sie-cluster/values.yaml`](../../../deploy/helm/sie-cluster/values.yaml)
- Runtime config parsing: [`config.rs`](../src/config.rs) and [`main.rs`](../src/main.rs)
- IPC schema parity check: [`tools/check_ipc_types_parity.py`](../../../tools/check_ipc_types_parity.py)
- response-chunk v1 envelope/limit pin: [`tools/check_response_chunk_protocol.py`](../../../tools/check_response_chunk_protocol.py)
- Gateway queue contract: [gateway architecture guide](../../sie_gateway/docs/architecture-guide.md)
