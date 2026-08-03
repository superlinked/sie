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

- [Worker runtime](../../../product/design/worker-runtime.md)
- [Deployment topologies](../../../product/design/deployment-topologies.md)
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
- Active local model loads are held with JetStream progress ACKs until
  `EnsureModelReady` returns ready; this preserves the delivery budget during
  cold starts. The progress delay is clamped below the pool consumer `ack_wait`
  even if the retry NAK delay is configured higher.
- Transient backend, decode, stream, and drain failures are NAKed.
- Reply publication failure, including a partial chunk sequence, leaves the
  JetStream message unacked for redelivery.

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
streaming cancellation contract uses `cancel.*`.

The greater of the work stream age and retry window is the active worker's
tombstone expiry horizon. A 100,000-entry process cap evicts the oldest
tombstone sooner under sustained cancellation volume. `work_cancel` is NATS
Core and therefore best effort across disconnects and restarts. JetStream
remains at-least-once, so an ACK failure can redeliver work, but a retained
tombstone filters that redelivery. Static inference already sent over backend
IPC is not preempted; the gateway has removed its collector and drops the late
result.

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
- `SignalGenerateCancel`
- `RunBatch`
- `ApplyModelConfig`
- `ReplaceModelConfigs`
- `Drain`

`tools/ci/check_ipc_types_parity.py` checks the Rust protocol schema against
`packages/sie_server/src/sie_server/ipc_types.py`.

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

`EnsureModelReady` returns readiness state and an optional `ModelDescriptor`.
The descriptor carries tokenizer path, tokenizer content hash, maximum sequence
length, output types, default text templates, and `supports_run_batch`.
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

Encode cost uses prepared-token length when available. Score tokenization stays
backend-owned, so its scheduler wrapper caches the same model-independent proxy
as Python: Unicode character count plus 1024 per media input, summed once per
query/document pair. The cached score estimate never enters the IPC schema,
never mutates or truncates inputs, and is not an authoritative token or billing
count. Extract currently uses unit cost.

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
emits worker identity, bundle, machine profile, readiness, and the current
bundle config hash. It also mirrors the latest `loaded_models` list reported by
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

Each notification is checked for trusted producer, bundle, epoch, and payload
size. Accepted deltas are forwarded to the colocated backend through
`ApplyModelConfig`. The backend returns the applied bundle config hash. The
sidecar stores that hash in `ConfigApplyState` only when it exactly matches a
non-empty control-plane hash. A mismatch or missing backend proof does not
advance the epoch, advertised hash, or loaded-model state, leaving the worker
quarantined from hash-bound work until reconciliation succeeds.

When `SIE_CONFIG_SERVICE_URL` is configured, the export reconciler fetches
`/v1/configs/epoch` and `/v1/configs/export` from `sie-config`. Bundle-relevant
exports are sent to the backend through `ReplaceModelConfigs`.

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
against newer weights. Successful non-streaming results echo that stable
execution hash so the gateway can bind response provenance to the exact worker
execution. Empty hashes remain accepted only for legacy, non-attested traffic.

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
- IPC schema parity check: [`tools/ci/check_ipc_types_parity.py`](../../../tools/ci/check_ipc_types_parity.py)
- response-chunk v1 envelope/limit pin: [`tools/ci/check_response_chunk_protocol.py`](../../../tools/ci/check_response_chunk_protocol.py)
- Gateway queue contract: [gateway architecture guide](../../sie_gateway/docs/architecture-guide.md)
