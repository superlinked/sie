# SIE Architecture

This document describes the current deployed architecture of the SIE control plane and inference gateway. It is verified against the public `packages/sie_gateway`, `packages/sie_config`, and `packages/sie_sdk` implementations. Every behavior described here is what the code does on this branch.

Top-level summary:

- `sie-gateway` (Rust) is the inference gateway. It routes ordinary encode, score, and extract requests through pool queues, capped logical batch pools and generation through worker direct-dispatch, all over JetStream, and serves read-side configuration.
- `sie-config` (Python) is the configuration control plane. It owns writes, persistence, and delta publication.
- Workers execute inference.
- NATS carries runtime messaging. JetStream carries inference work and the DLQ. Core pub/sub carries config deltas, worker health, and inference results.
- Kubernetes coordinates pool state. It is not on the inference request path.

The gateway's contract with `sie-config` is three shapes. Everything else below elaborates on them.

1. `GET /v1/configs/export` — full snapshot, consumed on cold start and on drift detection.
2. `GET /v1/configs/epoch` — monotonic integer plus compact fingerprints, polled every 30s to detect drift.
3. `sie.config.models._all` (NATS Core) — live deltas; missed ones are caught by (2).

The gateway's contract with admin tooling for post-write readiness is `GET /v1/configs/models/{id}/status`.

## 1. Services

| Service | Role | Publishes | Subscribes / Consumes |
|---|---|---|---|
| Client SDK | Submits inference requests | HTTP inference requests | HTTP inference responses |
| Admin / deploy tooling | Mutates config, reads control-plane state | HTTP config requests | HTTP config responses |
| `sie-gateway` (Rust) | JetStream-dispatched inference gateway: ordinary encode/score/extract are pool-queued, while capped logical batch pools and generation are worker-direct; read-side config; pool management; health and metrics | Queue work items, read-side config responses, pool state changes, DLQ republishes, metrics | Client inference requests, worker results, worker health, config deltas, K8s pool state |
| `sie-config` (Python) | Config control plane; single writer | Config write responses, read responses, snapshot/export responses, config deltas (NATS) | Admin/config-management requests |
| Workers | Execute inference | Result messages, health signals | Queue work items |
| NATS / JetStream | Message bus | — | — |
| Kubernetes API | Pool coordination | — | — |

Key terms used throughout this document:

- **Bundle**: a model-compatibility grouping. Answers "which workers can serve this model?". Defined by a `name`, a `priority`, an optional `engine`, and an `adapters[]` list of adapter module identifiers. The source of truth is `sie-config`'s filesystem (image-baked at `SIE_BUNDLES_DIR`, default `/app/bundles`); `sie-config` loads them at startup to drive write validation and exposes them via `GET /v1/configs/bundles` + `GET /v1/configs/bundles/{id}`. The gateway no longer bakes its own copy: `state::config_bootstrap::BootstrapClient::fetch_bundles` pulls them at startup and installs them into `ModelRegistry` via `install_bundles` before any model fetch. Neither service *persists* bundles, and there is no `POST /v1/configs/bundles` on either side — bundle changes are a `sie-config` redeploy and the gateway picks them up on its next bootstrap (or on the next poller-driven catch-up; see §4.2). A profile resolves to its compatible bundle with the lowest `priority` number. Remote adapters (`sie_server.adapters.remote.*`) are listed both in `default`, so a single server serves them, and in `remote` at priority 1, so in a cluster every remote profile, including a `model:remote` variant of a model served locally, resolves to the `remote` lane.
- **Pool**: an operational routing group of workers managed by the gateway. Answers "which compatible workers should this request be queued to right now?". Pools are a runtime concept stored in Kubernetes.
- **Config store**: `sie-config`'s backing store for persisted state. Per-model YAML files plus a plain-text monotonic `epoch` counter, laid out as `{base}/models/{id-with-/-as-__}.yaml` and `{base}/epoch`. The backend is selected by the `SIE_CONFIG_STORE_DIR` scheme: a local path (default — backed by an opt-in PVC in Helm), `s3://…`, or `gs://…`.
- **Config epoch**: a monotonic integer. Every `sie-config` write that creates new profiles bumps it by one. The gateway tracks it locally as `ConfigEpoch` for drift detection.
- **`bundle_config_hash`**: a content hash over each model's immutable weights revision and routable-profile fields (`adapter_path`, `max_batch_tokens`, `compute_precision`, `adapter_options`) for every model whose adapters match a bundle. Used by admin tooling and workers to tell whether a worker's local registry and weights identity match the gateway's expected config for this bundle. Every service scopes the hash by `sie-config`'s bundle definition, so it identifies the config a worker received, not the adapter code in the worker's image (§7).
- **`unsupported_models`**: the routable model ids (`model` or `model:profile`) covered by a worker's `bundle_config_hash` that the worker cannot serve, for example because its image predates an adapter that `sie-config` added to the bundle. Workers report it in their heartbeat; it is empty in steady state.

## 2. Inference Request Path

1. Client SDK sends HTTP to the gateway:
   - `POST /v1/encode/{*model}`
   - `POST /v1/score/{*model}`
   - `POST /v1/extract/{*model}`
   - `POST /v1/embeddings` — OpenAI-compatible JSON surface; the gateway accepts **string** or **list of strings** for `input`, supports `encoding_format=float` (default) or `base64`, translates to an internal **`POST /v1/encode/{model}`** with `items[].text` and `params.output_types=["dense"]`, then maps encode `items[].dense` vectors into OpenAI `data[].embedding` plus a rough `usage` estimate. Token-id / nested-array inputs are rejected with **`400`**. `dimensions` is accepted only when it equals the model's native dense width and otherwise rejected with **`400 unsupported_field`**, the same rule as the single-node server. The wrapper must buffer the whole encode reply before reshaping it, and that reply is an *amplification* of the request (N short texts in, N × `dim` floats out), so the buffer is bounded by its own **response**-specific cap, `MAX_COMPAT_RESPONSE_BODY` (16 MiB) — deliberately a separate constant from the request cap it used to be derived from, since the two quantities are unrelated and a change to ingress must not silently move this bound. The value holds today's effective bound rather than raising it: at the queue's item ceiling a float reply reaches ~90 MB at 1024 dims and ~224 MB at 2560 dims (`Qwen/Qwen3-Embedding-4B`, served here), the handler holds ~4 copies at once, and the deployed gateway is a 1 GiB container running 4 requests concurrently — so no bound both admits the largest real reply and fits. **Consequence, stated plainly: a batch above roughly 760 inputs at 1024 dims cannot be served on this surface.** The cure is an ingress item cap (terminal `400` before any GPU work), streaming assembly, or a larger container — each its own change. Over the cap the surface answers a **terminal `413`** naming the one remedy that works (fewer inputs — `encoding_format=base64` is inert here, because the field is never forwarded to `/v1/encode` and base64 is applied afterwards to the outer response), never a retryable `503`. Both that `413` and the unreadable-body `500` carry `GatewayOwnedFault`: the worker has already run and reported its units, so a metered composition must release the hold rather than bill for a response carrying zero embeddings.
2. Gateway resolves the request to a model, a bundle, a machine profile, a logical API pool, and the backing queue pool using its in-memory registry.
3. Gateway ordinarily publishes work to JetStream on the backing queue subject `sie.work.{pool}.{machine_profile}.{bundle}.{model}` and carries the logical pool as `admission_pool` in the work item. A capped logical batch pool instead targets one assigned worker on `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`.
4. A matching worker consumes the work item, verifies its logical pool assignment when `admission_pool` differs from its physical queue pool, and executes inference.
5. Worker publishes the result on `_INBOX.{router_id}.{request_id}` (NATS Core).
6. Gateway collects the result and returns the HTTP response.

Rules enforced on the inference path:

- The inference hot path does not call `sie-config`.
- Ordinary encode, score, and extract use pool-queue dispatch; capped logical batch pools and generation use worker direct-dispatch. Both paths use JetStream, with no direct-HTTP fallback. `src/handlers/proxy.rs` is the JetStream submission handler despite its name.
- If the queue transport is unavailable (no usable NATS client at init), the gateway returns `503`. It does not fall back to direct mode.
- Unknown model ids fast-fail with `404` whenever the in-memory `ModelRegistry` has been populated (either by the filesystem seed or by a successful bootstrap / delta from `sie-config`). In the pre-bootstrap edge case where the registry is still empty — no seed, no export applied yet — the proxy falls back to the caller-supplied bundle (or `"default"`) so an unseeded gateway can still publish work to a cold pool that a caller pinned via `X-SIE-Pool`. Once any model is registered, this fallback is disabled and the 404 contract applies.
- Automatic pool selection only considers healthy workers whose reported `bundle_config_hash` matches the gateway's expected hash. For the requested model, a lane (`pool`, `machine_profile`, `bundle`) is eligible only when no such worker in it lists the model in `unsupported_models`: every worker of a lane consumes the lane subject, so a worker that cannot serve the model would otherwise receive its items and NAK them, spending JetStream delivery attempts. Worker-direct rings (generation and capped logical pools) exclude workers that list the model. When every lane is excluded the gateway returns the `503 PROVISIONING` contract below and counts the exclusion in `sie.gateway.routing.unsupported_model_exclusions`. Explicit `X-SIE-Pool` selects a logical pool; the gateway publishes to that pool's backing queue pool (`PoolSpec.queue_pool`, default `default`) and stamps the logical pool into the work item as `admission_pool`. Deliveries whose hash is not the worker-sidecar's current hash, whose model the worker lists in `unsupported_models`, or whose logical pool does not assign that worker, are NAKed before backend IPC. Successful non-streaming results echo the execution hash; the gateway emits `X-SIE-Model-Revision` only when every successful result matches the hash and immutable revision captured atomically at routing time. The header value is the lowercase 64-hex executed bundle/config SHA-256, not the catalog weights revision (for example, a 40-hex Hugging Face commit). Buffered generation also follows this rule even though it internally collects worker chunks. True SSE responses (`stream: true`) always omit the header because headers precede the terminal result. Successful terminal SSE events may instead carry the worker-origin `execution_identity_sha256` / `execution_binding_sha256` pair, both lowercase 64-hex digests; absence remains valid for older or self-hosted deployments and does not attest execution. These digests are distinct from both the weights revision and the bundle/config hash.
- On scale-from-zero — i.e. no healthy worker registered for the `(bundle, machine_profile)` tuple and the caller did not pin an explicit pool — the gateway records pending demand for KEDA and returns a retryable `503` provisioning response with `Retry-After: 60`, `X-SIE-Error-Code: PROVISIONING`, and gateway version headers. SIE-native surfaces use the SDK retry envelope (`{"error":{"code","message"}}`); OpenAI-compatible surfaces (`/v1/generate`, `/v1/embeddings`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`) use the OpenAI error envelope, because standard OpenAI clients parse 2xx as successful model output. This applies whether or not the caller set `X-SIE-MACHINE-PROFILE`; default-routing clients get the same contract as profile-pinned clients.
- On no-consumer conditions for the JetStream publish, the gateway treats the miss as the same pre-execution provisioning state and returns the same retryable `503 PROVISIONING` contract (`Retry-After: 60`, `X-SIE-Error-Code: PROVISIONING`, and surface-specific error envelope). On backpressure conditions the gateway returns `503` with `Retry-After: 5`. Backpressure is evaluated twice: pool-wide against the `WORK_POOL_{pool}` stream's pending count (`SIE_GATEWAY_MAX_STREAM_PENDING`), and per lane (`pool`/`machine_profile`/`bundle`) against a gateway-local in-flight work-item count (`SIE_GATEWAY_MAX_LANE_IN_FLIGHT_ITEMS`). The per-lane decision is always computed and recorded on `sie.gateway.queue.lane_admission.decisions`, but it only sheds when `SIE_GATEWAY_LANE_BACKPRESSURE_ENFORCE` is set; with the flag off (the default) the pool-wide check is the sole gate. Both sheds produce the same `503` contract and both record pending demand for the exact physical lane so KEDA scales it.
- On queue result timeouts the gateway returns `504` with `X-SIE-Error-Code: GATEWAY_TIMEOUT` and `Retry-After: 5`. This means the gateway accepted and published the work item, but no worker result reached the gateway before `SIE_GATEWAY_REQUEST_TIMEOUT`; it must not be collapsed into worker-emitted `MODEL_LOADING`, which remains a separate retryable `503 MODEL_LOADING` signal.
- When workers report failure on every item of a batch with the **same retryable error code** in `WorkResult.error_code`, the gateway translates that into a `503` with the SDK-expected envelope (`error.code`, `Retry-After`, `X-SIE-Error-Code`) so the SDK auto-retries. Currently recognised codes: `RESOURCE_EXHAUSTED`, `MODEL_LOADING`, `LORA_LOADING`, and `QUEUE_FULL` (a worker that cannot serve the item now, such as a remote profile whose upstream is busy, unreachable, rate capped or behind an open circuit breaker; a remote profile whose upstream model is not ready answers `MODEL_LOADING`). A remote worker answers these upstream refusals at once with the upstream's wait rather than redelivering the item. Generation workers attach the operator's validated `oom_recovery.retry_after_s` value to a `RESOURCE_EXHAUSTED` terminal chunk, and a worker may set `WorkResult.retry_after_s` on a failed item; for `RESOURCE_EXHAUSTED`, `QUEUE_FULL` and `MODEL_LOADING` failures, the gateway uses the longest worker hint that qualifies as an integer in `1..=60` seconds, ignoring other hints. If no hint qualifies, or the code is `LORA_LOADING` even when a worker provides a hint, it uses the current five-second default compiled into the gateway. **Mixed batches** (different codes per item) keep going through the compatibility `500 all_items_failed` path with per-item `code` fields exposed in `details[]`, so callers can see exactly which items hit which failure mode. Workers reporting `RESOURCE_EXHAUSTED` are **not** marked unhealthy — losing an allocation race is not a worker-health signal.
- When **every** failed item carries **`MODEL_LOAD_FAILED`**, the gateway returns **`502 Bad Gateway`** with the SDK-style **`error`** object (`code`, `message`, `error_class`, `attempts`, `permanent`) — matching the terminal model-load contract consumed by SDK retry logic (no `Retry-After`; the SDK must not burn the `MODEL_LOADING` retry budget). The gateway synthesises conservative `attempts` / `permanent` fields on this queue path when the worker payload does not carry registry-shaped failure metadata.
- Every successful response and every `5xx` of an inference route carries `X-SIE-Served-By` (`local` or `remote`) once routing has resolved the dispatched model, and `X-SIE-Upstream` with the upstream's name when that model's default profile uses a remote adapter (`handlers::serving_disclosure`). The value is read from the dispatched route's configuration, as `sie_server.api.helpers.serving_disclosure_headers` reads it on the single server, so a `model:remote` variant or a remote-backed model reports `remote` and the model's local profile reports `local`. Client errors carry neither header. The compatibility routes (`/v1/embeddings`, `/v1/rerank`, `/v2/rerank`, `/v1/audio/transcriptions`) forward both through `is_openai_compat_forwarded_header`.

Encode / extract tuning fields (`output_types`, `instruction`, `options`, `labels`, `output_schema`, `is_query`) are read **only** from the nested JSON **`params`** object (and the msgpack analogue: a top-level **`params`** map). The score path continues to read `query` / `instruction` / `options` at the top level of the request body, matching `sie_server`.

Work-item and result payloads on the wire are **msgpack** (`rmp_serde`). JSON is used only for the HTTP request/response envelope when the client negotiates it (via `Content-Type` / `Accept`); the gateway transcodes msgpack ↔ JSON at the edge so the hot path between gateway and workers is always binary.
On SIE-native JSON requests, media byte fields such as `items[].images[].data`
are accepted as base64 strings at the HTTP edge and decoded into msgpack `bin`
before the gateway publishes the worker item, preserving the worker-side typed
`bytes` contract behind gateway-fronted clusters.

Gateway-generated **JSON error** bodies (validation, routing, auth, config read, pools) use a FastAPI-like envelope **`{"detail":{"code":"<STABLE_CODE>","message":"…", …}}`**. Extra diagnostic keys (e.g. `compatible_bundles`, `gpu`, `model`) live **inside** `detail` when present. This is intentionally **not** the same shape as SDK retry contracts, which remain **`{"error":{"code","message"}}`** for **503** retryable worker signals, SIE-native provisioning, and **`502`** **`MODEL_LOAD_FAILED`**. OpenAI-compatible provisioning uses the OpenAI error envelope. **200** success payloads keep their existing top-level shapes.

## 3. Configuration Write Path

All config mutations go to `sie-config`. The gateway has no write handler.

1. Admin or deploy tooling sends `POST /v1/configs/models` with a model config YAML body to `sie-config`. Auth: `_check_write_auth` (`packages/sie_config/src/sie_config/config_api.py`) requires a bearer matching `SIE_ADMIN_TOKEN` when that variable is set; if `SIE_ADMIN_TOKEN` is unset and a read-scoped token (`SIE_CONFIG_READ_TOKEN` or `SIE_AUTH_TOKEN`) is set, writes are rejected with `403` (a read token never grants write access); if no token is set (dev/local only) writes are accepted unauthenticated. Production deployments always set `SIE_ADMIN_TOKEN`.
2. **Pre-lock validation** (outside the per-app write lock, on the FastAPI event loop): parse the YAML body and — when a top-level `sie_id` is present — run `_validate_model_id` (regex + `..`/`\\` rejection + `status`-suffix rejection). Reject invalid input with `400` before any state is touched. The body is also checked against `packages/sie_config/src/sie_config/model_config.schema.json`, the JSON Schema of the worker's `ModelConfig` (a test keeps the two equal): an unknown key or a wrong value type at any level returns `422 validation_error` with one `{loc, message}` entry per field, instead of accepting a config that every worker rejects. JSON types are applied strictly, so values that pydantic's lax mode would coerce, such as a quoted number, are rejected too. Required fields are not enforced here, since append bodies are partial. `PUT` runs the same check.
3. **Critical section** under `_get_write_lock` (a lazy, per-app `asyncio.Lock` stored on `app.state`):
   1. `ModelRegistry.validate_model_config` — pure check against the in-memory registry (no mutation), run under the registry's internal `threading.RLock`. Validation rejects any write whose post-state resolves to zero routable bundles (`422`), so an `extends`-only profile cannot land a brand-new model that no worker bundle can serve. Appending a new `extends`-only profile to an already-routable model is allowed. When `SIE_UPSTREAM_NAMES` is set, each profile in the body whose adapter, after `extends`, is a remote adapter (`sie_server.adapters.remote.*`) must name one of those upstreams in `adapter_options.loadtime.upstream`, or the write is rejected with `422 validation_error`. `PUT` runs the same check. Profiles already stored are not re-checked. A remote worker applies the same rule to every profile, against its own upstreams file, when it applies the config.
   2. **Compute `created_profiles` vs `existing_profiles_skipped`.** The write is treated as append-only: profiles already present in the registry with a compatible body are reported as skipped, not rewritten. If there are no new profiles *and* at least one skip, an on-disk conflict check compares the incoming YAML against `ConfigStore`'s existing file for this model (guards against disk drift from the in-memory registry).
   3. **Persist to disk** via `ConfigStore.write_model` — **only if there are new profiles (`created_profiles` non-empty) and a `ConfigStore` is configured**. The persisted YAML is the result of a merge with the existing stored document (when one exists): new top-level fields are added, missing top-level fields are preserved, and same-key mutations to existing non-`profiles` metadata are rejected with `409 content_conflict` (the API is append-only for metadata too). Incoming `profiles` are merged profile-by-profile. No-op replays skip this step entirely. On the `LocalBackend`, `write_model` is atomic (`tempfile.mkstemp` → write → `fsync` → `Path.replace`, in `packages/sie_sdk/src/sie_sdk/storage.py`); on `S3Backend` / `GCSBackend` the write is a single object-store PUT (last-writer-wins, atomic at the object level).
   4. **Mutate the in-memory registry** via `ModelRegistry.add_model_config`. Runs under the registry's `RLock`; uses atomic dict-swap semantics so lock-free readers (e.g. `GET /models`, `GET /bundles`) never observe a torn state. The registry re-runs `_validate_config_locked` internally, so validation effectively runs twice (once from step 3.1, once from inside `add_model_config`) — the doubled cost is negligible and the redundancy is intentional.
   5. **Increment the epoch** via `ConfigStore.increment_epoch` — **only if `created_profiles` is non-empty and a `ConfigStore` is configured**. Read-modify-write; single-writer is enforced by the outer lock. `increment_epoch` always goes through plain `write_text` (no CAS), so running multiple `sie-config` replicas against a shared store would be unsafe — see §10 on single-replica deployment.
   6. **Publish NATS deltas** via `NatsPublisher.publish_config_notification` — **only if a publisher is configured, currently connected, and `created_profiles` is non-empty**. If the publisher is configured but not currently connected, the handler returns `503` instead of publishing. On `PartialPublishError` the handler emits a `nats_publish_partial` entry in `warnings` naming the failed bundles; on any other publish exception it emits a generic `nats_publish_failed` warning. The published `model_config` body is the merged YAML (not the partial incoming body), so every subscriber replays the same authoritative state sie-config wrote to disk.
4. `sie-config` returns a JSON response: `model_id`, `created_profiles`, `existing_profiles_skipped`, `warnings`, `routable_bundles_by_profile`, `router_id`.

Ordering guarantees:

- Within the critical section, persist (3.3) happens before registry mutation (3.4). A disk-write failure aborts the write before the in-memory state changes, so a later process restart cannot observe a registry-only model that has no backing YAML.
- The epoch is bumped strictly inside the lock, after a successful persist and registry apply. Two concurrent writers cannot lose an epoch bump.
- NATS publish happens inside the lock, so the `(bundle_id, epoch)` pairs reach the wire in strict monotonic order per bundle.
- No-op replays (every profile already present, no `created_profiles`) skip disk, epoch-bump, and NATS publish entirely. The response still returns normally, reporting `existing_profiles_skipped` with an empty `created_profiles`.
- `GET /v1/configs/export` also takes the same lock, after a per-app single-flight gate that lets one export build at a time, so a write waits behind at most one build however many exports are queued. The lock makes every exported snapshot a real serialization point: `(epoch, models)` always corresponds to a state that existed between two writes.

NATS publish behavior:

- One `ConfigNotification` message is published per affected bundle to `sie.config.models.{bundle_id}`, and a byte-identical copy to `sie.config.models._all`.
- Gateway code consumes `sie.config.models._all`. Worker-sidecar containers consume their bundle-scoped subject and forward accepted deltas over backend IPC `ApplyModelConfig`.
- If publish fails for a subset of bundles, `NatsPublisher` continues publishing to the remaining bundles and raises `PartialPublishError` at the end with the failed bundle list. The write itself is still durable on disk. The response `warnings` field names the failed bundles; gateway convergence is repaired by the poller/export path if `_all` is missed. Worker-sidecar convergence is repaired by the worker export reconciler when `SIE_CONFIG_SERVICE_URL` is set.

Idempotency:

- Writes accept an `Idempotency-Key` header. The state (`_IdempotencyState` on `request.app.state`) holds an LRU response cache and per-key in-flight records for deduplication. Each in-flight record carries the owner's payload hash, completion event, and terminal success/failure/cancellation outcome for current waiters.
- A duplicate key with the same body returns the cached response without re-executing.
- A duplicate key with a different body returns `422 idempotency_mismatch`.
- A duplicate key whose successful cached response has been LRU-evicted between the in-flight wait and the cache read returns a synthesized `200 idempotent_replay_evicted` response. The write is never executed twice for the same key.
- A duplicate key waiting on an in-flight owner that fails does not synthesize success. If the owner is cancelled before any irreversible write-side effect begins, current waiters see `503 idempotent_inflight_cancelled`, and a later non-overlapping retry can execute normally because failures are not stored in the replay cache. Once the owner enters the commit phase (durable store write or registry mutation), cancellation is deferred until the real outcome is known, so same-key waiters replay the committed success/failure instead of retrying against an unknown background write.

Model-ID validation (`_validate_model_id`):

- Rejects path-traversal patterns (`..`, `\\`).
- Requires `^[a-zA-Z0-9][a-zA-Z0-9._/-]*$`.
- Rejects IDs equal to `status` or ending in `/status`, because the gateway's `GET /v1/configs/models/{*id}` uses a `/status` suffix for worker-ack reporting and an overlapping model ID would make that URL ambiguous.

The gateway does not serve `POST /v1/configs/models`. The route is not registered. Axum returns `405 Method Not Allowed`. There is no `410 Gone` shim and no redirect body.

## 4. Gateway Bootstrap and Recovery

The gateway does not block startup on `sie-config`. Startup is:

1. Construct `ModelRegistry`. In default Helm deploys, `SIE_BUNDLES_DIR` and `SIE_MODELS_DIR` are unset and the registry's filesystem reload is a no-op (warns once that the dirs are missing, then proceeds with empty bundle and model maps). The optional `gateway.embeddedConfigs` / `gateway.configMap` overlays mount a ConfigMap at `/configs/{bundles,models}` and set those env vars; in that mode the registry loads the seed before bootstrap runs.
2. Initialize the other runtime subsystems that do not depend on `sie-config`: NATS manager (with config-delta subscription), worker health manager, optional filesystem config watcher, pool manager / K8s backend, and worker discovery.
3. Spawn two independent background tasks: `state::config_bootstrap::spawn_bootstrap_retry` and `state::config_poller::spawn`. They share the same `ModelRegistry`, `ConfigEpoch` (monotonic model-write counter), `BundlesHash` (sha256 fingerprint of `sie-config`'s loaded bundle set), and `BundleConfigHashesHash` (sha256 fingerprint of the per-bundle config-hash map). The poller skips its first tick (one `DEFAULT_POLL_INTERVAL` grace) so the initial bootstrap attempt can run first, and then reconciles on every subsequent tick regardless of whether the bootstrap task has completed yet.
4. Bind the HTTP listener. Kubernetes probes use plain-text endpoints implemented in `handlers/health.rs`:
   - **`GET /healthz`** — liveness. Always **`200 OK`** with body **`ok`** and **`Content-Type: text/plain; charset=utf-8`** (same wire shape as `sie_server`'s `/healthz`).
   - **`GET /readyz`** — readiness for **routing traffic to this gateway process**. When `SIE_CONFIG_SERVICE_URL` is set, it returns **`503 Service Unavailable`** + **`config bootstrap pending`** until this replica has applied its first complete `sie-config` snapshot (`bootstrap_once` returned `Ok`, from either the retry task or the poller), then **`200 OK`** + **`ok`** for the life of the process. The gate proves that this replica reached `sie-config` and, when `sie-config` requires a token, authenticated; it does not prove that the catalog is correct: the first bootstrap accepts whatever complete export `sie-config` serves. Without `SIE_CONFIG_SERVICE_URL` it returns `200` once the listener is serving. After the first snapshot it is independent of `sie-config` reachability, and it is always independent of worker health: a gateway with zero healthy workers must still receive the first inference request so it can return a retryable provisioning response and emit canonical `sie.gateway.pending_demand` telemetry for scale-from-zero. Worker availability is exposed on **`GET /health`** and by the inference response path itself.

The bootstrap/poller tasks are spawned **before** the HTTP listener binds, so requests hitting a freshly-bound replica can observe filesystem-seed-only state depending on timing. **`/readyz` waits only for the first complete snapshot** (see §11 caveat 3), which keeps a replica that cannot read its catalog (for example, because `sie-config` refuses its token) out of Service endpoints instead of serving a partial catalog. Later catch-up is tracked via **`GET /v1/configs/models/{id}/status`**, **`sie.gateway.config.bootstrap.degraded`**, and **`sie.gateway.config.applied_epoch`**; `/readyz` does not flip back when `sie-config` becomes unreachable after that.

### 4.1 Bootstrap retry

`state::config_bootstrap::spawn_bootstrap_retry` runs `bootstrap_once` in a retry loop until the first success, then exits. Ongoing reconciliation after that point is the poller's job (§4.2).

Startup short-circuits:

- If `SIE_CONFIG_SERVICE_URL` is unset, the task logs and returns immediately — there is nothing to bootstrap against. The gateway runs filesystem-seed-only.
- If constructing the HTTP client fails, the task sets `sie.gateway.config.bootstrap.degraded` to `1` and returns; it does not retry.

`bootstrap_once` runs a three-call sequence on `SIE_CONFIG_SERVICE_URL`, authenticated with the gateway's sie-config credential (`SIE_CONFIG_SERVICE_TOKEN`, a read-scoped token) as a bearer token:

1. `GET /v1/configs/epoch` — reads the compact control-plane fingerprints BEFORE we fetch bundles, so the hashes we store reflect the state we're about to catch up from (the pre-fetch ordering is load-bearing — see below). The `bundle_config_hashes_hash` fingerprint covers both worker-parity bundle config hashes and model-level pool ownership, because top-level `pool` changes affect routing/readiness even when the worker-applied profile hash is otherwise unchanged.
2. `GET /v1/configs/bundles` and per-id `GET /v1/configs/bundles/{id}` — the two-phase bundle fetch. The YAML bodies are parsed into `BundleInfo` values and retained for installation with the model export; this fetch does not install bundles independently.
3. `GET /v1/configs/export` — parses every exported model config, then installs the bundles and authoritative model set together through `ModelRegistry::replace_authoritative_surface_if_current`, guarded by the captured registry generation. Each parse error increments `failed` and prevents installation. If parsing succeeds but installation is rejected (for example, because a model is unroutable or the captured generation is stale), one installation error is logged and `failed` increments once. A failed export does not mutate the model/bundle surface. Export rows that carry no config body are skipped and are *not* counted as failed.

Returns `BootstrapOutcome { epoch, bundles_hash, bundle_config_hashes_hash, applied, failed, total }`.

The fetch captures the local registry generation before its first request.
Installation checks that generation under the registry's write lock and publishes
the model/bundle surface and exported bundle/pool hashes together. A delta,
reload, or another bootstrap that changes the registry during the fetch makes
the export stale: it is refused without changing the registry or advancing its
epoch/fingerprints, and the caller retries. This prevents an older in-flight
export from erasing a newer model update while leaving the epoch ahead of the
installed models. The network fetch never holds the registry write lock.
The poller retains a pending retry after a failed bootstrap, even if a concurrent
delta makes epochs and fingerprints match. Only a successful bootstrap clears
that retry; failed epoch reads leave it pending.

- If **any** model entry failed (`outcome.failed > 0`), `bootstrap_once` returns `BootstrapError::PartialApply` and **advances neither `ConfigEpoch`, `BundlesHash`, nor `BundleConfigHashesHash`**. The poller will detect drift on the next tick and retry.
- If all entries applied, `bootstrap_once` advances `ConfigEpoch` via `set_max(outcome.epoch)`, stores `outcome.bundles_hash` into `BundlesHash`, stores `outcome.bundle_config_hashes_hash` into `BundleConfigHashesHash`, marks the replica bootstrapped (`ConfigEpoch::mark_bootstrapped`, which `/readyz` reads), and the retry task exits.

Why fetch `/epoch` BEFORE `/bundles`: if a bundle is added on `sie-config` between our epoch read and our bundle list read, we install a bundle set *newer* than the hash reflects. The next poll tick sees `remote_hash != stored_hash` and re-fetches — a cheap self-heal. The inverse — storing a hash that's *newer* than the bundles we actually installed — would silently wedge the gateway on a stale registry, which is what the bundle-hash mechanism exists to prevent.

On failure (DNS, 5xx, timeout, decode error, partial apply):

- Records `sie.gateway.config.operations{operation="bootstrap", outcome=...}`.
- Sleeps with exponential backoff, initial 1s, doubling to a 60s cap.
- After 10 failed attempts or 5 minutes of sustained failure, sets `sie.gateway.config.bootstrap.degraded` gauge to `1` and escalates logging to `error!`. The gauge clears on the first success.
- The retry loop itself has no attempt ceiling — it keeps trying until one attempt succeeds (at which point the task exits) or the process is killed. `sie-config` is not a hard startup dependency.

During the bootstrap-retry window (before the first successful fetch):

- **`GET /readyz`** returns `503` when `SIE_CONFIG_SERVICE_URL` is set (rule above), so Kubernetes keeps the replica out of Service endpoints. Requests that reach the pod directly are still served from whatever models exist in the in-memory registry (typically filesystem-seed-only during this window).
- `GET /v1/configs/models/{id}` returns only filesystem-seeded models.
- `POST /v1/configs/resolve` for API-only models returns `404`. Admin tooling retries with backoff.
- `ConfigEpoch.get()` typically returns `0` and `GET /v1/configs/models/{id}/status` surfaces `config_epoch: 0`, which admin tooling uses to detect a pre-bootstrap replica. The epoch is not strictly pinned to `0`, however: if the NATS delta subscriber (§4.3) receives a valid `ConfigNotification` with a non-zero `epoch` before the first export succeeds, `ConfigEpoch::set_max` will advance the counter. When `SIE_CONFIG_SERVICE_URL` is set, `GET /readyz` is the reliable "has export ever succeeded?" signal; the `sie.gateway.config.bootstrap.degraded` gauge only reports sustained failure and stays `0` both before its threshold and after success.

### 4.2 Epoch poll

`state::config_poller::spawn` runs independently of bootstrap with `DEFAULT_POLL_INTERVAL = 30s`. It skips its first tick (to let the initial bootstrap attempt run without contention), then on every tick calls `GET /v1/configs/epoch` on `sie-config` (sending the same `SIE_CONFIG_SERVICE_TOKEN` bearer; `/epoch`, like every sie-config read route, accepts `SIE_CONFIG_READ_TOKEN`, `SIE_AUTH_TOKEN`, or `SIE_ADMIN_TOKEN`). The `/epoch` response carries three drift signals:

- `epoch` — monotonic counter bumped on every model-config write.
- `bundles_hash` — sha256 fingerprint over `sie-config`'s loaded bundle set. Bundles are filesystem artifacts inside the `sie-config` image, so their effective "version" is redeploy time, which the model-write counter does not observe.
- `bundle_config_hashes_hash` — sha256 fingerprint over the per-bundle `bundle_config_hash` map from `sie-config` plus model-level pool ownership. This catches no-store/filesystem-baseline redeploys where the epoch and bundle set stay unchanged but the expected routing hashes or model pool assignment move.

All signals funnel into the same `bootstrap_once` call — that function re-fetches bundles and models together, which is the correct blanket response to any control-plane drift.

Per-tick decisions:

- If `remote_epoch > local_epoch`, `remote_bundles_hash != stored_bundles_hash`, or `remote_bundle_config_hashes_hash != stored_bundle_config_hashes_hash` (with empty-string sentinels skipped), the poller calls `bootstrap_once`. On success, `ConfigEpoch`, `BundlesHash`, and `BundleConfigHashesHash` are updated.
- If `remote_epoch < local_epoch`, the local counter has run ahead of authority. Most innocent cause: `sie-config` lost its persisted epoch file and restarted at a lower value. Most worrying cause: a forged/untrusted NATS delta wedged the gateway ahead of authority (this is defense-in-depth alongside the producer allowlist — see §4.3). The poller refetches the full export first and only calls `ConfigEpoch::force_set(remote)` if the export succeeded; a failed recovery export keeps `local > remote`, so the next tick re-enters this branch and retries. Successful recovery replaces the gateway model set with the authority's export, including removals. Logged at ERROR.
- A remote hash of `""` is the documented "sie-config registry unavailable" / legacy-field sentinel. The poller skips that hash-mismatch branch rather than thrash fetching against a degraded control plane. The epoch branch still runs normally.
- Transient `/epoch` failures are logged and swallowed. The next tick retries. Neither the local epoch nor the stored hashes are reset on poll failure.
- The poller **always** compares local vs. remote, including when `local_epoch == 0` or `stored_bundles_hash == ""`. This matters for fresh clusters: if `sie-config`'s very first write (→ epoch 1) is lost on the wire, the gateway must catch `remote=1 > local=0` and trigger recovery. Gating on "non-zero local" would wedge a fresh gateway until a restart.

Operational consequence: adding or changing a bundle in `sie-config` propagates to running gateways within one `DEFAULT_POLL_INTERVAL` (≤ 30s worst case). No `kubectl rollout restart deployment/sie-gateway` is required for the gateway to learn the updated bundle set.

### 4.3 Live deltas

Independently of bootstrap and polling, `NatsManager` subscribes to `sie.config.models._all` (Core pub/sub) as soon as the NATS client is ready. For each incoming `ConfigNotification`:

- **Transport check first.** A notification with a reply subject or any `Nats-` header is dropped and counted as `rejected_untrusted`. `sie-config` publishes plain core messages, while a NATS user that can manage JetStream streams or consumers can make the server itself deliver stored bytes to any subject past its publish permissions (stream republish and direct-get replies carry `Nats-Stream`/`Nats-Sequence` headers, consumer deliveries carry a `$JS.ACK` reply subject). With the Helm chart's NATS users, only `sie-config` may publish on `sie.config.models.>`, so this check closes the remaining server-side path.
- **Producer-trust check next.** The `producer_id` (wire key `router_id`) must be in the trusted-producer allowlist. Default allowlist is `["sie-config"]`, configurable via `SIE_NATS_CONFIG_TRUSTED_PRODUCERS=a,b,c`; Helm sets it to the rendered config Deployment name (for example `sie-sie-cluster-config`) so Kubernetes pod names from that Deployment match. Matching is exact equality OR K8s pod-name prefix (`sie-config` also matches `sie-config-5f7b6d8c-kxwvr` and `sie-config-0`, but not `sie-configuration`). Untrusted notifications are dropped before any registry mutation or epoch advance; a log warning names the rejected producer. Set `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER=true` to disable validation (dev/local only — `main.rs` emits a startup warning when this is on).
- An **empty-or-whitespace `model_config` body** (pure epoch bump) advances `ConfigEpoch` via `set_max(notification.epoch)` and returns.
- A non-empty body is parsed as `ModelConfig` YAML. If the parse fails, the epoch is NOT advanced and the event is logged. The poller will catch up.
- On successful parse, `ModelRegistry.add_model_config` is called. If it returns `Ok`, the epoch is advanced. If it returns `Err` (e.g. append-only conflict, unroutable adapter), the epoch is NOT advanced.

`ConfigEpoch::set_max` uses a CAS loop over `AtomicU64`. The value only ever moves forward; a late-arriving lower-epoch delta cannot roll it back. `async_nats::Subscriber` survives reconnects transparently, so the config-delta path does not need explicit resubscribe logic — any messages published while the subscriber is disconnected are simply lost, which is the gap the poller closes.

## 5. Gateway Read Surface

The gateway serves these endpoints directly out of its in-memory `ModelRegistry`. They always reflect the gateway's own view, which may briefly lag `sie-config` during drift but is self-healing (§4.2).

- `GET /v1/configs/models` — lists every model the gateway currently knows about. Each entry includes `model_id`, `profiles`, `source`.
- `GET /v1/configs/models/{*id}` — dual-purpose wildcard dispatcher:
  - Plain path → YAML document describing the model (`sie_id`, `source`, `bundles`).
  - Path ending in `/status` → JSON worker-ack readiness for the model (see §6).
  - Disambiguation: if the `/status`-stripped ID is a known model, the endpoint returns the status view. Otherwise it falls back to interpreting the full path (including `/status`) as a model ID. `sie-config` refuses to register IDs ending in `/status`, so the ambiguous case cannot arise from legitimate writes.
- `GET /v1/configs/bundles` — lists every bundle the gateway knows about with its `priority`, `adapter_count`, and `connected_workers`.
- `GET /v1/configs/bundles/{id}` — YAML document describing the bundle.
- `POST /v1/configs/resolve` — resolves a `bundle:/model`-style spec to a specific bundle and returns its compatible bundles and profile names.

`sie-config` serves the same shapes (`GET /v1/configs/models`, `/models/{id}`, `/bundles`, `/bundles/{id}`, `POST /v1/configs/resolve`) backed by its authoritative store. In a healthy steady state the two views match.

`GET /v1/configs/models/{id}/status` is gateway-only. `sie-config` has no worker registry; it cannot answer per-replica readiness.

Concurrency on the gateway's `ModelRegistry`:

- Reads are lock-free. `ArcSwap<RegistrySnapshot>` lets readers clone an `Arc` pointer to the current snapshot without blocking.
- Writes (`add_model_config`, authoritative export replacement, `reload`) hold `write_lock: Mutex<()>` across the `load → mutate → store` cycle. Without this lock, two concurrent writers could both load the same base snapshot and silently drop one set of changes.
- `RegistrySnapshot` caches `bundle_config_hashes: HashMap<String, String>`, so the per-request worker-ack hash lookup (§6) is an `O(1)` map read rather than a SHA-256 over tens of KB of JSON.

## 6. Worker-Ack Status (`GET /v1/configs/models/{id}/status`)

Admin tooling uses this endpoint after a `sie-config` write to observe whether configured workers are reporting the expected `bundle_config_hash` for each affected bundle. Worker-sidecar containers advance that hash after their bundle-scoped NATS delta is accepted by backend IPC `ApplyModelConfig`, or after the worker export reconciler replaces the bundle-scoped backend registry/catalog view from `sie-config`. During that replacement the Python worker applies every entry its `ModelConfig` schema accepts; for a rejected entry it logs the model and reason and keeps that model's current registry entries, if any. It handles a live delta it rejects the same way, so a rejected delta advances the hash and is reported in `unsupported_models` instead of failing the apply. The sidecar still advances the advertised hash only when the worker's resulting hash equals the control-plane hash, and it commits the worker's `unsupported_models` together with that hash (§7).

Response shape:

```json
{
  "model_id": "BAAI/bge-m3",
  "config_epoch": 42,
  "all_bundles_acked": true,
  "no_bundles": false,
  "bundles": [
    {
      "bundle_id": "default",
      "pool": "default",
      "expected_bundle_config_hash": "sha256…",
      "total_eligible_workers": 3,
      "acked_workers": ["worker-1", "worker-2", "worker-3"],
      "pending_workers": [],
      "unsupported_workers": [],
      "acked": true
    }
  ],
  "source": "gateway-registry"
}
```

Computation rules:

- **Per-replica only.** The endpoint reports the worker registry state on this specific gateway pod. Fleet-wide readiness is the union across replicas; admin tooling fans out.
- **Case-sensitive bundle match.** A worker is counted as eligible only if `worker.bundle == bundle_id` exactly. Workers whose `bundle` field does not match exactly are skipped entirely — they contribute to neither `total_eligible_workers` nor `pending_workers`. The expected hash is computed against the canonical bundle ID from the registry, and the fleet is case-consistent in practice.
- **Per-model support.** A worker that reports the expected hash but lists the model, or one of its `model:profile` routes, in `unsupported_models` is reported in `unsupported_workers`, not `acked_workers`, and keeps `acked` false. During a rollout that adds an adapter to a bundle, workers on the earlier image appear here for the models that use the new adapter while they keep serving every other model.
- **Healthy-only.** Unhealthy workers (per `worker.healthy()`) are skipped entirely; they contribute to neither `total_eligible_workers` nor either worker list.
- **Zero-bundle models.** A model with no routable bundles is reported as `all_bundles_acked: false` and `no_bundles: true`. Returning `true` for "nothing to ack" would silently tell admin tooling "fully deployed" when there is nothing deployed.
- **`config_epoch` field.** Surfaces the gateway's local `ConfigEpoch` so callers can tell whether this replica has caught up to a recent write (`config_epoch` lower than the write's returned epoch means this replica is still catching up).

## 7. Cross-Service Hash Parity

`bundle_config_hash` is computed independently by `sie-config`, the gateway, and workers. The `sie-config` export/epoch surface keeps the global per-bundle hash for bootstrap and drift detection, while export snapshots and live NATS deltas also carry per-bundle/per-pool hashes for worker convergence. Worker sidecars apply only models whose top-level `pool` matches their own `SIE_POOL`, then advertise that pool-scoped hash. Gateway hot routing/readiness compares workers against the hash for the same materialized scope the worker serves: the default pool scope for unfiltered workers, or the named physical queue-pool scope for workers filtered by `SIE_POOL`. Logical API pools may draw from a different backing queue pool via `PoolSpec.queue_pool`; for the admin-tooling readiness flow to work, gateway and worker hashes must be byte-identical for that backing bundle/pool scope. A model config's top-level `pool` must therefore match the physical queue scope that will materialize it; dynamic logical pools backed by `default` should use default-scoped model configs unless a dedicated worker queue exists.

A model belongs to a bundle's hash when one of its profiles resolves to an adapter module in that bundle's `adapters` list. All three services use `sie-config`'s list for this. `sie-config` sends it as `bundle_adapters` (bundle id to adapter modules) with every export snapshot and live delta, and the worker sidecar forwards the list for its bundle over backend IPC (`ApplyModelConfig` / `ReplaceModelConfigs`). The Python worker then hashes the configs it received with that list, so a worker whose image predates an adapter still reproduces the control-plane hash. A worker that has not yet received a list, or whose `sie-config` does not send one, uses the list in its image's bundle file, as before. When the worker's schema rejects an export entry or a live delta, the worker hashes the received entry's routable fields, so the hash still identifies what the control plane sent.

Hash equality therefore no longer implies that the worker can serve every model it covers. The worker reports the routable ids it cannot serve as `unsupported_models`: those whose adapter module is in the control-plane list but not in its image's bundle file, and those whose export entry its schema rejected when the config it retains differs in hashed fields (or it has none). The sidecar commits the list together with the hash, publishes it in its heartbeat (omitted when empty), and NAKs work for a listed model before backend IPC. The gateway routes a model to a worker only when the hash matches and the model is not listed (§2). A worker that sends no list serves every model its hash covers: before this field existed, workers hashed with their image's adapter list, so a matching hash already implied that the image contains every covered adapter. The same holds for a hash that reaches a heartbeat without a list today. Backend IPC `Ping` carries no list, so the Python worker's `Ping` hash stays scoped by its image's bundle file, and a sidecar adopts it only while it has committed no config (at startup, including a sidecar restart next to a running worker). The gateway accepts at most 1024 ids per heartbeat. A worker that reports more counts as unable to serve any model until it reports fewer, because a dropped id would otherwise read as supported. A lane is eligible only when every hash-matching worker in it can serve the model (§2), so the whole lane of an overflowing worker is routed no model while it overflows. The Candle worker reports no list. It echoes the control-plane hash only after it has parsed and applied every entry of the export or delta, so an echoed hash implies full support. An entry it cannot parse fails the whole apply: the worker keeps its previous hash and the sidecar does not advance, so the bundle is held on that worker as before.

The hash is a SHA-256 over a JSON-serialized, sort-keys representation of a structured object. The outer shape is an array of models, each with `sie_id`, `revision`, and `profiles: [{name, config: {...4 fields...}}]`. `revision` is the immutable `hf_revision` commit SHA, or null for package-backed/unpinned development models. The inner `config` for each profile is strictly the four routable-profile fields: `adapter_path`, `max_batch_tokens`, `compute_precision`, `adapter_options`. Any field outside that whitelist is excluded from the hash so that non-routable edits (e.g. comments, description-style metadata) cannot unnecessarily invalidate worker acks. Python uses `orjson.dumps(..., OPT_SORT_KEYS)`; Rust mirrors this via `serde_json::to_vec` over `BTreeMap`-backed canonical types.

`adapter_options` is canonicalized before hashing:

- Python (`ModelRegistry.compute_bundle_config_hash`):

  ```python
  if isinstance(adapter_opts, dict) and not any(adapter_opts.values()):
      adapter_opts = None
  ```

- Rust (`CanonicalProfile::from_profile` → `canonicalize_adapter_options`): if `adapter_options` is an object and every value is Python-falsy (`null`, `false`, `0`, `0.0`, `""`, `[]`, `{}`), it is replaced with `None`; otherwise the value is preserved as-is.

The Rust predicate is intentionally a mirror of Python's `not any(values)`. Any divergence here would make the gateway's `expected_bundle_config_hash` in §6 disagree with every worker's advertised hash, and workers would sit in `pending_workers` indefinitely for any model whose config contained a value like `{"flag": 0}`.

## 8. Messaging Layers

NATS provides two messaging layers and this system uses both. The distinction matters for failure analysis.

- **NATS Core pub/sub** is fire-and-forget. A publisher sends a message on a subject; any subscriber currently connected receives it. If the subscriber is disconnected at that instant, the message is gone. No replay, no persistence, no acks.
- **JetStream** is a persistence layer over Core. Messages are stored, consumers get at-least-once delivery with acks, redelivery, and backpressure.

Per-subject transport:

| Path | Subject pattern | Transport | Notes |
|---|---|---|---|
| Pool inference work (`encode` / `score` / `extract`) | `sie.work.{pool}.{machine_profile}.{bundle}.{model}` | JetStream | Durability and max-delivery semantics required. Work-item payload is msgpack. One stream per pool (`WORK_POOL_{pool}`) captures ordinary non-generation work for the pool, while worker consumers filter one concrete machine-profile/bundle lane; see §8.2. |
| Worker direct-dispatch | `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}` | JetStream | Worker-specific stream used by generation and by capped logical batch pools that must target an assigned worker instead of allowing unassigned workers on the same backing queue to burn JetStream delivery attempts. |
| Batch direct cancel | `batch_cancel.{router_id}.{worker_id}.{request_id}` | NATS Core | Best-effort worker-scoped signal emitted only after non-streaming worker-direct fallback publishes are durably acked on the pool subject. Sidecars ACK-drop queued worker-direct encode/score/extract items for the request; pool fallback items are never cancelled by this signal. |
| Non-generation request abandonment | `work_cancel.{router_id}.{request_id}` | NATS Core | Best-effort request-wide signal emitted when the gateway abandons encode/score/extract work, including when a publish ACK fails and the stored item's fate is unknown. Active sidecars retain bounded namespaced tombstones and ACK-drop matching work before backend IPC. The signal is not replayed to disconnected or restarted workers, and static inference already past IPC is not preempted. |
| Inference results | `_INBOX.{router_id}.{request_id}` | NATS Core | Gateway is waiting synchronously; a brief blip after publish but before delivery means the result is lost and the client may retry (the gateway returns `504` with `X-SIE-Error-Code: GATEWAY_TIMEOUT` and `Retry-After: 5` — see §2). Result payload is msgpack. |
| Config deltas | `sie.config.models.{bundle}`, `sie.config.models._all` | NATS Core | Lightweight fan-out. Gateway durability comes from the snapshot/export path (section 4), not the bus. JSON payload (control plane, not hot path). The gateway subscribes on `_all`; worker-sidecar containers subscribe on their bundle subject and apply through backend IPC. |
| Worker health | `sie.health.>` | NATS Core | Ephemeral, last-heartbeat-wins. The gateway subscribes in `health_mode=nats` (see `discovery/nats_health.rs`) and supervises the subscriber task: reconnects normally resume in `async-nats`, but a terminated subscription stream is recreated with bounded backoff because there is no full-state health poller. Worker-sidecar containers publish this heartbeat and include the latest bundle hash after successful config apply, plus `unsupported_models` when that hash covers models the worker cannot serve. `packages/wire-fixtures/worker_status.json` pins the heartbeat field set for the sidecar and the gateway. Helm sidecar deployments set `health_mode=nats`; the gateway binary default remains `ws` so standalone/test deployments can use the Python WebSocket path. |
| DLQ advisories | `$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>` | NATS Core (advisory) | JetStream emits these; gateway replicas subscribe in `queue/dlq.rs` and publish to `DEAD_LETTERS` with a deterministic `Nats-Msg-Id`, so JetStream dedupes replica fan-out. Genuine advisories carry no reply subject and no headers; the listener drops any that do, because a stream republish can deliver forged advisories on this subject. |
| DLQ storage | `sie.dlq.{model_token}` | JetStream | Single stream `DEAD_LETTERS` (Limits retention, 24 h `max_age`) captures `sie.dlq.>`. Storage and replicas track the work-queue setting (`SIE_STREAM_STORAGE` / `SIE_STREAM_REPLICAS`, memory/1 by default), so a memory-backed DLQ is erased by the same broker restart that erases the work it recorded. `model_token` is derived from the advisory's original work subject by taking the model segment and replacing `/` with `_`. |

### 8.2 Work-stream configuration and the model-ID constraint

The gateway's `ensure_stream` (per pool, called lazily on first publish) creates a JetStream stream with:

```
name:      WORK_POOL_{pool}
subjects:  ["sie.work.{pool}.*.*.*"]
retention: WorkQueue
storage:   Memory by default (`SIE_STREAM_STORAGE`: memory | file)
replicas:  1 by default (`SIE_STREAM_REPLICAS`)
max_age:   1800s (shared gateway/worker default)
max_msgs:  100_000
discard:   New (reject publishes at the max_msgs cap)
```

Storage defaults to `Memory`, which means queued and delivered-but-unacked work lives only in the broker's RAM: a NATS restart erases it along with the `DEAD_LETTERS` record that would have named it, and since the gateway is queue-only that restart is also a full inference outage. `SIE_STREAM_STORAGE=file` (plus a JetStream file store on the broker) persists the streams at the cost of disk and publish latency; `SIE_STREAM_REPLICAS` raises the replica count on a clustered NATS. Helm renders both onto the gateway and the worker sidecars from `queueRouting.streamStorage` / `queueRouting.streamReplicas` so the two stream creators cannot disagree.

max_age defaults to 1800 s on both the gateway (`SIE_STREAM_MAX_AGE_S`) and worker sidecar. The default intentionally stays above `max_deliver * 30s` for the pool consumer so the original work payload remains fetchable after the retry/DLQ envelope trips. The `New` discard policy makes the broker reject a publish once the stream sits at `max_msgs`, so the failed ACK surfaces on the existing 503 queue-unavailable path; the async-nats default (`Old`) would instead silently discard the oldest queued work, whose clients then ride to a 504. `get_or_create_stream` / `add_stream` does not update an existing stream's config, so both gateway and sidecar reconcile existing stream `max_age`, `discard`, and `num_replicas` during stream ensure; keep the Helm value wired to both gateway and workers so they repair to the same target. `storage` is deliberately NOT reconciled: JetStream refuses an in-place storage-type change (err_code 10052, "stream configuration update can not change storage type" — the message store is already materialized), and repairing it would mean deleting and recreating the stream, destroying the very work durability is meant to protect. A mismatch is logged as a warning and the stream is left alone until an operator drains the pool and recreates it. The same policy covers `DEAD_LETTERS` (`queue/dlq.rs`), which shares the reconcile helpers in `queue/stream_durability.rs` — destroying the forensic record to change its storage type would be the failure the knob exists to prevent. Note NATS does not validate `num_replicas` against the real topology (a single-node server accepts and reports R3), so the Helm chart, not the broker, is what guards a replica count against the deployed NATS cluster size.

When a gateway or sidecar observes an existing `WORK_POOL_{pool}` stream, it
reconciles the stream subjects to exactly `sie.work.{pool}.*.*.*`. Legacy
subjects such as `sie.work.*.{pool}` are intentionally removed from the stream
configuration. This release is a cutover to the lane-aware subject shape, not a
mixed-version bridge: all gateways and workers in the cluster must use the new
shape, and any old queued work should be drained or purged before rollout.

The stream's subject filter `sie.work.{pool}.*.*.*` matches exactly one token each for machine profile, bundle, and model. Workers attach durable consumers with the narrower `sie.work.{pool}.{machine_profile}.{bundle}.*` filter, so multiple bundles can share a pool stream without stealing each other's work. NATS subject tokens legally contain `/`, so `BAAI/bge-m3` works directly. They cannot contain `.`, `*`, `>`, or whitespace, so any model whose `sie_id` contains one of those characters would — without normalization — expand into multiple tokens, not match the stream or consumer filters, and get rejected at the broker.

Initial publication does not wait for the JetStream ACK on the HTTP request
path. The dispatcher returns a transport-neutral durability completion, and a
detached monitor drains every ACK. Before spawning it, the handler begins one
request-scoped handoff lease for the exact physical lane. Complete success
releases that lease; an ACK failure retains request-scoped demand, notifies the
request driver, and cleans up the handoff when the request exits. The
queue-publish metric therefore describes submission, while the bounded
`publish_ack` queue event describes durability. The aggregate ACK completion is
capped at six seconds, and exactly one handler-owned task awaits it. Its future
set is proportional only to the already-admitted request batch, so it does not
introduce a second unbounded admission class. The task is shorter-lived than
the already in-flight result wait during a broker outage; there is no second
publisher-owned task per request.

To prevent that, the Rust gateway's `work_subject(pool, machine_profile, bundle, model)` in `packages/sie_gateway/src/queue/publisher.rs` calls a private `normalize_model_id` helper that mirrors the Python SDK (`sie_sdk.queue_types.normalize_model_id`): `/` → `__`, `.` → `_dot_`, and `*`/`>`/space → `_`. So `vidore/colqwen2.5-v0.2` on the default RTX 6000 default-bundle lane publishes on `sie.work.default.rtx6000.default.vidore__colqwen2_dot_5-v0_dot_2` — exactly six tokens, matches the lane consumer filter. The DLQ token extractor in `queue/dlq.rs` splits the advisory subject on `.` and takes `parts[5]`, which with the normalization in place is already a single safe token.

Worker direct-dispatch uses `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`, which has one additional token after the model. That subject intentionally does not match the pool stream's `sie.work.{pool}.*.*.*` filter. The sidecar binds this worker-specific stream at startup. It stays idle for ordinary pool-queued encode/score/extract traffic, but it is required for two cases: generation direct-dispatch and capped logical batch pools whose `admission_pool` assigns only a subset of workers on a shared backing queue. In the capped logical case the gateway builds an HRW ring from the assigned healthy workers for that lane and publishes to one worker subject, allowing lazy model load without unassigned peers repeatedly NAKing the same message into `max_deliver`. If a non-streaming worker-direct publish fails, the gateway immediately retries on the pool subject; if published work is still missing after the short fallback delay, it republishes the missing items once to the pool subject so a disappeared worker-specific consumer does not strand the request until the client timeout. The gateway emits `batch_cancel.{router_id}.{worker_id}.{request_id}` only after the pool fallback publishes are durably acked, and the original sidecar only ACK-drops matching queued worker-direct batch items that have not reached IPC. Sidecar `WorkResult` payloads also carry a direct-origin marker, so once the gateway has confirmed fallback for an item index it ignores any late direct-origin result for that index rather than allowing mixed direct/pool completion. In-flight direct work may still consume GPU, but it cannot win the collector after confirmed fallback.

The same isolation applies in the gateway. Ordinary `encode` / `score` / `extract` and `/v1/embeddings` requests stay on the pool JetStream subject and keep the pre-generation queue-publish hot path. All recognized inference endpoints — generation and non-generation alike — inject the active W3C trace context into the queue work-item envelope (`injects_queue_trace_context`), so the worker span attaches to the gateway span across the pool/sidecar boundary; unrecognized labels fail closed. This is independent of generation-specific gateway proxy spans: capped logical pool requests may use worker direct-dispatch to preserve admission semantics, and non-generation endpoints do not pay generation proxy-span overhead. When tracing export is enabled, non-generation endpoints open the lightweight `gateway.publish` span around queue publish; generation endpoints open `gateway.proxy_generate` / `gateway.proxy_chat` spans on their streaming paths. New gateway hot-path decisions should use the shared endpoint classifier and its regression tests, not one-off string checks.

The non-generation HTTP handler treats its result collector as exclusive
transport ownership. Cancellation during queue publication, a client drop,
handler timeout, explicit partial-publish or chunk-validation failure, and the
periodic expiry sweep all race to remove the collector. Only the removal winner
performs request abandonment: it releases chunk-memory reservations, terminates
the response path without partial success, publishes `work_cancel`, and deletes
exact-key offloaded payloads. Late results see no collector and are
dropped. JetStream stays at-least-once; abandonment does not claim exactly-once
delivery or execution. A failed durable publish ACK removes the collector the
same way and also publishes `work_cancel`, because a failed ACK does not prove
the broker discarded the item.

Every non-streaming work item also carries `deadline`, the publish timestamp
plus the request timeout, as absolute Unix seconds. Sidecars use it to keep
held deliveries leased and to size backend calls, count items found past it,
and, when `SIE_WORK_DEADLINE_ENFORCE=true` is set on the workers, ACK-drop
them before execution, which covers workers that never saw the `work_cancel`
signal. The comparison spans the gateway and worker clocks, so both must be
synchronised. Workers ignore a deadline more than `SIE_WORK_DEADLINE_MAX_BUDGET_S`
(default 180 s) after its timestamp and log a warning, so raising
`SIE_GATEWAY_REQUEST_TIMEOUT` above that value requires raising the worker
setting too. Streaming generation items omit it because their timeouts are
resolved per profile after publication.

The managed Modal dispatcher implements the same ownership contract without a
NATS cancellation hop. It registers an abort handle for each detached
non-generation dispatch task before allowing that task to enter the i6pn call,
and abandonment removes and aborts the task. Dropping an i6pn call before its
frame commits returns its credit; an interrupted write retires the channel; and
a committed call keeps its pending correlation until the transport settles the
credit. This safely releases gateway ownership, but it cannot preempt static
model execution after the remote call boundary.

### 8.3 Publisher/consumer ownership

Publishers and consumers:

- **Config deltas**: published exclusively by `sie-config`. The gateway's `NatsManager` is subscribe-only on `_all` and never publishes on these subjects. Delta payload includes a `router_id` field whose value is `sie-config`'s pod/host identifier. The Rust gateway deserializes this field into a struct member named `producer_id`; `router_id` is declared as a `#[serde(alias)]` so both the current `router_id` wire name and the wire-compatible `producer_id` name parse cleanly.
- **Queue work**: published exclusively by the gateway. Consumed by workers.
- **Results**: published exclusively by workers. Consumed by the originating gateway.

The `ConfigNotification` JSON payload carries the following fields, verified against `packages/sie_gateway/src/nats/manager.rs` and `packages/sie_config/src/sie_config/nats_publisher.py`:

- `router_id` on the wire (Python publisher key). The Rust gateway parses this into a struct field named `producer_id`; `router_id` is declared as a `#[serde(alias)]` on the Rust side for forward compatibility.
- `bundle_id`, `epoch`, `bundle_config_hash`, `model_id`, `pool`.
- `bundle_pool_config_hashes` — nested `bundle_id -> pool -> hash` map used by gateways and sidecars to compare worker acks against the same pool-scoped bundle projection the worker applied.
- `profiles_added` — list of profile names created by this write.
- `model_config` — raw YAML string of the full model config.
- `bundle_adapters` — `bundle_id -> [adapter module]` for the affected bundles: the lists the hashes in this message were scoped by (§7). The export snapshot carries the same map for every bundle. Older `sie-config` versions omit it.
- `affected_bundles` — list of every bundle affected by this write. Each affected bundle receives its own message, so a `_all` subscriber sees N messages per write with their respective `bundle_id` / `bundle_config_hash` populated.

### 8.4 Serialization formats

The hot path is always **msgpack** (`rmp_serde` on the gateway, `msgpack-python` / `msgpack-numpy` on workers). Specifically:

- `WorkItem` (gateway → worker on JetStream): msgpack. `model_id` is the route the work runs on. When routing dispatched the request to another route of the requested model (a profile variant), `display_model` carries the model id the caller asked for, so logs and accounting downstream can report it; it is omitted otherwise. `WorkDispatcher::publish_work` takes the same `display_model`. A remote attempt that stands in for a local refusal the gateway holds carries `fallback_reason`, the trigger of that refusal (`provisioning`, `model_loading`, `saturated` or `unhealthy`), on every fallback bridge and on a threshold bridge while local capacity wakes. The remote worker answers such an item it cannot serve now with a retryable error instead of redelivering it, so the gateway restores the local refusal at once. Other work, including a low-demand threshold route, omits the field. An admitted numerical remote attempt also carries `numerical_admission_sha256` (see [Numerical bridges](#numerical-bridges)), and the remote worker answers it at once in the same way, even on a threshold route.
- `WorkResult` (worker → gateway on the inbox subject): msgpack. Numpy arrays use `msgpack-numpy`'s extension-free encoding (maps with a `nd: true` sentinel + `type`, `shape`, `data`). When the gateway advertises `WorkItem.accepts_result_chunks: true`, a result that does not fit one NATS message may instead arrive as named-msgpack `result_chunk_v1` envelopes. This boolean negotiates v1 only; a future envelope version requires a new capability rather than reinterpreting the existing field. The gateway reassembles v1 transfers within the pending request lifetime, with a SHA-256 digest and fixed item/chunk/request/process memory limits, then decodes the reconstructed bytes as the same `WorkResult`. The gateway transcodes to native JSON arrays only when the client's `Accept` header asks for JSON.

JSON (`serde_json`) is used where payloads are low-frequency or human-oriented:

- Control plane: `POST /v1/configs/models` body, `GET /v1/configs/export` response, `GET /v1/configs/epoch` response.
- NATS config deltas: `ConfigNotification` is JSON.
- Worker-ack status endpoint responses.
- Operator-visible APIs: pool management, `/v1/models`, `/health`.
- Inference request/response envelopes, when the client opts into JSON via `Content-Type` / `Accept`. SDK clients typically use msgpack end-to-end.

## 9. Storage and Persistence

`sie-config` is the only service that persists configuration. Its backing store is:

- `{base_dir}/models/{model_id_with_slash_replaced_by_double_underscore}.yaml` — one YAML file per model.
- `{base_dir}/epoch` — a plain-text integer, monotonically incremented by `increment_epoch`.

Backend selection is URL-driven (`sie_sdk.storage.get_storage_backend`):

- Local filesystem (default).
- `s3://bucket/prefix` — S3.
- `gs://bucket/prefix` — GCS.
- `abfs(s)://container@account.dfs.core.windows.net/prefix` — Azure Blob / ADLS Gen2.
- `oss://bucket/prefix` — Alibaba OSS for discovery/cache copies only; mutable
  `ConfigStore` construction rejects it because OSS cannot satisfy epoch CAS.

The **local** backend's `write_text` is atomic: it writes via `tempfile.mkstemp` in the same directory as the destination, `fsync`s, then `Path.replace`s. A mid-write crash cannot leave the destination truncated or empty. The S3/GCS/Azure config backends implement `write_text` as a single-object write; object-store writes are last-writer-wins and observably atomic at the object level, but they do not use the tempfile + replace pattern. OSS is available to cache/discovery code but is rejected as a mutable config base. The atomicity property matters most for the `epoch` file: `ConfigStore.read_epoch` silently maps a malformed or empty integer to `0`, so a zero-byte epoch file would collapse the drift-detection mechanism (`remote == local == 0` would read as "in sync forever").

`ConfigStore.increment_epoch` is a naive read-modify-write. Its single-writer assumption is enforced at the FastAPI layer by the per-app write lock (§3).

The gateway has no persistent config store. `SIE_CONFIG_STORE_DIR` and `SIE_CONFIG_RESTORE` are `sie-config`-only and are not part of the gateway's env surface. The only persistent-looking inputs on the gateway are the optional filesystem seed directories (`SIE_BUNDLES_DIR`, `SIE_MODELS_DIR`); these are unset in the default deploy, where bundles and models are pulled from `sie-config` at startup, and only become non-empty when the `gateway.embeddedConfigs` / `gateway.configMap` overlays explicitly mount one.

## 10. Deployment Topology

- `sie-gateway` and `sie-config` are separate Kubernetes Deployments with separate images. Their public build definitions live in `packages/sie_gateway/Dockerfile` and `packages/sie_config/Dockerfile`; the Helm chart consumes tagged `ghcr.io/superlinked/sie-{config,gateway}:<tag>` images by default, and consumes the per-platform/bundle `sie-server:<tag>-<platform>-<bundle>` matrix when a worker pool is enabled.
- `sie-config` runs as **a single replica**. Multi-replica is blocked by the in-memory idempotency cache (see §11 "Known operational caveats"). Config persistence is opt-in: when `config.configStore.enabled: true` in the Helm values, a PVC is mounted at `/var/lib/sie-config` (default mount path) and `SIE_CONFIG_STORE_DIR` / `SIE_CONFIG_RESTORE` are set so the store survives pod restarts; with the default `enabled: false`, the store is ephemeral (backed by the pod filesystem) and a pod restart resets the epoch to 0 and drops every API-added model — `sie-config` does not subscribe to NATS (it is the sole publisher) so there is no replay path, and the registry rebuilds from the image-baked `/app/bundles` + `/app/models` baseline only. Every gateway replica's `state::config_poller` then detects `remote_epoch < local_epoch` and force-resets to match the restarted `sie-config`, picking up whatever baseline it now advertises. Operators who need API-added models to survive `sie-config` restarts must set `config.configStore.enabled: true`. The Helm chart (`deploy/helm/sie-cluster/values.yaml`) documents the SPOF posture.
- `sie-gateway` scales horizontally. Each replica maintains its own in-memory `ModelRegistry` and its own `ConfigEpoch`. They all converge on the same state via a combination of bootstrap, deltas, and the poller; they do not coordinate with each other.
- Bundles live on `sie-config`'s filesystem at `SIE_BUNDLES_DIR` (default `/app/bundles`, set via `sharedPaths.bundlesDir`) and are baked into its image. `sie-config` reads them at startup, validates writes against them, and re-serves them over HTTP at `GET /v1/configs/bundles{,/{id}}`. The gateway fetches that surface during `state::config_bootstrap::bootstrap` and installs it via `ModelRegistry::install_bundles` — there is no second copy. The `gateway.embeddedConfigs` / `gateway.configMap` overlays still exist for the rare case of running the gateway without `sie-config` (e.g., self-contained smoke tests); they mount a ConfigMap at `/configs/bundles` which the registry's filesystem reload picks up before the (no-op) bootstrap runs. In all cases, updating bundles is a `sie-config` redeploy and the gateway picks up the change on its next bootstrap or poller-driven reconcile, not via a runtime API call.
- Pools are stored in Kubernetes `ConfigMap`s and `Lease`s read/written by the gateway. `sie-config` is not involved in pool management.
- NATS / JetStream runs as its own workload. The gateway's `NatsManager` builds `async_nats::ConnectOptions` with `retry_on_initial_connect()` so the process does not fail to start if NATS is briefly unavailable; reconnect behavior after initial connect relies on `async-nats`'s default policy (indefinite reconnect with backoff). With `SIE_NATS_USER` / `SIE_NATS_PASSWORD` set it authenticates with `user_and_password`; an authentication failure surfaces as a NATS server-error event in the log while the client keeps retrying.
- NATS authentication: the Helm chart gives `sie-config`, the gateway, and the worker sidecars separate NATS users and limits each to the subjects in §8 (permission matrix in the chart README, "NATS authentication"). The gateway user publishes work, DLQ entries, and cancels, manages the `WORK_POOL_*` and `DEAD_LETTERS` streams (`$JS.API.STREAM.INFO.*`, `$JS.API.STREAM.CREATE.*`, `$JS.API.STREAM.UPDATE.*`), reads consumer info for backlog metrics, and subscribes to `sie.config.models._all`, `sie.health.>`, `_INBOX.>`, and the max-deliveries advisory. It cannot publish config deltas or delete streams. Worker sidecars use the inbox prefix `_INBOX_WORKER`, so the worker user can publish results into `_INBOX.{router_id}.{request_id}` without subscribing to the gateway's inboxes. The worker user keeps JetStream stream and consumer management, and the server does not check stream subjects against subscribe permissions, so worker credentials can still capture gateway inboxes into a stream; the chart README lists what the worker user can still do. Only gateway pods mount a Kubernetes API token.

## 11. Known Operational Caveats

1. **`sie-config` is a single point of failure for the control plane.** While the pod is down, config writes, authoritative config reads, and gateway catch-up fetches all fail. **The inference hot path is unaffected** — it never touches `sie-config`. New gateway pods enter the soft-degraded bootstrap-retry state (§4.1) and return **`503`** from **`GET /readyz`** until `sie-config` is reachable and their first export succeeds; already-bootstrapped pods stay ready. A rolling gateway update during a `sie-config` outage therefore stalls with the old pods serving instead of replacing them with pods that have no catalog. The SPOF is rooted in `_IdempotencyState` (in-memory, per-process). Multi-replica requires moving idempotency state to a shared backend (Redis, JetStream KV, a DB row per key). Until then, `replicas: 1` is a deliberate trade of availability for a correct idempotency contract.

2. **Config deltas use NATS Core pub/sub; deltas published during a gateway-NATS disconnect are lost.** Recovery is automatic via the config poller (§4.2). The worst-case staleness window is `DEFAULT_POLL_INTERVAL` plus one export round-trip. No pod restart is required. Shortening `DEFAULT_POLL_INTERVAL` trades load on `sie-config` for recovery time.

3. **Bootstrap is background-retried, not blocking the listener.** A gateway started while `sie-config` is unreachable, or while `sie-config` refuses its token, binds immediately but reports `503` from **`GET /readyz`** until the first successful export, so it receives no Service traffic in that window. API-added models are missing until the first successful export. `ConfigEpoch` typically stays at `0` during this window — but a NATS delta that arrives before the first export can move it above `0`, so admin tooling should use **`GET /readyz`** on a replica with `SIE_CONFIG_SERVICE_URL` set as the authoritative "has this replica ever reconciled with `sie-config`?" signal rather than `config_epoch == 0`. The `sie.gateway.config.bootstrap.degraded` gauge (set after 10 failed attempts or 5 minutes of sustained failure, cleared on first success) is a sustained-failure signal, not a completion signal: it is `0` both before that threshold and after success. **`GET /readyz` encodes only the first bootstrap completion, never worker health** (§4); it does not flip back on later `sie-config` outages.

4. **Partial NATS publish on `sie-config`.** If a write succeeds on disk and in the registry but the NATS publish fails for a subset of affected bundles, `sie-config` still returns `201` (or `200` for no-op replays) with a structured `warnings` entry naming the failed bundles. Workers on those bundles stay on the previous epoch until the gateway poller triggers a re-export.

5. **`POST /v1/configs/models` on the gateway returns `405 Method Not Allowed`.** There is no custom body and no redirect pointer. Tooling that still targets the gateway for writes must be updated to call `SIE_CONFIG_SERVICE_URL` directly.

6. **Write-response shape dropped worker-readiness fields.** `sie-config`'s `POST /v1/configs/models` response does not include `worker_ack_pending` / `acked_workers` / `pending_workers`. Admin tooling polls `GET /v1/configs/models/{id}/status` on each gateway replica instead (§6).

7. **Bundles are filesystem-only on `sie-config`.** There is no bundle-write API. Bundles change by redeploying the `sie-config` image (and the worker image, since adapters referenced in a bundle live in the worker). The gateway no longer bakes bundles — it fetches them from `sie-config` at bootstrap and re-fetches on any `bundles_hash` drift (§4.2) — so a bundle addition propagates to running gateway replicas within one `DEFAULT_POLL_INTERVAL` without a gateway redeploy. The natural deploy order is `sie-config` first (so the new bundle is advertised), then workers for the new bundle (so routable instances exist when the gateway learns about the bundle).

   A release can also add an adapter to an existing bundle. Once the new `sie-config` is running, a worker still on the earlier image reports the new bundle hash with the new adapter's models in `unsupported_models` (§7), so it keeps serving every other model of the bundle. A new adapter's models become routable in a lane once every worker in that lane that reports the hash can serve them, typically when the rollout has replaced the earlier workers; until then they get the retryable `503 PROVISIONING` response. This needs the earlier workers to run a release that already reports `unsupported_models`. Workers from before that release hash with their image's adapter list, so while they remain, the whole bundle is routable only on upgraded workers, as before. The first release with `unsupported_models` therefore protects the upgrade to the release after it, not its own upgrade. A release that removes an adapter from a bundle needs no special handling: `sie-config` stops exporting that adapter's models to the bundle, and workers of either image reproduce the hash.

   Side effect of this design: `sie-config` is a **cold-start** dependency for gateway replicas.

   - A fresh replica without a filesystem seed that cannot reach `sie-config` starts with zero bundles AND zero models. A replica seeded by `gateway.embeddedConfigs` / `gateway.configMap` starts with the seeded bundles and models instead.
   - Either way, a replica with `SIE_CONFIG_SERVICE_URL` set reports `503` from `/readyz` until its first export succeeds.
   - Typed inference requests (i.e., any caller that does not pin a pool) will return `404 model not found` until bootstrap completes.
   - The empty-registry fallback described in §2 only fires for callers that explicitly pin a pool via `X-SIE-Pool` — the proxy falls back to the caller-supplied bundle or `"default"`, so a warm pool stays reachable through a cold gateway.
   - An already-running replica is NOT affected by a `sie-config` outage: it keeps serving its last-known bundle set and receives model deltas over NATS when `sie-config` recovers.

   The `gateway.embeddedConfigs` / `gateway.configMap` Helm overlays statically seed the registry, so a seeded replica can serve the seeded catalog during a `sie-config` outage. They do not remove the readiness dependency while `SIE_CONFIG_SERVICE_URL` is set; only a deployment without `sie-config` (`config.enabled=false`, which leaves `SIE_CONFIG_SERVICE_URL` unset) becomes ready without it, at the cost of losing live bundle resync.

8. **`registry_unavailable` (503) on `sie-config`.** (This is **`sie-config`'s** `/readyz`, not the gateway's — same path string, different process.) If the `ModelRegistry` failed to initialize (e.g. malformed bundle YAML), `sie-config`'s `/readyz` returns 503 and every registry-dependent endpoint (`/v1/configs/models`, `/v1/configs/models/{id}`, `/v1/configs/bundles`, `/v1/configs/bundles/{id}`, `/v1/configs/resolve`, `POST /v1/configs/models`, `/v1/configs/export`) returns 503 with a structured `registry_unavailable` error body. `/epoch` is intentionally independent of the registry (it reads the epoch file directly), so the gateway's poller can still discover that `sie-config` is up but not serving — admin tooling should treat **`sie-config` `/readyz=503`** plus **`/epoch=200`** as "control plane is alive but wedged; inspect logs".

## 12. Interaction Rules (Invariants)

- Client SDKs call the gateway for inference, never `sie-config`.
- Admin/config tooling calls `sie-config` for writes.
- The gateway is not a config write authority. `POST /v1/configs/models` is not registered; requests get `405 Method Not Allowed`.
- `sie-config` does not proxy inference traffic.
- Workers do not become the config authority.
- NATS / JetStream is the runtime bus. Kubernetes coordinates pools, not inference.
- `sie-config` is the sole publisher on `sie.config.models.*`. The gateway is subscribe-only.
- The gateway's `ConfigEpoch` is not a source of truth; it is a local caching counter advanced by bootstrap, deltas, and the poller. `sie-config`'s `/epoch` endpoint is authoritative.
- `GET /v1/configs/export` is not an optimization. It is the only way a gateway can realign with `sie-config` after missed deltas.

## 13. Environment Variables

Gateway (`sie-gateway`):

- `SIE_BUNDLES_DIR`, `SIE_MODELS_DIR` — optional filesystem seed paths. Unset by default; only set by the `gateway.embeddedConfigs` / `gateway.configMap` Helm overlays which mount a ConfigMap at `/configs/{bundles,models}`. When unset, the registry's filesystem reload finds nothing and `state::config_bootstrap` fills both bundle and model state from `sie-config`.
- `SIE_CONFIG_SERVICE_URL` — base URL of `sie-config`. When set, `/readyz` returns `503` until the first complete export is applied. If unset, the bootstrap and poller tasks no-op and the gateway runs with whatever the (optional) filesystem seed loaded — typically empty in the default deploy, which means **no models will be served**. This is intended for local single-process tests only.
- `SIE_CONFIG_SERVICE_TOKEN` — bearer token the gateway-as-client presents on its `sie-config` reads (`GET /v1/configs/bundles`, `/bundles/{id}`, `/export`, `/epoch`). It should be `sie-config`'s read-scoped token (`SIE_CONFIG_READ_TOKEN` there), which the chart generates and always wires in (empty when the chart runs no `sie-config`). When the variable is set it decides, and a blank value sends no `Authorization` header (so `sie-config` must either be unauthenticated or the gateway will fail with `401`/`403`). When it is unset, the gateway falls back to `SIE_ADMIN_TOKEN` and, when `SIE_CONFIG_SERVICE_URL` is set, logs a deprecation warning at startup; it logs the same warning when the two are equal. `sie-config` accepts one read token at a time and every component reads it at start, so a rotation restarts `sie-config` first and then the gateways and worker sidecars; until they restart, they receive `403` on config reads.
- `SIE_ADMIN_TOKEN` — the token that `AuthLayer` requires for admin-gated mutations on the gateway itself (`POST/PUT/DELETE` on `/v1/configs/*`, `/v1/admin/*`, `/v1/pools/*`). If empty and the matching inbound request targets an admin path, the middleware fails closed with `403`. It is not sent to `sie-config` unless `SIE_CONFIG_SERVICE_TOKEN` is unset (the deprecated fallback above). The chart wires it only from `gateway.auth.adminTokenSecretName`, never from the `sie-config` admin token.
- `SIE_AUTH_TOKEN` / `SIE_AUTH_TOKENS` — tokens accepted by the gateway's own inference API (`/v1/encode`, `/v1/score`, `/v1/extract`) and for read-side pool/config routes. Not used for outbound calls to `sie-config`. When `SIE_AUTH_MODE` enables auth but this list is empty, every non-probe request returns `500`.
- `SIE_AUTH_MODE` — `token` (alias: `static`) enforces auth; `none` (default) disables it. Typos are fail-open-to-bypass by design; `audit_auth` logs a startup error naming the bad value so operators see it in `kubectl logs`.
- `SIE_AUTH_EXEMPT_OPERATIONAL` — when `true`, `/`, `/health`, and `/ws/*` are exempt from auth (they expose worker URLs, bundle assignments, queue depth, GPU inventory — treat as sensitive). Default `false`. **`/healthz` and `/readyz` are always exempt from auth** so kubelet probes never fail with **`401`/`403`** because of a missing bearer token. `/readyz` reports only process readiness and, when `SIE_CONFIG_SERVICE_URL` is set, first-bootstrap completion; worker health remains visible through `/health` and inference responses.
- `SIE_NATS_CONFIG_TRUSTED_PRODUCERS` — comma-separated producer allowlist for `sie.config.models._all`. Binary default is `sie-config`; Helm sets the release-scoped config Deployment name. Matching is exact OR K8s pod-name prefix (see §4.3). Untrusted notifications are dropped; the `config_poller` still closes the gap.
- `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER` — `true` disables producer validation entirely. Intended for local/dev. `main.rs` emits a startup warning when on.
- `SIE_NATS_URL` — NATS broker URL. Credentials in the URL are not used and are redacted in logs and `Config` debug output.
- `SIE_NATS_USER` / `SIE_NATS_PASSWORD` — NATS credentials. Both or neither; one alone fails startup. The password is redacted in `Config` debug output.

Config service (`sie-config`):

- `SIE_ADMIN_TOKEN` — write-auth, and also accepted on every read. Without it, writes are rejected if a read-scoped token (`SIE_CONFIG_READ_TOKEN` or `SIE_AUTH_TOKEN`) is set (a read token cannot implicitly grant write access). If no token is set, reads and writes are accepted unauthenticated — dev/local only; production always sets `SIE_ADMIN_TOKEN`.
- `SIE_CONFIG_READ_TOKEN` — read-auth for the gateways and worker sidecars, which present it as `SIE_CONFIG_SERVICE_TOKEN`. Accepted on every read route, including `GET /v1/configs/export`, and never on a write.
- `SIE_AUTH_TOKEN` — read-auth (the inference token) on every read route except `GET /v1/configs/export`, which holds the write lock and accepts only `SIE_CONFIG_READ_TOKEN` or `SIE_ADMIN_TOKEN`. Optional; the chart does not set it.
- `SIE_NATS_URL` — NATS broker URL. Publisher degrades gracefully if unreachable; mutations are blocked with `503` while the publisher is configured but disconnected. Logs show the URL with any userinfo redacted.
- `SIE_NATS_USER` / `SIE_NATS_PASSWORD` — NATS credentials (Helm sets user `sie-config`, the only user allowed to publish on `sie.config.models.>`). Both or neither; one alone fails startup.
- `SIE_CONFIG_STORE_DIR` — base directory for the on-disk config store. When the Helm `config.configStore.enabled` flag is true, this is set to the mounted PVC path and `SIE_CONFIG_RESTORE=true` enables startup replay from the store into the in-memory registry.
- `SIE_BUNDLES_DIR`, `SIE_MODELS_DIR` — bundle and model source directories. **`sie-config` is the source of truth**: it reads these at startup, validates writes against them, and re-serves the bundle list at `GET /v1/configs/bundles` for the gateway's bootstrap. Defaults via `sharedPaths.{bundlesDir,modelsDir}` (`/app/bundles`, `/app/models`) and image-baked.
- `SIE_UPSTREAM_NAMES` — comma-separated names of the upstreams the remote workers define. When it is set, even to an empty value, a write whose remote profile names any other upstream is rejected with `422` (§3). When it is unset, `sie-config` does not check upstream names and a remote worker rejects an undefined one when it applies the config. The Helm chart sets it from the same `upstreams` values as the remote lanes' upstreams file when a remote lane is enabled, and to an empty value otherwise.
- `SIE_LOG_LEVEL`, `SIE_LOG_JSON` — log verbosity and structured-JSON toggle (both also exposed as CLI flags).

Tracing (all runtimes):

- `SIE_TRACING_ENABLED` and an OTLP endpoint are both required before spans are exported by the gateway, worker sidecar, Python worker, or Rust worker. The trace-specific `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` takes precedence over the generic `OTEL_EXPORTER_OTLP_ENDPOINT`; whitespace-only values are ignored. All four runtimes install the W3C trace-context propagator even when the exporter is off, so inbound `traceparent` / `tracestate` can still flow through queue envelopes.
- `OTEL_SERVICE_NAME` controls the service name shown in the tracing backend. Helm defaults are `sie-gateway` for the gateway, `sie-worker-sidecar` for the sidecar, and `sie-server` for the worker container. If the Rust worker runs without the Helm-provided env, its binary default is `sie-server-rust`. Managed launchers set each child process's identity explicitly so gateway, Python worker, and sidecar signals remain distinct.
- `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG` configure head sampling. Helm and the managed Modal endpoint secret set them to `parentbased_traceidratio` / `0.05`, so an inbound sampled decision is honored and new traces are sampled at 5%.
- Every signal resolves its signal-specific endpoint before the generic OTLP endpoint and never inherits from another signal. Where transport is selectable, the signal-specific protocol likewise precedes the generic protocol. Dual-transport exporters accept only exact case-insensitive `grpc` and `http/protobuf`: `grpc` is the chart default, while `http/protobuf` is the managed Modal path because its collector edge publishes only OTLP/HTTP. The managed-only dispatcher requires explicit HTTP protobuf, and the provisioner supplies explicit trace, metric, and log settings. Unsupported values warn and disable only that signal; serving remains fail-open. The gateway, Python worker, and Rust sidecar attach the all-or-nothing `Modal-Key` / `Modal-Secret` pair only when managed proxy auth is enabled and the effective signal endpoint exactly matches the provisioner-recorded canonical `*.modal.run` HTTPS origin and OTLP path. Authenticated exporters do not follow redirects; an untrusted traces or metrics override disables that signal without stopping inference. The Python worker also has a local-dev `--tracing` CLI flag, which enables tracing to `localhost:4317`.
- Resources always include `deployment.environment` and `cloud.region`; deployments inject concrete values and local runs use `unknown`.
- Helm `observability.tracing.enabled` injects the exporter env block on gateway, worker, and sidecar containers. Rendering fails fast when no endpoint can be resolved from `observability.tracing.endpoint`, the bundled collector, or bundled Tempo.

## 14. Endpoint Reference

### `sie-gateway`

Inference (queue-only):

- `POST /v1/encode/{*model}`
- `POST /v1/score/{*model}`
- `POST /v1/extract/{*model}`
- `POST /v1/embeddings` — OpenAI JSON compatibility layer over encode (string or list of strings); see §2.

Config (read-side only):

- `GET /v1/configs/models`
- `GET /v1/configs/models/{*id}` — model YAML, or worker-ack status for `…/status` suffix.
- `GET /v1/configs/bundles`
- `GET /v1/configs/bundles/{id}`
- `POST /v1/configs/resolve`

Pools and operator:

- `GET /v1/pools`, `POST /v1/pools`, `GET /v1/pools/{name}`, `POST /v1/pools/{name}/renew`, `DELETE /v1/pools/{name}`
- `GET /v1/models`, `GET /v1/models/{*model}` — model catalogue. JSON objects mirror `sie_server` **`ModelInfo`** shape (including `inputs`, `outputs`, `dims`, `profiles`, worker-derived `loaded` / `state`, optional `last_error`, …). An unknown `model` id returns **`404`** with **`{"detail":{"code":"MODEL_NOT_FOUND",...}}`** (same envelope style as FastAPI validation errors on `sie_server`).
- `GET /healthz`, `GET /readyz`, `GET /health`
- `GET /ws/cluster-status`
- `GET /` — HTML status page (operator-facing, not a programmatic API).

Not registered on the gateway: `POST /v1/configs/models`, `GET /v1/configs/export`, `GET /v1/configs/epoch`. The export and epoch endpoints are consumed by the gateway as a client of `sie-config`.

### `sie-config`

Config writes (admin auth: `SIE_ADMIN_TOKEN`):

- `POST /v1/configs/models`
- `PUT /v1/configs/models/{model_id:path}`
- `DELETE /v1/configs/models/{model_id:path}`

Config reads (read auth: `SIE_CONFIG_READ_TOKEN`, `SIE_AUTH_TOKEN`, or `SIE_ADMIN_TOKEN`):

- `GET /v1/configs/models`
- `GET /v1/configs/models/{model_id:path}`
- `GET /v1/configs/bundles`
- `GET /v1/configs/bundles/{bundle_id}`
- `POST /v1/configs/resolve`
- `GET /v1/configs/epoch`
- `GET /v1/configs/export` — `SIE_CONFIG_READ_TOKEN` or `SIE_ADMIN_TOKEN` only

`sie-config` does not serve `GET /v1/configs/models/{id}/status`. That endpoint is gateway-only by design (no worker registry on `sie-config`).

## 15. Telemetry

The gateway emits each declared observation once through the canonical
OpenTelemetry facade and exports OTLP to a collector. It does not link a
Prometheus client, host an application `/metrics` route, or choose a backend.
The collector owns Prometheus exposition for OSS/KEDA and OTLP routing for
remote telemetry. The checked-in source of truth is `telemetry/contract.yaml`.

The gateway contract contains request count/duration/admission, five lane-value
families (pending demand, durable lane queue depth, active lease GPUs, pool warm
floor, and scale-worthy rejected requests), a local capacity-reconciliation
timestamp, and a same-label queue-snapshot timestamp.
KEDA state refreshes independently of HTTP at least every five seconds.
The physical catalog is frozen for the bounded process lifetime; physical lane
labels are `pool`, `machine_profile`, and `bundle`. For every catalog lane, the gateway reads the
exact `WORK_POOL_{pool}` durable consumer and records
`num_pending + num_ack_pending`; the latter keeps work visible while a worker
is loading a model or dies before ACK. The configured consumer name and exact
`sie.work.{pool}.{machine_profile}.{bundle}.*` filter are verified before any
value is emitted. `StreamNotFound` is a safe zero; `ConsumerNotFound` uses an
exact-subject retained-work check so another lane's consumer cannot hide
orphan work. A timeout, auth failure, filter mismatch, or counter overflow
updates neither that lane's queue value nor its freshness. Successful lanes and
the independent pending-demand, active-lease, and warm-floor families still
refresh, so one corrupt consumer ages out only its own KEDA input. The reader is
bounded by the 1,024-entry deployment catalog, 32 concurrent queries, and a
four-second whole-snapshot timeout. Each gateway hashes its process-start UUID
to a stable offset within the five-second cadence, preventing rollout-aligned
replicas from scanning in one burst. Helm caps metrics-enabled gateway replicas
at ten, and the live broker benchmark covers the resulting maximum 10,240
lookups per interval. Worker identity and heartbeat queue depth never enter
this KEDA series.

Prometheus target labels bind every query to one Helm release; the queue gauge
is sent to Better Stack under its canonical dotted name and naturally exports
as `sie_gateway_lane_queue_depth`. Each business series
must match a fresh timestamp from the same collector-authored
`producer_instance` and Prometheus-only `collector_generation` before
replica-safe aggregation, so exporter retention across a gateway restart
cannot revive stale capacity state and a collector accumulator restart cannot
silently continue an old counter series. The queue query uses its same-label
broker timestamp and requires a fresh queue sample count, distinguishing a
successful zero from a missing point. Broker, pool, and demand reads are
independently reconciled and OTel gauge collection can overlap recording, so
cross-family state is eventually consistent rather than export-atomic.
Reconciliation start time is captured before those reads and recorded after
the values; slow builds therefore age out, and the shared OSS/managed loop
skips missed ticks.

Resource identity always includes `service.name`, `service.instance.id`,
`deployment.environment`, and `cloud.region`. Deployments inject concrete
environment/region values; local runs use `unknown`. Ordinary point attributes
are bounded by the contract, and invalid catalog values collapse to `other`.
KEDA physical-lane attributes are stricter: they must resolve to an exact typed
catalog member, and unresolved control observations are omitted rather than
assigned a synthetic fallback lane.

### 15.1 Distributed tracing

Tracing uses W3C `traceparent` / `tracestate` propagation through the gateway queue envelopes described in §8.2 (`injects_queue_trace_context`). The SDKs are not instrumented today: a trace roots at the gateway unless the caller's own HTTP stack injects a context.

Non-streaming queue path (`/v1/encode`, `/v1/score`, `/v1/extract`, `/v1/embeddings`):

```
optional client traceparent
  -> gateway.publish
  -> sidecar.dispatch
  -> worker.run_batch
```

`/v1/embeddings` is rewritten to `/v1/encode`, and the gateway explicitly forwards `traceparent` / `tracestate` onto the queue work item. `gateway.publish` is opened only when the exporter is active; otherwise the gateway scopes the inbound context over publish so downstream workers can still attach to the caller's trace. The sidecar opens `sidecar.dispatch` around the coalesced batch, then serializes that span back onto IPC items so `worker.run_batch` nests under it. If the sidecar exporter is off, it falls back to the original gateway context.

On multi-GPU worker pods, this same sidecar hop is where non-generation work is assigned to the model's placed child GPU before IPC dispatch; `/v1/embeddings` inherits that behavior after the encode rewrite.

Generation path (`/v1/generate`, `/v1/chat/completions`):

```
optional client traceparent
  -> gateway.proxy_generate or gateway.proxy_chat
  -> worker-sidecar IPC ProcessGenerate
  -> worker.streaming_processor
```

Streaming generation goes through the worker-sidecar. The gateway opens `gateway.proxy_generate` (the OTel-renamed export of the internal `gateway.proxy` tracing span) for native generation or `gateway.proxy_chat` for OpenAI-compatible chat, injects that context into the streaming work item, and publishes the item to the worker-direct generation stream. The sidecar receives the generation item, resolves any offloaded payload, selects the model's placed child GPU, forwards `ProcessGenerate` over that child's IPC socket, and publishes Python's streaming events back to the gateway reply subject. The Python worker extracts the gateway trace context before `worker.streaming_processor`.

Multi-GPU caveat: one hot generation model is bound to one child at a time and is not replicated across children. Multiple generation models can be placed across children; the same model stays sticky to its child unless that child becomes unready or fails. Sharding happens inside a child: in a pool with `gpu.deviceGroup: true`, the single child owns every GPU in the pod, and a model whose profile declares `tensor_parallel_size` runs one engine across that many of them. The gateway and sidecar still see one child.

Batch fan-in parents on the first valid inbound context and records the remaining distinct valid contexts as OpenTelemetry links. The sidecar does this before `sidecar.dispatch`; Python and Rust workers do the same before `worker.run_batch`. The propagator is installed even when export is off, so trace context flows with tracing disabled. Runtime shutdown uses a bounded ~3 s trace flush. The Helm default sampler is `parentbased_traceidratio` at `0.05`, which preserves inbound sampling decisions and samples 5% of new roots.

Current span attributes:

| Span | Runtime | Attributes |
| ---- | ------- | ---------- |
| `gateway.publish` | Rust gateway, non-streaming queue publish | `sie.endpoint`, `sie.model`, `sie.pool`, `sie.publish_ms` |
| `sidecar.dispatch` | Rust worker sidecar | `sie.op`, `sie.model`, `sie.batch_id`, `sie.batch_size` |
| `worker.run_batch` | Python worker | `sie.model`, `sie.batch_id`, `sie.op`, `sie.adapter`, `sie.batch_size` |
| `worker.run_batch` | Rust worker | `sie.op`, `sie.model`, `sie.batch_id`, `sie.batch_size` |
| `gateway.proxy_generate` | Rust gateway, `/v1/generate` | `sie.endpoint`, `sie.request_id`, `sie.model` |
| `gateway.proxy_chat` | Rust gateway, `/v1/chat/completions` | `http.route`, `sie.routing_key_kind`, `sie.model`, `sie.request_id` |
| `worker.streaming_processor` | Python worker, streaming generation | `sie.request_id`, `sie.attempt_id`, `sie.model`, `sie.adapter` |

Operationally, §16.2 shows where the queue envelope is created in the request walk-through. The chart README's "Distributed Tracing (OTLP)" section documents the Helm enable switch. When tracing dashboards are enabled, traces surface in Grafana through the Tempo datasource (`uid: tempo`) and the `SIE Tracing` dashboard.

## 16. Worked Example: end-to-end request

This section walks a concrete `encode` request through the system so the moving parts in §1–§15 have an explicit reference trace.

### 16.1 Bundles and model → bundle resolution

A **bundle** defines which adapter module identifiers a worker deployment can run. The principal shapes are:

**`default` bundle** (priority `10`) — ~37 adapters for smaller BERT-class and cross-encoder models:

```yaml
# bundles/default.yaml
name: default
priority: 10
adapters:
- sie_server.adapters.bert_flash
- sie_server.adapters.bge_m3_flash
- sie_server.adapters.clip
- sie_server.adapters.colbert
- sie_server.adapters.sentence_transformer
- sie_server.adapters.splade_flash.adapter
# … 30+ more
```

**SGLang task-class bundles** — three logical worker lanes share one pinned
SGLang dependency stack and physical image while keeping generation, embedding,
and vision extraction queues and autoscaling independent:

```yaml
# bundles/sglang.yaml
name: sglang
priority: 20
adapters:
- sie_server.adapters.sglang.generation

# bundles/sglang-embedding.yaml
name: sglang-embedding
priority: 21
adapters:
- sie_server.adapters.sglang.embedding

# bundles/sglang-vision-extract.yaml
name: sglang-vision-extract
priority: 22
adapters:
- sie_server.adapters.sglang_vision_extract.adapter
```

The three bundles stay dependency-pin compatible. Kubernetes and Modal may reuse
the `sglang` image for the two non-generation logical bundles, but routing,
worker identity, metrics, queue subjects, and scaling always use the logical
bundle name.

**`candle` bundle** (priority `30`) — a Rust `sie-server-rust` IPC worker backed by native Candle execution:

```yaml
# bundles/candle.yaml
name: candle
priority: 30
engine: candle
adapters:
- sie_server_rust.adapters.candle
```

`ModelRegistry` maps a requested model/profile identity to a bundle by resolving the requested profile's `adapter_path` module, matching that module against each bundle's `adapters` list, then preferring the lowest-priority bundle in the match set. The base model id means the `default` profile; non-default profiles are routable as `base-model:profile` aliases. Native Candle lanes use explicit `:candle` identities for both side-by-side variants and Candle-only models unless the product intentionally introduces a separate public alias:

```
Model: BAAI/bge-m3
  adapter_path: sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter
  adapter module: sie_server.adapters.bge_m3_flash
  → matches "default" bundle
  → routes to "default" workers

Model: Qwen/Qwen3-Embedding-4B
  adapter_path: sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter
  adapter module: sie_server.adapters.sglang.embedding
  → matches "sglang-embedding" bundle
  → routes to "sglang-embedding" workers

Model: intfloat/e5-small-v2:candle
  base model: intfloat/e5-small-v2
  profile: candle
  adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
  adapter module: sie_server_rust.adapters.candle
  → matches "candle" bundle
  → routes to "candle" workers
```

The `engine` field on a bundle describes the worker runtime shape and can be used as an operator/debug filter, but normal product routing should choose a profile variant or explicit compatible bundle rather than asking application clients to know engine headers.

This mapping is computed at startup and whenever the model registry reloads (filesystem hot reload, NATS config delta, or successful bootstrap/poll). No per-model hand-wiring.

### 16.2 A single request from client to response

```bash
curl -X POST https://gateway/v1/encode/BAAI/bge-m3 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/msgpack" \
  --data-binary @request.msgpack
```

**Step 1 — Gateway resolves routing** (`handlers/proxy.rs`):

```
Model:    BAAI/bge-m3
Bundle:   default          (bge_m3_flash is in the default bundle)
Pool:     default          (no X-SIE-POOL header, no machine-profile override)
Machine:  default          (no X-SIE-MACHINE-PROFILE header)
Subject:  sie.work.default.default.default.BAAI__bge-m3
Stream:   WORK_POOL_default
```

The work subject is constructed by `work_subject(pool, machine_profile, bundle, model)` in `queue/publisher.rs`, which runs the lane tokens and model id through `normalize_model_id` (the Rust mirror of `sie_sdk.queue_types.normalize_model_id`: `/` → `__`, `.` → `_dot_`, `*`/`>`/space → `_`). The result is always exactly six dot-separated tokens so it matches the worker's `sie.work.{pool}.{machine_profile}.{bundle}.*` consumer filter — this is the guarantee that makes dotted ids like `vidore/colqwen2.5-v0.2` work. The DLQ path extracts the sixth token from the advisory subject, applies the same compatibility slash normalization, and publishes on `sie.dlq.{model_normalized}` — so `sie.work.default.default.default.BAAI__bge-m3` becomes `sie.dlq.BAAI__bge-m3` when a message hits max-deliveries.

**Step 2 — Gateway publishes work items to JetStream** (`queue/publisher.rs`). For a 2-item request the gateway publishes two msgpack work items:

```
WorkItem {
  work_item_id:  "abc-123.0",
  request_id:    "abc-123",
  item_index:    0,
  total_items:   2,
  operation:     "encode",
  model_id:      "BAAI/bge-m3",
  profile_id:    "default",
  display_model: "…",                       // the model the caller asked for; only when it differs from model_id
  pool_name:     "default",
  admission_pool: "default",
  machine_profile: "default",
  item:          { "text": "..." },       // or omitted + payload_ref if >1MB
  reply_subject: "_INBOX.gw-1.abc-123",
  timestamp:     1700000000.0,
  deadline:      1700000120.0,             // timestamp + request timeout; omitted when unknown
  accepts_result_chunks: true,              // supports result_chunk_v1 only; absent/false keeps one-shot
  bundle_config_hash: "a1b2c3…",
  …
}
```

JetStream publish submission is asynchronous: the gateway buffers publishes to the `async-nats` client and returns an owned durability future, so the request handler does not block serially on per-item JetStream ACK round-trips. Queue APIs admit at most 4096 items and overlap at most 64 initial sends. The one pending-demand handoff task drains the bounded ACK set under a six-second aggregate deadline. Ack outcomes are logged and recorded once as `sie.gateway.queue.events{event="publish_ack",outcome="success|ack_error"}`; the collector exposes that counter as `sie_gateway_queue_events_total`. Backpressure and no-consumer conditions are caught earlier via a cached stream-info check (`stream_info_cache`), so a stalled stream surfaces as a prompt `503` rather than a stuck request.

In parallel, the gateway registers a `ResultCollector` in its `DashMap<String, ResultCollector>` keyed by `request_id` and subscribes (once, app-wide) to `_INBOX.{router_id}.>` on NATS Core.

**Step 3 — Worker pulls and processes** (`sie-server`):

A worker in the `default` pool pulls the message from `WORK_POOL_default`:

1. Deserializes the msgpack payload.
2. Reads `model_id` and `operation`.
3. Checks whether `BAAI/bge-m3` is already loaded on the GPU; if not, loads it lazily.
4. Runs `BGEM3FlashAdapter` with the input text.
5. Produces a 1024-dimensional embedding (numpy array).
6. Publishes the msgpack-encoded result (with `msgpack-numpy` for the array) on `_INBOX.gw-1.abc-123` (NATS Core).
7. Acks the JetStream message.

Workers are multi-model: one `default` worker can serve BGE-M3, a cross-encoder, CLIP, etc. in interleaved fashion, loading and caching on demand.

**Step 4 — Gateway assembles the response** (`queue/publisher.rs`, `handlers/proxy.rs`):

On each inbox message, the gateway first runs `extract_request_id_fast` — a bounded scan of the raw msgpack bytes that locates the `request_id` (or, for array-shape `WorkResult` payloads, the first string field, which the current codec places at the `request_id` / `work_item_id` position) without a full deserialization. If the extracted ID is not present in the `pending_results` `DashMap`, the message is dropped immediately (already completed, duplicate redelivery, or stale). Only when it is live does the handler decode the payload. An exact `kind: result_chunk_v1` discriminator enters the v1 decoder and malformed exact-v1 envelopes fail closed; ordinary or future `WorkResult` maps remain compatible even if they add fields such as `total_bytes`. V1 accepts out-of-order and byte-identical duplicate chunks, and rejects identity/metadata conflicts, bad length/digest, more than 64 chunks, or more than 16 MiB per item. Chunk zero with a new digest replaces stale retry state. Chunk zero may also safely restart the same digest with a new canonical chunk count; bounded digest/layout tombstones stop delayed fragments from an abandoned attempt from mixing with or flipping the replacement. Unknown non-zero retry fragments are ignored.

Decoding and handling are separate stages. The subscription feeds one decode task, which does the fast-path skip and the single msgpack decode (`inbox_work_for_payload`) and then hands the typed envelope to a bounded worker pool keyed by request id (`queue/keyed_worker_pool.rs`). A request id always maps to the same shard and a shard drains its channel in order, awaiting each handler, so a request's messages are handled in arrival order — which streaming depends on, since a reordered `seq` is dropped as a duplicate or fails the stream as a `transport_failure` gap. Different requests take different shards and proceed concurrently. When a shard is saturated the decode task waits for a slot rather than dropping a result or a chunk; that degrades to the pre-pool behaviour of handling everything inline and is counted as `sie_gateway_queue_worker_pool_events_total{pool="inbox",event="saturated"}`. Completion-time object-store deletes do not run on either stage: `handle_result` and the streaming terminal funnel queue them onto a second pool (`pool="payload_cleanup"`), which sheds at its bound because the periodic `cleanup_expired` reconcile already retries orphaned keys.

Memory is admission-controlled before a transfer allocates its chunk vector. Each item conservatively reserves three times its encoded length (partial fragments, contiguous reassembly/decode copy, and the decoded result retained until all request items complete) plus 4 KiB of bookkeeping headroom. Reservations for accepted completed chunked items are held until request completion; completed results rejected by first-result or stale direct-fallback semantics release their reservation immediately. Reservations are capped at 64 MiB per request and 256 MiB gateway-wide. The process-wide ceiling is one eighth of the gateway's 2 GiB container limit, leaving 1.75 GiB for normal response buffers, JSON expansion, NATS, allocator overhead, and the rest of the process. Request-collector RAII releases reservations on success, validation failure, timeout, and publish failure. The current reservation is exposed as `sie_gateway_queue_result_chunk_reserved_bytes`; global exhaustion increments `sie_gateway_queue_result_chunk_rejections_total{reason="global_budget"}`. Invalid transfers terminate the pending request with a static, non-retry-hinted `503 transport_failure` instead of waiting for timeout.

The inbox remains a trusted in-cluster worker boundary, as it already was for one-shot `WorkResult`: a publisher that can write to another gateway's private inbox can name a live request. Chunk validation fails such a request closed, but never reflects the supplied ids, payload, or validation detail to the caller; the public message is the static transport error above.

Results are stored by `item_index` in the `ResultCollector`, and a oneshot completes when all items have arrived. The gateway then converts the collected msgpack results into the client-requested format — pass-through for `Accept: application/msgpack` or transcoded to JSON (with numpy-array expansion) for `Accept: application/json`.

Response headers include timing information:

```
X-SIE-Version: 0.2.0
X-SIE-Server-Version: 0.2.0
X-SIE-Request-Id: abc-123
X-SIE-Worker: <worker-id>
X-Queue-Publish-Time: 2.1
X-Queue-Wait-Time: 14.6
X-Queue-Time: 15.3
X-Inference-Time: 12.8
X-Tokenization-Time: 1.2
X-Postprocessing-Time: 0.4   # optional; only when the worker reports it
X-Payload-Fetch-Time: 0.0    # optional; only when payload-ref indirection was used
```

(HTTP header names are case-insensitive; the server emits them lowercase.)

### 16.3 Pools and capacity isolation

A **pool** is a group of GPU workers reserved for a specific workload — the answer to "I need guaranteed GPU capacity that isn't shared with other traffic." Every cluster has a `default` pool that uses all available workers. Custom pools are opt-in:

```
POST /v1/pools
{
  "name": "customer-acme",
  "gpus": {"l4-spot": 2},
  "gpu_caps": {"l4-spot": 4},
  "bundle": "sglang",
  "ttl_seconds": 3600
}
```

`gpus` is required capacity; `gpu_caps` is an optional assignment/view cap for that pool. `gpu_caps` is not an aggregate quota across several logical pools sharing the same backing queue; use separate physical queue pools for hard capacity separation. `bundle` is an optional assignment filter: when omitted, the logical pool may span multiple worker bundles and `status.assigned_workers[]` records each worker's actual bundle so `active_lease_gpus` can stay lane-aware. The `default` pool has zero requirements and no caps. The worker-sidecar polls pool status every 10 seconds before NATS pulls when the admission gate is enabled. Capped named physical pools fail closed unless this pod appears in `status.assigned_workers`; the `default` pool fails open during transient gateway/status errors so baseline capacity remains available. For logical API pools backed by a shared physical queue, the sidecar also tracks assigned logical pools from `GET /v1/pools` and NAKs work whose `admission_pool` does not assign this worker. HA gateways sort workers deterministically and persist named-pool assignment status.

Helm resolves every enabled worker entry through one canonical physical-lane
tuple `(queuePool, machineProfile, bundle)`, then reuses it for worker/sidecar
environment, heartbeat identity, Kubernetes labels/selectors, gateway
configured profiles, and KEDA queries. Tokens are trimmed, limited to 63
characters, validated against `^[A-Za-z0-9_-]+$`, and lowercased; explicit
blank values and post-normalization tuple collisions fail chart rendering.
Pool and bundle map keys remain stable Kubernetes/KEDA identities and are
validated separately as lowercase DNS-1123 labels.
Helm publishes those canonical machine profiles in
`SIE_GATEWAY_CONFIGURED_GPUS`, request aliases in
`SIE_GATEWAY_GPU_ALIASES`, and optional deploy-owned static queue pools in
`SIE_GATEWAY_STATIC_QUEUE_POOLS`.
It also publishes the exact tuple set in
`SIE_GATEWAY_CONFIGURED_PHYSICAL_LANES` (maximum 1024). The gateway resolves
every scale-driving observation against this catalog before it can enter the
pending-demand tracker or rejection metric. Unresolved caller input is omitted,
not relabeled into a fake KEDA lane. The tracker stores at most the finite
catalog, refreshes deadlines in place, and lets the five-second snapshot prune
expiry; request cardinality therefore cannot create entries or Tokio tasks.

Each physical queue pool gets its own JetStream stream (`WORK_POOL_{queue_pool}`) with subjects `sie.work.{queue_pool}.*.*.*`. Pool and queue-pool names are validated against `[A-Za-z0-9_-]` and canonicalized to lowercase at the API, gateway request, K8s restore/watch, and Helm render boundaries, so NATS subjects and KEDA labels do not split by case. Workers are deployed with `SIE_POOL={queue_pool}`, `SIE_MACHINE_PROFILE={profile}`, and `SIE_BUNDLE={bundle}`; they consume only from their concrete `sie.work.{queue_pool}.{profile}.{bundle}.*` lane. A logical API pool names the tenant/workload boundary and carries `PoolSpec.queue_pool` to select the physical queue lane; omitted `queue_pool` means `default`. Helm's baseline default puts enabled worker groups in `SIE_POOL=default`, so SDK calls can pass just the machine profile (`gpu="l4"`) for shared-cluster capacity, and API-created pools can borrow that same base capacity with `gpu="pool/l4"`. Dedicated queue deployments override Helm `queuePool` and must declare the same name under `queueRouting.staticQueuePools` before API pools can use it as a non-default `queue_pool`; callers can target the protected static pool directly or create a differently named logical pool backed by that queue, for example `{"name":"customer-acme-workload","queue_pool":"customer-acme"}` plus SDK `gpu="customer-acme-workload/l4"`. A cold `X-SIE-POOL` request without `X-SIE-MACHINE-PROFILE` records lane-specific `pending_demand` for every machine profile the pool can provision. Multi-profile pool demand therefore wakes all candidate lanes from zero instead of relying on an empty `machine_profile` label that no per-lane KEDA ScaledObject can observe. Registration does not clear demand: only a successful gateway work publish clears the exact target lane; other candidate lanes retain demand until work is published to them or the entry expires. This preserves scale-up pressure when registered workers are saturated.

A pool spec may also carry `minimum_worker_count` (a per-pool warm floor, default `0`). It keeps the first request from hitting a cold VM: every five seconds the gateway re-emits `sie.gateway.pool.warm_floor{pool,machine_profile,bundle}` for each backing queue lane the pool can name (clearing lanes that disappear), and a per-lane KEDA trigger reads the collector's Prometheus translation with `or vector(0)` so KEDA keeps machines warm. When multiple logical pools share one backing lane, the emitted value is the maximum requested floor for that physical lane, not a sum. Active lease capacity is labeled the same way. Default `0` emits nothing and leaves scale-from-zero unchanged.

A pool spec may also carry `pinned_models` (a per-pool set of always-loaded models, default empty), chosen from the models the gateway already tracks (`GET /v1/configs/models`); ids may be profile-qualified (`model-name:profile_name`, the registry's `{base}:{profile}` variant entries, with `model:default` folding to the base). The API validates each id against the registry and stores it canonicalized. Every 30 seconds the gateway emits the complete `sie.gateway.pool.pinned_model.loaded{pool,model}` snapshot through its canonical OTel facade: the value is `1` only when a healthy worker assigned to that logical pool reports the model loaded, and removed pools or pins are explicitly zeroed. Actually keeping the models loaded (worker preload) and lazy-load/LRU using spare capacity around the pinned set are worker-side and tracked separately. Default empty leaves lazy-loading unchanged.

Creating a usable custom pool has two supported shapes. For a dynamic logical pool over base capacity, call `POST /v1/pools` with the logical name and GPU requirements and omit `queue_pool`; it defaults to `default`, so no additional Helm worker group is required. Pool `name` is user/dynamic. Non-default `queue_pool` is infra/admin: it must name a Helm-rendered `queuePool`/`SIE_POOL` backing lane declared under `queueRouting.staticQueuePools`. For a dedicated physical queue, first add that worker group to Helm values and declare the same name under `queueRouting.staticQueuePools`, then create logical pools with `queue_pool` set to that queue. Lease-owned dynamic pools expire after their TTL unless renewed with `POST /v1/pools/{name}/renew`. The gateway bounds API-created pools independently of auth: a `minimum_worker_count` whose total across the pool's machine profiles, or `gpus` requirements whose sum, exceeds `SIE_GATEWAY_POOL_MAX_MINIMUM_WORKER_COUNT` (default 4) and `ttl_seconds` above `SIE_GATEWAY_POOL_MAX_TTL_S` (default 3600) are rejected with `400 INVALID_REQUEST`, and a create beyond `SIE_GATEWAY_MAX_POOLS` live API-created pools (default 64) is rejected with `403 POOL_OPERATION_FORBIDDEN`. Pools restored from Kubernetes or applied from another replica are held to the same TTL and warm-floor caps when their lease expiry and KEDA warm floor are computed. An API-created pool's active lease keeps at most its `gpus` requirement per machine profile warm, and at most `SIE_GATEWAY_POOL_MAX_MINIMUM_WORKER_COUNT` in total, however many workers it is assigned; a stored pool's per-lane warm floor is capped so its total fits the same budget. The live-pool cap is checked by the replica serving the create against every pool it knows, so concurrent creates on different replicas can briefly exceed it. `POST /v1/pools` refuses the name `default` with `403 POOL_OPERATION_FORBIDDEN`. The `default` pool and static Helm queue pools are operator-owned and exempt. Clients target either shape with an explicit `X-SIE-POOL` header, for example `X-SIE-POOL: customer-acme`, or SDK `gpu="customer-acme/l4"`. The `default` pool is protected and cannot be deleted. Missing named pools intentionally fail closed so capped/dynamic pool isolation is not weakened by a silent fallback.

Model configs registered via the control plane (`sie-config`) may carry a top-level `pool` field. That field is model routing/config assignment: omitted or empty means `default`, and a request that omits `X-SIE-POOL` defaults to the model's assigned pool. It does not create workers, KEDA ScaledObjects, or runtime pool leases. Worker counts, machine profiles, bundles, and autoscaling lanes are still Helm-driven physical capacity; `/v1/pools` and `queueRouting.staticQueuePools` only create or renew runtime admission/capacity state, optionally backed by existing `default` capacity through `PoolSpec.queue_pool`. In the Kubernetes/Helm composition, workers filter materialized model configs by their physical `SIE_POOL`, so a model config for a dedicated queue normally uses the physical queue-pool name; gateway requests may still target a logical pool whose `queue_pool` resolves to that physical name. A default-backed logical pool should not use a non-default model-config `pool` unless a matching dedicated worker queue exists. The managed Modal composition has one governed exception: a deployment-owned diagnostic admission queue may reuse the complete `default` catalog in a separate worker app. Dispatch stays on that physical queue, but bundle-hash and immutable-revision evidence are atomically scoped to the requested model's declared catalog pool. A known catalog model with a missing scoped hash fails closed rather than enabling the empty-hash wildcard; unknown sealed compatibility remains separately bounded.

### Numerical process diagnostics

NATS worker heartbeats may carry `numerical_process_inventory`: an observation
time in Unix milliseconds and a complete list of adapter children. The gateway
validates the optional metadata separately from worker health and exposes it in
cluster status, including workers excluded from direct dispatch by saturation.
Missing, malformed or oversized metadata clears the previous inventory without
discarding the heartbeat. The wire budget is 64 KiB and 256 children; unknown
fields and duplicate positive process IDs are rejected.

A process that holds a model's remote profile also reports that profile's
`remote_contract_sha256`, the `remote_execution_sha256` of its serving code and,
for a hybrid `encode` or `score` model, an `admission`: the local execution
identities its current evidence covers, the model contract, the outputs, the
expiry and a digest. The gateway checks the digests and identity formats, at most
eight identities, outputs from `dense`, `sparse`, `multivector` and `score`, and a
positive expiry, and drops a malformed inventory. The fields are absent when empty.
A gateway that predates them still accepts inventories from processes without a
remote profile and drops those that carry one.

Sidecars probe on dedicated IPC connections independently of readiness pings
and health publication, with a
five-second timeout and a ten-second observation lifetime. Shutdown tombstones
omit diagnostics. A snapshot describes the process and model configuration at
its observation time; it is not an atomic fleet proof, and on its own it grants
no routing or execution authority.

### Numerical bridges

A bare `encode` or `score` request for a local model with numerical outputs
(`dense`, `sparse`, `multivector` or `score`) bridges to the model's remote
profile only under a numerical admission. This covers fallback bridges, the
threshold route and `/v1/embeddings`, which wraps `encode`. Generation and
extraction surfaces never bridge such a model. For these models alone, the
gateway reads and parses the request body after route resolution and before
any bridge, threshold or load decision. An invalid body gets its `400` before
threshold demand is counted or load-only work is published. The parse lets the
gateway refuse a body-level `options.profile` selector, check the request
against the admission and replay the body remotely. The bridge decision is made
when the gateway is about to commit to the remote attempt:

- The request sets no instruction and no runtime option other than
  `is_query`, and every output it asks for (`dense` when an encode names none,
  `score` for score) is listed in the admission. Otherwise the remote process would refuse it, so the request
  stays local:
  - A request for an output that the model declares but the admission does not
    list stays local with the reason `unmeasured_request`.
  - A request with an instruction or another runtime option, or one that asks
    for an output the model does not declare, is not a bridge candidate on a
    fallback route. The
    trigger leaves it on its ordinary local path instead of committing to the
    local refusal, and no decision is recorded. On a `threshold` route it stays
    local with the reason `unmeasured_request`.
- An eligible remote worker is fresh and eligible for dispatch, reports
  `supports_execution_authority_v1`, `supports_numerical_admission_v1` and
  `supports_numerical_admission_subject_v1` (which builds that predate the
  admission subject never send), has
  the remote profile's exact configuration hash, bundle and pool, and has a
  unique worker name. Every child in its inventory must report the same
  admission for the bare model, together with the admission's own model
  contract, expiring at least five seconds after the decision.
- Every local process that could serve the model must be covered. That means
  each child of every worker on the model's local bundles and in the model's
  pool (`default` when it names none, because a worker loads only the models of
  its own pool), starting and degraded workers included, unless the worker lists
  the model as unsupported. Each such child must report an identity the
  admission lists and the admission's model contract. A worker without a
  complete inventory, a worker past the heartbeat timeout that has not been
  evicted, or a child without an identity or without the model, closes the
  bridge. The registry also closes it until it has heard worker health for one
  heartbeat timeout after the health subscription starts or resumes, or after
  a silence longer than that timeout, because a worker that has not reported
  yet looks absent. With no local worker the remote admission decides.
- The bridge pins one remote worker that both conditions admit. Selection checks
  again that the worker still reports `supports_numerical_admission_subject_v1` and the
  admitted digest. Its work items carry that digest as
  `numerical_admission_sha256` on the worker's `numerical-admission-v1` subject,
  which only a sidecar with the admission fence consumes, and the publisher
  refuses the digest on any other subject. The remote process derives its
  admission again before calling the upstream, and the sidecar answers a refused
  admitted item at once. A bridged caller then receives its local refusal with
  `X-SIE-Fallback-Error`. A threshold route holds no local refusal, so it
  answers `503 INFERENCE_ERROR` with the worker's `Retry-After` hint (1 to 60
  seconds, else 5).

Anything else keeps the request on its local route. A numerical plan that was
not admitted cannot select a worker, and a numerical bridge whose pinned worker
has no admitted digest is refused before publication. Each request's decision
is counted once, with the decision that applied, on
`sie.gateway.remote.numerical_admissions` with `admitted` or one of the refusal
reasons `no_admission`, `local_unobserved`, `uncovered_identity` and
`unmeasured_request`. Configuration load no longer refuses `threshold` routing
for numerical models; the admission gates each request instead.

### Remote routes under a model access policy

A deployment that installs a `ModelAccessPolicy` decides the implicit remote
routes of a bare model, whether a fallback trigger or a `threshold` decision
would take the route. A request that names the remote profile, directly or
through an alias, is governed by `visible` and `serving_refusal` like any other
model, so a caller who can see the remote profile can name it.

- **When the gateway asks.** Only about a route it admits on its own: a bare
  model whose routing names the remote profile and permits the reason, no
  caller profile, bundle, pool or engine selector, no `X-SIE-Remote: forbid`,
  a transport with execution authority v1, and no bridge already active.
- **The remote profile first.** The gateway requires
  `ModelAccessPolicy::visible` and `ModelAccessPolicy::serving_refusal` to
  pass for the remote profile. It asks `serving_refusal` with the request
  extensions minus the admission-outcome slot. A refusal keeps the local route,
  and it is neither returned nor recorded.
- **The question.** `ModelAccessPolicy::remote_route_admitted` receives the
  canonical model, the remote profile and a `RemoteRouteReason`:
  `Fallback(trigger)` or `Threshold`.
- **Once per request.** The gateway decides once per request, remote profile
  and reason, and reuses that decision.
- **The answer.** `false`, the default, keeps the local route. The caller
  receives the answer it would have received without a remote route.
- **Without a policy.** The gateway's own checks decide.
- **Governed generation.** A policy that governs generation routes keeps
  generation local, because a governed route has no remote route.
- **Model listing.** `/v1/models` and `/v1/models/{model}` show a bare model
  to a caller who may not see its remote profile without that profile and with
  the routing of a model that has no remote route, so the listing does not
  reveal the profile that `visible` hides.

### Lanes a transport reports cold

A dispatch transport that manages its own capacity can keep worker rows for a
lane with no ready capacity. `WorkDispatcher::lane_provisioning` reports such a
lane. It is `false` by default and is a non-publishing hint.

- **When the gateway reads it.** After the route resolves and before any
  dispatch.
- **The bridge.** When the hint is `true` and a `provisioning` bridge is
  admitted:
  1. The gateway wakes the lane through `publish_model_load` with the lane's
     pool target, recording pending demand as for any cold lane.
  2. It waits up to 2 s for the transport to accept the wake.
  3. It admits the bridge again, and makes the single remote attempt with
     `fallback_reason: provisioning`.
  - A wake that is not accepted, or a bridge that is no longer admitted after
    the wake, leaves the caller with the local `503 PROVISIONING`. A
    `model_loading` bridge admits its plan again after its load-only wake in the
    same way.
- **No bridge.** A request that is not bridged dispatches as before.
- **A remote attempt refused as `PROVISIONING`** received no answer. It
  restores the local refusal with `X-SIE-Fallback-Error: QUEUE_FULL`.
