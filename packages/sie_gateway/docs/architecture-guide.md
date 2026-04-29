# SIE Architecture

This document describes the current deployed architecture of the SIE control plane and inference edge. It is code-audited against the `sie-internal` monorepo (`packages/sie_gateway`, `packages/sie_config`, `packages/sie_sdk`). Every behavior described here is what the code does on this branch.

Top-level summary:

- `sie-gateway` (Rust) is the inference edge. It is queue-only for inference and serves read-side configuration.
- `sie-config` (Python) is the configuration control plane. It owns writes, persistence, and delta publication.
- Workers execute inference.
- NATS carries runtime messaging. JetStream carries inference work and the DLQ. Core pub/sub carries config deltas, worker health, and inference results.
- Kubernetes coordinates pool state. It is not on the inference request path.

The gateway's contract with `sie-config` is three shapes. Everything else below elaborates on them.

1. `GET /v1/configs/export` — full snapshot, consumed on cold start and on drift detection.
2. `GET /v1/configs/epoch` — monotonic integer, polled every 30s to detect drift.
3. `sie.config.models._all` (NATS Core) — live deltas; missed ones are caught by (2).

The gateway's contract with admin tooling for post-write readiness is `GET /v1/configs/models/{id}/status`.

## 1. Services

| Service | Role | Publishes | Subscribes / Consumes |
|---|---|---|---|
| Client SDK | Submits inference requests | HTTP inference requests | HTTP inference responses |
| Admin / deploy tooling | Mutates config, reads control-plane state | HTTP config requests | HTTP config responses |
| `sie-gateway` (Rust) | Queue-only inference edge; read-side config; pool management; health and metrics | Queue work items, inference results acks, read-side config responses, pool state changes, DLQ republishes, metrics | Client inference requests, worker results, worker health, config deltas, K8s pool state |
| `sie-config` (Python) | Config control plane; single writer | Config write responses, read responses, snapshot/export responses, config deltas (NATS) | Admin/config-management requests |
| Workers | Execute inference | Result messages, health signals | Queue work items |
| NATS / JetStream | Message bus | — | — |
| Kubernetes API | Pool coordination | — | — |

Key terms used throughout this document:

- **Bundle**: a model-compatibility grouping. Answers "which workers can serve this model?". Defined by a `name`, a `priority`, and an `adapters[]` list of Python module paths. The source of truth is `sie-config`'s filesystem (image-baked at `SIE_BUNDLES_DIR`, default `/app/bundles`); `sie-config` loads them at startup to drive write validation and exposes them via `GET /v1/configs/bundles` + `GET /v1/configs/bundles/{id}`. The gateway no longer bakes its own copy: `state::config_bootstrap::BootstrapClient::fetch_bundles` pulls them at startup and installs them into `ModelRegistry` via `install_bundles` before any model fetch. Neither service *persists* bundles, and there is no `POST /v1/configs/bundles` on either side — bundle changes are a `sie-config` redeploy and the gateway picks them up on its next bootstrap (or on the next poller-driven catch-up; see §4.2).
- **Pool**: an operational routing group of workers managed by the gateway. Answers "which compatible workers should this request be queued to right now?". Pools are a runtime concept stored in Kubernetes.
- **Config store**: `sie-config`'s backing store for persisted state. Per-model YAML files plus a plain-text monotonic `epoch` counter, laid out as `{base}/models/{id-with-/-as-__}.yaml` and `{base}/epoch`. The backend is selected by the `SIE_CONFIG_STORE_DIR` scheme: a local path (default — backed by an opt-in PVC in Helm), `s3://…`, or `gs://…`.
- **Config epoch**: a monotonic integer. Every `sie-config` write that creates new profiles bumps it by one. The gateway tracks it locally as `ConfigEpoch` for drift detection.
- **`bundle_config_hash`**: a content hash over the routable-profile fields (`adapter_path`, `max_batch_tokens`, `compute_precision`, `adapter_options`) for every model whose adapters match a bundle. Used by admin tooling and workers to tell "has this worker picked up the new config for this bundle yet?"

## 2. Inference Request Path

1. Client SDK sends HTTP to the gateway:
   - `POST /v1/encode/{*model}`
   - `POST /v1/score/{*model}`
   - `POST /v1/extract/{*model}`
2. Gateway resolves the request to a model, a bundle, a machine profile, and a queue pool using its in-memory registry.
3. Gateway publishes work to JetStream on `sie.work.{model}.{pool}`.
4. A matching worker consumes the work item and executes inference.
5. Worker publishes the result on `_INBOX.{router_id}.{request_id}` (NATS Core).
6. Gateway collects the result and returns the HTTP response.

Rules enforced on the inference path:

- The inference hot path does not call `sie-config`.
- The inference hot path is queue-only. There is no direct-HTTP fallback. `src/handlers/proxy.rs` is the queue-submission handler despite its name.
- If the queue transport is unavailable (no usable NATS client at init), the gateway returns `503`. It does not fall back to direct mode.
- Unknown model ids fast-fail with `404` whenever the in-memory `ModelRegistry` has been populated (either by the filesystem seed or by a successful bootstrap / delta from `sie-config`). In the pre-bootstrap edge case where the registry is still empty — no seed, no export applied yet — the proxy falls back to the caller-supplied bundle (or `"default"`) so an unseeded gateway can still publish work to a cold pool that a caller pinned via `X-SIE-Pool`. Once any model is registered, this fallback is disabled and the 404 contract applies.
- On scale-from-zero — i.e. no healthy worker registered for the `(bundle, machine_profile)` tuple and the caller did not pin an explicit pool — the gateway returns `202 provisioning` with `Retry-After: 120` and records pending demand for KEDA. This applies whether or not the caller set `X-SIE-MACHINE-PROFILE`; default-routing clients get the same contract as profile-pinned clients.
- On no-consumer conditions for the JetStream publish the gateway returns `503` with `Retry-After: 120`. On backpressure conditions the gateway returns `503` with `Retry-After: 5`.
- On queue result timeouts the gateway returns `503` with `X-SIE-Error-Code: MODEL_LOADING` and `Retry-After: 5`. The most common trigger is a worker cold-loading the target model on demand (worker NAKs the JetStream message and redelivers after load); the SDK retries this under the same `provision_timeout_s` budget used for worker-emitted `MODEL_LOADING` responses.
- When workers report failure on every item of a batch with the **same retryable error code** in `WorkResult.error_code`, the gateway translates that into a `503` with the SDK-expected envelope (`error.code`, `Retry-After`, `X-SIE-Error-Code`) so the SDK auto-retries. Currently recognised codes: `RESOURCE_EXHAUSTED` (worker-side OOM recovery exhausted — `Retry-After: 5`), `MODEL_LOADING`, and `LORA_LOADING`. **Mixed batches** (different codes per item) keep going through the legacy `500 all_items_failed` path with per-item `code` fields exposed in `details[]`, so callers can see exactly which items hit which failure mode. Workers reporting `RESOURCE_EXHAUSTED` are **not** marked unhealthy — losing an allocation race is not a worker-health signal.

Work-item and result payloads on the wire are **msgpack** (`rmp_serde`). JSON is used only for the HTTP request/response envelope when the client negotiates it (via `Content-Type` / `Accept`); the gateway transcodes msgpack ↔ JSON at the edge so the hot path between gateway and workers is always binary.

## 3. Configuration Write Path

All config mutations go to `sie-config`. The gateway has no write handler.

1. Admin or deploy tooling sends `POST /v1/configs/models` with a model config YAML body to `sie-config`. Auth: `_check_write_auth` (`packages/sie_config/src/sie_config/config_api.py`) requires a bearer matching `SIE_ADMIN_TOKEN` when that variable is set; if `SIE_ADMIN_TOKEN` is unset and `SIE_AUTH_TOKEN` is set, writes are rejected with `403` (the inference token never grants write access); if neither token is set (dev/local only) writes are accepted unauthenticated. Production deployments always set `SIE_ADMIN_TOKEN`.
2. **Pre-lock stage** (outside the per-app write lock, on the FastAPI event loop): parse the YAML body and — when a top-level `sie_id` is present — run `_validate_model_id` (regex + `..`/`\\` rejection + `status`-suffix rejection). Reject invalid input with `400` before any state is touched.
3. **Critical section** under `_get_write_lock` (a lazy, per-app `asyncio.Lock` stored on `app.state`):
   1. `ModelRegistry.validate_model_config` — pure check against the in-memory registry (no mutation), run under the registry's internal `threading.RLock`. Validation rejects any write whose post-state resolves to zero routable bundles (`422`), so an `extends`-only profile cannot land a brand-new model that no worker bundle can serve. Appending a new `extends`-only profile to an already-routable model is allowed.
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
- `GET /v1/configs/export` also takes the same lock (`_check_write_auth` — admin-only), so every exported snapshot is a real serialization point: `(epoch, models)` always corresponds to a state that existed between two writes.

NATS publish behavior:

- One `ConfigNotification` message is published per affected bundle to `sie.config.models.{bundle_id}`, and a byte-identical copy to `sie.config.models._all`.
- If publish fails for a subset of bundles, `NatsPublisher` continues publishing to the remaining bundles and raises `PartialPublishError` at the end with the failed bundle list. The write itself is still durable on disk. The response `warnings` field names the failed bundles; affected workers will stay on the previous epoch until the gateway's poller triggers a re-export.

Idempotency:

- Writes accept an `Idempotency-Key` header. The state (`_IdempotencyState` on `request.app.state`) holds an LRU response cache and per-key `asyncio.Event`s for in-flight deduplication.
- A duplicate key with the same body returns the cached response without re-executing.
- A duplicate key with a different body returns `422 idempotency_mismatch`.
- A duplicate key whose cached response has been LRU-evicted between the in-flight wait and the cache read returns a synthesized `200 idempotent_replay_evicted` response. The write is never executed twice for the same key.

Model-ID validation (`_validate_model_id`):

- Rejects path-traversal patterns (`..`, `\\`).
- Requires `^[a-zA-Z0-9][a-zA-Z0-9._/-]*$`.
- Rejects IDs equal to `status` or ending in `/status`, because the gateway's `GET /v1/configs/models/{*id}` uses a `/status` suffix for worker-ack reporting and an overlapping model ID would make that URL ambiguous.

The gateway does not serve `POST /v1/configs/models`. The route is not registered. Axum returns `405 Method Not Allowed`. There is no `410 Gone` shim and no redirect body.

## 4. Gateway Bootstrap and Recovery

The gateway does not block startup on `sie-config`. Startup is:

1. Construct `ModelRegistry`. In default Helm deploys, `SIE_BUNDLES_DIR` and `SIE_MODELS_DIR` are unset and the registry's filesystem reload is a no-op (warns once that the dirs are missing, then proceeds with empty bundle and model maps). The optional `gateway.embeddedConfigs` / `gateway.configMap` overlays mount a ConfigMap at `/configs/{bundles,models}` and set those env vars; in that mode the registry loads the seed before bootstrap runs.
2. Initialize the other runtime subsystems that do not depend on `sie-config`: NATS manager (with config-delta subscription), worker health manager, optional filesystem config watcher, pool manager / K8s backend, and worker discovery.
3. Spawn two independent background tasks: `state::config_bootstrap::spawn_bootstrap_retry` and `state::config_poller::spawn`. They share the same `ModelRegistry`, `ConfigEpoch` (monotonic model-write counter), and `BundlesHash` (sha256 fingerprint of `sie-config`'s loaded bundle set). The poller skips its first tick (one `DEFAULT_POLL_INTERVAL` grace) so the initial bootstrap attempt can run first, and then reconciles on every subsequent tick regardless of whether the bootstrap task has completed yet.
4. Bind the HTTP listener. `/readyz` returns `200` unconditionally — it does not gate on `sie-config` reachability.

The bootstrap/poller tasks are spawned **before** the HTTP listener binds, so requests hitting a freshly-bound replica can observe either filesystem-seed-only state or a partially-applied snapshot depending on timing. That is by design; `/readyz` does not gate on control-plane sync (see §11 caveat 3).

### 4.1 Bootstrap retry

`state::config_bootstrap::spawn_bootstrap_retry` runs `bootstrap_once` in a retry loop until the first success, then exits. Ongoing reconciliation after that point is the poller's job (§4.2).

Startup short-circuits:

- If `SIE_CONFIG_SERVICE_URL` is unset, the task logs and returns immediately — there is nothing to bootstrap against. The gateway runs filesystem-seed-only.
- If constructing the HTTP client fails, the task sets `sie_gateway_config_bootstrap_degraded` to `1` and returns; it does not retry.

`bootstrap_once` runs a three-call sequence, authenticated with `SIE_ADMIN_TOKEN` as a bearer token on `SIE_CONFIG_SERVICE_URL`:

1. `GET /v1/configs/epoch` — reads the bundle-set fingerprint BEFORE we fetch bundles, so the hash we store reflects the state we're about to catch up from (the pre-fetch ordering is load-bearing — see below).
2. `GET /v1/configs/bundles` and per-id `GET /v1/configs/bundles/{id}` — the two-phase bundle fetch. The YAML bodies are parsed into `BundleInfo` values and handed to `ModelRegistry::install_bundles`, which atomically replaces the registry's bundle set and recomputes every model's bundle associations.
3. `GET /v1/configs/export` — replays every exported model config into `ModelRegistry` via `add_model_config`. Malformed or unparseable entries are logged and counted as `failed`; they do not abort the loop. Export rows that carry no config body are skipped and are *not* counted as failed.

Returns `BootstrapOutcome { epoch, bundles_hash, applied, failed, total }`.

- If **any** model entry failed (`outcome.failed > 0`), `bootstrap_once` returns `BootstrapError::PartialApply` and **advances neither `ConfigEpoch` nor `BundlesHash`**. The poller will detect drift on the next tick and retry.
- If all entries applied, `bootstrap_once` advances `ConfigEpoch` via `set_max(outcome.epoch)`, stores `outcome.bundles_hash` into `BundlesHash`, and the retry task exits.

Why fetch `/epoch` BEFORE `/bundles`: if a bundle is added on `sie-config` between our epoch read and our bundle list read, we install a bundle set *newer* than the hash reflects. The next poll tick sees `remote_hash != stored_hash` and re-fetches — a cheap self-heal. The inverse — storing a hash that's *newer* than the bundles we actually installed — would silently wedge the gateway on a stale registry, which is what the bundle-hash mechanism exists to prevent.

On failure (DNS, 5xx, timeout, decode error, partial apply):

- Increments `sie_gateway_config_bootstrap_failures_total`.
- Sleeps with exponential backoff, initial 1s, doubling to a 60s cap.
- After 10 failed attempts or 5 minutes of sustained failure, sets `sie_gateway_config_bootstrap_degraded` gauge to `1` and escalates logging to `error!`. The gauge clears on the first success.
- The retry loop itself has no attempt ceiling — it keeps trying until one attempt succeeds (at which point the task exits) or the process is killed. `sie-config` is not a hard startup dependency.

During the bootstrap-retry window (before the first successful fetch):

- `/readyz` returns `200` unconditionally. The gateway serves filesystem-seed traffic.
- `GET /v1/configs/models/{id}` returns only filesystem-seeded models.
- `POST /v1/configs/resolve` for API-only models returns `404`. Admin tooling retries with backoff.
- `ConfigEpoch.get()` typically returns `0` and `GET /v1/configs/models/{id}/status` surfaces `config_epoch: 0`, which admin tooling uses to detect a pre-bootstrap replica. The epoch is not strictly pinned to `0`, however: if the NATS delta subscriber (§4.3) receives a valid `ConfigNotification` with a non-zero `epoch` before the first export succeeds, `ConfigEpoch::set_max` will advance the counter. The `config_bootstrap_degraded` gauge is the more reliable "has export ever succeeded?" signal.

### 4.2 Epoch poll

`state::config_poller::spawn` runs independently of bootstrap with `DEFAULT_POLL_INTERVAL = 30s`. It skips its first tick (to let the initial bootstrap attempt run without contention), then on every tick calls `GET /v1/configs/epoch` on `sie-config` (always sending `SIE_ADMIN_TOKEN` as the bearer — the `/epoch` handler is read-auth and accepts either `SIE_AUTH_TOKEN` or `SIE_ADMIN_TOKEN`, but the gateway-as-client only presents the admin token). The `/epoch` response carries **two** drift signals:

- `epoch` — monotonic counter bumped on every model-config write.
- `bundles_hash` — sha256 fingerprint over `sie-config`'s loaded bundle set. Bundles are filesystem artifacts inside the `sie-config` image, so their effective "version" is redeploy time, which the model-write counter does not observe.

Both signals funnel into the same `bootstrap_once` call — that function re-fetches bundles and models together, which is the correct blanket response to any control-plane drift.

Per-tick decisions:

- If `remote_epoch > local_epoch` **OR** `remote_bundles_hash != stored_bundles_hash` (with the empty-string "registry unavailable" sentinel skipped), the poller calls `bootstrap_once`. On success, `ConfigEpoch` advances and `BundlesHash` is updated.
- If `remote_epoch < local_epoch`, the local counter has run ahead of authority. Most innocent cause: `sie-config` lost its persisted epoch file and restarted at a lower value. Most worrying cause: a forged/untrusted NATS delta wedged the gateway ahead of authority (this is defense-in-depth alongside the producer allowlist — see §4.3). The poller refetches the full export first and only calls `ConfigEpoch::force_set(remote)` if the export succeeded; a failed recovery export keeps `local > remote`, so the next tick re-enters this branch and retries. Note: `ModelRegistry` is append-only for models (but not for bundles — `install_bundles` replaces), so any profiles admitted during the pre-recovery window remain in the registry after recovery — the export replay overlays authority on top of them rather than clearing them. A follow-up restart is the clean fix. Logged at ERROR.
- A remote `bundles_hash` of `""` is the documented "sie-config registry unavailable" sentinel (startup failure or hot reload in flight). The poller skips the hash-mismatch branch in that case rather than thrash fetching against a degraded control plane. The epoch branch still runs normally.
- Transient `/epoch` failures are logged and swallowed. The next tick retries. Neither the local epoch nor the stored bundles hash is reset on poll failure.
- The poller **always** compares local vs. remote, including when `local_epoch == 0` or `stored_bundles_hash == ""`. This matters for fresh clusters: if `sie-config`'s very first write (→ epoch 1) is lost on the wire, the gateway must catch `remote=1 > local=0` and trigger recovery. Gating on "non-zero local" would wedge a fresh gateway until a restart.

Operational consequence: adding or changing a bundle in `sie-config` propagates to running gateways within one `DEFAULT_POLL_INTERVAL` (≤ 30s worst case). No `kubectl rollout restart deployment/sie-gateway` is required — that was true of an earlier revision of this design, which relied on bundles being baked into the gateway image.

### 4.3 Live deltas

Independently of bootstrap and polling, `NatsManager` subscribes to `sie.config.models._all` (Core pub/sub) as soon as the NATS client is ready. For each incoming `ConfigNotification`:

- **Producer-trust check first.** The `producer_id` (wire key `router_id`) must be in the trusted-producer allowlist. Default allowlist is `["sie-config"]`, configurable via `SIE_NATS_CONFIG_TRUSTED_PRODUCERS=a,b,c`. Matching is exact equality OR K8s pod-name prefix (`sie-config` also matches `sie-config-5f7b6d8c-kxwvr` and `sie-config-0`, but not `sie-configuration`). Untrusted notifications are dropped before any registry mutation or epoch advance; a log warning names the rejected producer. Set `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER=true` to disable validation (dev/local only — `main.rs` emits a startup audit warning when this is on).
- An **empty-or-whitespace `model_config` body** (pure epoch bump) advances `ConfigEpoch` via `set_max(notification.epoch)` and returns.
- A non-empty body is parsed as `ModelConfig` YAML. If the parse fails, the epoch is NOT advanced and the event is logged. The poller will catch up.
- On successful parse, `ModelRegistry.add_model_config` is called. If it returns `Ok`, the epoch is advanced. If it returns `Err` (e.g. append-only conflict, unroutable adapter), the epoch is NOT advanced.

`ConfigEpoch::set_max` uses a CAS loop over `AtomicU64`. The value only ever moves forward; a late-arriving lower-epoch delta cannot roll it back. `async_nats::Subscriber` survives reconnects transparently, so the gateway does not need explicit resubscribe logic — any messages published while the subscriber is disconnected are simply lost, which is the gap the poller closes.

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
- Writes (`add_model_config`, `reload`) hold `write_lock: Mutex<()>` across the `load → mutate → store` cycle. Without this lock, two concurrent writers could both load the same base snapshot and silently drop one set of changes.
- `RegistrySnapshot` caches `bundle_config_hashes: HashMap<String, String>`, so the per-request worker-ack hash lookup (§6) is an `O(1)` map read rather than a SHA-256 over tens of KB of JSON.

## 6. Worker-Ack Status (`GET /v1/configs/models/{id}/status`)

Admin tooling uses this endpoint after a `sie-config` write to confirm that configured workers have picked up the new `bundle_config_hash` for each affected bundle.

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
      "expected_bundle_config_hash": "sha256…",
      "total_eligible_workers": 3,
      "acked_workers": ["worker-1", "worker-2", "worker-3"],
      "pending_workers": [],
      "acked": true
    }
  ],
  "source": "gateway-registry"
}
```

Computation rules:

- **Per-replica only.** The endpoint reports the worker registry state on this specific gateway pod. Fleet-wide readiness is the union across replicas; admin tooling fans out.
- **Case-sensitive bundle match.** A worker is counted as eligible only if `worker.bundle == bundle_id` exactly. Workers whose `bundle` field does not match exactly are skipped entirely — they contribute to neither `total_eligible_workers` nor `pending_workers`. The expected hash is computed against the canonical bundle ID from the registry, and the fleet is case-consistent in practice.
- **Healthy-only.** Unhealthy workers (per `worker.healthy()`) are skipped entirely; they contribute to neither `total_eligible_workers` nor either worker list.
- **Zero-bundle models.** A model with no routable bundles is reported as `all_bundles_acked: false` and `no_bundles: true`. Returning `true` for "nothing to ack" would silently tell admin tooling "fully deployed" when there is nothing deployed.
- **`config_epoch` field.** Surfaces the gateway's local `ConfigEpoch` so callers can tell whether this replica has caught up to a recent write (`config_epoch` lower than the write's returned epoch means this replica is still catching up).

## 7. Cross-Service Hash Parity

`bundle_config_hash` is computed independently by `sie-config` and the gateway, and also by workers. For the admin-tooling readiness flow to work, all three must produce byte-identical hashes for the same config.

The hash is a SHA-256 over a JSON-serialized, sort-keys representation of a structured object. The outer shape is an array of models, each with `sie_id` and `profiles: [{name, config: {...4 fields...}}]`, where the inner `config` for each profile is strictly the four routable-profile fields: `adapter_path`, `max_batch_tokens`, `compute_precision`, `adapter_options`. Any field outside that whitelist is excluded from the hash so that non-routable edits (e.g. comments, description-style metadata) cannot unnecessarily invalidate worker acks. Python uses `orjson.dumps(..., OPT_SORT_KEYS)`; Rust mirrors this via `serde_json::to_vec` over `BTreeMap`-backed canonical types.

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
| Inference work | `sie.work.{model}.{pool}` | JetStream | Durability and max-delivery semantics required. Work-item payload is msgpack. One stream per pool (`WORK_POOL_{pool}`) captures all models; see §8.2. |
| Inference results | `_INBOX.{router_id}.{request_id}` | NATS Core | Gateway is waiting synchronously; a brief blip after publish but before delivery means the result is lost and the client retries (the gateway returns `503` with `X-SIE-Error-Code: MODEL_LOADING` and `Retry-After: 5` — see §2). Result payload is msgpack. |
| Config deltas | `sie.config.models.{bundle}`, `sie.config.models._all` | NATS Core | Lightweight fan-out. Durability comes from the snapshot/export path (§4), not the bus. JSON payload (control plane, not hot path). Workers subscribe per-bundle; the gateway subscribes on `_all`. |
| Worker health | `sie.health.>` | NATS Core | Ephemeral, last-heartbeat-wins. The gateway subscribes in `health_mode=nats` (see `discovery/nats_health.rs`); the **default `health_mode=ws`** uses the WebSocket path, and a NATS-mode publisher is not present in the current worker code. Until a worker-side publisher is added, NATS health mode is effectively a no-op consumer. |
| DLQ advisories | `$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>` | NATS Core (advisory) | JetStream emits these; the gateway subscribes in `queue/dlq.rs`. |
| DLQ storage | `sie.dlq.{model_token}` | JetStream | Single stream `DEAD_LETTERS` (Limits retention, memory storage, 24 h `max_age`) captures `sie.dlq.>`. `model_token` is derived from the advisory's original work subject by taking the model segment and replacing `/` with `_`. |

### 8.2 Work-stream configuration and the model-ID constraint

The gateway's `ensure_stream` (per pool, called lazily on first publish) creates a JetStream stream with:

```
name:      WORK_POOL_{pool}
subjects:  ["sie.work.*.{pool}"]
retention: WorkQueue
storage:   Memory
max_age:   60s  (gateway-side)
max_msgs:  100_000
```

The Python worker (`sie_server.nats_pull_loop`) independently calls `add_stream` with the **same** name but a **different default `max_age` of 120 s** (`SIE_STREAM_MAX_AGE_S`). `get_or_create_stream` / `add_stream` does not update an existing stream's config, so whichever side races to the broker first wins. To avoid drift, pick one owner (either the gateway or the worker); the Helm values should set `SIE_STREAM_MAX_AGE_S` on workers to match the gateway's constant, or the gateway should read the same env.

The stream's subject filter `sie.work.*.{pool}` matches **exactly one token** in the model position. NATS subject tokens legally contain `/`, so `BAAI/bge-m3` works directly. They cannot contain `.`, `*`, `>`, or whitespace, so any model whose `sie_id` contains one of those characters would — without normalization — expand into multiple tokens, not match the stream filter, and get rejected at the broker.

To prevent that, the Rust gateway's `work_subject(model, pool)` in `packages/sie_gateway/src/queue/publisher.rs` calls a private `normalize_model_id` helper that mirrors the Python SDK (`sie_sdk.queue_types.normalize_model_id`): `/` → `__`, `.` → `_dot_`, and `*`/`>`/space → `_`. So `vidore/colqwen2.5-v0.2` publishes on `sie.work.vidore__colqwen2_dot_5-v0_dot_2.default` — exactly four tokens, matches the consumer filter. The DLQ token extractor in `queue/dlq.rs` splits the advisory subject on `.` and takes `parts[2]`, which with the normalization in place is already a single safe token; the existing `/` → `_` replacement there is now a no-op for correctly published subjects and a fallback for any legacy un-normalized messages still in-flight at the time of upgrade.

### 8.3 Publisher/consumer ownership

Publishers and consumers:

- **Config deltas**: published exclusively by `sie-config`. The gateway's `NatsManager` is subscribe-only on `_all` and never publishes on these subjects. Delta payload includes a `router_id` field whose value is `sie-config`'s pod/host identifier. The Rust gateway deserializes this field into a struct member named `producer_id`; `router_id` is declared as a `#[serde(alias)]` so both the current `router_id` wire name and any legacy `producer_id` name parse cleanly.
- **Queue work**: published exclusively by the gateway. Consumed by workers.
- **Results**: published exclusively by workers. Consumed by the originating gateway.

The `ConfigNotification` JSON payload carries the following fields, verified against `packages/sie_gateway/src/nats/manager.rs` and `packages/sie_config/src/sie_config/nats_publisher.py`:

- `router_id` on the wire (Python publisher key). The Rust gateway parses this into a struct field named `producer_id`; `router_id` is declared as a `#[serde(alias)]` on the Rust side for forward compatibility.
- `bundle_id`, `epoch`, `bundle_config_hash`, `model_id`.
- `profiles_added` — list of profile names created by this write.
- `model_config` — raw YAML string of the full model config.
- `affected_bundles` — list of every bundle affected by this write. Each affected bundle receives its own message, so a `_all` subscriber sees N messages per write with their respective `bundle_id` / `bundle_config_hash` populated.

### 8.4 Serialization formats

The hot path is always **msgpack** (`rmp_serde` on the gateway, `msgpack-python` / `msgpack-numpy` on workers). Specifically:

- `WorkItem` (gateway → worker on JetStream): msgpack.
- `WorkResult` (worker → gateway on the inbox subject): msgpack. Numpy arrays use `msgpack-numpy`'s extension-free encoding (maps with a `nd: true` sentinel + `dtype`, `shape`, `data`). The gateway transcodes to native JSON arrays only when the client's `Accept` header asks for JSON.

JSON (`serde_json`) is used where payloads are low-frequency or human-oriented:

- Control plane: `POST /v1/configs/models` body, `GET /v1/configs/export` response, `GET /v1/configs/epoch` response.
- NATS config deltas: `ConfigNotification` is JSON.
- Worker-ack status endpoint responses.
- Operator-visible APIs: pool management, `/v1/models`, `/health`, `/metrics`.
- Inference request/response envelopes, when the client opts into JSON via `Content-Type` / `Accept`. SDK clients typically use msgpack end-to-end.

## 9. Storage and Persistence

`sie-config` is the only service that persists configuration. Its backing store is:

- `{base_dir}/models/{model_id_with_slash_replaced_by_double_underscore}.yaml` — one YAML file per model.
- `{base_dir}/epoch` — a plain-text integer, monotonically incremented by `increment_epoch`.

Backend selection is URL-driven (`sie_sdk.storage.get_storage_backend`):

- Local filesystem (default).
- `s3://bucket/prefix` — S3.
- `gs://bucket/prefix` — GCS.

The **local** backend's `write_text` is atomic: it writes via `tempfile.mkstemp` in the same directory as the destination, `fsync`s, then `Path.replace`s. A mid-write crash cannot leave the destination truncated or empty. The cloud backends (`S3Backend`, `GCSBackend`) implement `write_text` as a single `put_object` / `upload_from_string`; object-store PUTs are last-writer-wins and observably atomic at the object level, but they do not use the tempfile + replace pattern. The atomicity property matters most for the `epoch` file: `ConfigStore.read_epoch` silently maps a malformed or empty integer to `0`, so a zero-byte epoch file would collapse the drift-detection mechanism (`remote == local == 0` would read as "in sync forever").

`ConfigStore.increment_epoch` is a naive read-modify-write. Its single-writer assumption is enforced at the FastAPI layer by the per-app write lock (§3).

The gateway has no persistent config store. `SIE_CONFIG_STORE_DIR` and `SIE_CONFIG_RESTORE` are `sie-config`-only and are not part of the gateway's env surface. The only persistent-looking inputs on the gateway are the optional filesystem seed directories (`SIE_BUNDLES_DIR`, `SIE_MODELS_DIR`); these are unset in the default deploy, where bundles and models are pulled from `sie-config` at startup, and only become non-empty when the `gateway.embeddedConfigs` / `gateway.configMap` overlays explicitly mount one.

## 10. Deployment Topology

- `sie-gateway` and `sie-config` are separate Kubernetes Deployments with separate images, built via `tools/mise_tasks/docker_task.py` (invoked as `mise run docker --{gateway,config,bake}`). On a `v*` git tag, `.github/workflows/release-docker.yml` publishes `ghcr.io/superlinked/sie-{config,gateway}:<tag>` and the per-platform/bundle `sie-server:<tag>-<platform>-<bundle>` matrix (plus floating `:latest` variants) to GHCR, which the Helm chart consumes by default. The `aws_docker`/`aws_deploy` tasks handle manual pushes to ECR/AR for cluster-local iteration.
- `sie-config` runs as **a single replica**. Multi-replica is blocked by the in-memory idempotency cache (see §11 "Known operational caveats"). Config persistence is opt-in: when `config.configStore.enabled: true` in the Helm values, a PVC is mounted at `/var/lib/sie-config` (default mount path) and `SIE_CONFIG_STORE_DIR` / `SIE_CONFIG_RESTORE` are set so the store survives pod restarts; with the default `enabled: false`, the store is ephemeral (backed by the pod filesystem) and a pod restart resets the epoch to 0 and drops every API-added model — `sie-config` does not subscribe to NATS (it is the sole publisher) so there is no replay path, and the registry rebuilds from the image-baked `/app/bundles` + `/app/models` baseline only. Every gateway replica's `state::config_poller` then detects `remote_epoch < local_epoch` and force-resets to match the restarted `sie-config`, picking up whatever baseline it now advertises. Operators who need API-added models to survive `sie-config` restarts must set `config.configStore.enabled: true`. The Helm chart (`deploy/helm/sie-cluster/values.yaml`) documents the SPOF posture.
- `sie-gateway` scales horizontally. Each replica maintains its own in-memory `ModelRegistry` and its own `ConfigEpoch`. They all converge on the same state via a combination of bootstrap, deltas, and the poller; they do not coordinate with each other.
- Bundles live on `sie-config`'s filesystem at `SIE_BUNDLES_DIR` (default `/app/bundles`, set via `sharedPaths.bundlesDir`) and are baked into its image. `sie-config` reads them at startup, validates writes against them, and re-serves them over HTTP at `GET /v1/configs/bundles{,/{id}}`. The gateway fetches that surface during `state::config_bootstrap::bootstrap` and installs it via `ModelRegistry::install_bundles` — there is no second copy. The legacy `gateway.embeddedConfigs` / `gateway.configMap` overlays still exist for the rare case of running the gateway without `sie-config` (e.g., self-contained smoke tests); they mount a ConfigMap at `/configs/bundles` which the registry's filesystem reload picks up before the (no-op) bootstrap runs. In all cases, updating bundles is a `sie-config` redeploy and the gateway picks up the change on its next bootstrap or poller-driven reconcile, not via a runtime API call.
- Pools are stored in Kubernetes `ConfigMap`s and `Lease`s read/written by the gateway. `sie-config` is not involved in pool management.
- NATS / JetStream runs as its own workload. The gateway's `NatsManager` builds `async_nats::ConnectOptions` with `retry_on_initial_connect()` so the process does not fail to start if NATS is briefly unavailable; reconnect behavior after initial connect relies on `async-nats`'s default policy (indefinite reconnect with backoff).

## 11. Known Operational Caveats

1. **`sie-config` is a single point of failure for the control plane.** While the pod is down, config writes, authoritative config reads, and gateway catch-up fetches all fail. **The inference hot path is unaffected** — it never touches `sie-config`. New gateway pods enter the soft-degraded bootstrap-retry state (§4.1) and serve filesystem-seed traffic until `sie-config` is reachable. The SPOF is rooted in `_IdempotencyState` (in-memory, per-process). Multi-replica requires moving idempotency state to a shared backend (Redis, JetStream KV, a DB row per key). Until then, `replicas: 1` is a deliberate trade of availability for a correct idempotency contract.

2. **Config deltas use NATS Core pub/sub; deltas published during a gateway-NATS disconnect are lost.** Recovery is automatic via the epoch poller (§4.2). The worst-case staleness window is `DEFAULT_POLL_INTERVAL` plus one export round-trip. No pod restart is required. Shortening `DEFAULT_POLL_INTERVAL` trades load on `sie-config` for recovery time.

3. **Bootstrap is background-retried, not blocking.** A gateway started while `sie-config` is unreachable will serve filesystem-seed traffic immediately. API-added models are missing until the first successful export. `ConfigEpoch` typically stays at `0` during this window — but a NATS delta that arrives before the first export can move it above `0`, so admin tooling should use the `sie_gateway_config_bootstrap_degraded` gauge (set after 10 failed attempts or 5 minutes of sustained failure, cleared on first success) as the authoritative "has this replica ever reconciled with `sie-config`?" signal rather than `config_epoch == 0`.

4. **Partial NATS publish on `sie-config`.** If a write succeeds on disk and in the registry but the NATS publish fails for a subset of affected bundles, `sie-config` still returns `201` (or `200` for no-op replays) with a structured `warnings` entry naming the failed bundles. Workers on those bundles stay on the previous epoch until the gateway poller triggers a re-export.

5. **`POST /v1/configs/models` on the gateway returns `405 Method Not Allowed`.** There is no custom body and no redirect pointer. Tooling that still targets the gateway for writes must be updated to call `SIE_CONFIG_SERVICE_URL` directly.

6. **Write-response shape dropped worker-readiness fields.** `sie-config`'s `POST /v1/configs/models` response does not include `worker_ack_pending` / `acked_workers` / `pending_workers`. Admin tooling polls `GET /v1/configs/models/{id}/status` on each gateway replica instead (§6).

7. **Bundles are filesystem-only on `sie-config`.** There is no bundle-write API. Bundles change by redeploying the `sie-config` image (and the worker image, since adapters referenced in a bundle live in the worker). The gateway no longer bakes bundles — it fetches them from `sie-config` at bootstrap and re-fetches on any `bundles_hash` drift (§4.2) — so a bundle addition propagates to running gateway replicas within one `DEFAULT_POLL_INTERVAL` without a gateway redeploy. The natural deploy order is `sie-config` first (so the new bundle is advertised), then workers for the new bundle (so routable instances exist when the gateway learns about the bundle).

   Side effect of this design: `sie-config` is a **cold-start** dependency for gateway replicas.

   - A fresh replica that cannot reach `sie-config` starts with zero bundles AND zero models.
   - Typed inference requests (i.e., any caller that does not pin a pool) will return `404 model not found` until bootstrap completes.
   - The empty-registry fallback described in §2 only fires for callers that explicitly pin a pool via `X-SIE-Pool` — the proxy falls back to the caller-supplied bundle or `"default"`, so a warm pool stays reachable through a cold gateway.
   - An already-running replica is NOT affected by a `sie-config` outage: it keeps serving its last-known bundle set and receives model deltas over NATS when `sie-config` recovers.

   The `gateway.embeddedConfigs` / `gateway.configMap` Helm overlays remain available as an escape hatch that statically seeds the registry and eliminates the cold-start dependency, at the cost of losing live bundle resync.

8. **`registry_unavailable` (503) on `sie-config`.** If the `ModelRegistry` failed to initialize (e.g. malformed bundle YAML), `sie-config`'s `/readyz` returns 503 and every registry-dependent endpoint (`/v1/configs/models`, `/v1/configs/models/{id}`, `/v1/configs/bundles`, `/v1/configs/bundles/{id}`, `/v1/configs/resolve`, `POST /v1/configs/models`, `/v1/configs/export`) returns 503 with a structured `registry_unavailable` error body. `/epoch` is intentionally independent of the registry (it reads the epoch file directly), so the gateway's poller can still discover that `sie-config` is up but not serving — admin tooling should treat `/readyz=503` plus `/epoch=200` as "control plane is alive but wedged; inspect logs".

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
- `SIE_CONFIG_SERVICE_URL` — base URL of `sie-config`. If unset, the bootstrap and poller tasks no-op and the gateway runs with whatever the (optional) filesystem seed loaded — typically empty in the default deploy, which means **no models will be served**. This is intended for local single-process tests only.
- `SIE_ADMIN_TOKEN` — (1) bearer token the gateway-as-client presents on `GET /v1/configs/export` and `GET /v1/configs/epoch`; (2) the token that `AuthLayer` requires for admin-gated mutations on the gateway itself (`POST/PUT/DELETE` on `/v1/configs/*`, `/v1/admin/*`, `/v1/pools/*`). If empty and the matching inbound request targets an admin path, the middleware fails closed with `403`. On outbound calls, if empty, no `Authorization` header is sent (so `sie-config` must either be unauthenticated or the gateway will fail with `401`/`403`).
- `SIE_AUTH_TOKEN` / `SIE_AUTH_TOKENS` — tokens accepted by the gateway's own inference API (`/v1/encode`, `/v1/score`, `/v1/extract`) and for read-side pool/config routes. Not used for outbound calls to `sie-config`. When `SIE_AUTH_MODE` enables auth but this list is empty, every non-probe request returns `500`.
- `SIE_AUTH_MODE` — `token` (alias: `static`) enforces auth; `none` (default) disables it. Typos are fail-open-to-bypass by design; `audit_auth` logs a startup error naming the bad value so operators see it in `kubectl logs`.
- `SIE_AUTH_EXEMPT_OPERATIONAL` — when `true`, `/`, `/health`, `/metrics`, and `/ws/*` are exempt from auth (they expose worker URLs, bundle assignments, queue depth, GPU inventory — treat as sensitive). Default `false`. `/healthz` and `/readyz` are **always** exempt because Kubernetes probes carry no credentials; a misconfiguration must not take the pod out of rotation.
- `SIE_NATS_CONFIG_TRUSTED_PRODUCERS` — comma-separated producer allowlist for `sie.config.models._all`. Default `sie-config`. Matching is exact OR K8s pod-name prefix (see §4.3). Untrusted notifications are dropped; the `config_poller` still closes the gap.
- `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER` — `true` disables producer validation entirely. Intended for local/dev. `main.rs` emits a startup audit warning when on.
- NATS connection variables per the existing gateway configuration.

Config service (`sie-config`):

- `SIE_ADMIN_TOKEN` — write-auth. Without it, writes are rejected if `SIE_AUTH_TOKEN` is also set (inference token cannot implicitly grant write access). If neither token is set, writes are accepted unauthenticated — dev/local only; production always sets `SIE_ADMIN_TOKEN`.
- `SIE_AUTH_TOKEN` — read-auth. Optional.
- `SIE_NATS_URL` — NATS broker URL. Publisher degrades gracefully if unreachable; mutations are blocked with `503` while the publisher is configured but disconnected.
- `SIE_CONFIG_STORE_DIR` — base directory for the on-disk config store. When the Helm `config.configStore.enabled` flag is true, this is set to the mounted PVC path and `SIE_CONFIG_RESTORE=true` enables startup replay from the store into the in-memory registry.
- `SIE_BUNDLES_DIR`, `SIE_MODELS_DIR` — bundle and model source directories. **`sie-config` is the source of truth**: it reads these at startup, validates writes against them, and re-serves the bundle list at `GET /v1/configs/bundles` for the gateway's bootstrap. Defaults via `sharedPaths.{bundlesDir,modelsDir}` (`/app/bundles`, `/app/models`) and image-baked.
- `SIE_LOG_LEVEL`, `SIE_LOG_JSON` — log verbosity and structured-JSON toggle (both also exposed as CLI flags).

## 14. Endpoint Reference

### `sie-gateway`

Inference (queue-only):

- `POST /v1/encode/{*model}`
- `POST /v1/score/{*model}`
- `POST /v1/extract/{*model}`

Config (read-side only):

- `GET /v1/configs/models`
- `GET /v1/configs/models/{*id}` — model YAML, or worker-ack status for `…/status` suffix.
- `GET /v1/configs/bundles`
- `GET /v1/configs/bundles/{id}`
- `POST /v1/configs/resolve`

Pools and operator:

- `GET /v1/pools`, `POST /v1/pools`, `GET /v1/pools/{name}`, `POST /v1/pools/{name}/renew`, `DELETE /v1/pools/{name}`
- `GET /v1/models`
- `GET /healthz`, `GET /readyz`, `GET /health`
- `GET /metrics`
- `GET /ws/cluster-status`
- `GET /` — HTML status page (operator-facing, not a programmatic API).

Not registered on the gateway: `POST /v1/configs/models`, `GET /v1/configs/export`, `GET /v1/configs/epoch`. The export and epoch endpoints are consumed by the gateway as a client of `sie-config`.

### `sie-config`

Config writes (admin auth):

- `POST /v1/configs/models`
- `GET /v1/configs/export`

Config reads (read auth unless noted):

- `GET /v1/configs/models`
- `GET /v1/configs/models/{model_id:path}`
- `GET /v1/configs/bundles`
- `GET /v1/configs/bundles/{bundle_id}`
- `POST /v1/configs/resolve`
- `GET /v1/configs/epoch`

`sie-config` does not serve `GET /v1/configs/models/{id}/status`. That endpoint is gateway-only by design (no worker registry on `sie-config`).

## 15. Metrics

Both services expose Prometheus metrics on `GET /metrics`. The gateway uses the `prometheus` Rust crate; `sie-config` uses `prometheus-client` (matches `sie-server`'s library choice). All metric names use the `sie_gateway_` or `sie_config_` prefix to make cross-service dashboards unambiguous.

### 15.1 Request and latency — gateway

Emitted from a Tower middleware (`MetricsLayer` in `middleware/metrics.rs`) that wraps the proxy routes. Putting it in middleware rather than inline in handlers is the reason these counters now cover every response including early rejections (validation, auth, capacity, timeout) — previously ~27 exit paths in `handlers/proxy.rs` were silently skipping the counter bump.

- `sie_gateway_requests_total{endpoint,status,machine_profile}` — counter.
- `sie_gateway_request_latency_seconds{endpoint,machine_profile}` — histogram. Wall-clock from middleware entry to response.

### 15.2 Routing outcomes — gateway

- `sie_gateway_provisioning_responses_total{machine_profile}` — counter. `202 Accepted` returned when no ready worker exists; paired with `sie_gateway_pending_demand` for KEDA scale-up.
- `sie_gateway_rejected_requests_total{machine_profile,bundle,reason}` — counter. In-handler emission because the `reason` label (`timeout`, `capacity`, `no_workers`, ...) is too granular for the middleware.
- `sie_gateway_pending_demand{machine_profile,bundle}` — gauge. KEDA trigger. Cleared by `clear_fulfilled_demand` when healthy workers appear.
- `sie_gateway_active_lease_gpus{machine_profile,bundle}` — gauge. KEDA trigger. Recomputed from the current pool list on every `update_pool_metrics`.

### 15.3 Worker and model state — gateway

- `sie_gateway_workers{status}` — gauge, healthy/unhealthy counts.
- `sie_gateway_worker_queue_depth{worker,machine_profile,bundle}` — gauge.
- `sie_gateway_worker_memory_used_bytes{worker,machine_profile,bundle}` — gauge.
- `sie_gateway_model_workers{model}` — gauge, worker count per model.

### 15.4 Queue and DLQ — gateway

- `sie_gateway_queue_publish_seconds{operation}` — histogram, JetStream publish latency.
- `sie_gateway_queue_items_published{operation}` — histogram, batch size.
- `sie_gateway_queue_result_wait_seconds{operation}` — histogram, time from publish to result receipt on the NATS Core inbox.
- `sie_gateway_queue_payload_offloads_total` — counter, large-payload offload events.
- `sie_gateway_queue_inbox_skips_total` — counter, fast-path request-id skips on the inbox.
- `sie_gateway_queue_ack_failures_total` — counter, fire-and-forget JetStream ack monitoring.
- `sie_gateway_dlq_events_total{stream,consumer}` — counter, **successful** DLQ forwards.
- `sie_gateway_dlq_republish_failures_total{stream,consumer}` — counter, DLQ publish failures. Companion to the success counter; without it a DLQ publish outage was invisible on the existing metric.

### 15.5 Pool lifecycle — gateway

- `sie_gateway_pool_events_total{event}` — counter. `event` is one of `created`, `renewed`, `deleted`, `expired`. Complements the `active_lease_gpus` gauge: the gauge shows current state, this counter shows the rate of churn. A 5-minute rate over `event="expired"` catches runaway expirations; a sustained `event="created"` rate catches runaway pool creation.

### 15.6 Config plane — gateway

- `sie_gateway_config_bootstrap_failures_total` — counter. Background bootstrap retry failures (see §10).
- `sie_gateway_config_bootstrap_degraded` — gauge, `0` or `1`. Set to `1` once the gateway has been serving the filesystem seed past the degraded threshold; cleared on the first successful bootstrap.
- `sie_gateway_config_epoch` — gauge. Highest-known control-plane epoch on this gateway. Updated by bootstrap fetch, NATS Core delta handler, and the epoch poller. Pairs with `sie_config_epoch` for drift alerting (§15.9).
- `sie_gateway_config_deltas_total{kind,result}` — counter. NATS Core config-delta processing outcomes. `kind` is the delta kind (`epoch_bump`, `model_added`); `result` is `applied`, `parse_error`, `apply_error`, or `rejected_untrusted`.
- `sie_gateway_nats_connected` — gauge, `0` or `1`. Flipped by the NATS client's connect/disconnect event callbacks. Unlabeled: the gateway uses a single `async-nats` client for JetStream (work + results) and NATS Core (config deltas), so a single gauge accurately represents the only connection-state dimension that exists today. If we later split clients per purpose, re-introduce a `stream` label at that point.

### 15.7 Request and latency — sie-config

Emitted from a Starlette ASGI middleware (`_PrometheusHTTPMiddleware` in `app_factory.py`) that wraps the FastAPI app. The `path` label is the **FastAPI route template** (e.g. `/v1/configs/models/{model_id:path}`), not the raw URL — otherwise per-model reads would each produce their own time series. The middleware intentionally skips `/metrics` itself so scrape traffic does not pollute the counter.

- `sie_config_http_requests_total{method,path,status}` — counter.
- `sie_config_http_request_duration_seconds{method,path}` — histogram. Wall-clock end-to-end inside the ASGI app.

### 15.8 Config plane state — sie-config

- `sie_config_epoch` — gauge. Authoritative epoch persisted in `ConfigStore`. Seeded on startup from `ConfigStore.read_epoch`, updated on every successful `increment_epoch`. `0` when no `ConfigStore` is configured (standalone / test), which is the correct "never bumped" value.
- `sie_config_models_total{source}` — gauge. `source` is `api` (added via `POST /v1/configs/models`, persisted in `ConfigStore`) or `filesystem` (seeded from the on-disk `bundles/` + `models/` directories). Refreshed on each successful write.
- `sie_config_nats_connected` — gauge, `0` or `1`. Flipped by `NatsPublisher` in the connect / disconnect / reconnect callbacks.
- `sie_config_nats_publishes_total{result}` — counter. One sample per `publish_config_notification` call. `result` is `success`, `partial`, or `failure`. `partial` means at least one bundle subject succeeded and at least one failed — operators treat this differently from `failure` because the gateway poller closes a partial gap within ~30 s.
- `sie_config_store_writes_total{op,result}` — counter. `op` is `write_model` or `increment_epoch`; `result` is `success` or `failure`. A sustained `failure` rate is a PVC / disk-backend issue.

### 15.9 Cross-service drift alert

The most important alert across both services. Fires when any gateway's view of the control plane lags the authoritative epoch published by `sie-config`:

```promql
(sie_config_epoch - on() sie_gateway_config_epoch) > 0
```

Sustained non-zero for >1 minute means either NATS Core deltas are not flowing (pub/sub has no replay, so a disconnected gateway silently falls behind) or the epoch poller is stuck. The poller normally closes this gap within its poll interval regardless of NATS health, so a sustained alert here is always an actionable incident.

## 16. Worked Example: end-to-end request

This section walks a concrete `encode` request through the system so the moving parts in §1–§15 have an explicit reference trace.

### 16.1 Bundles and model → bundle resolution

A **bundle** defines which Python adapter modules a worker deployment can run. Two shapes ship today:

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
- sie_server.adapters.splade_flash
# … 30+ more
```

**`sglang` bundle** (priority `20`) — one adapter for large LLM-based models using the SGLang inference engine:

```yaml
# bundles/sglang.yaml
name: sglang
priority: 20
adapters:
- sie_server.adapters.sglang
```

`ModelRegistry` maps a model to a bundle by matching the model's `adapter_path` module against each bundle's `adapters` list, then preferring the lowest-priority bundle in the match set:

```
Model: BAAI/bge-m3
  adapter_path: sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter
  adapter module: sie_server.adapters.bge_m3_flash
  → matches "default" bundle
  → routes to "default" workers

Model: Qwen/Qwen3-Embedding-4B
  adapter_path: sie_server.adapters.sglang:SGLangEmbeddingAdapter
  adapter module: sie_server.adapters.sglang
  → matches "sglang" bundle
  → routes to "sglang" workers
```

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
Subject:  sie.work.BAAI__bge-m3.default
Stream:   WORK_POOL_default
```

The work subject is constructed by `work_subject(model, pool)` in `queue/publisher.rs`, which runs the model id through `normalize_model_id` (the Rust mirror of `sie_sdk.queue_types.normalize_model_id`: `/` → `__`, `.` → `_dot_`, `*`/`>`/space → `_`). The result is always exactly four dot-separated tokens so it matches the worker's `sie.work.*.{pool}` consumer filter — this is the guarantee that makes dotted ids like `vidore/colqwen2.5-v0.2` work. The DLQ path extracts the third token from the advisory subject, replaces any leftover `/` with `_` as a legacy fallback, and publishes on `sie.dlq.{model_normalized}` — so `sie.work.BAAI__bge-m3.default` becomes `sie.dlq.BAAI__bge-m3` when a message hits max-deliveries.

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
  pool_name:     "default",
  machine_profile: "l4-spot",
  item:          { "text": "..." },       // or omitted + payload_ref if >1MB
  reply_subject: "_INBOX.gw-1.abc-123",
  bundle_config_hash: "a1b2c3…",
  …
}
```

JetStream publish is fire-and-forget: the gateway buffers publishes to the `async-nats` client and spawns a detached task to drain the acks, so the request handler does not block per-item on the JetStream round-trip. Ack failures are logged and counted via `sie_gateway_queue_ack_failures_total`. Backpressure and no-consumer conditions are caught earlier via a cached stream-info check (`stream_info_cache`), so a stalled stream surfaces as a prompt `503` rather than a stuck request.

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

On each inbox message, the gateway first runs `extract_request_id_fast` — a bounded scan of the raw msgpack bytes that locates the `request_id` (or, for array-shape `WorkResult` payloads, the first string field, which the current codec places at the `request_id` / `work_item_id` position) without a full deserialization. If the extracted ID is not present in the `pending_results` `DashMap`, the message is dropped immediately (already completed, duplicate redelivery, or stale). Only when it is live does the handler run `rmp_serde::from_slice` to decode the full `WorkResult`. This lets the gateway shed redelivered JetStream messages at near-memcmp cost.

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
  "bundle": "sglang",
  "ttl_seconds": 3600
}
```

Each pool gets its own JetStream stream (`WORK_POOL_{name}`) with subjects `sie.work.*.{name}`. Workers are deployed with `SIE_POOL={name}` and consume only from their own pool's stream. That is the isolation boundary.

Creating a usable custom pool is a two-step operation: add the pool to Helm values so a `StatefulSet` of workers actually exists, then register the pool on the gateway via `POST /v1/pools` so it tracks fulfillment. Clients then target the pool with an `X-SIE-POOL: customer-acme` header. Pools expire after their TTL unless renewed with `POST /v1/pools/{name}/renew`; the `default` pool is protected and cannot be deleted.

Model configs registered via the control plane (`sie-config`) are not tied to a pool — they describe what the cluster can serve, not who gets to serve it. Pool membership, worker counts, and GPU types are Helm-driven and independent of the config API.
