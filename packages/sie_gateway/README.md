# SIE Gateway (Rust)

Queue-only runtime gateway for elastic GPU inference deployments. It routes `encode`, `score`, and `extract` requests over NATS JetStream and owns pool coordination and worker health. Config writes live in `sie-config`; the gateway is a pure consumer of config state, bootstrapping from `GET /v1/configs/export` and subscribing to NATS deltas.

See [`docs/architecture-guide.md`](docs/architecture-guide.md) for the authoritative code-audited architecture document (covers both this service and `sie-config`).

## Features

- **Queue-only inference** — no direct worker HTTP fallback path
- **Worker discovery** — static URLs or Kubernetes service endpoints
- **Health monitoring** — WebSocket streaming or NATS heartbeats
- **Queue transport** — JetStream work publish, reply inbox collection, backpressure handling, and DLQ republish
- **Pool management** — named pools with TTLs, minimum worker counts, and Kubernetes-backed coordination
- **Model registry** — filesystem seed + background-retried snapshot from `sie-config` (`GET /v1/configs/export`) + live NATS deltas + periodic `GET /v1/configs/epoch` drift detection; the gateway is read-only
- **Config write cutover** — `POST /v1/configs/models` is not registered on the gateway. Requests receive `405 Method Not Allowed` from axum's default router. Writes belong to the control plane at `SIE_CONFIG_SERVICE_URL`
- **Worker-ack readiness** — `GET /v1/configs/models/{id}/status` reports per-replica `bundle_config_hash` acknowledgement plus the local `config_epoch` for admin tooling polling after a `sie-config` write
- **Config distribution** — authoritative deltas arrive on `sie.config.models.*` from `sie-config`; the gateway applies them to its in-memory registry
- **Auth** — static-token auth via `SIE_AUTH_TOKEN[S]`; config write idempotency belongs to `sie-config`
- **Demand tracking and readiness** — provisioning responses, pending-demand metrics, and worker ack checks after config changes
- **Observability** — Prometheus metrics, structured logging, audit middleware, and HTML/WebSocket status surfaces
- **Optional cloud storage** — `cloud-storage` feature enables S3/GCS payload backends; the Docker build enables it

## Quick Start

**Requirements:** Rust stable, NATS/JetStream for inference routing

```bash
# Build from the repo root (preferred contributor flow)
mise run gateway-build -- -r

# Direct cargo equivalent from this package directory
cargo build --release

# Run with static workers
SIE_NATS_URL=nats://localhost:4222 \
./packages/sie_gateway/target/release/sie-gateway serve \
  -w http://worker1:8080 \
  -w http://worker2:8080

# Run with Kubernetes discovery
SIE_NATS_URL=nats://localhost:4222 \
./packages/sie_gateway/target/release/sie-gateway serve \
  --kubernetes \
  --k8s-namespace sie \
  --k8s-service sie-worker
```

Notes:

- The gateway is queue-only: if no usable NATS client is available, inference requests return `503`.

## CLI

```text
sie-gateway serve [OPTIONS]

Key options:
  -p, --port <PORT>              Listen port (default: 8080)
      --host <HOST>              Listen host (default: 0.0.0.0)
  -w, --worker <WORKERS>         Worker URL(s), repeatable
      --kubernetes               Enable Kubernetes discovery
      --k8s-namespace <NS>       K8s namespace (default: default)
      --k8s-service <SVC>        K8s service name (default: sie-worker)
      --k8s-port <PORT>          K8s worker port (default: 8080)
  -l, --log-level <LEVEL>        Log level (default: info)
      --json-logs                Enable structured JSON logging
      --health-mode <MODE>       Health mode: ws|nats (default: ws)
      --bundles-dir <PATH>       Bundles directory
      --models-dir <PATH>        Models directory

sie-gateway version
```

Each `--flag` above has a matching `SIE_*` environment variable (see next section); CLI flags override env vars.

## Common Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIE_GATEWAY_PORT` | `8080` | Listen port |
| `SIE_GATEWAY_HOST` | `0.0.0.0` | Listen host |
| `SIE_GATEWAY_WORKERS` | | CSV of worker URLs |
| `SIE_GATEWAY_KUBERNETES` | `false` | Enable Kubernetes discovery |
| `SIE_GATEWAY_K8S_NAMESPACE` | `default` | K8s namespace |
| `SIE_GATEWAY_K8S_SERVICE` | `sie-worker` | K8s service name |
| `SIE_GATEWAY_K8S_PORT` | `8080` | K8s worker port |
| `SIE_GATEWAY_HEALTH_MODE` | `ws` | Health mode: `ws` or `nats` |
| `SIE_NATS_URL` | | NATS server URL. The process can start without it, but inference requests will return `503` until a usable client exists |
| `SIE_AUTH_MODE` | `none` | Auth mode for inbound requests: `none` disables, `token` (alias `static`) enforces. Unknown values fail-open-to-bypass; `main` logs a startup error naming the bad value |
| `SIE_AUTH_TOKENS` | | CSV of valid bearer tokens for inference and pool/config read endpoints. If unset, the singular `SIE_AUTH_TOKEN` is used as a fallback. When auth is enabled and this list is empty, non-probe requests return `500` |
| `SIE_AUTH_TOKEN` | | Singular alias for `SIE_AUTH_TOKENS` (fallback only; prefer the plural form) |
| `SIE_ADMIN_TOKEN` | | Admin bearer token the gateway (1) presents **as a client** to `sie-config` on `GET /v1/configs/export` and `GET /v1/configs/epoch`, and (2) requires inbound for admin-gated mutations: `POST/PUT/DELETE` on `/v1/configs/*`, `/v1/admin/*`, `/v1/pools/*`. If empty and an inbound request targets one of those paths, the middleware fails closed with `403` |
| `SIE_AUTH_EXEMPT_OPERATIONAL` | `false` | When `true`, `/`, `/health`, `/metrics`, and `/ws/*` are exempt from auth (they expose worker URLs, queue depth, GPU inventory). `/healthz` and `/readyz` are always exempt (K8s probes carry no creds). Default is fail-closed |
| `SIE_NATS_CONFIG_TRUSTED_PRODUCERS` | `sie-config` | CSV allowlist of `producer_id` values trusted to publish on `sie.config.models._all`. Matches exact OR K8s pod-name prefix (`sie-config` also matches `sie-config-5f7b6d8c-kxwvr`). Untrusted notifications are dropped; the epoch poller still closes the gap |
| `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER` | `false` | Disable producer validation entirely (dev/local only). `main` emits a startup audit warning when on |
| `SIE_LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warn`, `error`) |
| `SIE_LOG_JSON` | `false` | Structured JSON logging (for Loki) |
| `SIE_GATEWAY_REQUEST_TIMEOUT` | `30.0` | Request timeout in seconds |
| `SIE_GATEWAY_MAX_STREAM_PENDING` | `50000` | Max pending stream items per JetStream work stream |
| `SIE_GATEWAY_ENABLE_POOLS` | `false` | Enable pool management |
| `SIE_GATEWAY_HOT_RELOAD` | `false` | Enable filesystem watcher for bundle/model directories |
| `SIE_GATEWAY_WATCH_POLLING` | `false` | Use polling file-watcher instead of inotify/fsevents (alias: `SIE_GATEWAY_POLLING_WATCHER`). Useful on filesystems where native notifications are unreliable |
| `SIE_CONFIG_SERVICE_URL` | unset | Base URL of `sie-config`. When set, the gateway runs a background `GET /v1/configs/export` bootstrap on startup and a 30 s `GET /v1/configs/epoch` drift poller. When unset, the bootstrap/poller tasks no-op and the gateway runs filesystem-seed-only |
| `SIE_MULTI_ROUTER` | `false` | Multi-gateway coordination flag (wire-compatible name retained) |
| `SIE_GATEWAY_CONFIGURED_GPUS` | | CSV of configured GPU types |
| `SIE_BUNDLES_DIR` | `bundles` | Optional bundle filesystem seed. Unset in default Helm deploys: the gateway pulls bundles from `sie-config` via `GET /v1/configs/bundles{,/{id}}` at startup and the registry's filesystem reload is a no-op. Only set by the `gateway.embeddedConfigs` / `gateway.configMap` overlays which mount a ConfigMap at `/configs/bundles`. |
| `SIE_MODELS_DIR` | `models` | Optional model filesystem seed. Same semantics as `SIE_BUNDLES_DIR`: unset in default deploys, runtime model writes always go to `sie-config` and the gateway replays them via `GET /v1/configs/export`. |
| `SIE_PAYLOAD_STORE_URL` | `payload_store` | Payload offload store path; `s3://` and `gs://` require the `cloud-storage` feature |

## API Endpoints

### Health And Operator Surface

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | HTML status page |
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe |
| GET | `/health` | Cluster health JSON |
| GET | `/metrics` | Prometheus metrics |
| GET | `/ws/cluster-status` | WebSocket cluster status feed |
| GET | `/v1/models` | List available models |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/encode/{*model}` | Queue an encode request |
| POST | `/v1/score/{*model}` | Queue a score request |
| POST | `/v1/extract/{*model}` | Queue an extract request |

Common headers:

- `X-SIE-MACHINE-PROFILE`
- `X-SIE-POOL`
- `X-SIE-SDK-Version`

Common behaviors:

- `404` for unknown models once the in-memory registry has bootstrapped from `sie-config` (fast-fail; avoids queueing requests for typo'd model ids)
- `202` + `Retry-After` on scale-from-zero, whether or not `X-SIE-MACHINE-PROFILE` was set (records pending demand for KEDA)
- `503` + `Retry-After` for no-consumer or backpressure publish failures
- `503` + `X-SIE-Error-Code: MODEL_LOADING` + `Retry-After: 5` for queue result timeouts (typically a worker cold-loading the target model). SDK clients with `wait_for_capacity=True` retry under the existing `provision_timeout_s` budget.
- `503` + `X-SIE-Error-Code: RESOURCE_EXHAUSTED` + `Retry-After: 5` when every item in a batch fails with the same retryable code (`RESOURCE_EXHAUSTED` from worker-side OOM recovery exhaustion, `MODEL_LOADING` from a worker still warming up). The SDK auto-retries with bounded exponential backoff. Mixed batches keep returning `500 all_items_failed` with per-item `code` fields in the response body so callers can see which items hit which failure mode.

### Pool Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/pools` | List pools |
| POST | `/v1/pools` | Create pool |
| GET | `/v1/pools/{name}` | Get pool details |
| POST | `/v1/pools/{name}/renew` | Renew pool TTL |
| DELETE | `/v1/pools/{name}` | Delete pool (default pool protected) |

### Config API (read-only)

The gateway is a pure consumer of config state. Writes live in `sie-config`; the gateway returns `405 Method Not Allowed` for `POST /v1/configs/models`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/configs/models` | List model configs known to this replica |
| GET | `/v1/configs/models/{*id}` | Get model config YAML, or — when the path ends in `/status` and the prefix matches a known model — a worker-ack readiness JSON document for admin tooling |
| GET | `/v1/configs/bundles` | List bundle configs |
| GET | `/v1/configs/bundles/{id}` | Get bundle config |
| POST | `/v1/configs/resolve` | Resolve a bundle for a model |

`GET /v1/configs/models/{id}/status` is gateway-only (it reports per-replica worker-ack state from the gateway's in-memory `WorkerRegistry`). `sie-config` does not serve it.

Not registered on the gateway:

- `POST /v1/configs/models` — write, owned by `sie-config`.
- `GET /v1/configs/export`, `GET /v1/configs/epoch` — served by `sie-config`; the gateway is a *client* of both (bootstrap + drift poll). See `packages/sie_gateway/docs/architecture-guide.md` §4.

## Docker

The gateway image is built via the monorepo Docker tooling; see `tools/mise_tasks/docker.bash` and `deploy/` for chart/values.

## Testing

```bash
# Preferred repo-root contributor flow
mise run gateway-fmt              # applies rustfmt (default); add `-- --check` for CI-style check-only
mise run gateway-test
mise run gateway-clippy

# Direct cargo equivalents from this package directory
cargo fmt --all                   # append `--check` to verify without writing
cargo test
cargo clippy --all-targets -- -D warnings
```

## Project Structure

```text
src/
  main.rs                CLI parsing and async runtime startup
  server.rs              Axum routes and AppState
  config.rs              Config loading from env/CLI
  error.rs               AppError -> HTTP status mapping
  metrics.rs             Prometheus metrics
  handlers/
    health.rs            Health and status endpoints
    models.rs            GET /v1/models helpers
    pools.rs             Pool CRUD
    proxy.rs             Queue-only encode/score/extract routing
    config_api.rs        Read-only config API (GET /v1/configs/*, including /status dispatch); POST /v1/configs/models is NOT registered (gateway returns 405)
  middleware/
    auth.rs              Token authentication
    audit.rs             Request/response audit logging
  discovery/
    static_discovery.rs  Static worker list
    ws_health.rs         WebSocket worker health
    nats_health.rs       NATS worker health
    k8s_discovery.rs     Kubernetes endpoint discovery
  state/
    worker_registry.rs   Worker tracking and queue-pool resolution
    model_registry.rs    Model and bundle registry (in-memory)
    pool_manager.rs      Pool management
    k8s_pool_backend.rs  K8s ConfigMap/Lease pool storage
    k8s_pool_watcher.rs  K8s pool state watcher
    config_watcher.rs    Filesystem hot reload
    config_bootstrap.rs  Cold-start snapshot fetch from sie-config
    config_poller.rs     30 s epoch drift detector against sie-config
    config_epoch.rs      Monotonic config-epoch counter (AtomicU64 with CAS)
    demand_tracker.rs    Pending-demand tracking
  nats/
    manager.rs           NATS connection and config-delta subscription
  queue/
    publisher.rs         JetStream work publishing
    consumer.rs          Work consumption helpers
    dlq.rs               Dead-letter queue handling
    payload_store.rs     Payload offload storage
```

## Key Dependencies

- [axum](https://github.com/tokio-rs/axum) — HTTP framework
- [tokio](https://github.com/tokio-rs/tokio) — async runtime
- [clap](https://github.com/clap-rs/clap) — CLI parsing
- [async-nats](https://github.com/nats-io/nats.rs) — NATS/JetStream client
- [kube](https://github.com/kube-rs/kube) — Kubernetes client
- [prometheus](https://github.com/tikv/rust-prometheus) — metrics
- `serde`, `serde_json`, `serde_yaml` — config and payload serialization
