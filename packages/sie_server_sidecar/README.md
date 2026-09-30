# SIE Server Sidecar

`sie-server-sidecar` builds the `sie-server-sidecar` binary and `sie-server-sidecar`
container image.

The sidecar owns the queue-mode runtime around an inference adapter pod: NATS
JetStream consumption, batching and scheduling, IPC to the adapter, payload
fetching, result framing, ACK/NAK behavior, canonical telemetry, and readiness.

It does not load model weights or link GPU libraries. The colocated
`sie-server` container remains the Python adapter/model-execution process.

Public and runtime names:

- Kubernetes container: `worker-sidecar`
- Binary: `sie-server-sidecar`
- Image: `ghcr.io/superlinked/sie-server-sidecar`
- Metrics: push-only OTLP via the `sie.worker.*` contract; Prometheus
  compatibility is a collector/exporter concern

The sidecar also has a brokerless local-ingest mode for trusted colocated
callers. Its length-prefixed MessagePack protocol sends ordinary primitives
through the same Rust preparation, scheduling, admission, and backend pipeline
without NATS. Generation uses a separate streaming operation: each semantic
chunk is validated and emitted through the canonical v0.2 bounded writer. Each
connection also has fixed operation-count and retained-request-byte budgets;
when either is full, the reader stops accepting frame bodies until capacity
returns. Active envelope IDs are unique, so concurrent responses cannot become
ambiguous. Exactly one transport
terminal closes each operation, and cancellation or client disconnect is
forwarded to the Python generation processor with a finite drain.
`timeout_ms=0` leaves generation timing to the gateway. Rust normalizes legacy
base64 generation images to bounded MessagePack binary; Python accepts binary
and the legacy representation for mixed-version compatibility.

The local socket is an access-controlled process boundary, not an
authentication protocol. The listener restricts its socket node, validates a
domain-separated request digest and route identity, caps frame/media/request
sizes, and rejects unresolved payload references.

NATS credentials come from `SIE_NATS_USER` and `SIE_NATS_PASSWORD`, which are
read from the environment only and must be set together. Credentials embedded
in `SIE_NATS_URL` are not used. The subjects the Helm chart's `sie-worker` user
may use are listed in the architecture guide.

With the `cloud-storage` feature, payload reads support S3, GCS, Azure Blob,
and native `oss://` Alibaba OSS Signature V4. OSS requires `SIE_OSS_REGION` and
uses explicit environment credentials or ACK RRSA OIDC only; node metadata,
profiles, and credential-file fallbacks are excluded.

See [`docs/architecture-guide.md`](docs/architecture-guide.md) for the runtime
contract and deployment caveats.
