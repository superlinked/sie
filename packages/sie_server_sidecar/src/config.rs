//! Worker runtime configuration.
//!
//! All runtime knobs flow through CLI args / env vars in `main.rs`; this
//! struct is the one-shot snapshot passed into `run()`.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// NATS server URL (e.g. `nats://localhost:4222`). Required for the
    /// default NATS ingest (`run()`); `None` is valid only for the
    /// local-ingest mode (`run_local()`, P2.10 §4.6) which never
    /// touches NATS.
    pub nats_url: Option<String>,

    /// UDS path for the local-ingest listener (`SIE_SIDECAR_LOCAL_SOCKET`).
    /// Only read by `run_local()`; `None` on NATS deployments.
    pub local_socket_path: Option<PathBuf>,

    /// Pool name — drives the stream (`WORK_POOL_{pool}`), durable consumer
    /// (`{pool}_{machine_profile}_{bundle}`), and subject filters.
    pub pool: String,

    /// Bundle ID — forms part of the durable consumer name and subject lane
    /// so multiple bundles on the same pool don't step on each other.
    pub bundle: String,

    /// Primary Unix domain socket used to talk to the colocated adapter worker.
    pub ipc_socket_path: PathBuf,

    /// Unix domain sockets used to talk to adapter worker children. Defaults to
    /// [`Self::ipc_socket_path`] for the single-worker baseline.
    pub ipc_socket_paths: Vec<PathBuf>,

    /// Number of concurrent IPC connections to each adapter worker process. `1`
    /// preserves the legacy single-socket behaviour; higher values let
    /// the dispatcher's `SIE_MAX_CONCURRENT_BATCHES` actually drive
    /// parallel backend-side batches. Sourced from `SIE_IPC_POOL_SIZE`
    /// (see `main.rs`); when unset we default to
    /// `SIE_MAX_CONCURRENT_BATCHES`'s default (4).
    pub ipc_pool_size: usize,

    /// Per-RPC timeout for ordinary sidecar → adapter IPC calls. Sourced from
    /// `SIE_IPC_REQUEST_TIMEOUT_S`.
    pub ipc_request_timeout_s: u64,

    /// Timeout for sidecar → adapter `EnsureModelReady` calls. Must be at least
    /// as long as the slowest expected cold start; SGLang adapters may
    /// legitimately spend many minutes loading large models before they can
    /// answer the readiness handshake. Sourced from
    /// `SIE_MODEL_READY_TIMEOUT_S`.
    pub model_ready_timeout_s: u64,

    /// Optional payload store URL — if unset, workers expect items inline
    /// (large items will be rejected by the gateway's offload). Local paths
    /// point at a shared directory; `s3://…` / `gs://…` / `abfs://…` /
    /// `abfss://…` use cloud stores when built with `--features cloud-storage`.
    pub payload_store_url: Option<String>,

    /// Optional gateway URL used by the worker-side pool admission gate.
    /// When set and admission is enabled, the sidecar polls `/v1/pools`
    /// before pulling from NATS so it can enforce both the physical
    /// `SIE_POOL` assignment and logical `admission_pool` assignments backed
    /// by that queue.
    pub gateway_url: Option<String>,

    /// Optional bearer token for gateway pool-status reads.
    pub gateway_api_key: Option<String>,

    /// Whether the pool admission gate is enabled. The gate still no-ops
    /// when `gateway_url` is unset so local NATS-only harnesses continue to
    /// work.
    pub pool_admission_enabled: bool,

    /// Pool admission status check cadence.
    pub pool_admission_check_interval_ms: u64,

    /// Sleep duration while this worker is not admitted to pull.
    pub pool_admission_pause_ms: u64,

    /// How long to reuse the last successful admission decision after
    /// transient gateway/status errors.
    pub pool_admission_stale_after_ms: u64,

    /// Liveness/readiness probe HTTP port.
    pub probe_port: u16,

    /// Stable worker identifier surfaced in logs / `WorkResult.worker_id` /
    /// IPC `Ping`.
    pub worker_id: String,

    /// How often to send `Ping` RPCs to adapter worker processes.
    pub ping_interval_ms: u64,

    /// Multiplier applied to `ping_interval_ms` to compute the
    /// `/readyz` heartbeat-staleness threshold. The sidecar's
    /// readiness flips red once the most recent successful `Ping`
    /// is older than `ping_interval_ms * ready_stale_mult`.
    ///
    /// Default `3`. Override with `SIE_WORKER_READYZ_STALE_MULT`
    /// when ops want a looser bound (e.g. `5` to roughly match the
    /// adapter worker's historical 10 s window with a 2 s ping).
    /// `0` falls back to the default at construction time so a
    /// misconfigured env var doesn't make the pod look unready on
    /// the very tick after a successful ping.
    pub ready_stale_mult: u32,

    /// Machine-profile label this pod advertises in NATS
    /// heartbeats (e.g. `l4`, `a100`). Surfaced to the gateway
    /// only — the `X-SIE-MACHINE-PROFILE` route filter compares
    /// case-insensitively against this value. Empty disables the
    /// filter (route by bundle alone). Sourced from
    /// `SIE_MACHINE_PROFILE`; required so the queue lane is explicit.
    pub machine_profile: String,

    /// GPU count surfaced in heartbeats. Informational —
    /// `WorkerRegistry::update_worker` coerces `0 -> 1` so an
    /// unset value still routes. Sourced from `SIE_GPU_COUNT`.
    pub gpu_count: i32,

    /// Optional bundle-config hash echoed in heartbeats so admin
    /// tooling can correlate the worker's bundle revision with
    /// the gateway's model registry epoch. Empty is fine.
    pub bundle_config_hash: String,

    /// Optional URL for the `sie-config` control plane. When set, the
    /// sidecar polls `/v1/configs/epoch` and reconciles missed deltas from
    /// `/v1/configs/export`.
    pub config_service_url: Option<String>,

    /// Optional bearer token for `sie-config` export reads. Helm wires this
    /// from `SIE_ADMIN_TOKEN` when config auth is enabled because
    /// `/v1/configs/export` is admin-authenticated.
    pub config_service_token: Option<String>,

    /// Cadence for worker-side `/v1/configs/epoch` polling.
    pub config_poll_interval_ms: u64,

    /// Slow full-export reconciliation cadence. This covers no-config-store
    /// deployments where `sie-config` keeps epoch at `0`; `0` disables the
    /// periodic export audit after startup.
    pub config_full_export_interval_ms: u64,

    /// Trusted producer allowlist for `sie.config.models.<bundle>`
    /// notifications. Empty means trust any producer and is intended only for
    /// local/dev clusters.
    pub nats_config_trusted_producers: Vec<String>,

    /// Heartbeat interval for the NATS health publisher.
    /// Defaults to 5 s — same cadence as the gateway's
    /// `start_heartbeat_loop` so the staleness check has a 6×
    /// margin against the 30 s `heartbeat_timeout`.
    pub health_publish_interval_ms: u64,
}

impl WorkerConfig {
    pub fn stream_name(&self) -> String {
        format!("WORK_POOL_{}", self.pool)
    }

    pub fn stream_subject_filter(&self) -> String {
        format!("sie.work.{}.*.*.*", self.pool)
    }

    pub fn consumer_name(&self) -> String {
        // Matches `sie_sdk.queue_types.work_consumer_name(pool, machine, bundle)` so
        // Rust and Python adapter processes converge on the same durable consumer.
        format!(
            "{}_{}_{}",
            crate::subject::normalize_model_id(&self.pool),
            crate::subject::normalize_model_id(&self.machine_profile),
            crate::subject::normalize_model_id(&self.bundle)
        )
    }

    pub fn subject_filter(&self) -> String {
        format!(
            "sie.work.{}.{}.{}.*",
            self.pool,
            crate::subject::normalize_model_id(&self.machine_profile),
            crate::subject::normalize_model_id(&self.bundle)
        )
    }

    pub fn worker_stream_name(&self) -> String {
        format!(
            "WORK_WORKER_{}",
            crate::subject::normalize_model_id(&self.worker_id)
        )
    }

    pub fn worker_consumer_name(&self) -> String {
        format!(
            "gen-{}",
            crate::subject::normalize_model_id(&self.worker_id)
        )
    }

    pub fn worker_subject_filter(&self) -> String {
        format!(
            "sie.work.{}.{}.{}.*.{}",
            self.pool,
            crate::subject::normalize_model_id(&self.machine_profile),
            crate::subject::normalize_model_id(&self.bundle),
            crate::subject::normalize_model_id(&self.worker_id)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> WorkerConfig {
        WorkerConfig {
            nats_url: Some("nats://localhost:4222".into()),
            local_socket_path: None,
            pool: "l4".into(),
            bundle: "default".into(),
            ipc_socket_path: PathBuf::from("/tmp/sie-ipc.sock"),
            ipc_socket_paths: vec![PathBuf::from("/tmp/sie-ipc.sock")],
            ipc_pool_size: 1,
            ipc_request_timeout_s: 60,
            model_ready_timeout_s: 900,
            payload_store_url: None,
            gateway_url: None,
            gateway_api_key: None,
            pool_admission_enabled: true,
            pool_admission_check_interval_ms: 5_000,
            pool_admission_pause_ms: 1_000,
            pool_admission_stale_after_ms: 30_000,
            probe_port: 9095,
            worker_id: "worker-test".into(),
            ping_interval_ms: 2000,
            ready_stale_mult: 3,
            machine_profile: "l4".into(),
            gpu_count: 1,
            bundle_config_hash: String::new(),
            config_service_url: None,
            config_service_token: None,
            config_poll_interval_ms: 30_000,
            config_full_export_interval_ms: 300_000,
            nats_config_trusted_producers: vec!["sie-config".into()],
            health_publish_interval_ms: 5_000,
        }
    }

    #[test]
    fn stream_and_consumer_names_match_gateway_contract() {
        let c = sample();
        // Must agree with sie_gateway's NATS naming so publisher and
        // consumer land on the same stream/consumer.
        assert_eq!(c.stream_name(), "WORK_POOL_l4");
        assert_eq!(c.stream_subject_filter(), "sie.work.l4.*.*.*");
        assert_eq!(c.consumer_name(), "l4_l4_default");
        assert_eq!(c.subject_filter(), "sie.work.l4.l4.default.*");
        assert_eq!(c.worker_stream_name(), "WORK_WORKER_worker-test");
        assert_eq!(c.worker_consumer_name(), "gen-worker-test");
        assert_eq!(
            c.worker_subject_filter(),
            "sie.work.l4.l4.default.*.worker-test"
        );
    }

    #[test]
    fn subject_filter_contains_pool() {
        let mut c = sample();
        c.pool = "eval-h100".into();
        assert!(c.stream_subject_filter().starts_with("sie.work.eval-h100."));
        assert!(c.subject_filter().starts_with("sie.work.eval-h100."));
    }
}
