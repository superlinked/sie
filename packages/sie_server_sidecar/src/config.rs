//! Worker runtime configuration.
//!
//! All runtime knobs flow through CLI args / env vars in `main.rs`; this
//! struct is the one-shot snapshot passed into `run()`.

use std::path::PathBuf;

/// NATS user/password pair (`SIE_NATS_USER` / `SIE_NATS_PASSWORD`).
#[derive(Clone, PartialEq, Eq)]
pub struct NatsCredentials {
    pub user: String,
    pub password: String,
}

impl NatsCredentials {
    /// `None` when neither value is set, an error when only one is.
    pub fn from_parts(
        user: Option<String>,
        password: Option<String>,
    ) -> Result<Option<Self>, String> {
        let user = user.filter(|v| !v.is_empty());
        let password = password.filter(|v| !v.is_empty());
        match (user, password) {
            (None, None) => Ok(None),
            (Some(user), Some(password)) => Ok(Some(Self { user, password })),
            _ => Err("SIE_NATS_USER and SIE_NATS_PASSWORD must be set together".to_string()),
        }
    }

    /// Read `SIE_NATS_USER` / `SIE_NATS_PASSWORD`. They are environment-only
    /// (no CLI flags), so the password never appears in a process listing or
    /// in `--help`.
    pub fn from_env() -> Result<Option<Self>, String> {
        Self::from_parts(
            std::env::var("SIE_NATS_USER").ok(),
            std::env::var("SIE_NATS_PASSWORD").ok(),
        )
    }
}

impl std::fmt::Debug for NatsCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NatsCredentials")
            .field("user", &self.user)
            .field("password", &"<redacted>")
            .finish()
    }
}

/// `url` with any `user[:password]@` userinfo replaced by `<redacted>@`, for
/// logs. The client ignores userinfo in `SIE_NATS_URL`.
pub fn redact_url_userinfo(url: &str) -> String {
    url.split(',')
        .map(|server| {
            let authority_start = server.find("://").map_or(0, |i| i + 3);
            let authority_end = server[authority_start..]
                .find(['/', '?', '#'])
                .map_or(server.len(), |i| authority_start + i);
            match server[authority_start..authority_end].rfind('@') {
                Some(at) => format!(
                    "{}<redacted>{}",
                    &server[..authority_start],
                    &server[authority_start + at..]
                ),
                None => server.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Clone)]
pub struct WorkerConfig {
    /// NATS server URL (e.g. `nats://localhost:4222`). Required for the
    /// default NATS ingest (`run()`); `None` is valid only for the
    /// local-ingest mode (`run_local()`, P2.10 §4.6) which never
    /// touches NATS.
    pub nats_url: Option<String>,

    /// Credentials for the NATS connection; `None` connects without them.
    pub nats_credentials: Option<NatsCredentials>,

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
    /// `abfss://…` / `oss://…` use cloud stores when built with
    /// `--features cloud-storage`.
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

    /// Optional bearer token for the `sie-config` epoch and export reads,
    /// from `SIE_CONFIG_SERVICE_TOKEN`: a read-scoped sie-config token.
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

impl std::fmt::Debug for WorkerConfig {
    /// Hand-written so the bearer tokens (`gateway_api_key`,
    /// `config_service_token`) print only as present or absent, and the NATS
    /// password and URL userinfo never print.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Destructured without `..` so a new field fails to compile until it is
        // listed here, and so cannot be printed unredacted or silently omitted.
        let WorkerConfig {
            nats_url,
            nats_credentials,
            local_socket_path,
            pool,
            bundle,
            ipc_socket_path,
            ipc_socket_paths,
            ipc_pool_size,
            ipc_request_timeout_s,
            model_ready_timeout_s,
            payload_store_url,
            gateway_url,
            gateway_api_key,
            pool_admission_enabled,
            pool_admission_check_interval_ms,
            pool_admission_pause_ms,
            pool_admission_stale_after_ms,
            probe_port,
            worker_id,
            ping_interval_ms,
            ready_stale_mult,
            machine_profile,
            gpu_count,
            bundle_config_hash,
            config_service_url,
            config_service_token,
            config_poll_interval_ms,
            config_full_export_interval_ms,
            nats_config_trusted_producers,
            health_publish_interval_ms,
        } = self;
        f.debug_struct("WorkerConfig")
            .field("nats_url", &nats_url.as_deref().map(redact_url_userinfo))
            .field("nats_credentials", nats_credentials)
            .field("local_socket_path", local_socket_path)
            .field("pool", pool)
            .field("bundle", bundle)
            .field("ipc_socket_path", ipc_socket_path)
            .field("ipc_socket_paths", ipc_socket_paths)
            .field("ipc_pool_size", ipc_pool_size)
            .field("ipc_request_timeout_s", ipc_request_timeout_s)
            .field("model_ready_timeout_s", model_ready_timeout_s)
            .field("payload_store_url", payload_store_url)
            .field("gateway_url", gateway_url)
            .field(
                "gateway_api_key",
                &gateway_api_key.as_ref().map(|_| "<redacted>"),
            )
            .field("pool_admission_enabled", pool_admission_enabled)
            .field(
                "pool_admission_check_interval_ms",
                pool_admission_check_interval_ms,
            )
            .field("pool_admission_pause_ms", pool_admission_pause_ms)
            .field(
                "pool_admission_stale_after_ms",
                pool_admission_stale_after_ms,
            )
            .field("probe_port", probe_port)
            .field("worker_id", worker_id)
            .field("ping_interval_ms", ping_interval_ms)
            .field("ready_stale_mult", ready_stale_mult)
            .field("machine_profile", machine_profile)
            .field("gpu_count", gpu_count)
            .field("bundle_config_hash", bundle_config_hash)
            .field("config_service_url", config_service_url)
            .field(
                "config_service_token",
                &config_service_token.as_ref().map(|_| "<redacted>"),
            )
            .field("config_poll_interval_ms", config_poll_interval_ms)
            .field(
                "config_full_export_interval_ms",
                config_full_export_interval_ms,
            )
            .field(
                "nats_config_trusted_producers",
                nats_config_trusted_producers,
            )
            .field("health_publish_interval_ms", health_publish_interval_ms)
            .finish()
    }
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
            nats_credentials: None,
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
    fn debug_redacts_bearer_tokens() {
        let mut c = sample();
        c.gateway_api_key = Some("gateway-bearer-value".into());
        c.config_service_token = Some("config-read-value".into());
        let rendered = format!("{c:?}");
        assert!(!rendered.contains("gateway-bearer-value"));
        assert!(!rendered.contains("config-read-value"));
        assert!(rendered.contains("config_service_token: Some(\"<redacted>\")"));
        assert!(rendered.contains("gateway_api_key: Some(\"<redacted>\")"));
        assert!(rendered.contains("worker_id: \"worker-test\""));
    }

    #[test]
    fn nats_credentials_require_user_and_password_together() {
        assert_eq!(NatsCredentials::from_parts(None, None), Ok(None));
        assert_eq!(
            NatsCredentials::from_parts(Some(String::new()), Some(String::new())),
            Ok(None)
        );
        assert_eq!(
            NatsCredentials::from_parts(Some("sie-worker".into()), Some("pw".into())),
            Ok(Some(NatsCredentials {
                user: "sie-worker".into(),
                password: "pw".into(),
            }))
        );
        assert!(NatsCredentials::from_parts(Some("sie-worker".into()), None).is_err());
        assert!(NatsCredentials::from_parts(None, Some("pw".into())).is_err());
    }

    #[test]
    fn worker_config_debug_redacts_the_nats_password() {
        let mut cfg = sample();
        cfg.nats_credentials = Some(NatsCredentials {
            user: "sie-worker".into(),
            password: "nats-password-secret".into(),
        });
        cfg.nats_url = Some("nats://url-user:url-secret@nats:4222".into());
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("nats-password-secret"), "{dbg}");
        assert!(!dbg.contains("url-secret"), "{dbg}");
        assert!(dbg.contains("sie-worker"), "{dbg}");
    }

    #[test]
    fn redact_url_userinfo_hides_credentials_only() {
        assert_eq!(
            redact_url_userinfo("nats://nats-host:4222"),
            "nats://nats-host:4222"
        );
        assert_eq!(
            redact_url_userinfo("nats://user:secret@nats-host:4222"),
            "nats://<redacted>@nats-host:4222"
        );
        assert_eq!(
            redact_url_userinfo("tls://token@a:4222,nats://b:4222/x@y"),
            "tls://<redacted>@a:4222,nats://b:4222/x@y"
        );
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
