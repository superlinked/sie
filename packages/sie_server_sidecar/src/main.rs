//! sie-server-sidecar — sidecar driving the NATS queue for a sie-server pod.
//!
//! See `docs/architecture-guide.md` for the design. This binary is the
//! entrypoint; the library modules live in `lib.rs`.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use tracing::{error, info};

use sie_server_sidecar::config::WorkerConfig;
use sie_server_sidecar::config_subscriber::trusted_producers_from_env;
use sie_server_sidecar::dispatcher::default_max_concurrent_batches;
use sie_server_sidecar::{run, run_local};

#[derive(Parser, Debug)]
#[command(author, version, about = "SIE server sidecar", long_about = None)]
struct Cli {
    /// Work-ingest mode: `nats` (default; JetStream pull consumer) or
    /// `local` (UDS `publish_work` listener for the broker-less sidecar
    /// lane).
    #[arg(long, env = "SIE_SIDECAR_INGEST", default_value = "nats")]
    ingest: String,

    /// UDS path for the local-ingest listener. Required (and only read)
    /// when `--ingest local`.
    #[arg(long, env = "SIE_SIDECAR_LOCAL_SOCKET")]
    local_socket: Option<String>,

    /// Required for `--ingest nats`; unused in local mode.
    #[arg(long, env = "SIE_NATS_URL")]
    nats_url: Option<String>,

    #[arg(long, env = "SIE_POOL")]
    pool: String,

    #[arg(long, env = "SIE_BUNDLE")]
    bundle: String,

    #[arg(long, env = "SIE_IPC_SOCKET_PATH", default_value = "/tmp/sie-ipc.sock")]
    ipc_socket_path: String,

    /// Comma-separated IPC socket paths for per-GPU adapter worker children.
    /// Defaults to `SIE_IPC_SOCKET_PATH` when unset.
    #[arg(long, env = "SIE_IPC_SOCKET_PATHS")]
    ipc_socket_paths: Option<String>,

    /// Number of concurrent IPC connections to each adapter process.
    /// Defaults to `SIE_MAX_CONCURRENT_BATCHES` when unset (falling
    /// back to the dispatcher's own default of 4), so the IPC pool
    /// never becomes the binding constraint on the dispatcher's
    /// concurrency cap.
    #[arg(long, env = "SIE_IPC_POOL_SIZE")]
    ipc_pool_size: Option<usize>,

    /// Timeout, in seconds, for ordinary sidecar → adapter IPC calls.
    #[arg(long, env = "SIE_IPC_REQUEST_TIMEOUT_S", default_value = "60")]
    ipc_request_timeout_s: u64,

    /// Timeout, in seconds, for EnsureModelReady cold-load calls. Large SGLang
    /// models can spend minutes here while the subprocess loads and reaches
    /// health.
    #[arg(long, env = "SIE_MODEL_READY_TIMEOUT_S")]
    model_ready_timeout_s: Option<u64>,

    /// Adapter worker liveness budget in seconds. When set by Helm, the sidecar
    /// rejects an EnsureModelReady timeout kubelet would kill before completion.
    #[arg(long, env = "SIE_WORKER_LIVENESS_BUDGET_S")]
    worker_liveness_budget_s: Option<u64>,

    #[arg(long, env = "SIE_PAYLOAD_STORE_URL")]
    payload_store_url: Option<String>,

    /// Gateway URL used by the worker-side pool admission gate.
    #[arg(long, env = "SIE_GATEWAY_URL")]
    gateway_url: Option<String>,

    /// Bearer token for gateway pool-status reads.
    #[arg(long, env = "SIE_GATEWAY_API_KEY")]
    gateway_api_key: Option<String>,

    /// Enable/disable the worker-side pool admission gate.
    #[arg(long, env = "SIE_POOL_ADMISSION_ENABLED")]
    pool_admission_enabled: Option<String>,

    /// Pool admission check cadence, in seconds.
    #[arg(
        long,
        env = "SIE_POOL_ADMISSION_CHECK_INTERVAL_S",
        default_value = "5.0"
    )]
    pool_admission_check_interval_s: f64,

    /// Sleep duration while the worker is not admitted, in seconds.
    #[arg(long, env = "SIE_POOL_ADMISSION_PAUSE_S", default_value = "1.0")]
    pool_admission_pause_s: f64,

    /// How long to reuse the last successful admission decision after a
    /// transient gateway/status error, in seconds.
    #[arg(long, env = "SIE_POOL_ADMISSION_STALE_AFTER_S", default_value = "30.0")]
    pool_admission_stale_after_s: f64,

    #[arg(long, env = "SIE_WORKER_PROBE_PORT", default_value = "9095")]
    probe_port: u16,

    #[arg(long, env = "SIE_WORKER_ID")]
    worker_id: Option<String>,

    #[arg(long, env = "SIE_WORKER_PING_INTERVAL_MS", default_value = "2000")]
    ping_interval_ms: u64,

    /// Multiplier applied to `--ping-interval-ms` to derive the
    /// staleness threshold for `/readyz`. Default `3`, matching the
    /// sidecar readiness contract. `0` is treated
    /// as the default (see `WorkerConfig::ready_stale_mult`).
    #[arg(long, env = "SIE_WORKER_READYZ_STALE_MULT", default_value = "3")]
    ready_stale_mult: u32,

    /// Machine-profile label echoed in NATS health heartbeats and
    /// embedded in this worker's queue subject lane. Requests carrying
    /// `X-SIE-MACHINE-PROFILE` constrain route resolution to this value.
    /// Required. Helm sets it explicitly from the worker pool's
    /// machineProfile.
    #[arg(long, env = "SIE_MACHINE_PROFILE")]
    machine_profile: String,

    /// GPU count surfaced in heartbeats. Informational; the
    /// gateway coerces 0 to 1. Defaults to 1.
    #[arg(long, env = "SIE_GPU_COUNT", default_value = "1")]
    gpu_count: i32,

    /// Optional bundle-config hash echoed in heartbeats. Empty
    /// when the operator hasn't pinned a value.
    #[arg(long, env = "SIE_BUNDLE_CONFIG_HASH", default_value = "")]
    bundle_config_hash: String,

    /// Optional sie-config base URL. When set, the sidecar polls
    /// /v1/configs/epoch and reconciles missed config deltas from
    /// /v1/configs/export.
    #[arg(long, env = "SIE_CONFIG_SERVICE_URL")]
    config_service_url: Option<String>,

    /// Bearer token for sie-config export reads. Defaults from the shared
    /// SIE_ADMIN_TOKEN secret in Helm when config auth is enabled.
    #[arg(long, env = "SIE_ADMIN_TOKEN")]
    config_service_token: Option<String>,

    /// Worker-side config epoch poll interval in milliseconds.
    #[arg(
        long,
        env = "SIE_WORKER_CONFIG_POLL_INTERVAL_MS",
        default_value = "30000"
    )]
    config_poll_interval_ms: u64,

    /// Slow full-export reconcile interval in milliseconds. Set to 0 to
    /// disable after the startup export. Kept separate from the epoch poll
    /// so no-config-store deployments can still recover missed live deltas.
    #[arg(
        long,
        env = "SIE_WORKER_CONFIG_FULL_EXPORT_INTERVAL_MS",
        default_value = "300000"
    )]
    config_full_export_interval_ms: u64,

    /// Heartbeat interval (ms) for the NATS health publisher.
    /// Default 5_000 ms; must stay << gateway
    /// `heartbeat_timeout` (30 s) so the staleness check has
    /// margin.
    #[arg(long, env = "SIE_HEALTH_PUBLISH_INTERVAL_MS", default_value = "5000")]
    health_publish_interval_ms: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Release images enable cloud-storage: object_store selects aws-lc-rs
    // while NATS selects ring. Pick one provider before TLS clients start.
    let _ = rustls::crypto::ring::default_provider().install_default();

    init_tracing();

    let cli = Cli::parse();
    let ingest = validate_ingest_mode(
        &cli.ingest,
        cli.nats_url.as_deref(),
        cli.local_socket.as_deref(),
    )?;
    let pool = validate_lane_segment("SIE_POOL", cli.pool)?;
    let bundle = validate_lane_segment("SIE_BUNDLE", cli.bundle)?;
    let machine_profile = validate_lane_segment("SIE_MACHINE_PROFILE", cli.machine_profile)?;
    let model_ready_timeout_s = cli.model_ready_timeout_s.unwrap_or(900).max(1);
    validate_model_ready_liveness_budget(model_ready_timeout_s, cli.worker_liveness_budget_s)?;
    let ipc_socket_path = PathBuf::from(cli.ipc_socket_path);
    let ipc_socket_paths = parse_ipc_socket_paths(cli.ipc_socket_paths.as_deref())
        .unwrap_or_else(|| vec![ipc_socket_path.clone()]);
    validate_unique_ipc_socket_paths(&ipc_socket_paths)?;
    let config = WorkerConfig {
        nats_url: cli.nats_url,
        local_socket_path: cli.local_socket.map(Into::into),
        pool,
        bundle,
        ipc_socket_path,
        ipc_socket_paths,
        ipc_pool_size: cli
            .ipc_pool_size
            .filter(|&n| n > 0)
            .unwrap_or_else(default_max_concurrent_batches),
        ipc_request_timeout_s: cli.ipc_request_timeout_s.max(1),
        model_ready_timeout_s,
        payload_store_url: cli.payload_store_url,
        gateway_url: cli.gateway_url.filter(|url| !url.trim().is_empty()),
        gateway_api_key: cli.gateway_api_key.filter(|token| !token.trim().is_empty()),
        pool_admission_enabled: env_bool_value(cli.pool_admission_enabled.as_deref(), true),
        pool_admission_check_interval_ms: seconds_to_millis(
            cli.pool_admission_check_interval_s,
            1_000,
        ),
        pool_admission_pause_ms: seconds_to_millis(cli.pool_admission_pause_s, 100),
        pool_admission_stale_after_ms: seconds_to_millis(cli.pool_admission_stale_after_s, 0),
        probe_port: cli.probe_port,
        worker_id: cli.worker_id.unwrap_or_else(|| {
            std::env::var("HOSTNAME").unwrap_or_else(|_| format!("worker-{}", uuid::Uuid::new_v4()))
        }),
        ping_interval_ms: cli.ping_interval_ms,
        ready_stale_mult: cli.ready_stale_mult,
        machine_profile,
        gpu_count: cli.gpu_count,
        bundle_config_hash: cli.bundle_config_hash,
        config_service_url: cli.config_service_url.filter(|url| !url.trim().is_empty()),
        config_service_token: cli
            .config_service_token
            .filter(|token| !token.trim().is_empty()),
        config_poll_interval_ms: cli.config_poll_interval_ms.max(1_000),
        config_full_export_interval_ms: cli.config_full_export_interval_ms,
        nats_config_trusted_producers: trusted_producers_from_env(),
        health_publish_interval_ms: cli.health_publish_interval_ms,
    };

    info!(
        pool = %config.pool,
        bundle = %config.bundle,
        ipc = %config.ipc_socket_path.display(),
        ipc_sockets = config.ipc_socket_paths.len(),
        ipc_pool_size = config.ipc_pool_size,
        worker_id = %config.worker_id,
        ingest = %ingest,
        "sie-server-sidecar starting"
    );

    let result = match ingest {
        IngestMode::Nats => run(config).await.context("server sidecar run loop"),
        IngestMode::Local => run_local(config)
            .await
            .context("server sidecar local-ingest run loop"),
    };
    if let Err(e) = result {
        error!(error = ?e, "sie-server-sidecar exited with error");
        // Flush any pending OTLP spans before the hard exit.
        sie_server_sidecar::observability::tracing::shutdown_tracing();
        std::process::exit(1);
    }

    // Flush any pending OTLP spans before a clean exit.
    sie_server_sidecar::observability::tracing::shutdown_tracing();
    Ok(())
}

fn init_tracing() {
    sie_server_sidecar::observability::tracing::init_tracing();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IngestMode {
    Nats,
    Local,
}

impl std::fmt::Display for IngestMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Nats => "nats",
            Self::Local => "local",
        })
    }
}

/// Validate the ingest mode + its per-mode required argument. Kept out of
/// clap's value-parser so the error names the env vars operators actually
/// set (`SIE_SIDECAR_INGEST` / `SIE_NATS_URL` / `SIE_SIDECAR_LOCAL_SOCKET`).
fn validate_ingest_mode(
    raw: &str,
    nats_url: Option<&str>,
    local_socket: Option<&str>,
) -> anyhow::Result<IngestMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "nats" | "" => {
            if nats_url.is_none_or(|url| url.trim().is_empty()) {
                anyhow::bail!(
                    "SIE_NATS_URL is required when SIE_SIDECAR_INGEST=nats (the default)"
                );
            }
            Ok(IngestMode::Nats)
        }
        "local" => {
            if local_socket.is_none_or(|path| path.trim().is_empty()) {
                anyhow::bail!("SIE_SIDECAR_LOCAL_SOCKET is required when SIE_SIDECAR_INGEST=local");
            }
            Ok(IngestMode::Local)
        }
        other => anyhow::bail!("SIE_SIDECAR_INGEST must be 'nats' or 'local', got {other:?}"),
    }
}

fn validate_lane_segment(name: &str, raw: String) -> anyhow::Result<String> {
    let value = raw.trim();
    if value.is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    if value
        .chars()
        .any(|c| c.is_whitespace() || matches!(c, '.' | '*' | '>'))
    {
        anyhow::bail!("{name} must not contain '.', '*', '>', or whitespace");
    }
    Ok(value.to_string())
}

fn env_bool_value(raw: Option<&str>, default: bool) -> bool {
    let Some(raw) = raw else {
        return default;
    };
    let trimmed = raw.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return default;
    }
    !matches!(trimmed.as_str(), "0" | "false" | "no" | "off")
}

fn parse_ipc_socket_paths(raw: Option<&str>) -> Option<Vec<PathBuf>> {
    let raw = raw?;
    let paths: Vec<PathBuf> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    if paths.is_empty() {
        None
    } else {
        Some(paths)
    }
}

fn validate_unique_ipc_socket_paths(paths: &[PathBuf]) -> anyhow::Result<()> {
    let mut seen = HashSet::with_capacity(paths.len());
    for path in paths {
        if !seen.insert(path) {
            anyhow::bail!(
                "SIE_IPC_SOCKET_PATHS must not contain duplicate path '{}'",
                path.display()
            );
        }
    }
    Ok(())
}

fn seconds_to_millis(seconds: f64, min_ms: u64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return min_ms;
    }
    ((seconds * 1_000.0).round() as u64).max(min_ms)
}

fn validate_model_ready_liveness_budget(
    model_ready_timeout_s: u64,
    worker_liveness_budget_s: Option<u64>,
) -> anyhow::Result<()> {
    let Some(worker_liveness_budget_s) = worker_liveness_budget_s else {
        return Ok(());
    };
    if worker_liveness_budget_s == 0 {
        anyhow::bail!("SIE_WORKER_LIVENESS_BUDGET_S must be positive when set");
    }
    if model_ready_timeout_s >= worker_liveness_budget_s {
        anyhow::bail!(
            "SIE_MODEL_READY_TIMEOUT_S ({model_ready_timeout_s}s) must be lower than \
             SIE_WORKER_LIVENESS_BUDGET_S ({worker_liveness_budget_s}s) so kubelet liveness \
             does not kill the worker mid-load"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::parse_ipc_socket_paths;
    use super::validate_ingest_mode;
    use super::validate_lane_segment;
    use super::validate_model_ready_liveness_budget;
    use super::validate_unique_ipc_socket_paths;
    use super::IngestMode;

    #[test]
    fn ingest_mode_nats_requires_nats_url() {
        assert_eq!(
            validate_ingest_mode("nats", Some("nats://x:4222"), None).unwrap(),
            IngestMode::Nats
        );
        assert!(validate_ingest_mode("nats", None, None).is_err());
        assert!(validate_ingest_mode("nats", Some("  "), None).is_err());
    }

    #[test]
    fn ingest_mode_local_requires_socket() {
        assert_eq!(
            validate_ingest_mode("local", None, Some("/tmp/s.sock")).unwrap(),
            IngestMode::Local
        );
        assert!(validate_ingest_mode("local", Some("nats://x"), None).is_err());
        assert!(validate_ingest_mode("local", None, Some("")).is_err());
    }

    #[test]
    fn ingest_mode_rejects_unknown_values() {
        assert!(validate_ingest_mode("both", Some("nats://x"), Some("/tmp/s")).is_err());
    }

    #[test]
    fn validate_lane_segment_trims_valid_values() {
        assert_eq!(
            validate_lane_segment("SIE_POOL", " default-pool ".to_string()).unwrap(),
            "default-pool"
        );
    }

    #[test]
    fn validate_lane_segment_rejects_subject_wildcards_and_dot_tokens() {
        for raw in ["", "   ", "foo.bar", "foo*", "foo>", "foo bar"] {
            assert!(validate_lane_segment("SIE_POOL", raw.to_string()).is_err());
        }
    }

    #[test]
    fn validate_model_ready_liveness_budget_accepts_missing_or_larger_budget() {
        validate_model_ready_liveness_budget(900, None).unwrap();
        validate_model_ready_liveness_budget(900, Some(1080)).unwrap();
    }

    #[test]
    fn validate_model_ready_liveness_budget_rejects_impossible_budget() {
        assert!(validate_model_ready_liveness_budget(900, Some(0)).is_err());
        assert!(validate_model_ready_liveness_budget(900, Some(900)).is_err());
        assert!(validate_model_ready_liveness_budget(1200, Some(1080)).is_err());
    }

    #[test]
    fn parse_ipc_socket_paths_ignores_empty_segments() {
        let paths = parse_ipc_socket_paths(Some(" /a.sock, ,/b.sock ")).unwrap();
        assert_eq!(
            paths,
            vec![PathBuf::from("/a.sock"), PathBuf::from("/b.sock")]
        );
        assert!(parse_ipc_socket_paths(Some(" , ")).is_none());
    }

    #[test]
    fn validate_unique_ipc_socket_paths_rejects_duplicates() {
        let paths = parse_ipc_socket_paths(Some("/a.sock,/b.sock,/a.sock")).unwrap();

        assert!(validate_unique_ipc_socket_paths(&paths).is_err());
    }

    #[test]
    fn validate_unique_ipc_socket_paths_accepts_distinct_paths() {
        let paths = parse_ipc_socket_paths(Some("/a.sock,/b.sock")).unwrap();

        validate_unique_ipc_socket_paths(&paths).unwrap();
    }
}
