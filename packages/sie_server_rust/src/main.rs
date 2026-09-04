use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use sie_server_rust::candle_backend::{
    CandleBackend, CandleBackendConfig, CandleOomRecoveryConfig,
};
use sie_server_rust::ipc::{IpcServer, IpcServerConfig};
use sie_server_rust::ipc_types::SetPinnedModelsRequest;
use sie_server_rust::native_backend::NativeBackend;
use sie_server_rust::observability::{init_observability, shutdown_observability};
use tracing::{error, info};

#[derive(Debug, Parser)]
#[command(author, version, name = "sie-server-rust")]
#[command(about = "Rust SIE worker process backed by native inference engines")]
struct Cli {
    #[arg(long, env = "SIE_IPC_SOCKET_PATH", default_value = "/tmp/sie-ipc.sock")]
    ipc_socket_path: PathBuf,

    #[arg(long, env = "SIE_WORKER_ID", default_value = "sie-server-rust")]
    worker_id: String,

    #[arg(long, env = "SIE_BUNDLE", default_value = "candle")]
    bundle: String,

    #[arg(long, env = "SIE_CANDLE_BATCH_BUDGET", default_value = "64")]
    candle_batch_budget: u32,

    #[arg(long, env = "SIE_CANDLE_MAX_CONCURRENT_FORWARDS", default_value = "1")]
    candle_max_concurrent_forwards: usize,

    #[arg(long, env = "SIE_CANDLE_NORMALIZE")]
    candle_normalize: Option<String>,

    #[arg(long, env = "SIE_IDLE_EVICT_S")]
    idle_evict_s: Option<u64>,

    #[arg(long, env = "SIE_PRELOAD_MODELS")]
    preload_models: Option<String>,

    #[arg(long, env = "SIE_PINNED_MODELS")]
    pinned_models: Option<String>,

    #[arg(long, env = "SIE_SERVER_HOST", default_value = "0.0.0.0")]
    host: String,

    #[arg(long, env = "SIE_SERVER_PORT", default_value = "8080")]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_observability();

    if let Err(e) = run().await {
        error!(error = %e, "sie-server-rust failed");
        shutdown_observability();
        std::process::exit(1);
    }

    shutdown_observability();
    Ok(())
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let backend = NativeBackend::Candle(CandleBackend::new(
        CandleBackendConfig::new(
            cli.candle_batch_budget,
            env_bool_value(cli.candle_normalize.as_deref(), true),
            cli.candle_max_concurrent_forwards,
        )
        .with_idle_evict_s(idle_evict_duration(cli.idle_evict_s)?)
        .with_oom_recovery(CandleOomRecoveryConfig::from_env()?),
    ));
    if let Some(raw) = cli.preload_models.as_deref() {
        let models = csv_values(raw);
        if !models.is_empty() {
            let preload_count = backend.set_preload_models(&models);
            info!(preload_count, "configured startup preload models");
        }
    }
    if let Some(raw) = cli.pinned_models.as_deref() {
        let models = csv_values(raw);
        if !models.is_empty() {
            let response = backend.set_pinned_models(&SetPinnedModelsRequest { models });
            info!(
                pinned_count = response.pinned_count,
                "configured startup pinned models"
            );
        }
    }
    backend.start_idle_evictor();

    let shutdown_backend = backend.clone();
    let server = IpcServer::new(
        IpcServerConfig {
            socket_path: cli.ipc_socket_path,
            worker_id: cli.worker_id,
            bundle: cli.bundle,
            http_host: cli.host,
            http_port: cli.port,
        },
        backend,
    );

    info!("starting sie-server-rust");
    let result = server.run().await.context("run IPC server");
    shutdown_backend.stop_idle_evictor();
    result
}

fn csv_values(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
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

fn idle_evict_duration(raw: Option<u64>) -> anyhow::Result<Option<Duration>> {
    let Some(seconds) = raw else {
        return Ok(None);
    };
    if seconds == 0 {
        return Ok(None);
    }
    if !(10..=86_400).contains(&seconds) {
        anyhow::bail!("SIE_IDLE_EVICT_S must be 0 or between 10 and 86400 seconds; got {seconds}");
    }
    Ok(Some(Duration::from_secs(seconds)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_evict_duration_matches_python_bounds() {
        assert_eq!(idle_evict_duration(None).unwrap(), None);
        assert_eq!(idle_evict_duration(Some(0)).unwrap(), None);
        assert_eq!(
            idle_evict_duration(Some(10)).unwrap(),
            Some(Duration::from_secs(10))
        );
        assert!(idle_evict_duration(Some(9)).is_err());
        assert!(idle_evict_duration(Some(86_401)).is_err());
    }
}
