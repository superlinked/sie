mod config;
mod discovery;
mod error;
mod handlers;
mod metrics;
mod middleware;
mod nats;
mod queue;
mod server;
mod state;
mod types;

use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use tokio::net::TcpListener;
use tokio::signal;
use tracing::info;

use config::Config;
use discovery::nats_health::NatsHealthManager;
use discovery::static_discovery::StaticDiscovery;
use discovery::ws_health::WsHealthManager;
use nats::manager::NatsManager;
use queue::payload_store::create_payload_store;
use server::AppState;
use state::model_registry::ModelRegistry;
use state::pool_manager::PoolManager;
use state::worker_registry::WorkerRegistry;
use types::PoolState;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "sie-gateway",
    about = "SIE Gateway - Stateless request gateway for elastic cloud deployments"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Serve {
        #[arg(short = 'p', long, env = "SIE_GATEWAY_PORT", default_value_t = 8080)]
        port: u16,

        #[arg(long, env = "SIE_GATEWAY_HOST", default_value = "0.0.0.0")]
        host: String,

        #[arg(
            short = 'w',
            long = "worker",
            env = "SIE_GATEWAY_WORKERS",
            num_args = 1
        )]
        workers: Vec<String>,

        #[arg(long, env = "SIE_GATEWAY_KUBERNETES")]
        kubernetes: bool,

        #[arg(long, env = "SIE_GATEWAY_K8S_NAMESPACE", default_value = "default")]
        k8s_namespace: String,

        #[arg(long, env = "SIE_GATEWAY_K8S_SERVICE", default_value = "sie-worker")]
        k8s_service: String,

        #[arg(long, env = "SIE_GATEWAY_K8S_PORT", default_value_t = 8080)]
        k8s_port: u16,

        #[arg(short = 'l', long, env = "SIE_LOG_LEVEL", default_value = "info")]
        log_level: String,

        #[arg(long, env = "SIE_LOG_JSON")]
        json_logs: bool,

        #[arg(long, env = "SIE_GATEWAY_HEALTH_MODE", default_value = "ws")]
        health_mode: String,

        #[arg(long)]
        bundles_dir: Option<String>,

        #[arg(long)]
        models_dir: Option<String>,
    },
    Version,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Version => {
            println!("sie-gateway {} (rust)", VERSION);
        }
        Commands::Serve {
            port,
            host,
            workers,
            kubernetes,
            k8s_namespace,
            k8s_service,
            k8s_port,
            log_level,
            json_logs,
            health_mode,
            bundles_dir,
            models_dir,
        } => {
            // Load config from env vars
            let mut cfg = Config::load();

            // Override from CLI flags
            cfg.port = port;
            cfg.host = host;
            if !workers.is_empty() {
                cfg.worker_urls = workers;
            }
            cfg.use_kubernetes = cfg.use_kubernetes || kubernetes;
            cfg.k8s_namespace = k8s_namespace;
            cfg.k8s_service = k8s_service;
            cfg.k8s_port = k8s_port;
            cfg.log_level = log_level;
            cfg.json_logs = json_logs;
            cfg.health_mode = health_mode;
            if let Some(dir) = bundles_dir {
                cfg.bundles_dir = dir;
            }
            if let Some(dir) = models_dir {
                cfg.models_dir = dir;
            }

            setup_logging(&cfg.log_level, cfg.json_logs);

            if let Err(e) = run_server(cfg).await {
                tracing::error!(error = %e, "server error");
                std::process::exit(1);
            }
        }
    }
}

fn setup_logging(level: &str, json: bool) {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level_str = match level.to_lowercase().as_str() {
            "debug" => "debug",
            "warn" | "warning" => "warn",
            "error" => "error",
            _ => "info",
        };
        EnvFilter::new(level_str)
    });

    if json {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

async fn run_server(cfg: Config) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let config = Arc::new(cfg);

    // Pre-instantiate request/demand metric families with empty-label
    // sentinels so `/metrics` exposes them from the very first scrape,
    // before any proxied traffic has landed. See `init_metric_families`
    // for why dashboards on a freshly booted gateway need this.
    metrics::init_metric_families();

    // Log auth and NATS producer-trust configuration findings; does
    // not fail startup.
    for (level, msg) in config.audit_auth() {
        match level {
            crate::config::AuditLevel::Warn => tracing::warn!(audit = "auth", "{}", msg),
            crate::config::AuditLevel::Error => tracing::error!(audit = "auth", "{}", msg),
        }
    }
    for (level, msg) in config.audit_nats_producer_trust() {
        match level {
            crate::config::AuditLevel::Warn => {
                tracing::warn!(audit = "nats_config", "{}", msg)
            }
            crate::config::AuditLevel::Error => {
                tracing::error!(audit = "nats_config", "{}", msg)
            }
        }
    }

    // Gateway identity (UUID) — created early because K8s pool backend needs it
    let router_id = uuid::Uuid::new_v4().to_string();

    // Enable pools when kubernetes is active (matches Python behavior)
    let pools_enabled = config.enable_pools || config.use_kubernetes;

    // Set up pool manager (created early so on_worker_healthy callback can capture it)
    let mut pm = PoolManager::new(config.configured_gpus.clone());
    let mut k8s_pool_backend: Option<Arc<state::k8s_pool_backend::K8sPoolBackend>> = None;
    if config.use_kubernetes && pools_enabled {
        match state::k8s_pool_backend::K8sPoolBackend::new(&config.k8s_namespace, &router_id).await
        {
            Ok(backend) => {
                let backend = Arc::new(backend);
                pm = pm.with_k8s_backend(Arc::clone(&backend));
                k8s_pool_backend = Some(backend);
                info!("K8s pool backend enabled");
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to init K8s pool backend (continuing without)");
            }
        }
    }
    let pool_manager = Arc::new(pm);

    // Notify for immediate pool reassignment when a worker becomes healthy
    let worker_healthy_notify = Arc::new(tokio::sync::Notify::new());

    let notify_for_cb = Arc::clone(&worker_healthy_notify);
    let on_worker_healthy: Option<state::worker_registry::OnWorkerHealthy> = if pools_enabled {
        Some(Arc::new(move |_w: &types::WorkerState| {
            notify_for_cb.notify_one();
        }))
    } else {
        None
    };

    let registry = Arc::new(WorkerRegistry::new(
        Duration::from_secs(15),
        on_worker_healthy,
    ));

    // Set up discovery
    let discovery = StaticDiscovery::new(config.worker_urls.clone());

    // Set up model registry (before NATS — NatsManager depends on it)
    let model_registry = Arc::new(ModelRegistry::new(
        &config.bundles_dir,
        &config.models_dir,
        true,
    ));

    // Monotonic epoch counter shared between bootstrap, NATS delta handler,
    // epoch poller, and the /v1/configs/models/{id}/status endpoint. Starts
    // at 0 and advances on every successful bootstrap or delta.
    let config_epoch = state::config_epoch::ConfigEpoch::new();
    let bundles_hash = state::bundles_hash::BundlesHash::new();

    // Config persistence lives in sie-config now; the gateway is pure
    // consumer. It gets its authoritative snapshot via the background
    // bootstrap task and then tracks live changes through NATS deltas.
    let nats_manager = Arc::new(NatsManager::new_with_trusted_producers(
        router_id.clone(),
        config.nats_url.clone(),
        Arc::clone(&model_registry),
        config_epoch.clone(),
        config.nats_config_trusted_producers.clone(),
    ));
    if !config.nats_config_trusted_producers.is_empty() {
        tracing::info!(
            audit = "nats_config",
            trusted_producers = ?config.nats_config_trusted_producers,
            "NATS config-delta producer validation enabled",
        );
    }

    // Connect to NATS if URL is configured.
    //
    // `async_nats::Subscriber` survives reconnects transparently, so
    // `start_subscription` is a one-shot and we don't need to wire it
    // to `reconnect_notify` the way health and inbox do — those two
    // actually need to rebuild JetStream/request-reply state after a
    // server restart, but Core pub/sub subscriptions auto-resume.
    // Staleness caused by messages published during a disconnect is
    // covered by `state::config_poller`'s epoch drift detection.
    if !config.nats_url.is_empty() {
        if let Err(e) = nats_manager.connect().await {
            tracing::warn!(error = %e, "failed to connect to NATS (continuing without)");
        } else {
            nats_manager.start_subscription().await;
        }
    }

    // Set up health manager based on mode
    let mut ws_manager: Option<Arc<WsHealthManager>> = None;
    let mut nats_health_manager: Option<Arc<NatsHealthManager>> = None;
    let mut use_ws = true;

    if config.health_mode == "nats" && !config.nats_url.is_empty() {
        if let Some(client) = nats_manager.get_client().await {
            info!("using NATS health mode (shared connection)");
            let nats_mgr = Arc::new(NatsHealthManager::new(Arc::clone(&registry)));
            match nats_mgr.start(&client).await {
                Ok(()) => {
                    nats_mgr.start_heartbeat_loop().await;

                    // Reconnect listener: re-subscribe health on NATS reconnect
                    let nats_mgr_for_reconnect = Arc::clone(&nats_mgr);
                    let reconnect = nats_manager.reconnect_notify();
                    let nats_mgr_client_source = Arc::clone(&nats_manager);
                    tokio::spawn(async move {
                        loop {
                            reconnect.notified().await;
                            info!("NATS reconnected — re-subscribing health");
                            if let Some(client) = nats_mgr_client_source.get_client().await {
                                if let Err(e) = nats_mgr_for_reconnect.start(&client).await {
                                    tracing::warn!(error = %e, "failed to re-subscribe health on reconnect");
                                }
                            }
                        }
                    });

                    nats_health_manager = Some(nats_mgr);
                    use_ws = false;
                    info!("NATS health manager started");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to start NATS health manager, falling back to WS");
                }
            }
        } else {
            tracing::warn!(
                "NATS health mode requested but client not available, falling back to WS"
            );
        }
    }

    if use_ws {
        let ws_mgr = Arc::new(WsHealthManager::new(Arc::clone(&registry)));
        ws_mgr.start(discovery.get_worker_urls()).await;
        ws_mgr.start_heartbeat_loop().await;
        ws_manager = Some(ws_mgr);
    }

    // Set up file watcher for hot reload (if enabled)
    let _config_watcher = if config.hot_reload {
        match state::config_watcher::ConfigWatcher::start(
            Arc::clone(&model_registry),
            Duration::from_secs(1),
            config.watch_polling,
        ) {
            Ok(w) => {
                info!("config file watcher started");
                Some(w)
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to start config watcher (continuing without hot reload)");
                None
            }
        }
    } else {
        None
    };

    if pools_enabled {
        pool_manager.create_default_pool().await;
        // Restore pools from K8s backend on startup
        match pool_manager.restore_from_k8s().await {
            Ok(count) if count > 0 => {
                info!(count = count, "restored pools from K8s backend");
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "failed to restore pools from K8s");
            }
        }
    }

    // Start K8s pool watcher for multi-gateway coordination
    if config.multi_router && config.use_kubernetes && pools_enabled {
        if let Some(ref backend) = k8s_pool_backend {
            let watcher = Arc::new(state::k8s_pool_watcher::K8sPoolWatcher::new(
                backend.client().clone(),
                backend.namespace(),
                Arc::clone(&pool_manager),
            ));
            watcher.start().await;
            info!("K8s pool watcher started (multi-gateway coordination)");
        } else {
            tracing::warn!(
                "multi_router enabled but K8s pool backend not available, skipping pool watcher"
            );
        }
    }

    // Set up K8s discovery if enabled (requires WS health manager)
    if config.use_kubernetes {
        if let Some(ref ws_mgr) = ws_manager {
            match discovery::k8s_discovery::K8sDiscovery::new(
                &config.k8s_namespace,
                &config.k8s_service,
                config.k8s_port,
                Arc::clone(ws_mgr),
            )
            .await
            {
                Ok(k8s_disc) => {
                    let k8s_disc = Arc::new(k8s_disc);
                    k8s_disc.start().await;
                    info!("kubernetes discovery started");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to start k8s discovery");
                }
            }
        } else {
            info!("k8s discovery skipped (NATS health mode active)");
        }
    }

    // Static discovery periodic re-check (30s)
    if let Some(ref ws_mgr) = ws_manager {
        let static_urls: Vec<String> = discovery.get_worker_urls().to_vec();
        if !static_urls.is_empty() {
            info!(
                count = static_urls.len(),
                "static discovery re-check enabled (30s)"
            );
            let ws_mgr_clone = Arc::clone(ws_mgr);
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                interval.tick().await; // skip immediate first tick
                loop {
                    interval.tick().await;
                    for url in &static_urls {
                        ws_mgr_clone.add_worker(url.clone()).await;
                    }
                }
            });
        }
    }

    // Bootstrap + catch-up loop. The gateway does NOT block startup on
    // sie-config availability: it serves filesystem-seed traffic immediately
    // while a background task retries the export fetch with exponential
    // backoff. Once that first fetch succeeds, the epoch poller keeps the
    // local registry in sync by periodically checking sie-config's latest
    // epoch and triggering a re-export on drift (closes the NATS Core
    // pub/sub delta-loss gap).
    state::config_bootstrap::spawn_bootstrap_retry(
        config.config_service_url.as_deref(),
        config.config_service_token.as_deref(),
        Arc::clone(&model_registry),
        config_epoch.clone(),
        bundles_hash.clone(),
    );
    state::config_poller::spawn(
        config.config_service_url.as_deref(),
        config.config_service_token.as_deref(),
        Arc::clone(&model_registry),
        config_epoch.clone(),
        bundles_hash.clone(),
        state::config_poller::DEFAULT_POLL_INTERVAL,
    );

    // Set up queue mode (queue-only runtime) if NATS is available
    let work_publisher: Option<Arc<queue::publisher::WorkPublisher>> =
        if !config.nats_url.is_empty() {
            match nats_manager.get_client().await {
                Some(client) => {
                    let jetstream = async_nats::jetstream::new(client.clone());
                    let payload_store = create_payload_store(&config.payload_store_url).await;
                    let publisher = Arc::new(queue::publisher::WorkPublisher::new(
                        jetstream,
                        nats_manager.router_id().to_string(),
                        payload_store,
                        Duration::from_secs_f64(config.request_timeout),
                        config.max_stream_pending,
                    ));

                    if let Err(e) = publisher.start_inbox_subscription(&client).await {
                        tracing::warn!(error = %e, "failed to start inbox subscription");
                    }

                    let dlq_jetstream = async_nats::jetstream::new(client.clone());
                    if let Err(e) =
                        queue::dlq::DlqListener::start(dlq_jetstream, client.clone()).await
                    {
                        tracing::warn!(error = %e, "failed to start DLQ listener");
                    }

                    publisher.start_backpressure_monitor();

                    info!("queue mode enabled (JetStream)");
                    Some(publisher)
                }
                None => {
                    tracing::warn!("queue-only gateway started without an active NATS client");
                    None
                }
            }
        } else {
            None
        };

    // ── Background tasks ──────────────────────────────────────────

    // Pool lease expiration (every 60s)
    {
        let pm = Arc::clone(&pool_manager);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            interval.tick().await; // skip immediate first tick
            loop {
                interval.tick().await;
                let expired = pm.check_expired_leases().await;
                if !expired.is_empty() {
                    info!(count = expired.len(), pools = ?expired, "expired pool leases cleaned up");
                }
            }
        });
    }

    // Pool worker reassignment (triggered by on_worker_healthy or every 5s)
    {
        let pm = Arc::clone(&pool_manager);
        let reg = Arc::clone(&registry);
        let notify = Arc::clone(&worker_healthy_notify);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            interval.tick().await; // skip immediate first tick
            loop {
                tokio::select! {
                    _ = interval.tick() => {}
                    _ = notify.notified() => {}
                }
                let pools = pm.list_pools().await;
                let has_pending = pools.iter().any(|p| p.status.state == PoolState::Pending);
                if !has_pending {
                    continue;
                }

                let workers = reg.healthy_workers().await;
                let worker_tuples: Vec<(String, String, String, String)> = workers
                    .iter()
                    .map(|w| {
                        (
                            w.name.clone(),
                            w.url.clone(),
                            w.machine_profile.clone(),
                            w.bundle.clone(),
                        )
                    })
                    .collect();

                for pool in &pools {
                    if pool.status.state == PoolState::Pending {
                        pm.assign_workers(&pool.spec.name, &worker_tuples).await;
                    }
                }
            }
        });
    }

    // Queue result collector cleanup (every 10s, if queue mode)
    if let Some(ref publisher) = work_publisher {
        let pub_clone = Arc::clone(publisher);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                pub_clone.cleanup_expired().await;
            }
        });
    }

    // NATS reconnect inbox re-subscribe
    if let Some(ref publisher) = work_publisher {
        let pub_clone = Arc::clone(publisher);
        let nats_mgr_clone = Arc::clone(&nats_manager);
        let reconnect = nats_manager.reconnect_notify();
        tokio::spawn(async move {
            loop {
                reconnect.notified().await;
                info!("NATS reconnected — re-subscribing inbox and clearing caches");
                pub_clone.clear_caches();
                if let Some(client) = nats_mgr_clone.get_client().await {
                    if let Err(e) = pub_clone.start_inbox_subscription(&client).await {
                        tracing::warn!(error = %e, "failed to re-subscribe inbox on reconnect");
                    }
                }
            }
        });
    }

    // Keep a handle for graceful shutdown drain
    let shutdown_publisher = work_publisher.clone();

    let demand_tracker = Arc::new(state::demand_tracker::DemandTracker::new());

    let state = Arc::new(AppState {
        registry: Arc::clone(&registry),
        config: Arc::clone(&config),
        model_registry: Arc::clone(&model_registry),
        pool_manager: Arc::clone(&pool_manager),
        work_publisher,
        demand_tracker,
        config_epoch: config_epoch.clone(),
    });

    let app = server::create_router(state, Arc::clone(&config));

    info!(
        addr = %addr,
        version = VERSION,
        workers = discovery.get_worker_urls().len(),
        "starting SIE Gateway"
    );

    let listener = TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("server stopped");

    // Drain pending queue results (wait up to 5s for in-flight responses)
    if let Some(ref publisher) = shutdown_publisher {
        info!("draining pending queue results...");
        publisher.drain_pending(Duration::from_secs(5)).await;
    }

    // Stop health managers
    if let Some(ws_mgr) = ws_manager {
        ws_mgr.stop().await;
    }
    if let Some(nats_mgr) = nats_health_manager {
        nats_mgr.stop().await;
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("shutdown signal received (SIGINT)");
        }
        _ = terminate => {
            info!("shutdown signal received (SIGTERM)");
        }
    }
}
