use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::state::worker_registry::WorkerRegistry;
use crate::types::WorkerStatusMessage;

pub struct WsHealthManager {
    registry: Arc<WorkerRegistry>,
    cancels: Mutex<HashMap<String, tokio::sync::watch::Sender<()>>>,
}

impl WsHealthManager {
    pub fn new(registry: Arc<WorkerRegistry>) -> Self {
        Self {
            registry,
            cancels: Mutex::new(HashMap::new()),
        }
    }

    pub async fn start(&self, urls: &[String]) {
        info!(count = urls.len(), "discovered workers");

        for url in urls {
            self.start_connection(url.clone()).await;
        }
    }

    pub async fn start_heartbeat_loop(self: &Arc<Self>) {
        let registry = Arc::clone(&self.registry);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                let unhealthy = registry.check_heartbeats().await;
                for url in &unhealthy {
                    warn!(url = %url, "worker missed heartbeat");
                }
            }
        });
    }

    async fn start_connection(&self, url: String) {
        let mut cancels = self.cancels.lock().await;
        if cancels.contains_key(&url) {
            return;
        }
        let (tx, rx) = tokio::sync::watch::channel(());
        cancels.insert(url.clone(), tx);

        let registry = Arc::clone(&self.registry);
        tokio::spawn(async move {
            connect_to_worker(registry, url, rx).await;
        });
    }

    pub async fn add_worker(&self, url: String) {
        self.start_connection(url).await;
    }

    pub async fn remove_worker(&self, url: &str) {
        let mut cancels = self.cancels.lock().await;
        // Dropping the sender signals the receiver to stop
        cancels.remove(url);
        // Also remove from registry
        self.registry.remove_worker(url).await;
    }

    pub async fn stop(&self) {
        let mut cancels = self.cancels.lock().await;
        // Dropping all senders will signal receivers
        cancels.clear();
    }
}

async fn connect_to_worker(
    registry: Arc<WorkerRegistry>,
    url: String,
    mut cancel: tokio::sync::watch::Receiver<()>,
) {
    let ws_url = url
        .replace("http://", "ws://")
        .replace("https://", "wss://");
    let ws_url = format!("{}/ws/status", ws_url);

    loop {
        // Check cancellation
        if cancel.has_changed().unwrap_or(true) {
            // Check if sender is closed (dropped)
            if cancel.has_changed().is_err() {
                return;
            }
        }

        match run_ws_connection(&registry, &url, &ws_url, &mut cancel).await {
            Ok(()) => return, // Cancelled
            Err(e) => {
                warn!(url = %ws_url, error = %e, "websocket error");
            }
        }

        registry.mark_unhealthy(&url).await;
        info!(url = %ws_url, delay_s = 5, "reconnecting to worker");

        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(5)) => {}
            _ = cancel.changed() => {
                return;
            }
        }
    }
}

async fn run_ws_connection(
    registry: &Arc<WorkerRegistry>,
    http_url: &str,
    ws_url: &str,
    cancel: &mut tokio::sync::watch::Receiver<()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!(url = %ws_url, "connecting to worker");

    let (ws_stream, _) = tokio_tungstenite::connect_async(ws_url).await?;
    info!(url = %ws_url, "connected to worker");

    let (_, mut read) = ws_stream.split();

    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<WorkerStatusMessage>(&text) {
                            Ok(status) => {
                                registry.update_worker(http_url, status).await;
                            }
                            Err(e) => {
                                warn!(url = %http_url, error = %e, "invalid JSON from worker");
                            }
                        }
                    }
                    Some(Ok(Message::Binary(data))) => {
                        // Try msgpack first, then JSON (parity with NATS health handler)
                        let status: Option<WorkerStatusMessage> =
                            rmp_serde::from_slice(&data)
                                .ok()
                                .or_else(|| serde_json::from_slice(&data).ok());
                        match status {
                            Some(s) => {
                                registry.update_worker(http_url, s).await;
                            }
                            None => {
                                warn!(url = %http_url, "failed to parse binary WS message (tried msgpack and JSON)");
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        return Err("connection closed".into());
                    }
                    Some(Err(e)) => {
                        return Err(Box::new(e));
                    }
                    None => {
                        return Err("stream ended".into());
                    }
                    _ => {}
                }
            }
            _ = cancel.changed() => {
                return Ok(());
            }
        }
    }
}
