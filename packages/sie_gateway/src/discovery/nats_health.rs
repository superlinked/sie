use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use tracing::{info, warn};

use crate::state::worker_registry::WorkerRegistry;
use crate::types::WorkerStatusMessage;

const HEALTH_SUBJECT: &str = "sie.health.>";
const RESUBSCRIBE_INITIAL_DELAY: Duration = Duration::from_secs(1);
const RESUBSCRIBE_MAX_DELAY: Duration = Duration::from_secs(30);

pub struct NatsHealthManager {
    registry: Arc<WorkerRegistry>,
    cancel_tx: tokio::sync::watch::Sender<()>,
    cancel_rx: tokio::sync::watch::Receiver<()>,
}

impl NatsHealthManager {
    pub fn new(registry: Arc<WorkerRegistry>) -> Self {
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(());
        Self {
            registry,
            cancel_tx,
            cancel_rx,
        }
    }

    /// Subscribe to NATS health messages using a shared client (no separate connection).
    pub async fn start(
        &self,
        client: &async_nats::Client,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let sub = subscribe_health(client).await?;
        info!(subject = HEALTH_SUBJECT, "subscribed to NATS health");

        let registry = Arc::clone(&self.registry);
        let client = client.clone();
        let mut cancel_rx = self.cancel_rx.clone();

        tokio::spawn(async move {
            run_subscription_supervised(registry, client, sub, &mut cancel_rx).await;
        });

        Ok(())
    }

    pub async fn start_heartbeat_loop(&self) {
        let registry = Arc::clone(&self.registry);
        let mut cancel_rx = self.cancel_rx.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let sweep = registry.check_heartbeats().await;
                        for url in &sweep.unhealthy {
                            warn!(url = %url, "worker missed heartbeat (NATS)");
                        }
                        for url in &sweep.evicted {
                            info!(url = %url, "evicted stale worker after missed heartbeats (NATS)");
                        }
                    }
                    _ = cancel_rx.changed() => {
                        return;
                    }
                }
            }
        });
    }

    pub async fn stop(&self) {
        info!("stopping NATS health manager");
        let _ = self.cancel_tx.send(());
    }
}

async fn subscribe_health(
    client: &async_nats::Client,
) -> Result<async_nats::Subscriber, Box<dyn std::error::Error + Send + Sync>> {
    Ok(client.subscribe(HEALTH_SUBJECT).await?)
}

#[derive(Debug, PartialEq, Eq)]
enum SubscriptionExit {
    Ended { messages: u64 },
    Cancelled,
}

#[derive(Debug)]
struct ResubscribeBackoff {
    initial: Duration,
    next: Duration,
    max: Duration,
}

impl ResubscribeBackoff {
    fn new(initial: Duration, max: Duration) -> Self {
        Self {
            initial,
            next: initial,
            max,
        }
    }

    fn next_delay(&mut self) -> Duration {
        let delay = self.next;
        self.next = self.next.checked_mul(2).unwrap_or(self.max).min(self.max);
        delay
    }

    fn reset(&mut self) {
        self.next = self.initial;
    }
}

async fn run_subscription_supervised(
    registry: Arc<WorkerRegistry>,
    client: async_nats::Client,
    initial_sub: async_nats::Subscriber,
    cancel_rx: &mut tokio::sync::watch::Receiver<()>,
) {
    let mut backoff = ResubscribeBackoff::new(RESUBSCRIBE_INITIAL_DELAY, RESUBSCRIBE_MAX_DELAY);
    let mut next_sub = Some(initial_sub);

    loop {
        let sub = match next_sub.take() {
            Some(sub) => sub,
            None => match subscribe_health(&client).await {
                Ok(sub) => {
                    info!(subject = HEALTH_SUBJECT, "resubscribed to NATS health");
                    sub
                }
                Err(error) => {
                    let delay = backoff.next_delay();
                    warn!(
                        subject = HEALTH_SUBJECT,
                        error = %error,
                        delay_ms = delay.as_millis() as u64,
                        "failed to resubscribe to NATS health"
                    );
                    if sleep_or_cancel(delay, cancel_rx).await {
                        info!("NATS health subscription cancelled");
                        return;
                    }
                    continue;
                }
            },
        };

        match run_subscription_until_end(Arc::clone(&registry), sub, cancel_rx).await {
            SubscriptionExit::Cancelled => return,
            SubscriptionExit::Ended { messages } => {
                if messages > 0 {
                    backoff.reset();
                }
                let delay = backoff.next_delay();
                warn!(
                    subject = HEALTH_SUBJECT,
                    messages,
                    delay_ms = delay.as_millis() as u64,
                    "NATS health subscription stream ended; resubscribing"
                );
                if sleep_or_cancel(delay, cancel_rx).await {
                    info!("NATS health subscription cancelled");
                    return;
                }
            }
        }
    }
}

async fn sleep_or_cancel(
    delay: Duration,
    cancel_rx: &mut tokio::sync::watch::Receiver<()>,
) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(delay) => false,
        _ = cancel_rx.changed() => true,
    }
}

async fn run_subscription_until_end(
    registry: Arc<WorkerRegistry>,
    mut sub: async_nats::Subscriber,
    cancel_rx: &mut tokio::sync::watch::Receiver<()>,
) -> SubscriptionExit {
    info!("NATS health subscription handler started");
    let mut messages = 0u64;

    loop {
        tokio::select! {
            msg = sub.next() => {
                match msg {
                    Some(msg) => {
                        messages = messages.saturating_add(1);
                        handle_nats_message(&registry, msg).await;
                    }
                    None => {
                        return SubscriptionExit::Ended { messages };
                    }
                }
            }
            _ = cancel_rx.changed() => {
                info!("NATS health subscription cancelled");
                return SubscriptionExit::Cancelled;
            }
        }
    }
}

async fn handle_nats_message(registry: &WorkerRegistry, msg: async_nats::Message) {
    let subject = msg.subject.as_str();

    // Try msgpack first (compact binary format), then fall back to JSON
    let status: WorkerStatusMessage = if let Ok(s) = rmp_serde::from_slice(&msg.payload) {
        s
    } else if let Ok(s) = serde_json::from_slice(&msg.payload) {
        s
    } else {
        warn!(
            subject = %subject,
            "failed to parse NATS health message (tried msgpack and JSON)"
        );
        return;
    };

    handle_status_message(registry, subject, status).await;
}

async fn handle_status_message(
    registry: &WorkerRegistry,
    subject: &str,
    status: WorkerStatusMessage,
) {
    // Extract worker URL from the message or subject
    // The subject format is sie.health.<worker_identifier>
    // The worker URL should be in the message payload or derivable from the worker name
    let worker_url = if let Some(url) = extract_worker_url_from_status(&status, subject) {
        url
    } else {
        warn!(
            subject = %subject,
            "could not determine worker URL from NATS health message"
        );
        return;
    };

    if status.terminated {
        if registry.remove_worker(&worker_url).await {
            info!(url = %worker_url, subject = %subject, "worker removed via NATS tombstone");
        } else {
            info!(url = %worker_url, subject = %subject, "NATS tombstone for unknown worker ignored");
        }
        return;
    }

    let became_healthy = registry.update_worker(&worker_url, status).await;
    if became_healthy {
        info!(url = %worker_url, subject = %subject, "worker became healthy via NATS");
    }
}

fn extract_worker_url_from_status(status: &WorkerStatusMessage, subject: &str) -> Option<String> {
    // First, try the name field as a URL if it looks like one
    if status.name.starts_with("http://") || status.name.starts_with("https://") {
        return Some(status.name.clone());
    }

    // Extract the worker identifier from the NATS subject
    // Format: sie.health.<worker_id> where worker_id might be an IP:port or hostname
    let parts: Vec<&str> = subject.splitn(3, '.').collect();
    if parts.len() >= 3 {
        let worker_id = parts[2];
        // If it looks like a hostname:port or IP:port, construct URL
        if worker_id.contains(':') || worker_id.contains('.') {
            let url = format!("http://{worker_id}");
            return Some(url);
        }
        // Use the worker name from the status message with the subject as fallback
        if !status.name.is_empty() {
            return Some(format!("http://{}", status.name));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn status(ready: bool) -> WorkerStatusMessage {
        WorkerStatusMessage {
            supports_execution_authority_v1: false,
            name: "worker-1".into(),
            ready,
            gpu_count: 1,
            total_gpu_slots: None,
            ready_gpu_slots: None,
            machine_profile: "rtx6000".into(),
            pool_name: "default".into(),
            bundle: "sglang".into(),
            bundle_config_hash: "hash".into(),
            loaded_models: vec![],
            models: vec![],
            gpus: vec![],
            queue_depth: None,
            pending_cost: None,
            inflight_batches: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
            saturated: false,
            terminated: false,
            unsupported_models: Vec::new(),
        }
    }

    #[tokio::test]
    async fn tombstone_removes_registered_worker() {
        let registry = WorkerRegistry::new(Duration::from_secs(15), None);
        handle_status_message(&registry, "sie.health.worker-1", status(true)).await;
        assert_eq!(registry.workers().await.len(), 1);

        let mut tombstone = status(false);
        tombstone.terminated = true;
        handle_status_message(&registry, "sie.health.worker-1", tombstone).await;

        assert!(registry.workers().await.is_empty());
        assert!(registry.healthy_workers().await.is_empty());
    }

    #[test]
    fn resubscribe_backoff_doubles_to_cap() {
        let mut backoff = ResubscribeBackoff::new(Duration::from_secs(1), Duration::from_secs(5));
        assert_eq!(backoff.next_delay(), Duration::from_secs(1));
        assert_eq!(backoff.next_delay(), Duration::from_secs(2));
        assert_eq!(backoff.next_delay(), Duration::from_secs(4));
        assert_eq!(backoff.next_delay(), Duration::from_secs(5));
        assert_eq!(backoff.next_delay(), Duration::from_secs(5));
    }

    #[test]
    fn resubscribe_backoff_reset_returns_to_initial_delay() {
        let mut backoff = ResubscribeBackoff::new(Duration::from_secs(1), Duration::from_secs(5));
        assert_eq!(backoff.next_delay(), Duration::from_secs(1));
        assert_eq!(backoff.next_delay(), Duration::from_secs(2));
        backoff.reset();
        assert_eq!(backoff.next_delay(), Duration::from_secs(1));
    }
}
