//! Bundle-scoped config-delta subscriber.
//!
//! `sie-config` publishes validated model config deltas to both
//! `sie.config.models._all` (gateway replicas) and
//! `sie.config.models.<bundle>` (workers). The gateway applies `_all` into its
//! Rust registry; this module consumes the bundle-scoped subject in the
//! worker-sidecar and forwards the validated YAML over IPC so the backend-local
//! model registry changes in the process that serves inference.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use async_nats::Client;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock as AsyncRwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};
use tracing::{debug, info, warn};

use crate::backend::AdapterWorkerPool;
use crate::health_publisher::{SharedBundleConfigHash, SharedLoadedModels};
use crate::ipc_client::IpcError;
use crate::ipc_types::{
    ApplyModelConfigRequest, ReplaceModelConfigsRequest, ReplaceModelConfigsResponse,
};
use crate::observability::metrics::SidecarTelemetry;
#[cfg(test)]
use crate::runtime_state::RuntimeState;
use crate::shutdown::Shutdown;

const SUBJECT_PREFIX: &str = "sie.config.models";
pub const MAX_MODEL_CONFIG_BYTES: usize = 1_048_576;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(250);
const RETRY_MAX_DELAY: Duration = Duration::from_secs(5);
const DEFAULT_MODEL_POOL: &str = "default";

/// Default publisher allowlist. Matches the gateway config subscriber.
pub const DEFAULT_TRUSTED_PRODUCERS: &[&str] = &["sie-config"];

#[derive(Debug)]
pub struct ConfigApplyState {
    epoch: AtomicU64,
    bundle_config_hash: SharedBundleConfigHash,
    loaded_models: SharedLoadedModels,
    /// Config mutation is exclusive while inference takes a shared guard.
    /// This binds each execution to one stable backend registry revision.
    execution_barrier: AsyncRwLock<()>,
}

impl ConfigApplyState {
    pub fn new(initial_bundle_config_hash: String) -> Self {
        Self {
            epoch: AtomicU64::new(0),
            bundle_config_hash: Arc::new(RwLock::new(initial_bundle_config_hash)),
            loaded_models: Arc::new(RwLock::new(Vec::new())),
            execution_barrier: AsyncRwLock::new(()),
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    pub fn bundle_config_hash(&self) -> SharedBundleConfigHash {
        Arc::clone(&self.bundle_config_hash)
    }

    pub fn loaded_models(&self) -> SharedLoadedModels {
        Arc::clone(&self.loaded_models)
    }

    pub fn current_bundle_config_hash(&self) -> String {
        self.bundle_config_hash
            .read()
            .expect("bundle config hash lock poisoned")
            .clone()
    }

    pub fn accepts_bundle_config_hash(&self, expected_hash: &str) -> bool {
        if expected_hash.is_empty() {
            return true;
        }
        self.current_bundle_config_hash() == expected_hash
    }

    pub async fn lock_apply(&self) -> RwLockWriteGuard<'_, ()> {
        self.execution_barrier.write().await
    }

    pub async fn lock_execution(&self) -> RwLockReadGuard<'_, ()> {
        self.execution_barrier.read().await
    }

    pub fn set_epoch_max(&self, epoch: u64) -> bool {
        let mut current = self.epoch();
        while epoch > current {
            match self
                .epoch
                .compare_exchange(current, epoch, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
        false
    }

    pub fn force_epoch(&self, epoch: u64) {
        self.epoch.store(epoch, Ordering::Release);
    }

    pub fn set_bundle_hash(&self, hash: String) {
        let mut guard = self
            .bundle_config_hash
            .write()
            .expect("bundle config hash lock poisoned");
        *guard = hash;
    }

    pub fn set_loaded_models<I>(&self, models: I)
    where
        I: IntoIterator<Item = String>,
    {
        let mut deduped: Vec<String> = models
            .into_iter()
            .map(|model| model.trim().to_string())
            .filter(|model| !model.is_empty())
            .collect();
        deduped.sort();
        deduped.dedup();
        let mut guard = self
            .loaded_models
            .write()
            .expect("loaded models lock poisoned");
        *guard = deduped;
    }

    pub fn record_loaded_model(&self, model_id: &str) {
        let model_id = model_id.trim();
        if model_id.is_empty() {
            return;
        }
        let mut guard = self
            .loaded_models
            .write()
            .expect("loaded models lock poisoned");
        if guard.iter().any(|known| known == model_id) {
            return;
        }
        guard.push(model_id.to_string());
        guard.sort();
    }

    fn mark_applied(&self, epoch: u64, bundle_config_hash: String) {
        if epoch == 0 {
            self.set_bundle_hash(bundle_config_hash);
            return;
        }

        let mut current = self.epoch();
        while epoch > current {
            match self
                .epoch
                .compare_exchange(current, epoch, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    self.set_bundle_hash(bundle_config_hash);
                    return;
                }
                Err(observed) => current = observed,
            }
        }
    }

    pub fn mark_export_reconciled(
        &self,
        epoch: u64,
        bundle_config_hash: Option<String>,
        force_epoch: bool,
    ) -> bool {
        if force_epoch {
            self.force_epoch(epoch);
        } else {
            let current = self.epoch();
            if epoch < current {
                return false;
            }
            self.set_epoch_max(epoch);
        }
        if let Some(hash) = bundle_config_hash {
            self.set_bundle_hash(hash);
        }
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigNotification {
    /// Publisher identity. Python emits this field as `router_id`; keep the
    /// alias for the wire contract shared with the gateway.
    #[serde(alias = "router_id")]
    pub producer_id: String,
    pub bundle_id: String,
    pub epoch: u64,
    pub bundle_config_hash: String,
    #[serde(default)]
    pub bundle_pool_config_hashes: HashMap<String, HashMap<String, String>>,
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub profiles_added: Vec<String>,
    #[serde(default)]
    pub model_config: String,
    #[serde(default)]
    pub affected_bundles: Vec<String>,
    #[serde(default)]
    pub pool: Option<String>,
}

fn env_bool(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub fn trusted_producers_from_env() -> Vec<String> {
    if env_bool("SIE_NATS_CONFIG_TRUST_ANY_PRODUCER") {
        return Vec::new();
    }
    let custom: Vec<String> = std::env::var("SIE_NATS_CONFIG_TRUSTED_PRODUCERS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if custom.is_empty() {
        DEFAULT_TRUSTED_PRODUCERS
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    } else {
        custom
    }
}

#[derive(Debug, Clone)]
pub struct ConfigSubscriberOptions {
    pub trusted_producers: Vec<String>,
    pub pool: String,
}

struct SubscriberRuntime {
    bundle: String,
    pool: String,
    trusted_producers: Vec<String>,
    ipc: Arc<AdapterWorkerPool>,
    state: Arc<ConfigApplyState>,
    telemetry: SidecarTelemetry,
    shutdown: Arc<Shutdown>,
}

pub fn bundle_subject(bundle: &str) -> Result<String, String> {
    let trimmed = bundle.trim();
    if trimmed.is_empty() {
        return Err("bundle is empty".to_string());
    }
    if trimmed.contains('.') || trimmed.contains('*') || trimmed.contains('>') {
        return Err(format!(
            "bundle {trimmed:?} is not a single safe NATS subject token"
        ));
    }
    Ok(format!("{SUBJECT_PREFIX}.{trimmed}"))
}

pub fn is_trusted_producer(producer_id: &str, trusted_producers: &[String]) -> bool {
    if trusted_producers.is_empty() {
        return true;
    }
    trusted_producers.iter().any(|trusted| {
        trusted == producer_id
            || producer_id
                .strip_prefix(trusted.as_str())
                .is_some_and(|rest| rest.starts_with('-'))
    })
}

fn notification_targets_bundle(notification: &ConfigNotification, bundle: &str) -> bool {
    notification.bundle_id == bundle
}

fn normalize_pool_name(raw: &str) -> String {
    let pool = raw.trim().to_ascii_lowercase();
    if pool.is_empty() {
        DEFAULT_MODEL_POOL.to_string()
    } else {
        pool
    }
}

fn notification_targets_pool(notification: &ConfigNotification, pool: &str) -> bool {
    let Some(notification_pool) = notification.pool.as_deref() else {
        // Legacy sie-config payloads did not carry pool metadata; preserve
        // the old "bundle-only" apply behavior during mixed-version rollouts.
        return true;
    };
    normalize_pool_name(notification_pool) == normalize_pool_name(pool)
}

fn notification_bundle_hash_for_pool(notification: &ConfigNotification, pool: &str) -> String {
    let normalized_pool = normalize_pool_name(pool);
    match notification
        .bundle_pool_config_hashes
        .get(&notification.bundle_id)
    {
        Some(pool_hashes) => pool_hashes
            .get(&normalized_pool)
            .cloned()
            .unwrap_or_default(),
        None => notification.bundle_config_hash.clone(),
    }
}

pub(crate) fn verified_applied_bundle_hash(
    control_plane_hash: Option<&str>,
    worker_applied_hash: &str,
) -> Option<String> {
    match control_plane_hash.filter(|hash| !hash.is_empty()) {
        Some(control_plane_hash) if control_plane_hash == worker_applied_hash => {
            Some(control_plane_hash.to_string())
        }
        Some(_) => None,
        None => Some(worker_applied_hash.to_string()),
    }
}

fn notification_is_stale(epoch: u64, current_epoch: u64) -> bool {
    epoch < current_epoch || (epoch == current_epoch && epoch != 0)
}

fn record_delta(
    telemetry: &SidecarTelemetry,
    kind: &str,
    result: &str,
    applied_epoch: Option<u64>,
) {
    debug!(kind, result, "worker-config: notification outcome");
    telemetry.config_apply("notification", kind, result, applied_epoch);
}

fn retryable_ipc_error(e: &IpcError) -> bool {
    match e {
        IpcError::Io(io_err) => matches!(
            io_err.kind(),
            std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::ConnectionRefused
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::UnexpectedEof
                | std::io::ErrorKind::NotFound
                | std::io::ErrorKind::NotConnected
                | std::io::ErrorKind::WouldBlock
        ),
        IpcError::Timeout => true,
        _ => false,
    }
}

fn next_retry_delay(current: Duration) -> Duration {
    std::cmp::min(current.saturating_mul(2), RETRY_MAX_DELAY)
}

async fn wait_retry_or_shutdown(shutdown: &Shutdown, delay: Duration) -> bool {
    tokio::select! {
        biased;
        _ = shutdown.wait() => true,
        _ = sleep(delay) => false,
    }
}

/// Spawn the long-lived subscription task. Failures are logged and leave the
/// worker serving its startup config; inference should not crash just because
/// live config apply is temporarily unavailable.
pub fn spawn(
    nats: Client,
    bundle: String,
    ipc: Arc<AdapterWorkerPool>,
    state: Arc<ConfigApplyState>,
    telemetry: SidecarTelemetry,
    shutdown: Arc<Shutdown>,
    options: ConfigSubscriberOptions,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let subject = match bundle_subject(&bundle) {
            Ok(s) => s,
            Err(e) => {
                warn!(bundle = %bundle, error = %e, "worker-config: subscription disabled");
                return;
            }
        };

        let runtime = SubscriberRuntime {
            bundle,
            pool: options.pool,
            trusted_producers: options.trusted_producers,
            ipc,
            state,
            telemetry,
            shutdown,
        };

        if runtime.trusted_producers.is_empty() {
            warn!(
                subject = %subject,
                "worker-config: producer validation disabled; intended only for local/dev clusters"
            );
        } else {
            info!(
                subject = %subject,
                trusted_producers = ?runtime.trusted_producers,
                "worker-config: subscription started"
            );
        }

        let mut subscribe_delay = RETRY_BASE_DELAY;
        loop {
            if runtime.shutdown.is_fired() {
                debug!(subject = %subject, "worker-config: shutdown observed; stopping subscriber");
                return;
            }

            let mut subscriber = match nats.subscribe(subject.clone()).await {
                Ok(s) => {
                    subscribe_delay = RETRY_BASE_DELAY;
                    s
                }
                Err(e) => {
                    warn!(
                        subject = %subject,
                        error = %e,
                        retry_delay_ms = subscribe_delay.as_millis() as u64,
                        "worker-config: subscribe failed; retrying"
                    );
                    if wait_retry_or_shutdown(&runtime.shutdown, subscribe_delay).await {
                        return;
                    }
                    subscribe_delay = next_retry_delay(subscribe_delay);
                    continue;
                }
            };

            loop {
                let wait = runtime.shutdown.wait();
                tokio::select! {
                    biased;
                    _ = wait => {
                        debug!(subject = %subject, "worker-config: shutdown observed; stopping subscriber");
                        return;
                    }
                    maybe_msg = subscriber.next() => {
                        let Some(msg) = maybe_msg else {
                            warn!(
                                subject = %subject,
                                retry_delay_ms = subscribe_delay.as_millis() as u64,
                                "worker-config: subscription ended; resubscribing"
                            );
                            if wait_retry_or_shutdown(&runtime.shutdown, subscribe_delay).await {
                                return;
                            }
                            subscribe_delay = next_retry_delay(subscribe_delay);
                            break;
                        };
                        let notification: ConfigNotification = match serde_json::from_slice(&msg.payload) {
                            Ok(n) => n,
                            Err(e) => {
                                warn!(subject = %subject, error = %e, "worker-config: invalid notification JSON");
                                record_delta(&runtime.telemetry, "unknown", "parse_error", None);
                                continue;
                            }
                        };
                        apply_notification(&runtime, notification).await;
                    }
                }
            }
        }
    })
}

pub(crate) async fn apply_via_ipc_with_retry(
    ipc: Arc<AdapterWorkerPool>,
    shutdown: Arc<Shutdown>,
    req: ApplyModelConfigRequest,
    model_id: &str,
    epoch: u64,
) -> Result<Option<crate::ipc_types::ApplyModelConfigResponse>, IpcError> {
    let mut retry_delay = RETRY_BASE_DELAY;
    let mut attempt = 0_u64;
    loop {
        match ipc.apply_model_config(req.clone()).await {
            Ok(resp) => return Ok(Some(resp)),
            Err(e) if retryable_ipc_error(&e) && !shutdown.is_fired() => {
                attempt = attempt.saturating_add(1);
                if attempt == 1 || attempt == 5 || attempt == 30 || attempt.is_multiple_of(120) {
                    warn!(
                        model = %model_id,
                        epoch,
                        attempt,
                        retry_delay_ms = retry_delay.as_millis() as u64,
                        error = %e,
                        "worker-config: transient worker config apply failure; retrying"
                    );
                } else {
                    debug!(
                        model = %model_id,
                        epoch,
                        attempt,
                        error = %e,
                        "worker-config: worker config apply still waiting"
                    );
                }
                if wait_retry_or_shutdown(&shutdown, retry_delay).await {
                    return Ok(None);
                }
                retry_delay = next_retry_delay(retry_delay);
            }
            Err(e) => return Err(e),
        }
    }
}

pub(crate) async fn replace_via_ipc_with_retry(
    ipc: Arc<AdapterWorkerPool>,
    shutdown: Arc<Shutdown>,
    req: ReplaceModelConfigsRequest,
    epoch: u64,
) -> Result<Option<ReplaceModelConfigsResponse>, IpcError> {
    let mut retry_delay = RETRY_BASE_DELAY;
    let mut attempt = 0_u64;
    loop {
        match ipc.replace_model_configs(req.clone()).await {
            Ok(resp) => return Ok(Some(resp)),
            Err(e) if retryable_ipc_error(&e) && !shutdown.is_fired() => {
                attempt = attempt.saturating_add(1);
                if attempt == 1 || attempt == 5 || attempt == 30 || attempt.is_multiple_of(120) {
                    warn!(
                        epoch,
                        attempt,
                        retry_delay_ms = retry_delay.as_millis() as u64,
                        error = %e,
                        "worker-config: transient worker config replace failure; retrying"
                    );
                } else {
                    debug!(
                        epoch,
                        attempt,
                        error = %e,
                        "worker-config: worker config replace still waiting"
                    );
                }
                if wait_retry_or_shutdown(&shutdown, retry_delay).await {
                    return Ok(None);
                }
                retry_delay = next_retry_delay(retry_delay);
            }
            Err(e) => return Err(e),
        }
    }
}

async fn apply_notification(runtime: &SubscriberRuntime, notification: ConfigNotification) {
    let bundle = runtime.bundle.as_str();
    let trusted_producers = runtime.trusted_producers.as_slice();
    let state = runtime.state.as_ref();
    let kind = if notification.model_config.trim().is_empty() {
        "epoch_bump"
    } else {
        "model_config"
    };

    if !is_trusted_producer(&notification.producer_id, trusted_producers) {
        warn!(
            producer_id = %notification.producer_id,
            trusted_producers = ?trusted_producers,
            epoch = notification.epoch,
            model = %notification.model_id,
            "worker-config: rejecting notification from untrusted producer"
        );
        record_delta(&runtime.telemetry, kind, "rejected_untrusted", None);
        return;
    }

    if !notification_targets_bundle(&notification, bundle) {
        warn!(
            notification_bundle = %notification.bundle_id,
            worker_bundle = %bundle,
            affected_bundles = ?notification.affected_bundles,
            epoch = notification.epoch,
            "worker-config: rejecting notification for another bundle"
        );
        record_delta(&runtime.telemetry, kind, "rejected_bundle", None);
        return;
    }

    let current_epoch = state.epoch();
    if notification_is_stale(notification.epoch, current_epoch) {
        debug!(
            epoch = notification.epoch,
            current_epoch,
            model = %notification.model_id,
            "worker-config: dropping stale notification"
        );
        record_delta(&runtime.telemetry, kind, "stale", None);
        return;
    }

    if kind == "epoch_bump" {
        let _apply_guard = state.lock_apply().await;
        let current_epoch = state.epoch();
        if notification_is_stale(notification.epoch, current_epoch) {
            debug!(
                epoch = notification.epoch,
                current_epoch,
                model = %notification.model_id,
                "worker-config: dropping stale notification"
            );
            record_delta(&runtime.telemetry, kind, "stale", None);
            return;
        }
        state.set_epoch_max(notification.epoch);
        record_delta(
            &runtime.telemetry,
            kind,
            "applied",
            Some(notification.epoch),
        );
        return;
    }

    if !notification_targets_pool(&notification, runtime.pool.as_str()) {
        debug!(
            notification_bundle = %notification.bundle_id,
            worker_bundle = %bundle,
            notification_pool = ?notification.pool,
            worker_pool = %runtime.pool,
            epoch = notification.epoch,
            model = %notification.model_id,
            "worker-config: skipping notification for another pool"
        );
        record_delta(&runtime.telemetry, kind, "skipped_pool", None);
        return;
    }

    if notification.model_config.len() > MAX_MODEL_CONFIG_BYTES {
        warn!(
            model = %notification.model_id,
            bytes = notification.model_config.len(),
            max_bytes = MAX_MODEL_CONFIG_BYTES,
            "worker-config: rejecting oversized model_config"
        );
        record_delta(&runtime.telemetry, kind, "rejected_oversized", None);
        return;
    }

    let _apply_guard = state.lock_apply().await;
    let current_epoch = state.epoch();
    if notification_is_stale(notification.epoch, current_epoch) {
        debug!(
            epoch = notification.epoch,
            current_epoch,
            model = %notification.model_id,
            "worker-config: dropping stale notification"
        );
        record_delta(&runtime.telemetry, kind, "stale", None);
        return;
    }

    info!(
        from = %notification.producer_id,
        bundle = %notification.bundle_id,
        model = %notification.model_id,
        epoch = notification.epoch,
        profiles_added = ?notification.profiles_added,
        "worker-config: applying model config via IPC"
    );

    let req = ApplyModelConfigRequest {
        bundle_id: notification.bundle_id.clone(),
        model_id: notification.model_id.clone(),
        epoch: notification.epoch,
        bundle_config_hash: notification_bundle_hash_for_pool(&notification, runtime.pool.as_str()),
        profiles_added: notification.profiles_added.clone(),
        model_config: notification.model_config.clone(),
    };

    let resp = match apply_via_ipc_with_retry(
        Arc::clone(&runtime.ipc),
        Arc::clone(&runtime.shutdown),
        req,
        &notification.model_id,
        notification.epoch,
    )
    .await
    {
        Ok(Some(r)) => r,
        Ok(None) => {
            record_delta(&runtime.telemetry, kind, "shutdown", None);
            return;
        }
        Err(e) => {
            warn!(
                model = %notification.model_id,
                epoch = notification.epoch,
                error = %e,
                "worker-config: worker config apply failed; epoch not advanced"
            );
            record_delta(&runtime.telemetry, kind, "apply_error", None);
            return;
        }
    };

    if !resp.applied {
        warn!(
            model = %notification.model_id,
            epoch = notification.epoch,
            "worker-config: worker config reported applied=false; epoch not advanced"
        );
        record_delta(&runtime.telemetry, kind, "apply_rejected", None);
        return;
    }

    let control_plane_hash =
        notification_bundle_hash_for_pool(&notification, runtime.pool.as_str());
    let Some(applied_hash) =
        verified_applied_bundle_hash(Some(control_plane_hash.as_str()), &resp.bundle_config_hash)
    else {
        warn!(
            model = %notification.model_id,
            epoch = notification.epoch,
            control_plane_hash = %control_plane_hash,
            applied_hash = %resp.bundle_config_hash,
            "worker-config: worker config hash differs from control-plane hash; epoch and advertised state not advanced"
        );
        record_delta(&runtime.telemetry, kind, "hash_mismatch", None);
        return;
    };

    state.mark_applied(notification.epoch, applied_hash);
    // Telemetry catalog overflow/mismatch is fail-open for serving and the
    // facade emits at most one process-local warning before collapsing the
    // affected pair to `other`. Activate it only after the trusted hash and
    // advertised state have both committed.
    runtime
        .telemetry
        .extend_catalog(&notification.model_id, &notification.profiles_added);
    record_delta(
        &runtime.telemetry,
        kind,
        "applied",
        Some(notification.epoch),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;
    use tokio::sync::Mutex;

    fn notification(
        producer_id: &str,
        bundle_id: &str,
        pool: Option<&str>,
        model_id: &str,
    ) -> ConfigNotification {
        ConfigNotification {
            producer_id: producer_id.to_string(),
            bundle_id: bundle_id.to_string(),
            epoch: 1,
            bundle_config_hash: "hash-1".to_string(),
            bundle_pool_config_hashes: HashMap::new(),
            model_id: model_id.to_string(),
            profiles_added: vec!["default".to_string()],
            model_config: format!("sie_id: {model_id}\nprofiles:\n  default: {{}}\n"),
            affected_bundles: vec![bundle_id.to_string()],
            pool: pool.map(ToString::to_string),
        }
    }

    fn spawn_conditional_apply_server(
        listener: UnixListener,
        models: Arc<Mutex<Vec<String>>>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _addr)) = listener.accept().await else {
                    break;
                };
                let models = Arc::clone(&models);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0_u8; 4];
                        if stream.read_exact(&mut len_buf).await.is_err() {
                            break;
                        }
                        let len = u32::from_be_bytes(len_buf) as usize;
                        let mut frame = vec![0_u8; len];
                        if stream.read_exact(&mut frame).await.is_err() {
                            break;
                        }
                        let raw: serde_json::Value =
                            rmp_serde::from_slice(&frame).expect("decode request envelope");
                        let request_id = raw
                            .get("request_id")
                            .and_then(serde_json::Value::as_str)
                            .expect("request id");
                        let model_id = raw
                            .get("body")
                            .and_then(|body| body.get("model_id"))
                            .and_then(serde_json::Value::as_str)
                            .expect("model id")
                            .to_string();
                        models.lock().await.push(model_id.clone());

                        let response = serde_json::json!({
                            "version": crate::ipc_types::IPC_VERSION,
                            "request_id": request_id,
                            "ok": true,
                            "body": {
                                "applied": model_id != "failed/model",
                                "bundle_config_hash": "hash-1",
                                "config_version": 1_u64
                            }
                        });
                        let bytes =
                            rmp_serde::to_vec_named(&response).expect("encode response envelope");
                        stream
                            .write_all(&(bytes.len() as u32).to_be_bytes())
                            .await
                            .expect("write response length");
                        stream.write_all(&bytes).await.expect("write response");
                    }
                });
            }
        })
    }

    #[test]
    fn bundle_subject_requires_single_safe_token() {
        assert_eq!(
            bundle_subject("default").unwrap(),
            "sie.config.models.default"
        );
        assert!(bundle_subject("").is_err());
        assert!(bundle_subject("bad.bundle").is_err());
        assert!(bundle_subject("bad*").is_err());
        assert!(bundle_subject("bad>").is_err());
    }

    #[test]
    fn trusted_producer_matches_exact_and_pod_prefix() {
        let trusted = vec!["sie-config".to_string()];
        assert!(is_trusted_producer("sie-config", &trusted));
        assert!(is_trusted_producer("sie-config-5f7b6d8c-kxwvr", &trusted));
        assert!(is_trusted_producer("sie-config-0", &trusted));
        assert!(!is_trusted_producer("sie-configuration", &trusted));
        assert!(!is_trusted_producer("other", &trusted));
        assert!(is_trusted_producer("other", &[]));
    }

    #[test]
    fn notification_targets_bundle_requires_payload_bundle_match() {
        let mut notification = ConfigNotification {
            producer_id: "sie-config".into(),
            bundle_id: "default".into(),
            epoch: 1,
            bundle_config_hash: "h".into(),
            bundle_pool_config_hashes: HashMap::new(),
            model_id: "m".into(),
            profiles_added: vec![],
            model_config: "x".into(),
            affected_bundles: vec![],
            pool: None,
        };
        assert!(notification_targets_bundle(&notification, "default"));
        assert!(!notification_targets_bundle(&notification, "other"));
        notification.affected_bundles.push("other".into());
        assert!(!notification_targets_bundle(&notification, "other"));
    }

    #[test]
    fn notification_targets_pool_and_selects_pool_hash() {
        let notification = ConfigNotification {
            producer_id: "sie-config".into(),
            bundle_id: "candle".into(),
            epoch: 1,
            bundle_config_hash: "global-hash".into(),
            bundle_pool_config_hashes: HashMap::from([(
                "candle".to_string(),
                HashMap::from([
                    ("default".to_string(), "default-hash".to_string()),
                    ("customer-a".to_string(), "tenant-hash".to_string()),
                ]),
            )]),
            model_id: "m".into(),
            profiles_added: vec![],
            model_config: "x".into(),
            affected_bundles: vec!["candle".into()],
            pool: Some("Customer-A".into()),
        };

        assert!(notification_targets_pool(&notification, "customer-a"));
        assert!(!notification_targets_pool(&notification, "default"));
        assert_eq!(
            notification_bundle_hash_for_pool(&notification, "customer-a"),
            "tenant-hash"
        );
    }

    #[test]
    fn applied_bundle_hash_requires_control_plane_worker_agreement() {
        assert_eq!(
            verified_applied_bundle_hash(Some("control-hash"), "control-hash").as_deref(),
            Some("control-hash")
        );
        assert_eq!(
            verified_applied_bundle_hash(Some("control-hash"), "worker-hash"),
            None
        );
        assert_eq!(verified_applied_bundle_hash(Some("control-hash"), ""), None);
        assert_eq!(
            verified_applied_bundle_hash(Some(""), "worker-hash").as_deref(),
            Some("worker-hash")
        );
        assert_eq!(
            verified_applied_bundle_hash(None, "worker-hash").as_deref(),
            Some("worker-hash")
        );
    }

    #[test]
    fn config_notification_accepts_router_id_alias() {
        let json = r#"{
            "router_id": "sie-config-0",
            "bundle_id": "default",
            "epoch": 3,
            "bundle_config_hash": "hash",
            "model_id": "test/model",
            "profiles_added": ["default"],
            "model_config": "sie_id: test/model\n",
            "affected_bundles": ["default"]
        }"#;
        let parsed: ConfigNotification = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.producer_id, "sie-config-0");
        assert_eq!(parsed.bundle_id, "default");
        assert_eq!(parsed.epoch, 3);
        assert!(parsed.bundle_pool_config_hashes.is_empty());
        assert!(parsed.pool.is_none());
        assert!(notification_targets_pool(&parsed, "customer-a"));
    }

    #[test]
    fn apply_state_tracks_epoch_monotonically_and_hash() {
        let state = ConfigApplyState::new("initial".into());
        state.mark_applied(10, "h10".into());
        state.mark_applied(7, "h7".into());
        assert_eq!(state.epoch(), 10);
        assert_eq!(
            state.bundle_config_hash().read().unwrap().as_str(),
            "h10",
            "stale apply results must not roll the advertised hash backward"
        );
    }

    #[test]
    fn epoch_zero_model_config_is_not_stale_for_no_store_mode() {
        assert!(!notification_is_stale(0, 0));
        assert!(notification_is_stale(1, 1));
        assert!(notification_is_stale(0, 1));
    }

    #[test]
    fn epoch_zero_apply_updates_hash_without_advancing_epoch() {
        let state = ConfigApplyState::new("initial".into());
        state.mark_applied(0, "h0".into());
        assert_eq!(state.epoch(), 0);
        assert_eq!(state.bundle_config_hash().read().unwrap().as_str(), "h0");
    }

    #[test]
    fn live_apply_can_clear_bundle_hash() {
        let state = ConfigApplyState::new("old".into());
        state.mark_applied(4, String::new());
        assert_eq!(state.epoch(), 4);
        assert_eq!(state.bundle_config_hash().read().unwrap().as_str(), "");
    }

    #[test]
    fn export_reconcile_can_clear_bundle_hash() {
        let state = ConfigApplyState::new("old".into());
        assert!(state.mark_export_reconciled(4, Some(String::new()), false));
        assert_eq!(state.epoch(), 4);
        assert_eq!(state.bundle_config_hash().read().unwrap().as_str(), "");
    }

    #[test]
    fn set_loaded_models_are_trimmed_deduped_and_sorted() {
        let state = ConfigApplyState::new("initial".into());
        state.set_loaded_models(vec![
            "model/d".into(),
            "".into(),
            " model/c ".into(),
            "model/d".into(),
        ]);
        assert_eq!(
            state.loaded_models().read().unwrap().as_slice(),
            ["model/c".to_string(), "model/d".to_string()]
        );
    }

    #[test]
    fn only_current_hash_is_accepted_after_config_advance() {
        let state = ConfigApplyState::new("h0".into());
        assert!(state.accepts_bundle_config_hash(""));
        assert!(state.accepts_bundle_config_hash("h0"));
        assert!(!state.accepts_bundle_config_hash("h1"));

        state.mark_applied(1, "h1".into());
        assert!(!state.accepts_bundle_config_hash("h0"));
        assert!(state.accepts_bundle_config_hash("h1"));
        assert!(!state.accepts_bundle_config_hash("missing"));
    }

    #[tokio::test]
    async fn config_write_waits_for_inflight_execution_guard() {
        let state = Arc::new(ConfigApplyState::new("h0".into()));
        let execution = state.lock_execution().await;
        let writer_state = Arc::clone(&state);
        let writer = tokio::spawn(async move {
            let _write = writer_state.lock_apply().await;
            writer_state.set_bundle_hash("h1".into());
        });

        tokio::task::yield_now().await;
        assert!(!writer.is_finished());
        assert_eq!(state.current_bundle_config_hash(), "h0");

        drop(execution);
        writer.await.unwrap();
        assert_eq!(state.current_bundle_config_hash(), "h1");
    }

    #[test]
    fn retryable_ipc_errors_are_transport_or_timeout_only() {
        let refused = IpcError::Io(std::io::Error::from(std::io::ErrorKind::ConnectionRefused));
        let missing = IpcError::Io(std::io::Error::from(std::io::ErrorKind::NotFound));
        let server = IpcError::Server("bad config".into());
        let version = IpcError::VersionMismatch { got: 0 };

        assert!(retryable_ipc_error(&refused));
        assert!(retryable_ipc_error(&missing));
        assert!(retryable_ipc_error(&IpcError::Timeout));
        assert!(!retryable_ipc_error(&server));
        assert!(!retryable_ipc_error(&version));
    }

    #[tokio::test]
    async fn telemetry_catalog_expands_only_after_trusted_successful_apply() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let socket_path = tmp.path().join("ipc.sock");
        let listener = UnixListener::bind(&socket_path).expect("bind fake ipc socket");
        let applied_models = Arc::new(Mutex::new(Vec::new()));
        let server = spawn_conditional_apply_server(listener, Arc::clone(&applied_models));

        let runtime_state = Arc::new(RuntimeState::new());
        let telemetry = SidecarTelemetry::for_tests(&[]);
        let runtime = SubscriberRuntime {
            bundle: "default".to_string(),
            pool: "default".to_string(),
            trusted_producers: vec!["sie-config".to_string()],
            ipc: AdapterWorkerPool::new(&[socket_path], 1, 1, 1, runtime_state),
            state: Arc::new(ConfigApplyState::new(String::new())),
            telemetry: telemetry.clone(),
            shutdown: Arc::new(Shutdown::new()),
        };

        apply_notification(
            &runtime,
            notification("untrusted", "default", None, "untrusted/model"),
        )
        .await;
        apply_notification(
            &runtime,
            notification("sie-config", "other", None, "wrong-bundle/model"),
        )
        .await;
        apply_notification(
            &runtime,
            notification("sie-config", "default", Some("other"), "wrong-pool/model"),
        )
        .await;
        apply_notification(
            &runtime,
            notification("sie-config", "default", None, "failed/model"),
        )
        .await;

        assert_eq!(telemetry.allowed_model_count_for_tests(), 0);
        assert!(!telemetry.profile_allowed_for_tests("default"));

        apply_notification(
            &runtime,
            notification("sie-config", "default", None, "trusted/model"),
        )
        .await;

        assert!(telemetry.model_allowed_for_tests("trusted/model"));
        assert!(telemetry.profile_allowed_for_tests("default"));
        assert!(!telemetry.model_allowed_for_tests("untrusted/model"));
        assert!(!telemetry.model_allowed_for_tests("wrong-bundle/model"));
        assert!(!telemetry.model_allowed_for_tests("wrong-pool/model"));
        assert!(!telemetry.model_allowed_for_tests("failed/model"));
        assert_eq!(
            applied_models.lock().await.as_slice(),
            ["failed/model", "trusted/model"],
            "rejected notifications must not reach IPC"
        );
        server.abort();
    }

    #[tokio::test]
    async fn apply_via_ipc_does_not_eager_ensure_candle_model_ready() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let socket_path = tmp.path().join("ipc.sock");
        let listener = UnixListener::bind(&socket_path).expect("bind fake ipc socket");
        let methods = Arc::new(Mutex::new(Vec::<String>::new()));
        let server_methods = Arc::clone(&methods);

        let server = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _addr)) = listener.accept().await else {
                    break;
                };
                let methods = Arc::clone(&server_methods);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0_u8; 4];
                        if stream.read_exact(&mut len_buf).await.is_err() {
                            break;
                        }
                        let len = u32::from_be_bytes(len_buf) as usize;
                        let mut frame = vec![0_u8; len];
                        if stream.read_exact(&mut frame).await.is_err() {
                            break;
                        }
                        let raw: serde_json::Value =
                            rmp_serde::from_slice(&frame).expect("decode request envelope");
                        let method = raw
                            .get("method")
                            .and_then(serde_json::Value::as_str)
                            .expect("request method")
                            .to_string();
                        let request_id = raw
                            .get("request_id")
                            .and_then(serde_json::Value::as_str)
                            .expect("request id")
                            .to_string();
                        methods.lock().await.push(method.clone());

                        let body = match method.as_str() {
                            crate::ipc_types::METHOD_APPLY_MODEL_CONFIG => serde_json::json!({
                                "applied": true,
                                "bundle_config_hash": "hash",
                                "config_version": 1_u64
                            }),
                            crate::ipc_types::METHOD_ENSURE_MODEL_READY => serde_json::json!({
                                "state": "ready",
                                "batch_budget": 64_u32
                            }),
                            other => panic!("unexpected fake IPC method {other}"),
                        };
                        let response = serde_json::json!({
                            "version": crate::ipc_types::IPC_VERSION,
                            "request_id": request_id,
                            "ok": true,
                            "body": body
                        });
                        let bytes =
                            rmp_serde::to_vec_named(&response).expect("encode response envelope");
                        stream
                            .write_all(&(bytes.len() as u32).to_be_bytes())
                            .await
                            .expect("write response length");
                        stream.write_all(&bytes).await.expect("write response");
                    }
                });
            }
        });

        let runtime_state = Arc::new(RuntimeState::new());
        let ipc = AdapterWorkerPool::new(&[socket_path], 1, 1, 1, runtime_state);
        let shutdown = Arc::new(Shutdown::new());
        let resp = apply_via_ipc_with_retry(
            ipc,
            shutdown,
            ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "topk-io/Iso-ModernColBERT".to_string(),
                epoch: 7,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: "sie_id: topk-io/Iso-ModernColBERT\n".to_string(),
            },
            "topk-io/Iso-ModernColBERT",
            7,
        )
        .await
        .expect("apply via ipc")
        .expect("response");

        assert!(resp.applied);
        assert_eq!(
            methods.lock().await.as_slice(),
            [crate::ipc_types::METHOD_APPLY_MODEL_CONFIG],
            "config apply must not trigger Candle eager model readiness"
        );
        server.abort();
    }
}
