//! Periodic worker-side reconciliation against `sie-config`.
//!
//! Core NATS gives the sidecar a fast live path for
//! `sie.config.models.<bundle>`, but it does not replay messages that were
//! published while a worker was disconnected. This module closes that gap by
//! polling `GET /v1/configs/epoch` and fetching `GET /v1/configs/export` when
//! the control plane is ahead or when its compact per-bundle config fingerprint
//! changes. A slower full-export pass also covers legacy deployments where
//! `sie-config` cannot provide a compact fingerprint.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::backend::AdapterWorkerPool;
use crate::config_subscriber::{
    replace_via_ipc_with_retry, verified_applied_bundle_hash, ConfigApplyState,
    MAX_MODEL_CONFIG_BYTES,
};
use crate::ipc_client::IpcError;
use crate::ipc_types::{
    ReplaceModelConfigEntry, ReplaceModelConfigsRequest, ReplaceModelConfigsResponse,
};
use crate::observability::metrics::SidecarTelemetry;
use crate::shutdown::Shutdown;

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(30);
pub const DEFAULT_FULL_EXPORT_INTERVAL: Duration = Duration::from_secs(5 * 60);
const DEFAULT_MODEL_POOL: &str = "default";

#[derive(Debug, Clone)]
pub struct ReconcilerConfig {
    pub base_url: String,
    pub admin_token: Option<String>,
    pub bundle: String,
    pub pool: String,
    pub poll_interval: Duration,
    pub full_export_interval: Option<Duration>,
}

#[derive(Debug, Deserialize)]
struct EpochSnapshot {
    #[serde(default)]
    epoch: u64,
    #[serde(default)]
    bundle_config_hashes_hash: String,
}

#[derive(Debug, Deserialize)]
struct ExportSnapshot {
    #[serde(default)]
    epoch: u64,
    #[serde(default)]
    bundle_config_hashes: HashMap<String, String>,
    #[serde(default)]
    bundle_pool_config_hashes: HashMap<String, HashMap<String, String>>,
    #[serde(default)]
    models: Vec<ExportedModel>,
}

#[derive(Debug, Deserialize)]
struct ExportedModel {
    #[serde(default)]
    model_id: String,
    #[serde(default)]
    raw_yaml: Option<String>,
    #[serde(default)]
    model_config: Option<serde_json::Value>,
    #[serde(default)]
    affected_bundles: Vec<String>,
    #[serde(default)]
    pool: Option<String>,
}

impl ExportedModel {
    fn targets_bundle(&self, bundle: &str) -> bool {
        self.affected_bundles.iter().any(|b| b == bundle)
    }

    fn targets_pool(&self, pool: &str) -> bool {
        self.pool_name() == normalize_pool_name(pool)
    }

    fn pool_name(&self) -> String {
        if let Some(pool) = self.pool.as_deref() {
            return normalize_pool_name(pool);
        }
        self.model_config
            .as_ref()
            .and_then(|value| value.get("pool"))
            .and_then(|value| value.as_str())
            .map(normalize_pool_name)
            .unwrap_or_else(|| DEFAULT_MODEL_POOL.to_string())
    }

    fn config_body(&self) -> Result<Option<String>, serde_json::Error> {
        if let Some(raw) = self.raw_yaml.as_deref() {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                return Ok(Some(trimmed.to_string()));
            }
        }
        match &self.model_config {
            Some(value) if !value.is_null() => serde_json::to_string(value).map(Some),
            _ => Ok(None),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum ReconcileError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("unexpected status {status} from {url}: {body}")]
    BadStatus {
        status: u16,
        url: String,
        body: String,
    },
}

#[derive(Debug)]
struct ExportOutcome {
    epoch: u64,
    applied: usize,
    failed: usize,
    skipped: usize,
    shutdown: bool,
    state_updated: bool,
    export_signature: Option<blake3::Hash>,
}

#[derive(Debug, Clone, Copy)]
struct ExportReconcileOptions<'a> {
    force_epoch: bool,
    reason: &'a str,
    previous_export_signature: Option<blake3::Hash>,
}

struct ReconcileClient {
    base_url: String,
    admin_token: Option<String>,
    http: reqwest::Client,
}

struct ReconcileRuntime<'a> {
    client: &'a ReconcileClient,
    bundle: &'a str,
    pool: &'a str,
    ipc: Arc<AdapterWorkerPool>,
    state: Arc<ConfigApplyState>,
    telemetry: SidecarTelemetry,
    shutdown: Arc<Shutdown>,
}

#[derive(Clone, Copy)]
struct ReconcileScope<'a> {
    bundle: &'a str,
    pool: &'a str,
}

impl ReconcileClient {
    fn new(base_url: String, admin_token: Option<String>) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder().timeout(HTTP_TIMEOUT).build()?;
        Ok(Self {
            base_url,
            admin_token,
            http,
        })
    }

    async fn fetch_epoch(&self) -> Result<EpochSnapshot, ReconcileError> {
        let url = format!("{}/v1/configs/epoch", self.base_url.trim_end_matches('/'));
        let mut req = self.http.get(&url);
        if let Some(token) = &self.admin_token {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ReconcileError::BadStatus {
                status: status.as_u16(),
                url,
                body,
            });
        }
        Ok(resp.json().await?)
    }

    async fn fetch_export(&self) -> Result<ExportSnapshot, ReconcileError> {
        let url = format!("{}/v1/configs/export", self.base_url.trim_end_matches('/'));
        let mut req = self.http.get(&url);
        if let Some(token) = &self.admin_token {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ReconcileError::BadStatus {
                status: status.as_u16(),
                url,
                body,
            });
        }
        Ok(resp.json().await?)
    }
}

fn record_reconcile(
    telemetry: &SidecarTelemetry,
    operation: &str,
    result: &str,
    applied_epoch: Option<u64>,
) {
    debug!(result, "worker-config: export reconcile outcome");
    telemetry.config_apply("export", operation, result, applied_epoch);
}

fn normalize_pool_name(raw: &str) -> String {
    let pool = raw.trim().to_ascii_lowercase();
    if pool.is_empty() {
        DEFAULT_MODEL_POOL.to_string()
    } else {
        pool
    }
}

fn drift_reason(
    local_epoch: u64,
    remote: &EpochSnapshot,
    local_bundle_config_hashes_hash: &str,
    full_export_due: bool,
) -> Option<&'static str> {
    if remote.epoch > local_epoch {
        return Some("epoch_ahead");
    }
    if remote.epoch < local_epoch {
        return Some("epoch_rewind");
    }
    if !remote.bundle_config_hashes_hash.is_empty()
        && remote.bundle_config_hashes_hash != local_bundle_config_hashes_hash
    {
        return Some("bundle_config_hashes_hash");
    }
    if remote.epoch == 0 && full_export_due {
        return Some("periodic_export");
    }
    None
}

fn compute_export_signature(
    bundle: &str,
    bundle_config_hash: Option<&str>,
    models: &[ReplaceModelConfigEntry],
) -> blake3::Hash {
    fn update_len_prefixed(hasher: &mut blake3::Hasher, value: &str) {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }

    let mut hasher = blake3::Hasher::new();
    update_len_prefixed(&mut hasher, bundle);
    update_len_prefixed(&mut hasher, bundle_config_hash.unwrap_or_default());

    let mut entries: Vec<(&str, &str)> = models
        .iter()
        .map(|model| (model.model_id.as_str(), model.model_config.as_str()))
        .collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0).then_with(|| left.1.cmp(right.1)));
    hasher.update(&(entries.len() as u64).to_le_bytes());
    for (model_id, model_config) in entries {
        update_len_prefixed(&mut hasher, model_id);
        update_len_prefixed(&mut hasher, model_config);
    }
    hasher.finalize()
}

/// Spawn the optional export reconciler.
///
/// The task is disabled when `config.base_url` is empty. The caller owns the
/// returned handle and aborts it on shutdown, matching the config subscriber
/// and health publisher lifecycle.
pub fn spawn(
    config: Option<ReconcilerConfig>,
    ipc: Arc<AdapterWorkerPool>,
    state: Arc<ConfigApplyState>,
    telemetry: SidecarTelemetry,
    shutdown: Arc<Shutdown>,
) -> Option<JoinHandle<()>> {
    let config = config?;
    if config.base_url.trim().is_empty() {
        return None;
    }

    Some(tokio::spawn(async move {
        let client = match ReconcileClient::new(config.base_url.clone(), config.admin_token) {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, "worker-config: failed to build export reconciler client");
                record_reconcile(&telemetry, "full_export", "client_error", None);
                return;
            }
        };

        info!(
            base_url = %config.base_url,
            bundle = %config.bundle,
            pool = %config.pool,
            poll_interval_ms = config.poll_interval.as_millis() as u64,
            full_export_interval_ms = config.full_export_interval.map(|d| d.as_millis() as u64),
            "worker-config: export reconciler started"
        );

        let runtime = ReconcileRuntime {
            client: &client,
            bundle: &config.bundle,
            pool: &config.pool,
            ipc: Arc::clone(&ipc),
            state: Arc::clone(&state),
            telemetry,
            shutdown: Arc::clone(&shutdown),
        };
        let mut last_successful_export: Option<Instant> = None;
        let mut last_export_signature: Option<blake3::Hash> = None;
        let mut last_bundle_config_hashes_hash = match client.fetch_epoch().await {
            Ok(snapshot) => snapshot.bundle_config_hashes_hash,
            Err(e) => {
                warn!(
                    error = %e,
                    "worker-config: initial epoch fingerprint poll failed; startup export will proceed"
                );
                String::new()
            }
        };
        match reconcile_export(&runtime, false, "startup", None).await {
            Ok(outcome) if outcome.shutdown => return,
            Ok(outcome) if outcome.failed == 0 => {
                last_successful_export = Some(Instant::now());
                if let Some(signature) = outcome.export_signature {
                    last_export_signature = Some(signature);
                }
                info!(
                    epoch = outcome.epoch,
                    applied = outcome.applied,
                    skipped = outcome.skipped,
                    state_updated = outcome.state_updated,
                    "worker-config: startup export reconcile complete"
                );
            }
            Ok(outcome) => {
                warn!(
                    epoch = outcome.epoch,
                    applied = outcome.applied,
                    failed = outcome.failed,
                    skipped = outcome.skipped,
                    "worker-config: startup export reconcile partial; will retry"
                );
            }
            Err(e) => {
                warn!(error = %e, "worker-config: startup export reconcile failed; will retry");
                record_reconcile(&runtime.telemetry, "full_export", "fetch_error", None);
            }
        }

        let mut ticker = interval(config.poll_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await;

        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait() => return,
                _ = ticker.tick() => {
                    let local_epoch = state.epoch();
                    let remote = match client.fetch_epoch().await {
                        Ok(snapshot) => snapshot,
                        Err(e) => {
                            warn!(error = %e, "worker-config: epoch poll failed");
                            record_reconcile(&runtime.telemetry, "poll", "epoch_error", None);
                            continue;
                        }
                    };
                    let remote_epoch = remote.epoch;
                    let remote_bundle_config_hashes_hash =
                        remote.bundle_config_hashes_hash.clone();

                    let full_export_due = config.full_export_interval.is_some_and(|interval| {
                        last_successful_export
                            .map(|last| last.elapsed() >= interval)
                            .unwrap_or(true)
                    });

                    let Some(reason) = drift_reason(
                        local_epoch,
                        &remote,
                        &last_bundle_config_hashes_hash,
                        full_export_due,
                    ) else {
                        debug!(
                            local_epoch,
                            remote_epoch,
                            bundle_config_hashes_hash = %remote_bundle_config_hashes_hash,
                            "worker-config: epoch poll in sync"
                        );
                        continue;
                    };

                    let force_epoch = remote_epoch < local_epoch;
                    if force_epoch {
                        warn!(
                            local_epoch,
                            remote_epoch,
                            "worker-config: local epoch ahead of sie-config; export reconcile will force epoch after successful apply"
                        );
                    } else {
                        info!(
                            local_epoch,
                            remote_epoch,
                            reason,
                            bundle_config_hashes_hash = %remote_bundle_config_hashes_hash,
                            "worker-config: drift detected; fetching export"
                        );
                    }

                    match reconcile_export(&runtime, force_epoch, reason, last_export_signature)
                    .await
                    {
                        Ok(outcome) if outcome.shutdown => return,
                        Ok(outcome) if outcome.failed == 0 => {
                            last_successful_export = Some(Instant::now());
                            if let Some(signature) = outcome.export_signature {
                                last_export_signature = Some(signature);
                            }
                            if !remote_bundle_config_hashes_hash.is_empty() {
                                last_bundle_config_hashes_hash =
                                    remote_bundle_config_hashes_hash.clone();
                            }
                            info!(
                                epoch = outcome.epoch,
                                applied = outcome.applied,
                                skipped = outcome.skipped,
                                state_updated = outcome.state_updated,
                                reason,
                                "worker-config: export reconcile complete"
                            );
                        }
                        Ok(outcome) => {
                            warn!(
                                epoch = outcome.epoch,
                                applied = outcome.applied,
                                failed = outcome.failed,
                                skipped = outcome.skipped,
                                reason,
                                "worker-config: export reconcile partial; epoch/hash not advanced"
                            );
                        }
                        Err(e) => {
                            warn!(error = %e, reason, "worker-config: export reconcile failed");
                            record_reconcile(&runtime.telemetry, "full_export", "fetch_error", None);
                        }
                    }
                }
            }
        }
    }))
}

async fn reconcile_export(
    runtime: &ReconcileRuntime<'_>,
    force_epoch: bool,
    reason: &str,
    previous_export_signature: Option<blake3::Hash>,
) -> Result<ExportOutcome, ReconcileError> {
    let snapshot = runtime.client.fetch_export().await?;
    let ipc = Arc::clone(&runtime.ipc);
    let shutdown = Arc::clone(&runtime.shutdown);
    Ok(reconcile_export_snapshot(
        snapshot,
        ReconcileScope {
            bundle: runtime.bundle,
            pool: runtime.pool,
        },
        &runtime.state,
        &runtime.telemetry,
        ExportReconcileOptions {
            force_epoch,
            reason,
            previous_export_signature,
        },
        move |req, epoch| {
            let ipc = Arc::clone(&ipc);
            let shutdown = Arc::clone(&shutdown);
            async move { replace_via_ipc_with_retry(ipc, shutdown, req, epoch).await }
        },
    )
    .await)
}

async fn reconcile_export_snapshot<F, Fut>(
    snapshot: ExportSnapshot,
    scope: ReconcileScope<'_>,
    state: &ConfigApplyState,
    telemetry: &SidecarTelemetry,
    options: ExportReconcileOptions<'_>,
    mut apply: F,
) -> ExportOutcome
where
    F: FnMut(ReplaceModelConfigsRequest, u64) -> Fut,
    Fut: Future<Output = Result<Option<ReplaceModelConfigsResponse>, IpcError>>,
{
    let _apply_guard = state.lock_apply().await;

    if !options.force_epoch && snapshot.epoch < state.epoch() {
        record_reconcile(telemetry, "full_export", "stale_export", None);
        return ExportOutcome {
            epoch: snapshot.epoch,
            applied: 0,
            failed: 0,
            skipped: snapshot.models.len(),
            shutdown: false,
            state_updated: false,
            export_signature: None,
        };
    }

    let mut applied = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;
    let mut exported_models = Vec::new();
    let normalized_pool = normalize_pool_name(scope.pool);
    let control_plane_hash = match snapshot.bundle_pool_config_hashes.get(scope.bundle) {
        Some(pool_hashes) => Some(
            pool_hashes
                .get(&normalized_pool)
                .cloned()
                .unwrap_or_default(),
        ),
        None => snapshot.bundle_config_hashes.get(scope.bundle).cloned(),
    };
    for model in snapshot.models {
        if !model.targets_bundle(scope.bundle) || !model.targets_pool(&normalized_pool) {
            skipped += 1;
            continue;
        }

        let model_config = match model.config_body() {
            Ok(Some(body)) => body,
            Ok(None) => {
                skipped += 1;
                continue;
            }
            Err(e) => {
                warn!(
                    model = %model.model_id,
                    error = %e,
                    "worker-config: exported model_config could not be serialized"
                );
                failed += 1;
                continue;
            }
        };

        if model_config.len() > MAX_MODEL_CONFIG_BYTES {
            warn!(
                model = %model.model_id,
                bytes = model_config.len(),
                max_bytes = MAX_MODEL_CONFIG_BYTES,
                "worker-config: exported model_config is oversized"
            );
            failed += 1;
            continue;
        }

        exported_models.push(ReplaceModelConfigEntry {
            model_id: model.model_id.clone(),
            model_config,
        });
    }

    let export_signature = if failed == 0 {
        Some(compute_export_signature(
            scope.bundle,
            control_plane_hash.as_deref(),
            &exported_models,
        ))
    } else {
        None
    };

    if options.reason == "periodic_export"
        && export_signature.is_some()
        && export_signature == options.previous_export_signature
    {
        record_reconcile(telemetry, "full_export", "no_change", Some(snapshot.epoch));
        debug!(
            epoch = snapshot.epoch,
            skipped, "worker-config: periodic export unchanged; skipping replace"
        );
        return ExportOutcome {
            epoch: snapshot.epoch,
            applied: 0,
            failed: 0,
            skipped,
            shutdown: false,
            state_updated: false,
            export_signature,
        };
    }

    let state_updated = if failed == 0 {
        let req = ReplaceModelConfigsRequest {
            bundle_id: scope.bundle.to_string(),
            epoch: snapshot.epoch,
            bundle_config_hash: control_plane_hash.clone().unwrap_or_default(),
            models: exported_models,
        };
        let resp = match apply(req, snapshot.epoch).await {
            Ok(Some(resp)) => resp,
            Ok(None) => {
                record_reconcile(telemetry, "full_export", "shutdown", None);
                return ExportOutcome {
                    epoch: snapshot.epoch,
                    applied,
                    failed,
                    skipped,
                    shutdown: true,
                    state_updated: false,
                    export_signature,
                };
            }
            Err(e) => {
                warn!(
                    epoch = snapshot.epoch,
                    error = %e,
                    "worker-config: exported model replace failed"
                );
                record_reconcile(telemetry, "full_export", "partial", None);
                return ExportOutcome {
                    epoch: snapshot.epoch,
                    applied,
                    failed: failed + 1,
                    skipped,
                    shutdown: false,
                    state_updated: false,
                    export_signature,
                };
            }
        };

        if !resp.applied {
            warn!(
                epoch = snapshot.epoch,
                "worker-config: exported model replace returned applied=false"
            );
            record_reconcile(telemetry, "full_export", "partial", None);
            return ExportOutcome {
                epoch: snapshot.epoch,
                applied,
                failed: failed + 1,
                skipped,
                shutdown: false,
                state_updated: false,
                export_signature,
            };
        }

        applied = resp.applied_models.len();
        let Some(applied_hash) =
            verified_applied_bundle_hash(control_plane_hash.as_deref(), &resp.bundle_config_hash)
        else {
            warn!(
                epoch = snapshot.epoch,
                control_plane_hash = %control_plane_hash.as_deref().unwrap_or_default(),
                applied_hash = %resp.bundle_config_hash,
                "worker-config: worker config hash differs from control-plane export hash; advertised state not advanced"
            );
            record_reconcile(telemetry, "full_export", "hash_mismatch", None);
            return ExportOutcome {
                epoch: snapshot.epoch,
                applied,
                failed: failed + 1,
                skipped,
                shutdown: false,
                state_updated: false,
                export_signature: None,
            };
        };
        let state_updated =
            state.mark_export_reconciled(snapshot.epoch, Some(applied_hash), options.force_epoch);
        if state_updated {
            // Activate only the catalog belonging to the trusted, committed
            // worker state. Cardinality overflow remains fail-open for serving.
            telemetry.replace_catalog(&resp.applied_models, &resp.applied_profiles);
        }
        let result = if applied == 0 {
            "no_relevant_models"
        } else if state_updated {
            "applied"
        } else {
            "stale_export"
        };
        record_reconcile(
            telemetry,
            "full_export",
            result,
            state_updated.then_some(snapshot.epoch),
        );
        state_updated
    } else {
        record_reconcile(telemetry, "full_export", "partial", None);
        false
    };

    debug!(
        epoch = snapshot.epoch,
        applied,
        failed,
        skipped,
        reason = options.reason,
        pool = %normalized_pool,
        state_updated,
        "worker-config: export reconcile attempt finished"
    );

    ExportOutcome {
        epoch: snapshot.epoch,
        applied,
        failed,
        skipped,
        shutdown: false,
        state_updated,
        export_signature,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use tokio::sync::Mutex;

    fn exported_model_with_yaml(model_id: &str, bundles: &[&str], raw_yaml: &str) -> ExportedModel {
        ExportedModel {
            model_id: model_id.into(),
            raw_yaml: Some(raw_yaml.to_string()),
            model_config: None,
            affected_bundles: bundles.iter().map(|b| (*b).to_string()).collect(),
            pool: None,
        }
    }

    fn exported_model(model_id: &str, bundles: &[&str]) -> ExportedModel {
        exported_model_with_yaml(
            model_id,
            bundles,
            &format!("sie_id: {model_id}\nprofiles:\n  default: {{}}\n"),
        )
    }

    fn exported_model_with_pool(model_id: &str, bundles: &[&str], pool: &str) -> ExportedModel {
        ExportedModel {
            model_id: model_id.into(),
            raw_yaml: Some(format!(
                "sie_id: {model_id}\npool: {pool}\nprofiles:\n  default: {{}}\n"
            )),
            model_config: Some(serde_json::json!({"pool": pool})),
            affected_bundles: bundles.iter().map(|b| (*b).to_string()).collect(),
            pool: Some(pool.to_string()),
        }
    }

    fn scope<'a>(bundle: &'a str, pool: &'a str) -> ReconcileScope<'a> {
        ReconcileScope { bundle, pool }
    }

    fn reconcile_options(reason: &'static str) -> ExportReconcileOptions<'static> {
        ExportReconcileOptions {
            force_epoch: false,
            reason,
            previous_export_signature: None,
        }
    }

    #[test]
    fn drift_reason_detects_same_epoch_bundle_config_hashes_hash_change() {
        let remote = EpochSnapshot {
            epoch: 9,
            bundle_config_hashes_hash: "new-fingerprint".into(),
        };

        assert_eq!(
            drift_reason(9, &remote, "old-fingerprint", false),
            Some("bundle_config_hashes_hash")
        );
        assert_eq!(drift_reason(9, &remote, "new-fingerprint", true), None);

        let legacy_remote = EpochSnapshot {
            epoch: 9,
            bundle_config_hashes_hash: String::new(),
        };
        assert_eq!(
            drift_reason(9, &legacy_remote, "old-fingerprint", false),
            None
        );
    }

    #[test]
    fn drift_reason_preserves_epoch_zero_periodic_export_with_stable_fingerprint() {
        let remote = EpochSnapshot {
            epoch: 0,
            bundle_config_hashes_hash: "stable-fingerprint".into(),
        };

        assert_eq!(
            drift_reason(0, &remote, "stable-fingerprint", true),
            Some("periodic_export")
        );
        assert_eq!(drift_reason(0, &remote, "stable-fingerprint", false), None);

        let legacy_remote = EpochSnapshot {
            epoch: 0,
            bundle_config_hashes_hash: String::new(),
        };
        assert_eq!(
            drift_reason(0, &legacy_remote, "stable-fingerprint", true),
            Some("periodic_export")
        );
    }

    #[test]
    fn exported_model_targets_exact_bundle_only() {
        let model = ExportedModel {
            model_id: "m".into(),
            raw_yaml: Some("sie_id: m\n".into()),
            model_config: None,
            affected_bundles: vec!["default".into(), "vision".into()],
            pool: None,
        };
        assert!(model.targets_bundle("default"));
        assert!(!model.targets_bundle("def"));
        assert!(!model.targets_bundle("DEFAULT"));
    }

    #[test]
    fn raw_yaml_wins_over_structured_model_config() {
        let model = ExportedModel {
            model_id: "m".into(),
            raw_yaml: Some("sie_id: raw\n".into()),
            model_config: Some(serde_json::json!({"sie_id": "json"})),
            affected_bundles: vec!["default".into()],
            pool: None,
        };
        assert_eq!(model.config_body().unwrap().as_deref(), Some("sie_id: raw"));
    }

    #[test]
    fn structured_model_config_falls_back_to_json_yaml_subset() {
        let model = ExportedModel {
            model_id: "m".into(),
            raw_yaml: None,
            model_config: Some(serde_json::json!({"sie_id": "json"})),
            affected_bundles: vec!["default".into()],
            pool: None,
        };
        assert_eq!(
            model.config_body().unwrap().as_deref(),
            Some("{\"sie_id\":\"json\"}")
        );
    }

    #[tokio::test]
    async fn export_reconcile_applies_target_bundle_and_updates_epoch_hash() {
        let snapshot = ExportSnapshot {
            epoch: 7,
            bundle_config_hashes: HashMap::new(),
            bundle_pool_config_hashes: HashMap::new(),
            models: vec![
                exported_model("target-model", &["default"]),
                exported_model("other-model", &["vision"]),
            ],
        };
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&["seed-model"]);
        let applied = Arc::new(Mutex::new(Vec::new()));

        let outcome = reconcile_export_snapshot(
            snapshot,
            scope("default", "default"),
            &state,
            &telemetry,
            reconcile_options("test"),
            {
                let applied = Arc::clone(&applied);
                move |req, _epoch| {
                    let applied = Arc::clone(&applied);
                    async move {
                        applied.lock().await.push(req);
                        Ok(Some(ReplaceModelConfigsResponse {
                            applied: true,
                            bundle_config_hash: "hash-7".into(),
                            config_version: 7,
                            applied_models: vec![
                                "target-model".into(),
                                "target-model:fast-profile".into(),
                            ],
                            applied_profiles: vec!["default".into(), "fast-profile".into()],
                        }))
                    }
                }
            },
        )
        .await;

        assert_eq!(outcome.applied, 2);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.skipped, 1);
        assert!(outcome.state_updated);
        assert_eq!(state.epoch(), 7);
        assert_eq!(
            state.bundle_config_hash().read().unwrap().as_str(),
            "hash-7"
        );
        assert!(state.loaded_models().read().unwrap().is_empty());
        assert!(telemetry.model_allowed_for_tests("target-model"));
        assert!(telemetry.model_allowed_for_tests("target-model:fast-profile"));
        assert!(telemetry.profile_allowed_for_tests("default"));
        assert!(telemetry.profile_allowed_for_tests("fast-profile"));
        assert!(!telemetry.model_allowed_for_tests("other-model"));
        assert!(!telemetry.model_allowed_for_tests("seed-model"));
        assert_eq!(telemetry.allowed_model_count_for_tests(), 1);
        assert_eq!(telemetry.allowed_pair_count_for_tests(), 2);

        let applied = applied.lock().await;
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0].bundle_id, "default");
        assert_eq!(applied[0].epoch, 7);
        assert_eq!(applied[0].models.len(), 1);
        assert_eq!(applied[0].models[0].model_id, "target-model");
    }

    #[tokio::test]
    async fn export_reconcile_filters_pool_and_uses_pool_hash() {
        let snapshot = ExportSnapshot {
            epoch: 13,
            bundle_config_hashes: HashMap::from([(
                "candle".to_string(),
                "global-hash".to_string(),
            )]),
            bundle_pool_config_hashes: HashMap::from([(
                "candle".to_string(),
                HashMap::from([
                    ("default".to_string(), "default-pool-hash".to_string()),
                    ("customer-a".to_string(), "tenant-pool-hash".to_string()),
                ]),
            )]),
            models: vec![
                exported_model("default-model", &["candle"]),
                exported_model_with_pool("tenant-model", &["candle"], "customer-a"),
            ],
        };
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&[]);
        let applied = Arc::new(Mutex::new(Vec::new()));

        let outcome = reconcile_export_snapshot(
            snapshot,
            scope("candle", "default"),
            &state,
            &telemetry,
            reconcile_options("test"),
            {
                let applied = Arc::clone(&applied);
                move |req, _epoch| {
                    let applied = Arc::clone(&applied);
                    async move {
                        applied.lock().await.push(req);
                        Ok(Some(ReplaceModelConfigsResponse {
                            applied: true,
                            bundle_config_hash: "default-pool-hash".into(),
                            config_version: 13,
                            applied_models: vec!["default-model".into()],
                            applied_profiles: vec!["default".into()],
                        }))
                    }
                }
            },
        )
        .await;

        assert_eq!(outcome.applied, 1);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.skipped, 1);
        assert!(outcome.state_updated);
        assert_eq!(
            state.bundle_config_hash().read().unwrap().as_str(),
            "default-pool-hash"
        );
        assert!(state.loaded_models().read().unwrap().is_empty());

        let applied = applied.lock().await;
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0].bundle_id, "candle");
        assert_eq!(applied[0].bundle_config_hash, "default-pool-hash");
        assert_eq!(applied[0].models.len(), 1);
        assert_eq!(applied[0].models[0].model_id, "default-model");
    }

    #[tokio::test]
    async fn export_reconcile_partial_failure_does_not_advance_state() {
        let snapshot = ExportSnapshot {
            epoch: 8,
            bundle_config_hashes: HashMap::new(),
            bundle_pool_config_hashes: HashMap::new(),
            models: vec![
                exported_model("ok-model", &["default"]),
                exported_model("bad-model", &["default"]),
            ],
        };
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&["seed-model"]);

        let outcome = reconcile_export_snapshot(
            snapshot,
            scope("default", "default"),
            &state,
            &telemetry,
            reconcile_options("test"),
            |req, _epoch| async move {
                if req.models.iter().any(|model| model.model_id == "bad-model") {
                    Err(IpcError::Server("apply failed".into()))
                } else {
                    Ok(Some(ReplaceModelConfigsResponse {
                        applied: true,
                        bundle_config_hash: "hash-8".into(),
                        config_version: 8,
                        applied_models: req
                            .models
                            .into_iter()
                            .map(|model| model.model_id)
                            .collect(),
                        applied_profiles: vec!["default".into()],
                    }))
                }
            },
        )
        .await;

        assert_eq!(outcome.applied, 0);
        assert_eq!(outcome.failed, 1);
        assert!(!outcome.state_updated);
        assert_eq!(state.epoch(), 0);
        assert_eq!(state.bundle_config_hash().read().unwrap().as_str(), "old");
        assert!(telemetry.model_allowed_for_tests("seed-model"));
        assert!(!telemetry.model_allowed_for_tests("ok-model"));
        assert!(!telemetry.model_allowed_for_tests("bad-model"));
    }

    #[tokio::test]
    async fn export_reconcile_rejected_replace_does_not_change_telemetry_catalog() {
        let snapshot = ExportSnapshot {
            epoch: 8,
            bundle_config_hashes: HashMap::new(),
            bundle_pool_config_hashes: HashMap::new(),
            models: vec![exported_model("target-model", &["default"])],
        };
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&["seed-model"]);

        let outcome = reconcile_export_snapshot(
            snapshot,
            scope("default", "default"),
            &state,
            &telemetry,
            reconcile_options("test"),
            |_req, _epoch| async move {
                Ok(Some(ReplaceModelConfigsResponse {
                    applied: false,
                    bundle_config_hash: "hash-8".into(),
                    config_version: 8,
                    applied_models: vec!["target-model".into()],
                    applied_profiles: vec!["default".into()],
                }))
            },
        )
        .await;

        assert_eq!(outcome.failed, 1);
        assert!(!outcome.state_updated);
        assert!(telemetry.model_allowed_for_tests("seed-model"));
        assert!(!telemetry.model_allowed_for_tests("target-model"));
    }

    #[tokio::test]
    async fn export_reconcile_forced_epoch_rewind_only_after_success() {
        let snapshot = ExportSnapshot {
            epoch: 3,
            bundle_config_hashes: HashMap::new(),
            bundle_pool_config_hashes: HashMap::new(),
            models: vec![exported_model("target-model", &["default"])],
        };
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&[]);
        state.force_epoch(9);

        let outcome = reconcile_export_snapshot(
            snapshot,
            scope("default", "default"),
            &state,
            &telemetry,
            ExportReconcileOptions {
                force_epoch: true,
                ..reconcile_options("test")
            },
            |_req, _epoch| async move {
                Ok(Some(ReplaceModelConfigsResponse {
                    applied: true,
                    bundle_config_hash: "hash-3".into(),
                    config_version: 3,
                    applied_models: vec!["target-model".into()],
                    applied_profiles: vec!["default".into()],
                }))
            },
        )
        .await;

        assert_eq!(outcome.applied, 1);
        assert_eq!(outcome.failed, 0);
        assert!(outcome.state_updated);
        assert_eq!(state.epoch(), 3);
        assert_eq!(
            state.bundle_config_hash().read().unwrap().as_str(),
            "hash-3"
        );
    }

    #[tokio::test]
    async fn export_reconcile_rejects_control_plane_worker_hash_mismatch() {
        let snapshot = ExportSnapshot {
            epoch: 11,
            bundle_config_hashes: HashMap::from([(
                "default".to_string(),
                "control-hash".to_string(),
            )]),
            bundle_pool_config_hashes: HashMap::new(),
            models: vec![exported_model("target-model", &["default"])],
        };
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&[]);

        let outcome = reconcile_export_snapshot(
            snapshot,
            scope("default", "default"),
            &state,
            &telemetry,
            reconcile_options("test"),
            |_req, _epoch| async move {
                Ok(Some(ReplaceModelConfigsResponse {
                    applied: true,
                    bundle_config_hash: "python-hash".into(),
                    config_version: 11,
                    applied_models: vec!["target-model".into()],
                    applied_profiles: vec!["default".into()],
                }))
            },
        )
        .await;

        assert_eq!(outcome.applied, 1);
        assert_eq!(outcome.failed, 1);
        assert!(!outcome.state_updated);
        assert!(outcome.export_signature.is_none());
        assert_eq!(state.epoch(), 0);
        assert_eq!(state.bundle_config_hash().read().unwrap().as_str(), "old");
    }

    #[tokio::test]
    async fn export_reconcile_replaces_with_empty_bundle_snapshot() {
        let snapshot = ExportSnapshot {
            epoch: 12,
            bundle_config_hashes: HashMap::from([("default".to_string(), String::new())]),
            bundle_pool_config_hashes: HashMap::new(),
            models: vec![exported_model("other-model", &["vision"])],
        };
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&["seed-model"]);
        let replace_calls = Arc::new(Mutex::new(Vec::new()));

        let outcome = reconcile_export_snapshot(
            snapshot,
            scope("default", "default"),
            &state,
            &telemetry,
            reconcile_options("test"),
            {
                let replace_calls = Arc::clone(&replace_calls);
                move |req, _epoch| {
                    let replace_calls = Arc::clone(&replace_calls);
                    async move {
                        replace_calls.lock().await.push(req);
                        Ok(Some(ReplaceModelConfigsResponse {
                            applied: true,
                            bundle_config_hash: String::new(),
                            config_version: 12,
                            applied_models: vec![],
                            applied_profiles: vec![],
                        }))
                    }
                }
            },
        )
        .await;

        assert_eq!(outcome.applied, 0);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.skipped, 1);
        assert!(outcome.state_updated);
        assert_eq!(state.bundle_config_hash().read().unwrap().as_str(), "");
        assert!(state.loaded_models().read().unwrap().is_empty());
        assert_eq!(telemetry.allowed_model_count_for_tests(), 0);

        let replace_calls = replace_calls.lock().await;
        assert_eq!(replace_calls.len(), 1);
        assert!(replace_calls[0].models.is_empty());
    }

    #[tokio::test]
    async fn export_reconcile_skips_unchanged_periodic_snapshot_before_ipc() {
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&[]);
        let replace_calls = Arc::new(Mutex::new(Vec::new()));

        let initial = reconcile_export_snapshot(
            ExportSnapshot {
                epoch: 0,
                bundle_config_hashes: HashMap::from([(
                    "default".to_string(),
                    "control-hash".to_string(),
                )]),
                bundle_pool_config_hashes: HashMap::new(),
                models: vec![exported_model("target-model", &["default"])],
            },
            scope("default", "default"),
            &state,
            &telemetry,
            reconcile_options("startup"),
            {
                let replace_calls = Arc::clone(&replace_calls);
                move |req, _epoch| {
                    let replace_calls = Arc::clone(&replace_calls);
                    async move {
                        replace_calls.lock().await.push(req);
                        Ok(Some(ReplaceModelConfigsResponse {
                            applied: true,
                            bundle_config_hash: "control-hash".into(),
                            config_version: 1,
                            applied_models: vec!["target-model".into()],
                            applied_profiles: vec!["default".into()],
                        }))
                    }
                }
            },
        )
        .await;
        let signature = initial.export_signature.expect("initial signature");

        let unchanged = reconcile_export_snapshot(
            ExportSnapshot {
                epoch: 0,
                bundle_config_hashes: HashMap::from([(
                    "default".to_string(),
                    "control-hash".to_string(),
                )]),
                bundle_pool_config_hashes: HashMap::new(),
                models: vec![exported_model("target-model", &["default"])],
            },
            scope("default", "default"),
            &state,
            &telemetry,
            ExportReconcileOptions {
                previous_export_signature: Some(signature),
                ..reconcile_options("periodic_export")
            },
            {
                let replace_calls = Arc::clone(&replace_calls);
                move |req, _epoch| {
                    let replace_calls = Arc::clone(&replace_calls);
                    async move {
                        replace_calls.lock().await.push(req);
                        Ok(Some(ReplaceModelConfigsResponse {
                            applied: true,
                            bundle_config_hash: "control-hash".into(),
                            config_version: 2,
                            applied_models: vec!["target-model".into()],
                            applied_profiles: vec!["default".into()],
                        }))
                    }
                }
            },
        )
        .await;

        assert_eq!(unchanged.applied, 0);
        assert_eq!(unchanged.failed, 0);
        assert!(!unchanged.state_updated);
        assert_eq!(replace_calls.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn export_reconcile_applies_epoch_zero_periodic_snapshot_when_body_changes() {
        let state = ConfigApplyState::new("old".into());
        let telemetry = SidecarTelemetry::for_tests(&[]);
        let replace_calls = Arc::new(Mutex::new(Vec::new()));

        let initial = reconcile_export_snapshot(
            ExportSnapshot {
                epoch: 0,
                bundle_config_hashes: HashMap::from([(
                    "default".to_string(),
                    "stable-routing-hash".to_string(),
                )]),
                bundle_pool_config_hashes: HashMap::new(),
                models: vec![exported_model_with_yaml(
                    "target-model",
                    &["default"],
                    "sie_id: target-model\nprofiles:\n  default:\n    max_batch_tokens: 4096\n",
                )],
            },
            scope("default", "default"),
            &state,
            &telemetry,
            reconcile_options("startup"),
            {
                let replace_calls = Arc::clone(&replace_calls);
                move |req, _epoch| {
                    let replace_calls = Arc::clone(&replace_calls);
                    async move {
                        replace_calls.lock().await.push(req);
                        Ok(Some(ReplaceModelConfigsResponse {
                            applied: true,
                            bundle_config_hash: "stable-routing-hash".into(),
                            config_version: 1,
                            applied_models: vec!["target-model".into()],
                            applied_profiles: vec!["default".into()],
                        }))
                    }
                }
            },
        )
        .await;
        let signature = initial.export_signature.expect("initial signature");

        let changed = reconcile_export_snapshot(
            ExportSnapshot {
                epoch: 0,
                bundle_config_hashes: HashMap::from([(
                    "default".to_string(),
                    "stable-routing-hash".to_string(),
                )]),
                bundle_pool_config_hashes: HashMap::new(),
                models: vec![exported_model_with_yaml(
                    "target-model",
                    &["default"],
                    "sie_id: target-model\nprofiles:\n  default:\n    max_batch_tokens: 8192\n",
                )],
            },
            scope("default", "default"),
            &state,
            &telemetry,
            ExportReconcileOptions {
                previous_export_signature: Some(signature),
                ..reconcile_options("periodic_export")
            },
            {
                let replace_calls = Arc::clone(&replace_calls);
                move |req, _epoch| {
                    let replace_calls = Arc::clone(&replace_calls);
                    async move {
                        let applied_models = req
                            .models
                            .iter()
                            .map(|model| model.model_id.clone())
                            .collect();
                        replace_calls.lock().await.push(req);
                        Ok(Some(ReplaceModelConfigsResponse {
                            applied: true,
                            bundle_config_hash: "stable-routing-hash".into(),
                            config_version: 2,
                            applied_models,
                            applied_profiles: vec!["default".into()],
                        }))
                    }
                }
            },
        )
        .await;

        assert_eq!(changed.applied, 1);
        assert_eq!(changed.failed, 0);
        assert_eq!(replace_calls.lock().await.len(), 2);
    }
}
