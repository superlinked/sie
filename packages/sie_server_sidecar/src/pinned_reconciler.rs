//! Worker-side pinned-model reconciler.
//!
//! The gateway owns the pinned-model set per logical pool (`PoolSpec.pinned_models`,
//! set via the `/v1/pools` API or `SIE_GATEWAY_STATIC_QUEUE_POOLS`). This
//! module polls that source of truth on a timer, computes the union of pins for
//! logical pools that currently assign this worker, and pushes it to the backend
//! over IPC on change and after the backend process restarts, so the worker
//! eager-loads pinned models and excludes them from eviction.
//!
//! It is deliberately decoupled from the pool admission gate: a deployment may
//! run with `SIE_POOL_ADMISSION_ENABLED=false` (or use the default,
//! fail-open pool) and still want pinned models, so this reconciler runs
//! whenever a gateway URL is configured, independent of admission.
//!
//! There is no gateway push channel for pool changes, so delivery is poll-based
//! (mirroring `config_reconciler`). The backend `set_pinned_models` mutator is
//! idempotent, so a redundant push is harmless; we still suppress no-op pushes
//! via change detection.

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, info, warn};

use crate::backend::AdapterWorkerPool;
use crate::config::WorkerConfig;
use crate::shutdown::Shutdown;

const HTTP_TIMEOUT: Duration = Duration::from_secs(5);

struct PinnedReconciler {
    pool_name: String,
    worker_id: String,
    gateway_url: String,
    api_key: Option<String>,
    poll_interval: Duration,
    http: reqwest::Client,
    worker_pool: Arc<AdapterWorkerPool>,
    /// Last set pushed to the worker (sorted), or `None` before the first
    /// successful push. Starts `None` so the first authoritative fetch always
    /// pushes — even an empty set — replacing any deploy-time `SIE_PINNED_MODELS`
    /// bootstrap baseline.
    last_pushed: Option<Vec<String>>,
    /// Pool-side child-placement revision observed on the last successful push.
    /// The pinned set can stay identical while a pinned model moves to a new
    /// child after a child failure; in that case the per-child pin assignment must
    /// be pushed again.
    last_pushed_assignment_revision: u64,
    /// Whether the previous fetch failed, so we warn once on the transition into
    /// a failing state (a wedged gateway URL / api key) instead of only emitting
    /// debug logs that hide a never-converging reconciler.
    last_fetch_failed: bool,
}

impl PinnedReconciler {
    fn from_worker_config(
        config: &WorkerConfig,
        worker_pool: Arc<AdapterWorkerPool>,
    ) -> anyhow::Result<Option<Self>> {
        let Some(gateway_url) = config.gateway_url.as_deref().map(str::trim) else {
            debug!("pinned-reconciler: disabled because SIE_GATEWAY_URL is unset");
            return Ok(None);
        };
        if gateway_url.is_empty() {
            debug!("pinned-reconciler: disabled because SIE_GATEWAY_URL is empty");
            return Ok(None);
        }
        let http = reqwest::Client::builder().timeout(HTTP_TIMEOUT).build()?;
        Ok(Some(Self {
            pool_name: config.pool.clone(),
            worker_id: config.worker_id.clone(),
            gateway_url: gateway_url.trim_end_matches('/').to_string(),
            api_key: config.gateway_api_key.clone(),
            // Reuse the admission poll cadence. Decoupled from admission
            // otherwise (runs even when the gate is disabled); split into its
            // own field only if the cadences ever need to diverge.
            poll_interval: Duration::from_millis(config.pool_admission_check_interval_ms),
            http,
            worker_pool,
            last_pushed: None,
            last_pushed_assignment_revision: 0,
            last_fetch_failed: false,
        }))
    }

    async fn run(mut self, shutdown: Arc<Shutdown>) {
        info!(
            pool = %self.pool_name,
            worker_id = %self.worker_id,
            gateway = %self.gateway_url,
            poll_interval_ms = self.poll_interval.as_millis() as u64,
            "pinned-reconciler: started"
        );
        let mut ticker = interval(self.poll_interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // A restarted backend process has no pins. The restart also moves the
        // assignment revision, so this refresh pushes the set again at once
        // instead of at the next tick.
        let mut restarts = self.worker_pool.backend_restarts();
        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait() => return,
                Ok(()) = restarts.changed() => self.refresh().await,
                _ = ticker.tick() => self.refresh().await,
            }
        }
    }

    async fn refresh(&mut self) {
        let pools = match self.fetch_pools().await {
            // HTTP call succeeded. The pool list is authoritative, even when
            // empty: an empty assigned pin union clears any deploy-time
            // `SIE_PINNED_MODELS` bootstrap baseline.
            Ok(pools) => {
                self.last_fetch_failed = false;
                pools
            }
            // Network / auth / 5xx: keep the last set rather than wiping pins on
            // a glitch. Warn on the first failure so a wedged gateway URL or api
            // key (which would never converge) is visible, then drop to debug.
            Err(error) => {
                if self.last_fetch_failed {
                    debug!(pool = %self.pool_name, worker_id = %self.worker_id, error = %error, "pinned-reconciler: fetch still failing; keeping current pinned set");
                } else {
                    warn!(pool = %self.pool_name, worker_id = %self.worker_id, error = %error, "pinned-reconciler: pool fetch failed; keeping current pinned set, will retry");
                }
                self.last_fetch_failed = true;
                return;
            }
        };

        let current = parse_assigned_pinned_models(&pools, &self.worker_id, &self.pool_name);
        if current.is_empty() {
            let candidate_count = pinned_pool_count_for_home_queue(&pools, &self.pool_name);
            if candidate_count > 0 {
                debug!(
                    pool = %self.pool_name,
                    worker_id = %self.worker_id,
                    candidate_count,
                    "pinned-reconciler: pinned pools exist on this queue but none assign this worker"
                );
            }
        }
        let assignment_revision = self.worker_pool.pinned_assignment_revision();
        if self.last_pushed.as_ref() == Some(&current)
            && self.last_pushed_assignment_revision == assignment_revision
        {
            return;
        }

        match self.worker_pool.set_pinned_models(current.clone()).await {
            Ok(resp) => {
                if !resp.applied {
                    warn!(
                        pool = %self.pool_name,
                        worker_id = %self.worker_id,
                        count = current.len(),
                        pinned_count = resp.pinned_count,
                        "pinned-reconciler: worker pool rejected pinned set; will retry"
                    );
                    return;
                }
                info!(
                    pool = %self.pool_name,
                    worker_id = %self.worker_id,
                    count = current.len(),
                    applied = resp.applied,
                    "pinned-reconciler: pushed pinned set to worker"
                );
                self.last_pushed = Some(current);
                self.last_pushed_assignment_revision = assignment_revision;
            }
            Err(error) => {
                // Leave `last_pushed` unchanged so the next tick retries.
                warn!(pool = %self.pool_name, worker_id = %self.worker_id, error = %error, "pinned-reconciler: failed to push pinned set; will retry");
            }
        }
    }

    async fn fetch_pools(&self) -> Result<Vec<Value>, String> {
        let url = format!("{}/v1/pools", self.gateway_url);
        let mut req = self.http.get(&url).header("accept", "application/json");
        if let Some(token) = self.api_key.as_deref().filter(|token| !token.is_empty()) {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!(
                "status {} from pool list endpoint: {}",
                status.as_u16(),
                body.chars().take(120).collect::<String>()
            ));
        }
        let body = resp.json::<Value>().await.map_err(|e| e.to_string())?;
        parse_pool_list(&body).ok_or_else(|| "pool list response missing 'pools' array".to_string())
    }
}

fn parse_pool_list(body: &Value) -> Option<Vec<Value>> {
    body.get("pools")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| body.as_array().cloned())
}

fn normalize_pool_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

fn pool_queue_matches(pool: &Value, home_pool: &str) -> bool {
    let spec = pool.get("spec");
    let queue_pool = spec
        .and_then(|spec| spec.get("queue_pool"))
        .and_then(Value::as_str)
        .or_else(|| {
            spec.and_then(|spec| spec.get("name"))
                .and_then(Value::as_str)
        })
        .unwrap_or("default");
    normalize_pool_name(queue_pool) == normalize_pool_name(home_pool)
}

fn pool_assigns_worker(pool: &Value, worker_id: &str) -> bool {
    pool.get("status")
        .and_then(|status| status.get("assigned_workers"))
        .and_then(Value::as_array)
        .is_some_and(|workers| {
            workers.iter().any(|worker| {
                worker
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| name == worker_id)
            })
        })
}

fn parse_assigned_pinned_models(pools: &[Value], worker_id: &str, home_pool: &str) -> Vec<String> {
    let mut out: Vec<String> = pools
        .iter()
        .filter(|pool| pool_queue_matches(pool, home_pool))
        .filter(|pool| pool_assigns_worker(pool, worker_id))
        .flat_map(parse_pinned_models)
        .collect();
    out.sort();
    out.dedup();
    out
}

fn pinned_pool_count_for_home_queue(pools: &[Value], home_pool: &str) -> usize {
    pools
        .iter()
        .filter(|pool| pool_queue_matches(pool, home_pool))
        .filter(|pool| !parse_pinned_models(pool).is_empty())
        .count()
}

/// Extract `spec.pinned_models` from a pool JSON body as a sorted, deduped,
/// non-empty-trimmed list. Missing spec / field / non-array all yield an empty
/// set (the worker un-pins everything). Sorting makes change detection stable.
///
/// Ids are shipped gateway-canonical (bare or `model:profile`); worker-side
/// normalization owns exact profile handling. Python preserves non-default
/// profiles while lowercasing ids, and native Rust workers preserve HuggingFace
/// base-id casing while folding only `:default`, so dedup here stays on the
/// gateway form.
fn parse_pinned_models(pool: &Value) -> Vec<String> {
    let mut out: Vec<String> = pool
        .get("spec")
        .and_then(|spec| spec.get("pinned_models"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out.dedup();
    out
}

/// Spawn the pinned-model reconciler. Returns `None` (no task) when no gateway
/// URL is configured, matching the standalone / no-gateway deployment shape
/// where pinned models come only from the `SIE_PINNED_MODELS` env baseline.
pub fn spawn(
    config: &WorkerConfig,
    worker_pool: Arc<AdapterWorkerPool>,
    shutdown: Arc<Shutdown>,
) -> anyhow::Result<Option<JoinHandle<()>>> {
    let Some(reconciler) = PinnedReconciler::from_worker_config(config, worker_pool)? else {
        return Ok(None);
    };
    Ok(Some(tokio::spawn(async move {
        reconciler.run(shutdown).await;
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_sorted_deduped_set() {
        let pool = json!({"spec": {"pinned_models": ["b/model", "a/model", "b/model"]}});
        assert_eq!(parse_pinned_models(&pool), vec!["a/model", "b/model"]);
    }

    #[test]
    fn missing_spec_or_field_is_empty() {
        assert_eq!(parse_pinned_models(&json!({})), Vec::<String>::new());
        assert_eq!(
            parse_pinned_models(&json!({"spec": {}})),
            Vec::<String>::new()
        );
        assert_eq!(
            parse_pinned_models(&json!({"spec": {"pinned_models": []}})),
            Vec::<String>::new()
        );
    }

    #[test]
    fn filters_non_strings_and_blanks() {
        let pool = json!({"spec": {"pinned_models": ["org/a", 7, "", "  ", "org/b:fast"]}});
        assert_eq!(parse_pinned_models(&pool), vec!["org/a", "org/b:fast"]);
    }

    #[test]
    fn profile_qualified_ids_pass_through_unchanged() {
        // Normalization (strip :profile, lowercase) is the worker's job; the
        // reconciler ships gateway-canonical ids verbatim.
        let pool = json!({"spec": {"pinned_models": ["Org/Model:Fast"]}});
        assert_eq!(parse_pinned_models(&pool), vec!["Org/Model:Fast"]);
    }

    #[test]
    fn assigned_pool_union_uses_worker_and_home_queue() {
        let pools = vec![
            json!({
                "spec": {
                    "name": "tenant-a",
                    "queue_pool": "default",
                    "pinned_models": ["BAAI/bge-m3", "intfloat/e5"]
                },
                "status": {
                    "assigned_workers": [{"name": "worker-1"}]
                }
            }),
            json!({
                "spec": {
                    "name": "tenant-b",
                    "queue_pool": "default",
                    "pinned_models": ["BAAI/bge-m3", "Qwen/Qwen3"]
                },
                "status": {
                    "assigned_workers": [{"name": "worker-1"}]
                }
            }),
            json!({
                "spec": {
                    "name": "tenant-c",
                    "queue_pool": "other",
                    "pinned_models": ["other/model"]
                },
                "status": {
                    "assigned_workers": [{"name": "worker-1"}]
                }
            }),
            json!({
                "spec": {
                    "name": "tenant-d",
                    "queue_pool": "default",
                    "pinned_models": ["unassigned/model"]
                },
                "status": {
                    "assigned_workers": [{"name": "worker-2"}]
                }
            }),
        ];

        assert_eq!(
            parse_assigned_pinned_models(&pools, "worker-1", "default"),
            vec!["BAAI/bge-m3", "Qwen/Qwen3", "intfloat/e5"]
        );
    }

    #[test]
    fn assigned_pool_union_normalizes_home_queue() {
        let pools = vec![json!({
            "spec": {
                "name": "tenant-a",
                "queue_pool": "default",
                "pinned_models": ["BAAI/bge-m3"]
            },
            "status": {
                "assigned_workers": [{"name": "worker-1"}]
            }
        })];

        assert_eq!(
            parse_assigned_pinned_models(&pools, "worker-1", " DEFAULT "),
            vec!["BAAI/bge-m3"]
        );
    }

    #[test]
    fn missing_queue_pool_falls_back_to_pool_name_for_legacy_static_pools() {
        let pools = vec![json!({
            "spec": {
                "name": "l4",
                "pinned_models": ["BAAI/bge-m3"]
            },
            "status": {
                "assigned_workers": [{"name": "worker-1"}]
            }
        })];

        assert_eq!(
            parse_assigned_pinned_models(&pools, "worker-1", "l4"),
            vec!["BAAI/bge-m3"]
        );
    }
}
