use std::time::Duration;

use chrono::Utc;
use k8s_openapi::api::coordination::v1::Lease as K8sLease;
use k8s_openapi::api::coordination::v1::LeaseSpec;
use k8s_openapi::api::core::v1::ConfigMap;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{MicroTime, ObjectMeta};
use kube::api::{Api, ListParams, Patch, PatchParams};
use kube::Client;
use rand::Rng;
use tracing::{debug, info, warn};

use crate::types::pool::{Pool, PoolStatus};

pub(crate) const LABEL_MANAGED_BY: &str = "app.kubernetes.io/managed-by";
pub(crate) const LABEL_MANAGED_BY_VALUE: &str = "sie-gateway";
pub(crate) const LABEL_POOL: &str = "sie-gateway/pool";
pub(crate) const DATA_KEY: &str = "pool.json";
const LEASE_PREFIX: &str = "sie-pool-lease-";

const CAS_MAX_RETRIES: u32 = 10;
const CAS_BASE_DELAY_MS: f64 = 100.0;
const CAS_BACKOFF_FACTOR: f64 = 1.5;
const CAS_MAX_DELAY_MS: f64 = 5000.0;
const CAS_JITTER_FRACTION: f64 = 0.2;

/// Kubernetes-backed pool storage using ConfigMaps with optimistic concurrency.
/// Each pool is stored as a ConfigMap with the pool spec and status serialized as JSON.
/// Optimistic concurrency is enforced via `resourceVersion` on updates.
///
/// Also manages coordination.k8s.io/v1 Lease objects per pool for crash-safe TTL.
#[allow(dead_code)]
pub struct K8sPoolBackend {
    client: Client,
    namespace: String,
    holder_identity: String,
}

#[allow(dead_code)]
impl K8sPoolBackend {
    /// Returns a reference to the K8s client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Returns the namespace used for pool storage.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns the holder identity (gateway ID) for this backend.
    pub fn holder_identity(&self) -> &str {
        &self.holder_identity
    }

    pub async fn new(namespace: &str, holder_identity: &str) -> Result<Self, kube::Error> {
        let client = Client::try_default().await?;
        Ok(Self {
            client,
            namespace: namespace.to_string(),
            holder_identity: holder_identity.to_string(),
        })
    }

    pub fn with_client(client: Client, namespace: &str, holder_identity: &str) -> Self {
        Self {
            client,
            namespace: namespace.to_string(),
            holder_identity: holder_identity.to_string(),
        }
    }

    fn configmap_name(pool_name: &str) -> String {
        format!("sie-pool-{}", pool_name)
    }

    fn api(&self) -> Api<ConfigMap> {
        Api::namespaced(self.client.clone(), &self.namespace)
    }

    fn lease_api(&self) -> Api<K8sLease> {
        Api::namespaced(self.client.clone(), &self.namespace)
    }

    fn lease_name(pool_name: &str) -> String {
        format!("{}{}", LEASE_PREFIX, pool_name)
    }

    fn pool_labels(pool_name: &str) -> std::collections::BTreeMap<String, String> {
        let mut labels = std::collections::BTreeMap::new();
        labels.insert(
            LABEL_MANAGED_BY.to_string(),
            LABEL_MANAGED_BY_VALUE.to_string(),
        );
        labels.insert(LABEL_POOL.to_string(), pool_name.to_string());
        labels
    }

    /// Create or update a pool in Kubernetes.
    pub async fn save_pool(&self, pool: &Pool) -> Result<(), String> {
        let api = self.api();
        let cm_name = Self::configmap_name(&pool.spec.name);

        let pool_json =
            serde_json::to_string_pretty(pool).map_err(|e| format!("serialize pool: {}", e))?;

        let labels = Self::pool_labels(&pool.spec.name);

        let mut data = std::collections::BTreeMap::new();
        data.insert(DATA_KEY.to_string(), pool_json);

        let cm = ConfigMap {
            metadata: ObjectMeta {
                name: Some(cm_name.clone()),
                namespace: Some(self.namespace.clone()),
                labels: Some(labels),
                ..Default::default()
            },
            data: Some(data),
            ..Default::default()
        };

        // Use server-side apply for idempotent create-or-update
        let patch_params = PatchParams::apply("sie-gateway").force();
        api.patch(&cm_name, &patch_params, &Patch::Apply(cm))
            .await
            .map_err(|e| format!("save pool ConfigMap {}: {}", cm_name, e))?;

        debug!(pool = %pool.spec.name, configmap = %cm_name, "saved pool to K8s");
        Ok(())
    }

    /// Load a pool from Kubernetes.
    pub async fn load_pool(&self, name: &str) -> Result<Option<Pool>, String> {
        let api = self.api();
        let cm_name = Self::configmap_name(name);

        match api.get(&cm_name).await {
            Ok(cm) => {
                let data = cm
                    .data
                    .as_ref()
                    .and_then(|d| d.get(DATA_KEY))
                    .ok_or_else(|| format!("ConfigMap {} has no pool data", cm_name))?;

                let pool: Pool = serde_json::from_str(data)
                    .map_err(|e| format!("parse pool from ConfigMap {}: {}", cm_name, e))?;

                Ok(Some(pool))
            }
            Err(kube::Error::Api(err)) if err.code == 404 => Ok(None),
            Err(e) => Err(format!("get ConfigMap {}: {}", cm_name, e)),
        }
    }

    /// List all pools from Kubernetes.
    pub async fn list_pools(&self) -> Result<Vec<Pool>, String> {
        let api = self.api();
        let lp = ListParams::default()
            .labels(&format!("{}={}", LABEL_MANAGED_BY, LABEL_MANAGED_BY_VALUE));

        let cms = api
            .list(&lp)
            .await
            .map_err(|e| format!("list pool ConfigMaps: {}", e))?;

        let mut pools = Vec::new();
        for cm in cms.items {
            if let Some(data) = cm.data.as_ref().and_then(|d| d.get(DATA_KEY)) {
                match serde_json::from_str::<Pool>(data) {
                    Ok(pool) => pools.push(pool),
                    Err(e) => {
                        let name = cm.metadata.name.unwrap_or_default();
                        warn!(configmap = %name, error = %e, "failed to parse pool ConfigMap");
                    }
                }
            }
        }

        Ok(pools)
    }

    /// Delete a pool from Kubernetes.
    pub async fn delete_pool(&self, name: &str) -> Result<bool, String> {
        let api = self.api();
        let cm_name = Self::configmap_name(name);

        match api.delete(&cm_name, &Default::default()).await {
            Ok(_) => {
                info!(pool = %name, configmap = %cm_name, "deleted pool from K8s");
                Ok(true)
            }
            Err(kube::Error::Api(err)) if err.code == 404 => Ok(false),
            Err(e) => Err(format!("delete ConfigMap {}: {}", cm_name, e)),
        }
    }

    /// Update pool status with optimistic concurrency (resourceVersion check).
    /// Retries with exponential backoff on 409 Conflict (stale resourceVersion).
    pub async fn update_pool_status(&self, name: &str, status: &PoolStatus) -> Result<(), String> {
        let api = self.api();
        let cm_name = Self::configmap_name(name);
        let mut delay_ms = CAS_BASE_DELAY_MS;

        for attempt in 0..CAS_MAX_RETRIES {
            // Load current state (re-fetched each attempt to get latest resourceVersion)
            let cm = api
                .get(&cm_name)
                .await
                .map_err(|e| format!("get ConfigMap {}: {}", cm_name, e))?;

            let resource_version = cm
                .metadata
                .resource_version
                .ok_or_else(|| "ConfigMap missing resourceVersion".to_string())?;

            let data_str = cm
                .data
                .as_ref()
                .and_then(|d| d.get(DATA_KEY))
                .ok_or_else(|| format!("ConfigMap {} missing pool data", cm_name))?;

            let mut pool: Pool =
                serde_json::from_str(data_str).map_err(|e| format!("parse pool: {}", e))?;

            pool.status = status.clone();

            let pool_json = serde_json::to_string_pretty(&pool)
                .map_err(|e| format!("serialize pool: {}", e))?;

            let mut updated_data = std::collections::BTreeMap::new();
            updated_data.insert(DATA_KEY.to_string(), pool_json);

            let patch = serde_json::json!({
                "metadata": {
                    "resourceVersion": resource_version
                },
                "data": updated_data
            });

            match api
                .patch(&cm_name, &PatchParams::default(), &Patch::Merge(patch))
                .await
            {
                Ok(_) => {
                    if attempt > 0 {
                        debug!(
                            pool = %name,
                            attempts = attempt + 1,
                            "updated pool status in K8s after CAS retry"
                        );
                    } else {
                        debug!(pool = %name, "updated pool status in K8s");
                    }
                    return Ok(());
                }
                Err(kube::Error::Api(ref err)) if err.code == 409 => {
                    if attempt + 1 >= CAS_MAX_RETRIES {
                        return Err(format!(
                            "update ConfigMap {} failed after {} CAS retries (409 Conflict)",
                            cm_name, CAS_MAX_RETRIES
                        ));
                    }

                    // Apply jitter: multiply delay by a random factor in [1 - jitter, 1 + jitter)
                    let jitter_factor = {
                        let mut rng = rand::thread_rng();
                        rng.gen_range((1.0 - CAS_JITTER_FRACTION)..(1.0 + CAS_JITTER_FRACTION))
                    };
                    let sleep_ms = delay_ms * jitter_factor;

                    warn!(
                        pool = %name,
                        attempt = attempt + 1,
                        delay_ms = sleep_ms as u64,
                        "409 Conflict on pool status update, retrying"
                    );

                    tokio::time::sleep(Duration::from_secs_f64(sleep_ms / 1000.0)).await;

                    // Exponential backoff with cap
                    delay_ms = (delay_ms * CAS_BACKOFF_FACTOR).min(CAS_MAX_DELAY_MS);
                }
                Err(e) => {
                    return Err(format!(
                        "update ConfigMap {} (optimistic concurrency): {}",
                        cm_name, e
                    ));
                }
            }
        }

        // Should be unreachable due to the check inside the loop, but just in case
        Err(format!(
            "update ConfigMap {} failed after {} CAS retries",
            cm_name, CAS_MAX_RETRIES
        ))
    }

    // ── Lease CRUD (crash-safe pool TTL) ────────────────────────────

    /// Create or renew a Lease for the given pool. Uses server-side apply for
    /// idempotent create-or-update semantics.
    pub async fn create_or_renew_lease(
        &self,
        pool_name: &str,
        ttl_seconds: i32,
    ) -> Result<(), String> {
        let api = self.lease_api();
        let name = Self::lease_name(pool_name);
        let now = MicroTime(Utc::now());

        // Preserve the original acquireTime if the Lease already exists
        let acquire_time = match api.get(&name).await {
            Ok(existing) => existing
                .spec
                .and_then(|s| s.acquire_time)
                .unwrap_or_else(|| now.clone()),
            Err(_) => now.clone(),
        };

        let labels = Self::pool_labels(pool_name);

        let lease = K8sLease {
            metadata: ObjectMeta {
                name: Some(name.clone()),
                namespace: Some(self.namespace.clone()),
                labels: Some(labels),
                ..Default::default()
            },
            spec: Some(LeaseSpec {
                holder_identity: Some(self.holder_identity.clone()),
                lease_duration_seconds: Some(ttl_seconds),
                acquire_time: Some(acquire_time),
                renew_time: Some(now),
                ..Default::default()
            }),
        };

        let patch_params = PatchParams::apply("sie-gateway").force();
        api.patch(&name, &patch_params, &Patch::Apply(lease))
            .await
            .map_err(|e| format!("create/renew Lease {}: {}", name, e))?;

        debug!(pool = %pool_name, lease = %name, "created/renewed Lease in K8s");
        Ok(())
    }

    /// Renew an existing Lease by updating only the renewTime field.
    pub async fn renew_lease(&self, pool_name: &str) -> Result<(), String> {
        let api = self.lease_api();
        let name = Self::lease_name(pool_name);
        let now = MicroTime(Utc::now());

        let labels = Self::pool_labels(pool_name);

        // Server-side apply with full spec so renewTime is updated.
        // We must re-state holderIdentity and leaseDurationSeconds to keep them.
        // Load the existing lease to preserve acquireTime and leaseDurationSeconds.
        let existing = api
            .get(&name)
            .await
            .map_err(|e| format!("get Lease {} for renewal: {}", name, e))?;

        let existing_spec = existing.spec.unwrap_or_default();

        let lease = K8sLease {
            metadata: ObjectMeta {
                name: Some(name.clone()),
                namespace: Some(self.namespace.clone()),
                labels: Some(labels),
                ..Default::default()
            },
            spec: Some(LeaseSpec {
                holder_identity: Some(self.holder_identity.clone()),
                lease_duration_seconds: existing_spec.lease_duration_seconds,
                acquire_time: existing_spec.acquire_time,
                renew_time: Some(now),
                ..Default::default()
            }),
        };

        let patch_params = PatchParams::apply("sie-gateway").force();
        api.patch(&name, &patch_params, &Patch::Apply(lease))
            .await
            .map_err(|e| format!("renew Lease {}: {}", name, e))?;

        debug!(pool = %pool_name, lease = %name, "renewed Lease in K8s");
        Ok(())
    }

    /// Delete a Lease for the given pool. Returns true if deleted, false if not found.
    pub async fn delete_lease(&self, pool_name: &str) -> Result<bool, String> {
        let api = self.lease_api();
        let name = Self::lease_name(pool_name);

        match api.delete(&name, &Default::default()).await {
            Ok(_) => {
                info!(pool = %pool_name, lease = %name, "deleted Lease from K8s");
                Ok(true)
            }
            Err(kube::Error::Api(err)) if err.code == 404 => Ok(false),
            Err(e) => Err(format!("delete Lease {}: {}", name, e)),
        }
    }

    /// List all pool Leases managed by sie-gateway and return the pool names of
    /// expired ones (where `now - renewTime > leaseDurationSeconds`).
    pub async fn list_expired_leases(&self) -> Result<Vec<String>, String> {
        let api = self.lease_api();
        let lp = ListParams::default()
            .labels(&format!("{}={}", LABEL_MANAGED_BY, LABEL_MANAGED_BY_VALUE));

        let leases = api
            .list(&lp)
            .await
            .map_err(|e| format!("list Leases: {}", e))?;

        let now = Utc::now();
        let mut expired = Vec::new();

        for lease in leases.items {
            let pool_name = lease
                .metadata
                .labels
                .as_ref()
                .and_then(|l| l.get(LABEL_POOL))
                .cloned()
                .unwrap_or_default();

            if pool_name.is_empty() {
                continue;
            }

            let spec = match lease.spec {
                Some(s) => s,
                None => continue,
            };

            let duration_secs = match spec.lease_duration_seconds {
                Some(d) => d as i64,
                None => continue,
            };

            // Use renewTime, falling back to acquireTime
            let last_time = spec.renew_time.or(spec.acquire_time);

            let last_dt = match last_time {
                Some(MicroTime(dt)) => dt,
                None => continue,
            };

            let elapsed = now.signed_duration_since(last_dt).num_seconds();
            if elapsed > duration_secs {
                debug!(
                    pool = %pool_name,
                    elapsed_s = elapsed,
                    ttl_s = duration_secs,
                    "Lease expired"
                );
                expired.push(pool_name);
            }
        }

        Ok(expired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::pool::{PoolSpec, PoolState};
    use std::collections::HashMap;

    use http::{Request, Response};
    use kube::client::Body;
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::{Arc, Mutex};
    use tokio::time::Instant;

    #[test]
    fn test_configmap_name() {
        assert_eq!(
            K8sPoolBackend::configmap_name("default"),
            "sie-pool-default"
        );
        assert_eq!(
            K8sPoolBackend::configmap_name("eval-l4"),
            "sie-pool-eval-l4"
        );
    }

    #[test]
    fn test_pool_serialization_for_configmap() {
        let pool = Pool {
            spec: PoolSpec {
                name: "test".to_string(),
                bundle: Some("default".to_string()),
                gpus: {
                    let mut m = HashMap::new();
                    m.insert("l4-spot".to_string(), 2);
                    m
                },
                ttl_seconds: Some(600),
                minimum_worker_count: 2,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: 1000.0,
                last_renewed: 1000.0,
            },
        };

        let json = serde_json::to_string(&pool).unwrap();
        let parsed: Pool = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.spec.name, "test");
        assert_eq!(parsed.spec.gpus.get("l4-spot"), Some(&2));
        assert_eq!(parsed.status.state, PoolState::Pending);
    }

    #[test]
    fn test_label_constants() {
        assert_eq!(LABEL_MANAGED_BY, "app.kubernetes.io/managed-by");
        assert_eq!(LABEL_MANAGED_BY_VALUE, "sie-gateway");
        assert_eq!(LABEL_POOL, "sie-gateway/pool");
    }

    #[test]
    fn test_lease_name() {
        assert_eq!(
            K8sPoolBackend::lease_name("default"),
            "sie-pool-lease-default"
        );
        assert_eq!(
            K8sPoolBackend::lease_name("eval-l4"),
            "sie-pool-lease-eval-l4"
        );
    }

    #[test]
    fn test_cas_retry_constants() {
        // Verify the retry parameters match the specification (PRD S9.3)
        assert_eq!(CAS_MAX_RETRIES, 10);
        assert!((CAS_BASE_DELAY_MS - 100.0).abs() < f64::EPSILON);
        assert!((CAS_BACKOFF_FACTOR - 1.5).abs() < f64::EPSILON);
        assert!((CAS_MAX_DELAY_MS - 5000.0).abs() < f64::EPSILON);
        assert!((CAS_JITTER_FRACTION - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cas_backoff_sequence() {
        // Verify the exponential backoff sequence (without jitter) caps at 5s
        let mut delay = CAS_BASE_DELAY_MS;
        let mut delays = vec![delay];
        for _ in 1..CAS_MAX_RETRIES {
            delay = (delay * CAS_BACKOFF_FACTOR).min(CAS_MAX_DELAY_MS);
            delays.push(delay);
        }
        // First delay should be 100ms
        assert!((delays[0] - 100.0).abs() < f64::EPSILON);
        // Second delay should be 150ms
        assert!((delays[1] - 150.0).abs() < f64::EPSILON);
        // All delays should be <= 5000ms
        for d in &delays {
            assert!(*d <= CAS_MAX_DELAY_MS);
        }
        // Verify delays increase geometrically: 100, 150, 225, 337.5, ...
        assert!((delays[9] - 3844.3359375).abs() < 0.01);
    }

    #[test]
    fn test_cas_jitter_range() {
        // Verify jitter produces values in expected range
        let base = 1000.0;
        let low = base * (1.0 - CAS_JITTER_FRACTION);
        let high = base * (1.0 + CAS_JITTER_FRACTION);
        assert!((low - 800.0).abs() < f64::EPSILON);
        assert!((high - 1200.0).abs() < f64::EPSILON);
    }

    // ── Integration test helpers ─────────────────────────────────────

    fn make_test_pool(name: &str) -> Pool {
        Pool {
            spec: PoolSpec {
                name: name.to_string(),
                bundle: Some("default".to_string()),
                gpus: {
                    let mut m = HashMap::new();
                    m.insert("l4-spot".to_string(), 1);
                    m
                },
                ttl_seconds: Some(300),
                minimum_worker_count: 1,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: 1000.0,
                last_renewed: 1000.0,
            },
        }
    }

    fn make_configmap_response(pool: &Pool, resource_version: &str) -> Vec<u8> {
        let pool_json = serde_json::to_string_pretty(pool).unwrap();
        let mut data = BTreeMap::new();
        data.insert(DATA_KEY.to_string(), pool_json);
        let cm = serde_json::json!({
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": format!("sie-pool-{}", pool.spec.name),
                "namespace": "test-ns",
                "resourceVersion": resource_version
            },
            "data": data
        });
        serde_json::to_vec(&cm).unwrap()
    }

    fn make_error_response(code: u16, reason: &str, message: &str) -> Vec<u8> {
        let status = serde_json::json!({
            "kind": "Status",
            "apiVersion": "v1",
            "metadata": {},
            "status": "Failure",
            "message": message,
            "reason": reason,
            "code": code
        });
        serde_json::to_vec(&status).unwrap()
    }

    fn make_mock_client(responses: Vec<(u16, Vec<u8>)>) -> (Client, Arc<Mutex<usize>>) {
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let call_count = Arc::new(Mutex::new(0usize));
        let svc = tower::service_fn({
            let call_count = Arc::clone(&call_count);
            move |_req: Request<Body>| {
                let queue = Arc::clone(&queue);
                let call_count = Arc::clone(&call_count);
                async move {
                    *call_count.lock().unwrap() += 1;
                    let (status_code, body) = queue
                        .lock()
                        .unwrap()
                        .pop_front()
                        .expect("mock service: no more responses queued");
                    let resp = Response::builder()
                        .status(status_code)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap();
                    Ok::<_, std::convert::Infallible>(resp)
                }
            }
        });
        (Client::new(svc, "test-ns"), call_count)
    }

    // ── Integration tests for CAS retry loop ─────────────────────────

    #[tokio::test]
    async fn test_update_pool_status_first_attempt_success() {
        let pool = make_test_pool("mypool");
        let cm_body = make_configmap_response(&pool, "100");
        let patched_body = make_configmap_response(&pool, "101");

        let responses = vec![(200, cm_body), (200, patched_body)];
        let (client, _call_count) = make_mock_client(responses);
        let backend = K8sPoolBackend::with_client(client, "test-ns", "router-1");

        let new_status = PoolStatus {
            state: PoolState::Active,
            assigned_workers: Vec::new(),
            created_at: 1000.0,
            last_renewed: 2000.0,
        };

        let result = backend.update_pool_status("mypool", &new_status).await;
        assert!(result.is_ok(), "expected Ok, got: {:?}", result);
    }

    #[tokio::test]
    async fn test_update_pool_status_409_triggers_retry() {
        tokio::time::pause();

        let pool = make_test_pool("mypool");
        let cm_body_v1 = make_configmap_response(&pool, "100");
        let conflict_body = make_error_response(409, "Conflict", "the object has been modified");
        let cm_body_v2 = make_configmap_response(&pool, "200");
        let patched_body = make_configmap_response(&pool, "201");

        let responses = vec![
            (200, cm_body_v1),
            (409, conflict_body),
            (200, cm_body_v2),
            (200, patched_body),
        ];
        let (client, _call_count) = make_mock_client(responses);
        let backend = K8sPoolBackend::with_client(client, "test-ns", "router-1");

        let new_status = PoolStatus {
            state: PoolState::Active,
            assigned_workers: Vec::new(),
            created_at: 1000.0,
            last_renewed: 2000.0,
        };

        let result = backend.update_pool_status("mypool", &new_status).await;
        assert!(result.is_ok(), "expected Ok after retry, got: {:?}", result);
    }

    #[tokio::test]
    async fn test_update_pool_status_non_409_error_no_retry() {
        let pool = make_test_pool("mypool");
        let cm_body = make_configmap_response(&pool, "100");
        let error_body = make_error_response(500, "InternalError", "internal server error");

        let responses = vec![(200, cm_body), (500, error_body)];
        let (client, call_count) = make_mock_client(responses);
        let backend = K8sPoolBackend::with_client(client, "test-ns", "router-1");

        let new_status = PoolStatus::default();
        let result = backend.update_pool_status("mypool", &new_status).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("optimistic concurrency"),
            "expected 'optimistic concurrency' in error, got: {}",
            err
        );
        // Verify no retry: exactly 2 requests (1 GET + 1 PATCH)
        assert_eq!(
            *call_count.lock().unwrap(),
            2,
            "expected exactly 2 requests (no retry)"
        );
    }

    #[tokio::test]
    async fn test_update_pool_status_retry_exhaustion() {
        tokio::time::pause();

        let pool = make_test_pool("mypool");
        let conflict_body = make_error_response(409, "Conflict", "the object has been modified");

        let mut responses = Vec::new();
        for i in 0..CAS_MAX_RETRIES {
            let cm_body = make_configmap_response(&pool, &format!("{}", 100 + i));
            responses.push((200, cm_body));
            responses.push((409, conflict_body.clone()));
        }

        let (client, _call_count) = make_mock_client(responses);
        let backend = K8sPoolBackend::with_client(client, "test-ns", "router-1");

        let new_status = PoolStatus::default();
        let result = backend.update_pool_status("mypool", &new_status).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("failed after 10 CAS retries"),
            "expected retry exhaustion message, got: {}",
            err
        );
        assert!(
            err.contains("409 Conflict"),
            "expected '409 Conflict' in error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_update_pool_status_backoff_increases() {
        tokio::time::pause();

        let pool = make_test_pool("mypool");
        let conflict_body = make_error_response(409, "Conflict", "the object has been modified");

        let mut responses = Vec::new();
        for i in 0..3u32 {
            let cm_body = make_configmap_response(&pool, &format!("{}", 100 + i));
            responses.push((200, cm_body));
            responses.push((409, conflict_body.clone()));
        }
        let cm_body = make_configmap_response(&pool, "103");
        let patched_body = make_configmap_response(&pool, "104");
        responses.push((200, cm_body));
        responses.push((200, patched_body));

        let (client, _call_count) = make_mock_client(responses);
        let backend = K8sPoolBackend::with_client(client, "test-ns", "router-1");

        let new_status = PoolStatus::default();
        let start = Instant::now();
        let result = backend.update_pool_status("mypool", &new_status).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "expected Ok, got: {:?}", result);

        // With paused time, the total elapsed time should reflect the sum of
        // backoff delays: ~100ms + ~150ms + ~225ms = ~475ms
        // With jitter factor [0.8, 1.2], the min total is 80+120+180=380ms
        // and the max total is 120+180+270=570ms
        let elapsed_ms = elapsed.as_millis();
        assert!(
            (380..=570).contains(&elapsed_ms),
            "expected total backoff between 380ms and 570ms, got {elapsed_ms}ms"
        );
    }

    #[tokio::test]
    async fn test_update_pool_status_get_failure_no_retry() {
        let error_body =
            make_error_response(404, "NotFound", "configmaps \"sie-pool-noexist\" not found");

        let responses = vec![(404, error_body)];
        let (client, call_count) = make_mock_client(responses);
        let backend = K8sPoolBackend::with_client(client, "test-ns", "router-1");

        let new_status = PoolStatus::default();
        let result = backend.update_pool_status("noexist", &new_status).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("get ConfigMap"),
            "expected 'get ConfigMap' in error, got: {}",
            err
        );
        // Verify no retry: exactly 1 request (GET only)
        assert_eq!(
            *call_count.lock().unwrap(),
            1,
            "expected exactly 1 request (no retry)"
        );
    }
}
