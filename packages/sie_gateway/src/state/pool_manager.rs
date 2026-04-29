use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::state::k8s_pool_backend::K8sPoolBackend;
use crate::types::pool::{AssignedWorker, Pool, PoolSpec, PoolState, PoolStatus};

pub const DEFAULT_POOL_NAME: &str = "default";
const DEFAULT_LEASE_DURATION_S: f64 = 1200.0; // 20 minutes

#[derive(Debug)]
pub struct DefaultPoolProtectedError;

impl std::fmt::Display for DefaultPoolProtectedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cannot delete the default pool '{}'", DEFAULT_POOL_NAME)
    }
}

impl std::error::Error for DefaultPoolProtectedError {}

#[derive(Debug)]
pub struct InvalidMachineProfileError {
    pub invalid_profiles: Vec<String>,
    pub valid_profiles: Vec<String>,
}

impl std::fmt::Display for InvalidMachineProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unknown machine profiles: {:?}. Valid profiles: {:?}",
            self.invalid_profiles, self.valid_profiles
        )
    }
}

impl std::error::Error for InvalidMachineProfileError {}

pub struct PoolManager {
    pools: RwLock<HashMap<String, Pool>>,
    lease_duration_s: f64,
    configured_profiles: Vec<String>,
    /// Optional K8s ConfigMap backend for pool persistence.
    k8s_backend: Option<Arc<K8sPoolBackend>>,
}

impl PoolManager {
    pub fn new(configured_profiles: Vec<String>) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            lease_duration_s: DEFAULT_LEASE_DURATION_S,
            configured_profiles,
            k8s_backend: None,
        }
    }

    /// Attach a K8s pool backend for persistent pool storage.
    #[allow(dead_code)]
    pub fn with_k8s_backend(mut self, backend: Arc<K8sPoolBackend>) -> Self {
        self.k8s_backend = Some(backend);
        self
    }

    /// Restore pools from the K8s backend on startup.
    pub async fn restore_from_k8s(&self) -> Result<usize, String> {
        let backend = match &self.k8s_backend {
            Some(b) => b,
            None => return Ok(0),
        };

        let k8s_pools = backend.list_pools().await?;
        let mut pools = self.pools.write().await;
        let mut count = 0;
        for pool in k8s_pools {
            if !pools.contains_key(&pool.spec.name) {
                info!(pool = %pool.spec.name, "restored pool from K8s");
                pools.insert(pool.spec.name.clone(), pool);
                count += 1;
            }
        }
        Ok(count)
    }

    fn now_secs() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// Compute the Lease TTL in whole seconds from the pool spec or global default.
    fn lease_ttl_seconds(pool: &Pool) -> i32 {
        pool.spec
            .ttl_seconds
            .map(|s| s.min(i32::MAX as u64) as i32)
            .unwrap_or((DEFAULT_LEASE_DURATION_S as u64).min(i32::MAX as u64) as i32)
    }

    pub async fn create_default_pool(&self) {
        if self.configured_profiles.is_empty() {
            info!("no machine profiles configured, skipping default pool creation");
            return;
        }

        let gpus: HashMap<String, u32> = self
            .configured_profiles
            .iter()
            .map(|p| (p.clone(), 999))
            .collect();

        match self
            .create_pool(DEFAULT_POOL_NAME, gpus, None, None, 0)
            .await
        {
            Ok(_) => {
                info!(
                    profiles = ?self.configured_profiles,
                    "created default pool"
                );
            }
            Err(e) => {
                warn!(error = %e, "failed to create default pool");
            }
        }
    }

    pub async fn create_pool(
        &self,
        name: &str,
        gpus: HashMap<String, u32>,
        bundle: Option<String>,
        ttl_seconds: Option<u64>,
        minimum_worker_count: u32,
    ) -> Result<Pool, Box<dyn std::error::Error + Send + Sync>> {
        // Validate profiles
        if !self.configured_profiles.is_empty() {
            let invalid: Vec<String> = gpus
                .keys()
                .filter(|k| {
                    !self
                        .configured_profiles
                        .iter()
                        .any(|p| p.eq_ignore_ascii_case(k))
                })
                .cloned()
                .collect();

            if !invalid.is_empty() {
                return Err(Box::new(InvalidMachineProfileError {
                    invalid_profiles: invalid,
                    valid_profiles: self.configured_profiles.clone(),
                }));
            }
        }

        let now = Self::now_secs();
        let mut pools = self.pools.write().await;

        // Idempotent: return existing pool if found
        if let Some(existing) = pools.get_mut(name) {
            existing.status.last_renewed = now;
            let result = existing.clone();
            drop(pools); // release write lock before K8s call

            // Renew the K8s Lease (best-effort, skip default pool)
            if name != DEFAULT_POOL_NAME {
                if let Some(ref backend) = self.k8s_backend {
                    let ttl = Self::lease_ttl_seconds(&result);
                    if let Err(e) = backend.create_or_renew_lease(name, ttl).await {
                        warn!(error = %e, pool = name, "failed to renew K8s Lease");
                    }
                }
            }

            crate::metrics::POOL_EVENTS
                .with_label_values(&["renewed"])
                .inc();
            return Ok(result);
        }

        let pool = Pool {
            spec: PoolSpec {
                name: name.to_string(),
                bundle,
                gpus,
                ttl_seconds,
                minimum_worker_count,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: now,
                last_renewed: now,
            },
        };

        pools.insert(name.to_string(), pool.clone());
        drop(pools); // release write lock before K8s call
        info!(pool = name, "created pool");
        crate::metrics::POOL_EVENTS
            .with_label_values(&["created"])
            .inc();

        // Persist to K8s backend (best-effort)
        if let Some(ref backend) = self.k8s_backend {
            if let Err(e) = backend.save_pool(&pool).await {
                warn!(error = %e, pool = name, "failed to persist pool to K8s");
            }

            // Create K8s Lease for crash-safe TTL (skip default pool)
            if name != DEFAULT_POOL_NAME {
                let ttl = Self::lease_ttl_seconds(&pool);
                if let Err(e) = backend.create_or_renew_lease(name, ttl).await {
                    warn!(error = %e, pool = name, "failed to create K8s Lease");
                }
            }
        }

        Ok(pool)
    }

    pub async fn get_pool(&self, name: &str) -> Option<Pool> {
        let pools = self.pools.read().await;
        pools.get(name).cloned()
    }

    pub async fn list_pools(&self) -> Vec<Pool> {
        let pools = self.pools.read().await;
        pools.values().cloned().collect()
    }

    pub async fn delete_pool(&self, name: &str) -> Result<bool, DefaultPoolProtectedError> {
        if name == DEFAULT_POOL_NAME {
            return Err(DefaultPoolProtectedError);
        }

        let mut pools = self.pools.write().await;
        let removed = pools.remove(name).is_some();
        drop(pools); // release write lock before K8s call

        if removed {
            info!(pool = name, "deleted pool");
            crate::metrics::POOL_EVENTS
                .with_label_values(&["deleted"])
                .inc();
            if let Some(ref backend) = self.k8s_backend {
                if let Err(e) = backend.delete_pool(name).await {
                    warn!(error = %e, pool = name, "failed to delete pool from K8s");
                }
                if let Err(e) = backend.delete_lease(name).await {
                    warn!(error = %e, pool = name, "failed to delete K8s Lease");
                }
            }
        }
        Ok(removed)
    }

    pub async fn renew_pool(&self, name: &str) -> bool {
        let found = {
            let mut pools = self.pools.write().await;
            if let Some(pool) = pools.get_mut(name) {
                pool.status.last_renewed = Self::now_secs();
                true
            } else {
                false
            }
        }; // write lock dropped here

        // Renew the K8s Lease (best-effort, skip default pool)
        if found && name != DEFAULT_POOL_NAME {
            if let Some(ref backend) = self.k8s_backend {
                if let Err(e) = backend.renew_lease(name).await {
                    warn!(error = %e, pool = name, "failed to renew K8s Lease");
                }
            }
        }

        if found {
            crate::metrics::POOL_EVENTS
                .with_label_values(&["renewed"])
                .inc();
        }
        found
    }

    pub async fn assign_workers(
        &self,
        pool_name: &str,
        available_workers: &[(String, String, String, String)], // (name, url, gpu, bundle)
    ) -> bool {
        let mut pools = self.pools.write().await;
        let pool = match pools.get_mut(pool_name) {
            Some(p) => p,
            None => return false,
        };

        let filtered: Vec<&(String, String, String, String)> =
            if let Some(ref bundle_filter) = pool.spec.bundle {
                available_workers
                    .iter()
                    .filter(|(_, _, _, b)| b == bundle_filter)
                    .collect()
            } else {
                available_workers.iter().collect()
            };

        // Group by GPU type (lowercase)
        let mut workers_by_gpu: HashMap<String, Vec<&(String, String, String, String)>> =
            HashMap::new();
        for w in &filtered {
            workers_by_gpu
                .entry(w.2.to_lowercase())
                .or_default()
                .push(w);
        }

        let mut assigned: Vec<AssignedWorker> = Vec::new();
        let mut all_met = true;

        for (gpu_type, count_needed) in &pool.spec.gpus {
            let gpu_lower = gpu_type.to_lowercase();
            let available = workers_by_gpu.get_mut(&gpu_lower);

            let count = *count_needed as usize;
            match available {
                Some(workers) if workers.len() >= count => {
                    for w in workers.drain(..count) {
                        assigned.push(AssignedWorker {
                            name: w.0.clone(),
                            url: w.1.clone(),
                            gpu: w.2.clone(),
                        });
                    }
                }
                Some(workers) => {
                    all_met = false;
                    let take = workers.len();
                    for w in workers.drain(..take) {
                        assigned.push(AssignedWorker {
                            name: w.0.clone(),
                            url: w.1.clone(),
                            gpu: w.2.clone(),
                        });
                    }
                }
                None => {
                    all_met = false;
                }
            }
        }

        pool.status.assigned_workers = assigned;
        let meets_minimum =
            pool.status.assigned_workers.len() as u32 >= pool.spec.minimum_worker_count;
        pool.status.state = if all_met && !pool.status.assigned_workers.is_empty() && meets_minimum
        {
            PoolState::Active
        } else {
            PoolState::Pending
        };

        all_met
    }

    #[allow(dead_code)]
    pub async fn get_all_assigned_urls(&self) -> HashSet<String> {
        let pools = self.pools.read().await;
        let mut urls = HashSet::new();
        for pool in pools.values() {
            if pool.status.state == PoolState::Active {
                for w in &pool.status.assigned_workers {
                    urls.insert(w.url.clone());
                }
            }
        }
        urls
    }

    /// Apply a pool received from a remote gateway (via K8s watch).
    /// Inserts the pool if it does not exist, or updates it if the incoming
    /// pool has a more recent `last_renewed` timestamp.
    /// Does NOT write back to K8s (the event already came from K8s).
    pub async fn apply_remote_pool(&self, pool: Pool) {
        let name = pool.spec.name.clone();

        // Skip the default pool -- each gateway manages its own default pool
        if name == DEFAULT_POOL_NAME {
            return;
        }

        let mut pools = self.pools.write().await;
        if let Some(existing) = pools.get(&name) {
            // Only update if the incoming pool is newer
            if pool.status.last_renewed <= existing.status.last_renewed {
                return;
            }
        }

        info!(pool = %name, "applied remote pool from K8s watch");
        pools.insert(name, pool);
    }

    /// Remove a pool that was deleted by a remote gateway (via K8s watch).
    /// Does NOT write back to K8s (the event already came from K8s).
    pub async fn remove_remote_pool(&self, name: &str) {
        // Never delete the default pool via watch events
        if name == DEFAULT_POOL_NAME {
            return;
        }

        let mut pools = self.pools.write().await;
        if pools.remove(name).is_some() {
            info!(pool = %name, "removed remote pool via K8s watch");
        }
    }

    pub async fn check_expired_leases(&self) -> Vec<String> {
        let now = Self::now_secs();
        let mut expired = Vec::new();

        {
            let pools = self.pools.read().await;
            for (name, pool) in pools.iter() {
                if name == DEFAULT_POOL_NAME {
                    continue;
                }
                let ttl = pool
                    .spec
                    .ttl_seconds
                    .map(|s| s as f64)
                    .unwrap_or(self.lease_duration_s);
                if now - pool.status.last_renewed > ttl {
                    expired.push(name.clone());
                }
            }
        }

        // Also check K8s Leases for pools that may have been abandoned by a
        // crashed gateway (their software TTL timer never fires).
        if let Some(ref backend) = self.k8s_backend {
            match backend.list_expired_leases().await {
                Ok(k8s_expired) => {
                    for name in k8s_expired {
                        if name == DEFAULT_POOL_NAME || expired.contains(&name) {
                            continue;
                        }
                        let pools = self.pools.read().await;
                        let in_memory = pools.contains_key(&name);
                        drop(pools);

                        if in_memory {
                            info!(pool = %name, "K8s Lease expired (crash-safe TTL)");
                            expired.push(name);
                        } else {
                            // Orphaned Lease from a crashed gateway -- clean up
                            // K8s resources without adding to the expired vector.
                            info!(pool = %name, "orphaned K8s Lease cleanup");
                            if let Err(e) = backend.delete_pool(&name).await {
                                warn!(pool = %name, error = %e, "failed to delete orphaned pool from K8s");
                            }
                            if let Err(e) = backend.delete_lease(&name).await {
                                warn!(pool = %name, error = %e, "failed to delete orphaned Lease from K8s");
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "failed to list expired K8s Leases");
                }
            }
        }

        for name in &expired {
            {
                let mut pools = self.pools.write().await;
                if let Some(pool) = pools.get_mut(name) {
                    pool.status.state = PoolState::Expired;
                }
                pools.remove(name);
            }
            // Persist deletion to K8s backend (ConfigMap + Lease)
            if let Some(ref backend) = self.k8s_backend {
                if let Err(e) = backend.delete_pool(name).await {
                    warn!(pool = %name, error = %e, "failed to delete expired pool from K8s");
                }
                if let Err(e) = backend.delete_lease(name).await {
                    warn!(pool = %name, error = %e, "failed to delete expired Lease from K8s");
                }
            }
            info!(pool = %name, "cleaned up expired pool");
            crate::metrics::POOL_EVENTS
                .with_label_values(&["expired"])
                .inc();
        }

        expired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);

        let pool = pm.create_pool("test", gpus, None, None, 0).await.unwrap();
        assert_eq!(pool.spec.name, "test");
        assert_eq!(pool.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_create_pool_idempotent() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);

        let pool1 = pm
            .create_pool("test", gpus.clone(), None, None, 0)
            .await
            .unwrap();
        let pool2 = pm.create_pool("test", gpus, None, None, 0).await.unwrap();
        assert_eq!(pool1.spec.name, pool2.spec.name);
    }

    #[tokio::test]
    async fn test_delete_default_pool_fails() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        let result = pm.delete_pool(DEFAULT_POOL_NAME).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let deleted = pm.delete_pool("test").await.unwrap();
        assert!(deleted);

        let pool = pm.get_pool("test").await;
        assert!(pool.is_none());
    }

    #[tokio::test]
    async fn test_assign_workers() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let workers = vec![
            (
                "w1".to_string(),
                "http://w1:8080".to_string(),
                "l4-spot".to_string(),
                "default".to_string(),
            ),
            (
                "w2".to_string(),
                "http://w2:8080".to_string(),
                "l4-spot".to_string(),
                "default".to_string(),
            ),
            (
                "w3".to_string(),
                "http://w3:8080".to_string(),
                "a100".to_string(),
                "default".to_string(),
            ),
        ];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 2);
    }

    #[tokio::test]
    async fn test_assign_workers_partial() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 3);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let workers = vec![(
            "w1".to_string(),
            "http://w1:8080".to_string(),
            "l4-spot".to_string(),
            "default".to_string(),
        )];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(!all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_minimum_worker_count_enforced() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        // Require at least 3 workers total
        pm.create_pool("test", gpus, None, None, 3).await.unwrap();

        // Provide only 1 matching worker — GPU requirement met but worker count too low
        let workers = vec![(
            "w1".to_string(),
            "http://w1:8080".to_string(),
            "l4-spot".to_string(),
            "default".to_string(),
        )];

        let all_met = pm.assign_workers("test", &workers).await;
        // GPU count met (1/1) but minimum_worker_count (3) not met → stays Pending
        assert!(all_met); // all GPU types met
        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Pending);
        assert_eq!(pool.status.assigned_workers.len(), 1);
    }

    #[tokio::test]
    async fn test_minimum_worker_count_met() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);
        // Require at least 2 workers
        pm.create_pool("test", gpus, None, None, 2).await.unwrap();

        let workers = vec![
            (
                "w1".to_string(),
                "http://w1:8080".to_string(),
                "l4-spot".to_string(),
                "default".to_string(),
            ),
            (
                "w2".to_string(),
                "http://w2:8080".to_string(),
                "l4-spot".to_string(),
                "default".to_string(),
            ),
        ];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);
        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 2);
    }

    #[tokio::test]
    async fn test_renew_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let renewed = pm.renew_pool("test").await;
        assert!(renewed);

        let not_renewed = pm.renew_pool("nonexistent").await;
        assert!(!not_renewed);
    }

    #[tokio::test]
    async fn test_invalid_machine_profile() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("invalid-gpu".to_string(), 1);

        let result = pm.create_pool("test", gpus, None, None, 0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_per_pool_ttl_respected() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);

        // Pool with short TTL (1 second)
        pm.create_pool("short-ttl", gpus.clone(), None, Some(1), 0)
            .await
            .unwrap();

        // Pool with no TTL (uses global default = 1200s)
        pm.create_pool("default-ttl", gpus, None, None, 0)
            .await
            .unwrap();

        // Backdate last_renewed so the short-TTL pool is expired
        {
            let mut pools = pm.pools.write().await;
            if let Some(p) = pools.get_mut("short-ttl") {
                p.status.last_renewed -= 5.0; // 5s ago, exceeds 1s TTL
            }
            if let Some(p) = pools.get_mut("default-ttl") {
                p.status.last_renewed -= 5.0; // 5s ago, well within 1200s TTL
            }
        }

        let expired = pm.check_expired_leases().await;
        assert!(expired.contains(&"short-ttl".to_string()));
        assert!(!expired.contains(&"default-ttl".to_string()));

        // short-ttl should be removed
        assert!(pm.get_pool("short-ttl").await.is_none());
        // default-ttl should still exist
        assert!(pm.get_pool("default-ttl").await.is_some());
    }

    #[tokio::test]
    async fn test_create_default_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string(), "a100-40gb".to_string()]);

        pm.create_default_pool().await;

        let pool = pm.get_pool(DEFAULT_POOL_NAME).await;
        assert!(pool.is_some());

        let pool = pool.unwrap();
        assert_eq!(pool.spec.gpus.len(), 2);
        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&999));
        assert_eq!(pool.spec.gpus.get("a100-40gb"), Some(&999));
    }

    #[tokio::test]
    async fn test_restore_from_k8s_no_backend() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        // No K8s backend → returns Ok(0)
        let count = pm.restore_from_k8s().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_k8s_backend_field_defaults_none() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        assert!(pm.k8s_backend.is_none());
    }

    #[tokio::test]
    async fn test_apply_remote_pool_skips_default() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        let original = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();

        let pool = Pool {
            spec: PoolSpec {
                name: DEFAULT_POOL_NAME.to_string(),
                bundle: None,
                gpus: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: 9999.0,
                last_renewed: 9999.0,
            },
        };

        pm.apply_remote_pool(pool).await;

        // Should still have the locally-created default pool, not the remote one
        let found = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();
        // The remote pool had empty gpus; the original default pool has l4-spot=999
        assert_eq!(found.spec.gpus.get("l4-spot"), Some(&999));
        assert_eq!(found.status.created_at, original.status.created_at);
    }

    #[tokio::test]
    async fn test_remove_remote_pool_skips_default() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        pm.remove_remote_pool(DEFAULT_POOL_NAME).await;

        assert!(pm.get_pool(DEFAULT_POOL_NAME).await.is_some());
    }

    #[tokio::test]
    async fn test_remove_remote_pool_nonexistent() {
        let pm = PoolManager::new(vec![]);

        // Should not panic or error
        pm.remove_remote_pool("nonexistent").await;
    }
}
