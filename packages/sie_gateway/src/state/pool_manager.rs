use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::state::k8s_pool_backend::K8sPoolBackend;
use crate::types::pool::{AssignedWorker, Pool, PoolSpec, PoolState, PoolStatus};

pub const DEFAULT_POOL_NAME: &str = "default";
const DEFAULT_LEASE_DURATION_S: f64 = 1200.0; // 20 minutes
const TIMESTAMP_TOLERANCE_S: f64 = 0.001;
type WorkerAssignment = (String, String, String, String, String);

#[derive(Debug)]
pub enum PoolDeletionProtectedError {
    Default,
    Static(String),
}

impl std::fmt::Display for PoolDeletionProtectedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolDeletionProtectedError::Default => {
                write!(f, "Cannot delete the default pool '{}'", DEFAULT_POOL_NAME)
            }
            PoolDeletionProtectedError::Static(name) => {
                write!(f, "Cannot delete static Helm queue pool '{}'", name)
            }
        }
    }
}

impl std::error::Error for PoolDeletionProtectedError {}

#[derive(Debug)]
pub struct DefaultPoolMutationError;

impl std::fmt::Display for DefaultPoolMutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cannot modify the default pool '{}'", DEFAULT_POOL_NAME)
    }
}

impl std::error::Error for DefaultPoolMutationError {}

#[derive(Debug)]
pub struct StaticPoolMutationError {
    pub name: String,
}

impl std::fmt::Display for StaticPoolMutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cannot modify static Helm queue pool '{}'", self.name)
    }
}

impl std::error::Error for StaticPoolMutationError {}

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

#[derive(Debug)]
pub struct InvalidPoolNameError {
    pub name: String,
}

impl std::fmt::Display for InvalidPoolNameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invalid pool name '{}': only [A-Za-z0-9_-] are allowed (max 128 chars)",
            self.name
        )
    }
}

impl std::error::Error for InvalidPoolNameError {}

#[derive(Debug)]
pub struct UnknownQueuePoolError {
    pub queue_pool: String,
}

impl std::fmt::Display for UnknownQueuePoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unknown backing queue_pool '{}': non-default queue pools must be declared under queueRouting.staticQueuePools",
            self.queue_pool
        )
    }
}

impl std::error::Error for UnknownQueuePoolError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolAdmissionStatus {
    pub cap: u32,
    pub assigned_count: usize,
}

/// Capacity-reconciler view of one logical pool. Deliberately excludes pinned
/// models, GPU cap maps, timestamps, and persistence metadata so the
/// five-second KEDA loop clones only the state that can affect capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityPoolSnapshot {
    pub name: String,
    pub queue_pool: String,
    pub bundle: String,
    pub machine_profiles: Vec<String>,
    pub minimum_worker_count: u32,
    pub state: PoolState,
    pub assigned_workers: Vec<AssignedWorker>,
}

impl CapacityPoolSnapshot {
    pub(crate) fn from_pool(pool: &Pool) -> Self {
        Self {
            name: pool.spec.name.clone(),
            queue_pool: normalize_queue_pool(&pool.spec.queue_pool),
            bundle: pool
                .spec
                .bundle
                .as_deref()
                .unwrap_or("default")
                .trim()
                .to_ascii_lowercase(),
            machine_profiles: machine_profiles_for_pool(pool),
            minimum_worker_count: pool.spec.minimum_worker_count,
            state: pool.status.state,
            assigned_workers: pool.status.assigned_workers.clone(),
        }
    }
}

pub struct PoolManager {
    pools: RwLock<HashMap<String, Pool>>,
    lease_duration_s: f64,
    configured_profiles: Vec<String>,
    static_pool_names: RwLock<HashSet<String>>,
    /// Optional K8s ConfigMap backend for pool persistence.
    k8s_backend: Option<Arc<K8sPoolBackend>>,
}

impl PoolManager {
    pub fn new(configured_profiles: Vec<String>) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            lease_duration_s: DEFAULT_LEASE_DURATION_S,
            configured_profiles,
            static_pool_names: RwLock::new(HashSet::new()),
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
        let static_pool_names = self.static_pool_names.read().await.clone();
        let mut pools = self.pools.write().await;
        let mut count = 0;
        for mut pool in k8s_pools {
            if !Self::should_restore_pool_from_k8s(&pool) {
                continue;
            }
            if let Err(e) = validate_pool_name(&pool.spec.name) {
                warn!(pool = %pool.spec.name, error = %e, "skipping invalid pool from K8s restore");
                continue;
            }
            pool.spec.name = normalize_pool_name(&pool.spec.name);
            pool.spec.queue_pool = normalize_queue_pool(&pool.spec.queue_pool);
            if let Err(e) = validate_pool_name(&pool.spec.queue_pool) {
                warn!(pool = %pool.spec.name, queue_pool = %pool.spec.queue_pool, error = %e, "skipping pool with invalid backing queue pool from K8s restore");
                continue;
            }
            if !known_queue_pool_from_names(&static_pool_names, &pool.spec.queue_pool) {
                warn!(
                    pool = %pool.spec.name,
                    queue_pool = %pool.spec.queue_pool,
                    "skipping pool with unknown backing queue pool from K8s restore"
                );
                continue;
            }
            if static_pool_names.contains(&normalize_pool_name(&pool.spec.name)) {
                continue;
            }
            if pool_key_for_name(&pools, &pool.spec.name).is_none() {
                info!(pool = %pool.spec.name, "restored pool from K8s");
                pools.insert(pool.spec.name.clone(), pool);
                count += 1;
            }
        }
        Ok(count)
    }

    fn should_restore_pool_from_k8s(pool: &Pool) -> bool {
        !pool.spec.name.eq_ignore_ascii_case(DEFAULT_POOL_NAME)
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

    fn normalize_gpus_and_caps(
        mut gpus: HashMap<String, u32>,
        gpu_caps: &HashMap<String, u32>,
    ) -> HashMap<String, u32> {
        for gpu_type in gpu_caps.keys() {
            if !gpus
                .keys()
                .any(|required_gpu| required_gpu.eq_ignore_ascii_case(gpu_type))
            {
                gpus.insert(gpu_type.clone(), 0);
            }
        }
        gpus
    }

    fn validate_pool_profiles(
        &self,
        gpus: &HashMap<String, u32>,
        gpu_caps: &HashMap<String, u32>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.configured_profiles.is_empty() {
            return Ok(());
        }

        let invalid: Vec<String> = gpus
            .keys()
            .chain(gpu_caps.keys())
            .filter(|k| {
                !self
                    .configured_profiles
                    .iter()
                    .any(|p| p.eq_ignore_ascii_case(k))
            })
            .cloned()
            .collect();

        if invalid.is_empty() {
            Ok(())
        } else {
            Err(Box::new(InvalidMachineProfileError {
                invalid_profiles: invalid,
                valid_profiles: self.configured_profiles.clone(),
            }))
        }
    }

    async fn is_static_pool(&self, name: &str) -> bool {
        self.static_pool_names
            .read()
            .await
            .contains(&normalize_pool_name(name))
    }

    async fn is_known_queue_pool(&self, name: &str) -> bool {
        let static_pool_names = self.static_pool_names.read().await;
        known_queue_pool_from_names(&static_pool_names, name)
    }

    pub async fn sync_static_pools(
        &self,
        specs: &[PoolSpec],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let mut normalized_specs = Vec::new();
        let mut desired_names = HashSet::new();
        for spec in specs {
            validate_pool_name(&spec.name)?;
            let name = normalize_pool_name(&spec.name);
            if name.is_empty() || name == DEFAULT_POOL_NAME {
                continue;
            }

            let gpus = Self::normalize_gpus_and_caps(spec.gpus.clone(), &spec.gpu_caps);
            self.validate_pool_profiles(&gpus, &spec.gpu_caps)?;
            let bundle = spec.bundle.as_ref().and_then(|bundle| {
                let bundle = bundle.trim().to_string();
                if bundle.is_empty() {
                    None
                } else {
                    Some(bundle)
                }
            });
            let static_spec = PoolSpec {
                name: name.clone(),
                queue_pool: name.clone(),
                bundle,
                gpus,
                gpu_caps: spec.gpu_caps.clone(),
                ttl_seconds: None,
                minimum_worker_count: spec.minimum_worker_count,
                pinned_models: spec.pinned_models.clone(),
            };
            if !desired_names.insert(static_spec.name.clone()) {
                return Err(format!("duplicate static queue pool '{}'", name).into());
            }
            normalized_specs.push(static_spec);
        }
        normalized_specs.sort_by(|a, b| a.name.cmp(&b.name));

        let previous_names = {
            let mut static_pool_names = self.static_pool_names.write().await;
            let previous_names = static_pool_names.clone();
            *static_pool_names = desired_names.clone();
            previous_names
        };

        let now = Self::now_secs();
        let mut pools = self.pools.write().await;
        for name in previous_names.difference(&desired_names) {
            if pools.remove(name).is_some() {
                info!(pool = %name, "removed static Helm queue pool");
            }
        }
        let shadowed_names: Vec<String> = pools
            .keys()
            .filter(|name| {
                let normalized = normalize_pool_name(name);
                desired_names.contains(&normalized) && **name != normalized
            })
            .cloned()
            .collect();
        for name in &shadowed_names {
            if pools.remove(name).is_some() {
                info!(
                    pool = %name,
                    static_pool = %normalize_pool_name(name),
                    "removed case-variant pool shadowed by static Helm queue pool"
                );
            }
        }

        let mut count = 0;
        for spec in normalized_specs {
            count += 1;
            if let Some(existing) = pools.get_mut(&spec.name) {
                if existing.spec != spec {
                    existing.spec = spec;
                    existing.status.state = PoolState::Pending;
                    existing.status.assigned_workers.clear();
                    existing.status.last_renewed = now;
                    info!(pool = %existing.spec.name, "updated static Helm queue pool");
                } else {
                    existing.status.last_renewed = now;
                }
                continue;
            }

            let name = spec.name.clone();
            pools.insert(
                name.clone(),
                Pool {
                    spec,
                    status: PoolStatus {
                        state: PoolState::Pending,
                        assigned_workers: Vec::new(),
                        created_at: now,
                        last_renewed: now,
                    },
                },
            );
            info!(pool = %name, "created static Helm queue pool");
        }
        drop(pools);

        if let Some(ref backend) = self.k8s_backend {
            let mut backend_cleanup_names = desired_names.clone();
            backend_cleanup_names.extend(shadowed_names);
            for name in &backend_cleanup_names {
                if let Err(e) = backend.delete_pool(name).await {
                    warn!(pool = %name, error = %e, "failed to delete stale dynamic pool state for static Helm queue pool");
                }
                if let Err(e) = backend.delete_lease(name).await {
                    warn!(pool = %name, error = %e, "failed to delete stale dynamic pool lease for static Helm queue pool");
                }
            }
        }

        Ok(count)
    }

    pub async fn create_default_pool(&self) {
        if self.configured_profiles.is_empty() {
            info!("no machine profiles configured, skipping default pool creation");
            return;
        }

        let gpus: HashMap<String, u32> = self
            .configured_profiles
            .iter()
            .map(|p| (p.clone(), 0))
            .collect();

        match self
            .create_pool(DEFAULT_POOL_NAME, gpus, None, None, 0, Vec::new())
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
        pinned_models: Vec<String>,
    ) -> Result<Pool, Box<dyn std::error::Error + Send + Sync>> {
        self.create_pool_with_caps(
            name,
            gpus,
            HashMap::new(),
            bundle,
            ttl_seconds,
            minimum_worker_count,
            pinned_models,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create_pool_with_caps(
        &self,
        name: &str,
        gpus: HashMap<String, u32>,
        gpu_caps: HashMap<String, u32>,
        bundle: Option<String>,
        ttl_seconds: Option<u64>,
        minimum_worker_count: u32,
        pinned_models: Vec<String>,
    ) -> Result<Pool, Box<dyn std::error::Error + Send + Sync>> {
        self.create_pool_with_caps_on_queue(
            name,
            DEFAULT_POOL_NAME,
            gpus,
            gpu_caps,
            bundle,
            ttl_seconds,
            minimum_worker_count,
            pinned_models,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create_pool_with_caps_on_queue(
        &self,
        name: &str,
        queue_pool: &str,
        gpus: HashMap<String, u32>,
        gpu_caps: HashMap<String, u32>,
        bundle: Option<String>,
        ttl_seconds: Option<u64>,
        minimum_worker_count: u32,
        pinned_models: Vec<String>,
    ) -> Result<Pool, Box<dyn std::error::Error + Send + Sync>> {
        let raw_name = name;
        validate_pool_name(name)?;
        let name = normalize_pool_name(name);
        if name == DEFAULT_POOL_NAME && raw_name != DEFAULT_POOL_NAME {
            return Err(Box::new(DefaultPoolMutationError));
        }
        validate_pool_name(queue_pool)?;
        let queue_pool = normalize_queue_pool(queue_pool);
        let gpus = Self::normalize_gpus_and_caps(gpus, &gpu_caps);
        self.validate_pool_profiles(&gpus, &gpu_caps)?;
        let is_static_pool = self.is_static_pool(&name).await;
        if is_static_pool {
            return Err(Box::new(StaticPoolMutationError {
                name: raw_name.to_string(),
            }));
        }
        if !self.is_known_queue_pool(&queue_pool).await {
            return Err(Box::new(UnknownQueuePoolError {
                queue_pool: queue_pool.clone(),
            }));
        }

        let now = Self::now_secs();
        let mut pools = self.pools.write().await;

        // Idempotent: return/update the existing pool, including
        // case-variant requests for the same logical pool name.
        if let Some(pool_key) = pool_key_for_name(&pools, &name) {
            let existing = pools
                .get_mut(&pool_key)
                .expect("pool key resolved from the same map");
            let spec_changed = existing.spec.bundle != bundle
                || existing.spec.queue_pool != queue_pool
                || existing.spec.gpus != gpus
                || existing.spec.gpu_caps != gpu_caps
                || existing.spec.ttl_seconds != ttl_seconds
                || existing.spec.minimum_worker_count != minimum_worker_count
                || existing.spec.pinned_models != pinned_models;

            if pool_key.eq_ignore_ascii_case(DEFAULT_POOL_NAME) && spec_changed {
                return Err(Box::new(DefaultPoolMutationError));
            }

            existing.status.last_renewed = now;
            if spec_changed {
                existing.spec.bundle = bundle;
                existing.spec.queue_pool = queue_pool.clone();
                existing.spec.gpus = gpus;
                existing.spec.gpu_caps = gpu_caps;
                existing.spec.ttl_seconds = ttl_seconds;
                existing.spec.minimum_worker_count = minimum_worker_count;
                existing.spec.pinned_models = pinned_models.clone();
                existing.status.state = PoolState::Pending;
                existing.status.assigned_workers.clear();
            }
            let result = existing.clone();
            drop(pools); // release write lock before K8s call

            // Persist spec updates and renew the K8s Lease (best-effort,
            // skip default pool).
            if !pool_key.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
                if let Some(ref backend) = self.k8s_backend {
                    if spec_changed {
                        if let Err(e) = backend.save_pool(&result).await {
                            warn!(error = %e, pool = %pool_key, "failed to persist pool update to K8s");
                        }
                    }
                    let ttl = Self::lease_ttl_seconds(&result);
                    if let Err(e) = backend.create_or_renew_lease(&pool_key, ttl).await {
                        warn!(error = %e, pool = %pool_key, "failed to renew K8s Lease");
                    }
                }
            }

            return Ok(result);
        }

        let pool = Pool {
            spec: PoolSpec {
                name: name.clone(),
                queue_pool,
                bundle,
                gpus,
                gpu_caps,
                ttl_seconds,
                minimum_worker_count,
                pinned_models,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: now,
                last_renewed: now,
            },
        };

        pools.insert(name.clone(), pool.clone());
        drop(pools); // release write lock before K8s call
        info!(pool = %name, "created pool");

        // Persist dynamic named pools to K8s. The default pool is synthetic
        // per gateway replica and is not a lease-owned runtime pool.
        if !name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
            if let Some(ref backend) = self.k8s_backend {
                if let Err(e) = backend.save_pool(&pool).await {
                    warn!(error = %e, pool = %name, "failed to persist pool to K8s");
                }
                let ttl = Self::lease_ttl_seconds(&pool);
                if let Err(e) = backend.create_or_renew_lease(&name, ttl).await {
                    warn!(error = %e, pool = %name, "failed to create K8s Lease");
                }
            }
        }

        Ok(pool)
    }

    pub async fn get_pool(&self, name: &str) -> Option<Pool> {
        let pools = self.pools.read().await;
        let key = pool_key_for_name(&pools, name)?;
        pools.get(&key).cloned()
    }

    pub async fn capped_lane_status(
        &self,
        pool_name: &str,
        machine_profile: &str,
        bundle: &str,
    ) -> Option<PoolAdmissionStatus> {
        if machine_profile.is_empty() {
            return None;
        }

        let pools = self.pools.read().await;
        let key = pool_key_for_name(&pools, pool_name)?;
        let pool = pools.get(&key)?;
        let cap = pool
            .spec
            .gpu_caps
            .iter()
            .find(|(profile, _)| profile.eq_ignore_ascii_case(machine_profile))
            .map(|(_, cap)| *cap)?;
        let assigned_count = pool
            .status
            .assigned_workers
            .iter()
            .filter(|worker| worker.gpu.eq_ignore_ascii_case(machine_profile))
            .filter(|worker| worker.bundle.eq_ignore_ascii_case(bundle))
            .count();

        Some(PoolAdmissionStatus {
            cap,
            assigned_count,
        })
    }

    /// Return the assigned worker names for a capped concrete lane.
    ///
    /// `None` means the pool/profile is uncapped, so callers should not apply
    /// an admission filter. `Some(empty)` means admission is enabled for this
    /// profile but no worker is currently admitted. This mirrors the
    /// worker-side pool-admission gate: named pools with missing status fail
    /// closed, while the default pool remains fail-open.
    pub async fn admitted_worker_names_for_capped_lane(
        &self,
        pool_name: &str,
        machine_profile: &str,
        bundle: &str,
    ) -> Option<HashSet<String>> {
        if machine_profile.trim().is_empty() {
            return None;
        }

        let pools = self.pools.read().await;
        let Some(key) = pool_key_for_name(&pools, pool_name) else {
            return if pool_name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
                None
            } else {
                Some(HashSet::new())
            };
        };
        let pool = pools.get(&key)?;
        let cap = pool
            .spec
            .gpu_caps
            .iter()
            .find(|(profile, _)| profile.eq_ignore_ascii_case(machine_profile))
            .map(|(_, cap)| *cap)?;
        if cap == 0 {
            return Some(HashSet::new());
        }

        Some(
            pool.status
                .assigned_workers
                .iter()
                .filter(|worker| worker.gpu.eq_ignore_ascii_case(machine_profile))
                .filter(|worker| worker.bundle.eq_ignore_ascii_case(bundle))
                .map(|worker| worker.name.clone())
                .collect(),
        )
    }

    /// Every machine profile the pool can provision, sorted for a
    /// deterministic emit order. Used to fan pending demand across all cold
    /// lanes when a gpu-agnostic request cannot name a single profile: a
    /// pool serves every bundle from every profile it declares, so any of
    /// them may end up running the work, and KEDA must be able to observe
    /// `pending_demand{pool,machine_profile,bundle}` on each. Empty when the
    /// pool is unknown.
    pub async fn demand_profiles_for_pool(&self, pool_name: &str) -> Vec<String> {
        {
            let pools = self.pools.read().await;
            if let Some(key) = pool_key_for_name(&pools, pool_name) {
                if let Some(pool) = pools.get(&key) {
                    let mut profiles = machine_profiles_for_pool(pool);
                    profiles.sort();
                    return profiles;
                }
            }
        }
        // The default pool is implicit (never stored in `pools`): its cold
        // lanes are the cluster's configured machine profiles. Lowercased to
        // match the KEDA `machine_profile` label the chart renders.
        if normalize_pool_name(pool_name) == DEFAULT_POOL_NAME {
            let mut profiles: Vec<String> = self
                .configured_profiles
                .iter()
                .map(|p| p.to_lowercase())
                .collect();
            profiles.sort();
            profiles.dedup();
            return profiles;
        }
        Vec::new()
    }

    pub async fn queue_pool_for_pool(&self, pool_name: &str) -> Option<String> {
        let pools = self.pools.read().await;
        let key = pool_key_for_name(&pools, pool_name)?;
        let pool = pools.get(&key)?;
        Some(normalize_queue_pool(&pool.spec.queue_pool))
    }

    pub async fn list_pools(&self) -> Vec<Pool> {
        let pools = self.pools.read().await;
        pools.values().cloned().collect()
    }

    /// Clone only the fields required by the KEDA capacity reconciler.
    pub async fn capacity_pools(&self) -> Vec<CapacityPoolSnapshot> {
        let pools = self.pools.read().await;
        pools
            .values()
            .map(CapacityPoolSnapshot::from_pool)
            .collect()
    }

    pub async fn delete_pool(&self, name: &str) -> Result<bool, PoolDeletionProtectedError> {
        if name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
            return Err(PoolDeletionProtectedError::Default);
        }
        if self.is_static_pool(name).await {
            return Err(PoolDeletionProtectedError::Static(name.to_string()));
        }

        let mut pools = self.pools.write().await;
        let key = pool_key_for_name(&pools, name).unwrap_or_else(|| name.to_string());
        let removed = pools.remove(&key).is_some();
        drop(pools); // release write lock before K8s call

        if removed {
            info!(pool = %key, "deleted pool");
            if let Some(ref backend) = self.k8s_backend {
                if let Err(e) = backend.delete_pool(&key).await {
                    warn!(error = %e, pool = %key, "failed to delete pool from K8s");
                }
                if let Err(e) = backend.delete_lease(&key).await {
                    warn!(error = %e, pool = %key, "failed to delete K8s Lease");
                }
            }
        }
        Ok(removed)
    }

    pub async fn renew_pool(&self, name: &str) -> bool {
        let is_static_pool = self.is_static_pool(name).await;
        let found_key = {
            let mut pools = self.pools.write().await;
            if let Some(key) = pool_key_for_name(&pools, name) {
                if let Some(pool) = pools.get_mut(&key) {
                    pool.status.last_renewed = Self::now_secs();
                    Some(key)
                } else {
                    None
                }
            } else {
                None
            }
        }; // write lock dropped here

        // Renew the K8s Lease (best-effort, skip default and static Helm pools)
        if let Some(key) = found_key.as_ref() {
            if let Some(ref backend) = self.k8s_backend {
                if !key.eq_ignore_ascii_case(DEFAULT_POOL_NAME) && !is_static_pool {
                    if let Err(e) = backend.renew_lease(key).await {
                        warn!(error = %e, pool = %key, "failed to renew K8s Lease");
                    }
                }
            }
        }

        found_key.is_some()
    }

    pub async fn assign_workers(
        &self,
        pool_name: &str,
        available_workers: &[WorkerAssignment], // (name, url, gpu, bundle, queue_pool)
    ) -> bool {
        let is_static_pool = self.is_static_pool(pool_name).await;
        let mut pools = self.pools.write().await;
        let pool_key = match pool_key_for_name(&pools, pool_name) {
            Some(key) => key,
            None => return false,
        };
        let pool = pools
            .get_mut(&pool_key)
            .expect("pool key resolved from the same map");
        let queue_pool = normalize_queue_pool(&pool.spec.queue_pool);

        let filtered: Vec<&WorkerAssignment> = available_workers
            .iter()
            .filter(|(_, _, _, _, worker_queue_pool)| {
                worker_consumes_pool(worker_queue_pool, &queue_pool)
            })
            .filter(|(_, _, _, bundle, _)| match pool.spec.bundle.as_ref() {
                Some(bundle_filter) => bundle == bundle_filter,
                None => true,
            })
            .collect();

        // Group by GPU type (lowercase)
        let mut workers_by_gpu: HashMap<String, Vec<&WorkerAssignment>> = HashMap::new();
        for w in &filtered {
            workers_by_gpu
                .entry(w.2.to_lowercase())
                .or_default()
                .push(w);
        }
        for workers in workers_by_gpu.values_mut() {
            // HA gateway replicas discover the same K8s endpoint set and each
            // computes capped admission locally. Stable ordering keeps the
            // admitted pod set convergent no matter how HashMap iteration lands.
            workers.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        }

        let mut assigned: Vec<AssignedWorker> = Vec::new();
        let mut all_met = true;
        let gpu_caps: HashMap<String, u32> = pool
            .spec
            .gpu_caps
            .iter()
            .map(|(gpu, cap)| (gpu.to_lowercase(), *cap))
            .collect();

        for (gpu_type, required_count) in &pool.spec.gpus {
            let gpu_lower = gpu_type.to_lowercase();
            let available = workers_by_gpu.get_mut(&gpu_lower);
            let available_count = available.as_ref().map(|workers| workers.len()).unwrap_or(0);
            let required = *required_count as usize;

            if available_count < required {
                all_met = false;
            }

            if let Some(workers) = available {
                let cap = gpu_caps
                    .get(&gpu_lower)
                    .map(|cap| *cap as usize)
                    .unwrap_or(usize::MAX);
                let take = workers.len().min(cap);
                for w in workers.drain(..take) {
                    assigned.push(AssignedWorker {
                        name: w.0.clone(),
                        url: w.1.clone(),
                        gpu: w.2.clone(),
                        bundle: w.3.clone(),
                    });
                }
            }
        }

        let state = if pool_key.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
            if assigned.is_empty() {
                PoolState::Pending
            } else {
                PoolState::Active
            }
        } else if all_met {
            PoolState::Active
        } else {
            PoolState::Pending
        };
        let new_status = PoolStatus {
            state,
            assigned_workers: assigned,
            created_at: pool.status.created_at,
            last_renewed: pool.status.last_renewed,
        };
        let status_changed = pool.status != new_status;
        pool.status = new_status.clone();
        let backend = self.k8s_backend.clone();
        drop(pools);

        if status_changed && !pool_key.eq_ignore_ascii_case(DEFAULT_POOL_NAME) && !is_static_pool {
            if let Some(ref backend) = backend {
                if let Err(e) = backend.update_pool_status(&pool_key, &new_status).await {
                    warn!(error = %e, pool = %pool_key, "failed to persist pool assignment status to K8s");
                }
            }
        }

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
    /// pool has a more recent `last_renewed` timestamp. Status-only updates may
    /// keep `last_renewed` unchanged, because assignment changes should not
    /// extend the pool lease.
    /// Does NOT write back to K8s (the event already came from K8s).
    pub async fn apply_remote_pool(&self, mut pool: Pool) {
        let raw_name = pool.spec.name.clone();

        // Skip the default pool -- each gateway manages its own default pool
        if raw_name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
            return;
        }
        if let Err(e) = validate_pool_name(&raw_name) {
            warn!(pool = %raw_name, error = %e, "skipping invalid pool from K8s watch");
            return;
        }
        let name = normalize_pool_name(&raw_name);
        pool.spec.name = name.clone();
        pool.spec.queue_pool = normalize_queue_pool(&pool.spec.queue_pool);
        if let Err(e) = validate_pool_name(&pool.spec.queue_pool) {
            warn!(pool = %name, queue_pool = %pool.spec.queue_pool, error = %e, "skipping invalid backing queue pool from K8s watch");
            return;
        }
        if !self.is_known_queue_pool(&pool.spec.queue_pool).await {
            warn!(
                pool = %name,
                queue_pool = %pool.spec.queue_pool,
                "skipping pool with unknown backing queue pool from K8s watch"
            );
            return;
        }
        if self.is_static_pool(&name).await {
            return;
        }

        let mut pools = self.pools.write().await;
        let pool_key = pool_key_for_name(&pools, &name);
        if let Some(existing_key) = pool_key.as_ref() {
            let existing = pools
                .get(existing_key)
                .expect("pool key resolved from the same map");
            let renewed_delta = pool.status.last_renewed - existing.status.last_renewed;
            if renewed_delta < -TIMESTAMP_TOLERANCE_S {
                return;
            }

            if renewed_delta.abs() <= TIMESTAMP_TOLERANCE_S {
                if pool.spec != existing.spec {
                    return;
                }
                if pool.status == existing.status {
                    return;
                }
            }

            if existing_key != &name {
                pools.remove(existing_key);
            }
        }

        info!(pool = %name, "applied remote pool from K8s watch");
        pools.insert(name, pool);
    }

    /// Remove a pool that was deleted by a remote gateway (via K8s watch).
    /// Does NOT write back to K8s (the event already came from K8s).
    pub async fn remove_remote_pool(&self, name: &str) {
        // Never delete the default pool via watch events
        if name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
            return;
        }
        if self.is_static_pool(name).await {
            return;
        }

        let mut pools = self.pools.write().await;
        if let Some(key) = pool_key_for_name(&pools, name) {
            pools.remove(&key);
            info!(pool = %key, "removed remote pool via K8s watch");
        }
    }

    pub async fn check_expired_leases(&self) -> Vec<String> {
        let now = Self::now_secs();
        let mut expired = Vec::new();
        let static_pool_names = self.static_pool_names.read().await.clone();

        {
            let pools = self.pools.read().await;
            for (name, pool) in pools.iter() {
                if name.eq_ignore_ascii_case(DEFAULT_POOL_NAME)
                    || static_pool_names.contains(&normalize_pool_name(name))
                {
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
                        if name.eq_ignore_ascii_case(DEFAULT_POOL_NAME)
                            || expired.contains(&name)
                            || static_pool_names.contains(&normalize_pool_name(&name))
                        {
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
        }

        expired
    }
}

fn worker_consumes_pool(worker_pool: &str, pool_name: &str) -> bool {
    normalize_pool_name(worker_pool) == normalize_pool_name(pool_name)
}

pub(crate) fn normalize_pool_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

fn normalize_queue_pool(name: &str) -> String {
    let normalized = normalize_pool_name(name);
    if normalized.is_empty() {
        DEFAULT_POOL_NAME.to_string()
    } else {
        normalized
    }
}

fn known_queue_pool_from_names(static_pool_names: &HashSet<String>, name: &str) -> bool {
    let queue_pool = normalize_queue_pool(name);
    queue_pool == DEFAULT_POOL_NAME || static_pool_names.contains(&queue_pool)
}

/// Deduplicated, lowercased machine-profile set for a pool, taken from the
/// union of `spec.gpus` (requirements) and `spec.gpu_caps` (admission caps).
/// This is the same set `demand_profiles_for_pool` fans pending demand across;
/// the warm-floor snapshot keeps all of them so each lane gets its own
/// `sie.gateway.pool.warm_floor` series.
pub(crate) fn machine_profiles_for_pool(pool: &Pool) -> Vec<String> {
    let mut profiles: Vec<String> = Vec::new();
    for profile in pool.spec.gpus.keys().chain(pool.spec.gpu_caps.keys()) {
        if profiles
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(profile))
        {
            continue;
        }
        profiles.push(profile.to_lowercase());
    }
    profiles
}

fn validate_pool_name(name: &str) -> Result<(), InvalidPoolNameError> {
    if !name.is_empty()
        && name.len() <= 128
        && !name.eq_ignore_ascii_case("_default")
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
    {
        Ok(())
    } else {
        Err(InvalidPoolNameError {
            name: name.to_string(),
        })
    }
}

fn pool_key_for_name(pools: &HashMap<String, Pool>, name: &str) -> Option<String> {
    if pools.contains_key(name) {
        return Some(name.to_string());
    }
    let normalized = normalize_pool_name(name);
    pools
        .keys()
        .find(|pool_name| normalize_pool_name(pool_name) == normalized)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(
        name: &str,
        url: &str,
        gpu: &str,
        bundle: &str,
    ) -> (String, String, String, String, String) {
        worker_in_pool(name, url, gpu, bundle, DEFAULT_POOL_NAME)
    }

    fn worker_in_pool(
        name: &str,
        url: &str,
        gpu: &str,
        bundle: &str,
        pool: &str,
    ) -> (String, String, String, String, String) {
        (
            name.to_string(),
            url.to_string(),
            gpu.to_string(),
            bundle.to_string(),
            pool.to_string(),
        )
    }

    fn static_queue_pool(name: &str, profile: &str) -> PoolSpec {
        let mut gpus = HashMap::new();
        gpus.insert(profile.to_string(), 0);
        PoolSpec {
            name: name.to_string(),
            queue_pool: name.to_string(),
            bundle: None,
            gpus,
            gpu_caps: HashMap::new(),
            ttl_seconds: None,
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }
    }

    #[test]
    fn test_validate_pool_name_matches_route_contract() {
        for name in ["default", "customer-acme", "bench_1", "A100Pool"] {
            validate_pool_name(name).unwrap();
        }

        for name in ["", "foo.bar", "foo*", "foo>", "foo bar", " foo", "_default"] {
            assert!(validate_pool_name(name).is_err(), "{name:?} should fail");
        }
        assert!(validate_pool_name(&"a".repeat(129)).is_err());
        validate_pool_name(&"a".repeat(128)).unwrap();
    }

    #[test]
    fn test_normalize_pool_name_canonicalizes_case() {
        assert_eq!(normalize_pool_name("Customer-Acme"), "customer-acme");
    }

    #[tokio::test]
    async fn test_create_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);

        let pool = pm
            .create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();
        assert_eq!(pool.spec.name, "test");
        assert_eq!(pool.spec.queue_pool, DEFAULT_POOL_NAME);
        assert_eq!(pool.status.state, PoolState::Pending);
        assert_eq!(pool.spec.minimum_worker_count, 0);
    }

    #[tokio::test]
    async fn test_create_pool_rejects_unknown_non_default_queue_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        let err = pm
            .create_pool_with_caps_on_queue(
                "tenant",
                "missing-queue",
                gpus,
                HashMap::new(),
                None,
                None,
                0,
                vec![],
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("Unknown backing queue_pool"));
    }

    #[tokio::test]
    async fn test_create_pool_accepts_static_non_default_queue_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.sync_static_pools(&[static_queue_pool("bench", "l4-spot")])
            .await
            .unwrap();

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        let pool = pm
            .create_pool_with_caps_on_queue(
                "tenant",
                "bench",
                gpus,
                HashMap::new(),
                None,
                None,
                0,
                vec![],
            )
            .await
            .unwrap();

        assert_eq!(pool.spec.name, "tenant");
        assert_eq!(pool.spec.queue_pool, "bench");
    }

    #[tokio::test]
    async fn test_create_pool_stores_minimum_worker_count() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);

        // Trailing arg is the warm floor; it must round-trip onto the spec.
        let pool = pm
            .create_pool("test", gpus, None, None, 3, vec![])
            .await
            .unwrap();
        assert_eq!(pool.spec.minimum_worker_count, 3);
    }

    #[tokio::test]
    async fn test_create_pool_stores_pinned_models() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);

        // Trailing arg is the pinned-model set; it must round-trip onto the spec.
        let pool = pm
            .create_pool("test", gpus, None, None, 0, vec!["BAAI/bge-m3".to_string()])
            .await
            .unwrap();
        assert_eq!(pool.spec.pinned_models, vec!["BAAI/bge-m3"]);
    }

    #[tokio::test]
    async fn test_create_pool_idempotent() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);

        let pool1 = pm
            .create_pool("test", gpus.clone(), None, None, 0, vec![])
            .await
            .unwrap();
        let pool2 = pm
            .create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();
        assert_eq!(pool1.spec.name, pool2.spec.name);
    }

    #[tokio::test]
    async fn test_create_pool_existing_updates_spec() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            "test",
        )];
        pm.assign_workers("test", &workers).await;

        let mut updated_gpus = HashMap::new();
        updated_gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        let pool = pm
            .create_pool_with_caps("test", updated_gpus, gpu_caps, None, Some(60), 0, vec![])
            .await
            .unwrap();

        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(pool.spec.gpu_caps.get("l4-spot"), Some(&2));
        assert_eq!(pool.spec.ttl_seconds, Some(60));
        assert_eq!(pool.status.state, PoolState::Pending);
        assert!(pool.status.assigned_workers.is_empty());
    }

    #[tokio::test]
    async fn test_create_pool_rejects_default_mutation() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        let result = pm
            .create_pool(DEFAULT_POOL_NAME, gpus, None, None, 0, vec![])
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_pool_rejects_invalid_pool_names() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        for name in ["", "bench.l4", "bench*", "bench>", "bench pool", "_default"] {
            let mut gpus = HashMap::new();
            gpus.insert("l4-spot".to_string(), 1);
            let result = pm.create_pool(name, gpus, None, None, 0, vec![]).await;
            assert!(result.is_err(), "{name:?} should fail");
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("Invalid pool name"));
        }
    }

    #[tokio::test]
    async fn test_create_pool_rejects_default_case_variant() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        let result = pm.create_pool("Default", gpus, None, None, 0, vec![]).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("default pool"));
        let pools = pm.list_pools().await;
        assert_eq!(pools.len(), 1);
        assert_eq!(pools[0].spec.name, DEFAULT_POOL_NAME);

        let idempotent_case_variant = pm
            .create_pool("Default", HashMap::new(), None, None, 0, vec![])
            .await;
        assert!(idempotent_case_variant.is_err());
    }

    #[tokio::test]
    async fn test_create_pool_updates_case_variant_dynamic_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("Bench", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let mut updated_gpus = HashMap::new();
        updated_gpus.insert("l4-spot".to_string(), 2);
        let pool = pm
            .create_pool("bench", updated_gpus, None, Some(60), 0, vec![])
            .await
            .unwrap();

        assert_eq!(pool.spec.name, "bench");
        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&2));
        assert_eq!(pool.spec.ttl_seconds, Some(60));
        let pools = pm.list_pools().await;
        assert_eq!(pools.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_default_pool_fails() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        let result = pm.delete_pool(DEFAULT_POOL_NAME).await;
        assert!(result.is_err());

        let case_variant = pm.delete_pool("Default").await;
        assert!(case_variant.is_err());
    }

    #[tokio::test]
    async fn test_delete_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let deleted = pm.delete_pool("test").await.unwrap();
        assert!(deleted);

        let pool = pm.get_pool("test").await;
        assert!(pool.is_none());
    }

    #[tokio::test]
    async fn test_sync_static_pools_creates_uncapped_non_expiring_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let specs = vec![PoolSpec {
            name: "company-a".to_string(),
            queue_pool: DEFAULT_POOL_NAME.to_string(),
            bundle: None,
            gpus,
            gpu_caps: HashMap::new(),
            ttl_seconds: Some(1),
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }];

        let count = pm.sync_static_pools(&specs).await.unwrap();
        assert_eq!(count, 1);

        let pool = pm.get_pool("company-a").await.unwrap();
        assert_eq!(pool.spec.ttl_seconds, None);
        assert!(pool.spec.gpu_caps.is_empty());
        assert!(pm
            .admitted_worker_names_for_capped_lane("company-a", "l4-spot", "default")
            .await
            .is_none());

        {
            let mut pools = pm.pools.write().await;
            pools.get_mut("company-a").unwrap().status.last_renewed -=
                DEFAULT_LEASE_DURATION_S + 10.0;
        }

        let expired = pm.check_expired_leases().await;
        assert!(!expired.contains(&"company-a".to_string()));
        assert!(pm.get_pool("company-a").await.is_some());
    }

    #[tokio::test]
    async fn test_static_pool_caps_are_assignable_and_lane_scoped() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 1);
        let specs = vec![PoolSpec {
            name: "bench".to_string(),
            queue_pool: DEFAULT_POOL_NAME.to_string(),
            bundle: None,
            gpus: HashMap::new(),
            gpu_caps,
            ttl_seconds: None,
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }];

        pm.sync_static_pools(&specs).await.unwrap();
        let pool = pm.get_pool("bench").await.unwrap();
        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&0));

        let workers = vec![
            worker_in_pool("w2", "http://w2:8080", "l4-spot", "default", "bench"),
            worker_in_pool("w1", "http://w1:8080", "l4-spot", "default", "bench"),
            worker("default-worker", "http://w3:8080", "l4-spot", "default"),
        ];
        assert!(pm.assign_workers("bench", &workers).await);

        let admitted = pm
            .admitted_worker_names_for_capped_lane("bench", "l4-spot", "default")
            .await
            .expect("static cap should make admission lane-scoped");
        assert_eq!(admitted, HashSet::from(["w1".to_string()]));
    }

    #[tokio::test]
    async fn test_static_pool_rejects_api_mutation_and_delete() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let specs = vec![PoolSpec {
            name: "bench".to_string(),
            queue_pool: DEFAULT_POOL_NAME.to_string(),
            bundle: None,
            gpus,
            gpu_caps: HashMap::new(),
            ttl_seconds: None,
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }];
        pm.sync_static_pools(&specs).await.unwrap();

        let mut changed_gpus = HashMap::new();
        changed_gpus.insert("l4-spot".to_string(), 1);
        let result = pm
            .create_pool("bench", changed_gpus, None, Some(60), 0, vec![])
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("static Helm queue pool"));

        let delete_result = pm.delete_pool("bench").await;
        assert!(delete_result.is_err());
        assert!(delete_result
            .unwrap_err()
            .to_string()
            .contains("static Helm queue pool"));
        assert!(pm.renew_pool("bench").await);
    }

    #[tokio::test]
    async fn test_static_pool_case_variants_are_protected() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let specs = vec![PoolSpec {
            name: "Bench".to_string(),
            queue_pool: DEFAULT_POOL_NAME.to_string(),
            bundle: None,
            gpus,
            gpu_caps: HashMap::new(),
            ttl_seconds: None,
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }];
        pm.sync_static_pools(&specs).await.unwrap();

        assert_eq!(pm.get_pool("Bench").await.unwrap().spec.name, "bench");

        let mut changed_gpus = HashMap::new();
        changed_gpus.insert("l4-spot".to_string(), 1);
        let result = pm
            .create_pool("BENCH", changed_gpus, None, Some(60), 0, vec![])
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("static Helm queue pool"));

        let delete_result = pm.delete_pool("BENCH").await;
        assert!(delete_result.is_err());
        assert!(delete_result
            .unwrap_err()
            .to_string()
            .contains("static Helm queue pool"));
        assert!(pm.renew_pool("BENCH").await);
    }

    #[tokio::test]
    async fn test_static_pool_rejects_invalid_name() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let specs = vec![PoolSpec {
            name: " bench".to_string(),
            queue_pool: DEFAULT_POOL_NAME.to_string(),
            bundle: None,
            gpus: HashMap::from([("l4-spot".to_string(), 0)]),
            gpu_caps: HashMap::new(),
            ttl_seconds: None,
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }];

        let result = pm.sync_static_pools(&specs).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid pool name"));
        assert!(pm.list_pools().await.is_empty());
    }

    #[tokio::test]
    async fn test_static_pool_rejects_duplicate_case_variants() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let specs = vec![
            PoolSpec {
                name: "Bench".to_string(),
                queue_pool: DEFAULT_POOL_NAME.to_string(),
                bundle: None,
                gpus: gpus.clone(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
                pinned_models: Vec::new(),
            },
            PoolSpec {
                name: "bench".to_string(),
                queue_pool: DEFAULT_POOL_NAME.to_string(),
                bundle: None,
                gpus,
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
                pinned_models: Vec::new(),
            },
        ];

        let result = pm.sync_static_pools(&specs).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("duplicate static queue pool 'bench'"));
        assert!(pm.list_pools().await.is_empty());
    }

    #[tokio::test]
    async fn test_static_pool_removes_shadowed_case_variant_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut dynamic_gpus = HashMap::new();
        dynamic_gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("Bench", dynamic_gpus, None, Some(60), 0, vec![])
            .await
            .unwrap();

        let mut static_gpus = HashMap::new();
        static_gpus.insert("l4-spot".to_string(), 0);
        let specs = vec![PoolSpec {
            name: "bench".to_string(),
            queue_pool: DEFAULT_POOL_NAME.to_string(),
            bundle: None,
            gpus: static_gpus,
            gpu_caps: HashMap::new(),
            ttl_seconds: None,
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }];
        pm.sync_static_pools(&specs).await.unwrap();

        let pools = pm.list_pools().await;
        assert_eq!(pools.len(), 1);
        assert_eq!(pools[0].spec.name, "bench");
        assert_eq!(pools[0].spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(pm.get_pool("Bench").await.unwrap().spec.name, "bench");
    }

    #[tokio::test]
    async fn test_static_pool_ignores_remote_k8s_watch_events() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let specs = vec![PoolSpec {
            name: "bench".to_string(),
            queue_pool: DEFAULT_POOL_NAME.to_string(),
            bundle: None,
            gpus,
            gpu_caps: HashMap::new(),
            ttl_seconds: None,
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }];
        pm.sync_static_pools(&specs).await.unwrap();

        let mut remote_gpus = HashMap::new();
        remote_gpus.insert("l4-spot".to_string(), 1);
        let remote = Pool {
            spec: PoolSpec {
                name: "bench".to_string(),
                queue_pool: DEFAULT_POOL_NAME.to_string(),
                bundle: Some("other".to_string()),
                gpus: remote_gpus,
                gpu_caps: HashMap::new(),
                ttl_seconds: Some(60),
                minimum_worker_count: 1,
                pinned_models: Vec::new(),
            },
            status: PoolStatus {
                state: PoolState::Active,
                assigned_workers: vec![AssignedWorker {
                    name: "remote-worker".to_string(),
                    url: "http://remote-worker:8080".to_string(),
                    gpu: "l4-spot".to_string(),
                    bundle: "other".to_string(),
                }],
                created_at: 9999.0,
                last_renewed: 9999.0,
            },
        };

        pm.apply_remote_pool(remote).await;
        let found = pm.get_pool("bench").await.unwrap();
        assert_eq!(found.spec.bundle, None);
        assert_eq!(found.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(found.spec.ttl_seconds, None);
        assert_eq!(found.status.state, PoolState::Pending);
        assert!(found.status.assigned_workers.is_empty());

        pm.remove_remote_pool("bench").await;
        assert!(pm.get_pool("bench").await.is_some());
    }

    #[tokio::test]
    async fn test_assign_workers() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);
        pm.create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![
            worker("w1", "http://w1:8080", "l4-spot", "default"),
            worker("w2", "http://w2:8080", "l4-spot", "default"),
            worker("w3", "http://w3:8080", "a100", "default"),
        ];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 2);
    }

    #[tokio::test]
    async fn test_assign_workers_required_count_does_not_cap_assignment() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 3);
        pm.create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let workers: Vec<(String, String, String, String, String)> = (1..=5)
            .map(|i| {
                worker_in_pool(
                    &format!("w{}", i),
                    &format!("http://w{}:8080", i),
                    "l4-spot",
                    "default",
                    DEFAULT_POOL_NAME,
                )
            })
            .collect();

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 5);
    }

    #[tokio::test]
    async fn test_assign_workers_honors_gpu_cap() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 3);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 4);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        let workers: Vec<(String, String, String, String, String)> = (1..=5)
            .map(|i| {
                worker_in_pool(
                    &format!("w{}", i),
                    &format!("http://w{}:8080", i),
                    "l4-spot",
                    "default",
                    DEFAULT_POOL_NAME,
                )
            })
            .collect();

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 4);
    }

    #[tokio::test]
    async fn test_assign_workers_cap_selection_is_deterministic() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![
            worker("w3", "http://w3:8080", "l4-spot", "default"),
            worker("w1", "http://w1:8080", "l4-spot", "default"),
            worker("w2", "http://w2:8080", "l4-spot", "default"),
        ];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        let names: Vec<&str> = pool
            .status
            .assigned_workers
            .iter()
            .map(|worker| worker.name.as_str())
            .collect();
        assert_eq!(names, vec!["w1", "w2"]);
    }

    #[tokio::test]
    async fn test_dynamic_pool_assigns_default_backing_queue_when_capped() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 1);
        pm.create_pool_with_caps("bench", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![
            worker("default-worker", "http://w1:8080", "l4-spot", "default"),
            worker_in_pool(
                "bench-worker",
                "http://w2:8080",
                "l4-spot",
                "default",
                "bench",
            ),
        ];

        let all_met = pm.assign_workers("bench", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("bench").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 1);
        assert_eq!(pool.status.assigned_workers[0].name, "default-worker");
    }

    #[tokio::test]
    async fn test_dynamic_pool_assigns_default_backing_queue_without_caps() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("bench", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![worker(
            "default-worker",
            "http://w1:8080",
            "l4-spot",
            "default",
        )];

        let all_met = pm.assign_workers("bench", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("bench").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 1);
        assert_eq!(pool.status.assigned_workers[0].name, "default-worker");
    }

    #[tokio::test]
    async fn test_assign_workers_honors_custom_backing_queue_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.sync_static_pools(&[static_queue_pool("bench", "l4-spot")])
            .await
            .unwrap();

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool_with_caps_on_queue(
            "tenant",
            "bench",
            gpus,
            HashMap::new(),
            None,
            None,
            0,
            vec![],
        )
        .await
        .unwrap();

        let workers = vec![
            worker("default-worker", "http://w1:8080", "l4-spot", "default"),
            worker_in_pool(
                "bench-worker",
                "http://w2:8080",
                "l4-spot",
                "default",
                "bench",
            ),
        ];

        let all_met = pm.assign_workers("tenant", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("tenant").await.unwrap();
        assert_eq!(pool.spec.queue_pool, "bench");
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 1);
        assert_eq!(pool.status.assigned_workers[0].name, "bench-worker");
    }

    #[tokio::test]
    async fn test_gpu_cap_without_requirement_defaults_required_to_zero() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let gpus = HashMap::new();
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        let workers: Vec<(String, String, String, String, String)> = (1..=3)
            .map(|i| {
                worker_in_pool(
                    &format!("w{}", i),
                    &format!("http://w{}:8080", i),
                    "l4-spot",
                    "default",
                    DEFAULT_POOL_NAME,
                )
            })
            .collect();

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 2);
    }

    #[tokio::test]
    async fn test_zero_required_pool_can_be_active_without_workers() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        pm.create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = Vec::new();
        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert!(pool.status.assigned_workers.is_empty());
    }

    #[tokio::test]
    async fn test_capped_lane_status_reports_cap_and_assigned_count() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            DEFAULT_POOL_NAME,
        )];
        pm.assign_workers("test", &workers).await;

        let lane_status = pm
            .capped_lane_status("test", "L4-SPOT", "default")
            .await
            .expect("lane should be capped");
        assert_eq!(lane_status.cap, 2);
        assert_eq!(lane_status.assigned_count, 1);
        assert!(pm
            .capped_lane_status("test", "a100", "default")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_admitted_worker_names_for_capped_lane_filters_assignment() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 1);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![
            worker("w1", "http://w1:8080", "l4-spot", "default"),
            worker("w2", "http://w2:8080", "l4-spot", "default"),
        ];
        pm.assign_workers("test", &workers).await;

        let admitted = pm
            .admitted_worker_names_for_capped_lane("test", "L4-SPOT", "default")
            .await
            .expect("profile should be capped");
        assert_eq!(admitted, HashSet::from(["w1".to_string()]));
    }

    #[tokio::test]
    async fn test_admitted_worker_names_for_capped_lane_respects_bundle() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![
            worker("w1", "http://w1:8080", "l4-spot", "default"),
            worker("w2", "http://w2:8080", "l4-spot", "sglang"),
        ];
        pm.assign_workers("test", &workers).await;

        let admitted = pm
            .admitted_worker_names_for_capped_lane("test", "l4-spot", "default")
            .await
            .expect("profile should be capped");
        assert_eq!(admitted, HashSet::from(["w1".to_string()]));

        let default_status = pm
            .capped_lane_status("test", "l4-spot", "default")
            .await
            .expect("default lane should be capped");
        assert_eq!(default_status.assigned_count, 1);
        let missing_status = pm
            .capped_lane_status("test", "l4-spot", "rerank")
            .await
            .expect("missing bundle lane should still be capped");
        assert_eq!(missing_status.assigned_count, 0);
    }

    #[tokio::test]
    async fn test_admitted_worker_names_for_uncapped_lane_returns_none() {
        let pm = PoolManager::new(vec!["l4-spot".to_string(), "a100".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("a100".to_string(), 1);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        assert!(pm
            .admitted_worker_names_for_capped_lane("test", "l4-spot", "default")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_admitted_worker_names_for_zero_cap_and_missing_named_pool_fail_closed() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 0);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0, vec![])
            .await
            .unwrap();

        assert_eq!(
            pm.admitted_worker_names_for_capped_lane("test", "l4-spot", "default")
                .await,
            Some(HashSet::new())
        );
        assert_eq!(
            pm.admitted_worker_names_for_capped_lane("missing", "l4-spot", "default")
                .await,
            Some(HashSet::new())
        );
        assert!(pm
            .admitted_worker_names_for_capped_lane(DEFAULT_POOL_NAME, "l4-spot", "default")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_assign_workers_partial() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 3);
        pm.create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            "test",
        )];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(!all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_renew_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("test", gpus, None, None, 0, vec![])
            .await
            .unwrap();

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

        let result = pm.create_pool("test", gpus, None, None, 0, vec![]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_per_pool_ttl_respected() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);

        // Pool with short TTL (1 second)
        pm.create_pool("short-ttl", gpus.clone(), None, Some(1), 0, vec![])
            .await
            .unwrap();

        // Pool with no TTL (uses global default = 1200s)
        pm.create_pool("default-ttl", gpus, None, None, 0, vec![])
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
        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(pool.spec.gpus.get("a100-40gb"), Some(&0));
        assert!(pool.spec.gpu_caps.is_empty());
    }

    #[tokio::test]
    async fn test_default_pool_state_tracks_eligible_workers() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        pm.create_default_pool().await;

        let no_workers: Vec<WorkerAssignment> = Vec::new();
        pm.assign_workers(DEFAULT_POOL_NAME, &no_workers).await;
        let pending = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();
        assert_eq!(pending.status.state, PoolState::Pending);
        assert!(pending.status.assigned_workers.is_empty());

        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            DEFAULT_POOL_NAME,
        )];
        pm.assign_workers(DEFAULT_POOL_NAME, &workers).await;
        let active = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();
        assert_eq!(active.status.state, PoolState::Active);
        assert_eq!(active.status.assigned_workers.len(), 1);
    }

    #[tokio::test]
    async fn test_default_pool_requires_explicit_worker_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        pm.create_default_pool().await;

        let workers = vec![
            worker_in_pool("empty", "http://w1:8080", "l4-spot", "default", ""),
            worker_in_pool(
                "_default",
                "http://w2:8080",
                "l4-spot",
                "default",
                "_default",
            ),
        ];
        pm.assign_workers(DEFAULT_POOL_NAME, &workers).await;

        let pool = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();
        assert_eq!(pool.status.state, PoolState::Pending);
        assert!(pool.status.assigned_workers.is_empty());
    }

    #[tokio::test]
    async fn test_restore_from_k8s_no_backend() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        // No K8s backend → returns Ok(0)
        let count = pm.restore_from_k8s().await.unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_restore_from_k8s_skips_default_pool() {
        let pool = Pool {
            spec: PoolSpec {
                name: DEFAULT_POOL_NAME.to_string(),
                queue_pool: DEFAULT_POOL_NAME.to_string(),
                bundle: None,
                gpus: HashMap::new(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
                pinned_models: Vec::new(),
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: 9999.0,
                last_renewed: 9999.0,
            },
        };

        assert!(!PoolManager::should_restore_pool_from_k8s(&pool));

        let mut case_variant = pool;
        case_variant.spec.name = "Default".to_string();
        assert!(!PoolManager::should_restore_pool_from_k8s(&case_variant));
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
                queue_pool: DEFAULT_POOL_NAME.to_string(),
                bundle: None,
                gpus: HashMap::new(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
                pinned_models: Vec::new(),
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
        // The remote pool had empty gpus; the original default pool has l4-spot=0
        assert_eq!(found.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(found.status.created_at, original.status.created_at);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_same_timestamp_applies_status_update() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let local = pm
            .create_pool("bench", gpus.clone(), None, Some(60), 0, vec![])
            .await
            .unwrap();

        let mut remote = local.clone();
        remote.status.state = PoolState::Active;
        remote.status.assigned_workers = vec![AssignedWorker {
            name: "worker-a".to_string(),
            url: "http://worker-a:8080".to_string(),
            gpu: "l4-spot".to_string(),
            bundle: "default".to_string(),
        }];

        pm.apply_remote_pool(remote).await;

        let found = pm.get_pool("bench").await.unwrap();
        assert_eq!(found.status.state, PoolState::Active);
        assert_eq!(found.status.assigned_workers.len(), 1);
        assert_eq!(found.spec.gpus, gpus);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_same_timestamp_keeps_local_spec() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let local = pm
            .create_pool(
                "bench",
                gpus,
                Some("default".to_string()),
                Some(60),
                0,
                vec![],
            )
            .await
            .unwrap();

        let mut remote = local.clone();
        remote.spec.bundle = Some("other".to_string());
        remote.status.state = PoolState::Active;

        pm.apply_remote_pool(remote).await;

        let found = pm.get_pool("bench").await.unwrap();
        assert_eq!(found.spec.bundle, Some("default".to_string()));
        assert_eq!(found.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_timestamp_tolerance_keeps_local_spec() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let local = pm
            .create_pool(
                "bench",
                gpus,
                Some("default".to_string()),
                Some(60),
                0,
                vec![],
            )
            .await
            .unwrap();

        let mut remote = local.clone();
        remote.status.last_renewed += TIMESTAMP_TOLERANCE_S / 2.0;
        remote.spec.bundle = Some("other".to_string());
        remote.status.state = PoolState::Active;

        pm.apply_remote_pool(remote).await;

        let found = pm.get_pool("bench").await.unwrap();
        assert_eq!(found.spec.bundle, Some("default".to_string()));
        assert_eq!(found.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_case_variant_updates_existing_logical_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        let local = pm
            .create_pool("Bench", gpus, None, Some(60), 0, vec![])
            .await
            .unwrap();

        let mut remote = local.clone();
        remote.spec.name = "bench".to_string();
        remote.spec.ttl_seconds = Some(120);
        remote.status.last_renewed += 1.0;

        pm.apply_remote_pool(remote).await;

        let pools = pm.list_pools().await;
        assert_eq!(pools.len(), 1);
        let found = pm.get_pool("Bench").await.unwrap();
        assert_eq!(found.spec.name, "bench");
        assert_eq!(found.spec.ttl_seconds, Some(120));
    }

    #[tokio::test]
    async fn test_apply_remote_pool_skips_invalid_pool_name() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let remote = Pool {
            spec: PoolSpec {
                name: "bench.l4".to_string(),
                queue_pool: DEFAULT_POOL_NAME.to_string(),
                bundle: None,
                gpus: HashMap::from([("l4-spot".to_string(), 1)]),
                gpu_caps: HashMap::new(),
                ttl_seconds: Some(60),
                minimum_worker_count: 0,
                pinned_models: Vec::new(),
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: 1000.0,
                last_renewed: 1000.0,
            },
        };

        pm.apply_remote_pool(remote).await;

        assert!(pm.list_pools().await.is_empty());
    }

    #[tokio::test]
    async fn test_remove_remote_pool_skips_default() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        pm.remove_remote_pool(DEFAULT_POOL_NAME).await;

        assert!(pm.get_pool(DEFAULT_POOL_NAME).await.is_some());
    }

    #[tokio::test]
    async fn test_remove_remote_pool_case_variant_removes_existing_logical_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("Bench", gpus, None, Some(60), 0, vec![])
            .await
            .unwrap();

        pm.remove_remote_pool("bench").await;

        assert!(pm.get_pool("Bench").await.is_none());
    }

    #[tokio::test]
    async fn test_remove_remote_pool_nonexistent() {
        let pm = PoolManager::new(vec![]);

        // Should not panic or error
        pm.remove_remote_pool("nonexistent").await;
    }
}
