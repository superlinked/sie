use std::sync::Arc;
use std::time::Duration;

use futures_util::TryStreamExt;
use k8s_openapi::api::core::v1::ConfigMap;
use kube::api::Api;
use kube::runtime::watcher::{self, Event};
use kube::Client;
use rand::prelude::*;
use tracing::{debug, info, warn};

use crate::state::k8s_pool_backend::{
    DATA_KEY, LABEL_MANAGED_BY, LABEL_MANAGED_BY_VALUE, LABEL_POOL,
};
use crate::state::pool_manager::PoolManager;
use crate::types::pool::Pool;

const CONFIGMAP_NAME_PREFIX: &str = "sie-pool-";

pub struct K8sPoolWatcher {
    client: Client,
    namespace: String,
    pool_manager: Arc<PoolManager>,
}

impl K8sPoolWatcher {
    pub fn new(client: Client, namespace: &str, pool_manager: Arc<PoolManager>) -> Self {
        Self {
            client,
            namespace: namespace.to_string(),
            pool_manager,
        }
    }

    pub async fn start(self: Arc<Self>) {
        tokio::spawn(async move {
            self.watch_loop().await;
        });
    }

    async fn watch_loop(&self) {
        let mut backoff = Duration::from_secs(1);
        let max_backoff = Duration::from_secs(60);
        let backoff_factor = 1.5f64;

        loop {
            match self.run_watch().await {
                Ok(()) => {
                    info!("k8s pool watch ended");
                    return;
                }
                Err(e) => {
                    warn!(error = %e, "k8s pool watch error, will retry");

                    // Add jitter (20%)
                    let jitter_factor = {
                        let mut rng = rand::thread_rng();
                        rng.gen_range(0.8..1.2)
                    };
                    let sleep_duration =
                        Duration::from_secs_f64(backoff.as_secs_f64() * jitter_factor);

                    tokio::time::sleep(sleep_duration).await;

                    // Exponential backoff with cap
                    backoff = Duration::from_secs_f64(
                        (backoff.as_secs_f64() * backoff_factor).min(max_backoff.as_secs_f64()),
                    );
                }
            }
        }
    }

    async fn run_watch(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let configmaps: Api<ConfigMap> = Api::namespaced(self.client.clone(), &self.namespace);

        info!(
            namespace = %self.namespace,
            "starting k8s pool ConfigMap watch"
        );

        let label_selector = format!("{}={}", LABEL_MANAGED_BY, LABEL_MANAGED_BY_VALUE);
        let stream = watcher::watcher(
            configmaps,
            watcher::Config::default().labels(&label_selector),
        );

        futures_util::pin_mut!(stream);

        while let Some(event) = stream.try_next().await? {
            match event {
                Event::Apply(cm) | Event::InitApply(cm) => {
                    self.handle_apply(&cm).await;
                }
                Event::Delete(cm) => {
                    self.handle_delete(&cm).await;
                }
                Event::Init => {
                    info!("k8s pool watch initial list starting");
                }
                Event::InitDone => {
                    info!("k8s pool watch initial list complete");
                }
            }
        }

        Ok(())
    }

    async fn handle_apply(&self, cm: &ConfigMap) {
        let pool = match extract_pool_from_configmap(cm) {
            Some(p) => p,
            None => {
                let name = cm.metadata.name.as_deref().unwrap_or("<unknown>");
                debug!(configmap = %name, "skipping ConfigMap without valid pool data");
                return;
            }
        };

        self.pool_manager.apply_remote_pool(pool).await;
    }

    async fn handle_delete(&self, cm: &ConfigMap) {
        let pool_name = match extract_pool_name(cm) {
            Some(n) => n,
            None => {
                let name = cm.metadata.name.as_deref().unwrap_or("<unknown>");
                debug!(configmap = %name, "skipping delete for ConfigMap without pool name");
                return;
            }
        };

        self.pool_manager.remove_remote_pool(&pool_name).await;
    }
}

/// Extract a Pool from a ConfigMap's `data["pool.json"]` field.
fn extract_pool_from_configmap(cm: &ConfigMap) -> Option<Pool> {
    let data = cm.data.as_ref()?;
    let json_str = data.get(DATA_KEY)?;
    serde_json::from_str::<Pool>(json_str).ok()
}

/// Extract the pool name from a ConfigMap, preferring the `sie-gateway/pool` label
/// and falling back to stripping the `sie-pool-` prefix from the ConfigMap name.
fn extract_pool_name(cm: &ConfigMap) -> Option<String> {
    // Try label first
    if let Some(labels) = &cm.metadata.labels {
        if let Some(pool_name) = labels.get(LABEL_POOL) {
            if !pool_name.is_empty() {
                return Some(pool_name.clone());
            }
        }
    }

    // Fall back to ConfigMap name with prefix stripped
    if let Some(ref name) = cm.metadata.name {
        if let Some(pool_name) = name.strip_prefix(CONFIGMAP_NAME_PREFIX) {
            if !pool_name.is_empty() {
                return Some(pool_name.to_string());
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use std::collections::{BTreeMap, HashMap};

    use crate::types::pool::{PoolSpec, PoolState, PoolStatus};

    fn make_pool(name: &str, last_renewed: f64) -> Pool {
        Pool {
            spec: PoolSpec {
                name: name.to_string(),
                bundle: None,
                gpus: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: 1000.0,
                last_renewed,
            },
        }
    }

    fn make_configmap(
        name: &str,
        labels: Option<BTreeMap<String, String>>,
        pool: Option<&Pool>,
    ) -> ConfigMap {
        let data = pool.map(|p| {
            let mut d = BTreeMap::new();
            d.insert(DATA_KEY.to_string(), serde_json::to_string(p).unwrap());
            d
        });

        ConfigMap {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                labels,
                ..Default::default()
            },
            data,
            ..Default::default()
        }
    }

    #[test]
    fn test_extract_pool_from_configmap() {
        let pool = make_pool("test-pool", 1000.0);
        let mut labels = BTreeMap::new();
        labels.insert(
            LABEL_MANAGED_BY.to_string(),
            LABEL_MANAGED_BY_VALUE.to_string(),
        );
        labels.insert(LABEL_POOL.to_string(), "test-pool".to_string());

        let cm = make_configmap("sie-pool-test-pool", Some(labels), Some(&pool));
        let extracted = extract_pool_from_configmap(&cm).unwrap();
        assert_eq!(extracted.spec.name, "test-pool");
        assert_eq!(extracted.status.last_renewed, 1000.0);
    }

    #[test]
    fn test_extract_pool_from_invalid_configmap() {
        let mut data = BTreeMap::new();
        data.insert(DATA_KEY.to_string(), "not valid json!!!".to_string());

        let cm = ConfigMap {
            metadata: ObjectMeta {
                name: Some("sie-pool-bad".to_string()),
                ..Default::default()
            },
            data: Some(data),
            ..Default::default()
        };

        assert!(extract_pool_from_configmap(&cm).is_none());
    }

    #[test]
    fn test_extract_pool_from_empty_configmap() {
        let cm = ConfigMap {
            metadata: ObjectMeta {
                name: Some("sie-pool-empty".to_string()),
                ..Default::default()
            },
            data: None,
            ..Default::default()
        };

        assert!(extract_pool_from_configmap(&cm).is_none());
    }

    #[test]
    fn test_extract_pool_name_from_label() {
        let mut labels = BTreeMap::new();
        labels.insert(LABEL_POOL.to_string(), "my-pool".to_string());

        let cm = make_configmap("sie-pool-my-pool", Some(labels), None);
        assert_eq!(extract_pool_name(&cm).unwrap(), "my-pool");
    }

    #[test]
    fn test_extract_pool_name_from_configmap_name_fallback() {
        let cm = make_configmap("sie-pool-eval-l4", None, None);
        assert_eq!(extract_pool_name(&cm).unwrap(), "eval-l4");
    }

    #[test]
    fn test_extract_pool_name_none() {
        let cm = ConfigMap {
            metadata: ObjectMeta {
                name: Some("unrelated-configmap".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(extract_pool_name(&cm).is_none());
    }

    #[tokio::test]
    async fn test_apply_remote_pool() {
        let pm = Arc::new(PoolManager::new(vec![]));
        let pool = make_pool("remote-pool", 2000.0);

        pm.apply_remote_pool(pool).await;

        let found = pm.get_pool("remote-pool").await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().status.last_renewed, 2000.0);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_does_not_overwrite_newer() {
        let pm = Arc::new(PoolManager::new(vec![]));

        // Insert a pool with last_renewed = 2000
        let newer = make_pool("test-pool", 2000.0);
        pm.apply_remote_pool(newer).await;

        // Try to apply an older version with last_renewed = 1000
        let older = make_pool("test-pool", 1000.0);
        pm.apply_remote_pool(older).await;

        // Should still have the newer version
        let found = pm.get_pool("test-pool").await.unwrap();
        assert_eq!(found.status.last_renewed, 2000.0);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_same_timestamp_keeps_original() {
        let pm = Arc::new(PoolManager::new(vec![]));

        // Insert a pool with last_renewed = 1000 and a specific bundle
        let mut original = make_pool("tie-pool", 1000.0);
        original.spec.bundle = Some("original-bundle".to_string());
        pm.apply_remote_pool(original).await;

        // Apply a remote pool with the same name but different data and same timestamp
        let mut incoming = make_pool("tie-pool", 1000.0);
        incoming.spec.bundle = Some("incoming-bundle".to_string());
        pm.apply_remote_pool(incoming).await;

        // Original pool data should be retained (not overwritten)
        let found = pm.get_pool("tie-pool").await.unwrap();
        assert_eq!(found.spec.bundle, Some("original-bundle".to_string()));
    }

    #[tokio::test]
    async fn test_remove_remote_pool() {
        let pm = Arc::new(PoolManager::new(vec![]));

        // Insert a pool
        let pool = make_pool("to-remove", 1000.0);
        pm.apply_remote_pool(pool).await;
        assert!(pm.get_pool("to-remove").await.is_some());

        // Remove it
        pm.remove_remote_pool("to-remove").await;
        assert!(pm.get_pool("to-remove").await.is_none());
    }

    #[tokio::test]
    async fn test_remove_remote_pool_protects_default() {
        let pm = Arc::new(PoolManager::new(vec!["l4-spot".to_string()]));
        pm.create_default_pool().await;

        pm.remove_remote_pool("default").await;

        // Default pool should still exist
        assert!(pm.get_pool("default").await.is_some());
    }
}
