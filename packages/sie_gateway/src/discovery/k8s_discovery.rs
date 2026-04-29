use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use futures_util::TryStreamExt;
use k8s_openapi::api::core::v1::Endpoints;
use kube::api::{Api, ListParams};
use kube::runtime::watcher::{self, Event};
use kube::Client;
use rand::prelude::*;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::discovery::ws_health::WsHealthManager;

pub struct K8sDiscovery {
    client: Client,
    namespace: String,
    service: String,
    port: u16,
    ws_manager: Arc<WsHealthManager>,
    /// Track known worker URLs to detect stale workers on Apply events.
    known_urls: Mutex<HashSet<String>>,
}

impl K8sDiscovery {
    pub async fn new(
        namespace: &str,
        service: &str,
        port: u16,
        ws_manager: Arc<WsHealthManager>,
    ) -> Result<Self, kube::Error> {
        let client = Client::try_default().await?;
        Ok(Self {
            client,
            namespace: namespace.to_string(),
            service: service.to_string(),
            port,
            ws_manager,
            known_urls: Mutex::new(HashSet::new()),
        })
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
                    // Watch ended cleanly (e.g., shutdown)
                    info!("k8s watch ended");
                    return;
                }
                Err(e) => {
                    warn!(error = %e, "k8s watch error, will retry");

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
        let endpoints: Api<Endpoints> = Api::namespaced(self.client.clone(), &self.namespace);
        let _lp = ListParams::default().fields(&format!("metadata.name={}", self.service));

        info!(
            namespace = %self.namespace,
            service = %self.service,
            "starting k8s endpoints watch"
        );

        let stream = watcher::watcher(
            endpoints,
            watcher::Config::default()
                .labels("")
                .fields(&format!("metadata.name={}", self.service)),
        );

        futures_util::pin_mut!(stream);

        while let Some(event) = stream.try_next().await? {
            match event {
                Event::Apply(ep) | Event::InitApply(ep) => {
                    self.handle_endpoints_update(&ep).await;
                }
                Event::Delete(ep) => {
                    self.handle_endpoints_delete(&ep).await;
                }
                Event::Init => {
                    info!("k8s watch initial list starting");
                }
                Event::InitDone => {
                    info!("k8s watch initial list complete");
                }
            }
        }

        Ok(())
    }

    async fn handle_endpoints_update(&self, endpoints: &Endpoints) {
        let new_urls: HashSet<String> = self.extract_worker_urls(endpoints).into_iter().collect();

        // Diff against known URLs to remove stale workers
        let mut known = self.known_urls.lock().await;
        let stale: Vec<String> = known.difference(&new_urls).cloned().collect();
        for url in &stale {
            self.ws_manager.remove_worker(url).await;
        }
        if !stale.is_empty() {
            info!(count = stale.len(), "removed stale k8s workers");
        }

        for url in &new_urls {
            self.ws_manager.add_worker(url.clone()).await;
        }

        info!(count = new_urls.len(), "k8s endpoints updated");
        *known = new_urls;
    }

    async fn handle_endpoints_delete(&self, endpoints: &Endpoints) {
        let urls = self.extract_worker_urls(endpoints);
        info!(count = urls.len(), "k8s endpoints deleted");

        for url in &urls {
            self.ws_manager.remove_worker(url).await;
        }
    }

    fn extract_worker_urls(&self, endpoints: &Endpoints) -> Vec<String> {
        let mut urls = Vec::new();

        if let Some(subsets) = &endpoints.subsets {
            for subset in subsets {
                if let Some(addresses) = &subset.addresses {
                    for addr in addresses {
                        let url = format!("http://{}:{}", addr.ip, self.port);
                        urls.push(url);
                    }
                }
            }
        }

        urls
    }
}
