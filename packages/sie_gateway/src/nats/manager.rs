//! NATS subscriber that feeds the in-memory `ModelRegistry`.
//!
//! `sie-config` is the sole publisher on `sie.config.models.*`; this manager
//! subscribes to `sie.config.models._all` and applies each incoming
//! `ConfigNotification` to the local registry. The gateway never publishes on
//! these subjects.
//!
//! This layer is intentionally fragile: NATS Core pub/sub has no replay, so
//! deltas published during a gateway↔NATS disconnect are lost. Reconciliation
//! is not handled here — `state::config_poller` detects drift against
//! `sie-config`'s `/v1/configs/epoch` and triggers a full
//! `GET /v1/configs/export` re-fetch.

use std::sync::Arc;
use std::time::Duration;

use async_nats::Client;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::metrics;
use crate::state::config_epoch::ConfigEpoch;
use crate::state::model_registry::ModelRegistry;
use crate::types::model::ModelConfig;

const SUBJECT_ALL: &str = "sie.config.models._all";

/// Default trusted-producer allowlist for `sie.config.models._all`. Only
/// `sie-config` is expected to publish on these subjects; every other
/// `producer_id` is dropped in `apply_notification`. Matched as exact
/// equality **or** K8s pod-name prefix (`sie-config` also matches
/// `sie-config-5f7b6d8c-kxwvr` / `sie-config-0`) — see
/// `is_trusted_producer`. Override via `new_with_trusted_producers`;
/// pass an empty `Vec` to disable validation (`main.rs` does this
/// behind `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER=true`).
///
/// Used only by `NatsManager::new` (a test convenience); the release
/// binary constructs the manager with a config-driven list. Annotated
/// `allow(dead_code)` outside `cfg(test)` so strict clippy passes.
#[cfg_attr(not(test), allow(dead_code))]
pub const DEFAULT_TRUSTED_PRODUCERS: &[&str] = &["sie-config"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigNotification {
    /// Identifier of the publisher that emitted this notification. In the
    /// current topology this is always `sie-config`. Accepts `router_id` as
    /// an alias because that is the key name Python publishes on the wire.
    #[serde(alias = "router_id")]
    pub producer_id: String,
    pub bundle_id: String,
    pub epoch: u64,
    pub bundle_config_hash: String,
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub profiles_added: Vec<String>,
    #[serde(default)]
    pub model_config: String,
    #[serde(default)]
    pub affected_bundles: Vec<String>,
}

pub struct NatsManager {
    client: RwLock<Option<Client>>,
    gateway_id: String,
    nats_url: String,
    model_registry: Arc<ModelRegistry>,
    config_epoch: ConfigEpoch,
    reconnect_notify: Arc<tokio::sync::Notify>,
    /// Allowlist of `producer_id` values whose notifications are applied.
    /// An empty `Vec` disables validation (trust any producer).
    trusted_producers: Vec<String>,
}

impl NatsManager {
    /// Construct a manager with the default trusted-producer allowlist.
    /// Test convenience; the release binary uses
    /// `new_with_trusted_producers` with the operator-driven list.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(
        gateway_id: String,
        nats_url: String,
        model_registry: Arc<ModelRegistry>,
        config_epoch: ConfigEpoch,
    ) -> Self {
        Self::new_with_trusted_producers(
            gateway_id,
            nats_url,
            model_registry,
            config_epoch,
            DEFAULT_TRUSTED_PRODUCERS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        )
    }

    /// Construct a manager with an explicit trusted-producer allowlist.
    /// Pass an empty `Vec` to disable producer validation.
    pub fn new_with_trusted_producers(
        gateway_id: String,
        nats_url: String,
        model_registry: Arc<ModelRegistry>,
        config_epoch: ConfigEpoch,
        trusted_producers: Vec<String>,
    ) -> Self {
        Self {
            client: RwLock::new(None),
            gateway_id,
            nats_url,
            model_registry,
            config_epoch,
            reconnect_notify: Arc::new(tokio::sync::Notify::new()),
            trusted_producers,
        }
    }

    /// Test-only: current trusted-producer allowlist.
    #[cfg(test)]
    pub fn trusted_producers(&self) -> &[String] {
        &self.trusted_producers
    }

    /// Identifier of this gateway replica, used by other subsystems (pools,
    /// inbox, queue) as the `router_id` component of their NATS subjects.
    pub fn router_id(&self) -> &str {
        &self.gateway_id
    }

    pub fn reconnect_notify(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.reconnect_notify)
    }

    pub async fn connect(self: &Arc<Self>) -> Result<(), async_nats::ConnectError> {
        if self.nats_url.is_empty() {
            info!("NATS URL not configured, skipping connection");
            return Ok(());
        }

        let reconnect_notify_clone = Arc::clone(&self.reconnect_notify);

        let client = async_nats::ConnectOptions::new()
            .retry_on_initial_connect()
            .connection_timeout(Duration::from_secs(5))
            .reconnect_delay_callback(|_attempts| Duration::from_secs(2))
            .event_callback(move |event| {
                let notify = reconnect_notify_clone.clone();
                async move {
                    match event {
                        async_nats::Event::Disconnected => {
                            warn!("NATS disconnected");
                            metrics::set_nats_connected(false);
                        }
                        async_nats::Event::Connected => {
                            info!("NATS reconnected");
                            metrics::set_nats_connected(true);
                            notify.notify_waiters();
                        }
                        async_nats::Event::SlowConsumer(id) => {
                            warn!(subscription = id, "NATS slow consumer detected");
                        }
                        async_nats::Event::ServerError(err) => {
                            error!(error = %err, "NATS server error");
                        }
                        async_nats::Event::ClientError(err) => {
                            error!(error = %err, "NATS client error");
                        }
                        other => {
                            info!(event = ?other, "NATS event");
                        }
                    }
                }
            })
            .connect(&self.nats_url)
            .await?;

        info!(url = %self.nats_url, gateway_id = %self.gateway_id, "connected to NATS");
        // Initial connect: flip the gauge to 1. Subsequent
        // `Disconnected` / `Connected` events update it via the
        // event callback above.
        metrics::set_nats_connected(true);
        *self.client.write().await = Some(client);

        Ok(())
    }

    pub async fn get_client(&self) -> Option<Client> {
        self.client.read().await.clone()
    }

    pub async fn start_subscription(self: &Arc<Self>) {
        let guard = self.client.read().await;
        let client = match guard.as_ref() {
            Some(c) => c.clone(),
            None => return,
        };
        drop(guard);

        let subscriber = match client.subscribe(SUBJECT_ALL.to_string()).await {
            Ok(s) => s,
            Err(e) => {
                error!(error = %e, "failed to subscribe to {}", SUBJECT_ALL);
                return;
            }
        };

        let manager = Arc::clone(self);
        tokio::spawn(async move {
            manager.handle_subscription(subscriber).await;
        });

        info!(subject = SUBJECT_ALL, "NATS subscription started");
    }

    async fn handle_subscription(&self, mut subscriber: async_nats::Subscriber) {
        while let Some(msg) = subscriber.next().await {
            let notification: ConfigNotification = match serde_json::from_slice(&msg.payload) {
                Ok(n) => n,
                Err(e) => {
                    warn!(error = %e, "failed to parse config notification");
                    continue;
                }
            };

            info!(
                from = %notification.producer_id,
                bundle = %notification.bundle_id,
                epoch = notification.epoch,
                "received config notification"
            );

            self.apply_notification(&notification).await;
        }

        warn!("NATS subscription ended");
    }

    /// Returns `true` when `producer_id` is allowed to publish config
    /// notifications. An empty allowlist means "trust everyone".
    ///
    /// Matches exact equality **or** a Kubernetes-style pod-name prefix
    /// (`"sie-config"` matches both `"sie-config"` and
    /// `"sie-config-5f7b6d8c-kxwvr"`). Python's `NatsPublisher` sources
    /// the `router_id` from `POD_NAME`, which is the Deployment- or
    /// StatefulSet-managed pod name (e.g. `sie-config-5f7b6d8c-kxwvr`
    /// or `sie-config-0`), so an exact match alone would reject every
    /// notification in a real cluster. The prefix check requires a
    /// literal `-` separator so `sie-config` does not match, e.g.,
    /// `sie-configuration`.
    fn is_trusted_producer(&self, producer_id: &str) -> bool {
        if self.trusted_producers.is_empty() {
            return true;
        }
        self.trusted_producers.iter().any(|trusted| {
            trusted == producer_id
                || producer_id
                    .strip_prefix(trusted.as_str())
                    .is_some_and(|rest| rest.starts_with('-'))
        })
    }

    /// Apply a single incoming notification to the in-memory registry.
    ///
    /// The registry's append-only semantics handle duplicate/stale replays
    /// safely (equal profiles are skipped, truly divergent ones surface as
    /// append-only conflicts, which we log as warnings rather than crashing).
    ///
    /// Epoch-advance policy: `ConfigEpoch` is **only** advanced when the
    /// delta either (a) has no body and is therefore a pure epoch bump, or
    /// (b) parses cleanly and the registry accepts it. If the body fails to
    /// parse or the registry rejects it, the epoch stays where it is so
    /// `state::config_poller` still sees drift (local < remote) and triggers
    /// a full re-export. This prevents a malformed or schema-incompatible
    /// delta from silently claiming we're caught up while the registry
    /// actually missed the change.
    ///
    /// Notifications from untrusted producers are dropped before any
    /// registry mutation or epoch advance; see `is_trusted_producer`.
    async fn apply_notification(&self, notification: &ConfigNotification) {
        // Decide the `kind` label once up-front so every outcome
        // branch attaches the same, accurate kind to its counter:
        // an empty `model_config` is a pure epoch bump regardless of
        // which branch we end up in.
        let kind = if notification.model_config.trim().is_empty() {
            "epoch_bump"
        } else {
            "model_added"
        };

        if !self.is_trusted_producer(&notification.producer_id) {
            warn!(
                producer_id = %notification.producer_id,
                trusted_producers = ?self.trusted_producers,
                epoch = notification.epoch,
                model = %notification.model_id,
                "rejecting config notification from untrusted producer; epoch NOT advanced"
            );
            metrics::CONFIG_DELTAS
                .with_label_values(&[kind, "rejected_untrusted"])
                .inc();
            return;
        }

        if kind == "epoch_bump" {
            // Pure epoch bump (no config body to apply). Advance the
            // counter so the poller doesn't treat this as drift.
            self.config_epoch.set_max(notification.epoch);
            metrics::CONFIG_DELTAS
                .with_label_values(&[kind, "applied"])
                .inc();
            return;
        }

        let config: ModelConfig = match serde_yaml::from_str(&notification.model_config) {
            Ok(config) => config,
            Err(e) => {
                warn!(
                    error = %e,
                    model = %notification.model_id,
                    epoch = notification.epoch,
                    "failed to parse model_config from config notification; epoch NOT advanced (poller will recover)"
                );
                metrics::CONFIG_DELTAS
                    .with_label_values(&[kind, "parse_error"])
                    .inc();
                return;
            }
        };

        match self.model_registry.add_model_config(config) {
            Ok(_) => {
                self.config_epoch.set_max(notification.epoch);
                metrics::CONFIG_DELTAS
                    .with_label_values(&[kind, "applied"])
                    .inc();
            }
            Err(e) => {
                warn!(
                    error = %e,
                    model = %notification.model_id,
                    epoch = notification.epoch,
                    "failed to apply model_config from config notification; epoch NOT advanced (poller will recover)"
                );
                metrics::CONFIG_DELTAS
                    .with_label_values(&[kind, "apply_error"])
                    .inc();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;

    #[test]
    fn test_config_notification_serde() {
        let notification = ConfigNotification {
            producer_id: "sie-config".to_string(),
            bundle_id: "default".to_string(),
            epoch: 12345,
            bundle_config_hash: "abc123".to_string(),
            model_id: "BAAI/bge-m3".to_string(),
            profiles_added: vec!["default".to_string(), "fast".to_string()],
            model_config: "name: bge-m3\nprofiles: {}\n".to_string(),
            affected_bundles: vec!["default".to_string(), "premium".to_string()],
        };

        let json = serde_json::to_string(&notification).unwrap();
        let parsed: ConfigNotification = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.producer_id, "sie-config");
        assert_eq!(parsed.bundle_id, "default");
        assert_eq!(parsed.epoch, 12345);
        assert_eq!(parsed.bundle_config_hash, "abc123");
        assert_eq!(parsed.model_id, "BAAI/bge-m3");
        assert_eq!(parsed.profiles_added, vec!["default", "fast"]);
        assert_eq!(parsed.affected_bundles, vec!["default", "premium"]);
    }

    #[test]
    fn test_config_notification_accepts_router_id_wire_key() {
        // Python's `sie_config.nats_publisher` emits the publisher identity
        // under the key `router_id`. The Rust struct stores it in
        // `producer_id` via `#[serde(alias = "router_id")]`.
        let json = r#"{
            "router_id": "sie-config-abc",
            "bundle_id": "default",
            "epoch": 1,
            "bundle_config_hash": "x"
        }"#;
        let parsed: ConfigNotification = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.producer_id, "sie-config-abc");
        assert!(parsed.model_id.is_empty());
        assert!(parsed.profiles_added.is_empty());
        assert!(parsed.model_config.is_empty());
        assert!(parsed.affected_bundles.is_empty());
    }

    #[test]
    fn test_reconnect_notify_exists() {
        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            "", "", false,
        ));
        let mgr = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            registry,
            ConfigEpoch::new(),
        );
        let notify = mgr.reconnect_notify();
        notify.notify_waiters();
    }

    #[test]
    fn test_default_trusted_producers_is_sie_config() {
        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            "", "", false,
        ));
        let mgr = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            registry,
            ConfigEpoch::new(),
        );
        assert_eq!(mgr.trusted_producers(), &["sie-config".to_string()]);
        // Exact match (non-K8s / local process): sie-config is allowed.
        assert!(mgr.is_trusted_producer("sie-config"));
        // K8s Deployment pod-name prefix (`{deploy-name}-{hash}-{rand}`).
        assert!(mgr.is_trusted_producer("sie-config-5f7b6d8c-kxwvr"));
        // K8s StatefulSet pod-name prefix (`{sts-name}-{ordinal}`).
        assert!(mgr.is_trusted_producer("sie-config-0"));
        // Dash-separator is required: substring without dash does NOT match.
        assert!(!mgr.is_trusted_producer("sie-configuration"));
        // Random unrelated publisher is rejected.
        assert!(!mgr.is_trusted_producer("evil-publisher"));
        assert!(!mgr.is_trusted_producer(""));
    }

    #[test]
    fn test_empty_allowlist_trusts_everyone() {
        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            "", "", false,
        ));
        let mgr = NatsManager::new_with_trusted_producers(
            "gw1".to_string(),
            String::new(),
            registry,
            ConfigEpoch::new(),
            Vec::new(),
        );
        assert!(mgr.is_trusted_producer("sie-config"));
        assert!(mgr.is_trusted_producer("anyone-else"));
        assert!(mgr.is_trusted_producer(""));
    }

    #[test]
    fn test_custom_allowlist_blocks_non_members() {
        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            "", "", false,
        ));
        let mgr = NatsManager::new_with_trusted_producers(
            "gw1".to_string(),
            String::new(),
            registry,
            ConfigEpoch::new(),
            vec!["sie-config".to_string(), "staging-config".to_string()],
        );
        assert!(mgr.is_trusted_producer("sie-config"));
        assert!(mgr.is_trusted_producer("staging-config"));
        // Prefix match applies to every entry in the allowlist.
        assert!(mgr.is_trusted_producer("staging-config-0"));
        assert!(mgr.is_trusted_producer("sie-config-abc"));
        assert!(!mgr.is_trusted_producer("evil-publisher"));
        // No dash separator → no prefix match.
        assert!(!mgr.is_trusted_producer("sie-configuration"));
    }

    #[tokio::test]
    async fn test_apply_notification_rejects_untrusted_producer() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let bundles_dir = temp_dir.path().join("bundles");
        let models_dir = temp_dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));
        let epoch = ConfigEpoch::new();
        let manager = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            Arc::clone(&registry),
            epoch.clone(),
        );

        // Pure epoch bump from an untrusted producer — must NOT advance
        // the local epoch (that is the core attack surface: a forged
        // bump would wedge config_poller's catch-up trigger).
        let notification = ConfigNotification {
            producer_id: "rogue-publisher".to_string(),
            bundle_id: "default".to_string(),
            epoch: 99,
            bundle_config_hash: String::new(),
            model_id: String::new(),
            profiles_added: Vec::new(),
            model_config: String::new(),
            affected_bundles: Vec::new(),
        };
        manager.apply_notification(&notification).await;
        assert_eq!(
            epoch.get(),
            0,
            "untrusted producer must not advance the config epoch"
        );

        // And the same producer sending a real body must also be dropped.
        let notification_with_body = ConfigNotification {
            producer_id: "rogue-publisher".to_string(),
            bundle_id: "default".to_string(),
            epoch: 1,
            bundle_config_hash: String::new(),
            model_id: "evil/model".to_string(),
            profiles_added: vec!["default".to_string()],
            model_config: r#"
sie_id: evil/model
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"#
            .to_string(),
            affected_bundles: vec!["default".to_string()],
        };
        manager.apply_notification(&notification_with_body).await;
        assert_eq!(
            registry.get_model_profile_names("evil/model").len(),
            0,
            "untrusted producer must not be able to register a model"
        );
        assert_eq!(
            epoch.get(),
            0,
            "untrusted producer must not advance the config epoch"
        );
    }

    #[tokio::test]
    async fn test_apply_notification_applies_new_profile() {
        use crate::types::model::{ModelConfig, ProfileConfig};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let bundles_dir = temp_dir.path().join("bundles");
        let models_dir = temp_dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));

        // Pre-seed an existing config so the test doubles as a no-crash check
        // for the append-only pathway.
        let seed = ModelConfig {
            name: "test/model".to_string(),
            adapter_module: None,
            default_bundle: None,
            profiles: {
                let mut profiles = HashMap::new();
                profiles.insert(
                    "default".to_string(),
                    ProfileConfig {
                        adapter_path: Some(
                            "sie_server.adapters.sentence_transformer:Adapter".to_string(),
                        ),
                        max_batch_tokens: Some(4096),
                        compute_precision: None,
                        adapter_options: None,
                        extends: None,
                    },
                );
                profiles
            },
        };
        registry.add_model_config(seed).unwrap();

        let manager = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            Arc::clone(&registry),
            ConfigEpoch::new(),
        );

        let notification = ConfigNotification {
            producer_id: "sie-config".to_string(),
            bundle_id: "default".to_string(),
            epoch: 3,
            bundle_config_hash: "hash3".to_string(),
            model_id: "test/model".to_string(),
            profiles_added: vec!["fast".to_string()],
            model_config: r#"
sie_id: test/model
profiles:
  fast:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 8192
"#
            .to_string(),
            affected_bundles: vec!["default".to_string()],
        };

        manager.apply_notification(&notification).await;

        let mut profiles = registry.get_model_profile_names("test/model");
        profiles.sort();
        assert_eq!(profiles, vec!["default".to_string(), "fast".to_string()]);
    }

    #[tokio::test]
    async fn test_apply_notification_empty_body_is_noop() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let bundles_dir = temp_dir.path().join("bundles");
        let models_dir = temp_dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();

        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));
        let manager = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            Arc::clone(&registry),
            ConfigEpoch::new(),
        );

        let notification = ConfigNotification {
            producer_id: "sie-config".to_string(),
            bundle_id: "default".to_string(),
            epoch: 1,
            bundle_config_hash: "h".to_string(),
            model_id: "m".to_string(),
            profiles_added: vec![],
            model_config: "   \n\t".to_string(),
            affected_bundles: vec![],
        };

        manager.apply_notification(&notification).await;
        assert!(registry.get_model_profile_names("m").is_empty());
    }

    #[tokio::test]
    async fn test_apply_notification_bad_yaml_is_logged_not_crash() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let bundles_dir = temp_dir.path().join("bundles");
        let models_dir = temp_dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();

        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));
        let manager = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            Arc::clone(&registry),
            ConfigEpoch::new(),
        );

        let notification = ConfigNotification {
            producer_id: "sie-config".to_string(),
            bundle_id: "default".to_string(),
            epoch: 1,
            bundle_config_hash: "h".to_string(),
            model_id: "malformed/model".to_string(),
            profiles_added: vec![],
            model_config: "::: not valid yaml :::".to_string(),
            affected_bundles: vec![],
        };

        manager.apply_notification(&notification).await;
        assert!(registry
            .get_model_profile_names("malformed/model")
            .is_empty());
    }

    #[tokio::test]
    async fn test_apply_notification_ignores_unknown_fields() {
        let json = r#"{
            "producer_id": "sie-config",
            "bundle_id": "default",
            "epoch": 1,
            "bundle_config_hash": "x",
            "model_id": "unknown/model",
            "future_field_we_dont_know_about": 42,
            "model_config": ""
        }"#;
        let parsed: ConfigNotification = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.producer_id, "sie-config");
        assert_eq!(parsed.model_id, "unknown/model");
    }

    #[tokio::test]
    async fn test_apply_notification_advances_epoch_on_pure_bump() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let bundles_dir = temp_dir.path().join("bundles");
        let models_dir = temp_dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));
        let epoch = ConfigEpoch::new();
        let manager = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            Arc::clone(&registry),
            epoch.clone(),
        );

        // Empty body → treated as a pure epoch bump, counter advances.
        let notification = ConfigNotification {
            producer_id: "sie-config".to_string(),
            bundle_id: "default".to_string(),
            epoch: 42,
            bundle_config_hash: "h".to_string(),
            model_id: "m".to_string(),
            profiles_added: vec![],
            model_config: String::new(),
            affected_bundles: vec![],
        };
        manager.apply_notification(&notification).await;
        assert_eq!(epoch.get(), 42);

        // A lower-epoch delta must not move the counter backward.
        let older = ConfigNotification {
            epoch: 7,
            ..notification.clone()
        };
        manager.apply_notification(&older).await;
        assert_eq!(epoch.get(), 42);
    }

    #[tokio::test]
    async fn test_apply_notification_does_not_advance_epoch_on_bad_yaml() {
        // If the notification body fails to parse, the epoch must NOT
        // advance. Otherwise the poller would treat the replica as
        // caught up and skip the re-export that would actually heal the
        // registry.
        let temp_dir = tempfile::TempDir::new().unwrap();
        let bundles_dir = temp_dir.path().join("bundles");
        let models_dir = temp_dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));
        let epoch = ConfigEpoch::new();
        let manager = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            Arc::clone(&registry),
            epoch.clone(),
        );

        let notification = ConfigNotification {
            producer_id: "sie-config".to_string(),
            bundle_id: "default".to_string(),
            epoch: 99,
            bundle_config_hash: "h".to_string(),
            model_id: "broken/model".to_string(),
            profiles_added: vec![],
            model_config: "::: not yaml :::".to_string(),
            affected_bundles: vec!["default".to_string()],
        };
        manager.apply_notification(&notification).await;
        assert_eq!(
            epoch.get(),
            0,
            "epoch must stay at 0 so the poller still detects drift"
        );
        assert!(
            registry.get_model_profile_names("broken/model").is_empty(),
            "broken config should not leak into registry"
        );
    }

    #[tokio::test]
    async fn test_apply_notification_does_not_advance_epoch_on_registry_reject() {
        // Apply-reject branch: if `add_model_config` returns an error
        // (e.g. append-only conflict, unroutable adapter), the epoch
        // must NOT advance.
        let temp_dir = tempfile::TempDir::new().unwrap();
        let bundles_dir = temp_dir.path().join("bundles");
        let models_dir = temp_dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = Arc::new(crate::state::model_registry::ModelRegistry::new(
            &bundles_dir,
            &models_dir,
            true,
        ));
        let epoch = ConfigEpoch::new();
        let manager = NatsManager::new(
            "gw1".to_string(),
            String::new(),
            Arc::clone(&registry),
            epoch.clone(),
        );

        // adapter_path points at a module no bundle advertises — registry
        // will reject with "Adapter(s) not in any known bundle".
        let notification = ConfigNotification {
            producer_id: "sie-config".to_string(),
            bundle_id: "default".to_string(),
            epoch: 55,
            bundle_config_hash: "h".to_string(),
            model_id: "reject/model".to_string(),
            profiles_added: vec![],
            model_config: r#"
sie_id: reject/model
profiles:
  default:
    adapter_path: unknown.module:Adapter
"#
            .to_string(),
            affected_bundles: vec!["default".to_string()],
        };
        manager.apply_notification(&notification).await;
        assert_eq!(
            epoch.get(),
            0,
            "epoch must stay at 0 so the poller still detects drift"
        );
    }
}
