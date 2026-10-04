use std::sync::Arc;
use std::time::Duration;

use async_nats::jetstream;
use reqwest::Url;

use crate::config::{Config, NatsCredentials};
use crate::server::AppState;
use crate::state::model_registry::ModelRegistryGeneration;
use crate::state::threshold_coordinator::{
    ThresholdCoordinator, ThresholdSampler, THRESHOLD_SAMPLE_INTERVAL,
};

pub struct ThresholdSettings {
    url: String,
    credentials: NatsCredentials,
    replicas: usize,
}

impl ThresholdSettings {
    pub fn from_env(config: &Config) -> Result<Option<Self>, String> {
        let flag =
            std::env::var("SIE_THRESHOLD_ROUTING_ENABLED").unwrap_or_else(|_| "false".into());
        Self::parse(
            &flag,
            &std::env::var("SIE_THRESHOLD_NATS_URL").unwrap_or_default(),
            &std::env::var("SIE_THRESHOLD_NATS_REPLICAS").unwrap_or_else(|_| "1".into()),
            config,
        )
    }

    fn parse(
        flag: &str,
        url: &str,
        replicas: &str,
        config: &Config,
    ) -> Result<Option<Self>, String> {
        match flag {
            "false" => return Ok(None),
            "true" => {}
            _ => return Err("SIE_THRESHOLD_ROUTING_ENABLED must be true or false".into()),
        }
        let parsed = Url::parse(url)
            .map_err(|_| "threshold control endpoint must be an absolute NATS URL")?;
        if !matches!(parsed.scheme(), "nats" | "tls")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
            || !matches!(parsed.path(), "" | "/")
        {
            return Err(
                "threshold control endpoint must be a NATS URL without credentials or extra fields"
                    .into(),
            );
        }
        let endpoint = url
            .parse::<async_nats::ServerAddr>()
            .map_err(|_| "invalid threshold control endpoint")?;
        for server in config.nats_url.split(',') {
            let queue = server
                .parse::<async_nats::ServerAddr>()
                .map_err(|_| "invalid inference NATS endpoint")?;
            if queue.host() == endpoint.host() && queue.port() == endpoint.port() {
                return Err("threshold state requires a separate broker endpoint".into());
            }
        }
        let credentials = config
            .nats_credentials()?
            .ok_or("threshold control endpoint requires gateway authentication")?;
        let replicas = replicas
            .parse::<usize>()
            .map_err(|_| "threshold replicas must be 1 or 3")?;
        if !matches!(replicas, 1 | 3) {
            return Err("threshold replicas must be 1 or 3".into());
        }
        Ok(Some(Self {
            url: url.to_string(),
            credentials,
            replicas,
        }))
    }
}

pub(crate) struct ThresholdBinding {
    pub generation: ModelRegistryGeneration,
    pub epoch: u64,
    pub coordinator: ThresholdCoordinator,
}

/// Own the isolated connection and sampler. Registry pointer fencing prevents
/// any decision from a superseded snapshot from suppressing local warm-up.
pub fn spawn_threshold_runtime(
    state: Arc<AppState>,
    settings: ThresholdSettings,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut context = None;
        let mut binding: Option<Arc<ThresholdBinding>> = None;
        let mut sampler = ThresholdSampler::default();
        let mut ticks = tokio::time::interval(THRESHOLD_SAMPLE_INTERVAL);
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticks.tick().await;
            let epoch = state.config_epoch.get();
            if epoch == 0 || !state.config_epoch.is_bootstrapped() {
                state.model_registry.clear_threshold_binding();
                binding = None;
                continue;
            }
            if context.is_none() {
                let connect = async_nats::ConnectOptions::new()
                    .user_and_password(
                        settings.credentials.user.clone(),
                        settings.credentials.password.clone(),
                    )
                    .connection_timeout(Duration::from_secs(2))
                    .connect(&settings.url);
                context = match tokio::time::timeout(Duration::from_secs(2), connect).await {
                    Ok(Ok(client)) => Some(jetstream::new(client)),
                    _ => continue,
                };
            }
            if binding.as_ref().is_none_or(|old| {
                old.epoch != epoch
                    || state
                        .model_registry
                        .with_current_generation(&old.generation, || ())
                        .is_none()
            }) {
                state.model_registry.clear_threshold_binding();
                binding = None;
                sampler = ThresholdSampler::default();
                let Ok((generation, targets)) = state.model_registry.threshold_targets(epoch)
                else {
                    continue;
                };
                if targets.is_empty() {
                    continue;
                }
                let Ok(coordinator) = ThresholdCoordinator::connect(
                    context.as_ref().expect("connected context"),
                    settings.replicas,
                    targets,
                )
                .await
                else {
                    continue;
                };
                if state.config_epoch.get() != epoch || !state.config_epoch.is_bootstrapped() {
                    continue;
                }
                let current = Arc::new(ThresholdBinding {
                    generation,
                    epoch,
                    coordinator,
                });
                state
                    .model_registry
                    .install_threshold_binding(Arc::clone(&current));
                binding = Some(current);
            }
            if let Some(current) = &binding {
                // Standby sampling normally returns unavailable; its requests
                // still read decisions bound to the elected sampler's lease.
                let _ = current.coordinator.sample(&mut sampler).await;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn nonzero_delta_epoch_cannot_acquire_authority_before_complete_bootstrap() {
        use crate::handlers::test_support::{TestGateway, ThresholdBroker, HYBRID_GENERATE_MODEL};
        let Some(broker) = ThresholdBroker::start().await else {
            return;
        };
        let model = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: threshold\n  fallback_profile: remote\n  wake_above: 1\n  sleep_below: 0.5\n  window_s: 1\n  cooldown_s: 1\n"
        );
        let gateway = TestGateway::with_threshold_routing(&[&model], true).await;
        gateway.state.config_epoch.set_max(7);
        let runtime = spawn_threshold_runtime(
            Arc::clone(&gateway.state),
            ThresholdSettings {
                url: format!(
                    "nats://127.0.0.1:{}",
                    broker.context.client().server_info().port
                ),
                credentials: NatsCredentials {
                    user: "sie-gateway".into(),
                    password: "GatewayThresholdTestPassword0123456789".into(),
                },
                replicas: 1,
            },
        );
        tokio::time::sleep(Duration::from_millis(1100)).await;
        assert!(!gateway.state.config_epoch.is_bootstrapped());
        assert!(broker
            .context
            .get_key_value("SIE_THRESHOLD_COUNTS")
            .await
            .is_err());
        assert!(gateway
            .state
            .model_registry
            .threshold_remote_route("acme/chat", 7)
            .is_none());
        gateway.state.config_epoch.mark_bootstrapped();
        tokio::time::timeout(Duration::from_secs(4), async {
            loop {
                if broker
                    .context
                    .get_key_value("SIE_THRESHOLD_LEASE")
                    .await
                    .is_ok()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("complete bootstrap allows coordination");
        runtime.abort();
        let _ = runtime.await;
    }

    #[test]
    fn flag_requires_an_isolated_authenticated_control_endpoint() {
        let mut config = Config::load();
        config.nats_url = "nats://queue.example:4222".into();
        config.nats_user = "sie-gateway".into();
        config.nats_password = "gateway-test-password".into();
        assert!(ThresholdSettings::parse("false", "", "", &config)
            .unwrap()
            .is_none());
        assert!(
            ThresholdSettings::parse("true", "nats://control.example:4222", "1", &config)
                .unwrap()
                .is_some()
        );
        for url in [
            "nats://queue.example:4222",
            "nats://user:secret@control.example:4222",
            "nats://control.example:4222?token=secret",
            "https://control.example:4222",
            "nats://control.example:4222/path",
        ] {
            assert!(ThresholdSettings::parse("true", url, "1", &config).is_err());
        }
        assert!(ThresholdSettings::parse("yes", "", "1", &config).is_err());
        assert!(
            ThresholdSettings::parse("true", "nats://control.example:4222", "2", &config).is_err()
        );
        config.nats_password.clear();
        assert!(
            ThresholdSettings::parse("true", "nats://control.example:4222", "1", &config).is_err()
        );
    }
}
