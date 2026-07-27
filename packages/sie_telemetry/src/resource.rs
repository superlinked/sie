//! OTel resource identity shared by every SIE Rust producer.
//!
//! `service.name` plus `service.instance.id`, `deployment.environment`,
//! `cloud.region` (OTel semantic conventions), and — when the consumer
//! passes its `CARGO_PKG_VERSION` — `service.version`. Deployments inject
//! the real values; local processes retain the complete contract with
//! `unknown`.

use std::sync::OnceLock;

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;

use crate::env::cleaned_env;

pub const UNKNOWN_RESOURCE_VALUE: &str = "unknown";

/// Build the contract resource from explicit values.
///
/// Uses an explicit (empty-builder) resource: the default builder runs the
/// `OTEL_RESOURCE_ATTRIBUTES` detector; an injected `service.namespace`
/// would change Prometheus `job` and silently disconnect KEDA selectors.
///
/// `service_version` is the CONSUMER's `env!("CARGO_PKG_VERSION")` — it must
/// be passed in, never read here, or every crate would report this crate's
/// version. `None` omits the attribute (the sidecar's historical shape).
pub fn resource_from_values(
    service_name: &str,
    instance_id: &str,
    deployment_environment: &str,
    cloud_region: &str,
    service_version: Option<&str>,
) -> Resource {
    let mut attributes = vec![
        KeyValue::new("service.instance.id", instance_id.to_string()),
        KeyValue::new("deployment.environment", deployment_environment.to_string()),
        KeyValue::new("cloud.region", cloud_region.to_string()),
    ];
    if let Some(version) = service_version {
        attributes.push(KeyValue::new("service.version", version.to_string()));
    }
    Resource::builder_empty()
        .with_service_name(service_name.to_string())
        .with_attributes(attributes)
        .build()
}

/// The shared substrate-instance prefix chain: explicit override, then the
/// Modal task id.
pub fn instance_prefix_env() -> Option<String> {
    cleaned_env("SIE_TELEMETRY_INSTANCE_ID").or_else(|| cleaned_env("MODAL_TASK_ID"))
}

/// Compose stable substrate placement with a process-start UUID. The suffix
/// prevents a restarted container from refreshing the previous process's
/// freshness series when the pod/container prefix is reused.
pub fn service_instance_id(configured_prefix: Option<&str>) -> String {
    let process_start_uuid = process_start_uuid();
    compose_service_instance_id(configured_prefix, &process_start_uuid)
}

pub fn compose_service_instance_id(
    configured_prefix: Option<&str>,
    process_start_uuid: &str,
) -> String {
    configured_prefix
        .map(str::trim)
        .map(|prefix| prefix.trim_end_matches('/'))
        .filter(|prefix| !prefix.is_empty())
        .map(|prefix| format!("{prefix}/{process_start_uuid}"))
        .unwrap_or_else(|| process_start_uuid.to_string())
}

/// Stable within this process and unique across process starts.
pub fn process_start_uuid() -> String {
    static PROCESS_START_UUID: OnceLock<String> = OnceLock::new();
    PROCESS_START_UUID
        .get_or_init(|| uuid::Uuid::new_v4().to_string())
        .clone()
}

#[cfg(test)]
mod tests {
    use opentelemetry::Key;

    use super::*;

    #[test]
    fn resource_contains_the_complete_identity_with_optional_version() {
        let resource =
            resource_from_values("sie-worker", "boot-123", "dev", "us-east-1", Some("9.9.9"));
        assert_eq!(
            resource.get(&Key::new("service.name")),
            Some("sie-worker".into())
        );
        assert_eq!(
            resource.get(&Key::new("service.instance.id")),
            Some("boot-123".into())
        );
        assert_eq!(
            resource.get(&Key::new("deployment.environment")),
            Some("dev".into())
        );
        assert_eq!(
            resource.get(&Key::new("cloud.region")),
            Some("us-east-1".into())
        );
        assert_eq!(
            resource.get(&Key::new("service.version")),
            Some("9.9.9".into())
        );

        let versionless = resource_from_values("s", "i", "d", "r", None);
        assert_eq!(versionless.get(&Key::new("service.version")), None);
    }

    #[test]
    fn fallback_instance_id_is_stable_and_uuid_shaped() {
        let first = service_instance_id(None);
        let second = service_instance_id(None);
        assert_eq!(first, second);
        assert!(uuid::Uuid::parse_str(&first).is_ok());
    }

    #[test]
    fn configured_prefix_is_stable_per_process_and_changes_on_restart() {
        let current = service_instance_id(Some(" modal-task/worker/ "));
        let repeated = service_instance_id(Some("modal-task/worker"));
        assert_eq!(current, repeated);
        let suffix = current
            .strip_prefix("modal-task/worker/")
            .expect("configured substrate prefix");
        assert!(uuid::Uuid::parse_str(suffix).is_ok());

        let first_process = uuid::Uuid::new_v4().to_string();
        let restarted_process = uuid::Uuid::new_v4().to_string();
        assert_ne!(
            compose_service_instance_id(Some("modal-task/worker"), &first_process),
            compose_service_instance_id(Some("modal-task/worker"), &restarted_process)
        );
    }
}
