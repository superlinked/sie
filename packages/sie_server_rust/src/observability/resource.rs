//! Resource identity shared by every Rust worker OpenTelemetry signal.
//!
//! Identity composition lives in `sie-telemetry` (#2339); this module keeps
//! the worker's env-precedence chains and service identity.

use opentelemetry_sdk::Resource;

pub(crate) use sie_telemetry::env::cleaned_env;
use sie_telemetry::resource::{
    instance_prefix_env, resource_from_values, service_instance_id, UNKNOWN_RESOURCE_VALUE,
};

pub const SERVICE_NAME: &str = "sie-worker";

/// Build the process resource shared by traces and metrics.
pub fn telemetry_resource() -> Resource {
    let instance_id = service_instance_id(instance_prefix_env().as_deref());
    let deployment_environment = cleaned_env("SIE_OTEL_DEPLOYMENT_ENVIRONMENT")
        .or_else(|| cleaned_env("SIE_DEPLOYMENT_ENV"))
        .unwrap_or_else(|| UNKNOWN_RESOURCE_VALUE.to_string());
    let cloud_region = cleaned_env("SIE_OTEL_CLOUD_REGION")
        .or_else(|| cleaned_env("SIE_CLOUD_REGION"))
        .or_else(|| cleaned_env("AWS_REGION"))
        .or_else(|| cleaned_env("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|| UNKNOWN_RESOURCE_VALUE.to_string());

    resource_from_values(
        SERVICE_NAME,
        &instance_id,
        &deployment_environment,
        &cloud_region,
        Some(env!("CARGO_PKG_VERSION")),
    )
}
