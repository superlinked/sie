//! OTLP span/metric exporter construction for the selected transport.
//!
//! HTTP carries Modal proxy authentication when configured; gRPC keeps the
//! historical in-cluster path. Metric exporters pin LowMemory (delta)
//! temporality per the telemetry contract.

use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::metrics::Temporality;

use crate::env::cleaned_env;
use crate::proxy::{
    authenticated_http_client, modal_proxy_auth_enabled, modal_proxy_headers_from_values,
    validate_modal_proxy_transport,
};
use crate::transport::{OtlpProtocol, SignalExportConfig};

/// Build the span exporter, reading the Modal proxy posture from the
/// environment.
pub fn build_span_exporter(
    config: &SignalExportConfig,
) -> Result<opentelemetry_otlp::SpanExporter, String> {
    build_span_exporter_from_values(
        config,
        modal_proxy_auth_enabled(),
        cleaned_env("SIE_OTEL_PROXY_AUTH_ORIGIN").as_deref(),
        cleaned_env("SIE_MODAL_PROXY_TOKEN_ID").as_deref(),
        cleaned_env("SIE_MODAL_PROXY_TOKEN_SECRET").as_deref(),
    )
}

/// Build the metric exporter, reading the Modal proxy posture from the
/// environment.
pub fn build_metric_exporter(
    config: &SignalExportConfig,
) -> Result<opentelemetry_otlp::MetricExporter, String> {
    build_metric_exporter_from_values(
        config,
        modal_proxy_auth_enabled(),
        cleaned_env("SIE_OTEL_PROXY_AUTH_ORIGIN").as_deref(),
        cleaned_env("SIE_MODAL_PROXY_TOKEN_ID").as_deref(),
        cleaned_env("SIE_MODAL_PROXY_TOKEN_SECRET").as_deref(),
    )
}

pub fn build_span_exporter_from_values(
    config: &SignalExportConfig,
    proxy_auth_enabled: bool,
    allowed_origin: Option<&str>,
    token_id: Option<&str>,
    token_secret: Option<&str>,
) -> Result<opentelemetry_otlp::SpanExporter, String> {
    validate_modal_proxy_transport(config.protocol, proxy_auth_enabled)?;
    match config.protocol {
        OtlpProtocol::Grpc => opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&config.endpoint)
            .build()
            .map_err(|error| format!("build OTLP/gRPC span exporter: {error}")),
        OtlpProtocol::Http => {
            let mut builder = opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_endpoint(&config.endpoint);
            let headers = modal_proxy_headers_from_values(
                &config.endpoint,
                "/v1/traces",
                proxy_auth_enabled,
                allowed_origin,
                token_id,
                token_secret,
            )?;
            if proxy_auth_enabled {
                builder = builder
                    .with_http_client(authenticated_http_client()?)
                    .with_headers(headers);
            }
            builder
                .build()
                .map_err(|error| format!("build OTLP/HTTP span exporter: {error}"))
        }
    }
}

pub fn build_metric_exporter_from_values(
    config: &SignalExportConfig,
    proxy_auth_enabled: bool,
    allowed_origin: Option<&str>,
    token_id: Option<&str>,
    token_secret: Option<&str>,
) -> Result<opentelemetry_otlp::MetricExporter, String> {
    validate_modal_proxy_transport(config.protocol, proxy_auth_enabled)?;
    match config.protocol {
        OtlpProtocol::Grpc => opentelemetry_otlp::MetricExporter::builder()
            .with_temporality(Temporality::LowMemory)
            .with_tonic()
            .with_endpoint(&config.endpoint)
            .build()
            .map_err(|error| format!("build OTLP/gRPC metric exporter: {error}")),
        OtlpProtocol::Http => {
            let mut builder = opentelemetry_otlp::MetricExporter::builder()
                .with_temporality(Temporality::LowMemory)
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_endpoint(&config.endpoint);
            let headers = modal_proxy_headers_from_values(
                &config.endpoint,
                "/v1/metrics",
                proxy_auth_enabled,
                allowed_origin,
                token_id,
                token_secret,
            )?;
            if proxy_auth_enabled {
                builder = builder
                    .with_http_client(authenticated_http_client()?)
                    .with_headers(headers);
            }
            builder
                .build()
                .map_err(|error| format!("build OTLP/HTTP metric exporter: {error}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_sdk::metrics::exporter::PushMetricExporter;

    #[test]
    fn both_http_protobuf_exporters_build_with_and_without_modal_auth() {
        let trace = SignalExportConfig {
            endpoint: "https://collector.example/v1/traces".to_string(),
            protocol: OtlpProtocol::Http,
        };
        build_span_exporter_from_values(&trace, false, None, None, None)
            .expect("OTLP/HTTP span exporter");

        let metrics = SignalExportConfig {
            endpoint: "https://workspace--collector.modal.run/v1/metrics".to_string(),
            protocol: OtlpProtocol::Http,
        };
        build_metric_exporter_from_values(
            &metrics,
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            Some("secret"),
        )
        .expect("authenticated OTLP/HTTP metric exporter");
    }

    #[tokio::test]
    async fn metric_exporters_pin_low_memory_temporality_for_both_transports() {
        for config in [
            SignalExportConfig {
                endpoint: "http://127.0.0.1:4317".to_string(),
                protocol: OtlpProtocol::Grpc,
            },
            SignalExportConfig {
                endpoint: "https://collector.example/v1/metrics".to_string(),
                protocol: OtlpProtocol::Http,
            },
        ] {
            let exporter = build_metric_exporter_from_values(&config, false, None, None, None)
                .expect("build metric exporter");
            assert_eq!(
                PushMetricExporter::temporality(&exporter),
                Temporality::LowMemory
            );
        }
    }
}
