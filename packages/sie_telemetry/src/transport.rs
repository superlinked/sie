//! OTLP transport/endpoint/protocol resolution shared by every SIE Rust
//! producer. Producers record one semantic OTel stream; this module only
//! chooses how that stream reaches the collector: gRPC for the in-cluster
//! default, or OTLP/HTTP protobuf for the managed Modal edge.

use crate::env::{cleaned_env, sie_metrics_enabled, sie_tracing_enabled};

/// Selected OTLP transport for an exporter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OtlpProtocol {
    /// gRPC/tonic — the in-cluster/Helm collector on `:4317` (the default).
    Grpc,
    /// HTTP `http/protobuf` — the managed Modal collector on `:4318`.
    Http,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalExportConfig {
    pub endpoint: String,
    pub protocol: OtlpProtocol,
}

/// Trace-enable flag plus the resolved metric endpoint (the gateway/sidecar
/// init shape: metrics resolve to a plain endpoint string; the protocol is
/// re-read at exporter build time).
#[derive(Debug, PartialEq, Eq)]
pub struct SignalEndpoints {
    pub tracing_enabled: bool,
    pub metrics_enabled: bool,
    pub metrics: Option<String>,
}

/// Resolve the trace/metric signal gates + metric endpoint from the standard
/// `SIE_*_ENABLED` / `OTEL_EXPORTER_OTLP_*` variables.
pub fn configured_signal_endpoints() -> SignalEndpoints {
    let metrics_endpoint = cleaned_env("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT");
    let generic_endpoint = cleaned_env("OTEL_EXPORTER_OTLP_ENDPOINT");
    signal_endpoints_from_values(
        sie_tracing_enabled(),
        sie_metrics_enabled(),
        metrics_endpoint.as_deref(),
        generic_endpoint.as_deref(),
        otlp_metrics_protocol().unwrap_or(OtlpProtocol::Grpc),
    )
}

pub fn signal_endpoints_from_values(
    tracing_enabled: bool,
    metrics_enabled: bool,
    metrics_endpoint: Option<&str>,
    generic_endpoint: Option<&str>,
    metrics_protocol: OtlpProtocol,
) -> SignalEndpoints {
    let metrics = if !metrics_enabled {
        None
    } else if let Some(explicit) = metrics_endpoint {
        Some(explicit.to_string())
    } else {
        generic_endpoint
            .map(|base| derive_metrics_endpoint(base, metrics_protocol, None, Some(base)))
    };
    SignalEndpoints {
        tracing_enabled,
        metrics_enabled,
        metrics,
    }
}

/// Resolve the trace exporter config from the standard OTEL variables,
/// gated on the caller-provided enable flag.
pub fn trace_export_config(enabled: bool) -> Result<Option<SignalExportConfig>, String> {
    trace_export_config_from_values(
        enabled,
        cleaned_env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_ENDPOINT").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_PROTOCOL").as_deref(),
    )
}

pub fn trace_export_config_from_values(
    enabled: bool,
    traces_endpoint: Option<&str>,
    generic_endpoint: Option<&str>,
    traces_protocol: Option<&str>,
    generic_protocol: Option<&str>,
) -> Result<Option<SignalExportConfig>, String> {
    if !enabled {
        return Ok(None);
    }
    let protocol = select_signal_protocol(traces_protocol, generic_protocol, "traces")?;
    let endpoint = if let Some(explicit) = traces_endpoint {
        explicit.to_string()
    } else {
        let Some(base) = generic_endpoint else {
            return Ok(None);
        };
        signal_endpoint(base, protocol, "/v1/traces")
    };
    Ok(Some(SignalExportConfig { endpoint, protocol }))
}

/// Resolve the metric exporter config from the standard OTEL variables,
/// gated on the caller-provided enable flag (the `sie_server_rust` shape;
/// gateway/sidecar resolve metrics via [`configured_signal_endpoints`]).
pub fn metric_export_config(enabled: bool) -> Result<Option<SignalExportConfig>, String> {
    metric_export_config_from_values(
        enabled,
        cleaned_env("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_ENDPOINT").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_PROTOCOL").as_deref(),
    )
}

pub fn metric_export_config_from_values(
    enabled: bool,
    metric_endpoint: Option<&str>,
    generic_endpoint: Option<&str>,
    metric_protocol: Option<&str>,
    generic_protocol: Option<&str>,
) -> Result<Option<SignalExportConfig>, String> {
    if !enabled {
        return Ok(None);
    }
    let protocol = select_signal_protocol(metric_protocol, generic_protocol, "metrics")?;
    let endpoint = if let Some(explicit) = metric_endpoint {
        explicit.to_string()
    } else {
        let Some(generic_endpoint) = generic_endpoint else {
            return Ok(None);
        };
        signal_endpoint(generic_endpoint, protocol, "/v1/metrics")
    };
    Ok(Some(SignalExportConfig { endpoint, protocol }))
}

/// Signal-specific setting wins over the generic setting, then the gRPC
/// default. A signal never inherits another signal's specific setting.
pub fn select_signal_protocol(
    signal_protocol: Option<&str>,
    generic_protocol: Option<&str>,
    signal_name: &str,
) -> Result<OtlpProtocol, String> {
    protocol_from_raw(signal_protocol.or(generic_protocol), signal_name)
}

/// Only exact `grpc` and `http/protobuf` are accepted (case-insensitive,
/// trimmed); absence preserves the historical gRPC path; anything else —
/// including an empty string — fails closed instead of silently changing
/// transport (the gateway/sidecar strictness; call sites read env vars via
/// `cleaned_env`, so an empty value never reaches this in production).
pub fn protocol_from_raw(raw: Option<&str>, signal_name: &str) -> Result<OtlpProtocol, String> {
    match raw.map(str::trim) {
        None => Ok(OtlpProtocol::Grpc),
        Some(value) if value.eq_ignore_ascii_case("grpc") => Ok(OtlpProtocol::Grpc),
        Some(value) if value.eq_ignore_ascii_case("http/protobuf") => Ok(OtlpProtocol::Http),
        Some(value) => Err(format!(
            "unsupported OTLP {signal_name} protocol {value:?}; expected grpc or http/protobuf"
        )),
    }
}

/// Metrics-specific transport override, then generic, then the gRPC default.
/// Metrics never inherit a trace-specific setting.
pub fn otlp_metrics_protocol() -> Result<OtlpProtocol, String> {
    select_signal_protocol(
        cleaned_env("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_PROTOCOL").as_deref(),
        "metrics",
    )
}

/// Append the per-signal path to a generic base endpoint for the HTTP
/// transport; gRPC uses the base as-is.
pub fn signal_endpoint(base: &str, protocol: OtlpProtocol, signal_path: &str) -> String {
    match protocol {
        OtlpProtocol::Grpc => base.to_string(),
        OtlpProtocol::Http if base.ends_with(signal_path) => base.to_string(),
        OtlpProtocol::Http => format!("{}{signal_path}", base.trim_end_matches('/')),
    }
}

/// Derive the metric endpoint from an endpoint seed (gateway/sidecar
/// metrics-resolution shape).
pub fn derive_metrics_endpoint(
    endpoint_seed: &str,
    protocol: OtlpProtocol,
    metrics_override: Option<&str>,
    base_override: Option<&str>,
) -> String {
    if let Some(explicit) = metrics_override {
        return explicit.to_string();
    }
    match protocol {
        OtlpProtocol::Grpc => endpoint_seed.to_string(),
        OtlpProtocol::Http => {
            if endpoint_seed.ends_with("/v1/metrics") {
                endpoint_seed.to_string()
            } else if let Some(base) = base_override {
                format!("{}/v1/metrics", base.trim_end_matches('/'))
            } else {
                format!("{}/v1/metrics", endpoint_seed.trim_end_matches('/'))
            }
        }
    }
}

/// Return only the scheme/host/explicit-port origin for diagnostics.
///
/// Operator-provided OTLP URLs may contain credentials in userinfo, path,
/// query, or fragment components. Those fields must never reach process logs.
pub fn endpoint_origin_for_log(endpoint: &str) -> String {
    let Ok(parsed) = reqwest::Url::parse(endpoint) else {
        return "<redacted>".to_string();
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        return "<redacted>".to_string();
    }
    let Some(host) = parsed.host_str() else {
        return "<redacted>".to_string();
    };
    let host = if host.starts_with('[') && host.ends_with(']') {
        host.to_string()
    } else if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    match parsed.port() {
        Some(port) => format!("{}://{host}:{port}", parsed.scheme()),
        None => format!("{}://{host}", parsed.scheme()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_precedence_is_signal_specific_then_generic_then_grpc() {
        assert_eq!(
            select_signal_protocol(Some("http/protobuf"), Some("grpc"), "traces")
                .expect("supported trace protocol"),
            OtlpProtocol::Http
        );
        assert_eq!(
            select_signal_protocol(None, Some("http/protobuf"), "metrics")
                .expect("supported generic protocol"),
            OtlpProtocol::Http
        );
        assert_eq!(
            select_signal_protocol(Some("grpc"), Some("http/protobuf"), "metrics")
                .expect("specific metric protocol wins"),
            OtlpProtocol::Grpc
        );
        assert_eq!(
            select_signal_protocol(None, None, "metrics").expect("absent protocol defaults"),
            OtlpProtocol::Grpc
        );
        assert_eq!(
            protocol_from_raw(Some("HTTP/PROTOBUF"), "traces")
                .expect("protocol matching is case insensitive"),
            OtlpProtocol::Http
        );
    }

    #[test]
    fn unsupported_protocols_fail_closed_instead_of_changing_transport() {
        for unsupported in ["", "http/json", "thrift", "json", "http"] {
            let error = protocol_from_raw(Some(unsupported), "metrics")
                .expect_err("unsupported protocols must not select an exporter");
            assert!(error.contains(unsupported));
            assert!(error.contains("metrics"));
            assert!(error.contains("grpc or http/protobuf"));
        }
    }

    #[test]
    fn signal_specific_endpoints_win_and_generic_http_gets_signal_paths() {
        assert_eq!(
            trace_export_config_from_values(
                true,
                Some("https://trace.example/custom"),
                Some("https://generic.example"),
                Some("http/protobuf"),
                None,
            ),
            Ok(Some(SignalExportConfig {
                endpoint: "https://trace.example/custom".to_string(),
                protocol: OtlpProtocol::Http,
            }))
        );
        assert_eq!(
            trace_export_config_from_values(
                true,
                None,
                Some("https://collector.example/"),
                None,
                Some("http/protobuf"),
            )
            .expect("valid trace protocol")
            .expect("generic trace endpoint")
            .endpoint,
            "https://collector.example/v1/traces"
        );
        assert_eq!(
            metric_export_config_from_values(
                true,
                None,
                Some("https://collector.example/"),
                None,
                Some("http/protobuf"),
            )
            .expect("valid metric protocol")
            .expect("generic metric endpoint")
            .endpoint,
            "https://collector.example/v1/metrics"
        );
    }

    #[test]
    fn disabled_or_endpoint_free_signals_do_not_build_export_configs() {
        assert_eq!(
            trace_export_config_from_values(false, Some("http://collector:4317"), None, None, None),
            Ok(None)
        );
        assert_eq!(
            metric_export_config_from_values(true, None, None, None, None),
            Ok(None)
        );
    }

    #[test]
    fn endpoint_log_origin_removes_every_credential_bearing_url_component() {
        assert_eq!(
            endpoint_origin_for_log(
                "https://telemetry-user:telemetry-secret@collector.example:8443/v1/traces?api_key=secret#fragment"
            ),
            "https://collector.example:8443"
        );
        assert_eq!(
            endpoint_origin_for_log("http://[2001:db8::1]:4317/v1/metrics?token=secret"),
            "http://[2001:db8::1]:4317"
        );
        assert_eq!(endpoint_origin_for_log("collector:4317"), "<redacted>");
        assert_eq!(
            endpoint_origin_for_log("ftp://user:secret@collector.example/path"),
            "<redacted>"
        );
    }

    #[test]
    fn derive_metrics_endpoint_honors_override_and_appends_http_path() {
        assert_eq!(
            derive_metrics_endpoint(
                "https://a.example",
                OtlpProtocol::Http,
                Some("https://explicit.example/v1/metrics"),
                None
            ),
            "https://explicit.example/v1/metrics"
        );
        assert_eq!(
            derive_metrics_endpoint("http://collector:4317", OtlpProtocol::Grpc, None, None),
            "http://collector:4317"
        );
        assert_eq!(
            derive_metrics_endpoint("https://a.example/", OtlpProtocol::Http, None, None),
            "https://a.example/v1/metrics"
        );
        assert_eq!(
            derive_metrics_endpoint(
                "https://seed.example/v1/traces",
                OtlpProtocol::Http,
                None,
                Some("https://base.example/")
            ),
            "https://base.example/v1/metrics"
        );
    }
}
