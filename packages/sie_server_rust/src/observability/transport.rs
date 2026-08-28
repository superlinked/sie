//! OTLP transport selection for native-worker traces and metrics.
//!
//! The transport/exporter plumbing lives in the shared `sie-telemetry`
//! crate (#2339); this module keeps the worker-local pieces — the rustls
//! crypto-provider install and the `SIE_TRACING_ENABLED`-gated wrappers —
//! and re-exports the shared names its siblings consume.

pub(crate) use sie_telemetry::transport::{
    endpoint_origin_for_log, metric_export_config, trace_export_config, SignalExportConfig,
};

pub(crate) fn install_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

pub(crate) fn build_span_exporter(
    config: &SignalExportConfig,
) -> Result<opentelemetry_otlp::SpanExporter, String> {
    // Historical worker posture: (re-)install the ring provider before any
    // exporter build; install_default is a no-op once a provider is set.
    install_crypto_provider();
    sie_telemetry::exporters::build_span_exporter(config)
}

pub(crate) fn build_metric_exporter(
    config: &SignalExportConfig,
) -> Result<opentelemetry_otlp::MetricExporter, String> {
    install_crypto_provider();
    sie_telemetry::exporters::build_metric_exporter(config)
}
