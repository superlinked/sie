//! OpenTelemetry tracer- and meter-provider setup for the gateway.
//!
//! Trace, metric, and privacy-safe log OTLP exporters are independently gated.
//! Each signal accepts only its signal-specific endpoint or the generic OTLP
//! endpoint; no signal inherits configuration from a sibling signal. Logs keep
//! trace-correlated sampling even when span export is off by
//! using a local, no-export tracer that honors the configured OTel sampler.
//! The W3C [`TraceContextPropagator`] is installed globally regardless
//! of that exporter gate so that inbound `traceparent` headers still
//! propagate through to the worker via the JetStream / queue work
//! envelope. Without a span exporter, spans are not exported; logs-only mode
//! may still create local no-export request contexts so sampled log records
//! remain trace-correlated.
//!
//! Transport (#1740 reconciliation): the exporter transport is chosen by
//! `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` (falling back to
//! `OTEL_EXPORTER_OTLP_PROTOCOL`) — `grpc` (the default) preserves the
//! in-cluster/Helm OTLP/gRPC `:4317` path, while exact `http/protobuf` selects
//! the OTLP/HTTP exporter the MANAGED Modal collector publishes
//! (only OTLP/HTTP `:4318` is reachable through the Modal edge). Under the
//! opt-in Modal proxy-auth posture the `Modal-Key` / `Modal-Secret` pair rides
//! the HTTP exporter only for the exact provisioner-recorded collector origin,
//! with redirects disabled, so the push clears the edge without making the
//! credential endpoint-controlled —
//! exactly as the lane sender does (`worker_runtime._build_lane_span_exporter`).
//!
//! When metrics are enabled the gateway installs an OTel `MeterProvider` that
//! pushes the canonical instruments to the collector's `/v1/metrics`.
//! Prometheus exposition is owned by the collector, never by this producer.
//! Managed deployments may also install a dedicated OTel
//! Logs provider for one compile-time allowlisted, sampled request-completion
//! event.  It is deliberately not a bridge from `tracing`/stdout: arbitrary
//! application messages and exception chains are outside the privacy contract.
//! Setup is fail-open: a bad endpoint or exporter build can never crash the
//! gateway — it degrades to no export.
//!
//! [`TraceContextPropagator`]: opentelemetry_sdk::propagation::TraceContextPropagator

use std::env;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

use opentelemetry::global;
use opentelemetry::logs::{AnyValue, LogRecord as _, Logger as _, LoggerProvider as _, Severity};
use opentelemetry::trace::{SpanContext, TracerProvider as _};
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::logs::{SdkLogger, SdkLoggerProvider};
#[cfg(feature = "telemetry-benchmarks")]
use opentelemetry_sdk::metrics::ManualReader;
use opentelemetry_sdk::metrics::{Instrument, PeriodicReader, SdkMeterProvider, Stream};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{SdkTracerProvider, Tracer};
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use crate::observability::metrics::{
    RequestCompletionObservation, ACTIVE_LEASE_GPUS_METRIC_NAME,
    KEDA_SCALE_UP_REJECTION_REASON_CARDINALITY, LANE_QUEUE_DEPTH_METRIC_NAME,
    LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME, PENDING_DEMAND_METRIC_NAME,
    POOL_WARM_FLOOR_METRIC_NAME, REJECTED_REQUESTS_METRIC_NAME,
};
use crate::state::demand_tracker::MAX_CONFIGURED_PHYSICAL_LANES;

// The OTLP transport/proxy/resource plumbing is shared via `sie-telemetry`
// (#2339); the imports keep every historical call-site and test name
// resolvable inside this module.
use sie_telemetry::env::{cleaned_env, sie_tracing_enabled, tracing_flag_set};
use sie_telemetry::exporters::build_span_exporter;
#[cfg(test)]
use sie_telemetry::proxy::modal_proxy_headers_from_values;
use sie_telemetry::proxy::{
    authenticated_http_client, modal_proxy_auth_enabled,
    modal_proxy_headers as modal_proxy_headers_for_endpoint, validate_modal_proxy_transport,
};
#[cfg(test)]
use sie_telemetry::resource::compose_service_instance_id;
pub(crate) use sie_telemetry::resource::process_start_uuid;
use sie_telemetry::resource::{resource_from_values, service_instance_id};
use sie_telemetry::transport::{
    configured_signal_endpoints, endpoint_origin_for_log, otlp_metrics_protocol,
    select_signal_protocol, trace_export_config, OtlpProtocol, SignalExportConfig,
};

/// Legacy single-arg test seam for the shared protocol parser.
#[cfg(test)]
fn protocol_from_raw(raw: Option<&str>) -> Result<OtlpProtocol, String> {
    sie_telemetry::transport::protocol_from_raw(raw, "traces")
}
#[cfg(any(test, feature = "telemetry-benchmarks"))]
use opentelemetry_sdk::metrics::Temporality;
#[cfg(test)]
use sie_telemetry::transport::{
    derive_metrics_endpoint, signal_endpoints_from_values, trace_export_config_from_values,
};
#[cfg(test)]
use std::collections::HashMap;

/// Service name used when `OTEL_SERVICE_NAME` is not set.
const DEFAULT_SERVICE_NAME: &str = "sie-gateway";
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();
static LOCAL_LOG_CONTEXT_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();
static METER_PROVIDER: OnceLock<SdkMeterProvider> = OnceLock::new();
static LOGGER_PROVIDER: OnceLock<SdkLoggerProvider> = OnceLock::new();
static REQUEST_LOGGER: OnceLock<SdkLogger> = OnceLock::new();

/// Bounded flush deadline (ms) so process exit can't stall on an unreachable collector.
const TRACING_SHUTDOWN_TIMEOUT_MS: u64 = 3_000;

/// Default and maximum periodic metric export interval. KEDA consumes these
/// observations after a second collector/Prometheus scrape hop, so every
/// topology must keep the producer interval at or below five seconds.
const DEFAULT_METRICS_EXPORT_INTERVAL_MS: u64 = 5_000;

/// Exact upper bound for the scale-worthy rejection stream: four declared
/// reasons for each member of the finite physical-lane catalog.
pub(crate) const KEDA_REJECTED_REQUESTS_CARDINALITY_LIMIT: usize =
    MAX_CONFIGURED_PHYSICAL_LANES * KEDA_SCALE_UP_REJECTION_REASON_CARDINALITY;

/// Whether the canonical OTLP metrics provider was installed successfully.
/// Lifecycle code may use this to avoid starting telemetry-only reconcilers;
/// metric facades never use it to select a second instrumentation backend.
pub fn metrics_exporter_enabled() -> bool {
    METER_PROVIDER.get().is_some()
}

/// Install a synchronous in-memory provider for downstream release
/// microbenchmarks. The feature is enabled only for test targets: production
/// builds retain the normal OTLP-only initialization path.
#[cfg(feature = "telemetry-benchmarks")]
#[doc(hidden)]
// Consumed only through the lib target (cloud-gateway benchmarks). The bin
// target compiles the same module tree, where workspace feature unification
// can switch the feature on with no in-crate caller — dead code there only.
#[allow(dead_code)]
pub fn install_metrics_benchmark_provider() {
    assert!(
        METER_PROVIDER.get().is_none(),
        "benchmark meter provider must be installed before telemetry is enabled"
    );
    let provider = SdkMeterProvider::builder()
        .with_reader(
            ManualReader::builder()
                .with_temporality(Temporality::LowMemory)
                .build(),
        )
        .with_view(keda_metric_cardinality_view)
        .build();
    global::set_meter_provider(provider.clone());
    assert!(
        METER_PROVIDER.set(provider).is_ok(),
        "benchmark meter provider must be installed exactly once"
    );
}

/// The only managed log event emitted by the first privacy-safe producer.
/// Both event name and body are fixed; no generic message reaches OTLP.
pub const REQUEST_COMPLETION_LOG_EVENT: &str = "inference.request.completed";
const REQUEST_COMPLETION_LOG_SCHEMA: &str = "2";

/// True when the OTLP trace exporter/provider is installed.
pub fn exporter_enabled() -> bool {
    TRACER_PROVIDER.get().is_some()
}

/// Whether any request-level telemetry signal has a live provider.
///
/// Router composition calls this only after [`init_tracing`]. Checking the
/// installed providers rather than environment flags preserves fail-open
/// behavior: if every requested exporter fails to initialize, inference runs
/// without installing request middleware or doing hidden OTel work.
pub fn request_telemetry_enabled() -> bool {
    request_telemetry_enabled_from_state(
        exporter_enabled(),
        metrics_exporter_enabled(),
        REQUEST_LOGGER.get().is_some(),
    )
}

fn request_telemetry_enabled_from_state(traces: bool, metrics: bool, safe_logs: bool) -> bool {
    traces || metrics || safe_logs
}

/// Initialise OpenTelemetry + tracing-subscriber for the gateway.
///
/// Pipeline:
///   1. Install the global W3C [`TraceContextPropagator`] so the
///      `traceparent` / `tracestate` headers extract into a
///      `opentelemetry::Context`. **Always runs**, even without
///      an exporter — propagation is the load-bearing piece for
///      worker-side correlation.
///   2. Independently resolve trace, metric, and log endpoints. Tracing requires
///      `SIE_TRACING_ENABLED`; metrics require `SIE_METRICS_ENABLED`; safe logs
///      require `SIE_OTLP_LOGS_ENABLED`. Signals can roll out independently and
///      setup failures remain fail-open.
///   3. If tracing is enabled and configured, build a [`SdkTracerProvider`]
///      with an OTLP exporter (gRPC or HTTP per [`OtlpProtocol`]), attach a
///      [`tracing_opentelemetry::OpenTelemetryLayer`] so existing
///      `tracing::*` spans become OTel spans and set the provider as global.
///      When `SIE_OTLP_LOGS_ENABLED` is truthy, independently install a batched
///      [`SdkLoggerProvider`] for the one allowlisted request-completion record.
///      If spans are not exported, a local no-export tracer still provides the
///      parent-based sampled context required by that record.
pub fn init_tracing(level: &str, json: bool) {
    // Idempotency guard. The subscriber's ``.init()`` panics on a
    // second call ("a global default trace dispatcher has already
    // been set"). Tests that spin up the gateway in-process across
    // multiple cases would otherwise abort on the second case.
    use std::sync::atomic::{AtomicBool, Ordering};
    static INIT_GUARD: AtomicBool = AtomicBool::new(false);
    if INIT_GUARD.swap(true, Ordering::SeqCst) {
        tracing::debug!("init_tracing called more than once; skipping subsequent init");
        return;
    }

    // Step 1: always install the propagator. Even without an OTLP
    // exporter the gateway needs to *extract* inbound trace headers
    // and *inject* them into the work envelope so the worker side
    // (which runs the heavy adapter call) can continue the trace.
    global::set_text_map_propagator(TraceContextPropagator::new());

    let endpoints = configured_signal_endpoints();
    let (trace_config, trace_config_error) = match trace_export_config(sie_tracing_enabled()) {
        Ok(config) => (config, None),
        Err(error) => {
            eprintln!(
                "warn: failed to resolve OTLP trace exporter; continuing without trace export"
            );
            (None, Some(error))
        }
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level_str = match level.to_lowercase().as_str() {
            "debug" => "debug",
            "warn" | "warning" => "warn",
            "error" => "error",
            _ => "info",
        };
        EnvFilter::new(level_str)
    });

    // Try to build the OTel tracer. On any failure the gateway
    // continues with the fmt-only subscriber (and propagator-only
    // OTel state) — operators get logs and worker-side trace
    // correlation, just no spans on the gateway itself. FAIL-OPEN
    // (mirrors #1870's lane wrap): telemetry setup must never crash the edge.
    let exported_tracer = trace_config
        .as_ref()
        .and_then(|config| match init_tracer(config) {
            Ok(t) => Some(t),
            Err(_) => {
                eprintln!(
                    "warn: failed to init OTLP trace exporter; continuing without trace export"
                );
                None
            }
        });
    let tracing_initialized = exported_tracer.is_some();
    let logs_endpoint_source = safe_logs_enabled()
        .then(configured_logs_endpoint_source)
        .flatten();
    let logs_initialized = logs_endpoint_source.as_deref().is_some_and(init_logs);
    let local_log_context_tracer = if exported_tracer.is_none() && logs_initialized {
        Some(init_local_log_context_tracer())
    } else {
        None
    };

    // Metrics initialise before request instruments are first bound to the
    // global meter, and independently of trace setup or safe-log activation.
    let metrics_initialized = endpoints.metrics.as_deref().is_some_and(init_metrics);

    // Place the (boxed) OTel layer FIRST, then the fmt and filter
    // layers. `OpenTelemetryLayer<S, T>: Layer<S>` so when we box it
    // against the inner `Registry` we get a `Box<dyn Layer<Registry>
    // + Send + Sync>` that `Option<L>: Layer<S> where L: Layer<S>`
    // composes cleanly. Doing it the other way round (OTel last) hits
    // a wall because the boxed `dyn Layer<Registry>` does not satisfy
    // `Layer<Layered<...>>`.
    let otel_layer_boxed: Option<Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync>> =
        exported_tracer.or(local_log_context_tracer).map(|t| {
            let l: Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync> =
                Box::new(tracing_opentelemetry::layer().with_tracer(t));
            l
        });

    let base = tracing_subscriber::registry().with(otel_layer_boxed);
    if json {
        base.with(filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        base.with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    if let Some(config) = trace_config.as_ref() {
        if tracing_initialized {
            tracing::info!(
                endpoint = %endpoint_origin_for_log(&config.endpoint),
                protocol = ?config.protocol,
                "OpenTelemetry tracing initialized"
            );
        }
        if !tracing_initialized {
            tracing::warn!(
                endpoint = %endpoint_origin_for_log(&config.endpoint),
                "OpenTelemetry trace exporter disabled after setup failure"
            );
        }
    } else if trace_config_error.is_some() {
        tracing::warn!("OpenTelemetry trace exporter disabled after configuration failure");
    } else if endpoints.tracing_enabled {
        tracing::warn!("SIE_TRACING_ENABLED set but no trace OTLP endpoint; tracing disabled");
    }
    if safe_logs_enabled() && logs_endpoint_source.is_none() {
        tracing::warn!("SIE_OTLP_LOGS_ENABLED set but no log OTLP endpoint; logs disabled");
    } else if logs_endpoint_source.is_some() && !logs_initialized {
        tracing::warn!("OpenTelemetry safe-log exporter disabled after setup failure");
    } else if let Some(source) = logs_endpoint_source.as_deref() {
        if let Ok(protocol) = otlp_logs_protocol() {
            tracing::info!(
                endpoint = %endpoint_origin_for_log(&otlp_logs_endpoint(source, protocol)),
                event = REQUEST_COMPLETION_LOG_EVENT,
                "OpenTelemetry safe-log export initialized"
            );
        }
    }
    if let Some(ep) = endpoints.metrics.as_deref() {
        if metrics_initialized {
            tracing::info!(
                endpoint = %endpoint_origin_for_log(ep),
                metric = super::metrics::REQUESTS_METRIC_NAME,
                "OpenTelemetry metric export initialized"
            );
        }
    } else if endpoints.metrics_enabled {
        tracing::warn!("SIE_METRICS_ENABLED set but no metric OTLP endpoint; metrics disabled");
    }
    if !endpoints.tracing_enabled && !endpoints.metrics_enabled {
        tracing::debug!(
            "trace and metric exporters disabled; W3C propagator installed (no exporter)"
        );
    }
}

fn safe_logs_enabled() -> bool {
    let raw = env::var("SIE_OTLP_LOGS_ENABLED").ok();
    tracing_flag_set(raw.as_deref())
}

fn configured_logs_endpoint_source() -> Option<String> {
    logs_endpoint_source_from_values(
        cleaned_env("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_ENDPOINT").as_deref(),
    )
}

fn logs_endpoint_source_from_values(
    logs_endpoint: Option<&str>,
    generic_endpoint: Option<&str>,
) -> Option<String> {
    logs_endpoint.or(generic_endpoint).map(str::to_string)
}

/// OTLP transport for the exporters (#1740 transport reconciliation). Reads
/// `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` then `OTEL_EXPORTER_OTLP_PROTOCOL`;
/// only exact `grpc` and `http/protobuf` are accepted, while absent stays gRPC.
/// Mirrors the lane's `_otlp_traces_protocol`.
#[cfg(test)]
fn otlp_protocol() -> Result<OtlpProtocol, String> {
    select_signal_protocol(
        cleaned_env("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_PROTOCOL").as_deref(),
        "traces",
    )
}

/// Logs-specific transport override, then generic, then the gRPC default.
/// Logs never inherit trace configuration.
fn otlp_logs_protocol() -> Result<OtlpProtocol, String> {
    select_signal_protocol(
        cleaned_env("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_PROTOCOL").as_deref(),
        "logs",
    )
}

/// Build the OTel resource stamped on every span/metric: `service.name` plus
/// `deployment.environment` / `cloud.region` (OTel semantic conventions) so
/// staging/prod and per-region telemetry are distinguishable in Better Stack —
/// parity with the lane's `_lane_resource_attributes`. Deployments inject the
/// real values; local processes retain the complete contract with `unknown`.
fn otlp_resource(service_name: &str) -> Resource {
    let deployment_environment = cleaned_env("SIE_OTEL_DEPLOYMENT_ENVIRONMENT")
        .or_else(|| cleaned_env("SIE_DEPLOYMENT_ENV"))
        .unwrap_or_else(|| "unknown".to_string());
    let cloud_region = cleaned_env("SIE_OTEL_CLOUD_REGION")
        .or_else(|| cleaned_env("SIE_CLOUD_REGION"))
        .or_else(|| cleaned_env("AWS_REGION"))
        .or_else(|| cleaned_env("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|| "unknown".to_string());
    let instance_prefix =
        cleaned_env("SIE_TELEMETRY_INSTANCE_ID").or_else(|| cleaned_env("MODAL_TASK_ID"));
    let instance_id = service_instance_id(instance_prefix.as_deref());
    otlp_resource_from_values(
        service_name,
        &instance_id,
        &deployment_environment,
        &cloud_region,
    )
}

fn otlp_resource_from_values(
    service_name: &str,
    instance_id: &str,
    deployment_environment: &str,
    cloud_region: &str,
) -> Resource {
    resource_from_values(
        service_name,
        instance_id,
        deployment_environment,
        cloud_region,
        Some(env!("CARGO_PKG_VERSION")),
    )
}

fn init_tracer(config: &SignalExportConfig) -> Result<Tracer, String> {
    let service_name =
        env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());

    let exporter = build_span_exporter(config)?;

    let provider = SdkTracerProvider::builder()
        .with_resource(otlp_resource(&service_name))
        .with_batch_exporter(exporter)
        .build();
    let tracer = provider.tracer("sie-gateway");
    global::set_tracer_provider(provider.clone());
    let _ = TRACER_PROVIDER.set(provider);
    Ok(tracer)
}

/// Create sampled request contexts for the safe-log producer without adding a
/// span processor/exporter. The SDK's default config reads
/// `OTEL_TRACES_SAMPLER(_ARG)`, including parent-based ratios, so logs-only mode
/// has the same volume/correlation policy as tracing mode without exporting a
/// duplicate span stream.
fn init_local_log_context_tracer() -> Tracer {
    let service_name =
        env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());
    let provider = local_log_context_provider(&service_name);
    let tracer = provider.tracer("sie-gateway.safe-log-context");
    let _ = LOCAL_LOG_CONTEXT_PROVIDER.set(provider);
    tracer
}

fn local_log_context_provider(service_name: &str) -> SdkTracerProvider {
    SdkTracerProvider::builder()
        .with_resource(otlp_resource(service_name))
        .build()
}

/// Install the MeterProvider used by the canonical telemetry facade.
/// FAIL-OPEN: any exporter/build failure degrades to no metric export and never
/// propagates (telemetry must never crash inference).
fn init_metrics(endpoint: &str) -> bool {
    let protocol = match otlp_metrics_protocol() {
        Ok(protocol) => protocol,
        Err(_) => {
            eprintln!(
                "warn: failed to init OTLP metric exporter; continuing without metric export"
            );
            return false;
        }
    };
    let exporter = match build_metric_exporter(endpoint, protocol) {
        Ok(exporter) => exporter,
        Err(_) => {
            eprintln!(
                "warn: failed to init OTLP metric exporter; continuing without metric export"
            );
            return false;
        }
    };
    let reader = PeriodicReader::builder(exporter)
        .with_interval(metrics_export_interval(
            cleaned_env("OTEL_METRIC_EXPORT_INTERVAL").as_deref(),
        ))
        .build();
    let service_name =
        env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());
    let provider = SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(otlp_resource(&service_name))
        .with_view(keda_metric_cardinality_view)
        .build();
    global::set_meter_provider(provider.clone());
    let _ = METER_PROVIDER.set(provider);
    true
}

/// Override the SDK's default 2,000-series ceiling for every KEDA-filtered
/// stream. Lane snapshots have one point per catalog member; the rejection
/// counter has four scale-worthy reasons per member. These limits are exact,
/// finite, and prevent a valid high-index lane from collapsing into the OTel
/// overflow series (which PromQL's exact lane filters cannot see).
pub(crate) fn keda_metric_cardinality_view(instrument: &Instrument) -> Option<Stream> {
    let limit = match instrument.name() {
        PENDING_DEMAND_METRIC_NAME
        | LANE_QUEUE_DEPTH_METRIC_NAME
        | LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME
        | ACTIVE_LEASE_GPUS_METRIC_NAME
        | POOL_WARM_FLOOR_METRIC_NAME => MAX_CONFIGURED_PHYSICAL_LANES,
        REJECTED_REQUESTS_METRIC_NAME => KEDA_REJECTED_REQUESTS_CARDINALITY_LIMIT,
        _ => return None,
    };
    Some(
        Stream::builder()
            .with_cardinality_limit(limit)
            .build()
            .expect("constant KEDA cardinality limits must be valid"),
    )
}

fn metrics_export_interval(raw: Option<&str>) -> Duration {
    let millis = raw
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value >= 1_000)
        .map(|value| value.min(DEFAULT_METRICS_EXPORT_INTERVAL_MS))
        .unwrap_or(DEFAULT_METRICS_EXPORT_INTERVAL_MS);
    Duration::from_millis(millis)
}

fn build_metric_exporter(
    endpoint: &str,
    protocol: OtlpProtocol,
) -> Result<opentelemetry_otlp::MetricExporter, String> {
    sie_telemetry::exporters::build_metric_exporter(&SignalExportConfig {
        endpoint: endpoint.to_string(),
        protocol,
    })
}

/// Install the privacy-safe managed log producer.
///
/// This provider has no `tracing_subscriber`/`log` appender.  The only caller
/// is [`record_inference_completion_log`], whose body and attribute key set are
/// fixed in this module.  That closed surface is the producer-side redaction
/// boundary; stdout and arbitrary exception records remain Modal-local.
fn init_logs(endpoint_seed: &str) -> bool {
    let protocol = match otlp_logs_protocol() {
        Ok(protocol) => protocol,
        Err(_) => {
            eprintln!("warn: failed to init OTLP log exporter; continuing without managed logs");
            return false;
        }
    };
    let endpoint = otlp_logs_endpoint(endpoint_seed, protocol);
    let exporter = match build_log_exporter(&endpoint, protocol) {
        Ok(exporter) => exporter,
        Err(_) => {
            eprintln!("warn: failed to init OTLP log exporter; continuing without managed logs");
            return false;
        }
    };
    let service_name =
        env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());
    let provider = SdkLoggerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(otlp_resource(&service_name))
        .build();
    let logger = provider.logger("sie-gateway.request-completion");
    let _ = LOGGER_PROVIDER.set(provider);
    let _ = REQUEST_LOGGER.set(logger);
    true
}

fn otlp_logs_endpoint(endpoint_seed: &str, protocol: OtlpProtocol) -> String {
    derive_logs_endpoint(
        endpoint_seed,
        protocol,
        cleaned_env("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT").as_deref(),
        cleaned_env("OTEL_EXPORTER_OTLP_ENDPOINT").as_deref(),
    )
}

fn derive_logs_endpoint(
    endpoint_seed: &str,
    protocol: OtlpProtocol,
    logs_override: Option<&str>,
    base_override: Option<&str>,
) -> String {
    if let Some(explicit) = logs_override {
        return explicit.to_string();
    }
    match protocol {
        OtlpProtocol::Grpc => endpoint_seed.to_string(),
        OtlpProtocol::Http => {
            if endpoint_seed.ends_with("/v1/logs") {
                endpoint_seed.to_string()
            } else if let Some(base) = base_override {
                format!("{}/v1/logs", base.trim_end_matches('/'))
            } else {
                format!("{}/v1/logs", endpoint_seed.trim_end_matches('/'))
            }
        }
    }
}

fn build_log_exporter(
    endpoint: &str,
    protocol: OtlpProtocol,
) -> Result<opentelemetry_otlp::LogExporter, String> {
    let proxy_auth_enabled = modal_proxy_auth_enabled();
    validate_modal_proxy_transport(protocol, proxy_auth_enabled)?;
    match protocol {
        OtlpProtocol::Http => {
            let mut builder = opentelemetry_otlp::LogExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_endpoint(endpoint);
            let headers = modal_proxy_headers_for_endpoint(endpoint, "/v1/logs")?;
            if proxy_auth_enabled {
                builder = builder
                    .with_http_client(authenticated_http_client()?)
                    .with_headers(headers);
            }
            builder
                .build()
                .map_err(|e| format!("build OTLP/HTTP log exporter: {e}"))
        }
        OtlpProtocol::Grpc => opentelemetry_otlp::LogExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .map_err(|e| format!("build OTLP/gRPC log exporter: {e}")),
    }
}

/// Emit the one privacy-safe managed application log.
///
/// The record is deliberately omitted unless it can be joined to a valid,
/// sampled gateway span.  The metric above remains 100%; logs follow the same
/// parent-based sampling decision as the trace to avoid orphan records and an
/// accidental independent-volume policy. Every field below is bounded and
/// produced by gateway code. Customer identifiers, credential material,
/// request bodies, URLs, headers, and error text are not accepted by this API.
pub fn record_inference_completion_log(
    span_context: Option<&SpanContext>,
    observation: &RequestCompletionObservation<'_>,
) {
    let Some(span_context) = sampled_log_span_context(span_context) else {
        return;
    };
    let Some(logger) = REQUEST_LOGGER.get() else {
        return;
    };

    logger.emit(build_request_completion_log_record(
        logger,
        span_context,
        observation,
    ));
}

fn sampled_log_span_context(span_context: Option<&SpanContext>) -> Option<&SpanContext> {
    span_context.filter(|cx| cx.is_valid() && cx.is_sampled())
}

fn build_request_completion_log_record(
    logger: &SdkLogger,
    span_context: &SpanContext,
    observation: &RequestCompletionObservation<'_>,
) -> opentelemetry_sdk::logs::SdkLogRecord {
    let status = super::metrics::bounded_http_status(observation.status);
    let mut record = logger.create_log_record();
    record.set_timestamp(SystemTime::now());
    record.set_severity_number(Severity::Info);
    record.set_severity_text("INFO");
    record.set_body(AnyValue::from(REQUEST_COMPLETION_LOG_EVENT));
    record.set_trace_context(
        span_context.trace_id(),
        span_context.span_id(),
        Some(span_context.trace_flags()),
    );
    record.add_attribute("event.name", REQUEST_COMPLETION_LOG_EVENT);
    record.add_attribute("event.schema.version", REQUEST_COMPLETION_LOG_SCHEMA);
    record.add_attribute("operation", observation.operation.to_string());
    record.add_attribute("outcome", super::metrics::request_outcome(status));
    record.add_attribute("http.status_code", i64::from(status));
    record.add_attribute(
        "model",
        super::metrics::sanitize_model_label(observation.model),
    );
    record.add_attribute(
        "machine_profile",
        super::metrics::sanitize_label(observation.machine_profile),
    );
    record.add_attribute("duration_ms", observation.duration_s.max(0.0) * 1_000.0);
    record.add_attribute("admission_outcome", observation.admission_outcome.as_str());
    record
}

/// Graceful shutdown — flush any pending spans/metrics.
///
/// Called from `main.rs` on the way out so the OTLP exporters have a
/// chance to drain their batch before the process exits.
pub fn shutdown_tracing() {
    if let Some(provider) = TRACER_PROVIDER.get() {
        let _ = provider.shutdown_with_timeout(Duration::from_millis(TRACING_SHUTDOWN_TIMEOUT_MS));
    }
    if let Some(provider) = LOCAL_LOG_CONTEXT_PROVIDER.get() {
        let _ = provider.shutdown_with_timeout(Duration::from_millis(TRACING_SHUTDOWN_TIMEOUT_MS));
    }
    if let Some(provider) = METER_PROVIDER.get() {
        let _ = provider.shutdown();
    }
    if let Some(provider) = LOGGER_PROVIDER.get() {
        let _ = provider.shutdown_with_timeout(Duration::from_millis(TRACING_SHUTDOWN_TIMEOUT_MS));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry::trace::{
        Span as _, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState, Tracer as _,
    };
    use opentelemetry::{Context, Key};
    use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;
    use std::thread;

    // Serialize env-mutating tests to avoid races — env vars are process-global
    // (mirrors the ENV_LOCK pattern in config.rs).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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

    struct CapturedHttpRequest {
        method: String,
        path: String,
        headers: HashMap<String, String>,
        body: Vec<u8>,
    }

    fn capture_one_http_request() -> (String, thread::JoinHandle<CapturedHttpRequest>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind trace capture receiver");
        let base = format!("http://{}", listener.local_addr().expect("capture address"));
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept trace export");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("set capture timeout");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let count = stream.read(&mut buffer).expect("read trace export");
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
                let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if request.len() >= header_end + 4 + content_length {
                    break;
                }
            }
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .expect("complete HTTP request headers");
            let header_text = String::from_utf8_lossy(&request[..header_end]);
            let mut request_line = header_text
                .lines()
                .next()
                .expect("HTTP request line")
                .split_whitespace();
            let method = request_line
                .next()
                .expect("HTTP request method")
                .to_string();
            let path = request_line.next().expect("HTTP request path").to_string();
            let headers = header_text
                .lines()
                .skip(1)
                .filter_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    Some((name.to_ascii_lowercase(), value.trim().to_string()))
                })
                .collect();
            let body = request[header_end + 4..].to_vec();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-protobuf\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .expect("respond to trace export");
            CapturedHttpRequest {
                method,
                path,
                headers,
                body,
            }
        });
        (base, handle)
    }

    #[test]
    fn otlp_resource_contains_complete_gateway_identity() {
        let resource = otlp_resource_from_values("sie-gateway", "boot-123", "staging", "us-east-1");

        assert_eq!(
            resource.get(&Key::new("service.name")),
            Some("sie-gateway".into())
        );
        assert_eq!(
            resource.get(&Key::new("service.instance.id")),
            Some("boot-123".into())
        );
        assert_eq!(
            resource.get(&Key::new("deployment.environment")),
            Some("staging".into())
        );
        assert_eq!(
            resource.get(&Key::new("cloud.region")),
            Some("us-east-1".into())
        );
    }

    #[test]
    fn service_instance_id_appends_process_start_uuid_to_substrate_prefix() {
        let first_process = uuid::Uuid::new_v4().to_string();
        let restarted_process = uuid::Uuid::new_v4().to_string();

        assert_eq!(
            compose_service_instance_id(Some(" pod-uid/gateway/ "), &first_process),
            format!("pod-uid/gateway/{first_process}")
        );
        assert_ne!(
            compose_service_instance_id(Some("pod-uid/gateway"), &first_process),
            compose_service_instance_id(Some("pod-uid/gateway"), &restarted_process)
        );
        assert_eq!(
            compose_service_instance_id(None, &first_process),
            first_process
        );
    }

    #[test]
    fn service_instance_id_is_stable_within_the_process() {
        let first = service_instance_id(Some("pod-uid/gateway"));
        let second = service_instance_id(Some("pod-uid/gateway"));

        assert_eq!(first, second);
        let suffix = first
            .strip_prefix("pod-uid/gateway/")
            .expect("configured substrate prefix");
        assert!(uuid::Uuid::parse_str(suffix).is_ok());
    }

    #[test]
    fn logs_endpoint_is_resolved_independently_of_trace_export() {
        assert_eq!(
            logs_endpoint_source_from_values(
                Some("https://logs.example/v1/logs"),
                Some("https://generic.example"),
            ),
            Some("https://logs.example/v1/logs".to_string())
        );
        assert_eq!(
            logs_endpoint_source_from_values(None, Some("https://generic.example"),),
            Some("https://generic.example".to_string())
        );
        assert_eq!(
            logs_endpoint_source_from_values(None, None),
            None,
            "logs must never inherit a trace-specific endpoint"
        );
    }

    #[test]
    fn logs_only_context_honors_parent_based_sampler_without_exporter() {
        let _env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prior_sampler = env::var("OTEL_TRACES_SAMPLER").ok();
        let prior_arg = env::var("OTEL_TRACES_SAMPLER_ARG").ok();
        env::set_var("OTEL_TRACES_SAMPLER", "parentbased_always_off");
        env::remove_var("OTEL_TRACES_SAMPLER_ARG");

        let provider = local_log_context_provider("sie-gateway-test");
        let tracer = provider.tracer("logs-only-test");
        let root = tracer.start("root");
        assert!(!root.span_context().is_sampled());

        let sampled_parent = SpanContext::new(
            TraceId::from(1_u128),
            SpanId::from(1_u64),
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        );
        let parent = Context::new().with_remote_span_context(sampled_parent);
        let child = tracer.start_with_context("child", &parent);
        assert!(child.span_context().is_sampled());
        provider.shutdown().expect("shutdown local provider");

        match prior_sampler {
            Some(value) => env::set_var("OTEL_TRACES_SAMPLER", value),
            None => env::remove_var("OTEL_TRACES_SAMPLER"),
        }
        match prior_arg {
            Some(value) => env::set_var("OTEL_TRACES_SAMPLER_ARG", value),
            None => env::remove_var("OTEL_TRACES_SAMPLER_ARG"),
        }
    }

    #[test]
    fn tracing_flag_set_accepts_only_positive_truthy_values() {
        assert!(tracing_flag_set(Some("true")));
        assert!(tracing_flag_set(Some("1")));
        assert!(tracing_flag_set(Some("yes")));
        assert!(tracing_flag_set(Some("TRUE")));
        assert!(tracing_flag_set(Some(" true ")));

        assert!(!tracing_flag_set(Some("false")));
        assert!(!tracing_flag_set(Some("")));
        assert!(!tracing_flag_set(Some("   ")));
        assert!(!tracing_flag_set(None));
    }

    #[test]
    fn exporter_enabled_is_false_before_exporter_init() {
        assert!(!exporter_enabled());
    }

    #[test]
    fn request_telemetry_gate_requires_at_least_one_live_signal() {
        assert!(!request_telemetry_enabled_from_state(false, false, false));
        assert!(request_telemetry_enabled_from_state(true, false, false));
        assert!(request_telemetry_enabled_from_state(false, true, false));
        assert!(request_telemetry_enabled_from_state(false, false, true));
    }

    #[test]
    fn metric_export_interval_never_exceeds_keda_contract() {
        assert_eq!(metrics_export_interval(None), Duration::from_secs(5));
        assert_eq!(
            metrics_export_interval(Some("2000")),
            Duration::from_secs(2)
        );
        assert_eq!(
            metrics_export_interval(Some("30000")),
            Duration::from_secs(5)
        );
        assert_eq!(
            metrics_export_interval(Some("invalid")),
            Duration::from_secs(5)
        );
        assert_eq!(metrics_export_interval(Some("500")), Duration::from_secs(5));
    }

    #[test]
    fn protocol_selection_http_vs_grpc() {
        assert_eq!(
            protocol_from_raw(Some("http/protobuf")),
            Ok(OtlpProtocol::Http)
        );
        assert_eq!(
            protocol_from_raw(Some(" HTTP/PROTOBUF ")),
            Ok(OtlpProtocol::Http)
        );
        assert_eq!(protocol_from_raw(Some("grpc")), Ok(OtlpProtocol::Grpc));
        assert_eq!(protocol_from_raw(None), Ok(OtlpProtocol::Grpc));
        for unsupported in ["", "http", "http/json", "thrift"] {
            assert!(protocol_from_raw(Some(unsupported)).is_err());
        }
    }

    #[test]
    fn metrics_endpoint_derives_from_generic_endpoint() {
        assert_eq!(
            derive_metrics_endpoint(
                "https://collector.modal.run",
                OtlpProtocol::Http,
                None,
                None,
            ),
            "https://collector.modal.run/v1/metrics"
        );
        // HTTP with only a base endpoint: append the metrics path.
        assert_eq!(
            derive_metrics_endpoint(
                "https://collector.modal.run/anything",
                OtlpProtocol::Http,
                None,
                Some("https://collector.modal.run/"),
            ),
            "https://collector.modal.run/v1/metrics"
        );
        // gRPC uses the base endpoint for every signal.
        assert_eq!(
            derive_metrics_endpoint("http://collector:4317", OtlpProtocol::Grpc, None, None),
            "http://collector:4317"
        );
        // An explicit metrics endpoint always wins.
        assert_eq!(
            derive_metrics_endpoint(
                "https://collector.modal.run/v1/traces",
                OtlpProtocol::Http,
                Some("https://other/v1/metrics"),
                None,
            ),
            "https://other/v1/metrics"
        );
    }

    #[test]
    fn metrics_only_configuration_builds_metrics_without_a_tracer_endpoint() {
        let endpoints = signal_endpoints_from_values(
            false,
            true,
            Some("https://collector.example/v1/metrics"),
            None,
            OtlpProtocol::Http,
        );
        assert!(!endpoints.tracing_enabled);
        assert!(endpoints.metrics_enabled);
        assert_eq!(
            endpoints.metrics.as_deref(),
            Some("https://collector.example/v1/metrics")
        );
        build_metric_exporter(
            endpoints.metrics.as_deref().expect("metrics endpoint"),
            OtlpProtocol::Http,
        )
        .expect("metrics-only exporter must build while tracing is off");

        let trace_only = signal_endpoints_from_values(false, true, None, None, OtlpProtocol::Http);
        assert_eq!(
            trace_only.metrics, None,
            "metrics must never inherit a trace endpoint"
        );
    }

    #[tokio::test]
    async fn metric_exporters_pin_low_memory_temporality_for_both_transports() {
        let _env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prior_proxy_auth = env::var("SIE_MODAL_PROXY_AUTH").ok();
        env::remove_var("SIE_MODAL_PROXY_AUTH");
        for (endpoint, protocol) in [
            ("http://127.0.0.1:4317", OtlpProtocol::Grpc),
            ("https://collector.example/v1/metrics", OtlpProtocol::Http),
        ] {
            let exporter =
                build_metric_exporter(endpoint, protocol).expect("build metric exporter");
            assert_eq!(
                PushMetricExporter::temporality(&exporter),
                Temporality::LowMemory
            );
        }
        match prior_proxy_auth {
            Some(value) => env::set_var("SIE_MODAL_PROXY_AUTH", value),
            None => env::remove_var("SIE_MODAL_PROXY_AUTH"),
        }
    }

    #[test]
    fn tracing_and_metrics_gates_are_independent() {
        let traces_only = signal_endpoints_from_values(
            true,
            false,
            Some("http://collector:4317"),
            None,
            OtlpProtocol::Grpc,
        );
        assert!(traces_only.tracing_enabled);
        assert_eq!(traces_only.metrics, None);

        let disabled = signal_endpoints_from_values(
            false,
            false,
            Some("http://collector:4317"),
            Some("http://collector:4317"),
            OtlpProtocol::Grpc,
        );
        assert!(!disabled.tracing_enabled);
        assert_eq!(disabled.metrics, None);

        let separate_endpoints = signal_endpoints_from_values(
            true,
            true,
            None,
            Some("https://metrics.example/"),
            OtlpProtocol::Http,
        );
        assert_eq!(
            separate_endpoints.metrics.as_deref(),
            Some("https://metrics.example/v1/metrics"),
            "the generic metrics base must not inherit the trace-specific origin"
        );
    }

    #[test]
    fn trace_export_config_keeps_specific_endpoint_and_paths_generic_http_base() {
        assert_eq!(
            trace_export_config_from_values(
                true,
                Some("https://trace.example/custom"),
                Some("https://generic.example"),
                Some("http/protobuf"),
                Some("grpc"),
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
            ),
            Ok(Some(SignalExportConfig {
                endpoint: "https://collector.example/v1/traces".to_string(),
                protocol: OtlpProtocol::Http,
            }))
        );
        assert_eq!(
            trace_export_config_from_values(true, None, Some("http://collector:4317"), None, None,),
            Ok(Some(SignalExportConfig {
                endpoint: "http://collector:4317".to_string(),
                protocol: OtlpProtocol::Grpc,
            }))
        );
        assert!(trace_export_config_from_values(
            true,
            Some("https://trace.example/v1/traces"),
            None,
            Some("http/json"),
            Some("http/protobuf"),
        )
        .is_err());
    }

    #[test]
    fn generic_only_http_trace_config_posts_to_standard_trace_path() {
        let _env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prior_proxy_auth = env::var("SIE_MODAL_PROXY_AUTH").ok();
        env::remove_var("SIE_MODAL_PROXY_AUTH");
        let (base, capture) = capture_one_http_request();
        let config =
            trace_export_config_from_values(true, None, Some(&base), None, Some("http/protobuf"))
                .expect("valid HTTP protocol")
                .expect("generic trace endpoint");
        let exporter = build_span_exporter(&config).expect("build capture exporter");
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter)
            .build();
        provider.tracer("trace-path-capture").start("capture").end();
        provider.shutdown().expect("flush capture span");
        let captured = capture.join().expect("capture receiver");
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/v1/traces");
        match prior_proxy_auth {
            Some(value) => env::set_var("SIE_MODAL_PROXY_AUTH", value),
            None => env::remove_var("SIE_MODAL_PROXY_AUTH"),
        }
    }

    #[test]
    fn generic_only_http_metrics_posts_protobuf_to_standard_metric_path() {
        let _env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prior_proxy_auth = env::var("SIE_MODAL_PROXY_AUTH").ok();
        env::remove_var("SIE_MODAL_PROXY_AUTH");
        let (base, capture) = capture_one_http_request();
        let endpoint = derive_metrics_endpoint(&base, OtlpProtocol::Http, None, None);
        let exporter = build_metric_exporter(&endpoint, OtlpProtocol::Http)
            .expect("build metric capture exporter");
        let reader = PeriodicReader::builder(exporter).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        provider
            .meter("metric-path-capture")
            .u64_counter("sie.gateway.requests")
            .build()
            .add(1, &[]);
        provider.force_flush().expect("flush capture metric");

        let captured = capture.join().expect("capture receiver");
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/v1/metrics");
        assert_eq!(
            captured.headers.get("content-type").map(String::as_str),
            Some("application/x-protobuf")
        );
        assert!(!captured.body.is_empty());
        assert!(captured
            .body
            .windows(b"sie.gateway.requests".len())
            .any(|window| window == b"sie.gateway.requests"));
        let _ = provider.shutdown();
        match prior_proxy_auth {
            Some(value) => env::set_var("SIE_MODAL_PROXY_AUTH", value),
            None => env::remove_var("SIE_MODAL_PROXY_AUTH"),
        }
    }

    #[test]
    fn logs_endpoint_derives_from_generic_endpoint() {
        assert_eq!(
            derive_logs_endpoint(
                "https://collector.modal.run",
                OtlpProtocol::Http,
                None,
                None,
            ),
            "https://collector.modal.run/v1/logs"
        );
        assert_eq!(
            derive_logs_endpoint(
                "https://collector.modal.run/anything",
                OtlpProtocol::Http,
                None,
                Some("https://collector.modal.run/"),
            ),
            "https://collector.modal.run/v1/logs"
        );
        assert_eq!(
            derive_logs_endpoint("http://collector:4317", OtlpProtocol::Grpc, None, None),
            "http://collector:4317"
        );
        assert_eq!(
            derive_logs_endpoint(
                "https://collector.modal.run/v1/traces",
                OtlpProtocol::Http,
                Some("https://other/v1/logs"),
                None,
            ),
            "https://other/v1/logs"
        );
    }

    #[test]
    fn metrics_protocol_is_resolved_independently_of_traces() {
        // Hold the shared lock across every env mutation, protocol resolution,
        // and restoration below so this can't race other env-touching tests in
        // the same binary (poison is benign — we only guard ordering).
        let _env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // The metrics signal honors OTEL_EXPORTER_OTLP_METRICS_PROTOCOL rather
        // than inheriting the traces-specific setting: with traces pinned to
        // gRPC and metrics to HTTP (generic unset), each resolver picks its own.
        let prior_metrics = env::var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL").ok();
        let prior_traces = env::var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL").ok();
        let prior_generic = env::var("OTEL_EXPORTER_OTLP_PROTOCOL").ok();

        env::set_var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "http/protobuf");
        env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc");
        env::remove_var("OTEL_EXPORTER_OTLP_PROTOCOL");

        assert_eq!(otlp_metrics_protocol(), Ok(OtlpProtocol::Http));
        assert_eq!(otlp_protocol(), Ok(OtlpProtocol::Grpc));

        // Restore prior process env so parallel/subsequent tests are unaffected.
        match prior_metrics {
            Some(v) => env::set_var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", v),
            None => env::remove_var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL"),
        }
        match prior_traces {
            Some(v) => env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", v),
            None => env::remove_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"),
        }
        if let Some(v) = prior_generic {
            env::set_var("OTEL_EXPORTER_OTLP_PROTOCOL", v);
        }
    }

    #[test]
    fn metrics_protocol_never_falls_back_to_traces_transport() {
        let _env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let prior_metrics = env::var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL").ok();
        let prior_traces = env::var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL").ok();
        let prior_generic = env::var("OTEL_EXPORTER_OTLP_PROTOCOL").ok();

        // A trace-specific setting never influences metrics.
        env::remove_var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL");
        env::remove_var("OTEL_EXPORTER_OTLP_PROTOCOL");
        env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf");
        assert_eq!(otlp_metrics_protocol(), Ok(OtlpProtocol::Grpc));

        // In-cluster path: nothing set also defaults to gRPC (:4317 plaintext).
        env::remove_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL");
        assert_eq!(otlp_metrics_protocol(), Ok(OtlpProtocol::Grpc));

        // A metrics-specific override still wins over the traces transport.
        env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf");
        env::set_var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "grpc");
        assert_eq!(otlp_metrics_protocol(), Ok(OtlpProtocol::Grpc));

        match prior_metrics {
            Some(v) => env::set_var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", v),
            None => env::remove_var("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL"),
        }
        match prior_traces {
            Some(v) => env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", v),
            None => env::remove_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"),
        }
        match prior_generic {
            Some(v) => env::set_var("OTEL_EXPORTER_OTLP_PROTOCOL", v),
            None => env::remove_var("OTEL_EXPORTER_OTLP_PROTOCOL"),
        }
    }

    #[test]
    fn logs_protocol_never_falls_back_to_traces_transport() {
        let _env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let prior_logs = env::var("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL").ok();
        let prior_traces = env::var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL").ok();
        let prior_generic = env::var("OTEL_EXPORTER_OTLP_PROTOCOL").ok();

        env::remove_var("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL");
        env::remove_var("OTEL_EXPORTER_OTLP_PROTOCOL");
        env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf");
        assert_eq!(otlp_logs_protocol(), Ok(OtlpProtocol::Grpc));

        env::set_var("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", " HTTP/PROTOBUF ");
        assert_eq!(otlp_logs_protocol(), Ok(OtlpProtocol::Http));
        env::set_var("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/json");
        assert!(otlp_logs_protocol().is_err());

        match prior_logs {
            Some(value) => env::set_var("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", value),
            None => env::remove_var("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL"),
        }
        match prior_traces {
            Some(value) => env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", value),
            None => env::remove_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"),
        }
        match prior_generic {
            Some(value) => env::set_var("OTEL_EXPORTER_OTLP_PROTOCOL", value),
            None => env::remove_var("OTEL_EXPORTER_OTLP_PROTOCOL"),
        }
    }

    #[tokio::test]
    async fn http_span_exporter_builds_with_tls_and_batches_under_tokio() {
        // #1878 build-path guard. This CANNOT reproduce the deployed panic
        // (which only fires when the batch thread actually pushes to a live
        // HTTPS collector — only a redeploy proves the wire) but it locks in the
        // two compile/build-time preconditions the fix depends on:
        //   1. the OTLP/HTTP (reqwest) span exporter BUILDS against an HTTPS
        //      endpoint — i.e. a rustls TLS feature is compiled in; and
        //   2. the batch span processor CONSTRUCTS inside a Tokio runtime.
        // With the async `reqwest-client` the batch thread had no reactor and
        // panicked at export; the blocking client (this build) drives its own.
        let _env_guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let prior_traces = env::var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL").ok();
        let prior_generic = env::var("OTEL_EXPORTER_OTLP_PROTOCOL").ok();
        env::remove_var("OTEL_EXPORTER_OTLP_PROTOCOL");
        env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf");

        let exporter = build_span_exporter(&SignalExportConfig {
            endpoint: "https://collector.example/v1/traces".to_string(),
            protocol: OtlpProtocol::Http,
        })
        .expect("OTLP/HTTP span exporter must build against HTTPS (TLS feature enabled)");
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build();
        // No spans recorded, so shutdown drains an empty batch (no network) and
        // just confirms the dedicated batch thread was constructed and stops.
        let _ = provider.shutdown_with_timeout(Duration::from_millis(500));

        match prior_traces {
            Some(v) => env::set_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", v),
            None => env::remove_var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"),
        }
        if let Some(v) = prior_generic {
            env::set_var("OTEL_EXPORTER_OTLP_PROTOCOL", v);
        }
    }

    #[test]
    fn modal_proxy_headers_require_exact_trusted_signal_endpoint() {
        let trusted = modal_proxy_headers_from_values(
            "https://workspace--collector.modal.run/v1/traces",
            "/v1/traces",
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            Some("secret"),
        )
        .expect("trusted managed endpoint");
        assert_eq!(trusted.get("Modal-Key").map(String::as_str), Some("id"));
        assert_eq!(
            trusted.get("Modal-Secret").map(String::as_str),
            Some("secret")
        );
        for (id, secret) in [
            (Some("id"), None),
            (None, Some("secret")),
            (None, None),
            (Some(""), Some("secret")),
            (Some("id"), Some("   ")),
        ] {
            assert!(modal_proxy_headers_from_values(
                "https://workspace--collector.modal.run/v1/traces",
                "/v1/traces",
                true,
                Some("https://workspace--collector.modal.run"),
                id,
                secret,
            )
            .is_err());
        }

        for endpoint in [
            "https://attacker.example/v1/traces",
            "https://sibling--collector.modal.run/v1/traces",
            "https://workspace--collector.modal.run.evil.example/v1/traces",
            "https://user@workspace--collector.modal.run/v1/traces",
            "http://workspace--collector.modal.run/v1/traces",
            "https://workspace--collector.modal.run:8443/v1/traces",
            "https://workspace--collector.modal.run/wrong-path",
            "not a URL",
        ] {
            assert!(modal_proxy_headers_from_values(
                endpoint,
                "/v1/traces",
                true,
                Some("https://workspace--collector.modal.run"),
                Some("id"),
                Some("secret"),
            )
            .is_err());
        }
    }

    #[test]
    fn modal_proxy_headers_preserve_oss_and_guard_metrics_override() {
        let oss = modal_proxy_headers_from_values(
            "https://otel.example/v1/traces",
            "/v1/traces",
            false,
            None,
            Some("id"),
            Some("secret"),
        )
        .expect("managed auth is off");
        assert!(oss.is_empty());

        let managed_metrics = modal_proxy_headers_from_values(
            "https://workspace--collector.modal.run/v1/metrics",
            "/v1/metrics",
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            Some("secret"),
        )
        .expect("trusted managed metrics endpoint");
        assert_eq!(managed_metrics.len(), 2);
        assert_eq!(
            managed_metrics.get("Modal-Key").map(String::as_str),
            Some("id")
        );
        assert_eq!(
            managed_metrics.get("Modal-Secret").map(String::as_str),
            Some("secret")
        );

        let override_endpoint = derive_metrics_endpoint(
            "https://workspace--collector.modal.run/v1/traces",
            OtlpProtocol::Http,
            Some("https://attacker.example/v1/metrics"),
            None,
        );
        assert!(modal_proxy_headers_from_values(
            &override_endpoint,
            "/v1/metrics",
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            Some("secret"),
        )
        .is_err());

        let logs_override = derive_logs_endpoint(
            "https://workspace--collector.modal.run/v1/traces",
            OtlpProtocol::Http,
            Some("https://attacker.example/v1/logs"),
            None,
        );
        assert!(modal_proxy_headers_from_values(
            &logs_override,
            "/v1/logs",
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            Some("secret"),
        )
        .is_err());
    }

    #[tokio::test]
    async fn managed_proxy_auth_rejects_grpc_and_builds_bounded_no_redirect_client_in_runtime() {
        assert!(validate_modal_proxy_transport(OtlpProtocol::Grpc, true).is_err());
        assert!(validate_modal_proxy_transport(OtlpProtocol::Http, true).is_ok());
        authenticated_http_client().expect("no-redirect blocking client must build");
    }

    #[test]
    fn completion_log_is_fixed_bounded_and_explicitly_correlated() {
        let span_context = SpanContext::new(
            TraceId::from(0x4bf9_2f35_77b3_4da6_a3ce_929d_0e0e_4736),
            SpanId::from(0x00f0_67aa_0ba9_02b7),
            TraceFlags::SAMPLED,
            false,
            TraceState::default(),
        );
        let provider = SdkLoggerProvider::builder().build();
        let logger = provider.logger("test");
        let observation = RequestCompletionObservation {
            operation: "encode",
            status: 200,
            model: "BAAI/bge-m3",
            machine_profile: "l4-spot",
            duration_s: 0.125,
            admission_outcome: super::super::metrics::AdmissionOutcome::Admitted,
        };
        let record = build_request_completion_log_record(&logger, &span_context, &observation);

        assert_eq!(record.event_name(), None);
        assert_eq!(
            record.body(),
            Some(&AnyValue::from(REQUEST_COMPLETION_LOG_EVENT))
        );
        assert_eq!(record.severity_number(), Some(Severity::Info));
        let trace = record.trace_context().expect("explicit correlation");
        assert_eq!(trace.trace_id, span_context.trace_id());
        assert_eq!(trace.span_id, span_context.span_id());
        assert_eq!(trace.trace_flags, Some(TraceFlags::SAMPLED));

        let attributes: HashMap<_, _> = record
            .attributes_iter()
            .map(|(key, value)| (key.as_str(), value.clone()))
            .collect();
        assert_eq!(attributes.len(), 9);
        assert_eq!(
            attributes["event.name"],
            AnyValue::from("inference.request.completed")
        );
        assert_eq!(attributes["event.schema.version"], AnyValue::from("2"));
        assert_eq!(attributes["operation"], AnyValue::from("encode"));
        assert_eq!(attributes["outcome"], AnyValue::from("success"));
        assert_eq!(attributes["http.status_code"], AnyValue::from(200_i64));
        assert_eq!(attributes["model"], AnyValue::from("BAAI/bge-m3"));
        assert_eq!(attributes["machine_profile"], AnyValue::from("l4-spot"));
        assert_eq!(attributes["duration_ms"], AnyValue::from(125.0_f64));
        assert_eq!(attributes["admission_outcome"], AnyValue::from("admitted"));
        for forbidden in [
            "request_id",
            "url",
            "header",
            "error",
            "api_key",
            "email",
            "prompt",
            "document",
            "embedding",
        ] {
            assert!(!attributes.contains_key(forbidden));
        }
    }

    #[test]
    fn completion_log_collapses_invalid_status_to_contract_zero() {
        let span_context = SpanContext::new(
            TraceId::from(1_u128),
            SpanId::from(1_u64),
            TraceFlags::SAMPLED,
            false,
            TraceState::default(),
        );
        let provider = SdkLoggerProvider::builder().build();
        let logger = provider.logger("test");
        let observation = RequestCompletionObservation {
            operation: "encode",
            status: 700,
            model: "invalid model value with spaces",
            machine_profile: "invalid/profile",
            duration_s: 0.0,
            admission_outcome: super::super::metrics::AdmissionOutcome::Admitted,
        };
        let record = build_request_completion_log_record(&logger, &span_context, &observation);
        let attributes: HashMap<_, _> = record
            .attributes_iter()
            .map(|(key, value)| (key.as_str(), value.clone()))
            .collect();

        assert_eq!(attributes["outcome"], AnyValue::from("other"));
        assert_eq!(attributes["http.status_code"], AnyValue::from(0_i64));
        assert_eq!(attributes["model"], AnyValue::from("other"));
        assert_eq!(attributes["machine_profile"], AnyValue::from("other"));
    }

    #[test]
    fn completion_log_requires_a_valid_sampled_span() {
        let sampled = SpanContext::new(
            TraceId::from(1_u128),
            SpanId::from(1_u64),
            TraceFlags::SAMPLED,
            false,
            TraceState::default(),
        );
        let unsampled = SpanContext::new(
            TraceId::from(1_u128),
            SpanId::from(1_u64),
            TraceFlags::default(),
            false,
            TraceState::default(),
        );
        assert!(sampled_log_span_context(Some(&sampled)).is_some());
        assert!(sampled_log_span_context(Some(&unsampled)).is_none());
        assert!(sampled_log_span_context(Some(&SpanContext::NONE)).is_none());
        assert!(sampled_log_span_context(None).is_none());
    }
}
