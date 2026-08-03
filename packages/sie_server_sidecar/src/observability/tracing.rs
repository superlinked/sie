//! OpenTelemetry trace + bounded-metric provider setup for the sidecar.
//!
//! Mirrors `packages/sie_gateway/src/observability/tracing.rs`. Trace and metric
//! export are independently gated by `SIE_TRACING_ENABLED` and
//! `SIE_METRICS_ENABLED`; either signal may use its specific OTLP endpoint or
//! the generic endpoint. Transport follows the OTel env contract:
//! signal-specific protocol first, then the generic protocol, with gRPC as the
//! default. HTTP/protobuf carries the Modal proxy-auth headers only for the
//! provisioner-recorded collector origin, with redirects disabled.
//! The W3C [`TraceContextPropagator`] is installed globally regardless
//! of that exporter gate so that the inbound gateway
//! `traceparent` (carried on the work envelope) still propagates
//! through to the adapter worker. Without an exporter the sidecar
//! itself records no spans, but the IDs continue to flow — the
//! `into_run_batch_item_with_trace` fallback in the dispatcher copies
//! the gateway context onto the wire items unchanged.
//!
//! [`TraceContextPropagator`]: opentelemetry_sdk::propagation::TraceContextPropagator

use std::sync::OnceLock;
use std::time::Duration;

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::metrics::{Instrument, PeriodicReader, SdkMeterProvider, Stream};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{SdkTracerProvider, Tracer};
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Service name used when `OTEL_SERVICE_NAME` is not set.
const DEFAULT_SERVICE_NAME: &str = "sie-worker-sidecar";
const UNKNOWN_RESOURCE_VALUE: &str = "unknown";
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();
static METER_PROVIDER: OnceLock<SdkMeterProvider> = OnceLock::new();

/// Bounded flush deadline (ms) so process exit can't stall on an unreachable collector.
const TRACING_SHUTDOWN_TIMEOUT_MS: u64 = 3_000;

/// KEDA's local control path requires a fresh point at least every five seconds.
const METRICS_EXPORT_INTERVAL_S: u64 = 5;
const _: () = assert!(METRICS_EXPORT_INTERVAL_S <= 5);

/// True only after the canonical MeterProvider was installed successfully.
pub fn metrics_provider_enabled() -> bool {
    METER_PROVIDER.get().is_some()
}

// The OTLP transport/proxy/resource plumbing is shared via `sie-telemetry`
// (#2339); the imports below keep every historical call-site and test name
// resolvable inside this module.
use sie_telemetry::env::{cleaned_env, sie_tracing_enabled};
use sie_telemetry::exporters::build_span_exporter;
use sie_telemetry::resource::{instance_prefix_env, resource_from_values, service_instance_id};
use sie_telemetry::transport::{
    configured_signal_endpoints, endpoint_origin_for_log, otlp_metrics_protocol,
    trace_export_config, OtlpProtocol, SignalExportConfig,
};
// Test-only shared names: the unit tests below exercise the extracted
// implementations through this module's historical seams via `use super::*`.
#[cfg(test)]
use opentelemetry_sdk::metrics::Temporality;
#[cfg(test)]
use sie_telemetry::env::tracing_flag_set;
#[cfg(test)]
use sie_telemetry::proxy::{
    authenticated_http_client, modal_proxy_headers_from_values, validate_modal_proxy_transport,
};
#[cfg(test)]
use sie_telemetry::resource::compose_service_instance_id;
#[cfg(test)]
use sie_telemetry::transport::{
    derive_metrics_endpoint, signal_endpoints_from_values, trace_export_config_from_values,
};
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::env;

/// Initialise OpenTelemetry + tracing-subscriber for the sidecar.
///
/// Pipeline:
///   1. Install the global W3C [`TraceContextPropagator`] so the
///      inbound `traceparent` / `tracestate` (read off the work
///      envelope) extract into an `opentelemetry::Context`. **Always
///      runs**, even without an exporter — propagation is the
///      load-bearing piece for worker-side correlation.
///   2. Independently resolve trace and metric exporters. Tracing needs
///      `SIE_TRACING_ENABLED`; canonical metrics need `SIE_METRICS_ENABLED`.
///      Either signal can roll out alone and either setup failure is fail-open.
///   3. If tracing is active, attach a
///      [`tracing_opentelemetry::OpenTelemetryLayer`] so the
///      `sidecar.dispatch` `tracing::*` span becomes an OTel span. Otherwise,
///      install only the JSON fmt layer while propagation remains active.
///
/// Logs are always emitted as JSON, matching the sidecar's prior
/// behaviour. `RUST_LOG` is honoured via `EnvFilter`.
pub fn init_tracing() {
    // Idempotency guard. The subscriber's `.init()` panics on a second
    // call ("a global default trace dispatcher has already been set").
    // The integration tests spin the sidecar up in-process across
    // multiple cases and would otherwise abort on the second case.
    use std::sync::atomic::{AtomicBool, Ordering};
    static INIT_GUARD: AtomicBool = AtomicBool::new(false);
    if INIT_GUARD.swap(true, Ordering::SeqCst) {
        tracing::debug!("init_tracing called more than once; skipping subsequent init");
        return;
    }

    // Step 1: always install the propagator. Even without an OTLP
    // exporter the sidecar needs to *extract* the inbound gateway
    // context and *inject* it onto the wire items so the worker side
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

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Try to build the OTel tracer. On any failure the sidecar
    // continues with the fmt-only subscriber (and propagator-only OTel
    // state) — operators get logs and worker-side trace correlation,
    // just no `sidecar.dispatch` spans.
    let tracer = trace_config
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
    let tracing_initialized = tracer.is_some();

    // Metrics initialise independently of trace setup. This intentionally runs
    // before `RuntimeState::new()` binds the instruments to the global meter.
    let metrics_initialized = endpoints.metrics.as_deref().is_some_and(init_metrics);

    // Place the (boxed) OTel layer FIRST, then the fmt and filter
    // layers — see the gateway's note: boxing against the inner
    // `Registry` is what makes `Option<L>: Layer<S>` compose; OTel-last
    // fails the `Layer<Layered<…>>` bound.
    let otel_layer_boxed: Option<Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync>> =
        tracer.map(|t| {
            let l: Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync> =
                Box::new(tracing_opentelemetry::layer().with_tracer(t));
            l
        });

    tracing_subscriber::registry()
        .with(otel_layer_boxed)
        .with(filter)
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    if let Some(config) = trace_config.as_ref() {
        if tracing_initialized {
            tracing::info!(
                endpoint = %endpoint_origin_for_log(&config.endpoint),
                protocol = ?config.protocol,
                "OpenTelemetry tracing initialized"
            );
        }
    } else if trace_config_error.is_some() {
        tracing::warn!("OpenTelemetry trace exporter disabled after configuration failure");
    } else if endpoints.tracing_enabled {
        tracing::warn!("SIE_TRACING_ENABLED set but no trace OTLP endpoint; tracing disabled");
    }
    if let Some(ep) = endpoints.metrics.as_deref() {
        if metrics_initialized {
            tracing::info!(
                endpoint = %endpoint_origin_for_log(ep),
                "OpenTelemetry bounded metric export initialized"
            );
        }
    } else if endpoints.metrics_enabled {
        tracing::warn!("SIE_METRICS_ENABLED set but no metric OTLP endpoint; metrics disabled");
    }
    if !endpoints.tracing_enabled && !endpoints.metrics_enabled {
        tracing::debug!("telemetry exporters disabled; W3C propagator installed (no exporter)");
    }
}

/// Trace transport: trace-specific setting, then generic, then gRPC.
#[cfg(test)]
fn select_otlp_protocol(
    trace_protocol: Option<&str>,
    generic_protocol: Option<&str>,
) -> Result<OtlpProtocol, String> {
    sie_telemetry::transport::select_signal_protocol(trace_protocol, generic_protocol, "traces")
}

/// Metrics transport: metrics-specific setting, then generic, then gRPC.
#[cfg(test)]
fn select_metrics_protocol(
    metrics_protocol: Option<&str>,
    generic_protocol: Option<&str>,
) -> Result<OtlpProtocol, String> {
    sie_telemetry::transport::select_signal_protocol(metrics_protocol, generic_protocol, "metrics")
}

/// Resolve the sidecar service identity while honoring every non-empty operator
/// override. The managed launcher sets its child override explicitly; the
/// library must not reinterpret a deliberate OSS/Helm value.
fn sidecar_service_name(configured: Option<&str>) -> String {
    match configured.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) => value.to_string(),
        None => DEFAULT_SERVICE_NAME.to_string(),
    }
}

/// Build the resource attributes shared with the gateway and managed lanes.
/// The sidecar's historical shape: no `service.version` attribute and no
/// `SIE_DEPLOYMENT_ENV` / `AWS_REGION` fallbacks (unlike gateway + worker —
/// drift flagged in #2339, deliberately not normalized here).
fn otlp_resource(service_name: &str) -> Resource {
    otlp_resource_from_values(
        service_name,
        &service_instance_id(instance_prefix_env().as_deref()),
        cleaned_env("SIE_OTEL_DEPLOYMENT_ENVIRONMENT").as_deref(),
        cleaned_env("SIE_OTEL_CLOUD_REGION").as_deref(),
        cleaned_env("SIE_CLOUD_REGION").as_deref(),
    )
}

fn otlp_resource_from_values(
    service_name: &str,
    service_instance_id: &str,
    deployment_environment: Option<&str>,
    otel_cloud_region: Option<&str>,
    cloud_region: Option<&str>,
) -> Resource {
    resource_from_values(
        service_name,
        service_instance_id,
        deployment_environment.unwrap_or(UNKNOWN_RESOURCE_VALUE),
        otel_cloud_region
            .or(cloud_region)
            .unwrap_or(UNKNOWN_RESOURCE_VALUE),
        None,
    )
}

fn init_tracer(config: &SignalExportConfig) -> Result<Tracer, String> {
    let service_name = sidecar_service_name(cleaned_env("OTEL_SERVICE_NAME").as_deref());

    let exporter = build_span_exporter(config)?;

    let provider = SdkTracerProvider::builder()
        .with_resource(otlp_resource(&service_name))
        .with_batch_exporter(exporter)
        .build();
    let tracer = provider.tracer("sie-worker-sidecar");
    global::set_tracer_provider(provider.clone());
    let _ = TRACER_PROVIDER.set(provider);
    Ok(tracer)
}

/// Install the MeterProvider that pushes the bounded dotted queue/batch
/// contract. Fail-open: exporter or endpoint failures only disable telemetry.
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
        .with_interval(Duration::from_secs(METRICS_EXPORT_INTERVAL_S))
        .build();
    let service_name = sidecar_service_name(cleaned_env("OTEL_SERVICE_NAME").as_deref());
    let provider = SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(otlp_resource(&service_name))
        .with_view(sidecar_metric_cardinality_view)
        .build();
    global::set_meter_provider(provider.clone());
    let _ = METER_PROVIDER.set(provider);
    true
}

/// Apply the checked-in finite-domain ceiling to every sidecar instrument.
/// Most limits are below the SDK default. Batch fill and generation-loading
/// are intentionally higher because their comprehensive bounded dimensions
/// have a larger valid Cartesian product; keeping explicit limits prevents a
/// valid catalog label from becoming `otel.metric.overflow`.
pub(crate) fn sidecar_metric_cardinality_view(instrument: &Instrument) -> Option<Stream> {
    let limit = super::metrics::sidecar_metric_cardinality_limit(instrument.name())?;
    Some(
        Stream::builder()
            .with_cardinality_limit(limit)
            .build()
            .expect("constant sidecar cardinality limits must be valid"),
    )
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

/// Graceful shutdown — flush any pending spans and metric points.
///
/// Called from `main.rs` on the way out so the OTLP exporter has a
/// chance to drain its batch before the process exits.
pub fn shutdown_tracing() {
    if let Some(provider) = TRACER_PROVIDER.get() {
        let _ = provider.shutdown_with_timeout(Duration::from_millis(TRACING_SHUTDOWN_TIMEOUT_MS));
    }
    if let Some(provider) = METER_PROVIDER.get() {
        let _ = provider.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry::trace::{Span as _, Tracer as _};
    use opentelemetry::{Key, Value};
    use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;
    use std::thread;

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
    fn protocol_selection_honors_trace_specific_then_generic() {
        assert_eq!(
            select_otlp_protocol(Some("http/protobuf"), Some("grpc")),
            Ok(OtlpProtocol::Http)
        );
        assert_eq!(
            select_otlp_protocol(None, Some(" HTTP/PROTOBUF ")),
            Ok(OtlpProtocol::Http)
        );
        assert_eq!(
            select_otlp_protocol(Some("grpc"), Some("http/protobuf")),
            Ok(OtlpProtocol::Grpc)
        );
        assert_eq!(select_otlp_protocol(None, None), Ok(OtlpProtocol::Grpc));
        assert_eq!(
            select_metrics_protocol(Some("http/protobuf"), Some("grpc")),
            Ok(OtlpProtocol::Http)
        );
        assert_eq!(
            select_metrics_protocol(None, Some("grpc")),
            Ok(OtlpProtocol::Grpc)
        );
        assert_eq!(select_metrics_protocol(None, None), Ok(OtlpProtocol::Grpc));
        for unsupported in ["http", "http/json", "thrift"] {
            assert!(select_metrics_protocol(Some(unsupported), None).is_err());
        }
    }

    #[test]
    fn http_metrics_endpoint_is_derived_and_explicit_override_wins() {
        assert_eq!(
            derive_metrics_endpoint(
                "https://workspace--collector.modal.run",
                OtlpProtocol::Http,
                None,
                None,
            ),
            "https://workspace--collector.modal.run/v1/metrics"
        );
        assert_eq!(
            derive_metrics_endpoint(
                "https://workspace--collector.modal.run",
                OtlpProtocol::Http,
                None,
                Some("https://workspace--collector.modal.run/"),
            ),
            "https://workspace--collector.modal.run/v1/metrics"
        );
        assert_eq!(
            derive_metrics_endpoint(
                "https://workspace--collector.modal.run/v1/traces",
                OtlpProtocol::Http,
                Some("https://workspace--collector.modal.run/custom-metrics"),
                None,
            ),
            "https://workspace--collector.modal.run/custom-metrics"
        );
        assert_eq!(
            derive_metrics_endpoint("http://collector:4317", OtlpProtocol::Grpc, None, None,),
            "http://collector:4317"
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
            .u64_counter("sie.worker.ipc.requests")
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
        assert!(captured
            .body
            .windows(b"sie.worker.ipc.requests".len())
            .any(|window| window == b"sie.worker.ipc.requests"));
        let _ = provider.shutdown();
        match prior_proxy_auth {
            Some(value) => env::set_var("SIE_MODAL_PROXY_AUTH", value),
            None => env::remove_var("SIE_MODAL_PROXY_AUTH"),
        }
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
    fn modal_proxy_headers_require_exact_trusted_endpoint_and_complete_pair() {
        let headers = modal_proxy_headers_from_values(
            "https://workspace--collector.modal.run/v1/traces",
            "/v1/traces",
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            Some("secret"),
        )
        .expect("trusted endpoint");
        assert_eq!(headers.get("Modal-Key").map(String::as_str), Some("id"));
        assert_eq!(
            headers.get("Modal-Secret").map(String::as_str),
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

        assert!(modal_proxy_headers_from_values(
            "https://otel.example/v1/traces",
            "/v1/traces",
            false,
            None,
            Some("id"),
            Some("secret"),
        )
        .expect("OSS endpoint when managed auth is off")
        .is_empty());

        let metric_headers = modal_proxy_headers_from_values(
            "https://workspace--collector.modal.run/v1/metrics",
            "/v1/metrics",
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            Some("secret"),
        )
        .expect("trusted metrics endpoint");
        assert_eq!(
            metric_headers.get("Modal-Key").map(String::as_str),
            Some("id")
        );
        assert!(modal_proxy_headers_from_values(
            "https://workspace--collector.modal.run/v1/traces",
            "/v1/metrics",
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
    fn service_name_defaults_when_absent_and_honors_operator_override() {
        assert_eq!(sidecar_service_name(None), DEFAULT_SERVICE_NAME);
        assert_eq!(sidecar_service_name(Some("sie-worker")), "sie-worker");
        assert_eq!(
            sidecar_service_name(Some("custom-sidecar")),
            "custom-sidecar"
        );
    }

    #[test]
    fn resource_contains_environment_and_preferred_region() {
        let resource = otlp_resource_from_values(
            DEFAULT_SERVICE_NAME,
            "pod-uid/worker-sidecar",
            Some("staging"),
            Some("us-east-1"),
            Some("fallback-region"),
        );
        assert_eq!(
            resource.get(&Key::new("service.name")),
            Some(Value::from(DEFAULT_SERVICE_NAME))
        );
        assert_eq!(
            resource.get(&Key::new("service.instance.id")),
            Some(Value::from("pod-uid/worker-sidecar"))
        );
        assert_eq!(
            resource.get(&Key::new("deployment.environment")),
            Some(Value::from("staging"))
        );
        assert_eq!(
            resource.get(&Key::new("cloud.region")),
            Some(Value::from("us-east-1"))
        );
    }

    #[test]
    fn resource_always_contains_required_attributes() {
        let resource =
            otlp_resource_from_values(DEFAULT_SERVICE_NAME, "process-boot-uuid", None, None, None);
        assert_eq!(
            resource.get(&Key::new("service.name")),
            Some(Value::from(DEFAULT_SERVICE_NAME))
        );
        assert_eq!(
            resource.get(&Key::new("service.instance.id")),
            Some(Value::from("process-boot-uuid"))
        );
        assert_eq!(
            resource.get(&Key::new("deployment.environment")),
            Some(Value::from(UNKNOWN_RESOURCE_VALUE))
        );
        assert_eq!(
            resource.get(&Key::new("cloud.region")),
            Some(Value::from(UNKNOWN_RESOURCE_VALUE))
        );
    }

    #[test]
    fn instance_id_appends_stable_process_uuid_to_substrate_prefix() {
        let prefixed = service_instance_id(Some(" pod-uid/worker-sidecar/ "));
        let repeated = service_instance_id(Some("pod-uid/worker-sidecar"));
        assert_eq!(prefixed, repeated);
        let suffix = prefixed
            .strip_prefix("pod-uid/worker-sidecar/")
            .expect("configured substrate prefix");
        assert!(uuid::Uuid::parse_str(suffix).is_ok());

        let first = service_instance_id(None);
        let second = service_instance_id(Some(""));
        assert!(!first.is_empty());
        assert_eq!(first, second, "fallback must be stable across all signals");
        assert_eq!(first, suffix);
    }

    #[test]
    fn restarted_process_never_reuses_prefixed_instance_id() {
        let first_process = uuid::Uuid::new_v4().to_string();
        let restarted_process = uuid::Uuid::new_v4().to_string();

        assert_ne!(
            compose_service_instance_id(Some("pod-uid/worker-sidecar"), &first_process),
            compose_service_instance_id(Some("pod-uid/worker-sidecar"), &restarted_process)
        );
    }

    #[test]
    fn http_exporter_builds_with_tls() {
        build_span_exporter(&SignalExportConfig {
            endpoint: "https://collector.example/v1/traces".to_string(),
            protocol: OtlpProtocol::Http,
        })
        .expect("OTLP/HTTP exporter must build against an HTTPS endpoint");
        build_metric_exporter("https://collector.example/v1/metrics", OtlpProtocol::Http)
            .expect("OTLP/HTTP metric exporter must build against an HTTPS endpoint");
    }
}
