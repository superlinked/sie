//! OpenTelemetry tracer-provider setup for the Rust worker.
//!
//! The OTLP exporter is enabled only when `SIE_TRACING_ENABLED` is truthy and
//! an OTLP endpoint is configured via `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or
//! `OTEL_EXPORTER_OTLP_ENDPOINT`. The W3C propagator is installed globally
//! regardless of that exporter gate so inbound `traceparent` strings on IPC
//! `RunBatchItem`s still extract into an `opentelemetry::Context`.

use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{SdkTracerProvider, Tracer};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use sie_telemetry::env::sie_tracing_enabled;

use super::resource::telemetry_resource;
use super::transport::{
    build_span_exporter, endpoint_origin_for_log, trace_export_config, SignalExportConfig,
};

static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

/// Bounded flush deadline (ms) so process exit can't stall on an unreachable collector.
const TRACING_SHUTDOWN_TIMEOUT_MS: u64 = 3_000;

/// Initialise OpenTelemetry + tracing-subscriber for the Rust worker.
pub fn init_tracing() {
    // Idempotency guard. The subscriber's `.init()` panics on a second call.
    static INIT_GUARD: AtomicBool = AtomicBool::new(false);
    if INIT_GUARD.swap(true, Ordering::SeqCst) {
        tracing::debug!("init_tracing called more than once; skipping subsequent init");
        return;
    }

    global::set_text_map_propagator(TraceContextPropagator::new());

    let tracing_enabled = sie_tracing_enabled();
    let (export_config, transport_error) = match trace_export_config(tracing_enabled) {
        Ok(config) => (config, None),
        Err(error) => (None, Some(error)),
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let json_logs = env_bool_value(env::var("SIE_LOG_JSON").ok().as_deref(), false);

    let tracer = export_config
        .as_ref()
        .and_then(|config| match init_tracer(config) {
            Ok(t) => Some(t),
            Err(_) => {
                eprintln!("warn: failed to init OTLP exporter; continuing without exporter");
                None
            }
        });
    let exporter_enabled = tracer.is_some();

    let mut layers: Vec<Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync>> = Vec::new();
    if let Some(t) = tracer {
        layers.push(Box::new(tracing_opentelemetry::layer().with_tracer(t)));
    }

    let fmt_layer_boxed: Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync> = if json_logs {
        Box::new(tracing_subscriber::fmt::layer().json().with_target(false))
    } else {
        Box::new(tracing_subscriber::fmt::layer().with_target(false))
    };
    layers.push(fmt_layer_boxed);

    tracing_subscriber::registry()
        .with(layers)
        .with(filter)
        .init();

    if exporter_enabled {
        let config = export_config
            .as_ref()
            .expect("exporter_enabled implies export config is set");
        tracing::info!(endpoint = %endpoint_origin_for_log(&config.endpoint), protocol = ?config.protocol, "OpenTelemetry tracing initialized");
    } else if transport_error.is_some() {
        tracing::warn!("invalid OTLP trace transport configuration; tracing disabled");
    } else if export_config.is_none() {
        tracing::debug!(
            "SIE_TRACING_ENABLED not truthy or OTLP endpoint not set; W3C propagator installed (no exporter)"
        );
    } else {
        tracing::warn!("OTLP exporter init failed; W3C propagator installed (no exporter)");
    }
}

fn env_bool_value(raw: Option<&str>, default: bool) -> bool {
    let Some(raw) = raw else {
        return default;
    };
    let trimmed = raw.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return default;
    }
    !matches!(trimmed.as_str(), "0" | "false" | "no" | "off")
}

fn init_tracer(config: &SignalExportConfig) -> Result<Tracer, String> {
    let exporter = build_span_exporter(config)?;

    let provider = SdkTracerProvider::builder()
        .with_resource(telemetry_resource())
        .with_batch_exporter(exporter)
        .build();
    let tracer = provider.tracer("sie-worker");
    global::set_tracer_provider(provider.clone());
    let _ = TRACER_PROVIDER.set(provider);
    Ok(tracer)
}

/// Graceful shutdown: flush any pending spans.
pub fn shutdown_tracing() {
    if let Some(provider) = TRACER_PROVIDER.get() {
        let _ = provider.shutdown_with_timeout(Duration::from_millis(TRACING_SHUTDOWN_TIMEOUT_MS));
    }
}
