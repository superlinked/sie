//! Gateway-side observability: W3C Trace Context propagation.
//!
//! Owns the OpenTelemetry tracer-provider setup, the global W3C
//! propagator install, and the helpers for extracting an inbound
//! HTTP request's parent context plus injecting the active span's
//! context into the JetStream work envelope.
//!
//! The split between [`tracing`] (used throughout the gateway for
//! structured logs) and OpenTelemetry is intentional: `tracing::*`
//! spans become OTel spans via the [`tracing_opentelemetry`] layer,
//! so existing log call-sites contribute to the trace without code
//! changes. The helpers in [`propagation`] are only needed at the
//! *edges* — inbound HTTP and outbound NATS publish.

pub mod metrics;
pub mod propagation;
pub mod tracing;
