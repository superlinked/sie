//! Sidecar-side observability: bounded OTLP metrics, tracing, and W3C Trace Context.
//!
//! The sidecar sits on the queue hop between the gateway and the
//! adapter worker. It owns the OpenTelemetry tracer-provider setup, the
//! global W3C propagator install, and the helpers for extracting the
//! inbound gateway context off the work envelope plus injecting the
//! `sidecar.dispatch` span's context onto the IPC `RunBatchItem`s.
//!
//! Mirrors `packages/sie_gateway/src/observability/`. Unlike the
//! gateway, the propagation boundary is W3C *strings* carried on the
//! work envelope (the sidecar has no inbound `HeaderMap`), so the
//! extractor here is `HashMap`-backed rather than header-backed.

pub mod metrics;
pub mod propagation;
pub mod tracing;
