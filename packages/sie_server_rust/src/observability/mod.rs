//! Rust worker observability: one OTLP metrics/tracing pipeline.
//!
//! The native worker is the leaf process on the queue-mode inference path. It
//! owns provider setup and helpers for extracting `traceparent` / `tracestate`
//! strings carried on IPC `RunBatchItem`s.

pub mod metrics;
pub mod propagation;
mod resource;
pub mod tracing;
mod transport;

pub fn init_observability() {
    transport::install_crypto_provider();
    tracing::init_tracing();
    metrics::init_metrics();
}

pub fn shutdown_observability() {
    metrics::shutdown_metrics();
    tracing::shutdown_tracing();
}
