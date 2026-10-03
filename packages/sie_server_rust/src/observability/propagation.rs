//! W3C Trace Context extraction helpers for the Rust worker.
//!
//! Shared implementation in `sie-telemetry` (#2339): the worker extracts the
//! W3C `traceparent` / `tracestate` strings it receives on
//! [`crate::ipc_types::RunBatchItem`] via the global propagator installed in
//! [`super::tracing::init_tracing`] so `worker.run_batch` can attach to the
//! inbound trace.

pub use sie_telemetry::propagation::{extract_context_from_w3c, remote_span_context};
