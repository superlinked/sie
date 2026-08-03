//! W3C Trace Context propagation helpers for the sidecar.
//!
//! The sidecar sits between two propagation boundaries on the queue
//! hop:
//!
//! 1. **Inbound envelope → in-process Context**. The gateway serialised
//!    its span into the work envelope's `traceparent` / `tracestate`
//!    strings ([`crate::work_types::WorkItem`]). We extract them via
//!    the globally-installed propagator and parent the
//!    `sidecar.dispatch` span on the result.
//!
//! 2. **In-process Context → outbound IPC item**. Before sending the
//!    [`crate::ipc_types::RunBatchItem`] to the adapter worker we
//!    serialise the active `sidecar.dispatch` span back into the two
//!    header strings and write them onto the wire item, so the
//!    worker's `worker.run_batch` span nests under the sidecar span.
//!
//! Both directions go through the **same** propagator instance — the
//! global W3C propagator installed in [`super::tracing::init_tracing`]
//! — so the wire format is identical in both directions. Unlike the
//! gateway, the inbound carrier is a pair of W3C *strings* (not an
//! HTTP `HeaderMap`), so the extractor here is `HashMap`-backed.

pub use sie_telemetry::propagation::{
    extract_context_from_w3c, inject_context, inject_current_context, remote_span_context,
};
