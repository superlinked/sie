//! W3C Trace Context propagation helpers.
//!
//! The gateway sits at two propagation boundaries:
//!
//! 1. **Inbound HTTP → in-process Context**. The client sends a
//!    `traceparent` (and optional `tracestate`) header. We extract it
//!    via the globally-installed [`TextMapPropagator`] and attach the
//!    resulting context to a new gateway-side span.
//!
//! 2. **In-process Context → outbound NATS envelope**. Before
//!    publishing the [`crate::queue::publisher::WorkItem`] we
//!    serialise the current context (the gateway span) back into the
//!    two header strings and write them into the envelope. The
//!    worker re-extracts on the other side.
//!
//! Both directions go through the **same** [`TextMapPropagator`]
//! instance — the global W3C propagator installed in
//! [`super::tracing::init_tracing`] — so the wire format is
//! guaranteed identical in both directions.

pub use sie_telemetry::propagation::{extract_context_from_headers, inject_current_context};
