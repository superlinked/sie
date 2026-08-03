//! Shared OpenTelemetry *plumbing* for the SIE Rust crates (#2339).
//!
//! Extracted from the three copy-forked `observability/` stacks
//! (`sie_gateway`, `sie_server_sidecar`, `sie_server_rust`): OTLP
//! transport/endpoint/protocol resolution, Modal proxy authentication,
//! resource identity, exporter construction, env helpers, and W3C trace
//! propagation. Extraction preserved each consumer's behavior — the crate
//! exposes `_from_values` seams and parameters (e.g. `service_version`)
//! wherever the consumers deliberately differ.
//!
//! Deliberately NOT here: signal *definitions* — metric names, views,
//! cardinality limits, subscriber/layer wiring, the gateway's safe-log
//! pipeline. Those are per-crate surfaces (their name sets are disjoint;
//! see `telemetry/contract.yaml` and the parity test).

pub mod env;
pub mod exporters;
pub mod propagation;
pub mod proxy;
pub mod resource;
pub mod transport;
