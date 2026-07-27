//! Library target for the SIE gateway.
//!
//! Exposes the gateway's module tree as a library so a downstream
//! composition crate can reuse the HTTP router, registries, and the
//! [`queue::dispatch::WorkDispatcher`] seam without forking the
//! binary. The `sie-gateway` binary keeps its own module tree in
//! `main.rs`; the two targets compile the same module files
//! independently — an accepted POC tradeoff (a single-tree split where
//! `main.rs` consumes the lib is deferred).
//!
//! Everything here is a plain re-declaration of the binary's modules;
//! no lib-only code lives in this file.

pub mod config;
pub mod discovery;
pub mod endpoint;
pub mod error;
pub mod handlers;
pub mod health_mode;
pub mod http_error;
pub mod middleware;
pub mod nats;
pub mod observability;
pub mod openapi;
pub mod queue;
pub mod routing;
pub mod server;
pub mod state;
pub mod types;
