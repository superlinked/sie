//! Rust inference-engine integrations for SIE server workers.
//!
//! This crate owns engine-specific client/runtime code. Queueing, NATS,
//! SIE work-item routing, and result publishing stay in
//! `sie-server-sidecar`; that crate adapts these clients to the SIE
//! backend trait.

pub mod candle_backend;
mod candle_bert_flash;
pub mod candle_embedding;
mod candle_gte_rope;
mod candle_layers;
mod candle_modernbert;
mod candle_residency;
mod candle_rope;
mod candle_splade;
mod candle_xlm_roberta;
pub mod ipc;
pub mod ipc_types;
pub mod native_backend;
pub mod observability;
pub mod text_prep;
