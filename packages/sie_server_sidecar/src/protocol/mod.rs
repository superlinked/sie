//! Wire protocol definitions shared by the sidecar and Python adapter.
//!
//! The Rust IPC schema is intentionally kept in one module and checked against
//! `packages/sie_server/src/sie_server/ipc_types.py` by
//! `tools/ci/check_ipc_types_parity.py`.

pub mod ipc_types;
pub(crate) mod response_chunks;
