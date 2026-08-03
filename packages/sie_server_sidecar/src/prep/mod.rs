//! GPU-independent inference preparation for the SIE worker sidecar.
//!
//! This module owns the prep/protocol helpers that previously lived in a
//! standalone prep crate: text-template application, numpy-sentinel
//! encoding, typed output payload builders, and the sidecar-side error
//! taxonomy. Keeping them in-tree avoids a second local Rust package while
//! preserving the same wire behavior.

pub mod error;
pub mod media;
pub mod numpy_sentinel;
pub mod outcome;
pub mod output_types;
pub mod text_prep;
