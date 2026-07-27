//! Canonical identities shared by queue producers and result consumers.

/// Return the internal wire identity for one item in a request.
///
/// Queue result storage is indexed by `item_index`; this one-to-one mapping
/// prevents a second transfer identity from aliasing the same result slot.
pub fn canonical_work_item_id(request_id: &str, item_index: u32) -> String {
    format!("{request_id}.{item_index}")
}
