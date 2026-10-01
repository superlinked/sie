//! Per-item [`ItemOutcome`] helpers shared by every native backend.
//!
//! A native dense encoder ends up producing the same two payload
//! shapes: a successful dense-vector outcome and a failure outcome
//! with `PublishErrorAndAck` + an error code. Keep the boilerplate
//! here so the engine code in each architecture stays focused on the
//! forward pass.

use rmp::encode::{write_map_len, write_str};

use super::numpy_sentinel::{write_f32_vector, SentinelError};
use crate::ipc_types::{Disposition, EncodeBatchItem, ItemOutcome, ScoreBatchItem};

/// Routing handles we stash per-item *before* moving `items` into a
/// blocking task, so the failure path can synthesise outcomes
/// without cloning the full batch payload.
pub struct EncodeItemHandle {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
}

impl EncodeItemHandle {
    pub fn from_encode(bi: &EncodeBatchItem) -> Self {
        Self {
            work_item_id: bi.work_item_id.clone(),
            request_id: bi.request_id.clone(),
            item_index: bi.item_index,
        }
    }

    /// Convert the handle into a `PublishErrorAndAck` outcome.
    /// Used in the blocking-task panic / error fan-out paths where
    /// we no longer have the original `EncodeBatchItem`.
    pub fn into_error(self, code: &'static str, message: String) -> ItemOutcome {
        ItemOutcome {
            work_item_id: self.work_item_id,
            request_id: self.request_id,
            item_index: self.item_index,
            disposition: Disposition::PublishErrorAndAck,
            result_msgpack: Vec::new(),
            nak_delay_ms: None,
            error: Some(message),
            error_code: Some(code.into()),
            inference_ms: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            raw_output: None,
            units: None,
            retry_after_s: None,
        }
    }
}

/// Build an `PublishErrorAndAck` outcome from a live
/// [`EncodeBatchItem`]. Used inside the engine when we still hold
/// the original item and want to emit a per-item error alongside
/// other successful results.
pub fn encode_error_outcome(item: &EncodeBatchItem, code: &str, message: String) -> ItemOutcome {
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::PublishErrorAndAck,
        result_msgpack: Vec::new(),
        nak_delay_ms: None,
        error: Some(message),
        error_code: Some(code.into()),
        inference_ms: None,
        tokenization_ms: None,
        postprocessing_ms: None,
        raw_output: None,
        units: None,
        retry_after_s: None,
    }
}

/// Same as [`encode_error_outcome`] but for score-batch items.
pub fn encode_error_outcome_score(
    item: &ScoreBatchItem,
    code: &str,
    message: String,
) -> ItemOutcome {
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::PublishErrorAndAck,
        result_msgpack: Vec::new(),
        nak_delay_ms: None,
        error: Some(message),
        error_code: Some(code.into()),
        inference_ms: None,
        tokenization_ms: None,
        postprocessing_ms: None,
        raw_output: None,
        units: None,
        retry_after_s: None,
    }
}

/// Build the per-item `result_msgpack` bytes for a dense-only
/// encoder outcome, matching the Python queue executor's wire shape
/// verbatim:
///
/// ```text
/// {
///   "dense": {
///     "dims":   <hidden>,
///     "dtype":  "float32",
///     "values": <numpy_sentinel for [hidden] f32>,
///   },
/// }
/// ```
///
/// Top-level keys are packed as msgpack **str** (Python source
/// strings). The `values` entry uses the [`numpy_sentinel`][super::numpy_sentinel]
/// map form so the SDK's `msgpack_numpy.patch()`ed decoder round-
/// trips the array back into a real `numpy.ndarray`, exactly as it
/// does for the HTTP/queue Python paths.
///
/// If you later extend this to emit sparse / multivector, add the
/// corresponding sibling keys (`"sparse"`, `"multivector"`) before
/// the final map-len update; see `_wrap_encode_output` in
/// `sie_server/queue_executor.py` for the reference shape.
pub fn build_dense_payload(vec: &[f32], hidden: usize) -> Result<Vec<u8>, SentinelError> {
    if vec.len() != hidden {
        return Err(SentinelError::MsgpackWrite(format!(
            "dense dims mismatch: hidden={} vec_len={}",
            hidden,
            vec.len()
        )));
    }
    let mut buf = Vec::with_capacity(32 + vec.len() * 4);

    // Top-level map: {"dense": ...}.
    write_map_len(&mut buf, 1).map_err(|e| SentinelError::MsgpackWrite(e.to_string()))?;
    write_str(&mut buf, "dense").map_err(|e| SentinelError::MsgpackWrite(e.to_string()))?;

    // Inner "dense" envelope: {"dims", "dtype", "values"}.
    write_map_len(&mut buf, 3).map_err(|e| SentinelError::MsgpackWrite(e.to_string()))?;
    write_str(&mut buf, "dims").map_err(|e| SentinelError::MsgpackWrite(e.to_string()))?;
    rmp::encode::write_uint(&mut buf, hidden as u64)
        .map_err(|e| SentinelError::MsgpackWrite(e.to_string()))?;
    write_str(&mut buf, "dtype").map_err(|e| SentinelError::MsgpackWrite(e.to_string()))?;
    write_str(&mut buf, "float32").map_err(|e| SentinelError::MsgpackWrite(e.to_string()))?;
    write_str(&mut buf, "values").map_err(|e| SentinelError::MsgpackWrite(e.to_string()))?;
    write_f32_vector(&mut buf, vec)?;

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden: Python side runs
    ///   `msgpack.packb({"dense": {"dims": 3, "dtype": "float32",
    ///                             "values": np.array([1,2,3], dtype=f32)}},
    ///                  use_bin_type=True)`
    /// after `msgpack_numpy.patch()`. Captured from CPython (88 bytes).
    /// If you change the dense envelope, regenerate this with:
    /// `msgpack.packb(...).hex()` and paste the new hex here.
    const GOLDEN_DENSE_123: &[u8] = &[
        0x81, 0xa5, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x83, 0xa4, 0x64, 0x69, 0x6d, 0x73, 0x03, 0xa5,
        0x64, 0x74, 0x79, 0x70, 0x65, 0xa7, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x33, 0x32, 0xa6, 0x76,
        0x61, 0x6c, 0x75, 0x65, 0x73, 0x85, 0xc4, 0x02, 0x6e, 0x64, 0xc3, 0xc4, 0x04, 0x74, 0x79,
        0x70, 0x65, 0xa3, 0x3c, 0x66, 0x34, 0xc4, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0xc4, 0x00, 0xc4,
        0x05, 0x73, 0x68, 0x61, 0x70, 0x65, 0x91, 0x03, 0xc4, 0x04, 0x64, 0x61, 0x74, 0x61, 0xc4,
        0x0c, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
    ];

    #[test]
    fn build_dense_payload_matches_python_wire_format() {
        let buf = build_dense_payload(&[1.0, 2.0, 3.0], 3).unwrap();
        assert_eq!(
            buf, GOLDEN_DENSE_123,
            "dense payload must be byte-identical to msgpack-numpy Python output"
        );
    }

    #[test]
    fn build_dense_payload_rejects_dim_mismatch() {
        let err = build_dense_payload(&[1.0, 2.0], 3).unwrap_err();
        assert!(err.to_string().contains("dense dims mismatch"));
    }
}
