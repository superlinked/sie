//! Rust-side output shaping and result framing.
//!
//! Instead of Python building the full
//! `msgpack(WorkResult.result_msgpack)` envelope, the Python adapter
//! can emit a typed [`crate::ipc_types::RawOutput`] and let Rust
//! assemble the final on-wire bytes here.
//!
//! # Why this lives at top-level
//!
//! The canonical msgpack-numpy sentinel writer lives in
//! [`crate::prep::numpy_sentinel`]. This top-level module keeps the
//! dispatcher/publisher-facing output-shaping API.
//!
//! # Backend-agnostic
//!
//! The shapers below only touch host-side `Vec<f32>` / `Vec<String>`
//! data. They are identical whether the values come from a Python
//! adapter (copied off-device inside Python) or a future native
//! adapter (e.g. `tch-rs`'s `.to_kind(Kind::Float).to(Device::Cpu).data()`).
//! Moving output shaping into Rust lands the same wire-format
//! guarantee for every backend.
//!
//! # Non-destructive contract
//!
//! * Supported: `dense`, `sparse`, and `multivector` encode outputs,
//!   plus `score` rerank outputs.
//! * Deferred: `json` extract payloads, MUVERA post-processing, and
//!   quantization beyond the explicit shaper dtypes.
//! * When Python emits the legacy `result_msgpack` or when the Rust
//!   build is missing a shaper for a given [`crate::ipc_types::RawOutput`] variant,
//!   the publisher passes through the Python bytes unchanged.
//!
//! See `packages/sie_server_sidecar/docs/architecture-guide.md`
//! for the full design and
//! `packages/sie_server/src/sie_server/queue_executor.py::_wrap_encode_output`
//! for the Python reference wire shape this matches byte-for-byte.

pub mod numpy_sentinel;

use rmp::encode::{write_array_len, write_f64, write_map_len, write_nil, write_str, write_uint};

pub use numpy_sentinel::{
    write_f16_matrix_2d_from_f32, write_f16_matrix_2d_from_le_bytes, write_f32_matrix_2d,
    write_f32_vector, write_i32_vector, write_numpy_sentinel, SentinelError, DTYPE_F16, DTYPE_F32,
    DTYPE_I32,
};

/// Errors raised while shaping a `RawOutput` into final msgpack
/// bytes. Always recoverable: the publisher turns these into an
/// error `WorkResult` so one bad shaper can't drop the work item.
#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    #[error("dense output: expected values.len() == dim ({expected}), got {actual}")]
    DenseDimMismatch { expected: usize, actual: usize },

    #[error("score output: scores.len() ({scores}) != item_ids.len() ({items})")]
    ScoreLenMismatch { scores: usize, items: usize },

    #[error("sparse output: indices.len() ({indices}) != values.len() ({values})")]
    SparseLenMismatch { indices: usize, values: usize },

    #[error("multivector output: values.len() ({values}) != rows ({rows}) * cols ({cols})")]
    MultivectorShapeMismatch {
        values: usize,
        rows: usize,
        cols: usize,
    },

    #[error("msgpack write: {0}")]
    MsgpackWrite(String),

    #[error("sentinel write: {0}")]
    Sentinel(#[from] SentinelError),
}

fn wrap_err<E: std::fmt::Display>(e: E) -> ShapeError {
    ShapeError::MsgpackWrite(e.to_string())
}

/// Write the top-level encode-result map prefix, optionally echoing the
/// caller-provided item id before the embedding variant. The field order
/// mirrors Python's ``{"id": item_id, **output}`` framing exactly while
/// letting large multivectors stream directly into their final buffer.
fn write_encode_output_prefix(
    buf: &mut Vec<u8>,
    output_key: &str,
    item_id: Option<&str>,
) -> Result<(), ShapeError> {
    write_map_len(buf, if item_id.is_some() { 2 } else { 1 }).map_err(wrap_err)?;
    if let Some(item_id) = item_id {
        write_str(buf, "id").map_err(wrap_err)?;
        write_str(buf, item_id).map_err(wrap_err)?;
    }
    write_str(buf, output_key).map_err(wrap_err)?;
    Ok(())
}

/// L2-normalize a flat f32 vector in place.
///
/// Matches the Python adapter's `x / np.linalg.norm(x)` with the
/// same zero-norm guard (`norms = where(norms > 0, norms, 1.0)`).
/// Used when the Python side sets `DenseOutput.normalize = true` to
/// delegate the normalization step to Rust.
///
/// Stays numerically close to numpy: we sum `f64` partial squares
/// so a 1024-dim vector doesn't accumulate drift vs numpy's
/// per-row reduction (which also upcasts implicitly under BLAS).
pub fn l2_normalize_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let sq: f64 = values.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let norm = sq.sqrt();
    if norm <= 0.0 {
        return;
    }
    let norm = norm as f32;
    for v in values.iter_mut() {
        *v /= norm;
    }
}

/// Build the per-item `result_msgpack` bytes for a dense-only
/// encoder outcome, mirroring
/// [`_wrap_encode_output`][py-wrap] verbatim:
///
/// ```text
/// {
///   "dense": {
///     "dims":   <dim>,
///     "dtype":  "float32",
///     "values": <numpy_sentinel for [dim] f32>,
///   },
/// }
/// ```
///
/// Top-level keys are msgpack **str** (Python source-string keys),
/// matching what the SDK's `msgpack_numpy.patch()` decoder expects.
/// `values` uses the [`numpy_sentinel`] map form, identical byte
/// layout to
/// `msgpack.packb(np.array(values, dtype=np.float32))` after
/// `msgpack_numpy.patch()`.
///
/// [py-wrap]: ../../../../../../sie_server/src/sie_server/queue_executor.py
///
/// This wrapper delegates the actual msgpack writing to
/// [`crate::prep::outcome::build_dense_payload`], the canonical Rust
/// dense framer. It retains the sidecar's
/// stricter [`ShapeError::DenseDimMismatch`] runtime check (callers in
/// [`crate::publisher`] depend on that taxonomy) while letting the
/// `prep` module own the actual byte layout. Both paths still pass
/// the golden byte-for-byte tests in this module AND in
/// `prep::outcome::tests::build_dense_payload_matches_python_wire_format`.
pub fn build_dense_payload(values: &[f32], dim: usize) -> Result<Vec<u8>, ShapeError> {
    build_dense_payload_with_item_id(values, dim, None)
}

pub fn build_dense_payload_with_item_id(
    values: &[f32],
    dim: usize,
    item_id: Option<&str>,
) -> Result<Vec<u8>, ShapeError> {
    if values.len() != dim {
        return Err(ShapeError::DenseDimMismatch {
            expected: dim,
            actual: values.len(),
        });
    }
    if item_id.is_none() {
        return crate::prep::outcome::build_dense_payload(values, dim)
            .map_err(|e| ShapeError::MsgpackWrite(e.to_string()));
    }

    let mut buf =
        Vec::with_capacity(40 + item_id.map(str::len).unwrap_or_default() + values.len() * 4);
    write_encode_output_prefix(&mut buf, "dense", item_id)?;
    write_map_len(&mut buf, 3).map_err(wrap_err)?;
    write_str(&mut buf, "dims").map_err(wrap_err)?;
    write_uint(&mut buf, dim as u64).map_err(wrap_err)?;
    write_str(&mut buf, "dtype").map_err(wrap_err)?;
    write_str(&mut buf, "float32").map_err(wrap_err)?;
    write_str(&mut buf, "values").map_err(wrap_err)?;
    write_f32_vector(&mut buf, values)?;
    Ok(buf)
}

/// Build the per-item `result_msgpack` bytes for a score (rerank)
/// outcome, mirroring
/// [`_process_single_score`][py-score] verbatim:
///
/// ```text
/// [
///   {"item_id": "<id>", "score": <f64>, "rank": <uint>},
///   ...   // sorted by score desc
/// ]
/// ```
///
/// Matches the Python output byte-for-byte:
///
/// * top-level msgpack array
/// * each entry is a 3-key map, keys as msgpack `str` (`fixstr` for
///   `"item_id"` / `"score"` / `"rank"`)
/// * `score` packed as msgpack `float64` (`0xcb` + BE 8 bytes), same
///   as Python's `msgpack.packb(float)`
/// * `rank` packed as the shortest msgpack uint (fixint for the
///   usual small N), matching Python's rank integers
///
/// Ordering is by `scores[i]` descending, stable for ties (preserves
/// the caller's natural order), matching `scored.sort(key=..., reverse=True)`
/// in Python. (Python's `list.sort` is stable — Rust's
/// `sort_by` is not stable but `sort_by_cached_key` / `sort_by`
/// with a secondary key would be heavier; a stable sort is used
/// here via [`slice::sort_by`]'s stable alias
/// [`<[T]>::sort_by`].)
///
/// [py-score]: ../../../../../../sie_server/src/sie_server/queue_executor.py
pub fn build_score_payload(scores: &[f32], item_ids: &[String]) -> Result<Vec<u8>, ShapeError> {
    if scores.len() != item_ids.len() {
        return Err(ShapeError::ScoreLenMismatch {
            scores: scores.len(),
            items: item_ids.len(),
        });
    }
    // Stable sort by score descending. Python's `list.sort` is
    // stable; `<[T]>::sort_by` in stdlib is stable as well
    // (documented guarantee), so equal-scored items preserve their
    // input order across languages.
    let mut ordered: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ordered.sort_by(|a, b| {
        // Reverse: descending.
        // `partial_cmp` handles NaN by returning None; treat as
        // Equal so we never panic — Python does the same under
        // `reverse=True` (NaN comparisons yield False and numpy
        // scores are never NaN in practice).
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut buf = Vec::with_capacity(4 + ordered.len() * 48);
    write_array_len(&mut buf, ordered.len() as u32).map_err(wrap_err)?;
    for (rank, (orig_idx, score)) in ordered.iter().enumerate() {
        write_map_len(&mut buf, 3).map_err(wrap_err)?;
        write_str(&mut buf, "item_id").map_err(wrap_err)?;
        write_str(&mut buf, &item_ids[*orig_idx]).map_err(wrap_err)?;
        write_str(&mut buf, "score").map_err(wrap_err)?;
        // Python packs `float(score)` as f64 (msgpack `0xcb` + 8 BE
        // bytes). `rmp::write_f64` emits the same encoding.
        write_f64(&mut buf, f64::from(*score)).map_err(wrap_err)?;
        write_str(&mut buf, "rank").map_err(wrap_err)?;
        write_uint(&mut buf, rank as u64).map_err(wrap_err)?;
    }

    Ok(buf)
}

/// Build the per-item `result_msgpack` bytes for a `sparse` encode
/// outcome, mirroring the `sparse` branch of
/// [`_wrap_encode_output`][py-wrap] verbatim:
///
/// ```text
/// {
///   "sparse": {
///     "dims":    <u32> | nil,
///     "dtype":   "float32",
///     "indices": <numpy_sentinel for [n] i32>,
///     "values":  <numpy_sentinel for [n] f32>,
///   },
/// }
/// ```
///
/// Wire contract details:
///
/// * `dims` uses msgpack **nil** when `None` (matching Python's
///   `{"dims": None, ...}` which `msgpack.packb` emits as `0xc0`).
///   When present it's packed as the shortest-width uint (fixint for
///   small vocabs, `0xcd`/`0xce` for bigger — same as Python).
/// * `dtype` is always the literal `"float32"`. v1 does not cover
///   `float16` (the other case Python handles) to keep the wire
///   stable; models emitting float16 sparse still use the Python
///   path via the safety gate in `queue_executor`.
/// * `indices` dtype is always `"<i4"` (`int32`) because every
///   in-tree sparse adapter emits `np.int32` token ids (see e.g.
///   `splade_flash`, `gte_sparse_flash`, `sentence_transformer`).
///
/// [py-wrap]: ../../../../../../sie_server/src/sie_server/queue_executor.py
pub fn build_sparse_payload(
    indices: &[i32],
    values: &[f32],
    dims: Option<u32>,
) -> Result<Vec<u8>, ShapeError> {
    build_sparse_payload_with_item_id(indices, values, dims, None)
}

pub fn build_sparse_payload_with_item_id(
    indices: &[i32],
    values: &[f32],
    dims: Option<u32>,
    item_id: Option<&str>,
) -> Result<Vec<u8>, ShapeError> {
    if indices.len() != values.len() {
        return Err(ShapeError::SparseLenMismatch {
            indices: indices.len(),
            values: values.len(),
        });
    }
    let mut buf =
        Vec::with_capacity(72 + item_id.map(str::len).unwrap_or_default() + indices.len() * 8);

    write_encode_output_prefix(&mut buf, "sparse", item_id)?;

    write_map_len(&mut buf, 4).map_err(wrap_err)?;

    write_str(&mut buf, "dims").map_err(wrap_err)?;
    if let Some(d) = dims {
        write_uint(&mut buf, u64::from(d)).map_err(wrap_err)?;
    } else {
        write_nil(&mut buf).map_err(wrap_err)?;
    }

    write_str(&mut buf, "dtype").map_err(wrap_err)?;
    write_str(&mut buf, "float32").map_err(wrap_err)?;

    write_str(&mut buf, "indices").map_err(wrap_err)?;
    write_i32_vector(&mut buf, indices)?;

    write_str(&mut buf, "values").map_err(wrap_err)?;
    write_f32_vector(&mut buf, values)?;

    Ok(buf)
}

/// Build the per-item `result_msgpack` bytes for a `multivector`
/// encode outcome, mirroring the `multivector` branch of
/// [`_wrap_encode_output`][py-wrap]:
///
/// ```text
/// {
///   "multivector": {
///     "token_dims": <u32>,
///     "num_tokens": <u32>,
///     "dtype":      "float32" | "float16",
///     "values":     <numpy_sentinel for [num_tokens, token_dims] f32>,
///   },
/// }
/// ```
///
/// `float16` uses the same envelope and writes a msgpack-numpy
/// `"<f2"` sentinel. The bit-packed `binary` path (`shape[1] <
/// token_dims`, `dim/8` bytes per token) still stays on the Python
/// framing path.
///
/// `values.len() == num_tokens * token_dims` is required; the shaper
/// returns [`ShapeError::MultivectorShapeMismatch`] otherwise and
/// the publisher converts that into a typed error outcome rather
/// than publishing garbage.
///
/// [py-wrap]: ../../../../../../sie_server/src/sie_server/queue_executor.py
pub fn build_multivector_payload(
    values: &[f32],
    num_tokens: u32,
    token_dims: u32,
) -> Result<Vec<u8>, ShapeError> {
    build_multivector_payload_with_dtype(values, num_tokens, token_dims, "float32")
}

pub fn build_multivector_payload_with_dtype(
    values: &[f32],
    num_tokens: u32,
    token_dims: u32,
    dtype: &str,
) -> Result<Vec<u8>, ShapeError> {
    build_multivector_payload_with_dtype_and_item_id(values, num_tokens, token_dims, dtype, None)
}

pub fn build_multivector_payload_with_dtype_and_item_id(
    values: &[f32],
    num_tokens: u32,
    token_dims: u32,
    dtype: &str,
    item_id: Option<&str>,
) -> Result<Vec<u8>, ShapeError> {
    let expected = num_tokens as usize * token_dims as usize;
    if values.len() != expected {
        return Err(ShapeError::MultivectorShapeMismatch {
            values: values.len(),
            rows: num_tokens as usize,
            cols: token_dims as usize,
        });
    }

    let mut buf =
        Vec::with_capacity(72 + item_id.map(str::len).unwrap_or_default() + values.len() * 4);

    write_encode_output_prefix(&mut buf, "multivector", item_id)?;

    write_map_len(&mut buf, 4).map_err(wrap_err)?;

    write_str(&mut buf, "token_dims").map_err(wrap_err)?;
    write_uint(&mut buf, u64::from(token_dims)).map_err(wrap_err)?;

    write_str(&mut buf, "num_tokens").map_err(wrap_err)?;
    write_uint(&mut buf, u64::from(num_tokens)).map_err(wrap_err)?;

    write_str(&mut buf, "dtype").map_err(wrap_err)?;
    match dtype {
        "float32" => write_str(&mut buf, "float32").map_err(wrap_err)?,
        "float16" => write_str(&mut buf, "float16").map_err(wrap_err)?,
        other => {
            return Err(ShapeError::MsgpackWrite(format!(
                "unsupported multivector dtype {other:?}; expected float32 or float16"
            )));
        }
    }

    write_str(&mut buf, "values").map_err(wrap_err)?;
    match dtype {
        "float32" => write_f32_matrix_2d(&mut buf, values, num_tokens, token_dims)?,
        "float16" => write_f16_matrix_2d_from_f32(&mut buf, values, num_tokens, token_dims)?,
        _ => unreachable!("dtype validated above"),
    }

    Ok(buf)
}

pub fn build_multivector_payload_from_f16_bytes(
    values_f16: &[u8],
    num_tokens: u32,
    token_dims: u32,
) -> Result<Vec<u8>, ShapeError> {
    build_multivector_payload_from_f16_bytes_with_item_id(values_f16, num_tokens, token_dims, None)
}

pub fn build_multivector_payload_from_f16_bytes_with_item_id(
    values_f16: &[u8],
    num_tokens: u32,
    token_dims: u32,
    item_id: Option<&str>,
) -> Result<Vec<u8>, ShapeError> {
    let expected = num_tokens as usize * token_dims as usize * 2;
    if values_f16.len() != expected {
        return Err(ShapeError::MultivectorShapeMismatch {
            values: values_f16.len(),
            rows: num_tokens as usize,
            cols: token_dims as usize,
        });
    }

    let mut buf =
        Vec::with_capacity(72 + item_id.map(str::len).unwrap_or_default() + values_f16.len());

    write_encode_output_prefix(&mut buf, "multivector", item_id)?;

    write_map_len(&mut buf, 4).map_err(wrap_err)?;

    write_str(&mut buf, "token_dims").map_err(wrap_err)?;
    write_uint(&mut buf, u64::from(token_dims)).map_err(wrap_err)?;

    write_str(&mut buf, "num_tokens").map_err(wrap_err)?;
    write_uint(&mut buf, u64::from(num_tokens)).map_err(wrap_err)?;

    write_str(&mut buf, "dtype").map_err(wrap_err)?;
    write_str(&mut buf, "float16").map_err(wrap_err)?;

    write_str(&mut buf, "values").map_err(wrap_err)?;
    write_f16_matrix_2d_from_le_bytes(&mut buf, values_f16, num_tokens, token_dims)?;

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- l2_normalize_in_place ---------------------------------

    #[test]
    fn l2_normalize_unit_vector_stays_unit() {
        let mut v = vec![1.0_f32, 0.0, 0.0];
        l2_normalize_in_place(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn l2_normalize_scales_to_unit_length() {
        let mut v = vec![3.0_f32, 4.0];
        l2_normalize_in_place(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "got norm {norm}");
        // 3/5, 4/5
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_is_left_alone() {
        // Python does `where(norms > 0, norms, 1.0)` → zero vector
        // stays zero. Match that: no divide by zero, no NaN.
        let mut v = vec![0.0_f32; 5];
        l2_normalize_in_place(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn l2_normalize_empty_slice_is_noop() {
        let mut v: Vec<f32> = Vec::new();
        l2_normalize_in_place(&mut v);
        assert!(v.is_empty());
    }

    // ---- build_dense_payload -----------------------------------

    /// Golden: Python side runs
    ///   `msgpack.packb({"dense": {"dims": 3, "dtype": "float32",
    ///                             "values": np.array([1,2,3], dtype=f32)}},
    ///                  use_bin_type=True)`
    /// after `msgpack_numpy.patch()`. Captured from CPython (88 bytes).
    /// Same golden the `prep::outcome` tests use; duplicated here
    /// so the sidecar output-shaping path is validated independently.
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
            "dense payload must be byte-identical to _wrap_encode_output"
        );
    }

    #[test]
    fn build_dense_payload_rejects_dim_mismatch() {
        let err = build_dense_payload(&[1.0, 2.0], 3).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::DenseDimMismatch {
                expected: 3,
                actual: 2
            }
        ));
    }

    // ---- build_score_payload -----------------------------------

    /// Golden: Python `queue_executor._process_single_score` emits
    /// scores via `float(score_output.scores[i])` where
    /// `score_output.scores` is `np.float32`. That widens each
    /// score to `f64` **without** recovering full `f64` precision —
    /// `float(np.float32(0.9))` ≠ the Python literal `0.9`. The
    /// golden below reflects the real production path.
    ///
    /// Regenerate with:
    ///
    /// ```sh
    /// uv run --no-project --with msgpack --with numpy python -c "
    /// import msgpack, numpy as np
    /// scores = np.array([0.9, 0.5], dtype=np.float32)
    /// ids = ['a', 'b']
    /// scored = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    /// entries = [{'item_id': i, 'score': float(s), 'rank': r}
    ///            for r, (i, s) in enumerate(scored)]
    /// print(msgpack.packb(entries, use_bin_type=True).hex())"
    /// ```
    ///
    /// Captured: 65 bytes.
    const GOLDEN_SCORE_AB: &[u8] = &[
        0x92, 0x83, 0xa7, 0x69, 0x74, 0x65, 0x6d, 0x5f, 0x69, 0x64, 0xa1, 0x61, 0xa5, 0x73, 0x63,
        0x6f, 0x72, 0x65, 0xcb, 0x3f, 0xec, 0xcc, 0xcc, 0xc0, 0x00, 0x00, 0x00, 0xa4, 0x72, 0x61,
        0x6e, 0x6b, 0x00, 0x83, 0xa7, 0x69, 0x74, 0x65, 0x6d, 0x5f, 0x69, 0x64, 0xa1, 0x62, 0xa5,
        0x73, 0x63, 0x6f, 0x72, 0x65, 0xcb, 0x3f, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa4,
        0x72, 0x61, 0x6e, 0x6b, 0x01,
    ];

    #[test]
    fn build_score_payload_matches_python_wire_format_sorted_desc() {
        // Pass unsorted; the shaper must rank-sort desc like Python.
        // scores: a=0.9, b=0.5 → already desc on input, but exercises
        // the stable-sort path with a tied pair below.
        let scores = vec![0.9_f32, 0.5_f32];
        let ids = vec!["a".to_string(), "b".to_string()];
        let buf = build_score_payload(&scores, &ids).unwrap();
        assert_eq!(
            buf, GOLDEN_SCORE_AB,
            "score payload must be byte-identical to Python msgpack.packb"
        );
    }

    #[test]
    fn build_score_payload_reorders_and_reranks() {
        // Feed Python the same pair in reversed order — it still
        // emits a=rank0, b=rank1 after sorting desc.
        let scores = vec![0.5_f32, 0.9_f32];
        let ids = vec!["b".to_string(), "a".to_string()];
        let buf = build_score_payload(&scores, &ids).unwrap();
        assert_eq!(
            buf, GOLDEN_SCORE_AB,
            "score payload must sort desc before emission, regardless of input order"
        );
    }

    #[test]
    fn build_score_payload_stable_on_ties() {
        // Python's `list.sort(key=lambda x: x[1], reverse=True)` is
        // stable: equal-scored items keep their input order.
        // Assert the same here. Manually pack the expected output.
        let scores = vec![0.5_f32, 0.5_f32, 0.5_f32];
        let ids = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let buf = build_score_payload(&scores, &ids).unwrap();

        // Reconstruct expected Python bytes in-place — stable sort
        // means rank order is x=0, y=1, z=2.
        let mut expected = Vec::new();
        write_array_len(&mut expected, 3).unwrap();
        for (rank, id) in ["x", "y", "z"].iter().enumerate() {
            write_map_len(&mut expected, 3).unwrap();
            write_str(&mut expected, "item_id").unwrap();
            write_str(&mut expected, id).unwrap();
            write_str(&mut expected, "score").unwrap();
            write_f64(&mut expected, 0.5).unwrap();
            write_str(&mut expected, "rank").unwrap();
            write_uint(&mut expected, rank as u64).unwrap();
        }
        assert_eq!(buf, expected, "stable tie-breaking must match Python");
    }

    #[test]
    fn build_score_payload_rejects_length_mismatch() {
        let err = build_score_payload(&[1.0, 2.0], &["only-one".to_string()]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::ScoreLenMismatch {
                scores: 2,
                items: 1
            }
        ));
    }

    #[test]
    fn build_score_payload_empty_is_empty_array() {
        let buf = build_score_payload(&[], &[]).unwrap();
        // msgpack empty array is `0x90`.
        assert_eq!(buf, vec![0x90]);
    }

    // ---- build_sparse_payload ----------------------------------

    fn hex_to_bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Golden for `{"sparse": {"dims": None, "dtype": "float32",
    ///                         "indices": np.array([3, 7, 42], i32),
    ///                         "values":  np.array([0.5, 1.5, 2.5], f32)}}`.
    ///
    /// Regenerate via (keep in lockstep with the Python tests in
    /// `packages/sie_server/tests/test_stage1d_byte_identity.py`):
    ///
    /// ```sh
    /// uv run python -c "
    /// import msgpack, msgpack_numpy as m; m.patch()
    /// import numpy as np
    /// print(msgpack.packb({'sparse': {'dims': None, 'dtype': 'float32',
    ///     'indices': np.array([3,7,42], dtype=np.int32),
    ///     'values':  np.array([0.5,1.5,2.5], dtype=np.float32)}},
    ///     use_bin_type=True).hex())"
    /// ```
    const GOLDEN_SPARSE_NIL_DIM_HEX: &str = "81a673706172736584a464696d73c0a56474797065a7666c6f61743332a7696e646963657385c4026e64c3c40474797065a33c6934c4046b696e64c400c40573686170659103c40464617461c40c03000000070000002a000000a676616c75657385c4026e64c3c40474797065a33c6634c4046b696e64c400c40573686170659103c40464617461c40c0000003f0000c03f00002040";

    const GOLDEN_SPARSE_WITH_DIM_HEX: &str = "81a673706172736584a464696d73cd773aa56474797065a7666c6f61743332a7696e646963657385c4026e64c3c40474797065a33c6934c4046b696e64c400c40573686170659103c40464617461c40c03000000070000002a000000a676616c75657385c4026e64c3c40474797065a33c6634c4046b696e64c400c40573686170659103c40464617461c40c0000003f0000c03f00002040";

    const GOLDEN_SPARSE_EMPTY_HEX: &str = "81a673706172736584a464696d73cd03e8a56474797065a7666c6f61743332a7696e646963657385c4026e64c3c40474797065a33c6934c4046b696e64c400c40573686170659100c40464617461c400a676616c75657385c4026e64c3c40474797065a33c6634c4046b696e64c400c40573686170659100c40464617461c400";

    #[test]
    fn build_sparse_payload_nil_dim_matches_python() {
        let buf = build_sparse_payload(&[3, 7, 42], &[0.5, 1.5, 2.5], None).unwrap();
        assert_eq!(
            buf,
            hex_to_bytes(GOLDEN_SPARSE_NIL_DIM_HEX),
            "sparse (dims=None) bytes must match _wrap_encode_output output"
        );
    }

    #[test]
    fn build_sparse_payload_with_dim_matches_python() {
        // 30522 -> 0xcd 0x77 0x3a (msgpack uint16)
        let buf = build_sparse_payload(&[3, 7, 42], &[0.5, 1.5, 2.5], Some(30522)).unwrap();
        assert_eq!(
            buf,
            hex_to_bytes(GOLDEN_SPARSE_WITH_DIM_HEX),
            "sparse (dims=30522) bytes must match _wrap_encode_output output"
        );
    }

    #[test]
    fn build_sparse_payload_empty_matches_python() {
        let buf = build_sparse_payload(&[], &[], Some(1000)).unwrap();
        assert_eq!(
            buf,
            hex_to_bytes(GOLDEN_SPARSE_EMPTY_HEX),
            "empty sparse must match Python: shape=[0], data=empty"
        );
    }

    #[test]
    fn build_sparse_payload_rejects_length_mismatch() {
        let err = build_sparse_payload(&[1, 2], &[0.5], None).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::SparseLenMismatch {
                indices: 2,
                values: 1
            }
        ));
    }

    // ---- build_multivector_payload -----------------------------

    /// Golden: `{"multivector": {"token_dims": 4, "num_tokens": 3,
    ///                           "dtype": "float32",
    ///                           "values": np.arange(12, f32).reshape(3,4)}}`.
    const GOLDEN_MULTIVECTOR_3X4_HEX: &str = "81ab6d756c7469766563746f7284aa746f6b656e5f64696d7304aa6e756d5f746f6b656e7303a56474797065a7666c6f61743332a676616c75657385c4026e64c3c40474797065a33c6634c4046b696e64c400c4057368617065920304c40464617461c430000000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410000204100003041";

    #[test]
    fn build_multivector_payload_matches_python() {
        let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let buf = build_multivector_payload(&values, 3, 4).unwrap();
        assert_eq!(
            buf,
            hex_to_bytes(GOLDEN_MULTIVECTOR_3X4_HEX),
            "multivector [3, 4] f32 bytes must match _wrap_encode_output output"
        );
    }

    #[test]
    fn build_multivector_payload_supports_float16() {
        let values = vec![1.0, -2.0, 3.5, 0.0];
        let buf = build_multivector_payload_with_dtype(&values, 2, 2, "float16").unwrap();
        assert!(
            buf.windows("float16".len())
                .any(|window| window == b"float16"),
            "payload envelope must advertise float16"
        );
        assert!(
            buf.windows(DTYPE_F16.len())
                .any(|window| window == DTYPE_F16.as_bytes()),
            "numpy sentinel must use <f2 dtype"
        );
        assert!(
            buf.windows(4)
                .any(|window| window == [0x00, 0x3c, 0x00, 0xc0]),
            "payload data must contain little-endian f16 bytes for [1.0, -2.0]"
        );
    }

    #[test]
    fn build_multivector_payload_supports_preencoded_float16_bytes() {
        let values_f16 = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x43, 0x00, 0x00];
        let buf = build_multivector_payload_from_f16_bytes(&values_f16, 2, 2).unwrap();
        assert!(
            buf.windows("float16".len())
                .any(|window| window == b"float16"),
            "payload envelope must advertise float16"
        );
        assert!(
            buf.windows(DTYPE_F16.len())
                .any(|window| window == DTYPE_F16.as_bytes()),
            "numpy sentinel must use <f2 dtype"
        );
        assert!(
            buf.windows(values_f16.len())
                .any(|window| window == values_f16),
            "payload data must preserve supplied little-endian f16 bytes"
        );
    }

    #[test]
    fn build_multivector_payload_rejects_shape_mismatch() {
        // 3*4 = 12, feeding 10 values is wrong.
        let err = build_multivector_payload(&[0.0; 10], 3, 4).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::MultivectorShapeMismatch {
                values: 10,
                rows: 3,
                cols: 4,
            }
        ));
    }
}
