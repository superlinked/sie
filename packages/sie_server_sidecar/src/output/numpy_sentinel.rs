//! Build msgpack bytes that match `msgpack_numpy`'s on-wire format.
//!
//! The Python queue executor serializes a successful encode outcome
//! with `msgpack.packb(output, use_bin_type=True)` after
//! `msgpack_numpy.patch()` has installed a `default` encoder hook.
//! That hook rewrites any `numpy.ndarray` into a 5-key map — the
//! so-called **numpy sentinel**:
//!
//! ```text
//! {
//!   b"nd":    true,
//!   b"type":  "<f4",          // ndarray.dtype.str
//!   b"kind":  b"",            // ndarray.dtype.kind for structured dtypes
//!   b"shape": [d0, d1, ...],  // ndarray.shape
//!   b"data":  <raw bytes>,    // ndarray.tobytes() (little-endian, C-order)
//! }
//! ```
//!
//! All five keys are packed as **bin8** (not str) because
//! `msgpack_numpy.encode` uses Python `bytes` literals. The SDK's
//! decoder relies on this exact shape (see `_shared.py::parse_encode_results`
//! and gateway `convert_numpy_for_json::is_numpy_sentinel`), so we have
//! to match it byte-for-byte.
//!
//! Keeping this in one place also makes it trivial to unit-test the
//! output against a golden blob captured from Python.

use std::io::Write;

use rmp::encode::{write_array_len, write_bin, write_bool, write_map_len, write_str, write_uint};

/// Internal error type for sentinel writers. All writes go to a
/// `Vec<u8>`, so I/O can't fail in practice — the error exists so
/// callers can propagate without unwrapping.
#[derive(Debug, thiserror::Error)]
pub enum SentinelError {
    #[error("msgpack write: {0}")]
    MsgpackWrite(String),
    #[error("invalid matrix shape: rows*cols ({expected}) != data.len() ({actual})")]
    InvalidMatrixShape { expected: usize, actual: usize },
}

/// Small helper so the writers below stay readable: stringify any
/// error we get from `rmp::encode` and wrap it as `SentinelError`.
fn wrap_err<E: std::fmt::Display>(e: E) -> SentinelError {
    SentinelError::MsgpackWrite(e.to_string())
}

/// msgpack-numpy dtype string for a contiguous little-endian float32
/// array. This is `numpy.dtype("float32").str` on a little-endian
/// host, which is what the SDK expects.
pub const DTYPE_F32: &str = "<f4";

/// msgpack-numpy dtype string for a contiguous little-endian float16
/// array. This is `numpy.dtype("float16").str` on a little-endian
/// host.
pub const DTYPE_F16: &str = "<f2";

/// msgpack-numpy dtype string for int32 little-endian, used for
/// sparse indices.
pub const DTYPE_I32: &str = "<i4";

/// Write the 5-key numpy sentinel map into `buf`, with pre-encoded
/// raw bytes and explicit shape. The caller supplies:
///
/// * `dtype_str` — the `"<f4"`, `"<i4"`, etc. code. Must match the
///   byte layout of `data`.
/// * `shape`     — dimensions in C order (`[rows, cols]` for 2D).
/// * `data`      — raw little-endian bytes, `prod(shape) * itemsize`.
pub fn write_numpy_sentinel(
    buf: &mut Vec<u8>,
    dtype_str: &str,
    shape: &[u32],
    data: &[u8],
) -> Result<(), SentinelError> {
    // 5-key map: nd, type, kind, shape, data.
    write_map_len(buf, 5).map_err(wrap_err)?;

    write_bin(buf, b"nd").map_err(wrap_err)?;
    write_bool(buf, true).map_err(wrap_err)?;

    write_bin(buf, b"type").map_err(wrap_err)?;
    write_str(buf, dtype_str).map_err(wrap_err)?;

    write_bin(buf, b"kind").map_err(wrap_err)?;
    write_bin(buf, b"").map_err(wrap_err)?;

    write_bin(buf, b"shape").map_err(wrap_err)?;
    write_array_len(buf, shape.len() as u32).map_err(wrap_err)?;
    for dim in shape {
        // Shapes are always non-negative. Python packs these as the
        // smallest positive int — msgpack-numpy does the equivalent
        // of `msgpack.packb(int)`, which for `3` is `0x03` (fixint).
        // `write_uint` picks the shortest encoding (fixint for
        // values ≤127) to stay byte-identical with Python.
        write_uint(buf, u64::from(*dim)).map_err(wrap_err)?;
    }

    write_bin(buf, b"data").map_err(wrap_err)?;
    write_bin(buf, data).map_err(wrap_err)?;

    Ok(())
}

/// Convenience: write a 1-D float32 array as a numpy sentinel.
pub fn write_f32_vector(buf: &mut Vec<u8>, vec: &[f32]) -> Result<(), SentinelError> {
    // Flatten f32 slice into little-endian bytes. The host is
    // little-endian for every platform we ship on, so a bytemuck
    // cast would be zero-copy — but we keep it explicit and portable.
    let mut data = Vec::with_capacity(vec.len() * 4);
    for &f in vec {
        data.write_all(&f.to_le_bytes()).map_err(wrap_err)?;
    }
    write_numpy_sentinel(buf, DTYPE_F32, &[vec.len() as u32], &data)
}

/// Convenience: write a 1-D int32 array as a numpy sentinel.
pub fn write_i32_vector(buf: &mut Vec<u8>, vec: &[i32]) -> Result<(), SentinelError> {
    let mut data = Vec::with_capacity(vec.len() * 4);
    for &i in vec {
        data.write_all(&i.to_le_bytes()).map_err(wrap_err)?;
    }
    write_numpy_sentinel(buf, DTYPE_I32, &[vec.len() as u32], &data)
}

/// Convenience: write a 2-D float32 matrix `[rows, cols]` as a numpy
/// sentinel. `data` must be flattened in C (row-major) order —
/// exactly what `np.ndarray.tobytes()` emits for a contiguous array.
///
pub fn write_f32_matrix_2d(
    buf: &mut Vec<u8>,
    data: &[f32],
    rows: u32,
    cols: u32,
) -> Result<(), SentinelError> {
    let expected = (rows as usize).saturating_mul(cols as usize);
    if data.len() != expected {
        return Err(SentinelError::InvalidMatrixShape {
            expected,
            actual: data.len(),
        });
    }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &f in data {
        bytes.write_all(&f.to_le_bytes()).map_err(wrap_err)?;
    }
    write_numpy_sentinel(buf, DTYPE_F32, &[rows, cols], &bytes)
}

/// Convenience: write a 2-D float16 matrix `[rows, cols]` as a numpy
/// sentinel, rounding the supplied f32 values to IEEE 754 half
/// precision.
pub fn write_f16_matrix_2d_from_f32(
    buf: &mut Vec<u8>,
    data: &[f32],
    rows: u32,
    cols: u32,
) -> Result<(), SentinelError> {
    let expected = (rows as usize).saturating_mul(cols as usize);
    if data.len() != expected {
        return Err(SentinelError::InvalidMatrixShape {
            expected,
            actual: data.len(),
        });
    }
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &f in data {
        bytes
            .write_all(&f32_to_f16_bits(f).to_le_bytes())
            .map_err(wrap_err)?;
    }
    write_numpy_sentinel(buf, DTYPE_F16, &[rows, cols], &bytes)
}

/// Convenience: write a 2-D float16 matrix `[rows, cols]` from
/// already-rounded little-endian f16 bytes.
pub fn write_f16_matrix_2d_from_le_bytes(
    buf: &mut Vec<u8>,
    data: &[u8],
    rows: u32,
    cols: u32,
) -> Result<(), SentinelError> {
    let expected = (rows as usize)
        .saturating_mul(cols as usize)
        .saturating_mul(2);
    if data.len() != expected {
        return Err(SentinelError::InvalidMatrixShape {
            expected,
            actual: data.len(),
        });
    }
    write_numpy_sentinel(buf, DTYPE_F16, &[rows, cols], data)
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        let payload = (mantissa >> 13).max(1) as u16;
        return sign | 0x7c00 | payload;
    }

    let half_exp = exponent - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mantissa = (mantissa >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let remainder = mantissa & (round_bit - 1);
        if (mantissa & round_bit) != 0 && (remainder != 0 || (half_mantissa & 1) != 0) {
            half_mantissa = half_mantissa.wrapping_add(1);
        }
        return sign | half_mantissa;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mantissa >> 13) as u16);
    let round_bit = 0x0000_1000;
    let remainder = mantissa & (round_bit - 1);
    if (mantissa & round_bit) != 0 && (remainder != 0 || (half & 1) != 0) {
        half = half.wrapping_add(1);
    }
    half
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden: `msgpack.packb(np.array([1.0, 2.0, 3.0], dtype=np.float32))`
    /// after `msgpack_numpy.patch()`. Captured from CPython
    /// (see docs: 53 bytes).
    const GOLDEN_F32_123: &[u8] = &[
        0x85, 0xc4, 0x02, 0x6e, 0x64, 0xc3, 0xc4, 0x04, 0x74, 0x79, 0x70, 0x65, 0xa3, 0x3c, 0x66,
        0x34, 0xc4, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0xc4, 0x00, 0xc4, 0x05, 0x73, 0x68, 0x61, 0x70,
        0x65, 0x91, 0x03, 0xc4, 0x04, 0x64, 0x61, 0x74, 0x61, 0xc4, 0x0c, 0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
    ];

    /// Golden: `msgpack.packb(np.array([5, 17, 99], dtype=np.int32))`.
    const GOLDEN_I32: &[u8] = &[
        0x85, 0xc4, 0x02, 0x6e, 0x64, 0xc3, 0xc4, 0x04, 0x74, 0x79, 0x70, 0x65, 0xa3, 0x3c, 0x69,
        0x34, 0xc4, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0xc4, 0x00, 0xc4, 0x05, 0x73, 0x68, 0x61, 0x70,
        0x65, 0x91, 0x03, 0xc4, 0x04, 0x64, 0x61, 0x74, 0x61, 0xc4, 0x0c, 0x05, 0x00, 0x00, 0x00,
        0x11, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn f32_vector_matches_python_golden() {
        let mut buf = Vec::new();
        write_f32_vector(&mut buf, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            buf, GOLDEN_F32_123,
            "f32 vector sentinel doesn't match msgpack-numpy wire format"
        );
    }

    #[test]
    fn i32_vector_matches_python_golden() {
        let mut buf = Vec::new();
        write_i32_vector(&mut buf, &[5, 17, 99]).unwrap();
        assert_eq!(
            buf, GOLDEN_I32,
            "i32 vector sentinel doesn't match msgpack-numpy wire format"
        );
    }

    #[test]
    fn empty_f32_vector_roundtrips() {
        let mut buf = Vec::new();
        write_f32_vector(&mut buf, &[]).unwrap();
        // An empty vector should still be a 5-key map, shape [0], data zero bytes.
        assert_eq!(&buf[0..1], &[0x85], "should be map-of-5");
        // The last 3 bytes: c4 00 (empty bin data)
        assert_eq!(&buf[buf.len() - 2..], &[0xc4, 0x00]);
    }

    #[test]
    fn f32_matrix_rejects_shape_mismatch() {
        let mut buf = Vec::new();
        let err = write_f32_matrix_2d(&mut buf, &[1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert!(matches!(
            err,
            SentinelError::InvalidMatrixShape {
                expected: 4,
                actual: 3
            }
        ));
        assert!(buf.is_empty());
    }

    #[test]
    fn f16_matrix_uses_float16_dtype_and_half_bytes() {
        let mut buf = Vec::new();
        write_f16_matrix_2d_from_f32(&mut buf, &[1.0, -2.0], 1, 2).unwrap();
        assert!(
            buf.windows(DTYPE_F16.len())
                .any(|window| window == DTYPE_F16.as_bytes()),
            "sentinel must carry <f2 dtype"
        );
        assert!(
            buf.windows(4)
                .any(|window| window == [0x00, 0x3c, 0x00, 0xc0]),
            "sentinel data must contain little-endian f16 bytes for [1.0, -2.0]"
        );
    }

    #[test]
    fn f16_conversion_uses_round_to_nearest_even() {
        assert_eq!(f32_to_f16_bits(1.000_488_3), 0x3c00);
        assert_eq!(f32_to_f16_bits(1.001_464_8), 0x3c02);
    }
}
