"""Numeric-only MessagePack helpers for SDK request and response bodies."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

import numpy as np

try:
    _native_msgpack = import_module("msgpack._cmsgpack")
except ImportError:  # pragma: no cover - msgpack wheels use the C extension.
    _native_msgpack = import_module("msgpack.fallback")

_NativePacker = _native_msgpack.Packer
_native_unpackb = _native_msgpack.unpackb


_NUMERIC_DTYPE_CODES = frozenset(
    {
        "<f2",
        "<f4",
        "<f8",
        "<i2",
        "<i4",
        "<i8",
        "<u2",
        "<u4",
        "<u8",
        ">f2",
        ">f4",
        ">f8",
        ">i2",
        ">i4",
        ">i8",
        ">u2",
        ">u4",
        ">u8",
        "|b1",
        "|i1",
        "|u1",
    }
)
_MAX_NUMPY_DIMENSIONS = 8


class UnsafeMessagePackError(ValueError):
    """A MessagePack NumPy sentinel is outside the SDK's numeric wire contract."""


def _numeric_dtype(value: object) -> np.dtype[Any]:
    if not isinstance(value, str) or value not in _NUMERIC_DTYPE_CODES:
        raise UnsafeMessagePackError("MessagePack NumPy dtype is not an allowed numeric dtype")
    dtype = np.dtype(value)
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise UnsafeMessagePackError("MessagePack NumPy dtype is not a flat numeric dtype")
    return dtype


def _numeric_default(value: object) -> Mapping[bytes, object]:
    if isinstance(value, np.ndarray):
        dtype = _numeric_dtype(value.dtype.str)
        contiguous = np.ascontiguousarray(value, dtype=dtype)
        return {
            b"nd": True,
            b"type": dtype.str,
            b"kind": b"",
            b"shape": contiguous.shape,
            b"data": contiguous.tobytes(),
        }
    if isinstance(value, np.generic):
        dtype = _numeric_dtype(value.dtype.str)
        return {
            b"nd": False,
            b"type": dtype.str,
            b"data": value.tobytes(),
        }
    raise TypeError(f"cannot MessagePack-encode {type(value).__name__}")


def packb(value: object, *, use_bin_type: bool = True) -> bytes:
    """Pack one SDK body without installing process-global NumPy hooks."""
    return _NativePacker(default=_numeric_default, use_bin_type=use_bin_type).pack(value)


def _numeric_object_hook(value: dict[object, object]) -> object:
    if b"nd" not in value:
        if b"complex" in value:
            raise UnsafeMessagePackError("complex MessagePack NumPy values are not supported")
        return value

    is_array = value.get(b"nd")
    if not isinstance(is_array, bool):
        raise UnsafeMessagePackError("MessagePack NumPy nd marker must be boolean")
    if value.get(b"kind", b"") not in {b"", ""}:
        # In particular, kind=O invokes pickle.loads in msgpack-numpy and
        # kind=V admits attacker-selected structured dtype descriptors.
        raise UnsafeMessagePackError("object and structured MessagePack NumPy values are not supported")

    dtype = _numeric_dtype(value.get(b"type"))
    data = value.get(b"data")
    if not isinstance(data, bytes):
        raise UnsafeMessagePackError("MessagePack NumPy data must be bytes")

    if not is_array:
        if len(data) != dtype.itemsize:
            raise UnsafeMessagePackError("MessagePack NumPy scalar byte length does not match its dtype")
        return np.frombuffer(data, dtype=dtype, count=1)[0]

    raw_shape = value.get(b"shape")
    if not isinstance(raw_shape, list) or len(raw_shape) > _MAX_NUMPY_DIMENSIONS:
        raise UnsafeMessagePackError("MessagePack NumPy array shape is malformed")
    shape: list[int] = []
    elements = 1
    for raw_dimension in raw_shape:
        if not isinstance(raw_dimension, int) or isinstance(raw_dimension, bool) or raw_dimension < 0:
            raise UnsafeMessagePackError("MessagePack NumPy array dimension is malformed")
        dimension = int(raw_dimension)
        elements *= dimension
        shape.append(dimension)
    if elements * dtype.itemsize != len(data):
        raise UnsafeMessagePackError("MessagePack NumPy array byte length does not match its shape")
    return np.frombuffer(data, dtype=dtype).reshape(tuple(shape))


def unpackb(value: bytes, *, numeric_arrays: bool) -> Any:
    """Unpack one untrusted body with optional numeric-only NumPy decoding."""
    kwargs: dict[str, Any] = {"raw": False}
    if numeric_arrays:
        kwargs["object_hook"] = _numeric_object_hook
    return _native_unpackb(value, **kwargs)
