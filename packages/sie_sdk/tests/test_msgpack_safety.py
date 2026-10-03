from __future__ import annotations

from unittest.mock import patch

import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk._msgpack import UnsafeMessagePackError, packb, unpackb
from sie_sdk.jobs import decode_chunk_bytes


def _object_array_sentinel() -> dict[bytes, object]:
    return {
        b"nd": True,
        b"type": "|O",
        b"kind": b"O",
        b"shape": [1],
        b"data": b"attacker-controlled-pickle",
    }


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1, 2], dtype=np.int32),
        np.array([[1.0, 2.0]], dtype=np.float16),
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([1, 2], dtype=np.uint8),
        np.array([1, 2], dtype=np.uint16),
    ],
)
def test_numeric_msgpack_round_trip_preserves_array(array: np.ndarray) -> None:
    decoded = unpackb(packb({"value": array}), numeric_arrays=True)["value"]

    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == array.dtype
    assert decoded.shape == array.shape
    np.testing.assert_array_equal(decoded, array)


@pytest.mark.parametrize(
    "array",
    [
        np.array([object()], dtype=object),
        np.array([(1,)], dtype=[("value", "<i4")]),
    ],
)
def test_numeric_msgpack_encoder_rejects_object_and_structured_arrays(array: np.ndarray) -> None:
    with pytest.raises(UnsafeMessagePackError):
        packb({"value": array})


@pytest.mark.parametrize(
    "sentinel",
    [
        _object_array_sentinel(),
        {b"nd": True, b"type": "<i4", b"kind": b"V", b"shape": [1], b"data": b"\0\0\0\0"},
        {b"nd": True, b"type": "<f4", b"shape": [2], b"data": b"\0\0\0\0"},
        {b"nd": True, b"type": "<f4", b"shape": [True], b"data": b"\0\0\0\0"},
    ],
)
def test_numeric_msgpack_decoder_rejects_unsafe_or_malformed_arrays(
    sentinel: dict[bytes, object],
) -> None:
    raw = packb({"value": sentinel})

    with patch("msgpack_numpy.pickle.loads") as pickle_loads:
        with pytest.raises(UnsafeMessagePackError):
            unpackb(raw, numeric_arrays=True)

    pickle_loads.assert_not_called()


def test_job_chunk_decoders_never_unpickle_outer_or_inner_payloads() -> None:
    inner = packb({"dense": {"dims": 1, "values": _object_array_sentinel()}})
    outer = packb(
        [
            {"id": "inner", "success": True, "result_msgpack": inner},
            _object_array_sentinel(),
        ]
    )

    with patch("msgpack_numpy.pickle.loads") as pickle_loads:
        decoded = decode_chunk_bytes(outer)

    pickle_loads.assert_not_called()
    assert len(decoded) == 2
    assert decoded[0]["id"] == "inner"
    assert decoded[0]["dense"] is None


@pytest.mark.asyncio
async def test_client_construction_does_not_patch_process_global_msgpack() -> None:
    before = (msgpack.Packer, msgpack.Unpacker, msgpack.packb, msgpack.unpackb)

    sync_client = SIEClient("https://gateway.example.test")
    async_client = SIEAsyncClient("https://gateway.example.test")
    sync_client.close()
    await async_client.close()

    assert (msgpack.Packer, msgpack.Unpacker, msgpack.packb, msgpack.unpackb) == before
