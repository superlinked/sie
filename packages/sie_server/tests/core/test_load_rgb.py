"""Tests for the shared vision RGB image loader ``_load_rgb`` (issue #1540).

Pins the decode + RGB-convert idiom that the vision preprocessors used to
repeat: RGB passthrough preserves size, non-RGB modes are converted, and a
malformed media payload surfaces as ``InvalidMediaError`` (-> 400).
"""

from __future__ import annotations

import io

import pytest
from PIL import Image as PILImage
from sie_server.core.preprocessor.vision import _load_rgb
from sie_server.types.inputs import InvalidMediaError


def _png_bytes(mode: str, size: tuple[int, int] = (7, 5)) -> bytes:
    buf = io.BytesIO()
    PILImage.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


def test_rgb_passthrough_preserves_size() -> None:
    img = _load_rgb({"data": _png_bytes("RGB", (7, 5))})
    assert img.mode == "RGB"
    assert img.size == (7, 5)


def test_converts_grayscale_to_rgb() -> None:
    img = _load_rgb({"data": _png_bytes("L", (4, 9))})
    assert img.mode == "RGB"
    assert img.size == (4, 9)  # convert("RGB") preserves dimensions


def test_converts_rgba_to_rgb() -> None:
    img = _load_rgb({"data": _png_bytes("RGBA", (3, 3))})
    assert img.mode == "RGB"


def test_non_bytes_payload_raises_invalid_media() -> None:
    with pytest.raises(InvalidMediaError):
        _load_rgb({"data": "not-bytes"})


def test_missing_data_raises_invalid_media() -> None:
    with pytest.raises(InvalidMediaError):
        _load_rgb({})


def test_undecodable_bytes_raise_invalid_media() -> None:
    """Valid base64-decoded bytes that are not an image are a typed 400, not a 500."""
    with pytest.raises(InvalidMediaError, match=r"not a decodable image - at `\$\.items\[\*\]\.images\[0\]\.data`"):
        _load_rgb({"data": b"not an image at all"})


def test_undecodable_bytes_name_the_item_when_index_is_known() -> None:
    with pytest.raises(InvalidMediaError, match=r"at `\$\.items\[3\]\.images\[0\]\.data`"):
        _load_rgb({"data": b"not an image at all"}, item_index=3)
