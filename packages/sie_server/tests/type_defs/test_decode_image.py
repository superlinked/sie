"""Tests for the shared image decoder ``decode_image``.

``decode_image`` is the single seam every user-input ``PIL.Image.open`` routes
through. Valid base64 that is NOT decodable image bytes used to raise PIL's
``UnidentifiedImageError`` (an ``OSError`` subclass), which missed the
``ValueError`` -> 400 INVALID_INPUT mapping and surfaced as a 500
``INFERENCE_ERROR`` leaking a ``BytesIO`` repr. The helper must raise a typed
``InvalidMediaError`` whose message names the offending field in the same
JSON-path style msgspec uses for corrupt base64
("Invalid base64 encoded string - at `$.items[0].images[0].data`").
"""

import io

import pytest
from PIL import Image as PILImage
from sie_server.types.inputs import InvalidMediaError, decode_image

_NOT_AN_IMAGE = b"valid base64 decoded to these bytes, but they are not an image"


def _png_bytes(mode: str, size: tuple[int, int] = (7, 5)) -> bytes:
    buf = io.BytesIO()
    PILImage.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


class TestDecodeImageAccepts:
    """Decodable payloads come back as RGB PIL images."""

    def test_rgb_passthrough_preserves_size(self) -> None:
        img = decode_image({"data": _png_bytes("RGB", (7, 5))})
        assert img.mode == "RGB"
        assert img.size == (7, 5)

    def test_converts_grayscale_to_rgb(self) -> None:
        img = decode_image({"data": _png_bytes("L", (4, 9))})
        assert img.mode == "RGB"
        assert img.size == (4, 9)

    def test_converts_rgba_to_rgb(self) -> None:
        img = decode_image({"data": _png_bytes("RGBA", (3, 3))})
        assert img.mode == "RGB"


class TestDecodeImageRejectsUndecodable:
    """Non-image bytes raise InvalidMediaError (-> 400), never an OSError 500."""

    def test_non_image_bytes_raise_invalid_media(self) -> None:
        with pytest.raises(InvalidMediaError, match="image data is not a decodable image"):
            decode_image({"data": _NOT_AN_IMAGE})

    def test_message_has_wildcard_json_path_without_indices(self) -> None:
        with pytest.raises(InvalidMediaError, match=r"at `\$\.items\[\*\]\.images\[\*\]\.data`"):
            decode_image({"data": _NOT_AN_IMAGE})

    def test_message_names_item_and_image_index_when_known(self) -> None:
        with pytest.raises(InvalidMediaError, match=r"at `\$\.items\[2\]\.images\[1\]\.data`"):
            decode_image({"data": _NOT_AN_IMAGE}, item_index=2, image_index=1)

    def test_message_never_leaks_a_bytesio_repr(self) -> None:
        with pytest.raises(InvalidMediaError) as excinfo:
            decode_image({"data": _NOT_AN_IMAGE}, item_index=0, image_index=0)
        assert "BytesIO" not in str(excinfo.value)

    def test_truncated_image_bytes_raise_invalid_media(self) -> None:
        # A valid header with the tail cut off decodes lazily in PIL; the
        # eager load() must surface it as the same typed error.
        truncated = _png_bytes("RGB", (32, 32))[:40]
        with pytest.raises(InvalidMediaError, match="not a decodable image"):
            decode_image({"data": truncated})

    def test_error_is_a_value_error(self) -> None:
        """ValueError subclassing is what routes it to HTTP 400 INVALID_INPUT."""
        assert issubclass(InvalidMediaError, ValueError)


class TestDecodeImageRejectsNonBytes:
    """The media_bytes contract still applies before any PIL decode."""

    def test_str_data_is_rejected(self) -> None:
        with pytest.raises(InvalidMediaError, match="image data must be bytes, got str"):
            decode_image({"data": "aGVsbG8="})

    def test_missing_data_is_rejected(self) -> None:
        with pytest.raises(InvalidMediaError):
            decode_image({})
