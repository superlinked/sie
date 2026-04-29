"""Tests for wire format type guards in sie_server.types.inputs.

These tests verify the TypeGuard functions that replace isinstance() checks
for TypedDict types (not supported in Python 3.12+).
"""

from sie_server.types.inputs import (
    Item,
    is_audio_input,
    is_document_input,
    is_image_input,
    is_item,
    is_video_input,
)


class TestIsImageInput:
    """Tests for is_image_input type guard."""

    def test_valid_with_data_only(self) -> None:
        """Minimal valid ImageInput with just data."""
        assert is_image_input({"data": b"test"})

    def test_valid_with_data_and_format(self) -> None:
        """Full ImageInput with data and format."""
        assert is_image_input({"data": b"\x89PNG", "format": "png"})

    def test_valid_with_empty_bytes(self) -> None:
        """Empty bytes is still valid (validation happens elsewhere)."""
        assert is_image_input({"data": b""})

    def test_invalid_missing_data(self) -> None:
        """Missing data key should fail."""
        assert not is_image_input({})
        assert not is_image_input({"format": "png"})

    def test_invalid_data_wrong_type(self) -> None:
        """Data must be bytes, not string."""
        assert not is_image_input({"data": "string"})
        assert not is_image_input({"data": 123})
        assert not is_image_input({"data": None})

    def test_invalid_not_dict(self) -> None:
        """Must be a dict."""
        assert not is_image_input(None)
        assert not is_image_input("not a dict")
        assert not is_image_input(123)
        assert not is_image_input([])
        assert not is_image_input(b"bytes")


class TestIsAudioInput:
    """Tests for is_audio_input type guard."""

    def test_valid_with_data_only(self) -> None:
        """Minimal valid AudioInput with just data."""
        assert is_audio_input({"data": b"audio"})

    def test_valid_full(self) -> None:
        """Full AudioInput with all fields."""
        assert is_audio_input({"data": b"audio", "format": "wav", "sample_rate": 16000})

    def test_invalid_missing_data(self) -> None:
        """Missing data key should fail."""
        assert not is_audio_input({})
        assert not is_audio_input({"format": "wav"})

    def test_invalid_data_wrong_type(self) -> None:
        """Data must be bytes."""
        assert not is_audio_input({"data": "string"})

    def test_invalid_not_dict(self) -> None:
        """Must be a dict."""
        assert not is_audio_input(None)
        assert not is_audio_input("string")


class TestIsVideoInput:
    """Tests for is_video_input type guard."""

    def test_valid_with_data_only(self) -> None:
        """Minimal valid VideoInput with just data."""
        assert is_video_input({"data": b"video"})

    def test_valid_full(self) -> None:
        """Full VideoInput with all fields."""
        assert is_video_input({"data": b"video", "format": "mp4"})

    def test_invalid_missing_data(self) -> None:
        """Missing data key should fail."""
        assert not is_video_input({})

    def test_invalid_data_wrong_type(self) -> None:
        """Data must be bytes."""
        assert not is_video_input({"data": "string"})

    def test_invalid_not_dict(self) -> None:
        """Must be a dict."""
        assert not is_video_input(None)


class TestIsDocumentInput:
    """Tests for is_document_input type guard."""

    def test_valid_with_data_only(self) -> None:
        """Minimal valid DocumentInput with just data."""
        assert is_document_input({"data": b"%PDF-1.4"})

    def test_valid_full(self) -> None:
        """Full DocumentInput with format hint."""
        assert is_document_input({"data": b"%PDF-1.4", "format": "pdf"})

    def test_valid_with_empty_bytes(self) -> None:
        """Empty bytes is still valid (validation happens elsewhere)."""
        assert is_document_input({"data": b""})

    def test_invalid_missing_data(self) -> None:
        """Missing data key should fail."""
        assert not is_document_input({})
        assert not is_document_input({"format": "pdf"})

    def test_invalid_data_wrong_type(self) -> None:
        """Data must be bytes."""
        assert not is_document_input({"data": "string"})
        assert not is_document_input({"data": None})

    def test_invalid_not_dict(self) -> None:
        """Must be a dict."""
        assert not is_document_input(None)
        assert not is_document_input(b"raw bytes")


class TestIsItem:
    """Tests for is_item type guard."""

    def test_valid_empty_dict(self) -> None:
        """Empty dict is valid (all fields optional)."""
        assert is_item({})

    def test_valid_with_text(self) -> None:
        """Item with text."""
        assert is_item({"text": "hello"})

    def test_valid_with_images(self) -> None:
        """Item with images list."""
        assert is_item({"images": [{"data": b"img"}]})

    def test_valid_with_document(self) -> None:
        """Item with a document payload."""
        assert is_item({"document": {"data": b"%PDF-1.4", "format": "pdf"}})

    def test_valid_full(self) -> None:
        """Full Item with all fields."""
        assert is_item(
            {
                "id": "item1",
                "text": "hello",
                "images": [{"data": b"img"}],
                "document": {"data": b"%PDF-1.4", "format": "pdf"},
                "metadata": {"key": "value"},
            }
        )

    def test_valid_item_struct(self) -> None:
        """Item Struct instances are valid."""
        assert is_item(Item(text="hello"))
        assert is_item(Item())
        assert is_item(Item(document={"data": b"%PDF-1.4", "format": "pdf"}))

    def test_invalid_not_dict_or_struct(self) -> None:
        """Must be a dict or Item Struct."""
        assert not is_item(None)
        assert not is_item("string")
        assert not is_item([])
        assert not is_item(123)
