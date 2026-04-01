"""Tests for image conversion utilities."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sie_sdk.images import convert_item_images, to_jpeg_bytes


class TestToJpegBytes:
    """Tests for to_jpeg_bytes function."""

    def test_pil_image_rgb(self) -> None:
        """Convert RGB PIL image to JPEG bytes."""
        img = Image.new("RGB", (100, 100), color="red")
        result = to_jpeg_bytes(img)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # Verify it's valid JPEG (starts with FFD8)
        assert result[:2] == b"\xff\xd8"

    def test_pil_image_rgba(self) -> None:
        """Convert RGBA PIL image to JPEG bytes (alpha removed)."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        result = to_jpeg_bytes(img)

        assert isinstance(result, bytes)
        # Verify it's valid JPEG
        assert result[:2] == b"\xff\xd8"

    def test_pil_image_grayscale(self) -> None:
        """Convert grayscale PIL image to JPEG bytes."""
        img = Image.new("L", (100, 100), color=128)
        result = to_jpeg_bytes(img)

        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"

    def test_numpy_array_rgb(self) -> None:
        """Convert RGB numpy array to JPEG bytes."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red channel
        result = to_jpeg_bytes(arr)

        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"

    def test_numpy_array_grayscale(self) -> None:
        """Convert grayscale numpy array to JPEG bytes."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[50:, :] = 255
        result = to_jpeg_bytes(arr)

        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"

    def test_bytes_passthrough(self) -> None:
        """Bytes pass through unchanged."""
        # Create valid JPEG bytes
        img = Image.new("RGB", (10, 10), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        jpeg_bytes = buffer.getvalue()

        result = to_jpeg_bytes(jpeg_bytes)
        assert result == jpeg_bytes

    def test_file_path_string(self) -> None:
        """Load image from file path string."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (50, 50), color="green")
            img.save(f.name)

            result = to_jpeg_bytes(f.name)
            assert isinstance(result, bytes)
            assert result[:2] == b"\xff\xd8"

            # Cleanup
            Path(f.name).unlink()

    def test_file_path_object(self) -> None:
        """Load image from Path object."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (50, 50), color="blue")
            img.save(f.name)
            path = Path(f.name)

            result = to_jpeg_bytes(path)
            assert isinstance(result, bytes)
            assert result[:2] == b"\xff\xd8"

            # Cleanup
            path.unlink()

    def test_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            to_jpeg_bytes("/nonexistent/path/image.jpg")

    def test_unsupported_type(self) -> None:
        """Raise ValueError for unsupported types."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            to_jpeg_bytes(123)  # type: ignore[arg-type]

    def test_invalid_array_shape(self) -> None:
        """Raise ValueError for invalid array dimensions."""
        arr = np.zeros((10, 10, 10, 3), dtype=np.uint8)  # 4D array
        with pytest.raises(ValueError, match=r"Expected 2D .* or 3D"):
            to_jpeg_bytes(arr)

    def test_custom_quality(self) -> None:
        """Test custom JPEG quality setting."""
        img = Image.new("RGB", (100, 100), color="red")

        # Low quality should produce smaller file
        low_q = to_jpeg_bytes(img, quality=10)
        high_q = to_jpeg_bytes(img, quality=95)

        # Low quality should generally be smaller (not guaranteed for solid colors)
        assert isinstance(low_q, bytes)
        assert isinstance(high_q, bytes)


class TestConvertItemImages:
    """Tests for convert_item_images function."""

    def test_no_images(self) -> None:
        """Items without images pass through unchanged."""
        item = {"text": "hello world"}
        result = convert_item_images(item)
        assert result == {"text": "hello world"}

    def test_empty_images(self) -> None:
        """Empty images list passes through."""
        item = {"text": "hello", "images": []}
        result = convert_item_images(item)
        assert result == {"text": "hello", "images": []}

    def test_pil_images(self) -> None:
        """Convert PIL images in item to ImageInput wire format."""
        img1 = Image.new("RGB", (10, 10), color="red")
        img2 = Image.new("RGB", (10, 10), color="blue")
        item = {"text": "test", "images": [img1, img2]}

        result = convert_item_images(item)

        assert len(result["images"]) == 2
        # Should be list of dicts with "data" and "format" keys
        for img_input in result["images"]:
            assert isinstance(img_input, dict)
            assert "data" in img_input
            assert "format" in img_input
            assert img_input["format"] == "jpeg"
            assert isinstance(img_input["data"], bytes)
            assert img_input["data"][:2] == b"\xff\xd8"

    def test_image_input_dict(self) -> None:
        """Handle ImageInput dict format."""
        img = Image.new("RGB", (10, 10), color="green")
        item = {"images": [{"data": img, "format": "jpeg"}]}

        result = convert_item_images(item)

        assert len(result["images"]) == 1
        assert isinstance(result["images"][0], dict)
        assert isinstance(result["images"][0]["data"], bytes)
        assert result["images"][0]["format"] == "jpeg"

    def test_mixed_formats(self) -> None:
        """Handle mix of PIL images and ImageInput dicts."""
        img1 = Image.new("RGB", (10, 10), color="red")
        img2 = Image.new("RGB", (10, 10), color="blue")
        item = {"images": [img1, {"data": img2}]}

        result = convert_item_images(item)

        assert len(result["images"]) == 2
        for img_input in result["images"]:
            assert isinstance(img_input, dict)
            assert isinstance(img_input["data"], bytes)
