"""Image conversion utilities for SIE SDK.

PIL/array/path inputs are serialized as JPEG bytes for transport. Already
encoded bytes remain byte-identical and carry their detected format.

MessagePack primitives carry raw image bytes plus a truthful format token.
Already encoded JPEG, PNG, GIF, WebP, BMP, and TIFF bytes remain native;
unknown signatures (including currently unsupported HEIC/AVIF) fail closed.
Native JSON generation base64-encodes the same bytes at its final wire step.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type alias for all supported image input formats
ImageLike = Union[Image.Image, "NDArray[Any]", bytes, str, Path]

# Default JPEG quality for image transport.
DEFAULT_JPEG_QUALITY = 95
_MAX_IMAGE_FORMAT_LENGTH = 32


def _canonical_image_format(value: object) -> str:
    if not isinstance(value, str) or not (
        1 <= len(value) <= _MAX_IMAGE_FORMAT_LENGTH
        and all(character.isascii() and (character.isalnum() or character in ".+-") for character in value)
    ):
        msg = "Image format must be a short ASCII media-format token"
        raise ValueError(msg)
    normalized = value.lower()
    return "jpeg" if normalized in {"jpg", "jpe"} else normalized


def _detect_encoded_image_format(data: bytes) -> str:
    """Detect the transport format of already-encoded image bytes.

    This deliberately inspects signatures rather than decoding pixels: SDK
    serialization must stay cheap, while the server-side media loader remains
    responsible for full image validation. Unknown bytes fail closed instead
    of being silently labeled as JPEG.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "tiff"
    msg = "Could not detect encoded image format from bytes"
    raise ValueError(msg)


def _image_wire_value(image: ImageLike, declared_format: object = None) -> dict[str, Any]:
    if isinstance(image, bytes):
        data = image
        detected_format = _detect_encoded_image_format(data)
    else:
        data = to_jpeg_bytes(image)
        detected_format = "jpeg"

    if declared_format is not None:
        normalized_format = _canonical_image_format(declared_format)
        if normalized_format != detected_format:
            msg = f"Image format mismatch: declared '{normalized_format}', detected '{detected_format}'"
            raise ValueError(msg)
    return {"data": data, "format": detected_format}


def to_jpeg_bytes(
    image: ImageLike,
    *,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Convert various image formats to JPEG bytes for wire transport.

    Supports:
    - PIL.Image → JPEG bytes
    - np.ndarray (H,W,C) → PIL → JPEG bytes
    - bytes (JPEG/PNG) → pass through as-is
    - str/Path → read file → JPEG bytes

    Args:
        image: Image in any supported format.
        quality: JPEG quality (1-100). Default: 95.

    Returns:
        JPEG bytes ready for msgpack transport.

    Raises:
        ValueError: If image format is not supported.
        FileNotFoundError: If image path doesn't exist.
    """
    # Already bytes - assume JPEG/PNG, pass through
    if isinstance(image, bytes):
        return image

    # Path or string - read file
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            msg = f"Image file not found: {path}"
            raise FileNotFoundError(msg)

        # Read and convert to ensure consistent JPEG output
        pil_image = Image.open(path)
        return _pil_to_jpeg_bytes(pil_image, quality=quality)

    # PIL Image
    if isinstance(image, Image.Image):
        return _pil_to_jpeg_bytes(image, quality=quality)

    # NumPy array - convert to PIL first
    # Check for numpy array using duck typing to avoid import
    if hasattr(image, "shape") and hasattr(image, "dtype"):
        import numpy as np

        if not isinstance(image, np.ndarray):
            msg = f"Unsupported image type: {type(image)}"
            raise ValueError(msg)

        # Validate array shape
        if image.ndim not in (2, 3):
            msg = f"Expected 2D (grayscale) or 3D (H,W,C) array, got {image.ndim}D"
            raise ValueError(msg)

        # Convert to PIL
        pil_image = Image.fromarray(image)
        return _pil_to_jpeg_bytes(pil_image, quality=quality)

    msg = f"Unsupported image type: {type(image)}. Expected PIL.Image, np.ndarray, bytes, str, or Path."
    raise ValueError(msg)


def _pil_to_jpeg_bytes(image: Image.Image, *, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Convert PIL Image to JPEG bytes.

    Args:
        image: PIL Image object.
        quality: JPEG quality (1-100).

    Returns:
        JPEG bytes.
    """
    # Convert to RGB if necessary (JPEG doesn't support alpha)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    elif image.mode == "L":
        # Grayscale is fine for JPEG
        pass
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Encode to JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def convert_item_images(item: dict[str, Any]) -> dict[str, Any]:
    """Convert all images in an item to wire format for transport.

    Images are wrapped in ImageInput format: ``{"data": <bytes>,
    "format": <detected transport format>}``. PIL/array/path inputs are
    converted to JPEG; encoded bytes are preserved and signature-detected.

    Modifies the item in-place and returns it.

    Args:
        item: Item dict that may contain an 'images' field.

    Returns:
        The same item dict with images converted to ImageInput wire format.
    """
    if "images" not in item:
        return item

    images = item["images"]
    if not images:
        return item

    converted: list[dict[str, Any]] = []
    for img in images:
        # Handle ImageInput dict format (SDK user provided dict with "data" key)
        if isinstance(img, dict) and "data" in img:
            converted.append(_image_wire_value(img["data"], img.get("format")))
        else:
            # Direct image input (PIL.Image, ndarray, bytes, str/Path)
            converted.append(_image_wire_value(img))

    item["images"] = converted
    return item


def convert_images_for_json(images: Sequence[Any]) -> list[dict[str, str]]:
    """Convert native image inputs to the JSON generate wire envelope.

    Native tensor primitives carry image bytes directly in MessagePack. The
    native generate endpoint is JSON, so it uses the same ``{data, format}``
    envelope with standard-base64 text in ``data``. Conversion is centralized
    here so sync, async, buffered, and streaming clients cannot drift.
    """
    converted = convert_item_images({"images": list(images)}).get("images", [])
    return [
        {
            "data": base64.b64encode(image["data"]).decode("ascii"),
            "format": str(image.get("format") or "jpeg"),
        }
        for image in converted
    ]
