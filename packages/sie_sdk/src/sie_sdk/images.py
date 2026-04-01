"""Image conversion utilities for SIE SDK.

Per DESIGN.md Section 4.3, images are serialized as JPEG bytes for transport.
This module handles conversion from various input formats to JPEG bytes.

Wire format: raw JPEG bytes in msgpack (no base64 encoding).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type alias for all supported image input formats
ImageLike = Union[Image.Image, "NDArray[Any]", bytes, str, Path]

# Default JPEG quality (per design.md)
DEFAULT_JPEG_QUALITY = 95


def to_jpeg_bytes(
    image: ImageLike,
    *,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Convert various image formats to JPEG bytes for wire transport.

    Per DESIGN.md Section 4.3, supports:
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

    Per design.md Section 4.3, images are sent as JPEG bytes wrapped in
    ImageInput format: {"data": <bytes>, "format": "jpeg"}.

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
            img_data = img["data"]
            jpeg_bytes = to_jpeg_bytes(img_data)
            converted.append({"data": jpeg_bytes, "format": "jpeg"})
        else:
            # Direct image input (PIL.Image, ndarray, bytes, str/Path)
            jpeg_bytes = to_jpeg_bytes(img)
            converted.append({"data": jpeg_bytes, "format": "jpeg"})

    item["images"] = converted
    return item
