"""Audio and video conversion utilities for SIE SDK.

Wire format: raw media bytes in msgpack with an optional format hint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MediaLike = bytes | str | Path


def infer_media_format(source: str | Path) -> str | None:
    """Infer a media format hint from a path suffix."""
    suffix = Path(source).suffix.lower()
    return suffix.removeprefix(".") or None


def to_media_bytes(media: MediaLike, *, kind: str) -> tuple[bytes, str | None]:
    """Resolve an audio or video input to bytes and an optional format hint."""
    if isinstance(media, bytes):
        return media, None

    if isinstance(media, (str, Path)):
        path = Path(media)
        if not path.exists():
            msg = f"{kind.capitalize()} file not found: {path}"
            raise FileNotFoundError(msg)
        return path.read_bytes(), infer_media_format(path)

    msg = f"Unsupported {kind} type: {type(media)}. Expected bytes, str, or Path."
    raise TypeError(msg)


def _convert_media_field(item: dict[str, Any], field: str) -> None:
    media = item.get(field)
    if media is None:
        return

    if isinstance(media, dict):
        if "data" not in media:
            msg = f"{field.capitalize()} input must contain a 'data' field."
            raise ValueError(msg)
        data, inferred = to_media_bytes(media["data"], kind=field)
        converted: dict[str, Any] = {
            "data": data,
            "format": media.get("format", inferred),
        }
        if field == "audio" and "sample_rate" in media:
            converted["sample_rate"] = media["sample_rate"]
        item[field] = converted
        return

    data, inferred = to_media_bytes(media, kind=field)
    item[field] = {"data": data, "format": inferred}


def convert_item_media(item: dict[str, Any]) -> dict[str, Any]:
    """Convert an item's audio and video fields to their wire shapes in-place."""
    _convert_media_field(item, "audio")
    _convert_media_field(item, "video")
    return item
