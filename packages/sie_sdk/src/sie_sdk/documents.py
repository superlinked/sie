from __future__ import annotations

from pathlib import Path
from typing import Any

DocumentLike = bytes | str | Path

# Suffix → wire-format hint. Adapter validation decides what's actually accepted.
_SUFFIX_TO_FORMAT: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".rtf": "rtf",
    ".odt": "odt",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".csv": "csv",
}


def infer_document_format(source: str | Path) -> str | None:
    """Infer a document format hint from a path suffix.

    Returns None if the suffix is unknown so callers can decide whether to
    pass through (and let the server/adapter introspect the bytes) or raise.
    """
    suffix = Path(source).suffix.lower()
    return _SUFFIX_TO_FORMAT.get(suffix)


def to_document_bytes(document: DocumentLike) -> tuple[bytes, str | None]:
    """Resolve a document input to (bytes, format_hint).

    Accepts:
    - bytes: passed through; format hint is None (server may sniff)
    - str / Path: read from disk; format hint inferred from suffix

    Raises:
        FileNotFoundError: If a path-based input does not exist.
        TypeError: If the input is not bytes / str / Path.
    """
    if isinstance(document, bytes):
        return document, None

    if isinstance(document, (str, Path)):
        path = Path(document)
        if not path.exists():
            msg = f"Document file not found: {path}"
            raise FileNotFoundError(msg)
        return path.read_bytes(), infer_document_format(path)

    msg = f"Unsupported document type: {type(document)}. Expected bytes, str, or Path."
    raise TypeError(msg)


def convert_item_document(item: dict[str, Any]) -> dict[str, Any]:
    """Convert an item's document field to wire format in-place.

    Wire format mirrors ImageInput / AudioInput: ``{"data": <bytes>, "format": <str|None>}``.
    A caller-supplied ``format`` always wins over the inferred one.
    """
    if "document" not in item:
        return item

    document = item["document"]
    if document is None:
        return item

    if isinstance(document, dict) and "data" in document:
        data, inferred = to_document_bytes(document["data"])
        # `.get(key, default)` is a presence check — caller-supplied ``format=""``
        # or ``None`` signals "no hint" explicitly and must not be silently
        # replaced by the inferred suffix.
        fmt = document.get("format", inferred)
        item["document"] = {"data": data, "format": fmt}
        return item

    data, inferred = to_document_bytes(document)
    item["document"] = {"data": data, "format": inferred}
    return item
