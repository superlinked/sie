"""Literal source text and native SIE page extraction with provenance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from sie_sdk import SIEClient
from sie_sdk.types import Item

if TYPE_CHECKING:
    from cognee.tasks.ingestion.data_item import DataItem


@dataclass(frozen=True)
class SourceDocument:
    text: str
    source_id: str
    filename: str
    page_number: int | None = None

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.source_id.strip():
            raise ValueError("Source text and source_id must be nonempty")
        if not self.filename or PurePath(self.filename).name != self.filename or "\\" in self.filename:
            raise ValueError("filename must be a name without a directory path")
        if self.page_number is not None and (
            isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or self.page_number < 1
        ):
            raise ValueError("page_number must be a positive integer")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "content_sha256": hashlib.sha256(self.text.encode()).hexdigest(),
        }

    def to_data_item(self) -> DataItem:
        """Preserve OCR/author text verbatim, including URL-looking strings."""
        from cognee.tasks.ingestion.data_item import DataItem

        return DataItem(data=self.text, label=self.source_id, external_metadata=self.metadata, literal_text=True)


def ocr_page(
    client: SIEClient,
    *,
    model: str,
    image: bytes,
    source_id: str,
    filename: str,
    page_number: int,
    max_bytes: int = 10_000_000,
    provision_timeout_s: float = 180,
) -> SourceDocument:
    """Extract one rasterized page; PDF rendering belongs to the caller."""
    if not image or len(image) > max_bytes:
        raise ValueError("Page image is empty or exceeds max_bytes")
    reply = client.extract(
        model,
        Item(images=[image]),
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    if not isinstance(reply, dict) or reply.get("error"):
        raise ValueError("OCR returned an error or malformed response")
    entities = reply.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("OCR returned no text entities")
    texts = []
    for entity in entities:
        if not isinstance(entity, dict) or not isinstance(entity.get("text"), str):
            raise ValueError("OCR returned a malformed text entity")
        texts.append(entity["text"])
    return SourceDocument("\n".join(texts), source_id, filename, page_number)
