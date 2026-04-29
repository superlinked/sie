from __future__ import annotations

from typing import TYPE_CHECKING

from sie_server.core.prepared import ExtractPreparedItem
from sie_server.types.inputs import is_document_input

if TYPE_CHECKING:
    from sie_server.types.inputs import Item


def extract_item_cost(item: Item) -> int:
    """Return the batching cost for a single extract item.

    For document items the cost is the byte size of the raw document so that
    very large PDFs/DOCX inputs do not get bundled into a single batch with
    other heavy items. For text items the cost is the character count, which
    matches the historical behavior used by GLiNER/GLiClass adapters.
    """
    document = item.document
    if is_document_input(document):
        return len(document["data"])
    text = item.text
    return len(text) if text else 0


def build_extract_prepared_items(items: list[Item]) -> list[ExtractPreparedItem]:
    """Build PreparedItems for a batch of extract items."""
    return [ExtractPreparedItem(cost=extract_item_cost(item), original_index=i) for i, item in enumerate(items)]
