"""Undecodable image bytes raise a typed error across the col-family adapters.

Fix-the-class sweep: every adapter that opens user image bytes with PIL routes
through ``sie_server.types.inputs.decode_image``, which maps PIL's
``UnidentifiedImageError`` (an ``OSError`` subclass that misses the
``ValueError`` -> 400 INVALID_INPUT mapping) to ``InvalidMediaError`` with a
JSON-path message. These adapters have no dedicated test module, so their
loader seams are pinned here.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image
from sie_server.adapters.colpali import ColPaliAdapter
from sie_server.adapters.colqwen2 import ColQwen2Adapter
from sie_server.adapters.colqwen3 import ColQwen3Adapter
from sie_server.adapters.colsmol import ColSmolAdapter
from sie_server.adapters.nemo_colembed import NemoColEmbedAdapter
from sie_server.types.inputs import InvalidMediaError, Item

_NOT_AN_IMAGE = b"valid base64 decoded to these bytes, but they are not an image"


def _png_stub() -> dict[str, bytes]:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    return {"data": buf.getvalue()}


@pytest.mark.parametrize(
    "adapter_cls",
    [ColPaliAdapter, ColQwen2Adapter, ColQwen3Adapter, ColSmolAdapter],
)
def test_col_family_load_images_rejects_undecodable_bytes(adapter_cls: type) -> None:
    adapter = adapter_cls("unused")

    with pytest.raises(InvalidMediaError, match=r"not a decodable image - at `\$\.items\[\*\]\.images\[0\]\.data`"):
        adapter._load_images(Item(images=[{"data": _NOT_AN_IMAGE}]))


def test_col_family_names_the_offending_image_index() -> None:
    adapter = ColPaliAdapter("unused")

    with pytest.raises(InvalidMediaError, match=r"\.images\[1\]\.data"):
        adapter._load_images(Item(images=[_png_stub(), {"data": _NOT_AN_IMAGE}]))


def test_nemo_colembed_encode_images_names_the_item() -> None:
    adapter = NemoColEmbedAdapter("unused")
    items = [Item(images=[_png_stub()]), Item(images=[{"data": _NOT_AN_IMAGE}])]

    with pytest.raises(InvalidMediaError, match=r"at `\$\.items\[1\]\.images\[0\]\.data`"):
        adapter._encode_images(items, is_query=False)
