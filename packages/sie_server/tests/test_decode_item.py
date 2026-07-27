"""Tests for the single validating item decode (`decode_item`) and its
INVALID_INPUT mapping on the queue path. See issue #1537.

The HTTP path validates request items against the typed ``Item`` struct at the
seam; the queue/IPC path historically built ``Item(**kwargs)`` with no type
validation. ``decode_item`` closes that asymmetry by running the same ``Item``
contract through ``msgspec.convert``.
"""

from __future__ import annotations

import base64
import time

import msgspec
import pytest
from sie_server.ipc_types import EncodeBatchItem
from sie_server.queue_executor import _inference_exception_outcome
from sie_server.types.inputs import InvalidInputError, InvalidMediaError, decode_item, media_bytes
from sie_server.types.responses import ErrorCode


class TestDecodeItemAccepts:
    def test_valid_text(self) -> None:
        item = decode_item({"text": "hello", "id": "a"})
        assert item.text == "hello"
        assert item.id == "a"

    def test_remaps_content_to_text(self) -> None:
        # The SDK ships ``content``; the server expects ``text``.
        item = decode_item({"content": "hi"})
        assert item.text == "hi"

    def test_content_ignored_when_text_present(self) -> None:
        item = decode_item({"content": "ignored", "text": "kept"})
        assert item.text == "kept"

    def test_ignores_unknown_keys(self) -> None:
        # Parity with the prior behaviour (which filtered to known fields).
        item = decode_item({"text": "hi", "totally_unknown": 123})
        assert item.text == "hi"

    def test_base64_decodes_str_media_data(self) -> None:
        # Matches the HTTP JSON path: a base64 ``str`` in ``data`` becomes
        # ``bytes``, so ``media_bytes`` downstream gets a valid payload.
        raw = base64.b64encode(b"PNGDATA").decode("ascii")
        item = decode_item({"images": [{"data": raw, "format": "png"}]})
        assert item.images is not None
        assert media_bytes(item.images[0], kind="image") == b"PNGDATA"

    def test_keeps_bytes_media_data(self) -> None:
        item = decode_item({"images": [{"data": b"PNGDATA"}]})
        assert item.images is not None
        assert media_bytes(item.images[0], kind="image") == b"PNGDATA"


class TestDecodeItemRejects:
    def test_bad_text_type(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode_item({"text": 123})

    def test_non_list_images(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode_item({"images": "not-a-list"})

    def test_non_dict_image_element(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode_item({"images": [123]})

    def test_bad_metadata_type(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode_item({"metadata": "not-a-mapping"})

    def test_validation_error_is_value_error(self) -> None:
        # InvalidMediaError documents that both paths surface decode failures
        # as ValueError -> INVALID_INPUT; msgspec.ValidationError is one too.
        assert issubclass(msgspec.ValidationError, ValueError)


def _bi() -> EncodeBatchItem:
    return EncodeBatchItem(
        work_item_id="req-1.0",
        request_id="req-1",
        item_index=0,
        total_items=1,
        timestamp=time.time(),
        item={"text": "hi"},
    )


class TestInvalidInputMapping:
    def test_validation_error_maps_to_invalid_input(self) -> None:
        try:
            decode_item({"text": 123})
        except msgspec.ValidationError as exc:
            outcome = _inference_exception_outcome(_bi(), exc)
        assert outcome.disposition == "publish_error_and_ack"
        assert outcome.error_code == ErrorCode.INVALID_INPUT.value

    def test_invalid_media_maps_to_invalid_input(self) -> None:
        outcome = _inference_exception_outcome(_bi(), InvalidMediaError("bad media"))
        assert outcome.disposition == "publish_error_and_ack"
        assert outcome.error_code == ErrorCode.INVALID_INPUT.value

    def test_typed_adapter_input_error_maps_to_invalid_input(self) -> None:
        outcome = _inference_exception_outcome(_bi(), InvalidInputError("blank candidate"))
        assert outcome.disposition == "publish_error_and_ack"
        assert outcome.error_code == ErrorCode.INVALID_INPUT.value
