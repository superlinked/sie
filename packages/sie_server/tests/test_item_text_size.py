"""The per-item text size bound shared by the HTTP and queue ingress paths.

``item_size_error`` counts an item's text in UTF-8 bytes plus its metadata's
encoded size; the HTTP request types and ``decode_item`` (the queue path) both
reject an item over ``MAX_ITEM_TEXT_BYTES`` before anything tokenizes it.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pytest
from sie_server.types import inputs
from sie_server.types.inputs import Item, decode_item, item_size_error

CAP = inputs.MAX_ITEM_TEXT_BYTES
# msgpack framing of ``{"state": <str of 65,536 bytes or more>}``: map header,
# the key, and the string header.
STATE_FRAMING = len(msgspec.msgpack.encode({"state": "x" * 70_000})) - 70_000


def _exact_utf8(char: str, total: int) -> str:
    """A text of exactly ``total`` UTF-8 bytes, mostly ``char``."""
    width = len(char.encode("utf-8"))
    count = total // width
    return char * count + "x" * (total - count * width)


class TestLimit:
    def test_default_is_2_mib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_MAX_ITEM_TEXT_BYTES", raising=False)
        assert inputs._item_text_limit_from_env() == 2 * 1024 * 1024

    def test_env_overrides_the_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_MAX_ITEM_TEXT_BYTES", "4096")
        assert inputs._item_text_limit_from_env() == 4096

    @pytest.mark.parametrize("raw", ["0", "-1", "2MiB"])
    def test_env_rejects_a_non_positive_or_malformed_limit(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv("SIE_MAX_ITEM_TEXT_BYTES", raw)
        with pytest.raises(ValueError, match="SIE_MAX_ITEM_TEXT_BYTES must be a positive integer"):
            inputs._item_text_limit_from_env()

    def test_check_reads_the_configured_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(inputs, "MAX_ITEM_TEXT_BYTES", 8)
        assert item_size_error(Item(text="x" * 8), "items[0]") is None
        assert item_size_error(Item(text="x" * 9), "items[0]") == (
            "Field 'items[0]' must hold at most 8 bytes of UTF-8 text"
        )


class TestText:
    def test_ascii_text_at_the_cap_is_accepted(self) -> None:
        assert item_size_error(Item(text="x" * CAP), "items[0]") is None

    def test_ascii_text_over_the_cap_names_the_limit_and_item(self) -> None:
        error = item_size_error(Item(text="x" * (CAP + 1)), "items[3]")
        assert error == f"Field 'items[3]' must hold at most {CAP} bytes of UTF-8 text"

    @pytest.mark.parametrize("char", ["é", "中", "😀"])
    def test_multibyte_text_is_measured_in_utf8_bytes(self, char: str) -> None:
        at_cap = _exact_utf8(char, CAP)
        assert len(at_cap.encode("utf-8")) == CAP
        assert item_size_error(Item(text=at_cap), "items[0]") is None

        over = at_cap + "x"
        # Far fewer characters than bytes: a character count would admit it.
        assert len(over) < CAP
        assert item_size_error(Item(text=over), "items[0]") is not None

    def test_text_longer_than_the_cap_in_characters_is_rejected(self) -> None:
        assert item_size_error(Item(text="é" * (CAP + 1)), "items[0]") is not None

    def test_items_without_text_pass(self) -> None:
        assert item_size_error(Item(), "items[0]") is None
        assert item_size_error(Item(text=""), "items[0]") is None
        assert item_size_error(Item(images=[{"data": b"x" * (CAP + 1)}]), "items[0]") is None


class TestMetadata:
    def test_state_string_at_the_cap_is_accepted(self) -> None:
        item = Item(metadata={"state": "x" * (CAP - STATE_FRAMING)})
        assert item_size_error(item, "items[0]") is None

    def test_state_string_over_the_cap_is_rejected(self) -> None:
        item = Item(metadata={"state": "x" * (CAP - STATE_FRAMING + 1)})
        assert item_size_error(item, "items[1]") == (
            f"Field 'items[1]' must hold at most {CAP} bytes of UTF-8 text and metadata"
        )

    def test_multibyte_state_is_measured_in_utf8_bytes(self) -> None:
        state = _exact_utf8("中", CAP - STATE_FRAMING)
        assert item_size_error(Item(metadata={"state": state}), "items[0]") is None
        assert item_size_error(Item(metadata={"state": state + "x"}), "items[0]") is not None

    @pytest.mark.parametrize(
        "state",
        [
            {"ticket": "x" * (CAP // 2), "notes": "y" * (CAP // 2)},
            [{"role": "user", "content": "x" * (CAP // 4)}] * 5,
        ],
        ids=["object", "turns"],
    )
    def test_structured_state_over_the_cap_is_rejected(self, state: Any) -> None:
        assert item_size_error(Item(metadata={"state": state}), "items[0]") is not None

    def test_text_and_metadata_share_one_bound(self) -> None:
        half = "x" * (CAP // 2)
        assert item_size_error(Item(text=half), "items[0]") is None
        assert item_size_error(Item(metadata={"state": half}), "items[0]") is None
        assert item_size_error(Item(text=half, metadata={"state": half}), "items[0]") is not None

    def test_ordinary_metadata_passes(self) -> None:
        item = Item(
            text="Steve Jobs founded Apple.",
            metadata={"entities": [{"text": "Steve Jobs", "label": "person", "start": 0, "end": 10}]},
        )
        assert item_size_error(item, "items[0]") is None

    def test_json_integer_wider_than_64_bits_is_measured(self) -> None:
        metadata = msgspec.json.decode(b'{"n": 123456789012345678901234567890}')
        assert item_size_error(Item(text="x", metadata=metadata), "items[0]") is None


class TestDecodeItem:
    def test_queue_item_over_the_cap_is_a_validation_error_naming_the_item(self) -> None:
        with pytest.raises(msgspec.ValidationError, match=rf"'items\[7\]' must hold at most {CAP} bytes"):
            decode_item({"text": "x" * (CAP + 1)}, "items[7]")

    def test_queue_state_over_the_cap_is_rejected(self) -> None:
        with pytest.raises(msgspec.ValidationError, match="text and metadata"):
            decode_item({"metadata": {"state": "x" * CAP}}, "items[0]")

    def test_sdk_content_alias_is_measured_as_text(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            decode_item({"content": "x" * (CAP + 1)})

    def test_queue_item_at_the_cap_decodes(self) -> None:
        item = decode_item({"text": "x" * CAP, "metadata": {}}, "items[0]")
        assert item.text is not None
        assert len(item.text) == CAP
