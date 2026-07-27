from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from sie_server.core.prepared import LightOnOCRPayload, PreparedItem
from sie_server.types.inputs import InvalidMediaError, Item

if TYPE_CHECKING:
    from sie_server.adapters.lighton_ocr.adapter import LightOnOCRAdapter


class _FakeTextConfig:
    eos_token_id = 0


class _FakeConfig:
    text_config = _FakeTextConfig()


class _FakeTokenizer:
    pad_token_id = 0


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    @staticmethod
    def batch_decode(generated: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        # Receives only the post-slice generated tokens (adapter passes out[:, max_len:]);
        # _FakeModel echoes each row's last real input id as the first generated column.
        return [f"id{int(row[0])}" for row in generated]


class _FakeModel:
    """Records generate() calls and echoes each row's last real token as output."""

    dtype = torch.float32
    config = _FakeConfig()

    def __init__(self) -> None:
        self.generate_calls: list[dict] = []

    def generate(self, *, input_ids, attention_mask, pixel_values, image_sizes, **kwargs):
        self.generate_calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "pixel_values_shape": tuple(pixel_values.shape),
                "image_sizes": image_sizes.clone(),
            }
        )
        marker = input_ids[:, -1:].clone()  # last (right-aligned) real token per row
        return torch.cat([input_ids, marker], dim=1)


def _payload(input_id_tail: int, length: int, h: int, w: int):
    ids = torch.arange(1, length, dtype=torch.long)
    ids = torch.cat([ids, torch.tensor([input_id_tail], dtype=torch.long)])  # unique last token
    return LightOnOCRPayload(
        pixel_values=torch.ones(3, h, w),
        input_ids=ids,
        attention_mask=torch.ones(length, dtype=torch.long),
        image_sizes=torch.tensor([h, w], dtype=torch.long),
        original_size=(w, h),
    )


def _prepared(payload, index: int):
    return PreparedItem(payload=payload, cost=1, original_index=index)


class TestLightOnOCRAdapter:
    """Tests for LightOnOCRAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> LightOnOCRAdapter:
        """Create an adapter instance."""
        from sie_server.adapters.lighton_ocr.adapter import LightOnOCRAdapter

        return LightOnOCRAdapter(
            "lightonai/LightOnOCR-2-1B",
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: LightOnOCRAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_page_metering_replaces_generic_image_units(self, adapter: LightOnOCRAdapter) -> None:
        items = [Item(images=[{"data": b"page", "format": "png"}])]

        assert adapter.count_input_images(items) is None

    def test_dims(self, adapter: LightOnOCRAdapter) -> None:
        """Adapter reports empty dimensions (extraction model)."""
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises(self, adapter: LightOnOCRAdapter) -> None:
        """Encode raises NotImplementedError."""
        items = [Item(text="hello")]
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode(items, output_types=["dense"])

    def test_extract_before_load(self, adapter: LightOnOCRAdapter) -> None:
        """Extract before load raises RuntimeError."""
        from sie_server.types.inputs import ImageInput

        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    @pytest.mark.parametrize("images", [None, [], [{"data": b"one"}, {"data": b"two"}]])
    def test_extract_rejects_non_single_image_arrays(
        self,
        adapter: LightOnOCRAdapter,
        images: list[dict[str, bytes]] | None,
    ) -> None:
        adapter._model = _FakeModel()
        adapter._processor = _FakeProcessor()

        with pytest.raises(InvalidMediaError, match="exactly one image"):
            adapter.extract([Item(images=images)])

    def test_build_messages_default(self, adapter: LightOnOCRAdapter) -> None:
        """Default messages have system and user roles with image."""
        messages = adapter._build_messages(instruction=None)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are an OCR engine. Return the markdown representation of the document."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == [{"type": "image"}]

    def test_build_messages_with_instruction(self, adapter: LightOnOCRAdapter) -> None:
        """Instruction appends text content to user message."""
        messages = adapter._build_messages(instruction="Extract tables only")

        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "image"}
        assert content[1] == {"type": "text", "text": "Extract tables only"}

    def test_convert_output(self, adapter: LightOnOCRAdapter) -> None:
        """Markdown text is wrapped in Entity with label 'markdown'."""
        entities = adapter._convert_output("# Title\n\nSome text")

        assert len(entities) == 1
        assert entities[0]["text"] == "# Title\n\nSome text"
        assert entities[0]["label"] == "markdown"
        assert entities[0]["score"] == 1.0

    def test_convert_output_strips_whitespace(self, adapter: LightOnOCRAdapter) -> None:
        """Output text is stripped of leading/trailing whitespace."""
        entities = adapter._convert_output("  \n  # Title  \n  ")

        assert len(entities) == 1
        assert entities[0]["text"] == "# Title"

    def test_system_prompt_configurable(self) -> None:
        """Custom system_prompt is respected."""
        from sie_server.adapters.lighton_ocr.adapter import LightOnOCRAdapter

        custom_prompt = "Extract all text from the image."
        adapter = LightOnOCRAdapter(
            "lightonai/LightOnOCR-2-1B",
            system_prompt=custom_prompt,
        )

        messages = adapter._build_messages(instruction=None)
        assert messages[0]["content"] == custom_prompt

    def test_extract_preprocessed_batches_and_left_pads(self, adapter: LightOnOCRAdapter) -> None:
        """Heterogeneous payloads run through ONE batched generate with left-padding."""
        adapter._model = _FakeModel()
        adapter._processor = _FakeProcessor()
        adapter._device = "cpu"

        # Different prompt lengths (3/5/4) and pixel sizes -> exercises pad + left-pad.
        payloads = [_payload(501, 3, 2, 2), _payload(502, 5, 4, 3), _payload(503, 4, 3, 5)]
        prepared = [_prepared(p, i) for i, p in enumerate(payloads)]

        out = adapter._extract_preprocessed(
            items=[Item(text="x") for _ in range(3)], prepared_items=prepared, max_new_tokens=8, num_beams=1
        )

        # One batched generate (not three serial calls).
        assert len(adapter._model.generate_calls) == 1
        call = adapter._model.generate_calls[0]

        # input_ids left-padded to max length 5; shortest row (len 3) has 2 left pads.
        assert call["input_ids"].shape == (3, 5)
        assert torch.equal(call["input_ids"][0, :2], torch.zeros(2, dtype=torch.long))  # pad_id=0
        assert torch.equal(call["attention_mask"][0], torch.tensor([0, 0, 1, 1, 1]))
        # Real last token preserved at the right edge for every row.
        assert [int(call["input_ids"][i, -1]) for i in range(3)] == [501, 502, 503]

        # pixel_values zero-padded to per-batch max (H=4, W=5) and stacked.
        assert call["pixel_values_shape"] == (3, 3, 4, 5)
        # image_sizes carries TRUE (h, w), never the padded size.
        assert torch.equal(call["image_sizes"], torch.tensor([[2, 2], [4, 3], [3, 5]]))

        # Results decoded per row, in original order.
        assert [ents[0]["text"] for ents in out.entities] == ["id501", "id502", "id503"]
        assert out.pages == [1, 1, 1]

    def test_extract_preprocessed_respects_max_batch_images(self) -> None:
        """max_batch_images chunks a large batch into bounded sub-batches."""
        from sie_server.adapters.lighton_ocr.adapter import LightOnOCRAdapter

        adapter = LightOnOCRAdapter("lightonai/LightOnOCR-2-1B", max_batch_images=2)
        adapter._model = _FakeModel()
        adapter._processor = _FakeProcessor()
        adapter._device = "cpu"

        payloads = [_payload(500 + i, 4, 3, 3) for i in range(5)]
        prepared = [_prepared(p, i) for i, p in enumerate(payloads)]

        out = adapter._extract_preprocessed(
            items=[Item(text="x") for _ in range(5)], prepared_items=prepared, max_new_tokens=8, num_beams=1
        )

        # cap=2 over 5 items -> sub-batches of 2, 2, 1.
        assert [c["input_ids"].shape[0] for c in adapter._model.generate_calls] == [2, 2, 1]
        assert [ents[0]["text"] for ents in out.entities] == [f"id{500 + i}" for i in range(5)]

    def test_extract_preprocessed_max_batch_one_is_serial(self) -> None:
        """max_batch_images=1 issues one generate() per item, preserving order."""
        from sie_server.adapters.lighton_ocr.adapter import LightOnOCRAdapter

        adapter = LightOnOCRAdapter("lightonai/LightOnOCR-2-1B", max_batch_images=1)
        adapter._model = _FakeModel()
        adapter._processor = _FakeProcessor()
        adapter._device = "cpu"

        prepared = [_prepared(_payload(500 + i, 4, 3, 3), i) for i in range(3)]
        out = adapter._extract_preprocessed(
            items=[Item(text="x") for _ in range(3)], prepared_items=prepared, max_new_tokens=8, num_beams=1
        )

        assert [c["input_ids"].shape[0] for c in adapter._model.generate_calls] == [1, 1, 1]
        assert [ents[0]["text"] for ents in out.entities] == [f"id{500 + i}" for i in range(3)]

    def test_extract_preprocessed_mixed_payloads_fall_back(self, adapter: LightOnOCRAdapter) -> None:
        """Non-LightOnOCRPayload items route to _extract_single; payloads still batch, order kept."""
        adapter._model = _FakeModel()
        adapter._processor = _FakeProcessor()
        adapter._device = "cpu"

        def _fake_single(item, **kw):
            return [{"text": "fallback", "label": "markdown", "score": 1.0}]

        adapter._extract_single = _fake_single  # ty: ignore[invalid-assignment]

        # Middle item carries a non-LightOnOCRPayload payload -> fallback path.
        prepared = [
            _prepared(_payload(501, 4, 3, 3), 0),
            _prepared(object(), 1),
            _prepared(_payload(503, 4, 3, 3), 2),
        ]
        out = adapter._extract_preprocessed(
            items=[Item(text="x") for _ in range(3)], prepared_items=prepared, max_new_tokens=8, num_beams=1
        )

        # Only the two real payloads are batched (one generate call of size 2).
        assert [c["input_ids"].shape[0] for c in adapter._model.generate_calls] == [2]
        assert [ents[0]["text"] for ents in out.entities] == ["id501", "fallback", "id503"]
