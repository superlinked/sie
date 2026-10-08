from __future__ import annotations

import io
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from sie_server.adapters.lighton_ocr.adapter import LightOnOCR3Adapter
from sie_server.core.prepared import GlmOcrPayload, LightOnOCR3Payload, PreparedItem
from sie_server.core.preprocessor.vision import LightOnOCR3Preprocessor
from sie_server.types.inputs import InvalidMediaError, Item


def _item(width: int = 4) -> Item:
    buffer = io.BytesIO()
    Image.new("L", (width, 3)).save(buffer, format="PNG")
    return Item(images=[{"data": buffer.getvalue(), "format": "png"}])


def _inputs(marker: int = 4) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([[1, 2, marker]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "pixel_values": torch.ones((4, 12)),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 0, 1]], dtype=torch.long),
        "processor_extra": torch.tensor([0.5]),
    }


class _Processor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.decoded: list[torch.Tensor] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, **kwargs})
        return _inputs(messages[0]["content"][0]["image"].width)

    def decode(self, generated: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded.append(generated.clone())
        return f"  ![header](0,0,1000,100)\nraw {int(generated[0])}\n  "


class _Model:
    dtype = torch.bfloat16

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.device: str | None = None
        self.evaluated = False

    def to(self, device: str) -> _Model:
        self.device = device
        return self

    def eval(self) -> None:
        self.evaluated = True

    def generate(self, **kwargs: Any) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        return torch.cat((ids, ids[:, -1:] + 100), dim=1)


@pytest.fixture
def adapter() -> LightOnOCR3Adapter:
    value = LightOnOCR3Adapter("lightonai/LightOnOCR-3-4B", max_new_tokens=32)
    value._model = _Model()
    value._processor = _Processor()
    value._device = "cpu"
    value._create_preprocessor()
    return value


@pytest.mark.parametrize("instruction", [None, "", "grounding"])
def test_trained_messages_and_full_processor_payload(adapter: LightOnOCR3Adapter, instruction: str | None) -> None:
    result = adapter.extract([_item()], instruction=instruction)
    call = adapter._processor.calls[0]
    messages = call["messages"]
    assert [message["role"] for message in messages] == ["user"]
    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["image"].mode == "RGB"
    assert content[0]["image"].size == (4, 3)
    assert content[1:] == ([{"type": "text", "text": "grounding"}] if instruction == "grounding" else [])
    assert {key: value for key, value in call.items() if key != "messages"} == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "return_dict": True,
        "return_tensors": "pt",
    }
    generated = adapter._model.calls[0]
    assert generated["pixel_values"].shape == (4, 12)
    assert generated["pixel_values"].dtype == torch.bfloat16
    assert generated["processor_extra"].dtype == torch.bfloat16
    for key in ("input_ids", "attention_mask", "image_grid_thw", "mm_token_type_ids"):
        assert generated[key].dtype == torch.long
        assert torch.equal(generated[key], _inputs()[key])
    assert torch.equal(adapter._processor.decoded[0], torch.tensor([104]))
    assert generated["do_sample"] is True
    assert generated["num_beams"] == 1
    assert generated["temperature"] == 0.1
    assert generated["top_p"] == 1.0
    assert generated["max_new_tokens"] == 32
    assert result.entities == [[{"text": "  ![header](0,0,1000,100)\nraw 104\n  ", "label": "markdown", "score": 1.0}]]
    assert result.pages == [1]


def test_multi_image_order_and_generation_overrides(adapter: LightOnOCR3Adapter) -> None:
    result = adapter.extract([_item(2), _item(6)], options={"max_new_tokens": 8, "temperature": 0.2, "top_p": 0.9})
    assert [row[0]["text"] for row in result.entities] == [
        "  ![header](0,0,1000,100)\nraw 102\n  ",
        "  ![header](0,0,1000,100)\nraw 106\n  ",
    ]
    assert result.pages == [1, 1]
    assert len(adapter._model.calls) == 2
    assert all(call["max_new_tokens"] == 8 and call["temperature"] == 0.2 for call in adapter._model.calls)


@pytest.mark.parametrize("instruction", ["Grounding", " grounding", "Text Recognition:", "Extract tables only"])
def test_untrained_prompt_rejected_before_processing(adapter: LightOnOCR3Adapter, instruction: str) -> None:
    with pytest.raises(ValueError, match="supports only"):
        adapter.extract([_item()], instruction=instruction)
    assert adapter._processor.calls == []
    assert adapter._model.calls == []


@pytest.mark.parametrize("images", [None, [], [{"data": b"one"}, {"data": b"two"}]])
def test_single_image_contract_rejects_complete_request(adapter: LightOnOCR3Adapter, images: Any) -> None:
    with pytest.raises(InvalidMediaError, match="exactly one image"):
        adapter.extract([_item(), Item(images=images)])
    assert adapter._processor.calls == []
    assert adapter._model.calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"max_new_tokens": 33},
        {"temperature": 0},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_p": False},
        {"do_sample": False},
        {"num_beams": 2},
        {"enable_thinking": True},
    ],
)
def test_invalid_generation_controls_fail_before_model(adapter: LightOnOCR3Adapter, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"must|Unsupported"):
        adapter.extract([_item()], options=options)
    assert adapter._processor.calls == []
    assert adapter._model.calls == []


@pytest.mark.parametrize(
    "prepared", [[], [GlmOcrPayload(_inputs(), (4, 3))], [LightOnOCR3Payload(_inputs(), (4, 3), "grounding")]]
)
def test_prepared_payload_cannot_be_missing_or_relabelled(adapter: LightOnOCR3Adapter, prepared: list[Any]) -> None:
    with pytest.raises(ValueError, match=r"length|prompt mode"):
        adapter.extract([_item()], prepared_items=prepared)
    assert adapter._model.calls == []


def test_matching_grounding_payload_avoids_reprocessing(adapter: LightOnOCR3Adapter) -> None:
    payload = LightOnOCR3Payload(_inputs(), (4, 3), "grounding")
    prepared = PreparedItem(payload=payload, cost=1, original_index=0)
    result = adapter.extract([_item()], instruction="grounding", prepared_items=[prepared])
    assert adapter._processor.calls == []
    assert "![header]" in result.entities[0][0]["text"]
    assert payload.inputs["pixel_values"].dtype == torch.float32


@pytest.mark.parametrize("missing", ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"])
def test_incomplete_processor_dict_rejected(adapter: LightOnOCR3Adapter, missing: str) -> None:
    inputs = _inputs()
    del inputs[missing]
    with pytest.raises(ValueError, match="requires tensor"):
        adapter.extract([_item()], prepared_items=[LightOnOCR3Payload(inputs, (4, 3))])
    assert adapter._model.calls == []


def test_wrong_grid_or_double_batched_pixels_rejected_before_any_generation(adapter: LightOnOCR3Adapter) -> None:
    wrong = _inputs()
    wrong["image_grid_thw"] = torch.tensor([[1.0, 2.0, 2.0]])
    with pytest.raises(ValueError, match="one image"):
        adapter.extract(
            [_item(), _item()],
            prepared_items=[LightOnOCR3Payload(_inputs(), (4, 3)), LightOnOCR3Payload(wrong, (4, 3))],
        )
    assert adapter._model.calls == []
    wrong = _inputs()
    wrong["pixel_values"] = wrong["pixel_values"].unsqueeze(0)
    with pytest.raises(ValueError, match="one image"):
        adapter.extract([_item()], prepared_items=[LightOnOCR3Payload(wrong, (4, 3))])


def test_parallel_preparation_retains_order_and_prompt_mode(adapter: LightOnOCR3Adapter) -> None:
    batch = adapter._preprocessor.prepare([_item(2), _item(6)], config=None, instruction="grounding")
    assert [prepared.original_index for prepared in batch.items] == [0, 1]
    assert [prepared.payload.original_size for prepared in batch.items] == [(2, 3), (6, 3)]
    assert all(
        isinstance(prepared.payload, LightOnOCR3Payload) and prepared.payload.instruction == "grounding"
        for prepared in batch.items
    )
    assert batch.total_cost == 2
    assert batch.modality == "image"


@pytest.mark.parametrize("model", ["lightonai/LightOnOCR-3-4B", "lightonai/LightOnOCR-3-0.8B"])
def test_native_loader_pins_processor_and_qwen35_without_remote_code(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    processor = _Processor()
    loaded_model = _Model()
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def load_processor(name: str, **kwargs: Any) -> _Processor:
        calls.append(("processor", name, kwargs))
        return processor

    def load_model(name: str, **kwargs: Any) -> _Model:
        calls.append(("model", name, kwargs))
        return loaded_model

    transformers = ModuleType("transformers")
    transformers.__dict__.update(
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    revision = "a" * 40
    value = LightOnOCR3Adapter(model, revision=revision)
    value.load("cpu")
    assert calls == [
        ("processor", model, {"trust_remote_code": False, "revision": revision}),
        (
            "model",
            model,
            {
                "trust_remote_code": False,
                "revision": revision,
                "dtype": torch.float32,
                "attn_implementation": "sdpa",
            },
        ),
    ]
    assert loaded_model.device == "cpu"
    assert loaded_model.evaluated is True
    assert isinstance(value.get_preprocessor(), LightOnOCR3Preprocessor)
    assert value._resolve_dtype("cuda:0") == torch.bfloat16
    assert value.count_input_images([_item()]) is None
    value.unload()
    assert value.get_preprocessor() is None
    with pytest.raises(RuntimeError, match="not loaded"):
        value.extract([_item()])


@pytest.mark.parametrize(
    "options",
    [{"system_prompt": "OCR"}, {"user_text": "OCR"}, {"do_sample": False}, {"num_beams": 2}, {"enable_thinking": True}],
)
def test_loadtime_cannot_change_trained_contract(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="LightOnOCR-3"):
        LightOnOCR3Adapter("lightonai/LightOnOCR-3-4B", **options)


def test_cuda_float16_rejected() -> None:
    value = LightOnOCR3Adapter("lightonai/LightOnOCR-3-0.8B", compute_precision="float16")
    with pytest.raises(ValueError, match="does not support float16"):
        value._resolve_dtype("cuda:0")
