"""GLiNER adapter builds the model on the target device, with a CPU fallback on OOM."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from sie_server.adapters.gliner import GLiNERAdapter


class FakeGLiNER:
    def __init__(self) -> None:
        self.config = MagicMock(relations_layer=None, max_len=384)
        self.config.encoder_config = {"model_type": "bert"}
        self.model = MagicMock()
        self.data_processor = MagicMock()
        self.data_processor.transformer_tokenizer = MagicMock(model_max_length=512)

    def to(self, *args: Any, **kwargs: Any) -> FakeGLiNER:
        return self


def load_with(monkeypatch: pytest.MonkeyPatch, from_pretrained: MagicMock, device: str) -> GLiNERAdapter:
    module = ModuleType("gliner")
    module.GLiNER = MagicMock(from_pretrained=from_pretrained)  # ty:ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "gliner", module)
    adapter = GLiNERAdapter("fake/gliner", revision="0" * 40)
    with (
        patch("sie_server.adapters.gliner.install_transformers_security_fixes"),
        patch("sie_server.adapters.gliner.bound_gliner_words", return_value=False),
        patch("sie_server.adapters.gliner.gliner_prompt_counter", return_value=lambda *_: 0),
    ):
        adapter.load(device)
    return adapter


def test_gliner_builds_the_model_on_the_target_device(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[torch.device] = []

    def from_pretrained(*_args: Any, **_kwargs: Any) -> FakeGLiNER:
        seen.append(torch.get_default_device())
        return FakeGLiNER()

    load_with(monkeypatch, MagicMock(side_effect=from_pretrained), "cpu")

    assert seen == [torch.device("cpu")]


def test_gliner_passes_the_device_as_map_location(monkeypatch: pytest.MonkeyPatch) -> None:
    from_pretrained = MagicMock(return_value=FakeGLiNER())

    load_with(monkeypatch, from_pretrained, "cpu")

    assert from_pretrained.call_args.kwargs["map_location"] == "cpu"
    assert from_pretrained.call_args.kwargs["revision"] == "0" * 40


@pytest.mark.parametrize(
    "error",
    [torch.OutOfMemoryError("CUDA out of memory"), RuntimeError("MPS backend out of memory (MPS allocated: 7.9 GB)")],
)
def test_gliner_falls_back_to_cpu_construction_when_the_device_is_full(
    monkeypatch: pytest.MonkeyPatch, error: RuntimeError
) -> None:
    model = FakeGLiNER()
    seen: list[torch.device] = []

    def from_pretrained_once_full(*_args: Any, **_kwargs: Any) -> FakeGLiNER:
        if not seen:
            seen.append(torch.get_default_device())
            raise error
        seen.append(torch.get_default_device())
        return model

    from_pretrained = MagicMock(side_effect=from_pretrained_once_full)

    adapter = load_with(monkeypatch, from_pretrained, "cpu")

    assert from_pretrained.call_count == 2
    assert "map_location" not in from_pretrained.call_args_list[1].kwargs
    assert seen == [torch.device("cpu"), torch.device("cpu")]
    assert adapter._model is model


def test_gliner_propagates_other_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from_pretrained = MagicMock(side_effect=RuntimeError("size mismatch for token_rep_layer"))

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_with(monkeypatch, from_pretrained, "cpu")

    assert from_pretrained.call_count == 1
