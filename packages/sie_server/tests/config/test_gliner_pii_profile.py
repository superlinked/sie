"""GLiNER PII profile precision and loader behavior."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.core.inference import ComputePrecision
from sie_server.core.loader import load_adapter, load_model_config

_MODELS = Path(__file__).resolve().parents[2] / "models"
_MODEL = "urchade/gliner_multi_pii-v1"
_REVISION = "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"
_CONFIG = _MODELS / "urchade__gliner_multi_pii-v1.yaml"
_ADAPTER = "sie_server.adapters.gliner"


def test_pii_profile_pins_full_precision_and_preserves_recipe() -> None:
    config = load_model_config(_CONFIG)
    assert config.sie_id == config.hf_id == _MODEL
    assert config.hf_revision == _REVISION
    assert config.hf_tokenizer_dependencies == {
        "microsoft/mdeberta-v3-base": "a0484667b22365f84929a935b5e50a51f71f159d",
    }
    assert config.max_sequence_length == 384
    assert set(config.profiles) == {"default"}
    profile = config.resolve_profile("default")
    assert profile.compute_precision == "float32"
    assert profile.max_batch_tokens == 16384
    assert profile.adapter_path == f"{_ADAPTER}:GLiNERAdapter"
    assert profile.loadtime == profile.runtime == {}


@pytest.mark.parametrize("engine_precision", ["float16", "bfloat16", "float32"])
def test_pii_profile_overrides_engine_precision(engine_precision: ComputePrecision) -> None:
    adapter = load_adapter(
        load_model_config(_CONFIG), _MODELS, device="cuda:0", default_compute_precision=engine_precision
    )
    assert isinstance(adapter, GLiNERAdapter)
    assert adapter._compute_precision == "float32"
    assert adapter._revision == _REVISION
    assert adapter._threshold == 0.5
    assert adapter._flat_ner is True
    assert adapter._multi_label is False
    assert adapter._merge_adjacent_entities is False
    assert adapter._model is None


@pytest.fixture
def mocked_gliner(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    model = Mock(config=SimpleNamespace(relations_layer=None))
    model.to.return_value = model
    factory = Mock(return_value=model)
    module = ModuleType("gliner")
    monkeypatch.setattr(module, "GLiNER", SimpleNamespace(from_pretrained=factory), raising=False)
    monkeypatch.setitem(sys.modules, "gliner", module)
    monkeypatch.setattr(f"{_ADAPTER}.bound_gliner_words", Mock(return_value=False))
    monkeypatch.setattr(f"{_ADAPTER}.gliner_prompt_counter", Mock())
    return factory, model


def test_pii_cuda_load_uses_full_precision(mocked_gliner: tuple[Mock, Mock]) -> None:
    factory, model = mocked_gliner
    adapter = load_adapter(load_model_config(_CONFIG), _MODELS, device="cuda:0")
    adapter.load("cuda:0")
    factory.assert_called_once_with(_MODEL, map_location="cuda:0", revision=_REVISION)
    model.to.assert_called_once_with("cuda:0", dtype=torch.float32)


@pytest.mark.parametrize(
    ("precision", "dtype"), [("float16", torch.float16), ("bfloat16", torch.bfloat16), ("float32", torch.float32)]
)
def test_generic_gliner_keeps_precision_choices(
    mocked_gliner: tuple[Mock, Mock], precision: ComputePrecision, dtype: torch.dtype
) -> None:
    _, model = mocked_gliner
    assert GLiNERAdapter(_MODEL)._compute_precision == "float16"
    adapter = GLiNERAdapter(_MODEL, compute_precision=precision)
    adapter.load("cuda:0")
    model.to.assert_called_once_with("cuda:0", dtype=dtype)
