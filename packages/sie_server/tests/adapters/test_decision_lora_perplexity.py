"""Offline native Qwen3.5 LoRA lifecycle; run in the Transformers 5 bundle."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec
import pytest
import torch
from peft import LoraConfig
from safetensors.torch import save_file
from sie_server.adapters.pplx_decider.model import load_model
from sie_server.adapters.pplx_decider.runtime_lora import PerplexityDecisionLoraAdapter
from sie_server.types.inputs import Item

qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5", reason="requires the transformers5 bundle")
configuration = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")

SCHEMA = {
    "intent": {"type": "choice", "instructions": "classify", "criteria": {"refund": None, "support": None}},
    "urgent": {"type": "noul", "instructions": "urgent", "criteria": {"true": "urgent", "false": "normal"}},
}
ITEMS = [Item(text="refund"), Item(text="support")]


class TinyProcessor:
    """Deterministic text rows isolate the real backbone/readout from tokenizer assets."""

    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def __call__(self, *, text, **kwargs):
        row = [2 + value % 54 for value in text[0].encode()[:24]]
        return {"input_ids": torch.tensor([row]), "attention_mask": torch.ones(1, len(row), dtype=torch.long)}


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20261019)
    config = configuration.Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [3, 3, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "num_position_embeddings": 16,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    config._attn_implementation = "sdpa"
    native = qwen.Qwen3_5Model(config).eval()
    native.save_pretrained(tmp_path, safe_serialization=True)
    readout = torch.nn.Linear(32, 255, bias=False)
    save_file({"weight": readout.weight.detach()}, tmp_path / "readout.safetensors")
    model = load_model(
        tmp_path,
        {"temperature": 1.7, "pooling": "last", "attention_mode": "noncausal_full_attention"},
        "cpu",
        torch.float32,
    )
    del native
    paths = {}
    for name, target in [
        ("a", "backbone.language_model.layers.1.self_attn.q_proj"),
        ("b", "backbone.language_model.layers.0.linear_attn.in_proj_qkv"),
    ]:
        directory = tmp_path / name
        directory.mkdir()
        LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0, bias="none", target_modules=[target], task_type=None
        ).save_pretrained(directory)
        layer = model.get_submodule(target)
        generator = torch.Generator().manual_seed(ord(name))
        weights = {
            f"base_model.model.{target}.lora_A.weight": torch.randn(4, layer.in_features, generator=generator) * 0.2,
            f"base_model.model.{target}.lora_B.weight": torch.randn(layer.out_features, 4, generator=generator) * 0.2,
        }
        save_file(weights, directory / "adapter_model.safetensors")
        paths[name] = str(directory)
    instance = PerplexityDecisionLoraAdapter(tmp_path, lora_paths=paths, compute_precision="float32")
    instance._model = model
    instance._processor = TinyProcessor()
    instance._tokenizer = TinyProcessor()
    instance._device = "cpu"
    instance._codes = tuple(str(index) for index in range(255))
    instance._lora.bind(model)
    try:
        yield instance
    finally:
        instance.unload()
        torch.set_num_threads(threads)


def output(adapter, name):
    return msgspec.to_builtins(adapter.extract(ITEMS, output_schema=SCHEMA, options={"lora_adapter": name}))


def test_perplexity_real_attention_recurrence_lora_and_complete_base_restoration(adapter):
    model = adapter._model
    modules = dict(model.named_modules())
    tensors = {key: (parameter, parameter.detach().clone()) for key, parameter in model.named_parameters()}
    baseline = output(adapter, None)
    for name in ("a", "b"):
        assert adapter.load_lora(name) > 0
    expected = {name: output(adapter, name) for name in ("a", "b")}
    assert expected["a"] != baseline
    assert expected["b"] != baseline
    assert expected["a"] != expected["b"]
    assert output(adapter, None) == baseline
    with ThreadPoolExecutor(max_workers=3) as pool:
        names = ["a", None, "b"] * 3
        values = list(pool.map(lambda name: output(adapter, name), names))
    assert values == [baseline if name is None else expected[name] for name in names]
    adapter.unload_lora("b")
    assert output(adapter, "a") == expected["a"]
    adapter.load_lora("b")
    assert output(adapter, "b") == expected["b"]
    for name in ("a", "b"):
        adapter.unload_lora(name)
    assert output(adapter, None) == baseline
    assert dict(model.named_modules()) == modules
    for key, (original, value) in tensors.items():
        assert model.get_parameter(key) is original
        assert torch.equal(original, value)
