"""Real GLiClass checkpoints retain native scoring through runtime LoRA changes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import msgspec
import pytest
import torch
from gliclass import GLiClassModel, GLiClassModelConfig
from peft import LoraConfig
from safetensors.torch import save_file
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.adapters.gliclass.runtime_lora import GLiClassLoraAdapter
from sie_server.types.inputs import InvalidInputError, Item
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, DebertaV2Config, ModernBertConfig, PreTrainedTokenizerFast

ITEMS = [Item(text="refund request"), Item(text="technical support")]
GROUPS = {"intent": ["refund", "support"], "urgent": ["yes", "no"]}


@pytest.fixture(params=["bert", "deberta", "modernbert"])
def checkpoint(request, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "refund",
        "request",
        "technical",
        "support",
        "yes",
        "no",
        "intent",
        "urgent",
        ".",
        ":",
    ]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))  # noqa: S106 -- vocabulary, not a secret
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",  # noqa: S106 -- vocabulary entries, not secrets
        unk_token="[UNK]",  # noqa: S106
        cls_token="[CLS]",  # noqa: S106
        sep_token="[SEP]",  # noqa: S106
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.add_tokens(["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"], special_tokens=True)
    tokenizer.model_max_length = 128
    fields = {
        "vocab_size": len(tokenizer),
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
    }
    if request.param == "modernbert":
        encoder = ModernBertConfig(**fields, local_attention=16, reference_compile=False)
        targets = ["model.encoder_model.layers.0.attn.Wqkv", "model.encoder_model.layers.1.mlp.Wi"]
    elif request.param == "bert":
        encoder = BertConfig(**fields)
        targets = [
            "model.encoder_model.encoder.layer.0.attention.self.query",
            "model.encoder_model.encoder.layer.1.intermediate.dense",
        ]
    else:
        encoder = DebertaV2Config(
            **fields,
            relative_attention=True,
            position_buckets=16,
            pos_att_type=["p2c", "c2p"],
            position_biased_input=False,
        )
        targets = [
            "model.encoder_model.encoder.layer.0.attention.self.query_proj",
            "model.encoder_model.encoder.layer.1.intermediate.dense",
        ]
    config = GLiClassModelConfig(
        encoder_config=encoder,
        encoder_model="local-tiny",
        vocab_size=len(tokenizer),
        scorer_type="simple",
        class_token_index=tokenizer.convert_tokens_to_ids("<<LABEL>>"),
        text_token_index=tokenizer.convert_tokens_to_ids("<<SEP>>"),
        example_token_index=tokenizer.convert_tokens_to_ids("<<EXAMPLE>>"),
        prompt_first=False,
        dropout=0,
    )
    model = GLiClassModel(config).eval()
    directory = tmp_path / "model"
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    paths = {}
    for name, target in zip(("a", "b"), targets, strict=True):
        path = tmp_path / name
        path.mkdir()
        LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0, bias="none", task_type=None, target_modules=[target]
        ).save_pretrained(path)
        layer = model.get_submodule(target)
        generator = torch.Generator().manual_seed(ord(name))
        save_file(
            {
                f"base_model.model.{target}.lora_A.weight": torch.randn(4, layer.in_features, generator=generator)
                * 0.2,
                f"base_model.model.{target}.lora_B.weight": torch.randn(layer.out_features, 4, generator=generator)
                * 0.2,
            },
            path / "adapter_model.safetensors",
        )
        paths[name] = str(path)
    try:
        yield directory, paths
    finally:
        torch.set_num_threads(threads)


def result(adapter, selected):
    return msgspec.to_builtins(
        adapter.extract(
            ITEMS, options={"label_groups": GROUPS, "classification_type": "multi-label", "lora_adapter": selected}
        )
    )


def test_gliclass_real_adapters_isolation_and_exact_base_restoration(checkpoint):
    directory, paths = checkpoint
    ordinary = GLiClassAdapter(directory, max_seq_length=128)
    ordinary.load("cpu")
    # This tiny untrained DeBERTa has almost identical class-token features.
    # Independent sigmoids expose its real logit deltas; a two-class softmax
    # rounds the common shifts to exactly 0.5 and cannot be an effect oracle.
    baseline = msgspec.to_builtins(
        ordinary.extract(ITEMS, options={"label_groups": GROUPS, "classification_type": "multi-label"})
    )
    assert not ordinary.supports_lora()
    assert not hasattr(ordinary, "_lora")
    ordinary.unload()
    adapter = GLiClassLoraAdapter(directory, lora_paths=paths, max_seq_length=128)
    adapter.load("cpu")
    model = adapter._lora_model()
    modules = dict(model.named_modules())
    tensors = {name: (tensor, tensor.detach().clone()) for name, tensor in model.named_parameters()}
    try:
        for name in paths:
            assert adapter.load_lora(name) > 0
        expected = {name: result(adapter, name) for name in paths}
        assert expected["a"] != baseline
        assert expected["b"] != baseline
        assert expected["a"] != expected["b"]
        assert result(adapter, None) == baseline
        names = ["a", None, "b"] * 3
        with ThreadPoolExecutor(max_workers=3) as pool:
            actual = list(pool.map(lambda name: result(adapter, name), names))
        assert actual == [baseline if name is None else expected[name] for name in names]
        adapter.unload_lora("a")
        assert result(adapter, "b") == expected["b"]
        adapter.load_lora("a")
        assert result(adapter, "a") == expected["a"]
        for name in paths:
            adapter.unload_lora(name)
        assert result(adapter, None) == baseline
        assert dict(model.named_modules()) == modules
        for name, (original, values) in tensors.items():
            assert model.get_parameter(name) is original
            assert torch.equal(original, values)
        with pytest.raises(InvalidInputError, match="output_schema"):
            adapter.validate_extract_request(ITEMS, output_schema={"intent": {}}, options={"lora_adapter": "a"})
    finally:
        adapter.unload()
