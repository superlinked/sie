"""Real tiny decision models exercise adapter effects and exact base restoration."""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import huggingface_hub
import msgspec
import pytest
import torch
from gliner2 import BoundaryExtractor, SpanExtractor
from gliner2.configuration import ExtractorConfig
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import save_file
from sie_server.adapters.gliner2.decide import GLiNER2DecideAdapter
from sie_server.adapters.gliner2.decide_lora import GLiNER2DecideLoraAdapter
from sie_server.adapters.laya.adapter import LayaAdapter
from sie_server.adapters.laya.model import LayaDecisionHead, LayaModel
from sie_server.adapters.laya.runtime_lora import LayaLoraAdapter
from sie_server.adapters.lora import LoraStateUncertainError
from sie_server.config.model import ModelConfig
from sie_server.core.extract_cost import build_extract_prepared_items
from sie_server.core.model_loader import ModelLoader
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker.model_worker import PreformedExtractRequest
from sie_server.types.inputs import InvalidInputError, Item
from transformers import BertConfig, BertTokenizerFast, DebertaV2Config, ModernBertConfig, ModernBertModel

ITEMS = [Item(text="alice wants a refund"), Item(text="bob needs technical support")]
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "classify the request", "criteria": ["refund", "support"]},
    "urgent": {"type": "noul", "instructions": "is this urgent", "criteria": {"true": "urgent", "false": "normal"}},
}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def checkpoint_lora(root: Path, model: torch.nn.Module, name: str, targets: list[str]) -> str:
    path = root / name
    path.mkdir()
    LoraConfig(r=4, lora_alpha=8, lora_dropout=0, target_modules=targets, bias="none", task_type=None).save_pretrained(
        path
    )
    generator = torch.Generator().manual_seed(702 + ord(name[0]))
    weights = {}
    for target in targets:
        module = model.get_submodule(target)
        assert type(module) is torch.nn.Linear
        weights[f"base_model.model.{target}.lora_A.weight"] = (
            torch.randn(4, module.in_features, generator=generator) * 0.2
        )
        weights[f"base_model.model.{target}.lora_B.weight"] = (
            torch.randn(module.out_features, 4, generator=generator) * 0.2
        )
    save_file(weights, path / "adapter_model.safetensors")
    return str(path)


@pytest.fixture
def laya_checkpoint(tmp_path: Path) -> Path:
    directory = tmp_path / "laya"
    directory.mkdir()
    vocabulary = directory / "vocab.txt"
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "alice",
        "bob",
        "wants",
        "a",
        "refund",
        "needs",
        "technical",
        "support",
        "classify",
        "the",
        "request",
        "is",
        "this",
        "urgent",
        "normal",
        "true",
        "false",
        "yes",
        "no",
        ":",
        ".",
        "?",
        "choice",
        "noul",
    ]
    vocabulary.write_text("\n".join(tokens))
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    torch.manual_seed(20261019)
    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        local_attention=16,
        global_attn_every_n_layers=2,
        reference_compile=False,
        pad_token_id=0,
        attention_dropout=0,
        embedding_dropout=0,
        mlp_dropout=0,
    )
    config.save_pretrained(directory / "encoder")
    encoder = ModernBertModel(config)
    head = LayaDecisionHead(32, 1)
    model = LayaModel(encoder, head).eval()
    save_file(dict(model.state_dict()), directory / "model.safetensors")
    (directory / "rl_agent_config.json").write_text(json.dumps({"max_len": 128, "head_max_len": 64, "head_layers": 1}))
    return directory


def result(adapter: LayaAdapter, selected: str | None) -> Any:
    return msgspec.to_builtins(adapter.extract(ITEMS, output_schema=QUESTIONS, options={"lora": selected}))


def test_laya_real_named_effects_base_restoration_eviction_and_reload(laya_checkpoint: Path, tmp_path: Path):
    ordinary = LayaAdapter(laya_checkpoint)
    ordinary.load("cpu")
    baseline = msgspec.to_builtins(ordinary.extract(ITEMS, output_schema=QUESTIONS))
    assert ordinary._model is not None
    paths = {
        "a": checkpoint_lora(tmp_path, ordinary._model, "a", ["encoder.layers.0.attn.Wqkv"]),
        "b": checkpoint_lora(tmp_path, ordinary._model, "b", ["encoder.layers.1.mlp.Wi"]),
    }
    ordinary.unload()
    adapter = LayaLoraAdapter(laya_checkpoint, lora_paths=paths, max_loaded_loras=2)
    adapter.load("cpu")
    model = adapter._model
    assert model is not None
    modules = dict(model.named_modules())
    parameters = {name: (p, p.detach().clone(), p.requires_grad) for name, p in model.named_parameters()}
    try:
        assert result(adapter, None) == baseline
        assert adapter.lora_runtime_config().preloaded == frozenset()
        for name in paths:
            assert adapter.load_lora(name) > 0
        changed = {name: result(adapter, name) for name in paths}
        assert changed["a"] != baseline
        assert changed["b"] != baseline
        assert changed["a"] != changed["b"]
        assert result(adapter, None) == baseline
        assert dict(model.named_modules()) == modules
        assert not any(isinstance(m, LoraLinear) for m in model.modules())
        assert all(not p.requires_grad for ps in adapter._lora._loaded.values() for p in ps)
        with ThreadPoolExecutor(max_workers=4) as pool:
            names = ["a", "b", None] * 4
            outputs = list(pool.map(lambda selected: result(adapter, selected), names))
        assert outputs == [baseline if name is None else changed[name] for name in names]
        adapter.unload_lora("b")
        assert result(adapter, "a") == changed["a"]
        adapter.load_lora("b")
        assert result(adapter, "b") == changed["b"]
        adapter.unload_lora("a")
        adapter.unload_lora("b")
        assert result(adapter, None) == baseline
        assert dict(model.named_modules()) == modules
        for name, (original, value, flag) in parameters.items():
            assert model.get_parameter(name) is original
            assert torch.equal(original, value)
            assert original.requires_grad is flag
    finally:
        adapter.unload()
    assert not adapter._lora._resident_layers


def test_laya_admission_denies_source_and_invalid_schema_without_fetch(laya_checkpoint: Path, monkeypatch):
    adapter = LayaLoraAdapter(laya_checkpoint, lora_paths={"a": "org/allowed"})
    fetch = Mock(side_effect=AssertionError("must not fetch"))
    monkeypatch.setattr(adapter, "fetch_lora", fetch)
    with pytest.raises(InvalidInputError, match="declared"):
        adapter.validate_extract_request(ITEMS, output_schema=QUESTIONS, options={"lora": "org/forbidden"})
    with pytest.raises(InvalidInputError):
        adapter.validate_extract_request(ITEMS, output_schema={}, options={"lora": "a"})
    fetch.assert_not_called()


def test_ordinary_laya_has_no_lora_controller(laya_checkpoint: Path):
    ordinary = LayaAdapter(laya_checkpoint)
    assert not hasattr(ordinary, "_lora")
    assert not ordinary.supports_lora()
    assert not hasattr(ordinary, "validate_extract_request")


def test_native_decision_lora_uses_complete_cluster_copy_after_partial_local_cache(loaded_laya, tmp_path, monkeypatch):
    original, paths = loaded_laya
    checkpoint = original._model_name_or_path
    original.unload()
    local, cluster = tmp_path / "local-cache", tmp_path / "cluster-cache"
    revision = "a" * 40
    for root, files in (
        (local, ["adapter_config.json"]),
        (cluster, ["adapter_config.json", "adapter_model.safetensors"]),
    ):
        repository = root / "models--org--decision-lora"
        directory = repository / "snapshots" / revision
        directory.mkdir(parents=True)
        (repository / "refs").mkdir()
        (repository / "refs" / "main").write_text(revision)
        for filename in files:
            shutil.copyfile(Path(paths["a"]) / filename, directory / filename)
    monkeypatch.setenv("SIE_LOCAL_CACHE", str(local))
    monkeypatch.setenv("SIE_CLUSTER_CACHE", str(cluster))
    monkeypatch.setenv("SIE_HF_FALLBACK", "false")
    denied = Mock(side_effect=AssertionError("offline adapter attempted network access"))
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", denied)
    monkeypatch.setattr(huggingface_hub, "file_exists", denied)
    adapter = LayaLoraAdapter(
        checkpoint,
        lora_paths={"a": "org/decision-lora"},
        lora_revisions={"org/decision-lora": revision},
    )
    adapter.load("cpu")
    try:
        baseline = result(adapter, None)
        assert adapter.load_lora("a") > 0
        adapted = result(adapter, "a")
        assert adapted != baseline
        assert result(adapter, None) == baseline
        adapter.unload_lora("a")
        adapter.load_lora("a")
        assert result(adapter, "a") == adapted
        assert result(adapter, None) == baseline
        denied.assert_not_called()
    finally:
        adapter.unload()


@pytest.fixture(params=[("span", "bert"), ("span", "deberta"), ("boundary", "modernbert")])
def decide_checkpoint(request, tmp_path: Path, laya_checkpoint: Path) -> tuple[Path, str]:
    architecture, encoder_kind = request.param
    tokenizer = BertTokenizerFast.from_pretrained(laya_checkpoint / "tokenizer", local_files_only=True)
    fields = {
        "vocab_size": len(tokenizer),
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 256,
        "pad_token_id": 0,
    }
    if encoder_kind == "modernbert":
        encoder = ModernBertConfig(**fields, local_attention=16, reference_compile=False)
        target = "encoder.layers.0.attn.Wqkv"
    elif encoder_kind == "deberta":
        encoder = DebertaV2Config(
            **fields, relative_attention=True, pos_att_type=["p2c", "c2p"], max_relative_positions=64
        )
        target = "encoder.encoder.layer.0.attention.self.query_proj"
    else:
        encoder = BertConfig(**fields)
        target = "encoder.encoder.layer.0.attention.self.query"
    model_class = SpanExtractor if architecture == "span" else BoundaryExtractor
    torch.manual_seed(20261020)
    model = model_class(
        ExtractorConfig(model_name="generated-local", architecture=architecture, max_width=3),
        encoder_config=encoder,
        tokenizer=tokenizer,
    )
    model.eval()
    directory = tmp_path / "decide"
    model.save_pretrained(directory)
    return directory, target


def test_decide_real_named_effects_and_priming_isolation(decide_checkpoint, tmp_path):
    directory, target = decide_checkpoint
    ordinary = GLiNER2DecideAdapter(directory, max_seq_length=256)
    ordinary.load("cpu")
    baseline = msgspec.to_builtins(ordinary.extract(ITEMS, output_schema=QUESTIONS))
    # Real trainer exports can cover the entire span/classifier projection
    # closure even though decision scoring only calls the classifier today.
    head_targets = [
        name
        for name, layer in ordinary._model.named_modules()
        if type(layer) is torch.nn.Linear and name.startswith(("classifier.", "span_rep."))
    ]
    paths = {
        "a": checkpoint_lora(tmp_path, ordinary._model, "a", [target]),
        "b": checkpoint_lora(tmp_path, ordinary._model, "b", head_targets),
    }
    ordinary.unload()
    adapter = GLiNER2DecideLoraAdapter(directory, max_seq_length=256, lora_paths=paths)
    adapter.load("cpu")
    modules = dict(adapter._model.named_modules())
    try:
        for name in paths:
            adapter.load_lora(name)
        expected = {
            name: msgspec.to_builtins(adapter.extract(ITEMS, output_schema=QUESTIONS, options={"lora": name}))
            for name in paths
        }
        assert expected["a"] != baseline
        assert expected["b"] != baseline
        assert expected["a"] != expected["b"]
        groups = [{"items": ITEMS, "output_schema": QUESTIONS, "options": {"lora": name}} for name in [None, "a", "b"]]
        adapter.prime_extract_groups(groups)
        assert not adapter._primed
        for name in ["a", None, "b", None, "a"]:
            actual = msgspec.to_builtins(adapter.extract(ITEMS, output_schema=QUESTIONS, options={"lora": name}))
            assert actual == (baseline if name is None else expected[name])
            assert dict(adapter._model.named_modules()) == modules
        adapter.unload_lora("a")
        adapter.unload_lora("b")
        assert msgspec.to_builtins(adapter.extract(ITEMS, output_schema=QUESTIONS)) == baseline
    finally:
        adapter.unload()


@pytest.fixture
def loaded_laya(laya_checkpoint, tmp_path):
    ordinary = LayaAdapter(laya_checkpoint)
    ordinary.load("cpu")
    paths = {
        name: checkpoint_lora(tmp_path, ordinary._model, name, ["encoder.layers.0.attn.Wqkv"])
        for name in ("a", "b", "c")
    }
    ordinary.unload()
    adapter = LayaLoraAdapter(laya_checkpoint, lora_paths=paths, max_loaded_loras=2)
    adapter.load("cpu")
    try:
        yield adapter, paths
    finally:
        adapter.unload()


def test_laya_graph_bypass_and_base_restoration_even_after_forward_failure(loaded_laya, monkeypatch):
    adapter, _ = loaded_laya
    adapter.load_lora("a")
    baseline = result(adapter, None)
    graph = Mock()
    graph.run.return_value = None  # Native fallback, while recording any attempted replay.
    adapter._graphs = graph
    adapter._graph_tokens = 4096  # Exercise the CPU stand-in as a supported small graph shape.
    result(adapter, "a")
    graph.run.assert_not_called()
    assert adapter._graphs is graph
    assert result(adapter, None) == baseline
    graph.run.assert_called_once()
    model = adapter._model
    modules = dict(model.named_modules())
    with monkeypatch.context() as patch:
        patch.setattr(model, "forward_padded", Mock(side_effect=RuntimeError("forward sentinel")))
        with pytest.raises(RuntimeError, match="forward sentinel"):
            result(adapter, "a")
    assert adapter._lora._active is None
    assert dict(model.named_modules()) == modules
    assert adapter._graphs is graph
    assert result(adapter, None) == baseline
    adapter._graphs = None


def test_live_rebinding_refused_and_failed_removal_blocks_future_serving(loaded_laya, monkeypatch):
    adapter, _ = loaded_laya
    adapter.load_lora("a")
    with pytest.raises(RuntimeError, match="Close"):
        adapter._lora.bind(adapter._model)
    result(adapter, "a")
    monkeypatch.setattr(adapter._lora, "_remove", Mock(side_effect=RuntimeError("remove sentinel")))
    with pytest.raises(LoraStateUncertainError, match="removal"):
        adapter.unload_lora("a")
    for selected in (None, "a", "b"):
        with pytest.raises(LoraStateUncertainError, match="uncertain"):
            result(adapter, selected)


@pytest.mark.parametrize(
    ("field", "value"),
    [("kasa_config", {"enabled": True}), ("future_feature", True), ("future_feature", 0), ("future_feature", 0.0)],
)
def test_new_exporter_features_never_silently_disappear(loaded_laya, field, value):
    adapter, paths = loaded_laya
    config_path = Path(paths["a"]) / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))
    baseline = result(adapter, None)
    with pytest.raises(ValueError, match="Unsupported LoRA configuration"):
        adapter.load_lora("a")
    assert result(adapter, None) == baseline
    assert not adapter._lora._prepared
    assert not adapter._lora._loaded


@pytest.mark.parametrize(
    "selector",
    [
        {"lora_adapter": "a", "lora": "b"},
        {"lora_adapter": None, "lora": "a"},
        {"lora_id": "a"},
        {"lora_adapter": "undeclared"},
        {"lora_adapter": 3},
    ],
)
def test_selection_denied_before_fetch_or_forward(loaded_laya, monkeypatch, selector):
    adapter, _ = loaded_laya
    fetch = Mock(side_effect=AssertionError("fetch must not run"))
    forward = Mock(side_effect=AssertionError("forward must not run"))
    monkeypatch.setattr(adapter, "fetch_lora", fetch)
    monkeypatch.setattr(adapter._model, "forward_padded", forward)
    with pytest.raises(InvalidInputError):
        adapter.validate_extract_request(ITEMS, output_schema=QUESTIONS, options=selector)
    with pytest.raises(InvalidInputError):
        adapter.extract(ITEMS, output_schema=QUESTIONS, options=selector)
    fetch.assert_not_called()
    forward.assert_not_called()


@pytest.mark.asyncio
async def test_native_decision_worker_lazy_loading_and_lru(loaded_laya, monkeypatch):
    adapter, paths = loaded_laya
    fetch = Mock(wraps=adapter.fetch_lora)
    monkeypatch.setattr(adapter, "fetch_lora", fetch)
    config = ModelConfig.model_validate(
        {
            "sie_id": "decision",
            "hf_id": adapter._model_name_or_path,
            "tasks": {"extract": {}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.laya.runtime_lora:LayaLoraAdapter",
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": {"lora_paths": paths, "max_loaded_loras": 2}},
                }
            },
        }
    )
    registry = ModelRegistry()
    loader = ModelLoader(
        preprocessor_registry=MagicMock(),
        postprocessor_registry=MagicMock(),
        all_configs={"decision": config},
        registry_callbacks=registry,
    )
    loaded = loader._finish_load("decision", "cpu", adapter, config)
    registry._configs["decision"] = config
    registry._loaded["decision"] = loaded
    assert loaded.max_loras == 2
    assert loaded.loras == {}
    fetch.assert_not_called()
    await loaded.worker.start()
    try:
        for name in ("a", "b"):
            assert await registry.ensure_lora_loaded_async("decision", name) == (False, True)
            await asyncio.wait_for(asyncio.gather(*registry._background_tasks), 5)
        expected = {name: result(adapter, name) for name in (None, "a", "b")}

        async def submit(name):
            futures = await loaded.worker.submit_extract_preformed_batch(
                [
                    PreformedExtractRequest(
                        build_extract_prepared_items(ITEMS), ITEMS, output_schema=QUESTIONS, options={"lora": name}
                    )
                ],
                lora=name,
            )
            return msgspec.to_builtins((await futures[0]).output)

        names = [None, "a", "b"]
        actual = await asyncio.gather(*(submit(name) for name in names))
        assert actual == [expected[name] for name in names]
        assert await registry.ensure_lora_loaded_async("decision", "c") == (False, True)
        await asyncio.wait_for(asyncio.gather(*registry._background_tasks), 5)
        assert len(loaded.loras) == len(adapter._lora._loaded) == 2
        assert paths["c"] in adapter._lora._loaded
        assert result(adapter, None) == expected[None]
    finally:
        await loaded.worker.stop()
