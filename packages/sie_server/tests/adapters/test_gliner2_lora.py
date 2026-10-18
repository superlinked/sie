"""Offline GLiNER2 entity-LoRA mechanisms; no pretrained or performance claim."""

import asyncio
import gc
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import msgspec
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gliner2 import GLiNER2
from gliner2.model import ExtractorConfig
from gliner2.training.lora import _resolve_targets
from peft import LoraConfig, PeftModel
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import load_file, save_file
from sie_sdk import SIEClient
from sie_sdk.client.errors import ServerError
from sie_server.adapters.gliner2 import _lora
from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
from sie_server.adapters.gliner2.entity_lora import GLiNER2EntityLoraAdapter
from sie_server.adapters.lora import LoraStateUncertainError
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.api.extract import router as extract_router
from sie_server.config.model import ModelConfig
from sie_server.core.extract_cost import build_extract_prepared_items
from sie_server.core.loader import expand_profile_variants, load_adapter
from sie_server.core.model_loader import ModelLoader
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker.model_worker import PreformedExtractRequest
from sie_server.types.inputs import InvalidInputError, Item
from transformers import BertConfig, BertTokenizerFast, DebertaV2Config

TEXTS = ["Alice met Bob in London.", "Bob works with Alice in Paris."]
LABELS = ["person", "city"]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(params=["bert", "deberta"])
def raw_model(request: pytest.FixtureRequest, tmp_path: Path) -> GLiNER2:
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "alice",
        "bob",
        "london",
        "paris",
        "met",
        "works",
        "with",
        "in",
        ".",
        "person",
        "city",
        "entities",
        "topic",
        "yes",
        "no",
        "(",
        ")",
        ",",
        "|",
    ]
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=True)
    torch.manual_seed(20261017)
    settings = {
        "vocab_size": len(tokenizer),
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "max_position_embeddings": 256,
        "hidden_dropout_prob": 0,
        "attention_probs_dropout_prob": 0,
    }
    encoder = (
        BertConfig(**settings)
        if request.param == "bert"
        else DebertaV2Config(
            **settings,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
            max_relative_positions=64,
        )
    )
    model = GLiNER2(
        ExtractorConfig(model_name="generated-local", max_width=3, counting_layer="count_lstm"),
        encoder_config=encoder,
        tokenizer=tokenizer,
    )
    # Mechanical fixture only, fixed before every baseline: avoid vacuous empty
    # random-model entity output. This is not a trained extraction-quality test.
    count = [m for m in model.count_pred.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        count.weight.zero_()
        count.bias.zero_()
        count.bias[2] = 8
    return model.eval().to("cpu")


def make_checkpoint(root: Path, model: GLiNER2, alias: str, *, targets: list[str] | None = None) -> str:
    path = root / alias
    path.mkdir()
    resolved = targets or _resolve_targets(model, ["encoder.query", "encoder.value", "span_rep"])
    LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0, bias="none", target_modules=resolved, task_type=None, inference_mode=True
    ).save_pretrained(path)
    generator = torch.Generator().manual_seed(7910 + ord(alias[0]) - ord("a"))
    weights = {}
    for target in resolved:
        layer = model.get_submodule(target)
        assert type(layer) is torch.nn.Linear
        prefix = f"base_model.model.{target}"
        weights[f"{prefix}.lora_A.weight"] = torch.randn(4, layer.in_features, generator=generator) * 0.2
        weights[f"{prefix}.lora_B.weight"] = torch.randn(layer.out_features, 4, generator=generator) * 0.2
    save_file(weights, path / "adapter_model.safetensors")
    return str(path)


def adapter_for(model: GLiNER2, paths: dict[str, str]) -> GLiNER2EntityLoraAdapter:
    adapter = GLiNER2EntityLoraAdapter(
        "generated-local",
        lora_paths=paths,
        max_loaded_loras=2,
        compute_precision="float32",
        threshold=0,
        max_seq_length=64,
    )
    adapter._model = model
    adapter._device = "cpu"
    adapter._use_linear_word_splitter(model.processor, model.encoder.config)
    adapter._lora.bind(model)
    return adapter


def snapshot(model: GLiNER2) -> dict[str, Any]:
    return {
        "parameters": {n: (p, p.detach().clone(), p.requires_grad) for n, p in model.named_parameters()},
        "modules": {n: (m, m.training) for n, m in model.named_modules()},
        "attributes": {
            n: {
                key: (
                    key in m.__dict__,
                    m.__dict__.get(key),
                    dict(m.__dict__[key]) if isinstance(m.__dict__.get(key), dict) else None,
                )
                for key in ("peft_config", "_hf_peft_config_loaded")
            }
            for n, m in model.named_modules()
        },
    }


def assert_restored(model: GLiNER2, before: dict[str, Any]) -> None:
    assert set(dict(model.named_modules())) == set(before["modules"])
    assert set(dict(model.named_parameters())) == set(before["parameters"])
    for name, (parameter, value, flag) in before["parameters"].items():
        actual = model.get_parameter(name)
        assert actual is parameter
        assert torch.equal(actual, value)
        assert actual.requires_grad is flag
    for name, (module, training) in before["modules"].items():
        assert model.get_submodule(name) is module
        assert module.training is training
        for key, (present, value, contents) in before["attributes"][name].items():
            assert (key in module.__dict__) is present
            if present:
                assert module.__dict__[key] is value
                if contents is not None:
                    assert module.__dict__[key] == contents


def serve(adapter: GLiNER2EntityLoraAdapter, name: str | None) -> Any:
    return msgspec.to_builtins(adapter.extract([Item(text=t) for t in TEXTS], labels=LABELS, options={"lora": name}))


def assert_native_base_with_residents(adapter: GLiNER2EntityLoraAdapter) -> None:
    assert adapter._lora._resident_layers
    assert all(layer.disable_adapters for layer in adapter._lora._resident_layers.values())
    assert not any(isinstance(layer, LoraLinear) for layer in adapter._model.modules())
    assert all(adapter._model.get_submodule(name) is original for name, original in adapter._lora._originals.items())


@pytest.mark.parametrize("disjoint", [False, True])
def test_resident_base_uses_exact_original_tree_without_visiting_wrappers(
    raw_model: GLiNER2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disjoint: bool
) -> None:
    targets = _resolve_targets(raw_model, ["encoder.query", "encoder.value", "span_rep"])
    paths = {
        name: make_checkpoint(tmp_path, raw_model, name, targets=targets[index::2] if disjoint else targets)
        for index, name in enumerate(("a", "b"))
    }
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    baseline = serve(adapter, None)
    for name in paths:
        adapter.load_lora(name)
    assert dict(raw_model.named_modules()) == {name: value[0] for name, value in before["modules"].items()}
    assert dict(raw_model.named_parameters()) == {name: value[0] for name, value in before["parameters"].items()}
    layers = adapter._lora._resident_layers
    assert layers
    assert all(isinstance(layer, LoraLinear) for layer in layers.values())
    assert all(layer.get_base_layer() is before["modules"][name][0] for name, layer in layers.items())
    identities = {name: id(layer) for name, layer in layers.items()}
    with monkeypatch.context() as patch:
        for layer in layers.values():
            patch.setattr(layer, "forward", Mock(side_effect=AssertionError("base visited LoRA forward")))
            patch.setattr(layer, "train", Mock(side_effect=AssertionError("base eval visited LoRA wrapper")))
        assert serve(adapter, None) == baseline
    assert serve(adapter, "a") != baseline
    assert serve(adapter, "b") != baseline
    assert serve(adapter, None) == baseline
    assert identities == {name: id(layer) for name, layer in layers.items()}
    assert dict(raw_model.named_modules()) == {name: value[0] for name, value in before["modules"].items()}
    adapter.unload()
    assert not layers
    assert_restored(raw_model, before)


def test_real_named_effects_eviction_and_full_restoration(raw_model: GLiNER2, tmp_path: Path) -> None:
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b")}
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    processor = raw_model.processor
    baseline = serve(adapter, None)
    assert all(baseline["entities"])
    adapter.load_lora("a")
    adapter.load_lora("b")
    observed = []
    handles = []
    for name, layer in adapter._lora._resident_layers.items():
        if isinstance(layer, LoraLinear):

            def observe(module, args, output, name=name):
                observed.append(
                    (name, tuple(module.active_adapters), module.disable_adapters, bool(torch.isfinite(output).all()))
                )

            handles.append(layer.register_forward_hook(observe))
    try:
        a, b = serve(adapter, "a"), serve(adapter, "b")
        assert a != baseline
        assert b != baseline
        assert a != b
        assert observed
        assert all(row[3] for row in observed)
        assert any(not row[2] for row in observed)
        assert serve(adapter, None) == baseline
        assert adapter._model is raw_model
        assert raw_model.processor is processor
        assert all(not p.requires_grad for parameters in adapter._lora._loaded.values() for p in parameters)
        assert all(p.requires_grad is flag for p, flag in adapter._lora._grad)
    finally:
        for handle in handles:
            handle.remove()
    adapter.unload_lora("a")
    assert serve(adapter, "b") == b
    adapter.load_lora("a")
    assert serve(adapter, "a") == a
    parameter_refs = [weakref.ref(p) for params in adapter._lora._loaded.values() for p in params]
    adapter.unload_lora("b")
    adapter.unload_lora("a")
    assert serve(adapter, None) == baseline
    assert_restored(raw_model, before)
    gc.collect()
    assert all(ref() is None for ref in parameter_refs)
    adapter.unload()


def test_disjoint_eviction_releases_empty_wrappers_and_can_reattach(raw_model, tmp_path):
    targets = {
        name: _resolve_targets(raw_model, selected)
        for name, selected in (("a", ["encoder.query"]), ("b", ["span_rep"]))
    }
    paths = {name: make_checkpoint(tmp_path, raw_model, name, targets=targets[name]) for name in targets}
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    base = serve(adapter, None)
    for name in paths:
        adapter.load_lora(name)
    expected = {name: serve(adapter, name) for name in paths}
    removed_layers = [weakref.ref(adapter._lora._resident_layers[name]) for name in targets["b"]]
    adapter.unload_lora("b")
    assert set(adapter._lora._resident_layers) == set(targets["a"])
    assert all(reference() is None for reference in removed_layers)
    assert_native_base_with_residents(adapter)
    assert serve(adapter, None) == base
    assert serve(adapter, "a") == expected["a"]
    adapter.load_lora("b")
    assert serve(adapter, "b") == expected["b"]
    adapter.unload()
    assert_restored(raw_model, before)


@pytest.mark.parametrize("second", [False, True])
def test_partial_detach_attachment_failure_restores_native_tree_and_prior_resident(
    raw_model, tmp_path, monkeypatch, second
):
    a_targets = _resolve_targets(raw_model, ["encoder.query"])
    b_targets = _resolve_targets(raw_model, ["span_rep"])
    paths = {
        "a": make_checkpoint(tmp_path, raw_model, "a", targets=a_targets),
        "b": make_checkpoint(tmp_path, raw_model, "b", targets=b_targets),
    }
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    baseline = serve(adapter, None)
    expected = None
    if second:
        adapter.load_lora("a")
        expected = serve(adapter, "a")
    ordered = [name for name in adapter._lora._originals if name in b_targets]
    assert len(ordered) > 1
    parent_name, _, leaf = ordered[1].rpartition(".")
    parent = raw_model.get_submodule(parent_name)
    original = adapter._lora._originals[ordered[1]]
    assign = torch.nn.Module.__setattr__
    calls = 0

    def fail_once(owner, name, value):
        nonlocal calls
        if owner is parent and name == leaf and value is original:
            calls += 1
            if calls == 1:
                assert raw_model.get_submodule(ordered[0]) is adapter._lora._originals[ordered[0]]
                raise RuntimeError("partial detach sentinel")
        return assign(owner, name, value)

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.Module, "__setattr__", fail_once)
        with pytest.raises(RuntimeError, match="partial detach sentinel"):
            adapter.load_lora("b")
    assert calls >= 2
    assert not adapter._lora._uncertain
    assert paths["b"] not in adapter._lora._loaded
    assert serve(adapter, None) == baseline
    if second:
        assert set(adapter._lora._resident_layers) == set(a_targets)
        assert_native_base_with_residents(adapter)
        assert serve(adapter, "a") == expected
    else:
        assert not adapter._lora._resident_layers
        assert_restored(raw_model, before)
    adapter.load_lora("b")
    assert serve(adapter, "b") != baseline
    adapter.unload()
    assert_restored(raw_model, before)


def test_detached_adapter_precision_change_is_refused_before_selection(raw_model, tmp_path):
    adapter = adapter_for(raw_model, {"a": make_checkpoint(tmp_path, raw_model, "a")})
    adapter.load_lora("a")
    assert_native_base_with_residents(adapter)
    raw_model.half()
    with pytest.raises(LoraStateUncertainError, match="selection failed"):
        adapter.set_active_lora("a")
    with pytest.raises(LoraStateUncertainError, match="uncertain"):
        adapter.set_active_lora(None)
    adapter.unload()
    assert not adapter._lora._resident_layers


def test_partial_reattach_selection_failure_refuses_inference_until_teardown(raw_model, tmp_path, monkeypatch):
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b")}
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    for name in paths:
        adapter.load_lora(name)
    layers = list(adapter._lora._resident_layers.items())
    assert len(layers) > 1
    parent_name, _, leaf = layers[1][0].rpartition(".")
    parent = raw_model.get_submodule(parent_name)
    assign = torch.nn.Module.__setattr__

    def fail_reattach(owner, name, value):
        if owner is parent and name == leaf and value is layers[1][1]:
            assert raw_model.get_submodule(layers[0][0]) is layers[0][1]
            raise RuntimeError("partial reattach sentinel")
        return assign(owner, name, value)

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.Module, "__setattr__", fail_reattach)
        with pytest.raises(LoraStateUncertainError, match="selection failed"):
            adapter.set_active_lora("a")
        for name in (None, "a", "b"):
            with pytest.raises(LoraStateUncertainError, match="uncertain"):
                serve(adapter, name)
    adapter.unload()
    assert not adapter._lora._resident_layers
    assert_restored(raw_model, before)


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("removed", ["a", "b"])
@pytest.mark.parametrize("reset_to_base", [False, True])
@pytest.mark.parametrize("disjoint", [False, True])
def test_remove_last_selected_resident_synchronizes_controller_and_survivor(
    raw_model: GLiNER2,
    tmp_path: Path,
    order: tuple[str, str],
    removed: str,
    reset_to_base: bool,
    disjoint: bool,
) -> None:
    targets = {
        "a": _resolve_targets(raw_model, ["encoder.query"]),
        "b": _resolve_targets(raw_model, ["span_rep"]),
    }
    paths = {
        name: make_checkpoint(tmp_path, raw_model, name, targets=targets[name] if disjoint else None) for name in order
    }
    for index, parameter in enumerate(raw_model.parameters()):
        parameter.requires_grad_(index % 2 == 0)
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    base = serve(adapter, None)
    for name in order:
        adapter.load_lora(name)
    survivor = "a" if removed == "b" else "b"
    expected = serve(adapter, survivor)
    adapter.set_active_lora(removed)
    if reset_to_base:
        adapter.set_active_lora(None)
    removed_parameters = [weakref.ref(parameter) for parameter in adapter._lora._loaded[paths[removed]]]
    adapter.unload_lora(removed)
    assert set(adapter._lora._loaded) == {paths[survivor]}
    internal = PEFTLoRAMixin._peft_adapter_name(paths[survivor])
    assert adapter._lora._controller.active_adapter == internal
    assert adapter._lora._tuner().active_adapters == [internal]
    assert adapter._lora._active is None
    assert not adapter._lora._uncertain
    assert_native_base_with_residents(adapter)
    assert all(not parameter.requires_grad for values in adapter._lora._loaded.values() for parameter in values)
    assert all(parameter.requires_grad is flag for parameter, flag in adapter._lora._grad)
    assert all(reference() is None for reference in removed_parameters)
    assert serve(adapter, None) == base
    assert serve(adapter, survivor) == expected
    adapter.load_lora(removed)
    assert serve(adapter, None) == base
    adapter.unload_lora(removed)
    assert serve(adapter, survivor) == expected
    adapter.unload_lora(survivor)
    assert_restored(raw_model, before)
    adapter.unload()


@pytest.mark.parametrize("disjoint", [False, True])
def test_second_attach_rollback_restores_survivor_after_failed_adapter_became_active(
    raw_model: GLiNER2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disjoint: bool
) -> None:
    paths = {
        "a": make_checkpoint(
            tmp_path, raw_model, "a", targets=_resolve_targets(raw_model, ["span_rep"]) if disjoint else None
        ),
        "b": make_checkpoint(
            tmp_path, raw_model, "b", targets=_resolve_targets(raw_model, ["encoder.query"]) if disjoint else None
        ),
    }
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    base = serve(adapter, None)
    adapter.load_lora("a")
    expected = serve(adapter, "a")
    original = PeftModel.load_adapter

    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.set_adapter(kwargs["adapter_name"])
        raise RuntimeError("second attachment sentinel")

    monkeypatch.setattr(PeftModel, "load_adapter", fail)
    with pytest.raises(RuntimeError, match="second attachment sentinel"):
        adapter.load_lora("b")
    assert not adapter._lora._uncertain
    assert set(adapter._lora._loaded) == {paths["a"]}
    assert adapter._lora._prepared == {}
    internal = PEFTLoRAMixin._peft_adapter_name(paths["a"])
    assert adapter._lora._controller.active_adapter == internal
    assert adapter._lora._tuner().active_adapters == [internal]
    assert_native_base_with_residents(adapter)
    assert all(not parameter.requires_grad for values in adapter._lora._loaded.values() for parameter in values)
    assert serve(adapter, None) == base
    assert serve(adapter, "a") == expected
    monkeypatch.setattr(PeftModel, "load_adapter", original)
    adapter.load_lora("b")
    adapter.unload_lora("b")
    adapter.unload_lora("a")
    assert_restored(raw_model, before)
    adapter.unload()


def test_survivor_selection_failure_during_removal_refuses_all_inference(
    raw_model: GLiNER2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b")}
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    adapter.load_lora("a")
    adapter.load_lora("b")
    adapter.set_active_lora("b")
    monkeypatch.setattr(adapter._lora._controller, "set_adapter", Mock(side_effect=RuntimeError("selection sentinel")))
    with pytest.raises(LoraStateUncertainError, match="removal failed"):
        adapter.unload_lora("b")
    for name in (None, "a", "b"):
        with pytest.raises(LoraStateUncertainError, match="uncertain"):
            serve(adapter, name)
    adapter.unload()
    assert_restored(raw_model, before)


@pytest.mark.parametrize("teardown", [False, True])
def test_selection_and_final_unload_restore_mixed_grad_training_and_peft_layout(
    raw_model: GLiNER2, tmp_path: Path, teardown: bool
) -> None:
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b")}
    for index, parameter in enumerate(raw_model.parameters()):
        parameter.requires_grad_(index % 2 == 0)
    for index, module in enumerate(raw_model.modules()):
        module.training = index % 2 == 0
    raw_model.peft_config = {}
    raw_model._hf_peft_config_loaded = False
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    for name in ("a", "b"):
        adapter.load_lora(name)
        adapter.set_active_lora(name)
        assert all(not p.requires_grad for values in adapter._lora._loaded.values() for p in values)
        assert all(p.requires_grad is flag for p, flag in adapter._lora._grad)
        assert all(m.training is flag for m, flag in adapter._lora._training)
    if teardown:
        adapter.unload()
    else:
        adapter.unload_lora("a")
        adapter.unload_lora("b")
    assert_restored(raw_model, before)
    adapter.unload()


@pytest.mark.parametrize("second", [False, True])
def test_failed_real_injection_restores_base_and_prior_resident(
    raw_model: GLiNER2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second: bool
) -> None:
    targets = _resolve_targets(raw_model, ["encoder.query", "span_rep"])
    a_targets = [target for target in targets if target.startswith("encoder.")]
    b_targets = [target for target in targets if target.startswith("span_rep.")]
    paths = {
        "a": make_checkpoint(tmp_path, raw_model, "a", targets=a_targets),
        "b": make_checkpoint(tmp_path, raw_model, "b", targets=b_targets),
    }
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    baseline = serve(adapter, None)
    if second:
        adapter.load_lora("a")
        a = serve(adapter, "a")
        original = PeftModel.load_adapter

        def fail(self, *args, **kwargs):
            original(self, *args, **kwargs)
            raise RuntimeError("after real second injection")

        monkeypatch.setattr(PeftModel, "load_adapter", fail)
    else:
        original = PeftModel.from_pretrained

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("after real first injection")

        monkeypatch.setattr(PeftModel, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="after real"):
        adapter.load_lora("b")
    assert serve(adapter, None) == baseline
    if second:
        assert serve(adapter, "a") == a
        assert all(not isinstance(raw_model.get_submodule(target), LoraLinear) for target in b_targets)
        adapter.unload_lora("a")
    assert_restored(raw_model, before)
    adapter.unload()


def test_failed_removal_poisoning_refuses_base_and_selected_inference(
    raw_model: GLiNER2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = adapter_for(raw_model, {"a": make_checkpoint(tmp_path, raw_model, "a")})
    adapter.load_lora("a")
    monkeypatch.setattr(adapter._lora, "_remove", Mock(side_effect=RuntimeError("delete failed")))
    with pytest.raises(LoraStateUncertainError):
        adapter.unload_lora("a")
    for name in (None, "a"):
        with pytest.raises(LoraStateUncertainError):
            serve(adapter, name)
    adapter.unload()


@pytest.mark.parametrize("lora", [None, "a"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_schema": {}},
        {"instruction": ""},
        {"options": {"classification_task": "topic"}},
        {"options": {"multi_label": True}},
        {"options": {"positive_label": "yes"}},
        {"options": {"task": "classification"}},
        {"options": {"lora_id": "org/adapter"}},
    ],
)
def test_task_and_legacy_selection_denial_before_any_fetch(
    monkeypatch: pytest.MonkeyPatch, lora: str | None, kwargs: dict[str, Any]
) -> None:
    fetch = Mock(side_effect=AssertionError("must not fetch"))
    monkeypatch.setattr(_lora, "snapshot_download", fetch)
    adapter = GLiNER2EntityLoraAdapter("not-loaded", lora_paths={"a": "org/adapter"})
    kwargs = dict(kwargs)
    kwargs["options"] = {**kwargs.get("options", {}), "lora": lora}
    with pytest.raises(InvalidInputError):
        adapter.extract([Item(text="Alice")], labels=LABELS, **kwargs)
    fetch.assert_not_called()
    assert adapter.lora_runtime_config().preloaded == frozenset()
    assert adapter.lora_runtime_config().pinned == frozenset()


def test_ordinary_gliner2_remains_without_lora_manager_or_admission() -> None:
    adapter = GLiNER2Adapter("unloaded")
    assert not hasattr(adapter, "_lora")
    assert not hasattr(adapter, "validate_extract_request")
    assert not adapter.supports_lora()


@pytest.mark.parametrize("version", ["1.3.1", "1.3.3", "2.0.0rc1", "2.0.1", "2.1.0"])
def test_entity_lora_rejects_unreviewed_versions_before_binding(raw_model, monkeypatch, version):
    controller = _lora.GLiNER2Lora({"a": "org/a"}, None, 1)
    before = snapshot(raw_model)
    monkeypatch.setattr(_lora.importlib.metadata, "version", lambda _name: version)
    with pytest.raises(ValueError, match=r"requires GLiNER2 1\.3\.2 or 2\.0\.0"):
        controller.bind(raw_model)
    assert controller._model is None
    assert controller._closed
    assert not controller._originals
    assert snapshot(raw_model)["modules"] == before["modules"]


@pytest.mark.parametrize("value", [[], "", False, 0])
def test_direct_guard_rejects_empty_nonobject_options_before_model_use(value):
    adapter = GLiNER2EntityLoraAdapter("unloaded", lora_paths={"a": "org/a"})
    with pytest.raises(InvalidInputError, match="object"):
        adapter.extract([Item(text="Alice")], labels=LABELS, options=value)


@pytest.mark.parametrize("option", ["classification_task", "multi_label", "positive_label", "preload_loras"])
def test_entity_constructor_refuses_inherited_modes_and_preload(option):
    with pytest.raises(ValueError, match="load options"):
        GLiNER2EntityLoraAdapter("unloaded", lora_paths={"a": "org/a"}, **{option: None})


@pytest.mark.parametrize(
    "revisions", [{"a": "a" * 40}, {"org/other-profile": "main"}, {"org/a": "main"}, {1: "a" * 40}, [], False]
)
def test_revision_policy_never_drops_a_pin_silently(monkeypatch, revisions):
    fetch = Mock(side_effect=AssertionError("invalid declaration fetched"))
    monkeypatch.setattr(_lora, "snapshot_download", fetch)
    with pytest.raises(ValueError, match="revision"):
        GLiNER2EntityLoraAdapter("unloaded", lora_paths={"a": "org/a"}, lora_revisions=revisions)
    fetch.assert_not_called()


def test_actual_multi_profile_loader_preserves_canonical_pins_and_resolved_capacity(tmp_path, monkeypatch):
    config = ModelConfig.model_validate(
        {
            "sie_id": "entity",
            "hf_id": "unloaded",
            "tasks": {"extract": {}},
            "profiles": {
                name: {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.gliner2.entity_lora:GLiNER2EntityLoraAdapter",
                    "adapter_options": {"loadtime": {"lora_paths": {alias: {"id": source, "revision": pin}}}},
                }
                for name, alias, source, pin in [("default", "a", "org/a", "a" * 40), ("other", "b", "org/b", "b" * 40)]
            },
        }
    )
    fetch = Mock(side_effect=AssertionError("construction must not fetch"))
    monkeypatch.setattr(_lora, "snapshot_download", fetch)
    expanded = expand_profile_variants([config])
    assert config.lora_revisions() == {"org/a": "a" * 40, "org/b": "b" * 40}
    for route, expected in [("entity", {"org/a": "a" * 40}), ("entity:other", {"org/b": "b" * 40})]:
        adapter = load_adapter(expanded[route], tmp_path, device="cpu", max_loras_per_model=3)
        assert type(adapter) is GLiNER2EntityLoraAdapter
        assert adapter._lora._revisions == expected
        assert adapter.lora_runtime_config().max_loaded == 3
        assert adapter.lora_runtime_config().preloaded == frozenset()
    fetch.assert_not_called()


def test_unchanged_selection_avoids_flag_traversal_but_checks_poison_and_residency(raw_model, tmp_path, monkeypatch):
    adapter = adapter_for(raw_model, {"a": make_checkpoint(tmp_path, raw_model, "a")})
    freeze = Mock(wraps=adapter._lora._freeze_adapters)
    monkeypatch.setattr(adapter._lora, "_freeze_adapters", freeze)
    baseline = serve(adapter, None)
    freeze.assert_not_called()
    adapter.load_lora("a")
    freeze.reset_mock()
    assert serve(adapter, None) == baseline
    assert serve(adapter, None) == baseline
    freeze.assert_not_called()
    adapter.set_active_lora("a")
    freeze.assert_called_once()
    adapter.set_active_lora("a")
    freeze.assert_called_once()
    adapter.set_active_lora(None)
    assert freeze.call_count == 2
    adapter._lora._uncertain = True
    with pytest.raises(LoraStateUncertainError, match="uncertain"):
        adapter.set_active_lora(None)
    adapter.unload()
    assert adapter._lora._closed
    with pytest.raises(LoraStateUncertainError, match="uncertain"):
        adapter.set_active_lora(None)
    adapter._lora.bind(raw_model)
    adapter._lora.close()
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.set_active_lora(None)


def test_base_and_overlong_item_metering_match_generic_adapter(raw_model, tmp_path):
    paths = {"a": make_checkpoint(tmp_path, raw_model, "a")}
    ordinary = GLiNER2Adapter("generated-local", threshold=0, max_seq_length=64)
    ordinary._model = raw_model
    ordinary._use_linear_word_splitter(raw_model.processor, raw_model.encoder.config)
    items = [Item(text=TEXTS[0]), Item(text="Alice " * 80), Item(text=TEXTS[1])]
    baseline = msgspec.to_builtins(ordinary.extract(items, labels=LABELS))
    assert baseline["errors"][1]["code"] == "INPUT_TOO_LONG"
    assert baseline["input_token_counts"][1] == 0
    assert baseline["input_token_counts"][0] > 0
    adapter = GLiNER2EntityLoraAdapter("generated-local", lora_paths=paths, threshold=0, max_seq_length=64)
    adapter._model = raw_model
    for attr in ("_word_splitter", "_count_subwords", "_count_prompt", "_lower_text_first", "_quadratic_attention"):
        setattr(adapter, attr, getattr(ordinary, attr))
    adapter._lora.bind(raw_model)
    assert msgspec.to_builtins(adapter.extract(items, labels=LABELS)) == baseline
    adapter.load_lora("a")
    adapted = msgspec.to_builtins(adapter.extract(items, labels=LABELS, options={"lora": "a"}))
    assert adapted["entities"] != baseline["entities"]
    assert adapted["errors"] == baseline["errors"]
    assert adapted["input_token_counts"] == baseline["input_token_counts"]
    assert msgspec.to_builtins(adapter.extract(items, labels=LABELS)) == baseline
    adapter.unload()


def test_inference_failure_resets_adapter_and_preserves_original_error(raw_model, tmp_path, monkeypatch):
    adapter = adapter_for(raw_model, {"a": make_checkpoint(tmp_path, raw_model, "a")})
    baseline = serve(adapter, None)
    adapter.load_lora("a")
    original = raw_model.batch_extract_entities
    monkeypatch.setattr(raw_model, "batch_extract_entities", Mock(side_effect=RuntimeError("inference sentinel")))
    with pytest.raises(RuntimeError, match="inference sentinel"):
        serve(adapter, "a")
    assert adapter._lora._active is None
    assert_native_base_with_residents(adapter)
    monkeypatch.setattr(raw_model, "batch_extract_entities", original)
    assert serve(adapter, None) == baseline
    adapter.unload()


def test_failed_source_cleanup_poisoning_retains_file_ownership_until_teardown(raw_model, tmp_path, monkeypatch):
    path = make_checkpoint(tmp_path, raw_model, "a")
    adapter = adapter_for(raw_model, {"a": path})
    before = snapshot(raw_model)
    source = adapter.fetch_lora("a")
    directory = adapter._lora._prepared[path].directory
    original = directory.cleanup
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("owned cleanup sentinel")
        original()

    monkeypatch.setattr(directory, "cleanup", fail_once)
    with pytest.raises(LoraStateUncertainError, match="cleanup"):
        adapter.load_lora("a", source=source)
    assert Path(source).exists()
    assert path in adapter._lora._prepared
    for name in (None, "a"):
        with pytest.raises(LoraStateUncertainError, match="uncertain"):
            serve(adapter, name)
    adapter.unload()
    assert not Path(source).exists()
    assert_restored(raw_model, before)


def test_failed_second_rollback_poisoning_is_cleared_only_by_model_teardown(raw_model, tmp_path, monkeypatch):
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b")}
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    adapter.load_lora("a")
    original = PeftModel.load_adapter

    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected second failure")

    monkeypatch.setattr(PeftModel, "load_adapter", fail)
    monkeypatch.setattr(adapter._lora, "_remove", Mock(side_effect=RuntimeError("rollback sentinel")))
    with pytest.raises(LoraStateUncertainError, match="rollback"):
        adapter.load_lora("b")
    for name in (None, "a", "b"):
        with pytest.raises(LoraStateUncertainError, match="uncertain"):
            serve(adapter, name)
    adapter.unload()
    assert_restored(raw_model, before)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_type", "SEQ_CLS"),
        ("bias", "all"),
        ("modules_to_save", ["count_pred"]),
        ("init_lora_weights", "pissa"),
        ("rank_pattern", {"query": 8}),
        ("alpha_pattern", {"query": 16}),
        ("layer_replication", [[0, 1]]),
        ("trainable_token_indices", [1]),
        ("use_dora", True),
        ("target_modules", ["count_pred"]),
        ("target_modules", ["encoder.pooler.dense"]),
    ],
)
def test_unsafe_adapter_config_rejected_without_injection(raw_model, tmp_path, monkeypatch, field, value):
    path = make_checkpoint(tmp_path, raw_model, "a")
    config_path = Path(path) / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))
    adapter = adapter_for(raw_model, {"a": path})
    before = snapshot(raw_model)
    inject = Mock(side_effect=AssertionError("invalid config reached PEFT"))
    monkeypatch.setattr(PeftModel, "from_pretrained", inject)
    with pytest.raises(ValueError, match=r"LoRA|PEFT"):
        adapter.load_lora("a")
    inject.assert_not_called()
    assert adapter._lora._prepared == {}
    assert_restored(raw_model, before)
    adapter.unload()


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "integer", "nan", "overflow"])
def test_invalid_tensor_rejected_without_injection(raw_model, tmp_path, monkeypatch, fault):
    path = make_checkpoint(tmp_path, raw_model, "a")
    weight_path = Path(path) / "adapter_model.safetensors"
    weights = load_file(str(weight_path))
    first = next(iter(weights))
    if fault == "missing":
        del weights[first]
    elif fault == "extra":
        weights["base_model.model.count_pred.weight"] = torch.zeros(1)
    elif fault == "shape":
        weights[first] = weights[first][:-1]
    elif fault == "integer":
        weights[first] = weights[first].to(torch.int32)
    else:
        weights[first].fill_(float("nan") if fault == "nan" else 1e30)
    save_file(weights, weight_path)
    if fault == "overflow":
        raw_model.half()
    adapter = adapter_for(raw_model, {"a": path})
    before = snapshot(raw_model)
    inject = Mock(side_effect=AssertionError("invalid tensors reached PEFT"))
    monkeypatch.setattr(PeftModel, "from_pretrained", inject)
    with pytest.raises(ValueError, match="LoRA"):
        adapter.load_lora("a")
    inject.assert_not_called()
    assert adapter._lora._prepared == {}
    assert_restored(raw_model, before)
    adapter.unload()


def test_declared_pin_and_owned_snapshot_survive_source_changes(raw_model, tmp_path, monkeypatch):
    original = make_checkpoint(tmp_path, raw_model, "a")
    expected_weights = load_file(str(Path(original) / "adapter_model.safetensors"))
    pin = "a" * 40
    adapter = GLiNER2EntityLoraAdapter(
        "generated-local",
        lora_paths={"a": "org/a", "same": "org/a"},
        lora_revisions={"org/a": pin},
        compute_precision="float32",
    )
    adapter._model = raw_model
    adapter._lora.bind(raw_model)
    fetch = Mock(return_value=original)
    monkeypatch.setattr(_lora, "snapshot_download", fetch)
    with pytest.raises(ValueError, match="revision"):
        adapter.fetch_lora("a", "b" * 40)
    with pytest.raises(ValueError, match="declared"):
        adapter.fetch_lora("undeclared")
    fetch.assert_not_called()
    source = adapter.fetch_lora("same", pin)
    assert source != original
    fetch.assert_called_once_with(
        "org/a", revision=pin, allow_patterns=["adapter_config.json", "adapter_model.safetensors"]
    )
    (Path(original) / "adapter_config.json").write_text("{}")
    (Path(original) / "adapter_model.safetensors").write_bytes(b"broken")
    with pytest.raises(ValueError, match="revision"):
        adapter.load_lora("a", "b" * 40, source=source)
    size = adapter.load_lora("same", pin, source=source)
    assert size == sum(t.numel() * t.element_size() for t in expected_weights.values())
    assert not Path(source).exists()
    assert adapter.load_lora("a", pin) == size
    assert adapter._lora._prepared == {}
    fetch.assert_called_once()
    adapter.unload()


def test_capacity_failure_discards_only_owned_unattached_snapshot(raw_model, tmp_path):
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b")}
    adapter = adapter_for(raw_model, paths)
    adapter._lora._max_loaded = 1
    baseline = serve(adapter, None)
    adapter.load_lora("a")
    a = serve(adapter, "a")
    source = adapter.fetch_lora("b")
    with pytest.raises(ValueError, match="capacity"):
        adapter.load_lora("b", source=source)
    assert not Path(source).exists()
    assert adapter._lora._prepared == {}
    assert serve(adapter, None) == baseline
    assert serve(adapter, "a") == a
    adapter.unload()


def test_fetch_after_teardown_cannot_attach_to_rebound_model(raw_model, tmp_path, monkeypatch):
    original = make_checkpoint(tmp_path, raw_model, "a")
    adapter = adapter_for(raw_model, {"a": "org/a"})
    entered, release = threading.Event(), threading.Event()

    def fetch(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original

    monkeypatch.setattr(_lora, "snapshot_download", fetch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.fetch_lora, "a")
        try:
            assert entered.wait(3)
            adapter._lora.close()
            adapter._lora.bind(raw_model)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="changed"):
            future.result(timeout=5)
    assert adapter._lora._prepared == {}
    assert adapter._lora._loaded == {}
    assert not any(isinstance(m, LoraLinear) for m in raw_model.modules())
    adapter.unload()


def test_concurrent_direct_requests_serialize_actual_selection(raw_model, tmp_path):
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b")}
    adapter = adapter_for(raw_model, paths)
    for name in paths:
        adapter.load_lora(name)
    expected = {name: serve(adapter, name) for name in (None, "a", "b")}
    assert len({json.dumps(result, sort_keys=True) for result in expected.values()}) == 3
    names = [None, "a", "b"] * 4
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(lambda name: serve(adapter, name), names))
    assert actual == [expected[name] for name in names]
    assert adapter._lora._active is None
    adapter.unload()


def config_for(paths):
    return ModelConfig.model_validate(
        {
            "sie_id": "entity",
            "hf_id": "generated-local",
            "tasks": {"extract": {}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.gliner2.entity_lora:GLiNER2EntityLoraAdapter",
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": {"lora_paths": paths}},
                }
            },
        }
    )


@pytest.mark.asyncio
async def test_real_loader_registry_deferred_startup_mixed_worker_and_lru(raw_model, tmp_path, monkeypatch):
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b", "c")}
    adapter = adapter_for(raw_model, paths)
    fetch = Mock(wraps=adapter.fetch_lora)
    monkeypatch.setattr(adapter, "fetch_lora", fetch)
    registry = ModelRegistry()
    config = config_for(paths)
    loader = ModelLoader(
        preprocessor_registry=MagicMock(),
        postprocessor_registry=MagicMock(),
        all_configs={"entity": config},
        registry_callbacks=registry,
    )
    loaded = loader._finish_load("entity", "cpu", adapter, config)
    registry._configs["entity"] = config
    registry._loaded["entity"] = loaded
    assert loaded.max_loras == 2
    assert loaded.loras == {}
    fetch.assert_not_called()
    worker = loaded.worker
    await worker.start()
    try:
        for name in ("a", "b"):
            assert await registry.ensure_lora_loaded_async("entity", name) == (False, True)
            await asyncio.wait_for(asyncio.gather(*registry._background_tasks), 5)
        expected = {name: serve(adapter, name) for name in (None, "a", "b")}
        names = [None, "a", "b"] * 2

        async def submit(name):
            items = [Item(text=text) for text in TEXTS]
            futures = await worker.submit_extract_preformed_batch(
                [
                    PreformedExtractRequest(
                        build_extract_prepared_items(items),
                        items,
                        labels=LABELS,
                        options={"lora": name},
                    )
                ],
                lora=name,
            )
            return msgspec.to_builtins((await futures[0]).output)

        actual = await asyncio.gather(*(submit(name) for name in names))
        assert actual == [expected[name] for name in names]
        assert await registry.ensure_lora_loaded_async("entity", "c") == (False, True)
        await asyncio.wait_for(asyncio.gather(*registry._background_tasks), 5)
        assert len(loaded.loras) == len(adapter._lora._loaded) == 2
        assert paths["c"] in adapter._lora._loaded
        assert serve(adapter, None) == expected[None]
    finally:
        await worker.stop()
        adapter.unload()


def test_real_sdk_cold_load_and_concurrent_named_base_isolation(raw_model, tmp_path, monkeypatch):
    paths = {name: make_checkpoint(tmp_path, raw_model, name) for name in ("a", "b")}
    adapter = adapter_for(raw_model, paths)
    before = snapshot(raw_model)
    # Untimed evidence joins each wire item to its actual combined serving
    # batch. Different CPU batch shapes can legitimately round scores
    # differently; never compare against a guessed shape or an oracle envelope.
    observations = []
    current = threading.local()
    original_extract = adapter.extract
    original_batch = raw_model.batch_extract_entities
    original_forward = raw_model.encoder.forward

    def observed_extract(items, **kwargs):
        record = {
            "ids": [item.id for item in items],
            "texts": [item.text for item in items],
            "selection": (kwargs.get("options") or {}).get("lora"),
            "batches": [],
            "forwards": [],
        }
        assert not hasattr(current, "record")
        current.record = record
        try:
            return original_extract(items, **kwargs)
        finally:
            observations.append(record)
            del current.record

    def observed_batch(texts, *args, **kwargs):
        current.record["batches"].append({"texts": list(texts), "source": adapter._lora._active})
        return original_batch(texts, *args, **kwargs)

    def observed_forward(*args, **kwargs):
        assert not args
        current.record["forwards"].append(
            {
                "inputs": {key: value.tolist() for key, value in kwargs.items() if isinstance(value, torch.Tensor)},
                "source": adapter._lora._active,
                "native_projections": all(
                    raw_model.get_submodule(name) is original for name, original in adapter._lora._originals.items()
                ),
                "tuners": {
                    name: [] if layer.disable_adapters else list(layer.active_adapters)
                    for name, layer in raw_model.named_modules()
                    if isinstance(layer, LoraLinear)
                },
            }
        )
        return original_forward(**kwargs)

    monkeypatch.setattr(adapter, "extract", observed_extract)
    monkeypatch.setattr(raw_model, "batch_extract_entities", observed_batch)
    monkeypatch.setattr(raw_model.encoder, "forward", observed_forward)
    fetch = Mock(wraps=adapter.fetch_lora)
    monkeypatch.setattr(adapter, "fetch_lora", fetch)
    registry = ModelRegistry()
    config = config_for(paths)
    loader = ModelLoader(
        preprocessor_registry=MagicMock(),
        postprocessor_registry=MagicMock(),
        all_configs={"entity": config},
        registry_callbacks=registry,
    )
    loaded = loader._finish_load("entity", "cpu", adapter, config)
    registry._configs["entity"] = config
    registry._loaded["entity"] = loaded

    @asynccontextmanager
    async def lifespan(app):
        await loaded.worker.start()
        try:
            yield
        finally:
            await loaded.worker.stop()
            adapter.unload()

    async def finish_loads():
        await asyncio.wait_for(asyncio.gather(*registry._background_tasks), 5)

    app = FastAPI(lifespan=lifespan)
    app.include_router(extract_router)
    app.state.registry = registry
    statuses = []
    with TestClient(app, base_url="http://127.0.0.1", follow_redirects=False) as transport:
        transport.event_hooks["response"].append(lambda response: statuses.append(response.status_code))
        with SIEClient("http://127.0.0.1", http_client=transport) as client:

            def request(name, prefix="cold", repeats=1):
                return client.extract(
                    "entity",
                    [{"id": f"{prefix}-{index}", "text": text} for index, text in enumerate(TEXTS * repeats)],
                    labels=LABELS,
                    options={"lora": name},
                    provision_timeout_s=5,
                )

            expected = {None: request(None)}
            fetch.assert_not_called()
            for name in paths:
                # Existing extract SDK behavior is a ServerError. Make a new
                # call after readiness, without changing its retry policy.
                with pytest.raises(ServerError) as loading:
                    request(name)
                assert loading.value.status_code == 503
                assert loading.value.code == "LORA_LOADING"
                assert loading.value.retry_after > 0
                transport.portal.call(finish_loads)
                expected[name] = request(name)
            assert statuses == [200, 503, 200, 503, 200]
            assert expected["a"] != expected[None]
            assert expected["b"] != expected[None]
            assert expected["a"] != expected["b"]
            oracles = {}
            for name in (None, "a", "b"):
                for rows in (2, 4, 6):
                    start = len(observations)
                    result = request(name, f"oracle-{name}-{rows}", repeats=rows // 2)
                    records = observations[start:]
                    assert len(records) == 1
                    record = records[0]
                    result_by_id = {item["id"]: item for item in result}
                    assert set(record["ids"]) == set(result_by_id)
                    assert record["texts"] == [TEXTS[int(item_id.rsplit("-", 1)[1]) % 2] for item_id in record["ids"]]
                    assert record["batches"]
                    assert record["forwards"]
                    assert [text for batch in record["batches"] for text in batch["texts"]] == record["texts"]
                    source = paths[name] if name is not None else None
                    assert all(batch["source"] == source for batch in record["batches"])
                    assert all(forward["source"] == source for forward in record["forwards"])
                    active = [] if source is None else [PEFTLoRAMixin._peft_adapter_name(source)]
                    for forward in record["forwards"]:
                        if source is None:
                            assert forward["tuners"] == {}
                            assert forward["native_projections"]
                        else:
                            assert forward["tuners"]
                            assert not forward["native_projections"]
                            assert all(names == active for names in forward["tuners"].values())
                    oracles[name, rows] = {
                        "record": record,
                        "result": [result_by_id[item_id] for item_id in record["ids"]],
                    }
            statuses.clear()
            choices = [None, "a", "b"] * 4
            start = len(observations)
            with ThreadPoolExecutor(max_workers=3) as pool:
                actual = list(pool.map(lambda pair: request(pair[1], f"request-{pair[0]}"), enumerate(choices)))
            records = observations[start:]
            (tmp_path / "sdk-concurrent-results.json").write_text(
                json.dumps(
                    {
                        "choices": choices,
                        "actual": actual,
                        "observations": records,
                        "oracles": [
                            {"selection": name, "rows": rows, **oracle} for (name, rows), oracle in oracles.items()
                        ],
                    },
                    indent=2,
                )
            )
            expected_by_id = {}
            for record in records:
                ids = record["ids"]
                assert len(ids) in (2, 4, 6)
                selected = record["selection"]
                assert all(choices[int(item_id.split("-")[1])] == selected for item_id in ids)
                oracle = oracles[selected, len(ids)]
                assert record["texts"] == oracle["record"]["texts"]
                # Includes actual internal split order, tensor inputs/shapes,
                # canonical selection and effective tuner state, not just rows.
                assert record["batches"] == oracle["record"]["batches"]
                assert record["forwards"] == oracle["record"]["forwards"]
                for item_id, result in zip(ids, oracle["result"], strict=True):
                    assert item_id not in expected_by_id
                    expected_by_id[item_id] = {**result, "id": item_id}
            assert len(expected_by_id) == 2 * len(choices)
            assert actual == [
                [expected_by_id[f"request-{index}-{row}"] for row in range(2)] for index in range(len(choices))
            ]
            assert statuses == [200] * len(choices)
            assert fetch.call_count == 2
    assert_restored(raw_model, before)
