"""Real BERT/XLM-R/Transformers/PEFT lifecycle checks without model downloads."""

from __future__ import annotations

import asyncio
import builtins
import gc
import json
import socket
import subprocess
import sys
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
import torch
import yaml
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file, save_file
from sie_sdk import SIEClient
from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
from sie_server.adapters.cross_encoder.runtime_lora import CrossEncoderLoraAdapter
from sie_server.adapters.lora import LoraStateUncertainError, LoraUnavailableError
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.model import ModelConfig
from sie_server.core.loader import load_adapter
from sie_server.core.prepared import ScorePreparedItem
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.model_worker import PreformedScoreRequest
from sie_server.types.inputs import Item
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
)

QUERY = {"text": "how do i reset my password"}
DOCS = [{"text": "reset your password by email"}, {"text": "the capital is paris"}]


@pytest.fixture(scope="module", params=("bert", "xlmr"))
def checkpoint(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp(f"cross-encoder-{request.param}")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        *["how", "do", "i", "reset", "my", "password", "your", "by", "email", "the", "capital", "is", "paris"],
    ]
    vocab = list(dict.fromkeys(vocab))
    common = {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 96,
        "num_labels": 1,
        "hidden_dropout_prob": 0,
        "attention_probs_dropout_prob": 0,
    }
    if request.param == "bert":
        vocab_path = directory / "vocab.txt"
        vocab_path.write_text("\n".join(vocab))
        BertTokenizerFast(vocab_file=str(vocab_path), model_max_length=64).save_pretrained(directory)
        config = BertConfig(vocab_size=len(vocab), **{**common, "max_position_embeddings": 64})
        model_class = BertForSequenceClassification
    else:
        # Local WordLevel fixture with XLM-R pair boundaries; no SentencePiece or
        # pretrained checkpoint/adapter quality is being qualified by these tests.
        vocab[:5] = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        vocab.extend(["change", "london", "a", "city"])
        bos, pad, eos, unk, mask = vocab[:5]
        backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocab)}, unk_token=unk))
        backend.normalizer = normalizers.Lowercase()
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[("<s>", 0), ("</s>", 2)],
        )
        PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token=bos,
            pad_token=pad,
            eos_token=eos,
            unk_token=unk,
            mask_token=mask,
            model_max_length=64,
            model_input_names=["input_ids", "attention_mask"],
        ).save_pretrained(directory)
        config = XLMRobertaConfig(
            vocab_size=len(vocab),
            type_vocab_size=1,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            classifier_dropout=0,
            **{**common, "hidden_size": 32, "num_attention_heads": 4, "intermediate_size": 64},
        )
        model_class = XLMRobertaForSequenceClassification
    config.sbert_ce_default_activation_function = "torch.nn.modules.linear.Identity"
    with torch.random.fork_rng():
        torch.manual_seed(29 if request.param == "bert" else 20261017)
        model_class(config).save_pretrained(directory)
    return directory


def encoder_prefix(checkpoint):
    model_type = json.loads((checkpoint / "config.json").read_text())["model_type"]
    return {"bert": "bert.encoder", "xlm-roberta": "roberta.encoder"}[model_type]


def fixture_lora_bytes(checkpoint, *, targets=2):
    config = json.loads((checkpoint / "config.json").read_text())
    # Each generated projection has two float32 rank-4 matrices.
    return config["num_hidden_layers"] * targets * 2 * 4 * config["hidden_size"] * 4


def save_adapter(checkpoint, path, seed=1, *, targets=("query", "value"), **overrides):
    path.mkdir()
    LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=8,
        target_modules=list(targets),
        bias="none",
        lora_dropout=0,
        inference_mode=True,
    ).save_pretrained(path)
    if overrides:
        config_path = path / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config.update(overrides)
        config_path.write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(seed)
    weights = {}
    hidden_size = json.loads((checkpoint / "config.json").read_text())["hidden_size"]
    for layer in range(2):
        for target in targets:
            prefix = f"base_model.model.{encoder_prefix(checkpoint)}.layer.{layer}.attention.self.{target}"
            weights[f"{prefix}.lora_A.weight"] = torch.randn(4, hidden_size, generator=generator) * 0.4
            weights[f"{prefix}.lora_B.weight"] = torch.randn(hidden_size, 4, generator=generator) * 0.4
    save_file(weights, path / "adapter_model.safetensors")
    return str(path)


@pytest.fixture
def adapter(checkpoint, tmp_path):
    paths = {name: save_adapter(checkpoint, tmp_path / name, seed) for seed, name in enumerate(("a", "b", "c"), 1)}
    adapter = CrossEncoderLoraAdapter(checkpoint, lora_paths=paths, max_loaded_loras=2)
    adapter.load("cpu")
    yield adapter
    adapter.unload()


def scores(adapter):
    query, docs = Item(**QUERY), [Item(**doc) for doc in DOCS]
    direct = np.asarray(adapter.score(query, docs))
    pairs = adapter.score_pairs([query] * len(docs), docs)
    assert np.isfinite(direct).all()
    np.testing.assert_allclose(pairs.scores, direct, atol=1e-7, rtol=0)
    assert pairs.input_token_counts
    assert all(count > 0 for count in pairs.input_token_counts)
    return direct


def base_snapshot(adapter):
    outer = adapter._model
    return (
        outer,
        outer.model,
        outer.model.classifier,
        outer.tokenizer,
        outer.activation_fn,
        dict(outer.model.named_modules()),
        {name: value.detach().clone() for name, value in outer.model.state_dict().items()},
        [(parameter, parameter.requires_grad) for parameter in outer.model.parameters()],
        [(module, module.training) for module in outer.model.modules()],
    )


def assert_base_preserved(adapter, snapshot, *, unwrapped=False):
    outer, model, head, tokenizer, activation, modules, weights, grad, training = snapshot
    assert adapter._model is outer
    assert outer.model is model
    assert model.classifier is head
    assert outer.tokenizer is tokenizer
    assert outer.activation_fn is activation
    for name, original in modules.items():
        if name == "classifier" or name.startswith("classifier."):
            assert model.get_submodule(name) is original, name
    for name, expected in weights.items():
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(parent)
        if isinstance(module, BaseTunerLayer):
            module = module.get_base_layer()
        assert torch.equal(getattr(module, leaf), expected), name
    assert all(parameter.requires_grad == flag for parameter, flag in grad)
    assert all(module.training == flag for module, flag in training)
    if unwrapped:
        assert dict(model.named_modules()) == modules
        assert not hasattr(model, "peft_config")
        assert not model._hf_peft_config_loaded


def test_never_lora_has_no_manager_or_optional_imports(checkpoint, monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in {"sie_server.adapters.cross_encoder._lora", "sie_server.adapters.cross_encoder.runtime_lora"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    adapter = CrossEncoderAdapter(checkpoint)
    assert not hasattr(adapter, "_lora")
    assert not adapter.supports_lora()
    assert adapter.lora_capability() is None
    assert not adapter.supports_hot_lora_reload()
    assert adapter.lora_runtime_config() is None
    adapter.load("cpu")
    before = base_snapshot(adapter)
    baseline = scores(adapter)
    adapter.set_active_lora(None)
    np.testing.assert_array_equal(scores(adapter), baseline)
    assert_base_preserved(adapter, before)
    # Preserve the original base hook contract; registry capability checks
    # reject adapter requests before any model mutation.
    adapter.set_active_lora("a")
    np.testing.assert_array_equal(scores(adapter), baseline)
    adapter.unload()


def test_ordinary_module_import_and_hooks_stay_independent_of_runtime_lora(tmp_path):
    script = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name not in {
        'sie_server.adapters.cross_encoder.runtime_lora',
        'sie_server.adapters.cross_encoder._lora',
    }, name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
adapter = CrossEncoderAdapter('unused-local-construction-only', lora_paths={'a': 'unused'})
assert not hasattr(adapter, '_lora')
for name in ('unload', 'supports_lora', 'supports_hot_lora_reload', 'lora_runtime_config',
             'fetch_lora', 'discard_lora_source', 'load_lora', 'unload_lora', 'set_active_lora'):
    assert getattr(CrossEncoderAdapter, name) is getattr(BaseAdapter, name), name
assert not adapter.supports_lora()
assert 'sie_server.adapters.cross_encoder.runtime_lora' not in sys.modules
assert 'sie_server.adapters.cross_encoder._lora' not in sys.modules
"""
    result = subprocess.run(  # noqa: S603 -- fixed local import-only check, no model or network
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=45, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("declarations", [None, {}, []])
def test_explicit_runtime_class_requires_declarations(checkpoint, declarations):
    with pytest.raises(ValueError, match="requires nonempty lora_paths"):
        CrossEncoderLoraAdapter(checkpoint, lora_paths=declarations, lora_revisions={"org/other": "a" * 40})


@pytest.mark.parametrize("runtime", [False, True])
def test_real_loader_routes_only_the_explicit_profile_and_preserves_base_options(checkpoint, runtime):
    adapter_type = CrossEncoderLoraAdapter if runtime else CrossEncoderAdapter
    loadtime = {"max_length": 24, "trust_remote_code": True}
    if runtime:
        loadtime["lora_paths"] = {"a": {"id": "org/declared", "revision": "b" * 40}}
    config = ModelConfig.model_validate(
        {
            "sie_id": "test/explicit-cross-encoder",
            "weights_path": str(checkpoint),
            "hf_revision": "a" * 40,
            "inputs": {"text": True},
            "tasks": {"score": {}},
            "profiles": {
                "default": {
                    "adapter_path": f"{adapter_type.__module__}:{adapter_type.__name__}",
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": loadtime},
                }
            },
        }
    )
    adapter = load_adapter(config, checkpoint, device="cpu", attention_backend="eager", max_loras_per_model=2)
    assert type(adapter) is adapter_type
    assert adapter._model_name_or_path == str(checkpoint)
    assert adapter._max_length == 24
    assert adapter._trust_remote_code
    assert adapter._compute_precision == "float32"
    assert adapter._attn_implementation == "eager"
    assert adapter._revision == "a" * 40
    assert adapter_type.score is CrossEncoderAdapter.score
    assert adapter_type.score_pairs is CrossEncoderAdapter.score_pairs
    if runtime:
        assert adapter._lora._revisions == {"org/declared": "b" * 40}
        assert adapter.lora_runtime_config().max_loaded == 2
        assert adapter.lora_runtime_config().aliases == {"a": "org/declared"}
    else:
        assert not hasattr(adapter, "_lora")
        assert adapter.set_active_lora.__func__ is BaseAdapter.set_active_lora
        # Profile resolution is cached. Construct a distinct invalid config,
        # rather than mutating the already-resolved instance behind its cache.
        invalid = config.model_dump()
        invalid["profiles"]["default"]["adapter_options"]["loadtime"]["lora_paths"] = {"a": "org/declared"}
        with pytest.raises(ValueError, match=r"does not accept.*lora_paths"):
            load_adapter(ModelConfig.model_validate(invalid), checkpoint, device="cpu")


def test_pinned_cluster_adapter_preserves_lifecycle_with_network_denied(checkpoint, tmp_path, monkeypatch):
    cluster = tmp_path / "cluster"
    revision = "a" * 40
    cached = cluster / "models--org--adapter" / "snapshots" / revision
    cached.parent.mkdir(parents=True)
    save_adapter(checkpoint, cached)
    monkeypatch.setenv("SIE_LOCAL_CACHE", str(tmp_path / "local"))
    monkeypatch.setenv("SIE_CLUSTER_CACHE", str(cluster))
    monkeypatch.setenv("SIE_HF_FALLBACK", "false")
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))
    monkeypatch.setattr("huggingface_hub.snapshot_download", Mock(side_effect=AssertionError("Hub forbidden")))
    adapter = CrossEncoderLoraAdapter(
        checkpoint, lora_paths={"a": "org/adapter"}, lora_revisions={"org/adapter": revision}, max_loaded_loras=1
    )
    adapter.load("cpu")
    try:
        baseline = scores(adapter)
        original = base_snapshot(adapter)
        prepared = adapter.fetch_lora("a")
        adapter.load_lora("a", source=prepared)
        adapter.set_active_lora("a")
        assert not np.array_equal(scores(adapter), baseline)
        adapter.set_active_lora(None)
        np.testing.assert_array_equal(scores(adapter), baseline)
        adapter.unload_lora("a")
        assert_base_preserved(adapter, original, unwrapped=True)
    finally:
        adapter.unload()
    assert not Path(prepared).exists()


def test_actual_effect_restoration_head_and_final_module_identity(adapter, checkpoint):
    snapshot = base_snapshot(adapter)
    baseline = scores(adapter)
    np.testing.assert_array_equal(scores(adapter), baseline)
    runtime = adapter.lora_runtime_config()
    assert runtime is not None
    assert runtime.max_loaded == 2
    assert set(runtime.aliases) == {"a", "b", "c"}
    assert not runtime.preloaded
    assert not runtime.pinned
    with pytest.raises(LoraUnavailableError):
        adapter.set_active_lora("a")
    assert adapter.load_lora("a") == fixture_lora_bytes(checkpoint)
    # Successful attachment must leave base selected, including SEQ_CLS configs.
    np.testing.assert_array_equal(scores(adapter), baseline)
    adapter.set_active_lora("a")
    a = scores(adapter)
    assert not any(parameter.requires_grad for parameter in adapter._lora._parameters[adapter._lora._aliases["a"]])
    assert np.max(np.abs(a - baseline)) > 1e-6
    assert_base_preserved(adapter, snapshot)
    adapter.load_lora("b")
    np.testing.assert_array_equal(scores(adapter), baseline)
    adapter.set_active_lora("b")
    b = scores(adapter)
    assert np.max(np.abs(b - a)) > 1e-6
    adapter.set_active_lora(None)
    np.testing.assert_array_equal(scores(adapter), baseline)
    with pytest.raises(ValueError, match="capacity"):
        adapter.load_lora("c")
    adapter.unload_lora("a")
    with pytest.raises(LoraUnavailableError):
        adapter.set_active_lora("a")
    adapter.set_active_lora("b")
    np.testing.assert_array_equal(scores(adapter), b)
    adapter.unload_lora("b")
    assert_base_preserved(adapter, snapshot, unwrapped=True)
    np.testing.assert_array_equal(scores(adapter), baseline)
    adapter.load_lora("a")
    adapter.set_active_lora("a")
    np.testing.assert_array_equal(scores(adapter), a)


def test_resident_adapters_leave_the_exact_base_tree_and_preserve_layer_hooks(adapter, checkpoint):
    snapshot = base_snapshot(adapter)
    baseline = scores(adapter)
    prefix = encoder_prefix(checkpoint)
    original_query = snapshot[1].get_submodule(f"{prefix}.layer.0.attention.self.query")
    original_storage = [(parameter, parameter.data_ptr()) for parameter in snapshot[1].parameters()]
    original_calls, wrapper_calls = [], []
    base_hook = original_query.register_forward_hook(lambda *_: original_calls.append(True))
    adapter.load_lora("a")
    manager = adapter._lora
    assert dict(snapshot[1].named_modules()) == snapshot[5]
    wrapper = manager._resident_layers[f"{prefix}.layer.0.attention.self.query"]

    def observed_wrapper(module, inputs, output):
        assert inputs[0].shape[0] > 0
        assert torch.isfinite(output).all()
        assert not module.training
        wrapper_calls.append((tuple(module.active_adapters), module.disable_adapters))

    wrapper_hook = wrapper.register_forward_hook(observed_wrapper)
    adapter.load_lora("b")
    identities = {name: id(layer) for name, layer in manager._resident_layers.items()}
    assert dict(snapshot[1].named_modules()) == snapshot[5]
    assert len(manager._loaded) == 2
    assert sum(manager._loaded.values()) == 2 * fixture_lora_bytes(checkpoint)
    expected = {None: baseline}
    for selection in ("a", None, "b", None, "a", None, "b", None):
        old_wrapper_calls = len(wrapper_calls)
        old_original_calls = len(original_calls)
        adapter.set_active_lora(selection)
        actual = scores(adapter)
        assert len(original_calls) > old_original_calls
        if selection is None:
            assert dict(snapshot[1].named_modules()) == snapshot[5]
            assert len(wrapper_calls) == old_wrapper_calls
        else:
            assert len(wrapper_calls) > old_wrapper_calls
            expected_name = PEFTLoRAMixin._peft_adapter_name(manager._aliases[selection])
            assert all(
                names == (expected_name,) and not disabled for names, disabled in wrapper_calls[old_wrapper_calls:]
            )
            assert all(id(snapshot[1].get_submodule(name)) == identity for name, identity in identities.items())
            assert np.max(np.abs(actual - baseline)) > 1e-6
        if selection in expected:
            np.testing.assert_array_equal(actual, expected[selection])
        expected[selection] = actual
        assert_base_preserved(adapter, snapshot)
        assert all(parameter.data_ptr() == pointer for parameter, pointer in original_storage)
        assert {name: id(layer) for name, layer in manager._resident_layers.items()} == identities
    assert np.max(np.abs(expected["a"] - expected["b"])) > 1e-6
    base_hook.remove()
    wrapper_hook.remove()


def test_repeated_disabled_selection_checks_readiness_without_tree_or_peft_work(adapter, monkeypatch):
    adapter.load_lora("a")
    adapter.load_lora("b")
    adapter.set_active_lora(None)
    baseline = scores(adapter)
    snapshot = base_snapshot(adapter)
    manager = adapter._lora
    original_ready = manager._ready
    checks = []

    def ready():
        checks.append(True)
        original_ready()

    def unexpected(*_args, **_kwargs):
        pytest.fail("disabled selection traversed or changed adapter layers")

    with monkeypatch.context() as context:
        context.setattr(manager, "_ready", ready)
        for name in ("_attach_layers", "_detach_layers", "_restore_grad_flags"):
            context.setattr(manager, name, unexpected)
        for name in ("set_adapter", "enable_adapters", "disable_adapters"):
            context.setattr(adapter._model, name, unexpected)
        for _ in range(100):
            adapter.set_active_lora(None)
        assert len(checks) == 100
        np.testing.assert_array_equal(scores(adapter), baseline)
        assert dict(snapshot[1].named_modules()) == snapshot[5]
        manager._uncertain = True
        with pytest.raises(LoraStateUncertainError):
            adapter.set_active_lora(None)
        manager._uncertain = False


def test_disjoint_targets_evict_only_their_wrappers_and_final_unload_releases_weights(checkpoint, tmp_path):
    paths = {
        "a": save_adapter(checkpoint, tmp_path / "a", 1, targets=("query",)),
        "b": save_adapter(checkpoint, tmp_path / "b", 2, targets=("value",)),
    }
    adapter = CrossEncoderLoraAdapter(checkpoint, lora_paths=paths, max_loaded_loras=2)
    adapter.load("cpu")
    try:
        snapshot, baseline = base_snapshot(adapter), scores(adapter)
        adapter.load_lora("a")
        adapter.set_active_lora("a")
        a = scores(adapter)
        adapter.load_lora("b")
        manager = adapter._lora
        refs = {name: weakref.ref(layer) for name, layer in manager._resident_layers.items()}
        parameter_refs = [weakref.ref(p) for parameters in manager._parameters.values() for p in parameters]
        assert dict(snapshot[1].named_modules()) == snapshot[5]
        assert set(refs) == {
            f"{encoder_prefix(checkpoint)}.layer.{i}.attention.self.{target}"
            for i in range(2)
            for target in ("query", "value")
        }
        adapter.set_active_lora("b")
        b = scores(adapter)
        assert np.max(np.abs(a - baseline)) > 1e-6
        assert np.max(np.abs(b - baseline)) > 1e-6
        adapter.set_active_lora(None)
        np.testing.assert_array_equal(scores(adapter), baseline)
        adapter.unload_lora("a")
        gc.collect()
        assert all(ref() is None for name, ref in refs.items() if name.endswith("query"))
        assert all(ref() is not None for name, ref in refs.items() if name.endswith("value"))
        assert dict(snapshot[1].named_modules()) == snapshot[5]
        assert sum(manager._loaded.values()) == fixture_lora_bytes(checkpoint, targets=1)
        adapter.set_active_lora("b")
        np.testing.assert_array_equal(scores(adapter), b)
        adapter.unload_lora("b")
        gc.collect()
        assert all(ref() is None for ref in refs.values())
        assert all(ref() is None for ref in parameter_refs)
        assert not manager._resident_layers
        assert_base_preserved(adapter, snapshot, unwrapped=True)
        adapter.load_lora("a")
        adapter.set_active_lora("a")
        np.testing.assert_array_equal(scores(adapter), a)
    finally:
        adapter.unload()


@pytest.mark.parametrize("operation", ["attach", "detach"])
def test_partial_tree_swap_failure_poisoning_preserves_residency_until_teardown(
    adapter, checkpoint, monkeypatch, operation
):
    snapshot = base_snapshot(adapter)
    adapter.load_lora("a")
    adapter.load_lora("b")
    manager = adapter._lora
    if operation == "detach":
        adapter.set_active_lora("a")
    refs = [weakref.ref(layer) for layer in manager._resident_layers.values()]

    def partial_swap():
        name, layer = next(iter(manager._resident_layers.items()))
        parent_name, _, child = name.rpartition(".")
        setattr(
            manager._model.get_submodule(parent_name),
            child,
            layer if operation == "attach" else manager._originals[name],
        )
        raise RuntimeError("injected partial tree swap")

    with monkeypatch.context() as context:
        context.setattr(manager, f"_{operation}_layers", partial_swap)
        with pytest.raises(LoraStateUncertainError, match="Failed to select"):
            adapter.set_active_lora("a" if operation == "attach" else None)
    assert len(manager._loaded) == 2
    assert sum(manager._loaded.values()) == 2 * fixture_lora_bytes(checkpoint)
    with pytest.raises(LoraStateUncertainError, match="uncertain"):
        adapter.set_active_lora(None)
    adapter.unload()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert dict(snapshot[1].named_modules()) == snapshot[5]
    assert not manager._resident_layers
    assert not manager._loaded


def test_external_precision_change_with_detached_adapters_fails_closed(adapter):
    adapter.load_lora("a")
    adapter._model.to(dtype=torch.float64)
    with pytest.raises(LoraStateUncertainError, match="Failed to select"):
        adapter.set_active_lora("a")
    assert len(adapter._lora._loaded) == 1
    with pytest.raises(LoraStateUncertainError, match="uncertain"):
        adapter.set_active_lora(None)


def test_partial_detach_failure_during_load_removes_all_failed_adapter_weights(adapter, monkeypatch):
    snapshot, baseline = base_snapshot(adapter), scores(adapter)
    adapter.load_lora("a")
    adapter.set_active_lora("a")
    a = scores(adapter)
    manager = adapter._lora
    original = manager._detach_layers
    failed = False

    def fail_after_first_detached_layer():
        nonlocal failed
        if failed:
            return original()
        failed = True
        name, layer = next(iter(manager._resident_layers.items()))
        parent_name, _, child = name.rpartition(".")
        setattr(manager._model.get_submodule(parent_name), child, manager._originals[name])
        assert len(layer.lora_A) == 2
        raise RuntimeError("injected failure after detaching one updated layer")

    with monkeypatch.context() as context:
        context.setattr(manager, "_detach_layers", fail_after_first_detached_layer)
        with pytest.raises(RuntimeError, match="after detaching"):
            adapter.load_lora("b")
    assert len(manager._loaded) == 1
    assert all(len(layer.lora_A) == len(layer.lora_B) == 1 for layer in manager._resident_layers.values())
    assert dict(snapshot[1].named_modules()) == snapshot[5]
    np.testing.assert_array_equal(scores(adapter), baseline)
    adapter.set_active_lora("a")
    np.testing.assert_array_equal(scores(adapter), a)
    adapter.load_lora("b")
    assert len(manager._loaded) == 2


@pytest.mark.parametrize("existing", [False, True])
def test_failed_attachment_rolls_back_entire_base_or_only_failed_adapter(adapter, monkeypatch, existing):
    snapshot = base_snapshot(adapter)
    baseline = scores(adapter)
    if existing:
        adapter.load_lora("a")
        adapter.set_active_lora("a")
        a = scores(adapter)
    original = adapter._model.load_adapter

    def fail_after_injection(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected post-load failure")

    with monkeypatch.context() as context:
        context.setattr(adapter._model, "load_adapter", fail_after_injection)
        with pytest.raises(RuntimeError, match="injected post-load"):
            adapter.load_lora("b")
    assert_base_preserved(adapter, snapshot, unwrapped=not existing)
    assert dict(snapshot[1].named_modules()) == snapshot[5]
    np.testing.assert_array_equal(scores(adapter), baseline)
    assert adapter._lora._prepared == {}
    with pytest.raises(LoraUnavailableError):
        adapter.set_active_lora("b")
    if existing:
        adapter.set_active_lora("a")
        np.testing.assert_array_equal(scores(adapter), a)
    adapter.load_lora("b")
    adapter.set_active_lora("b")
    assert np.max(np.abs(scores(adapter) - baseline)) > 1e-6


@pytest.mark.parametrize("damage", ["missing", "empty", "foreign", "shape", "nan", "overflow", "duplicate"])
def test_bad_weights_rejected_before_model_mutation(adapter, damage):
    source = Path(adapter._lora._aliases["a"])
    weights_path = source / "adapter_model.safetensors"
    weights = load_file(weights_path)
    key = next(iter(weights))
    if damage == "missing":
        del weights[key]
    elif damage == "empty":
        weights.clear()
    elif damage == "foreign":
        weights["base_model.model.classifier.weight"] = torch.ones(1, 16)
    elif damage == "shape":
        weights[key] = torch.ones(3, 16)
    elif damage == "nan":
        weights[key][0, 0] = float("nan")
    elif damage == "overflow":
        weights[key] = torch.full(weights[key].shape, 1e300, dtype=torch.float64)
    else:
        weights[key.removeprefix("base_model.model.")] = weights[key].clone()
    save_file(weights, weights_path)
    snapshot = base_snapshot(adapter)
    with pytest.raises(ValueError, match="LoRA"):
        adapter.load_lora("a")
    assert_base_preserved(adapter, snapshot, unwrapped=True)
    assert adapter._lora._prepared == {}
    assert not adapter._lora._loaded


@pytest.mark.parametrize(
    "change",
    [
        {"modules_to_save": ["classifier"]},
        {"modules_to_save": ["classifier", "score"]},
        {"bias": "all"},
        {"target_modules": ["classifier"]},
        {"target_modules": ["classifier.dense"]},
        {"target_modules": ["classifier.out_proj"]},
        {"target_modules": ["out_proj"]},
        {"target_modules": ["dense"]},
        {"target_modules": ["does_not_exist"]},
        {"target_modules": "query"},
        {"init_lora_weights": "pissa"},
        {"use_dora": True},
        {"rank_pattern": {"query": 2}},
        {"target_parameters": ["classifier.weight"]},
        {"lora_alpha": float("inf")},
    ],
)
def test_unsafe_or_unqualified_config_is_rejected(adapter, change):
    path = Path(adapter._lora._aliases["a"]) / "adapter_config.json"
    config = json.loads(path.read_text())
    config.update(change)
    path.write_text(json.dumps(config))
    snapshot = base_snapshot(adapter)
    with pytest.raises(ValueError, match="LoRA"):
        adapter.load_lora("a")
    assert_base_preserved(adapter, snapshot, unwrapped=True)


def test_only_declared_private_snapshot_can_attach(adapter, checkpoint, tmp_path):
    unknown = save_adapter(checkpoint, tmp_path / "unknown")
    with pytest.raises(ValueError, match="not declared"):
        adapter.fetch_lora(unknown)
    with pytest.raises(ValueError, match="validated snapshot"):
        adapter.load_lora("a", source=unknown)
    prepared = Path(adapter.fetch_lora("a"))
    assert prepared.is_dir()
    assert prepared != Path(adapter._lora._aliases["a"])
    (Path(adapter._lora._aliases["a"]) / "adapter_model.safetensors").write_bytes(b"corrupted after validation")
    assert adapter.load_lora("a", source=str(prepared)) > 0
    assert not prepared.exists()


def test_uncertain_mutation_refuses_even_base_until_whole_model_unload(adapter, monkeypatch):
    adapter.load_lora("a")
    snapshot = base_snapshot(adapter)

    def fail(*args, **kwargs):
        raise RuntimeError("injected rollback failure")

    with monkeypatch.context() as context:
        context.setattr(adapter._model, "load_adapter", fail)
        context.setattr(adapter._lora, "_remove", fail)
        with pytest.raises(LoraStateUncertainError):
            adapter.load_lora("b")
    with pytest.raises(LoraStateUncertainError):
        adapter.set_active_lora(None)
    adapter.unload()
    assert not any(isinstance(module, BaseTunerLayer) for module in snapshot[1].modules())
    assert adapter._lora._outer is None
    assert adapter._lora._originals == {}
    adapter.load("cpu")
    adapter.load_lora("a")
    adapter.set_active_lora("a")
    assert np.isfinite(scores(adapter)).all()


def test_discard_and_model_unload_release_owned_sources(adapter):
    a = Path(adapter.fetch_lora("a"))
    adapter.discard_lora_source("a")
    assert not a.exists()
    b = Path(adapter.fetch_lora("b"))
    model = adapter._model.model
    adapter.unload()
    assert not b.exists()
    assert not adapter._lora._prepared
    assert not any(isinstance(module, BaseTunerLayer) for module in model.modules())
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.fetch_lora("a")


def test_revision_and_capacity_validation(checkpoint, tmp_path):
    path = save_adapter(checkpoint, tmp_path / "a")
    with pytest.raises(ValueError, match="positive integer"):
        CrossEncoderLoraAdapter(checkpoint, lora_paths={"a": path}, max_loaded_loras=0)
    adapter = CrossEncoderLoraAdapter(checkpoint, lora_paths={"a": path}, lora_revisions={path: "a" * 40})
    adapter.load("cpu")
    try:
        with pytest.raises(ValueError, match="differs"):
            adapter.fetch_lora("a", revision="b" * 40)
        with pytest.raises(ValueError, match="local LoRA"):
            adapter.fetch_lora("a")
    finally:
        adapter.unload()


def test_revisions_alone_do_not_enable_or_authorize_lora(checkpoint):
    adapter = CrossEncoderAdapter(checkpoint, lora_revisions={"other-profile/source": "a" * 40})
    assert not hasattr(adapter, "_lora")
    assert adapter.lora_capability() is None


@pytest.mark.parametrize("paths", [["org/source"], {"/private/source": "org/source"}, {"a": "b", "b": "c"}])
def test_aliases_are_public_names_and_cannot_shadow_sources(checkpoint, paths):
    with pytest.raises(ValueError, match=r"aliases|shadows"):
        CrossEncoderLoraAdapter(checkpoint, lora_paths=paths)


def test_fetch_rechecks_model_generation_before_publishing_source(adapter, monkeypatch):
    from sie_server.adapters.cross_encoder import _lora

    original = _lora.load_file

    def reload_during_fetch(path):
        weights = original(path)
        adapter.unload()
        adapter.load("cpu")
        return weights

    monkeypatch.setattr(_lora, "load_file", reload_during_fetch)
    with pytest.raises(RuntimeError, match="reloaded while"):
        adapter.fetch_lora("a")
    assert adapter._lora._prepared == {}


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("disjoint", [False, True])
def test_partial_injection_failure_preserves_old_adapters_and_base(adapter, monkeypatch, existing, disjoint):
    from peft.tuners.lora.model import LoraModel

    snapshot = base_snapshot(adapter)
    baseline = scores(adapter)
    if disjoint:
        for alias, target in (("a", "query"), ("b", "value")):
            path = Path(adapter._lora._aliases[alias])
            config = json.loads((path / "adapter_config.json").read_text())
            config["target_modules"] = [target]
            (path / "adapter_config.json").write_text(json.dumps(config))
            weights = {k: v for k, v in load_file(path / "adapter_model.safetensors").items() if f".{target}." in k}
            save_file(weights, path / "adapter_model.safetensors")
    if existing:
        adapter.load_lora("a")
        adapter.set_active_lora("a")
        a = scores(adapter)
    original = LoraModel._create_and_replace
    calls = 0

    def fail_second_layer(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected mid-attachment failure")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(LoraModel, "_create_and_replace", fail_second_layer)
        with pytest.raises(RuntimeError, match="mid-attachment"):
            adapter.load_lora("b")
    assert calls == 2
    assert_base_preserved(adapter, snapshot, unwrapped=not existing)
    assert dict(snapshot[1].named_modules()) == snapshot[5]
    if disjoint:
        assert all(name.endswith("query") for name in adapter._lora._resident_layers)
    np.testing.assert_array_equal(scores(adapter), baseline)
    if existing:
        adapter.set_active_lora("a")
        np.testing.assert_array_equal(scores(adapter), a)
    adapter.load_lora("b")


def test_source_cleanup_failure_retains_capacity_and_blocks_selection(adapter, monkeypatch):
    path = adapter.fetch_lora("a")
    canonical = adapter._lora._aliases["a"]
    temporary = adapter._lora._prepared[canonical].directory

    def fail_cleanup():
        raise OSError("injected cleanup failure")

    with monkeypatch.context() as context:
        context.setattr(temporary, "cleanup", fail_cleanup)
        with pytest.raises(LoraStateUncertainError, match="cleanup failed"):
            adapter.load_lora("a", source=path)
    assert canonical in adapter._lora._loaded
    assert canonical in adapter._lora._prepared
    with pytest.raises(LoraStateUncertainError):
        adapter.set_active_lora(None)
    adapter.unload()
    assert not Path(path).exists()


def test_source_cleanup_failure_cannot_skip_whole_model_teardown(adapter, monkeypatch):
    adapter.load_lora("a")
    original_model = adapter._model.model
    path = adapter.fetch_lora("b")
    temporary = adapter._lora._prepared[adapter._lora._aliases["b"]].directory

    def fail_cleanup():
        raise OSError("injected cleanup failure")

    with monkeypatch.context() as context:
        context.setattr(temporary, "cleanup", fail_cleanup)
        with pytest.raises(ExceptionGroup, match="owned LoRA"):
            adapter.unload()
    assert adapter._model is None
    assert adapter._lora._model is None
    assert not adapter._lora._loaded
    assert not any(isinstance(module, BaseTunerLayer) for module in original_model.modules())
    assert Path(path).exists()
    temporary.cleanup()


@pytest.mark.parametrize("kind", ["subclass", "quantized", "ordinary_roberta"])
def test_exact_model_class_and_unquantized_admission(adapter, kind):
    from sie_server.adapters.cross_encoder._lora import CrossEncoderLora

    original = adapter._model.model
    if kind == "subclass":

        class CustomClassifier(type(original)):
            pass

        model = CustomClassifier(original.config)
    elif kind == "quantized":
        model = type(original)(original.config)
        model.is_quantized = True
    else:
        model = RobertaForSequenceClassification(RobertaConfig(**original.config.to_dict()))
    before = dict(model.named_modules())
    manager = CrossEncoderLora(adapter._lora._aliases, None, 2)
    with pytest.raises(ValueError, match="standard, unquantized"):
        manager.bind(SimpleNamespace(model=model))
    assert dict(model.named_modules()) == before
    assert manager._model is None
    assert manager._encoder_prefix is None
    assert not hasattr(model, "peft_config")


def test_fully_qualified_encoder_dense_target_preserves_classifier(adapter, checkpoint):
    source = Path(adapter._lora._aliases["a"])
    prefix = encoder_prefix(checkpoint)
    target = f"{prefix}.layer.0.attention.output.dense"
    config_path = source / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["target_modules"] = [target]
    config_path.write_text(json.dumps(config))
    weights_path = source / "adapter_model.safetensors"
    weights = {
        key.replace("attention.self.query", "attention.output.dense"): value
        for key, value in load_file(weights_path).items()
        if ".layer.0.attention.self.query." in key
    }
    assert len(weights) == 2
    save_file(weights, weights_path)
    snapshot, baseline = base_snapshot(adapter), scores(adapter)
    adapter.load_lora("a")
    assert set(adapter._lora._resident_layers) == {target}
    adapter.set_active_lora("a")
    assert np.max(np.abs(scores(adapter) - baseline)) > 1e-6
    assert_base_preserved(adapter, snapshot)
    adapter.unload_lora("a")
    assert_base_preserved(adapter, snapshot, unwrapped=True)
    np.testing.assert_array_equal(scores(adapter), baseline)


def test_mixed_original_grad_flags_and_inference_training_survive_lifecycle(adapter):
    manager = adapter._lora
    manager.close()
    parameters = list(adapter._model.model.parameters())
    for index, parameter in enumerate(parameters):
        parameter.requires_grad_(index % 2 == 0)
    manager.bind(adapter._model)
    snapshot, baseline = base_snapshot(adapter), scores(adapter)
    assert {flag for _, flag in snapshot[7]} == {True, False}
    adapter.load_lora("a")
    adapter.load_lora("b")
    for selection in ("a", None, "b", None):
        adapter.set_active_lora(selection)
        actual = scores(adapter)
        assert_base_preserved(adapter, snapshot, unwrapped=False)
        assert all(not parameter.requires_grad for values in manager._parameters.values() for parameter in values)
        if selection is None:
            assert dict(adapter._model.model.named_modules()) == snapshot[5]
            np.testing.assert_array_equal(actual, baseline)
    adapter.unload_lora("a")
    adapter.unload_lora("b")
    assert_base_preserved(adapter, snapshot, unwrapped=True)
    np.testing.assert_array_equal(scores(adapter), baseline)


@pytest.mark.asyncio
async def test_real_preformed_worker_groups_loras_and_preserves_order(adapter, monkeypatch):
    adapter.load_lora("a")
    adapter.load_lora("b")
    expected = {}
    for selection in (None, "a", "b"):
        adapter.set_active_lora(selection)
        expected[selection] = scores(adapter)
    calls = []
    original = adapter.score_pairs

    def observed(queries, items, **kwargs):
        result = original(queries, items, **kwargs)
        calls.append((adapter._lora._active, len(items)))
        return result

    monkeypatch.setattr(adapter, "score_pairs", observed)
    selections = ["a", "b", "a", None]
    requests = [
        PreformedScoreRequest(
            prepared_items=[ScorePreparedItem(cost=20, original_index=i) for i in range(len(DOCS))],
            query=Item(**QUERY),
            items=[Item(**doc) for doc in DOCS],
            options={"lora": selection},
        )
        for selection in selections
    ]
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        futures = await worker.submit_score_preformed_batch(requests)
        results = await asyncio.wait_for(asyncio.gather(*futures), 10)
        for selection, result in zip(selections, results, strict=True):
            np.testing.assert_allclose(result.output.scores, expected[selection], atol=1e-6, rtol=0)
        assert calls == [(adapter._lora._aliases["a"], 4), (adapter._lora._aliases["b"], 2), (None, 2)]
    finally:
        await worker.stop()


@pytest.mark.usefixtures("_offline_apps")
def test_real_score_routes_authorize_aliases_isolate_batches_and_evict(
    checkpoint,
    tmp_path,
    monkeypatch,
    serve_on_loopback,
):
    from sie_server.adapters.cross_encoder._lora import CrossEncoderLora

    paths = {
        name: save_adapter(checkpoint, tmp_path / name, index) for index, name in enumerate(("a", "b", "c", "other"), 1)
    }
    models = tmp_path / "models"
    models.mkdir()
    config = {
        "sie_id": "test/cross-encoder-lora",
        "weights_path": str(checkpoint),
        "inputs": {"text": True},
        "tasks": {"score": {}},
        "max_sequence_length": 64,
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.cross_encoder.runtime_lora:CrossEncoderLoraAdapter",
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": {"lora_paths": {name: paths[name] for name in ("a", "b", "c")}}},
            },
            "other": {"extends": "default", "adapter_options": {"loadtime": {"lora_paths": {"other": paths["other"]}}}},
        },
    }
    (models / "model.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("SIE_MAX_LORAS_PER_MODEL", "2")
    monkeypatch.setenv("SIE_DISK_CACHE_ENABLED", "false")

    async def no_watcher(self):
        pass

    monkeypatch.setattr(ModelRegistry, "start_hot_reload", no_watcher)
    fetched = []
    original_fetch = CrossEncoderLora.fetch

    def observed_fetch(self, name, revision=None):
        fetched.append(name)
        return original_fetch(self, name, revision)

    monkeypatch.setattr(CrossEncoderLora, "fetch", observed_fetch)
    app = AppFactory.create_app(AppStateConfig(models_dir=str(models), device="cpu", remote_serving=False))
    model = config["sie_id"]
    with (
        serve_on_loopback(app) as url,
        httpx.Client(base_url=url, timeout=15, headers={"Accept": "application/json"}) as client,
        SIEClient(base_url=url, timeout_s=15, remote="forbid") as sdk,
    ):

        def request(selection=None, *, profile=None):
            options = {"lora": selection}
            if profile is not None:
                options["profile"] = profile
            return client.post(f"/v1/score/{model}", json={"query": QUERY, "items": DOCS, "options": options})

        def ready(selection=None):
            deadline = time.monotonic() + 20
            response = request(selection)
            statuses = [response.status_code]
            if selection is not None and response.status_code == 503:
                assert response.json()["detail"]["code"] == "LORA_LOADING"
                assert float(response.headers["Retry-After"]) > 0
            while response.status_code == 503 and time.monotonic() < deadline:
                time.sleep(0.02)
                response = request(selection)
                statuses.append(response.status_code)
            assert response.status_code == 200, response.text
            values = np.asarray(
                [entry["score"] for entry in sorted(response.json()["scores"], key=lambda row: row["item_id"])]
            )
            assert np.isfinite(values).all()
            return values, statuses

        baseline, _ = ready()
        assert fetched == []
        for rejected, profile in (
            ("https://example.invalid/adapter", None),
            (str(tmp_path / "not-declared"), None),
            ("other", None),
            (paths["other"], None),
            ("a", "other"),
        ):
            response = request(rejected, profile=profile)
            assert response.status_code == 400, response.text
            assert response.json()["detail"]["code"] == "INVALID_INPUT"
            assert fetched == []
        a, statuses = ready("a")
        assert statuses[0] == 503
        b, statuses = ready("b")
        assert statuses[0] == 503
        assert np.max(np.abs(a - baseline)) > 1e-6
        assert np.max(np.abs(b - a)) > 1e-6
        assert set(app.state.registry.loaded_lora_names(model)) == {"a", "b"}
        listed = client.get("/v1/models").json()["models"]
        metadata = next(item for item in listed if item["name"] == model)
        assert set(metadata["lora_adapters"]) == {"a", "b"}
        assert not any(source in json.dumps(metadata["lora_adapters"]) for source in paths.values())
        assert len(app.state.registry._loaded[model].loras) == 2
        assert all(
            value.memory_bytes == fixture_lora_bytes(checkpoint)
            for value in app.state.registry._loaded[model].loras.values()
        )
        expected = {None: baseline, "a": a, "b": b}
        selections = ["a", None, "b"] * 8
        # Preserve the original warmed wire assertion: no hidden cold-load retry.
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [(selection, executor.submit(ready, selection)) for selection in selections]
            for selection, future in futures:
                actual, statuses = future.result(timeout=20)
                assert statuses == [200]
                np.testing.assert_allclose(actual, expected[selection], atol=1e-6, rtol=0)

        def sdk_score(selection):
            result = sdk.score(
                model, QUERY, DOCS, options={"lora": selection}, provision_timeout_s=15, max_oom_retries=0
            )
            return np.asarray([row["score"] for row in sorted(result["scores"], key=lambda row: row["item_id"])])

        # The wire-level cold/denial checks above remain explicit. Concurrent
        # served selections exercise the supported SDK through the actual worker.
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [(selection, executor.submit(sdk_score, selection)) for selection in selections]
            for selection, future in futures:
                actual = future.result(timeout=20)
                np.testing.assert_allclose(actual, expected[selection], atol=1e-6, rtol=0)
        ready("b")  # A is now the least recently used resident.
        ready("c")
        assert set(app.state.registry.loaded_lora_names(model)) == {"b", "c"}
        assert not app.state.registry.is_lora_loaded(model, "a")
        np.testing.assert_allclose(ready()[0], baseline, atol=1e-6, rtol=0)
        assert fetched == [paths["a"], paths["b"], paths["c"]]
