"""Real local-model controls for the load-only ST custom-code boundary."""

from __future__ import annotations

import gc
import importlib
import json
import sys
import threading
import tomllib
import types
import uuid
import weakref
from collections import UserDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import sentence_transformers
import torch
import yaml
from packaging.requirements import Requirement
from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.base.modules import Dense, Module, Router
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sie_server.adapters import _sentence_transformer_trust as trust
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
from sie_server.adapters.cross_encoder.runtime_lora import CrossEncoderLoraAdapter
from sie_server.adapters.native_causal_cross_encoder.adapter import NativeCausalCrossEncoderAdapter
from sie_server.adapters.sentence_transformer import SentenceTransformerDenseAdapter, SentenceTransformerSparseAdapter
from sie_server.adapters.st_sparse_vision.adapter import SparseEncoderVisionAdapter
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertModel,
    BertTokenizerFast,
    dynamic_module_utils,
)


class _ControlledAutoImportMarker(BaseException):
    """Survive AutoProcessor's broad Exception fallback to prove code execution."""


def test_dependency_pins_match_supported_loading_boundary() -> None:
    root = Path(__file__).resolve().parents[4]
    server = root / "packages" / "sie_server"
    project = tomllib.loads((server / "pyproject.toml").read_text())
    requirement = next(
        Requirement(value)
        for value in project["project"]["dependencies"]
        if Requirement(value).name == "sentence-transformers"
    )
    (specifier,) = requirement.specifier
    assert specifier.operator == "=="
    assert specifier.version in trust._SUPPORTED_VERSIONS
    for bundle in ("default", "transformers5"):
        config = yaml.safe_load((server / "bundles" / f"{bundle}.yaml").read_text())
        assert config["deps"]["sentence-transformers"] == str(requirement.specifier)
    lock = tomllib.loads((root / "uv.lock").read_text())
    assert [package["version"] for package in lock["package"] if package["name"] == "sentence-transformers"] == [
        specifier.version
    ]


@pytest.fixture(autouse=True)
def _isolated_dynamic_module_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = str(tmp_path / "hf-modules")
    monkeypatch.setenv("HF_MODULES_CACHE", cache)
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", cache)
    # Transformers appends this cache to sys.path when loading custom code.
    monkeypatch.setattr(sys, "path", list(sys.path))
    control = types.ModuleType("_sie_st_trust_auto_import_control")
    control.__dict__["ControlledAutoImportMarker"] = _ControlledAutoImportMarker
    monkeypatch.setitem(sys.modules, control.__name__, control)


def _json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _model(path: Path, kind: type = SentenceTransformer, module_type: object = "modeling_marker.Marker") -> None:
    _json(path / "config_sentence_transformers.json", {"model_type": kind.__name__})
    _json(path / "modules.json", [{"idx": 0, "name": "0", "path": "", "type": module_type}])


def _marker(path: Path, monkeypatch: pytest.MonkeyPatch, *, pause: str = "") -> tuple[str, Path]:
    path.mkdir(parents=True, exist_ok=True)
    name = "modeling_marker_" + uuid.uuid4().hex
    marker = path / "executed.txt"
    (path / f"{name}.py").write_text(
        "from pathlib import Path\n"
        + pause
        + f"Path({str(marker)!r}).write_text('controlled import')\n"
        + "raise RuntimeError('controlled import marker')\nclass Marker:\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(path))
    return f"{name}.Marker", marker


def _nested(path: Path, class_ref: str, depth: int, *, tokenizer: bool = False) -> None:
    router = "sentence_transformers.models.Router"
    word = "sentence_transformers.models.WordEmbeddings"
    _model(path, module_type=router if depth else word)
    current = path
    for index in range(depth):
        child = router if index < depth - 1 else word if tokenizer else class_ref
        _json(
            current / "router_config.json",
            {"types": {"child": child}, "structure": {"query": ["child"]}, "parameters": {"default_route": "query"}},
        )
        current = current / "child"
    if tokenizer:
        _json(current / "wordembedding_config.json", {"tokenizer_class": class_ref})


@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder, CrossEncoder])
@pytest.mark.parametrize("allowed", [False, None, True])
def test_root_custom_module_requires_explicit_trust(tmp_path, monkeypatch, model_class, allowed):
    reference, marker = _marker(tmp_path, monkeypatch)
    _model(tmp_path, model_class, reference)
    expected = RuntimeError if allowed else ValueError
    with pytest.raises(expected, match="controlled import marker" if allowed else "requires trust_remote_code=True"):
        trust.load_sentence_transformer(model_class, str(tmp_path), trust_remote_code=allowed, local_files_only=True)
    assert marker.exists() is bool(allowed)


@pytest.mark.parametrize(("depth", "tokenizer"), [(1, False), (3, False), (0, True), (1, True), (3, True)])
@pytest.mark.parametrize("allowed", [False, None, True])
def test_recursive_router_and_tokenizer_gate(tmp_path, monkeypatch, depth, tokenizer, allowed):
    reference, marker = _marker(tmp_path, monkeypatch)
    _nested(tmp_path, reference, depth, tokenizer=tokenizer)
    expected = RuntimeError if allowed else ValueError
    with pytest.raises(expected, match="controlled import marker" if allowed else "requires trust_remote_code=True"):
        trust.load_sentence_transformer(
            SentenceTransformer, str(tmp_path), trust_remote_code=allowed, local_files_only=True
        )
    assert marker.exists() is bool(allowed)


@pytest.mark.parametrize(
    "reference",
    [
        None,
        [],
        {},
        "os.system",
        "sentence_transformersx.models.Router",
        "sentence_transformers.models..Router",
        "sentence_transformers.models.Router/../../modeling_marker.Marker",
        "sentence_transformers.models.Router:Marker",
        "sentence_transformers.SentenceTransformer",
        "sentence_transformers.util.import_from_string",
        "sentence_transformers.models.Router\x00",
    ],
)
def test_malformed_or_noncomponent_reference_never_imported(monkeypatch, reference):
    boundary = trust._NativeLoadBoundary()
    imported = []
    monkeypatch.setattr(importlib, "import_module", lambda *args, **kwargs: imported.append(args))
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        boundary.resolve_module(reference)
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        boundary.resolve_tokenizer(reference)
    assert imported == []


@pytest.mark.parametrize("nested", trust._KWARG_GROUPS)
@pytest.mark.parametrize("allowed", [False, None])
@pytest.mark.parametrize("mapping", [dict, UserDict])
def test_nested_kwargs_cannot_elevate_trust(tmp_path, monkeypatch, nested, allowed, mapping):
    reference, marker = _marker(tmp_path, monkeypatch)
    _nested(tmp_path, reference, 2, tokenizer=True)
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        trust.load_sentence_transformer(
            SentenceTransformer,
            str(tmp_path),
            trust_remote_code=allowed,
            local_files_only=True,
            **{nested: mapping({"trust_remote_code": True})},
        )
    assert not marker.exists()


@pytest.mark.parametrize("nested", trust._KWARG_GROUPS)
@pytest.mark.parametrize("value", [[("trust_remote_code", True)], "trust_remote_code", 1])
def test_nonmapping_nested_kwargs_fail_before_constructor(tmp_path, monkeypatch, nested, value):
    monkeypatch.setattr(SentenceTransformer, "__init__", lambda *args, **kwargs: pytest.fail("constructor reached"))
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        trust.load_sentence_transformer(SentenceTransformer, str(tmp_path), **{nested: value})


@pytest.mark.parametrize("elevated", ["processor_kwargs", "tokenizer_kwargs"])
def test_canonical_alias_collision_cannot_hide_elevated_trust(tmp_path, monkeypatch, elevated):
    monkeypatch.setattr(SentenceTransformer, "__init__", lambda *args, **kwargs: pytest.fail("constructor reached"))
    kwargs: dict[str, Mapping[str, bool | None]] = {
        "processor_kwargs": {"trust_remote_code": False},
        "tokenizer_kwargs": {"trust_remote_code": None},
    }
    kwargs[elevated] = UserDict({"trust_remote_code": True})
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        trust.load_sentence_transformer(
            SentenceTransformer,
            str(tmp_path),
            processor_kwargs=kwargs["processor_kwargs"],
            tokenizer_kwargs=kwargs["tokenizer_kwargs"],
        )


@pytest.mark.parametrize("allowed", [False, None])
@pytest.mark.parametrize("nested", trust._KWARG_GROUPS)
def test_nested_none_is_denying_and_caller_mapping_unchanged(tmp_path, monkeypatch, nested, allowed):
    original = UserDict({"trust_remote_code": None, "local_files_only": True})
    seen = []

    def construct(model, path, *, trust_remote_code, **kwargs):
        seen.append((trust_remote_code, kwargs[nested]))
        assert kwargs[nested] is not original

    monkeypatch.setattr(SentenceTransformer, "__init__", construct)
    loaded = trust.load_sentence_transformer(
        SentenceTransformer, str(tmp_path), trust_remote_code=allowed, **{nested: original}
    )
    assert seen == [(False, {"trust_remote_code": False, "local_files_only": True})]
    assert original == {"trust_remote_code": None, "local_files_only": True}
    assert trust._RESOLVER not in vars(loaded)


def _bert_checkpoint(path: Path, model_class: type, hf_class: type) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_labels=1,
    )
    hf_class(config).save_pretrained(path)


def _auto_marker(path: Path, kind: str, model_class: type) -> Path:
    name = "modeling_auto_marker_" + uuid.uuid4().hex
    marker = path / "auto-executed.txt"
    (path / f"{name}.py").write_text(
        "from pathlib import Path\n"
        "from _sie_st_trust_auto_import_control import ControlledAutoImportMarker\n"
        f"Path({str(marker)!r}).write_text('controlled Auto import')\n"
        "raise ControlledAutoImportMarker('controlled Auto import marker')\nclass Marker:\n    pass\n"
    )
    filename = "tokenizer_config.json" if kind == "tokenizer" else "config.json"
    config_path = path / filename
    config = json.loads(config_path.read_text())
    reference = f"{name}.Marker"
    if kind == "tokenizer":
        config["auto_map"] = {"AutoTokenizer": [reference, reference]}
    elif kind == "config":
        config["auto_map"] = {"AutoConfig": reference}
    else:
        auto_classes: dict[type, str] = {
            SentenceTransformer: "AutoModel",
            SparseEncoder: "AutoModelForMaskedLM",
            CrossEncoder: "AutoModelForSequenceClassification",
        }
        auto = auto_classes[model_class]
        config["auto_map"] = {auto: reference}
    _json(config_path, config)
    return marker


@pytest.mark.parametrize(
    ("model_class", "hf_class", "alias"),
    [
        (SentenceTransformer, BertModel, "tokenizer_kwargs"),
        (SparseEncoder, BertForMaskedLM, "tokenizer_kwargs"),
        (CrossEncoder, BertForSequenceClassification, "tokenizer_args"),
    ],
)
@pytest.mark.parametrize("allowed", [False, None, True])
@pytest.mark.parametrize("mapping", [dict, UserDict])
def test_actual_auto_tokenizer_alias_cannot_elevate_trust(tmp_path, model_class, hf_class, alias, allowed, mapping):
    _bert_checkpoint(tmp_path, model_class, hf_class)
    marker = _auto_marker(tmp_path, "tokenizer", model_class)
    with pytest.raises(
        _ControlledAutoImportMarker if allowed is True else ValueError,
        match="controlled Auto import marker" if allowed is True else "requires trust_remote_code=True",
    ):
        trust.load_sentence_transformer(
            model_class,
            str(tmp_path),
            trust_remote_code=allowed,
            device="cpu",
            local_files_only=True,
            **{alias: mapping({"trust_remote_code": True})},
        )
    assert marker.exists() is (allowed is True)


@pytest.mark.parametrize("kind", ["model", "config"])
@pytest.mark.parametrize("allowed", [False, None, True])
def test_actual_cross_encoder_auto_alias_cannot_elevate_trust(tmp_path, kind, allowed):
    _bert_checkpoint(tmp_path, CrossEncoder, BertForSequenceClassification)
    marker = _auto_marker(tmp_path, kind, CrossEncoder)
    alias = "automodel_args" if kind == "model" else "config_args"
    with pytest.raises(
        _ControlledAutoImportMarker if allowed is True else ValueError,
        match="controlled Auto import marker" if allowed is True else "requires trust_remote_code=True",
    ):
        trust.load_sentence_transformer(
            CrossEncoder,
            str(tmp_path),
            trust_remote_code=allowed,
            device="cpu",
            local_files_only=True,
            **{alias: UserDict({"trust_remote_code": True})},
        )
    assert marker.exists() is (allowed is True)


@pytest.mark.parametrize(
    ("kind", "key"),
    [
        ("model", "model_args"),
        ("model", "model_kwargs"),
        ("tokenizer", "tokenizer_args"),
        ("tokenizer", "processor_kwargs"),
        ("config", "config_args"),
        ("config", "config_kwargs"),
    ],
)
@pytest.mark.parametrize("allowed", [False, None, True])
def test_checkpoint_saved_kwargs_cannot_elevate_trust(tmp_path, kind, key, allowed):
    _bert_checkpoint(tmp_path, SentenceTransformer, BertModel)
    _model(tmp_path, module_type="sentence_transformers.models.Transformer")
    marker = _auto_marker(tmp_path, kind, SentenceTransformer)
    _json(tmp_path / "sentence_bert_config.json", {key: {"trust_remote_code": True}})
    if allowed is True:
        with pytest.raises(_ControlledAutoImportMarker, match="controlled Auto import marker"):
            trust.load_sentence_transformer(
                SentenceTransformer, str(tmp_path), trust_remote_code=True, device="cpu", local_files_only=True
            )
    else:
        # A native fallback is safe: the relevant invariant is no custom import.
        loaded = trust.load_sentence_transformer(
            SentenceTransformer, str(tmp_path), trust_remote_code=allowed, device="cpu", local_files_only=True
        )
        assert type(loaded) is SentenceTransformer
    assert marker.exists() is (allowed is True)


def test_private_loader_ignores_nested_import_trust_true(tmp_path, monkeypatch):
    reference, marker = _marker(tmp_path, monkeypatch)
    _nested(tmp_path, reference, 2, tokenizer=True)
    boundary = trust._NativeLoadBoundary()
    router = boundary.resolve_module("sentence_transformers.models.Router")
    assert issubclass(router, Router)
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        router.load(str(tmp_path), trust_remote_code=True, local_files_only=True)
    assert not marker.exists()


def test_known_aliases_resolve_to_native_classes_without_imports(monkeypatch):
    boundary = trust._NativeLoadBoundary()
    imported = []
    monkeypatch.setattr(importlib, "import_module", lambda *args, **kwargs: imported.append(args))
    for name in (
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Pooling.Pooling",
        "sentence_transformers.sentence_transformer.modules.Pooling",
        "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
    ):
        assert boundary.resolve_module(name) is Pooling
    for name in (
        "sentence_transformers.models.tokenizer.WhitespaceTokenizer",
        "sentence_transformers.models.tokenizer.WhitespaceTokenizer.WhitespaceTokenizer",
        "sentence_transformers.sentence_transformer.modules.tokenizer.whitespace.WhitespaceTokenizer",
    ):
        assert boundary.resolve_tokenizer(name) is WhitespaceTokenizer
    assert imported == []


@pytest.mark.parametrize("allowed", [False, None])
def test_existing_dense_activation_guard_preserved(tmp_path, monkeypatch, allowed):
    reference, marker = _marker(tmp_path, monkeypatch)
    Dense(4, 4).save(str(tmp_path))
    config = json.loads((tmp_path / "config.json").read_text())
    config["activation_function"] = reference
    _json(tmp_path / "config.json", config)
    native = trust._NativeLoadBoundary().resolve_module("sentence_transformers.models.Dense")
    assert issubclass(native, Dense)
    loaded = native.load(str(tmp_path), trust_remote_code=allowed, local_files_only=True)
    assert type(loaded) is Dense
    assert type(loaded.activation_function) is torch.nn.Tanh
    assert not marker.exists()


def test_changed_nested_loader_fails_closed(tmp_path, monkeypatch):
    _model(tmp_path, module_type="sentence_transformers.models.Router")
    monkeypatch.setattr(Router, "load", classmethod(lambda cls, *args, **kwargs: pytest.fail("changed loader reached")))
    with pytest.raises(ValueError, match="Unsupported Sentence Transformers"):
        trust.load_sentence_transformer(SentenceTransformer, str(tmp_path), local_files_only=True)


@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize("allowed", [False, None])
def test_native_loaded_objects_and_forwards_are_unchanged(tmp_path, router, allowed):
    tokenizer = WhitespaceTokenizer(vocab=["[PAD]", "hello", "world"], stop_words=[])
    word = WordEmbeddings(tokenizer, torch.arange(12, dtype=torch.float32).reshape(3, 4))
    modules: list[Module] = [word, Pooling(4)]
    if router:
        modules = [Router({"query": modules}, default_route="query")]
    native = SentenceTransformer(modules=modules, device="cpu", local_files_only=True)
    native.save(str(tmp_path), create_model_card=False)
    expected = native.encode(["hello world", "hello"], show_progress_bar=False)
    loaded = trust.load_sentence_transformer(
        SentenceTransformer, str(tmp_path), trust_remote_code=allowed, device="cpu", local_files_only=True
    )
    assert type(loaded) is SentenceTransformer
    assert trust._RESOLVER not in vars(loaded)
    assert loaded.forward.__func__ is SentenceTransformer.forward
    assert loaded.encode.__func__ is SentenceTransformer.encode
    assert [type(item) for item in loaded.modules()] == [type(item) for item in native.modules()]
    assert not any("facade" in type(item).__name__.lower() for item in loaded.modules())
    np.testing.assert_array_equal(loaded.encode(["hello world", "hello"], show_progress_bar=False), expected)


def test_temporary_binding_is_removed_on_constructor_failure(tmp_path, monkeypatch):
    _model(tmp_path)
    refs = []
    original = trust._NativeLoadBoundary.resolve_root

    def observe(boundary, model, *args, **kwargs):
        refs.append(weakref.ref(model))
        assert trust._RESOLVER in vars(model)
        return original(boundary, model, *args, **kwargs)

    monkeypatch.setattr(trust._NativeLoadBoundary, "resolve_root", observe)
    with pytest.raises(ValueError, match="requires trust_remote_code=True") as error:
        trust.load_sentence_transformer(SentenceTransformer, str(tmp_path), local_files_only=True)
    tb = error.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_name == "load_sentence_transformer":
            assert trust._RESOLVER not in vars(tb.tb_frame.f_locals["model"])
        tb = tb.tb_next
    del error
    gc.collect()
    assert len(refs) == 1
    assert refs[0]() is None


@pytest.mark.parametrize(
    ("model_class", "hf_class"),
    [(SentenceTransformer, BertModel), (SparseEncoder, BertForMaskedLM), (CrossEncoder, BertForSequenceClassification)],
)
def test_native_transformer_constructors_and_predictions(tmp_path, model_class, hf_class):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "one", "two"]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(vocab_path)).save_pretrained(tmp_path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_labels=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    hf_class(config).save_pretrained(tmp_path)
    native = model_class(str(tmp_path), device="cpu", trust_remote_code=False, local_files_only=True)
    loaded = trust.load_sentence_transformer(
        model_class, str(tmp_path), device="cpu", trust_remote_code=False, local_files_only=True
    )
    assert type(loaded) is model_class
    assert trust._RESOLVER not in vars(loaded)
    assert loaded.forward.__func__ is native.forward.__func__
    assert [type(item) for item in loaded.modules()] == [type(item) for item in native.modules()]
    if model_class is CrossEncoder:
        actual = loaded.predict([("hello", "world"), ("one", "two")], show_progress_bar=False)
        expected = native.predict([("hello", "world"), ("one", "two")], show_progress_bar=False)
        assert loaded.predict.__func__ is native.predict.__func__
    else:
        actual = loaded.encode(["hello world", "one two"], show_progress_bar=False)
        expected = native.encode(["hello world", "one two"], show_progress_bar=False)
        assert loaded.encode.__func__ is native.encode.__func__
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual.to_dense(), expected.to_dense(), rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_unknown_version_fails_before_constructor(tmp_path, monkeypatch):
    monkeypatch.setattr(sentence_transformers, "__version__", "5.8.0")
    monkeypatch.setattr(SentenceTransformer, "__init__", lambda *args, **kwargs: pytest.fail("constructor reached"))
    with pytest.raises(ValueError, match="Unsupported Sentence Transformers"):
        trust.load_sentence_transformer(SentenceTransformer, str(tmp_path), local_files_only=True)


class _LoadControl(types.ModuleType):
    entered: threading.Event
    release: threading.Event


def test_concurrent_true_false_loads_do_not_change_library_bindings(tmp_path, monkeypatch):
    control_name = "trust_control_" + uuid.uuid4().hex
    control = _LoadControl(control_name)
    control.entered = threading.Event()
    control.release = threading.Event()
    monkeypatch.setitem(sys.modules, control_name, control)
    pause = f"import {control_name} as control\ncontrol.entered.set()\nassert control.release.wait(10)\n"
    reference, marker = _marker(tmp_path, monkeypatch, pause=pause)
    _nested(tmp_path, reference, 2, tokenizer=True)
    loaders = (Router.load.__func__, WordEmbeddings.load.__func__)
    before = []
    for fn in loaders:
        assert isinstance(fn, types.FunctionType)
        before.append((fn, fn.__globals__, dict(fn.__globals__)))

    def allowed_load():
        with pytest.raises(RuntimeError, match="controlled import marker"):
            trust.load_sentence_transformer(
                SentenceTransformer, str(tmp_path), trust_remote_code=True, local_files_only=True
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(allowed_load)
        try:
            assert control.entered.wait(10)
            with pytest.raises(ValueError, match="requires trust_remote_code=True"):
                trust.load_sentence_transformer(
                    SentenceTransformer, str(tmp_path), trust_remote_code=False, local_files_only=True
                )
            assert not marker.exists()
        finally:
            control.release.set()
        future.result(timeout=10)
    assert marker.exists()
    assert loaders == (Router.load.__func__, WordEmbeddings.load.__func__)
    for fn, globals_object, entries in before:
        assert fn.__globals__ is globals_object
        assert entries.keys() == fn.__globals__.keys()
        assert all(fn.__globals__[name] is value for name, value in entries.items())


@pytest.mark.parametrize(
    ("adapter_class", "model_class"),
    [
        (SentenceTransformerDenseAdapter, SentenceTransformer),
        (SentenceTransformerSparseAdapter, SparseEncoder),
        (CrossEncoderAdapter, CrossEncoder),
        (CrossEncoderLoraAdapter, CrossEncoder),
        (SparseEncoderVisionAdapter, SparseEncoder),
    ],
)
@pytest.mark.parametrize("allowed", [False, None, True])
def test_actual_adapter_loads_enforce_root_trust(tmp_path, monkeypatch, adapter_class, model_class, allowed):
    reference, marker = _marker(tmp_path, monkeypatch)
    _model(tmp_path, model_class, reference)
    extra = (
        {"lora_paths": {"example": str(tmp_path / "unused-adapter")}}
        if adapter_class is CrossEncoderLoraAdapter
        else {}
    )
    adapter = adapter_class(str(tmp_path), trust_remote_code=allowed, **extra)
    registrations = []
    original = SparseEncoderVisionAdapter._register_remote_code_dir

    def register(instance):
        registrations.append(instance)
        original(instance)

    monkeypatch.setattr(SparseEncoderVisionAdapter, "_register_remote_code_dir", register)
    before = list(sys.path)
    try:
        with pytest.raises(
            RuntimeError if allowed is True else ValueError,
            match="controlled import marker" if allowed is True else "requires trust_remote_code=True",
        ):
            adapter.load("cpu")
        assert marker.exists() is (allowed is True)
        assert adapter._model is None
        assert bool(registrations) is (allowed is True and adapter_class is SparseEncoderVisionAdapter)
        if allowed is not True:
            assert sys.path == before
    finally:
        adapter.unload()


@pytest.mark.parametrize("allowed", [False, None, True])
def test_actual_dense_adapter_nested_config_cannot_elevate_trust(tmp_path, allowed):
    _bert_checkpoint(tmp_path, SentenceTransformer, BertModel)
    marker = _auto_marker(tmp_path, "config", SentenceTransformer)
    adapter = SentenceTransformerDenseAdapter(
        str(tmp_path), trust_remote_code=allowed, config_kwargs={"trust_remote_code": True}
    )
    try:
        with pytest.raises(
            _ControlledAutoImportMarker if allowed is True else ValueError,
            match="controlled Auto import marker" if allowed is True else "requires trust_remote_code=True",
        ):
            adapter.load("cpu")
        assert marker.exists() is (allowed is True)
        assert adapter._model is None
    finally:
        adapter.unload()


@pytest.mark.parametrize(
    ("adapter_class", "model_class", "hf_class"),
    [
        (SentenceTransformerDenseAdapter, SentenceTransformer, BertModel),
        (SentenceTransformerSparseAdapter, SparseEncoder, BertForMaskedLM),
        (CrossEncoderAdapter, CrossEncoder, BertForSequenceClassification),
    ],
)
@pytest.mark.parametrize("allowed", [False, None, True])
def test_actual_adapter_native_models_preserve_types_and_predictions(
    tmp_path, adapter_class, model_class, hf_class, allowed
):
    _bert_checkpoint(tmp_path, model_class, hf_class)
    native = model_class(str(tmp_path), device="cpu", trust_remote_code=False, local_files_only=True)
    adapter = adapter_class(str(tmp_path), trust_remote_code=allowed)
    try:
        adapter.load("cpu")
        loaded = adapter._model
        assert type(loaded) is model_class
        assert trust._RESOLVER not in vars(loaded)
        assert loaded.forward.__func__ is native.forward.__func__
        assert [type(item) for item in loaded.modules()] == [type(item) for item in native.modules()]
        if model_class is CrossEncoder:
            actual = loaded.predict([("hello", "world")], show_progress_bar=False)
            expected = native.predict([("hello", "world")], show_progress_bar=False)
        else:
            actual = loaded.encode(["hello world"], show_progress_bar=False)
            expected = native.encode(["hello world"], show_progress_bar=False)
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual.to_dense(), expected.to_dense(), rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(actual, expected)
    finally:
        adapter.unload()
    assert adapter._model is None


def test_native_causal_adapter_uses_guard_before_custom_config_import(tmp_path, monkeypatch):
    _bert_checkpoint(tmp_path, CrossEncoder, BertForSequenceClassification)
    marker = _auto_marker(tmp_path, "config", CrossEncoder)
    config = json.loads((tmp_path / "config.json").read_text())
    config["model_type"] = "controlled_custom_model_without_native_fallback"
    _json(tmp_path / "config.json", config)
    _json(
        tmp_path / "modules.json",
        [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
            {
                "idx": 1,
                "name": "1",
                "path": "1_LogitScore",
                "type": "sentence_transformers.cross_encoder.modules.logit_score.LogitScore",
            },
        ],
    )
    seen = []
    original = trust.load_sentence_transformer

    def observe(model_class, *args, **kwargs):
        seen.append((model_class, kwargs["trust_remote_code"]))
        return original(model_class, *args, **kwargs)

    monkeypatch.setattr(trust, "load_sentence_transformer", observe)
    adapter = NativeCausalCrossEncoderAdapter(str(tmp_path))
    try:
        with pytest.raises(ValueError, match="trust_remote_code"):
            adapter.load("cpu")
        assert seen == [(CrossEncoder, False)]
        assert not marker.exists()
        assert adapter._model is None
    finally:
        adapter.unload()
