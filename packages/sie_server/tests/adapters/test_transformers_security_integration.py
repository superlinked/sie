"""Offline controls at SIE's actual native/dependent model-load boundaries."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import transformers
from sie_server.adapters import _transformers_compat as compat
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
from sie_server.adapters.cross_encoder.runtime_lora import CrossEncoderLoraAdapter
from sie_server.adapters.docling.adapter import DoclingAdapter
from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter
from sie_server.adapters.sglang import generation as sglang_generation
from sie_server.adapters.sglang_vision_extract import adapter as sglang_vision
from sie_server.adapters.siglip.adapter import SiglipAdapter
from sie_server.types.inputs import Item
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertTokenizerFast,
    PretrainedConfig,
    modeling_utils,
)
from transformers.integrations import hub_kernels

_EXACT_TF4 = pytest.mark.skipif(transformers.__version__ != "4.57.6", reason="Exact TF4 backport controls")
_ATTACK = "sie-security-probe.invalid/owned-marker:attention"
_PRIVATE_FIELDS = (
    "_attn_implementation_internal",
    "_attn_implementation",
    "_experts_implementation_internal",
    "_experts_implementation",
)
_STRUCTURE_FIELDS = ("__dict__", "__class__", "attribute_map", "sub_configs")
_NATIVE_CONFIG_INIT = PretrainedConfig.__init__
_NATIVE_HUB_IS_KERNEL = hub_kernels.is_kernel
_NATIVE_MODEL_IS_KERNEL = modeling_utils.is_kernel


@pytest.fixture(autouse=True)
def offline_native_loader(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert not getattr(_NATIVE_CONFIG_INIT, "_sie_kernel_security_backport", False)
    monkeypatch.setattr(PretrainedConfig, "__init__", _NATIVE_CONFIG_INIT)
    monkeypatch.setattr(hub_kernels, "is_kernel", _NATIVE_HUB_IS_KERNEL)
    monkeypatch.setattr(modeling_utils, "is_kernel", _NATIVE_MODEL_IS_KERNEL)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def kernel_boundary(monkeypatch: pytest.MonkeyPatch) -> Mock:
    boundary = Mock(side_effect=AssertionError("Forbidden kernel loader reached"))
    monkeypatch.setattr(hub_kernels, "_kernels_available", True)
    monkeypatch.setattr(hub_kernels, "get_kernel", boundary, raising=False)
    return boundary


def _update_config(path: Path, **values: Any) -> None:
    config = json.loads(path.read_text())
    config.update(values)
    path.write_text(json.dumps(config))


def _tokenizer(directory: Path) -> BertTokenizerFast:
    directory.mkdir(exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alice", "bob", "met", "person", "."]
    vocabulary = directory / "vocab.txt"
    vocabulary.write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary), model_max_length=64)
    tokenizer.save_pretrained(directory)
    return tokenizer


def _bert_config(tokenizer: BertTokenizerFast) -> BertConfig:
    return BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=128,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        num_labels=1,
    )


@pytest.fixture
def bert_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    embedding, cross_encoder = tmp_path / "embedding", tmp_path / "cross-encoder"
    with torch.random.fork_rng():
        torch.manual_seed(81)
        BertModel(_bert_config(_tokenizer(embedding))).save_pretrained(embedding)
        BertForSequenceClassification(_bert_config(_tokenizer(cross_encoder))).save_pretrained(cross_encoder)
    return embedding, cross_encoder


@_EXACT_TF4
@pytest.mark.parametrize("field", _PRIVATE_FIELDS)
def test_direct_embedding_load_drops_serialized_private_settings(
    bert_checkpoints: tuple[Path, Path], kernel_boundary: Mock, field: str
) -> None:
    path, _ = bert_checkpoints
    tokens = torch.tensor([[2, 5, 6, 3]])
    native = BertModel.from_pretrained(path, attn_implementation="eager").eval()
    with torch.inference_mode():
        expected = native(tokens).last_hidden_state
    _update_config(path / "config.json", **{field: _ATTACK})
    adapter = PyTorchEmbeddingAdapter(path, attn_implementation="eager", compute_precision="float32")
    try:
        adapter.load("cpu")
        assert type(adapter._model) is BertModel
        assert "forward" not in adapter._model.__dict__
        assert adapter._model.config._attn_implementation == "eager"
        for key, value in native.state_dict().items():
            assert torch.equal(value, adapter._model.state_dict()[key])
        with torch.inference_mode():
            assert torch.equal(expected, adapter._model(tokens).last_hidden_state)
        kernel_boundary.assert_not_called()
    finally:
        adapter.unload()


@_EXACT_TF4
@pytest.mark.parametrize("field", _STRUCTURE_FIELDS)
def test_direct_load_rejects_config_structure_before_model_construction(
    bert_checkpoints: tuple[Path, Path], kernel_boundary: Mock, field: str
) -> None:
    path, _ = bert_checkpoints
    _update_config(path / "config.json", **{field: {"__class__": "marker"}})
    adapter = PyTorchEmbeddingAdapter(path)
    with pytest.raises(ValueError, match="config attribute structure"):
        adapter.load("cpu")
    assert adapter._model is None
    kernel_boundary.assert_not_called()


@_EXACT_TF4
@pytest.mark.parametrize("field", _PRIVATE_FIELDS)
def test_sglang_parent_processor_guards_actual_config_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    tokenizer = _tokenizer(tmp_path)
    _bert_config(tokenizer).save_pretrained(tmp_path)
    expected = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    _update_config(tmp_path / "config.json", **{field: _ATTACK})
    native = AutoConfig.from_pretrained
    observations = []

    def observed_config(cls: type, *args: Any, **kwargs: Any) -> Any:
        assert getattr(PretrainedConfig.__init__, "_sie_kernel_security_backport", False)
        config = native(*args, **kwargs)
        observations.append(config)
        return config

    monkeypatch.setattr(AutoConfig, "from_pretrained", classmethod(observed_config))
    engine = Mock()
    monkeypatch.setattr(sglang_vision.SGLangGenerationAdapter, "load", engine)
    adapter = sglang_vision.SGLangVisionExtractAdapter(tmp_path, trust_remote_code=False)
    start_loop = Mock()
    monkeypatch.setattr(adapter, "_start_request_loop", start_loop)
    try:
        adapter.load("cuda:0")  # Only the real CPU processor runs; the engine boundary is mocked.
        assert observations
        assert all(getattr(config, field, None) is None for config in observations)
        assert type(adapter._processor) is type(expected) is BertTokenizerFast
        assert adapter._processor(["alice", "bob"]) == expected(["alice", "bob"])
        engine.assert_called_once_with("cuda:0")
        start_loop.assert_called_once_with()
    finally:
        adapter.unload()


@_EXACT_TF4
@pytest.mark.parametrize("field", _STRUCTURE_FIELDS)
def test_sglang_parent_processor_rejects_config_structure_before_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    _bert_config(_tokenizer(tmp_path)).save_pretrained(tmp_path)
    _update_config(tmp_path / "config.json", **{field: {"__class__": "owned-marker"}})
    engine = Mock()
    monkeypatch.setattr(sglang_vision.SGLangGenerationAdapter, "load", engine)
    adapter = sglang_vision.SGLangVisionExtractAdapter(tmp_path, trust_remote_code=False)
    start_loop = Mock()
    monkeypatch.setattr(adapter, "_start_request_loop", start_loop)
    with pytest.raises(ValueError, match="config attribute structure"):
        adapter.load("cuda:0")
    assert adapter._processor is None
    engine.assert_not_called()
    start_loop.assert_not_called()


@_EXACT_TF4
def test_sglang_parent_processor_refuses_unsupported_version_before_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("transformers.__version__", "4.57.5")
    processor = Mock(side_effect=AssertionError("Processor must not run"))
    engine = Mock(side_effect=AssertionError("Engine must not run"))
    monkeypatch.setattr(AutoProcessor, "from_pretrained", processor)
    monkeypatch.setattr(sglang_vision.SGLangGenerationAdapter, "load", engine)
    adapter = sglang_vision.SGLangVisionExtractAdapter(tmp_path, trust_remote_code=False)
    with pytest.raises(RuntimeError, match=r"requires transformers==4\.57\.6"):
        adapter.load("cuda:0")
    assert adapter._processor is None
    processor.assert_not_called()
    engine.assert_not_called()


def _reasoning_tokenizer(directory: Path) -> BertTokenizerFast:
    tokenizer = _tokenizer(directory)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    tokenizer.save_pretrained(directory)
    _bert_config(tokenizer).save_pretrained(directory)
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    del config["tokenizer_class"]  # Exercise AutoTokenizer's actual AutoConfig fallback.
    config_path.write_text(json.dumps(config))
    return tokenizer


@_EXACT_TF4
@pytest.mark.parametrize("field", _PRIVATE_FIELDS)
def test_sglang_parent_reasoning_tokenizer_guards_actual_config_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    tokenizer = _reasoning_tokenizer(tmp_path)
    _update_config(tmp_path / "config.json", **{field: _ATTACK})
    native = AutoConfig.from_pretrained
    observations = []

    def observed_config(cls: type, *args: Any, **kwargs: Any) -> Any:
        assert getattr(PretrainedConfig.__init__, "_sie_kernel_security_backport", False)
        config = native(*args, **kwargs)
        observations.append(config)
        return config

    monkeypatch.setattr(AutoConfig, "from_pretrained", classmethod(observed_config))
    ids = sglang_generation._reasoning_control_token_ids(
        str(tmp_path), revision=None, trust_remote_code=False, reasoning_format="qwen3"
    )
    assert observations
    assert all(getattr(config, field, None) is None for config in observations)
    assert ids == {
        "open": tokenizer.convert_tokens_to_ids("<think>"),
        "close": tokenizer.convert_tokens_to_ids("</think>"),
        "newline": None,
    }


@_EXACT_TF4
@pytest.mark.parametrize("field", _STRUCTURE_FIELDS)
def test_sglang_parent_reasoning_tokenizer_rejects_structure_before_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    _reasoning_tokenizer(tmp_path)
    _update_config(tmp_path / "config.json", **{field: {"__class__": "owned-marker"}})
    reserve_port = Mock(side_effect=AssertionError("Engine resources must not be reserved"))
    monkeypatch.setattr(sglang_generation._server, "find_free_port", reserve_port)
    adapter = sglang_generation.SGLangGenerationAdapter(
        str(tmp_path), reasoning_parser="qwen3", thinking_budget=32, trust_remote_code=False
    )
    with pytest.raises(ValueError, match="config attribute structure"):
        adapter.load("cuda:0")
    assert adapter._thinking_budget_tokens is None
    reserve_port.assert_not_called()


@_EXACT_TF4
def test_sglang_parent_reasoning_tokenizer_refuses_unsupported_version_before_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("transformers.__version__", "4.57.5")
    tokenizer = Mock(side_effect=AssertionError("Tokenizer must not run"))
    reserve_port = Mock(side_effect=AssertionError("Engine resources must not be reserved"))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(sglang_generation._server, "find_free_port", reserve_port)
    adapter = sglang_generation.SGLangGenerationAdapter(
        str(tmp_path), reasoning_parser="qwen3", thinking_budget=32, trust_remote_code=False
    )
    with pytest.raises(RuntimeError, match=r"requires transformers==4\.57\.6"):
        adapter.load("cuda:0")
    assert adapter._thinking_budget_tokens is None
    tokenizer.assert_not_called()
    reserve_port.assert_not_called()


@_EXACT_TF4
@pytest.mark.parametrize("adapter_class", [CrossEncoderAdapter, CrossEncoderLoraAdapter])
@pytest.mark.parametrize("trust", [False, True])
def test_st_and_inherited_lora_load_install_before_native_constructor(
    bert_checkpoints: tuple[Path, Path], kernel_boundary: Mock, adapter_class: type, trust: bool
) -> None:
    _, path = bert_checkpoints
    _update_config(path / "config.json", _attn_implementation_internal=_ATTACK)
    kwargs = {"lora_paths": {"deferred": str(path / "never-read")}} if adapter_class is CrossEncoderLoraAdapter else {}
    adapter = adapter_class(path, trust_remote_code=trust, compute_precision="float32", **kwargs)
    try:
        adapter.load("cpu")
        assert type(adapter._model.model) is BertForSequenceClassification
        assert "forward" not in adapter._model.model.__dict__
        scores = adapter.score(Item(text="alice"), [Item(text="alice met bob"), Item(text="bob")])
        assert len(scores) == 2
        assert torch.isfinite(torch.tensor(scores)).all()
        kernel_boundary.assert_not_called()
    finally:
        adapter.unload()


@_EXACT_TF4
@pytest.mark.parametrize("variant", ["entities", "classification", "entity_lora"])
def test_actual_dependent_gliner2_load_and_inherited_variants(
    tmp_path: Path, kernel_boundary: Mock, variant: str
) -> None:
    gliner2 = pytest.importorskip("gliner2")
    from gliner2.model import ExtractorConfig
    from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
    from sie_server.adapters.gliner2.classification import GLiNER2ClassificationAdapter
    from sie_server.adapters.gliner2.entity_lora import GLiNER2EntityLoraAdapter

    checkpoint = tmp_path / "extractor"
    tokenizer = _tokenizer(checkpoint)
    with torch.random.fork_rng():
        torch.manual_seed(83)
        native = gliner2.GLiNER2(
            ExtractorConfig(model_name="generated-local", max_width=3, counting_layer="count_lstm"),
            encoder_config=_bert_config(tokenizer),
            tokenizer=tokenizer,
        )
        native.save_pretrained(str(checkpoint))
    _update_config(checkpoint / "encoder_config/config.json", _attn_implementation_internal=_ATTACK)
    adapter_class = {"entities": GLiNER2Adapter, "classification": GLiNER2ClassificationAdapter}.get(variant)
    if adapter_class is None:
        adapter = GLiNER2EntityLoraAdapter(checkpoint, lora_paths={"deferred": str(tmp_path / "never-read")})
    else:
        adapter = adapter_class(checkpoint)
    try:
        adapter.load("cpu")
        assert type(adapter._model) is type(native)
        assert type(adapter._model.encoder) is BertModel
        assert "forward" not in adapter._model.encoder.__dict__
        for name, value in native.state_dict().items():
            assert torch.equal(value, adapter._model.state_dict()[name])
        tokens = torch.tensor([[2, 5, 6, 3]])
        native.eval()
        adapter._model.eval()
        with torch.inference_mode():
            assert torch.equal(
                native.encoder(tokens).last_hidden_state, adapter._model.encoder(tokens).last_hidden_state
            )
        kernel_boundary.assert_not_called()
    finally:
        adapter.unload()


@_EXACT_TF4
def test_docling_guard_failure_is_not_swallowed_by_prewarm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("transformers.__version__", "4.57.5")
    adapter = DoclingAdapter("owned-fixture")
    converter = Mock(side_effect=AssertionError("Converter must not run"))
    monkeypatch.setattr(adapter, "_get_converter", converter)
    monkeypatch.setattr(adapter, "_resolve_artifacts_path", lambda: None)
    with pytest.raises(RuntimeError, match=r"requires transformers==4\.57\.6"):
        adapter.load("cpu")
    assert not adapter._loaded
    converter.assert_not_called()


@_EXACT_TF4
def test_docling_lazy_models_inherit_permanent_load_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kernel_boundary: Mock
) -> None:
    BertConfig().save_pretrained(tmp_path)
    _update_config(tmp_path / "config.json", _attn_implementation_internal=_ATTACK)
    adapter = DoclingAdapter("owned-fixture")
    monkeypatch.setattr(adapter, "_resolve_artifacts_path", lambda: None)
    calls: list[bool] = []

    def converter(ocr_enabled: bool) -> object:
        config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
        assert config._attn_implementation is None
        calls.append(ocr_enabled)
        return object()

    monkeypatch.setattr(adapter, "_make_converter", converter)
    monkeypatch.setattr(adapter, "_convert_bytes", lambda *args, **kwargs: ([], 0, None))
    adapter.load("cpu")
    assert adapter._loaded
    assert calls == [False, True]
    # The native lazy cache can reconstruct a converter after prewarming. It
    # still sees the permanent policy, without a request-time installer call.
    adapter._converters.clear()
    adapter._get_converter(ocr_enabled=True)
    assert calls == [False, True, True]
    kernel_boundary.assert_not_called()


@_EXACT_TF4
@pytest.mark.parametrize("variant", ["pplx", "nemo"])
def test_guard_precedes_dependent_and_fallback_native_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    monkeypatch.setattr("transformers.__version__", "4.57.5")
    model = Mock(side_effect=AssertionError("Model constructor must not run"))
    if variant == "pplx":
        module = importlib.import_module("sie_server.adapters.pplx_decider.adapter")
        monkeypatch.setattr(module, "checkpoint_config", lambda path: {})
        monkeypatch.setattr(module, "load_model", model)
        adapter = module.PerplexityDecisionAdapter(str(tmp_path))
    else:
        module = importlib.import_module("sie_server.adapters.nemo_colembed")
        adapter = module.NemoColEmbedAdapter(str(tmp_path))
        monkeypatch.setattr(adapter, "_load_v1_dynamic", model)
        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model)
    with pytest.raises(RuntimeError, match=r"requires transformers==4\.57\.6"):
        adapter.load("cpu")
    model.assert_not_called()


@pytest.mark.parametrize("hub", [False, True])
@pytest.mark.parametrize("hf_text", [False, True])
def test_open_clip_guards_available_transformers_before_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: bool, hf_text: bool
) -> None:
    open_clip = pytest.importorskip("open_clip")
    module = importlib.import_module("sie_server.adapters.siglip.adapter")
    text_config = {"context_length": 16, **({"hf_model_name": "owned-tiny"} if hf_text else {})}
    model_config = {"embed_dim": 8, "vision_cfg": {}, "text_cfg": text_config}
    selected: list[str] = []
    events: list[str] = []
    monkeypatch.setattr(module, "install_transformers_security_fixes", lambda: events.append("guard"))

    def get_config(name: str) -> dict[str, Any]:
        selected.append(name)
        raise AssertionError("A separate config preview cannot authorize the later factory read")

    def create(*args: Any, **kwargs: Any) -> tuple[Any, object, object]:
        assert events == ["guard"]
        events.append("construct")
        return SimpleNamespace(to=lambda **kw: None, eval=lambda: None), object(), object()

    monkeypatch.setattr(open_clip, "get_model_config", get_config)
    monkeypatch.setattr(open_clip, "create_model_and_transforms", create)
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda name: object())
    model_id = "ViT/owned"  # The real factory normalizes legacy slash names.
    if hub:
        model_id = "hf-hub:owned/tiny"
        (tmp_path / "open_clip_model.safetensors").write_bytes(b"owned-placeholder-not-loaded")
        (tmp_path / "open_clip_config.json").write_text(
            json.dumps(
                {
                    "model_cfg": model_config,
                    "preprocess_cfg": {
                        "mean": [0, 0, 0],
                        "std": [1, 1, 1],
                        "interpolation": "bicubic",
                        "resize_mode": "shortest",
                    },
                }
            )
        )
        monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(tmp_path))
        monkeypatch.setattr(open_clip, "add_model_config", lambda path: None)
    adapter = SiglipAdapter(
        "owned", backend="open_clip", open_clip_model_id=model_id, dense_dim=8, revision="a" * 40 if hub else None
    )
    adapter.load("cpu")
    assert events == ["guard", "construct"]
    assert selected == []


@_EXACT_TF4
def test_open_clip_native_model_refuses_installed_unsupported_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    open_clip = pytest.importorskip("open_clip")
    monkeypatch.setattr("transformers.__version__", "4.57.5")
    constructor = Mock(side_effect=AssertionError("Factory must not run"))
    monkeypatch.setattr(open_clip, "create_model_and_transforms", constructor)
    adapter = SiglipAdapter("owned", backend="open_clip", open_clip_model_id="ViT-B-32", dense_dim=8)
    with pytest.raises(RuntimeError, match=r"requires transformers==4\.57\.6"):
        adapter.load("cpu")
    constructor.assert_not_called()


@pytest.mark.parametrize("error", [ImportError("broken discovery"), ValueError("missing module spec")])
def test_open_clip_does_not_hide_transformers_discovery_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    open_clip = pytest.importorskip("open_clip")
    constructor = Mock(side_effect=AssertionError("Factory must not run"))
    monkeypatch.setattr(open_clip, "create_model_and_transforms", constructor)
    module = importlib.import_module("sie_server.adapters.siglip.adapter")
    monkeypatch.setattr(module, "find_spec", Mock(side_effect=error))
    adapter = SiglipAdapter("owned", backend="open_clip", open_clip_model_id="ViT-B-32", dense_dim=8)
    with pytest.raises(type(error), match=str(error)):
        adapter.load("cpu")
    constructor.assert_not_called()


def test_open_clip_installer_failure_prevents_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    open_clip = pytest.importorskip("open_clip")
    constructor = Mock(side_effect=AssertionError("Factory must not run"))
    monkeypatch.setattr(open_clip, "create_model_and_transforms", constructor)
    module = importlib.import_module("sie_server.adapters.siglip.adapter")
    monkeypatch.setattr(
        module, "install_transformers_security_fixes", Mock(side_effect=RuntimeError("Owned installation failure"))
    )
    adapter = SiglipAdapter("owned", backend="open_clip", open_clip_model_id="ViT-B-32", dense_dim=8)
    with pytest.raises(RuntimeError, match="Owned installation failure"):
        adapter.load("cpu")
    constructor.assert_not_called()


@pytest.mark.parametrize("version", ["5.3.0", "5.17.0"])
def test_open_clip_fixed_version_policy_preserves_native_bindings(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    open_clip = pytest.importorskip("open_clip")
    # This is a version-dispatch control, not a claim of loading those wheels.
    monkeypatch.setattr("transformers.__version__", version)
    before = (PretrainedConfig.__init__, hub_kernels.is_kernel, modeling_utils.is_kernel)
    model = SimpleNamespace(to=lambda **kw: None, eval=lambda: None)
    factory = Mock(return_value=(model, object(), object()))
    monkeypatch.setattr(open_clip, "create_model_and_transforms", factory)
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda name: object())
    adapter = SiglipAdapter("owned", backend="open_clip", open_clip_model_id="owned-native", dense_dim=8)
    adapter.load("cpu")
    after = (PretrainedConfig.__init__, hub_kernels.is_kernel, modeling_utils.is_kernel)
    assert all(left is right for left, right in zip(before, after, strict=True))
    factory.assert_called_once_with("owned-native")
    adapter.unload()


@pytest.mark.parametrize("kind", ["unknown", "absent", "malformed", "invalid-json", "empty"])
def test_open_clip_preserves_native_identifier_errors(tmp_path: Path, kind: str) -> None:
    open_clip = pytest.importorskip("open_clip")
    identifier = f"local-dir:{tmp_path / 'missing'}"
    if kind == "unknown":
        identifier = "owned-unregistered-model"
    elif kind == "empty":
        identifier = "local-dir:"
    elif kind in {"malformed", "invalid-json"}:
        identifier = f"local-dir:{tmp_path}"
        (tmp_path / "open_clip_config.json").write_text("{}" if kind == "malformed" else "{invalid-json")
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)) as native:
        open_clip.create_model_and_transforms(identifier)
    adapter = SiglipAdapter("owned", backend="open_clip", open_clip_model_id=identifier, dense_dim=8)
    with pytest.raises(type(native.value)) as guarded:
        adapter.load("cpu")
    assert str(guarded.value) == str(native.value)


@_EXACT_TF4
def test_actual_open_clip_hf_text_constructor_sees_guard(
    bert_checkpoints: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, kernel_boundary: Mock
) -> None:
    open_clip = pytest.importorskip("open_clip")
    from open_clip.model import CustomTextCLIP

    path, _ = bert_checkpoints
    _update_config(path / "config.json", _attn_implementation_internal=_ATTACK)
    config = {
        "embed_dim": 8,
        "vision_cfg": {"layers": 1, "width": 32, "head_width": 16, "patch_size": 4, "image_size": 8},
        "text_cfg": {
            "hf_model_name": str(path),
            "hf_model_pretrained": True,
            "hf_proj_type": "linear",
            "hf_pooler_type": "mean_pooler",
        },
    }
    monkeypatch.setattr(open_clip.factory, "_MODEL_CONFIGS", {**open_clip.factory._MODEL_CONFIGS, "owned-tiny": config})
    # Tokenizer choice is independent of the HF model loader under test.
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda name: object())
    adapter = SiglipAdapter(
        "owned", backend="open_clip", open_clip_model_id="owned-tiny", dense_dim=8, compute_precision="float32"
    )
    # OpenCLIP captured its AutoModel/AutoConfig aliases before installation.
    assert open_clip.hf_model.AutoModel is transformers.AutoModel
    assert open_clip.hf_model.AutoConfig is AutoConfig
    installed = None
    first_output = None
    for _ in range(2):
        try:
            with torch.random.fork_rng():
                torch.manual_seed(187)
                adapter.load("cpu")
            bindings = (PretrainedConfig.__init__, hub_kernels.is_kernel, modeling_utils.is_kernel)
            if installed is None:
                installed = bindings
            else:
                assert all(before is after for before, after in zip(installed, bindings, strict=True))
            assert type(adapter._model) is CustomTextCLIP
            assert type(adapter._model.text.transformer) is BertModel
            assert "forward" not in adapter._model.text.transformer.__dict__
            with torch.inference_mode():
                output = adapter._model.encode_text(torch.tensor([[2, 5, 6, 3]]))
            assert output.shape == (1, 8)
            assert torch.isfinite(output).all()
            if first_output is None:
                first_output = output
            else:
                assert torch.equal(first_output, output)
            kernel_boundary.assert_not_called()
        finally:
            adapter.unload()


@_EXACT_TF4
@pytest.mark.parametrize("case", ["absolute", "collision", "factory-change"])
def test_actual_open_clip_local_directory_cannot_bypass_guard(
    tmp_path: Path,
    bert_checkpoints: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    kernel_boundary: Mock,
    case: str,
) -> None:
    open_clip = pytest.importorskip("open_clip", minversion="3.0.0")
    path, _ = bert_checkpoints
    tokens = torch.tensor([[2, 5, 6, 3]])
    expected = BertModel.from_pretrained(path, add_pooling_layer=False).eval()
    _update_config(path / "config.json", _attn_implementation_internal=_ATTACK)
    vision = {"layers": 1, "width": 32, "head_width": 16, "patch_size": 4, "image_size": 8}
    native = {
        "embed_dim": 8,
        "vision_cfg": vision,
        "text_cfg": {"context_length": 8, "vocab_size": 49408, "width": 16, "heads": 2, "layers": 1},
    }
    hf = {
        "embed_dim": 8,
        "vision_cfg": vision,
        "text_cfg": {
            "hf_model_name": str(path),
            "hf_tokenizer_name": str(path),
            "hf_model_pretrained": True,
            "hf_proj_type": "linear",
            "hf_pooler_type": "mean_pooler",
        },
    }
    selected = tmp_path / "nested" / "selected"
    selected.mkdir(parents=True)
    # OpenCLIP 3 loads a local-dir tokenizer from that same directory, even
    # when text_cfg names a separate model path.
    _tokenizer(selected)
    config_path = selected / "open_clip_config.json"
    config_path.write_text(json.dumps({"model_cfg": hf}))
    monkeypatch.chdir(tmp_path)
    identifier = f"local-dir:{selected}" if case == "absolute" else "local-dir:nested/selected"
    if case == "collision":
        sibling = tmp_path / "nested-selected"
        sibling.mkdir()
        (sibling / "open_clip_config.json").write_text(json.dumps({"model_cfg": native}))
    elif case == "factory-change":
        # A file can change after a security preview but before the native
        # factory reads it. The permanent policy must already be installed.
        config_path.write_text(json.dumps({"model_cfg": native}))
        factory = open_clip.create_model_and_transforms

        def changed_factory(*args: Any, **kwargs: Any) -> Any:
            config_path.write_text(json.dumps({"model_cfg": hf}))
            return factory(*args, **kwargs)

        monkeypatch.setattr(open_clip, "create_model_and_transforms", changed_factory)
    adapter = SiglipAdapter("owned", backend="open_clip", open_clip_model_id=identifier, dense_dim=8)
    try:
        adapter.load("cpu")
        assert type(adapter._model.text.transformer) is BertModel
        assert "forward" not in adapter._model.text.transformer.__dict__
        assert adapter._model.text.transformer.config._attn_implementation == expected.config._attn_implementation
        with torch.inference_mode():
            assert torch.equal(
                adapter._model.text.transformer(tokens).last_hidden_state, expected(tokens).last_hidden_state
            )
        kernel_boundary.assert_not_called()
    finally:
        adapter.unload()


@pytest.mark.parametrize(
    "module",
    ["sie_server.adapters.remote.sie", "sie_server.adapters._transformers_compat"],
)
def test_optional_paths_do_not_import_transformers(module: str, tmp_path: Path) -> None:
    script = """
import importlib, importlib.abc, sys
class DenyTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'transformers' or fullname.startswith('transformers.'):
            raise AssertionError('Optional path imported Transformers')
sys.meta_path.insert(0, DenyTransformers())
importlib.import_module(sys.argv[1])
assert not any(name == 'transformers' or name.startswith('transformers.') for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 — fixed interpreter, inline script and parametrized internal module
        [sys.executable, "-c", script, module],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_ctranslate2_import_adds_no_model_loading_guard(tmp_path: Path) -> None:
    # CT2 already imports AutoTokenizer. Its actual adapter must not acquire a
    # native model-loading policy merely by importing the shared base classes.
    script = """
import importlib.abc, sys
class DenyCompat(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sie_server.adapters._transformers_compat':
            raise AssertionError('CT2 imported a native model-loading guard')
sys.meta_path.insert(0, DenyCompat())
from sie_server.adapters.ctranslate2.generation import CTranslate2GenerationAdapter, AutoTokenizer
assert CTranslate2GenerationAdapter.__module__ == 'sie_server.adapters.ctranslate2.generation'
assert AutoTokenizer.__module__ == 'transformers.models.auto.tokenization_auto'
assert 'sie_server.adapters._transformers_compat' not in sys.modules
"""
    result = subprocess.run(  # noqa: S603 — fixed interpreter and inline script
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("local_directory", [False, True])
def test_actual_native_open_clip_stays_available_without_transformers(tmp_path: Path, local_directory: bool) -> None:
    pytest.importorskip("open_clip", minversion="3.0.0" if local_directory else None)
    script = """
import importlib.machinery, importlib.util, json, pathlib, sys
# Model an absent optional package through actual import discovery in this
# subprocess; do not suppress a discoverable-but-broken package's errors.
class WithoutTransformers(importlib.machinery.PathFinder):
    @classmethod
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'transformers' or fullname.startswith('transformers.'):
            return None
        return super().find_spec(fullname, path, target)
sys.meta_path[:] = [WithoutTransformers if x is importlib.machinery.PathFinder else x for x in sys.meta_path]
assert importlib.util.find_spec('transformers') is None
import open_clip, torch
from sie_server.adapters.siglip.adapter import SiglipAdapter
torch.set_num_threads(1)
config = {
    'embed_dim': 8,
    'vision_cfg': {'layers': 1, 'width': 32, 'head_width': 16, 'patch_size': 4, 'image_size': 8},
    'text_cfg': {'context_length': 8, 'vocab_size': 49408, 'width': 16, 'heads': 2, 'layers': 1},
}
identifier = 'owned-native'
if sys.argv[1] == 'local':
    directory = pathlib.Path.cwd() / 'nested' / 'native'
    directory.mkdir(parents=True)
    (directory / 'open_clip_config.json').write_text(json.dumps({'model_cfg': config}))
    identifier = f'local-dir:{directory}'
else:
    open_clip.factory._MODEL_CONFIGS[identifier] = config
adapter = SiglipAdapter('owned', backend='open_clip', open_clip_model_id=identifier, dense_dim=8, compute_precision='float32')
try:
    adapter.load('cpu')
    assert type(adapter._model) is open_clip.model.CLIP
    with torch.inference_mode():
        result = adapter._model.encode_text(open_clip.tokenize(['hello'], context_length=8))
    assert result.shape == (1, 8) and torch.isfinite(result).all()
    assert not any(n == 'transformers' or n.startswith('transformers.') for n in sys.modules)
finally:
    adapter.unload()
"""
    result = subprocess.run(  # noqa: S603 — fixed interpreter and inline offline native fixture
        [sys.executable, "-c", script, "local" if local_directory else "registered"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
