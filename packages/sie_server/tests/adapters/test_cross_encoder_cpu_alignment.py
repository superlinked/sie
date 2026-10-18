"""Scalar-alignment and loading-boundary controls without model downloads."""

import ctypes
import gc
import json
import struct
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sie_server.adapters import cross_encoder
from sie_server.adapters.cross_encoder._cpu_alignment import normalize_bert_cpu_storage
from sie_server.types.inputs import Item
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast


@pytest.fixture
def model():
    config = BertConfig(
        vocab_size=12,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=1,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    config.sbert_ce_default_activation_function = "torch.nn.modules.linear.Identity"
    with torch.random.fork_rng():
        torch.manual_seed(776)
        return BertForSequenceClassification(config).eval()


def shifted(value, offset=1):
    """Equal packed bytes with an exact prospective pointer offset modulo64."""
    raw = value.detach().contiguous().numpy().tobytes()
    buffer = bytearray(len(raw) + 64)
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    start = (offset - address) % 64
    buffer[start : start + len(raw)] = raw
    result = torch.frombuffer(buffer, dtype=value.dtype, count=value.numel(), offset=start).reshape(value.shape)
    assert result.data_ptr() % 64 == offset
    return result


def query_weight(model):
    return model.bert.encoder.layer[0].attention.self.query.weight


def pointers(model):
    return {name: tensor.data_ptr() for name, tensor in model.state_dict().items()}


@pytest.mark.parametrize("offset", [0, 1, 2, 3, 4, 8, 12])
def test_scalar_alignment_preserves_values_parameter_identity_and_layout(model, offset):
    weight = query_weight(model)
    expected = weight.detach().clone()
    weight.data = shifted(weight, offset)
    before = weight.data_ptr()
    flag = weight.requires_grad
    hook = weight.register_hook(lambda gradient: gradient)
    normalize_bert_cpu_storage(model)
    assert query_weight(model) is weight
    assert weight.requires_grad is flag
    assert hook.id in weight._backward_hooks
    assert weight.stride() == expected.stride()
    assert weight.storage_offset() == 0
    assert weight.data_ptr() % 4 == 0
    assert (weight.data_ptr() == before) is (offset % 4 == 0)
    assert torch.equal(weight, expected)
    hook.remove()


@pytest.mark.parametrize("offset", [0, 1, 2, 3, 4, 8, 12])
@pytest.mark.parametrize("shape", [(1, 384), (2, 384), (108, 384), (4, 27, 384)])
def test_corrected_linear_matches_float64_reference_for_fixed_shapes_and_offsets(offset, shape):
    config = BertConfig(
        vocab_size=12,
        hidden_size=384,
        intermediate_size=384,
        num_hidden_layers=1,
        num_attention_heads=12,
        max_position_embeddings=32,
    )
    with torch.random.fork_rng():
        torch.manual_seed(20261016)
        model = BertForSequenceClassification(config)
    generator = torch.Generator().manual_seed(20261016)
    weight = torch.randn(384, 384, generator=generator) * 0.1
    bias = torch.randn(384, generator=generator) * 0.1
    inputs = torch.randn(shape, generator=generator)
    expected = inputs.numpy().astype(np.float64) @ weight.numpy().astype(np.float64).T + bias.numpy()
    parameter = query_weight(model)
    parameter.data = shifted(weight, offset)
    normalize_bert_cpu_storage(model)
    with torch.inference_mode():
        actual = torch.nn.functional.linear(inputs, parameter, bias).numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)


def test_same_object_ties_and_float_buffer_identity_survive(model):
    weight = query_weight(model)
    weight.data = shifted(weight, 2)
    model.register_parameter("same_weight", weight)
    buffer = shifted(torch.arange(8, dtype=torch.float32), 3)
    model.register_buffer("extra_float_buffer", buffer)
    expected = buffer.clone()
    normalize_bert_cpu_storage(model)
    assert model.same_weight is query_weight(model) is weight
    assert model.extra_float_buffer is buffer
    assert buffer.data_ptr() % 4 == 0
    assert torch.equal(buffer, expected)


@pytest.mark.parametrize(
    "alias_kind", ["parameter", "byte_buffer", "separate_storage_view", "disjoint_view", "empty_view"]
)
def test_distinct_shared_storage_is_rejected_before_any_copy(model, monkeypatch, alias_kind):
    weight = query_weight(model)
    raw = shifted(torch.arange(weight.numel() * 2, dtype=torch.float32), 1)
    weight.data = raw[: weight.numel()].reshape(weight.shape)
    if alias_kind == "parameter":
        model.register_parameter("alias", torch.nn.Parameter(weight.detach()))
    elif alias_kind == "byte_buffer":
        model.register_buffer("alias", weight.view(torch.uint8))
    elif alias_kind == "separate_storage_view":
        # A new Python/Torch storage object still points at overlapping bytes.
        array = (ctypes.c_ubyte * (weight.numel() * 4)).from_address(weight.data_ptr())
        model.register_buffer("alias", torch.frombuffer(array, dtype=torch.uint8))
    elif alias_kind == "empty_view":
        model.register_buffer("alias", weight.detach().flatten()[:0])
    else:
        model.register_buffer("alias", raw[weight.numel() :])
    before = pointers(model)

    def forbidden_clone(*args, **kwargs):
        raise AssertionError("alias checks must precede allocation")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden_clone)
    with pytest.raises(ValueError, match="storage overlaps"):
        normalize_bert_cpu_storage(model)
    assert pointers(model) == before


@pytest.mark.parametrize("layout", ["transpose", "offset"])
def test_unsupported_misaligned_layout_fails_before_mutation(model, layout):
    weight = query_weight(model)
    if layout == "transpose":
        weight.data = shifted(weight, 1).T
    else:
        raw = shifted(torch.zeros(weight.numel() + 1, dtype=torch.float32), 1)
        weight.data = raw[1:].reshape(weight.shape)
    before = pointers(model)
    with pytest.raises(ValueError, match="contiguous storage at offset zero"):
        normalize_bert_cpu_storage(model)
    assert pointers(model) == before


def test_all_copies_are_preflighted_before_mutation_and_failure_releases_copies(model, monkeypatch):
    for weight in model.parameters():
        weight.data = shifted(weight, 2)
    before = pointers(model)
    original_clone = torch.Tensor.clone
    allocations = []

    def fail_second(value, *args, **kwargs):
        if allocations:
            raise MemoryError("controlled copy failure")
        result = original_clone(value, *args, **kwargs)
        allocations.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.Tensor, "clone", fail_second)
    with pytest.raises(MemoryError, match="controlled copy failure"):
        normalize_bert_cpu_storage(model)
    assert pointers(model) == before
    gc.collect()
    assert all(reference() is None for reference in allocations)


def test_bad_replacement_alignment_is_rejected_before_mutation(model, monkeypatch):
    weight = query_weight(model)
    weight.data = shifted(weight, 1)
    before = pointers(model)
    monkeypatch.setattr(torch.Tensor, "clone", lambda value: value)
    with pytest.raises(ValueError, match="Could not create aligned"):
        normalize_bert_cpu_storage(model)
    assert pointers(model) == before


@pytest.mark.parametrize("unsupported", ["subclass", "quantized", "peft"])
def test_unsupported_models_are_unchanged(model, unsupported):
    if unsupported == "subclass":

        class CustomBert(BertForSequenceClassification):
            pass

        model = CustomBert(model.config)
    elif unsupported == "quantized":
        model.is_quantized = True
    else:
        model._hf_peft_config_loaded = True
    query_weight(model).data = shifted(query_weight(model), 1)
    before = pointers(model)
    normalize_bert_cpu_storage(model)
    assert pointers(model) == before


@pytest.mark.parametrize(
    ("system", "machine", "device"),
    [
        ("linux", "aarch64", "cpu"),
        ("darwin", "x86_64", "cpu"),
        ("darwin", "arm64", "mps"),
        ("darwin", "arm64", "cuda:0"),
    ],
)
def test_off_platform_load_does_not_invoke_alignment(monkeypatch, system, machine, device):
    outer = SimpleNamespace(max_length=0)
    cache_calls = []
    monkeypatch.setattr(cross_encoder.sys, "platform", system)
    monkeypatch.setattr(cross_encoder.platform, "machine", lambda: machine)
    monkeypatch.setattr(cross_encoder, "load_sentence_transformer", lambda *args, **kwargs: outer)
    # Both the model and its accelerator are simulated, including cache cleanup.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_calls.append("cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: cache_calls.append("mps"))

    def forbidden_alignment(*args, **kwargs):
        raise AssertionError("off-platform load must not normalize storage")

    monkeypatch.setattr(cross_encoder, "normalize_bert_cpu_storage", forbidden_alignment)
    adapter = cross_encoder.CrossEncoderAdapter("unused", max_length=17)
    adapter.load(device)
    assert adapter._model is outer
    assert outer.max_length == 17
    adapter.unload()
    assert adapter._model is None
    assert cache_calls == ([] if device == "cpu" else [device.split(":", 1)[0]])


def test_failed_normalization_never_publishes_outer(model, monkeypatch):
    weight = query_weight(model)
    weight.data = shifted(weight, 1)
    model.register_buffer("alias", weight.detach())
    outer = SimpleNamespace(model=model)
    monkeypatch.setattr(cross_encoder.sys, "platform", "darwin")
    monkeypatch.setattr(cross_encoder.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cross_encoder, "load_sentence_transformer", lambda *args, **kwargs: outer)
    adapter = cross_encoder.CrossEncoderAdapter("unused")
    with pytest.raises(ValueError, match="storage overlaps"):
        adapter.load("cpu")
    assert adapter._model is None
    adapter.unload()


def test_failed_copy_never_publishes_outer(model, monkeypatch):
    query_weight(model).data = shifted(query_weight(model), 2)
    before = pointers(model)
    monkeypatch.setattr(cross_encoder.sys, "platform", "darwin")
    monkeypatch.setattr(cross_encoder.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        cross_encoder, "load_sentence_transformer", lambda *args, **kwargs: SimpleNamespace(model=model)
    )

    def fail_copy(*args, **kwargs):
        raise MemoryError("controlled load copy failure")

    monkeypatch.setattr(torch.Tensor, "clone", fail_copy)
    adapter = cross_encoder.CrossEncoderAdapter("unused")
    with pytest.raises(MemoryError, match="controlled load copy failure"):
        adapter.load("cpu")
    assert adapter._model is None
    assert pointers(model) == before
    adapter.unload()


def test_failed_load_releases_temporary_outer_model_and_copies(model, monkeypatch):
    config = model.config
    references = []
    allocations = []
    original_clone = torch.Tensor.clone

    class Outer:
        def __init__(self, inner):
            self.model = inner

    def fail_second(value, *args, **kwargs):
        if allocations:
            raise MemoryError("controlled temporary-model failure")
        result = original_clone(value, *args, **kwargs)
        allocations.append(weakref.ref(result))
        return result

    def factory(*args, **kwargs):
        inner = BertForSequenceClassification(config)
        for parameter in inner.parameters():
            parameter.data = shifted(parameter, 2)
        outer = Outer(inner)
        references.extend([weakref.ref(outer), weakref.ref(inner), weakref.ref(query_weight(inner))])
        monkeypatch.setattr(torch.Tensor, "clone", fail_second)
        return outer

    monkeypatch.setattr(cross_encoder.sys, "platform", "darwin")
    monkeypatch.setattr(cross_encoder.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cross_encoder, "load_sentence_transformer", factory)
    adapter = cross_encoder.CrossEncoderAdapter("unused")

    def attempt():
        # Let the exception/traceback owner go out of scope before checking refs.
        with pytest.raises(MemoryError, match="controlled temporary-model failure"):
            adapter.load("cpu")

    attempt()
    assert adapter._model is None
    gc.collect()
    assert allocations
    assert all(reference() is None for reference in references + allocations)


def write_unpadded_checkpoint(model, tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "query", "one", "two", "three", "four", "five", "six"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tokenizer = BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"), model_max_length=32)
    tokenizer.save_pretrained(tmp_path)
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    content = path.read_bytes()
    header_length = struct.unpack("<Q", content[:8])[0]
    header = content[8 : 8 + header_length].rstrip(b" ")
    header += b" " * ((2 - len(header)) % 4)
    json.loads(header)
    path.write_bytes(struct.pack("<Q", len(header)) + header + content[8 + header_length :])
    assert (8 + len(header)) % 4 == 2
    return tokenizer


def test_real_unpadded_checkpoint_load_scoring_and_reload(model, tmp_path, monkeypatch):
    tokenizer = write_unpadded_checkpoint(model, tmp_path)
    monkeypatch.setattr(cross_encoder.sys, "platform", "darwin")
    monkeypatch.setattr(cross_encoder.platform, "machine", lambda: "arm64")
    unaligned_before = []

    def normalize_after_precondition(inner):
        names = [name for name, parameter in inner.named_parameters() if parameter.data_ptr() % 4]
        assert names, "the actual tiny checkpoint must reach normalization with unaligned parameters"
        unaligned_before.append(names)
        normalize_bert_cpu_storage(inner)

    monkeypatch.setattr(cross_encoder, "normalize_bert_cpu_storage", normalize_after_precondition)
    queries, documents = ["query", "query"], ["one two", "three four"]
    inputs = tokenizer(queries, documents, padding=True, return_tensors="pt")
    with torch.inference_mode():
        expected = model(**inputs).logits.flatten().numpy()
    adapter = cross_encoder.CrossEncoderAdapter(tmp_path, max_length=32)
    for _ in range(2):
        adapter.load("cpu")
        loaded = adapter._model.model
        assert all(parameter.data_ptr() % 4 == 0 for parameter in loaded.parameters())
        for name, value in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[name], value), name
        scores = adapter.score(Item(text="query"), [Item(text=text) for text in documents])
        pairs = adapter.score_pairs([Item(text=text) for text in queries], [Item(text=text) for text in documents])
        np.testing.assert_allclose(scores, expected, atol=1e-7, rtol=0)
        np.testing.assert_allclose(pairs.scores, expected, atol=1e-7, rtol=0)
        adapter.unload()
        assert adapter._model is None
    assert len(unaligned_before) == 2


def test_explicit_lora_binds_only_after_real_unpadded_base_is_aligned(model, tmp_path, monkeypatch):
    from sie_server.adapters.cross_encoder._lora import CrossEncoderLora
    from sie_server.adapters.cross_encoder.runtime_lora import CrossEncoderLoraAdapter

    write_unpadded_checkpoint(model, tmp_path)
    monkeypatch.setattr(cross_encoder.sys, "platform", "darwin")
    monkeypatch.setattr(cross_encoder.platform, "machine", lambda: "arm64")
    original_bind = CrossEncoderLora.bind
    observations = []

    def bind(manager, outer):
        assert all(parameter.data_ptr() % 4 == 0 for parameter in outer.model.parameters())
        for name, value in model.state_dict().items():
            assert torch.equal(outer.model.state_dict()[name], value), name
        observations.append(pointers(outer.model))
        original_bind(manager, outer)

    monkeypatch.setattr(CrossEncoderLora, "bind", bind)
    adapter = CrossEncoderLoraAdapter(tmp_path, lora_paths={"a": "org/never-fetched"})
    adapter.load("cpu")
    assert len(observations) == 1
    assert pointers(adapter._model.model) == observations[0]
    assert adapter._lora._outer is adapter._model
    assert adapter._lora._model is adapter._model.model
    for name, layer in adapter._lora._originals.items():
        assert layer is adapter._model.model.get_submodule(name)
    adapter.unload()
    assert adapter._model is None
    assert adapter._lora._model is None
