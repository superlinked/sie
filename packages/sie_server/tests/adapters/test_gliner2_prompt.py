"""Exact prompt admission against the installed GLiNER2 inference compiler.

No model weights: the real convenience methods and collator compile every row;
only the encoder is replaced. These tests also run unchanged against 1.3.1,
1.3.2 and the transformers5 bundle's GLiNER2 2.0.0.
"""

from __future__ import annotations

import copy
import gc
import random
import weakref
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from sie_server.adapters._prompt_limit import PromptLimit
from sie_server.adapters.gliner2._prompt import prompt_counter, structure_schema
from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
from sie_server.types.inputs import InvalidInputError, Item

from .test_gliner2_long_text import gliner2_engine, make_adapter, make_processor

_TEXT = "Alice works at Acme."
_LABELS = ["person", "organization"]
_STRUCTURES = {
    "_sie_root": [
        "name::str::The person's full name",
        "kind::[very long first choice|other choice]::str::The person's role",
        "aliases::list",
    ]
}
_TASKS = {"sentiment": {"labels": ["yes", "no"], "multi_label": True, "cls_threshold": 0.25}}


def _cache_info(processor: Any) -> Any:
    cached = getattr(processor, "_tokenize_cached", None)
    return cached.cache_info() if cached is not None else None


def native_model(processor: Any) -> Any:
    """Real runtime schema construction, with its final batch extraction captured."""
    model = SimpleNamespace(processor=processor, batches=[], schemas=[])
    runtime = gliner2_engine.GLiNER2
    for name in ("create_schema", "_parse_field_spec", "_json_schema", "_classification_schema"):
        if hasattr(runtime, name):
            setattr(model, name, MethodType(getattr(runtime, name), model))

    def capture(texts: list[str], schema: Any, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        model.schemas.append(copy.deepcopy(schema.build()))
        model.batches.append(processor.collate_fn_inference([(text, schema) for text in texts]))
        return [{} for _ in texts]

    model.batch_extract = capture
    return model


def schemas(task: str, model: Any) -> tuple[Any, str, Any]:
    if task == "entities":
        return model.create_schema().entities(_LABELS), "batch_extract_entities", _LABELS
    if task == "relations":
        return model.create_schema().relations(_LABELS), "batch_extract_relations", _LABELS
    if task == "structured":
        return structure_schema(model, _STRUCTURES), "batch_extract_json", _STRUCTURES
    return (
        model.create_schema().classification("sentiment", **_TASKS["sentiment"]),
        "batch_classify_text",
        _TASKS,
    )


@pytest.mark.parametrize("task", ["entities", "relations", "classification", "structured"])
@pytest.mark.parametrize("text", [_TEXT, "", "( ) [SEP_TEXT] café İ"])
def test_count_matches_real_convenience_collator_prefix(task: str, text: str) -> None:
    processor = make_processor()
    model = native_model(processor)
    schema, method, arguments = schemas(task, model)
    before = copy.deepcopy(schema.build())
    getattr(gliner2_engine.GLiNER2, method)(model, [text], arguments)
    assert model.schemas == [before]
    # Use the exact empty-record compiler as an independent prefix oracle.
    empty = processor.transform_and_format("", copy.deepcopy(before))
    count = prompt_counter(processor)(schema)
    assert count == len(empty.input_ids) > 0
    batch = model.batches[0]
    assert batch.input_ids[0, :count].tolist() == empty.input_ids
    assert batch.original_lengths[0] > count
    assert schema.build() == before
    # JSON choices are on the document side of SEP_TEXT and still count.
    if task == "structured":
        assert empty.text_tokens
        schema_only = sum(kind != "text" for kind, *_rest in empty.mapped_indices)
        assert count > schema_only


@pytest.mark.parametrize("task", ["entities", "relations", "classification", "structured"])
def test_count_is_deterministic_without_global_rng_or_serving_state_writes(
    monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    processor = make_processor()
    processor.is_training = True
    processor.sampling_config.max_num_labels = 1
    model = native_model(processor)
    schema, _, _ = schemas(task, model)
    state = copy.deepcopy(processor.__dict__)
    cache_info = _cache_info(processor)
    rng = random.getstate()
    count = prompt_counter(processor)
    for method in ("choice", "random", "shuffle", "sample", "betavariate"):
        monkeypatch.setattr(random, method, Mock(side_effect=AssertionError("counter used process RNG")))
    assert count(schema) == count(schema) > 0
    assert processor.is_training is True
    assert processor.sampling_config == state["sampling_config"]
    assert processor.tokenizer.vocab == state["tokenizer"].vocab
    assert processor._token_cache == state["_token_cache"]
    assert _cache_info(processor) == cache_info
    assert random.getstate() == rng


@pytest.mark.parametrize("task", ["entities", "relations", "classification", "structured"])
def test_large_prompt_is_not_truncated_by_training_sampling_or_document_limit(task: str) -> None:
    processor = make_processor()
    processor.is_training = True
    processor.sampling_config.max_num_labels = 1
    model = native_model(processor)
    labels = [f"label{n:04d}" for n in range(1200)]
    if task == "structured":
        schema = structure_schema(model, {"_sie_root": [f"field{n:04d}::str::value description" for n in range(1200)]})
    elif task == "classification":
        schema = model.create_schema().classification("topic", labels)
    else:
        schema = getattr(model.create_schema(), task)(labels)
    count = prompt_counter(processor)(schema)
    assert count > 1200
    # Inference's document max_len never clips schema/prefix tokens.
    processor.change_mode(is_training=False)
    batch = processor.collate_fn_inference([(_TEXT, schema)], max_len=1)
    assert batch.original_lengths[0] == count + len(processor.tokenizer.tokenize("alice"))
    with pytest.raises(InvalidInputError, match="at most 100 tokens"):
        PromptLimit("GLiNER2", 100).check(["short"], lambda: count, task)


@pytest.mark.parametrize("task", ["entities", "relations", "classification", "structured"])
def test_exact_boundary_accepts_equal_rejects_one_less_before_document_work(
    monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    adapter, model = make_adapter()
    kwargs: dict[str, Any] = {"labels": _LABELS}
    item = Item(text=_TEXT)
    if task == "relations":
        item.metadata = {"entities": [{"text": "Alice", "start": 0, "end": 5}]}
    elif task == "classification":
        kwargs["options"] = {"classification_task": "topic"}
    elif task == "structured":
        kwargs = {
            "output_schema": {
                "type": "object",
                "properties": {"field": {"type": "string", "enum": ["first choice", "second choice"]}},
            }
        }
    adapter.extract([item], **kwargs)
    count = model.inputs[0].original_lengths[0] - sum(
        len(model.processor.tokenizer.tokenize(word))
        for word, *_rest in model.processor.word_splitter(_TEXT, lower=True)
    )
    assert list(adapter._prompt_limit._counts.values()) == [count]
    assert count > 1
    adapter._prompt_limit = PromptLimit("GLiNER2", count)
    assert adapter.extract([item], **kwargs).errors is None
    model.inputs.clear()
    adapter._prompt_limit = PromptLimit("GLiNER2", count - 1)
    for name in ("_read", "_doc_input_token_counts", "_classification_windows", "_windowed_input_token_counts"):
        monkeypatch.setattr(adapter, name, Mock(side_effect=AssertionError("document work before rejection")))
    for _ in range(2):
        with pytest.raises(InvalidInputError, match=f"take {count} tokens"):
            adapter.extract([item], **kwargs)
    assert model.inputs == []


def test_compiler_exception_is_not_replaced_with_fallback_or_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, model = make_adapter()
    # The copy uses the actual type's method, so failure must propagate.
    monkeypatch.setattr(type(model.processor), "transform_and_format", Mock(side_effect=ValueError("broken schema")))
    monkeypatch.setattr(adapter, "_read", Mock(side_effect=AssertionError("document read")))
    for _ in range(2):
        with pytest.raises(ValueError, match="broken schema"):
            adapter.extract([Item(text=_TEXT)], labels=_LABELS)
    assert not adapter._prompt_limit._counts
    assert model.inputs == []


def test_extended_classification_contract_is_not_silently_ignored() -> None:
    processor = make_processor()
    schema = native_model(processor).create_schema().classification("topic", ["yes"], prompt="new prompt")
    with pytest.raises(RuntimeError, match="schema changed"):
        prompt_counter(processor)(schema)


def test_aggregate_character_limit_precedes_schema_build_and_compilation() -> None:
    adapter, model = make_adapter(max_prompt_tokens=20)
    model.create_schema = Mock(side_effect=AssertionError("schema build before character admission"))
    with pytest.raises(InvalidInputError, match="characters"):
        adapter.extract(
            [Item(text=_TEXT)],
            output_schema={"type": "object", "properties": {"field": {"type": "string", "description": "d" * 1000}}},
        )
    model.create_schema.assert_not_called()


def test_cache_preserves_task_and_order_and_does_not_retain_strings() -> None:
    adapter, _ = make_adapter()
    count = Mock(wraps=adapter._count_prompt)
    adapter._count_prompt = count
    for labels in (["one", "two"], ["two", "one"], ["one", "two"]):
        adapter.extract([Item(text=_TEXT)], labels=labels)
    assert count.call_count == 2
    for n in range(270):
        adapter.extract([Item(text=_TEXT)], labels=[f"label {n}"])
    assert len(adapter._prompt_limit._counts) == 256
    assert all(
        type(key) is bytes and len(key) == 32 and type(value) is int
        for key, value in adapter._prompt_limit._counts.items()
    )


def test_private_fast_tokenizer_state_and_cache_are_independent(tmp_path: Any) -> None:
    transformers = pytest.importorskip("transformers")
    processor_module = pytest.importorskip("gliner2.processor")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalice\nperson\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    processor = processor_module.SchemaTransformer(tokenizer=tokenizer)
    processor.tokenizer.backend_tokenizer.enable_truncation(1)
    before = processor.tokenizer.backend_tokenizer.to_str()
    cache = _cache_info(processor)
    counter = prompt_counter(processor)
    schema = native_model(processor).create_schema().entities(["person"])
    assert counter(schema) > 1
    assert processor.tokenizer.backend_tokenizer.to_str() == before
    assert _cache_info(processor) == cache


def test_load_resets_counts_and_unload_releases_private_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    first = native_model(make_processor())
    second = native_model(make_processor())
    models = iter([first, second])
    package = pytest.importorskip("gliner2")
    monkeypatch.setattr(package.GLiNER2, "from_pretrained", lambda *_args, **_kwargs: next(models))
    adapter = GLiNER2Adapter("fake", compute_precision="float32")
    adapter.load("cpu")
    assert adapter._count_prompt is not None
    counter_ref = weakref.ref(adapter._count_prompt)
    adapter._prompt_tokens(
        _LABELS, key=("entities", tuple(_LABELS)), schema=lambda: first.create_schema().entities(_LABELS)
    )
    assert len(adapter._prompt_limit._counts) == 1
    adapter.unload()
    gc.collect()
    assert adapter._model is None
    assert adapter._count_prompt is None
    assert counter_ref() is None
    adapter.load("cpu")
    assert not adapter._prompt_limit._counts
    assert adapter._model is second
    adapter.unload()


def test_repeated_choice_tokens_defeat_the_old_estimate() -> None:
    adapter, model = make_adapter()
    choices = [f"long choice number {n:02d}" for n in range(20)]
    output_schema = {"type": "object", "properties": {"kind": {"type": "string", "enum": choices}}}
    structures = adapter._json_schema_to_structures(output_schema)
    specs = structures["_sie_root"]
    actual = prompt_counter(model.processor)(structure_schema(model, structures))
    assert adapter._count_subwords is not None
    old_estimate = sum(adapter._count_subwords(specs)) + 2 * len(specs) + len(choices) + 8
    assert actual > old_estimate
    adapter._prompt_limit = PromptLimit("GLiNER2", old_estimate)
    with pytest.raises(InvalidInputError, match=f"take {actual} tokens"):
        adapter.extract([Item(text=_TEXT)], output_schema=output_schema)
    assert model.inputs == []


def test_schema_and_labels_remain_unchanged_across_repeated_calls() -> None:
    adapter, _ = make_adapter()
    output_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "x::y", "enum": ["first value", "second value"]},
            "name": {"type": "string", "title": "Full name"},
        },
    }
    before = copy.deepcopy(output_schema)
    for _ in range(2):
        adapter.extract([Item(text=_TEXT)], output_schema=output_schema)
    assert output_schema == before
    labels = [" one ", "two"]
    original = labels.copy()
    adapter.extract([Item(text=_TEXT)], labels=labels)
    assert labels == original
