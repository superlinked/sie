"""Native GLiNER2 contracts using toy tokenization and mocked loaders only."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import yaml
from sie_server.adapters.gliner2 import entities as entities_module
from sie_server.adapters.gliner2.decisions import MARKERS
from sie_server.adapters.gliner2.entities import GLiNER2EntitiesAdapter
from sie_server.adapters.gliner2.words import PACKAGE_PATTERN, LinearWordSplitter
from sie_server.types.inputs import InvalidInputError, Item


class WhitespaceTokenSplitter:
    """The pinned native splitter's original-source lowercasing contract."""

    _PATTERN = PACKAGE_PATTERN

    def __call__(self, text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
        for match in self._PATTERN.finditer(text):
            word = match.group()
            yield word.lower() if lower else word, match.start(), match.end()


class ToyTokenizer:
    """Subwords can be many per word, or empty, as in the native processor."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize(self, word: str) -> list[str]:
        self.calls.append(word)
        return [f"subword-{character}" for character in word if character != "\u2060"]


class ToySchema:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def entities(self, labels: list[str]) -> ToySchema:
        self.labels = list(labels)
        return self

    def build(self) -> dict[str, Any]:
        return {"entities": dict.fromkeys(self.labels, "")}


class ToyProcessor:
    """Single-row collator with the pinned API, punctuation and segment mapping."""

    def __init__(self) -> None:
        self.tokenizer = ToyTokenizer()
        self.word_splitter: Any = WhitespaceTokenSplitter()
        self._tokenize_cached: Callable[[str], list[str]] = self.tokenizer.tokenize
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.truncate_words: int | None = None
        self.training = True
        self.corrupt_length = False

    def change_mode(self, *, is_training: bool) -> None:
        self.training = is_training

    def collate_fn_inference(self, batch: list[tuple[str, dict[str, Any]]], **kwargs: Any) -> SimpleNamespace:
        assert len(batch) == 1
        text, schema = batch[0]
        self.calls.append((text, schema, kwargs))
        model_text = text if text.endswith((".", "!", "?")) else text + "."
        words = list(self.word_splitter(model_text))
        if self.truncate_words is not None:
            words = words[: self.truncate_words]
        # Five structural tokens, plus one marker and label subwords per type.
        mappings = [("schema", 0, 0)] * 4
        for index, label in enumerate(schema["entities"]):
            mappings.append(("schema", index + 1, 0))
            mappings.extend([("schema", index + 1, 0)] * len(self._tokenize_cached(label)))
        mappings.append(("sep", 0, 1))
        for index, (word, _, _) in enumerate(words):
            mappings.extend([("text", index, 1)] * len(self._tokenize_cached(word)))
        return SimpleNamespace(
            original_lengths=[len(mappings) + int(self.corrupt_length)],
            original_texts=[model_text],
            mapped_indices=[mappings],
            start_mappings=[[start for _, start, _ in words]],
            end_mappings=[[end for _, _, end in words]],
        )


class ToyModel:
    def __init__(self, architecture: str = "boundary") -> None:
        self.architecture = architecture
        self.config = SimpleNamespace(max_len=1)
        self.processor = ToyProcessor()
        self.to = Mock()
        self.eval = Mock()
        self.calls: list[tuple[list[str], list[str], dict[str, Any]]] = []
        self.results: dict[str, Any] = {}
        self.batch_result: Any = None

    def create_schema(self) -> ToySchema:
        return ToySchema()

    def batch_extract_entities(self, texts: list[str], labels: list[str], **kwargs: Any) -> Any:
        self.calls.append((texts, list(labels), kwargs))
        if self.batch_result is not None:
            return self.batch_result
        return [self.results.get(text, {"entities": {}}) for text in texts]


@pytest.fixture
def native_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Mock, Mock, ToyModel]:
    model = ToyModel()
    loader = Mock(return_value=model)
    module = ModuleType("gliner2")
    module.AutoExtractor = SimpleNamespace(from_pretrained=loader)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    snapshot = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(entities_module, "snapshot_download", snapshot)
    return loader, snapshot, model


@pytest.fixture
def loaded(native_loader: tuple[Mock, Mock, ToyModel]) -> tuple[GLiNER2EntitiesAdapter, ToyModel]:
    adapter = GLiNER2EntitiesAdapter("publisher/model", revision="a" * 40)
    adapter.load("cpu")
    return adapter, native_loader[2]


def native_counts(model: ToyModel, text: str, labels: list[str]) -> tuple[int, int, int]:
    batch = model.processor.collate_fn_inference([(text, ToySchema().entities(labels).build())])
    document = sum(mapping[0] == "text" for mapping in batch.mapped_indices[0])
    return batch.original_lengths[0], document, batch.original_lengths[0] - document


@pytest.mark.parametrize("marker", MARKERS)
def test_structural_markers_fail_before_native_schema_or_collation(loaded, marker):
    adapter, model = loaded
    model.create_schema = Mock(side_effect=AssertionError("the model must not read malformed labels"))
    with pytest.raises(InvalidInputError, match="structural tokens"):
        adapter.extract([Item(text="Alice.")], labels=[f"person {marker}", "organization"])
    model.create_schema.assert_not_called()
    assert model.processor.calls == []
    assert model.calls == []


def span(text: str, start: int, end: int, *, confidence: Any = 0.9) -> dict[str, Any]:
    return {"text": text[start:end], "start": start, "end": end, "confidence": confidence}


@pytest.mark.parametrize("architecture", ["boundary", "span"])
@pytest.mark.parametrize(
    ("precision", "dtype"), [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)]
)
def test_pinned_snapshot_and_native_auto_loader(
    native_loader: tuple[Mock, Mock, ToyModel], architecture: str, precision: Any, dtype: torch.dtype
) -> None:
    loader, snapshot, model = native_loader
    model.architecture = architecture
    adapter = GLiNER2EntitiesAdapter("fastino/model", revision="b" * 40, compute_precision=precision)
    adapter.load("cpu")
    snapshot.assert_called_once_with(
        repo_id="fastino/model", revision="b" * 40, allow_patterns=list(entities_module._CHECKPOINT_FILES)
    )
    loader.assert_called_once_with(snapshot.return_value, map_location="cpu", quantize=False)
    model.to.assert_called_once_with(dtype=dtype)
    model.eval.assert_called_once()
    assert isinstance(model.processor.word_splitter, LinearWordSplitter)
    assert not model.processor.word_splitter.lower_text_first
    assert not model.processor.training
    assert adapter._architecture == architecture


def test_local_checkpoint_stays_local(native_loader: tuple[Mock, Mock, ToyModel], tmp_path: Path) -> None:
    loader, snapshot, _ = native_loader
    adapter = GLiNER2EntitiesAdapter(tmp_path, revision="c" * 40)
    adapter.load("cpu")
    snapshot.assert_not_called()
    loader.assert_called_once_with(str(tmp_path), map_location="cpu", quantize=False)


def test_optional_native_loader_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner2", ModuleType("gliner2"))
    with pytest.raises(RuntimeError, match=r"gliner2==2\.0\.0"):
        GLiNER2EntitiesAdapter("publisher/model").load("cpu")


@pytest.mark.parametrize("architecture", [None, "classification", "unknown"])
def test_unrelated_architectures_fail(native_loader: tuple[Mock, Mock, ToyModel], architecture: Any) -> None:
    native_loader[2].architecture = architecture
    adapter = GLiNER2EntitiesAdapter("publisher/model")
    with pytest.raises(RuntimeError, match="native span or boundary"):
        adapter.load("cpu")
    assert adapter._model is None


def test_source_lowercasing_and_splitter_are_verified(native_loader: tuple[Mock, Mock, ToyModel]) -> None:
    model = native_loader[2]

    class LegacyWhitespaceTokenSplitter(WhitespaceTokenSplitter):
        def __call__(self, text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
            yield from super().__call__(text.lower() if lower else text, lower=False)

    LegacyWhitespaceTokenSplitter.__name__ = "WhitespaceTokenSplitter"
    model.processor.word_splitter = LegacyWhitespaceTokenSplitter()
    with pytest.raises(RuntimeError, match="source-preserving"):
        GLiNER2EntitiesAdapter("publisher/model").load("cpu")


def test_unknown_splitter_fails(native_loader: tuple[Mock, Mock, ToyModel]) -> None:
    native_loader[2].processor.word_splitter = object()
    with pytest.raises(RuntimeError, match="source-preserving"):
        GLiNER2EntitiesAdapter("publisher/model").load("cpu")


def test_full_source_exact_labels_and_native_api(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    text = "A long document ends with Alice"
    labels = [" Person: full legal name ", "Company / employer"]
    model.results[text] = {"entities": {labels[0]: [span(text, len(text) - 5, len(text))]}}
    row_tokens, document_tokens, prompt_tokens = native_counts(model, text, labels)
    model.processor.calls.clear()

    output = adapter.extract([Item(text=text)], labels=labels, options={"threshold": 0.6})

    assert output.errors is None
    assert output.entities == [
        [{"text": "Alice", "start": len(text) - 5, "end": len(text), "label": labels[0], "score": 0.9}]
    ]
    assert output.input_token_counts == [document_tokens]
    assert output.data == [{"encoded_row_token_count": row_tokens, "schema_prompt_token_count": prompt_tokens}]
    assert model.processor.calls == [
        (
            text,
            {"entities": dict.fromkeys(labels, "")},
            {"max_len": None, "error_policy": "raise", "architecture": "boundary"},
        )
    ]
    assert model.calls == [
        (
            [text],
            labels,
            {"batch_size": 1, "threshold": 0.6, "include_confidence": True, "include_spans": True, "max_len": None},
        )
    ]
    assert model.config.max_len == 1
    assert document_tokens < row_tokens


def test_exact_limit_and_one_subword_overflow(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    labels = ["person"]
    exact = "Alice."
    too_long = "Alicee."
    adapter._max_seq_length = native_counts(model, exact, labels)[0]

    output = adapter.extract([Item(text=exact), Item(text=too_long)], labels=labels)

    assert output.errors is not None
    assert output.errors[0] is None
    assert output.errors[1] is not None
    assert output.errors[1].code == "INPUT_TOO_LONG"
    assert output.input_token_counts == [6, 0]
    assert output.data is not None
    assert output.data[1] == {}
    assert model.calls[0][0] == [exact]


@pytest.mark.parametrize("limit_kind", ["row", "prompt"])
def test_schema_only_overflow_rejects_every_item_without_forward(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], limit_kind: str
) -> None:
    adapter, model = loaded
    labels = ["A descriptive entity label"]
    _, _, prompt = native_counts(model, "x.", labels)
    if limit_kind == "row":
        adapter._max_seq_length = prompt - 1
    else:
        adapter._max_prompt_tokens = prompt - 1
    output = adapter.extract([Item(text="x."), Item(text="y.")], labels=labels)
    assert output.errors is not None
    assert all(error is not None and error.code == "INPUT_TOO_LONG" for error in output.errors)
    assert output.entities == [[], []]
    assert output.input_token_counts == [0, 0]
    assert not model.calls


def test_long_single_word_is_measured_as_subwords(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    adapter._max_seq_length = 30
    output = adapter.extract([Item(text="https://example.com/" + "a" * 40)], labels=["url"])
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[0].code == "INPUT_TOO_LONG"
    assert output.input_token_counts == [0]
    assert not model.calls


@pytest.mark.parametrize("bound", ["characters", "words", "word_characters"])
def test_source_admission_bounds_fail_before_collation_or_inference(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], bound: str
) -> None:
    adapter, model = loaded
    if bound == "characters":
        adapter._max_seq_length = 24
        text = "x " * (24 * entities_module._MAX_SOURCE_CHARS_PER_ROW_TOKEN // 2 + 1)
    elif bound == "words":
        adapter._max_seq_length = 24
        text = "." * (24 * entities_module._MAX_SOURCE_WORDS_PER_ROW_TOKEN + 1)
    else:
        text = "x" * (entities_module._MAX_SOURCE_WORD_CHARS + 1)
    output = adapter.extract([Item(text=text), Item(text="Alice.")], labels=["person"])
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[0].code == "INPUT_TOO_LONG"
    assert output.errors[1] is None
    assert output.input_token_counts == [0, 6]
    assert [call[0] for call in model.processor.calls] == ["Alice."]
    assert model.calls[0][0] == ["Alice."]


@pytest.mark.parametrize("bound", ["characters", "labels"])
def test_prompt_admission_bounds_fail_before_collation_or_inference(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], bound: str
) -> None:
    adapter, model = loaded
    adapter._max_prompt_tokens = 1
    labels = ["x" * 33] if bound == "characters" else ["x", "y"]
    output = adapter.extract([Item(text="Alice.")], labels=labels)
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[0].code == "INPUT_TOO_LONG"
    assert output.input_token_counts == [0]
    assert not model.processor.calls
    assert not model.calls


def test_exact_label_character_admission_bound(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    labels = ["x" * 128]
    output = adapter.extract([Item(text="Alice.")], labels=labels)
    assert output.errors is None
    assert model.calls[0][1] == labels
    model.processor.calls.clear()
    model.calls.clear()
    with pytest.raises(InvalidInputError, match="128 characters"):
        adapter.extract([Item(text="Alice.")], labels=["x" * 129])
    assert not model.processor.calls
    assert not model.calls


def test_mixed_batch_non_text_and_oversized_items_are_isolated(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    adapter._max_seq_length = 24
    output = adapter.extract(
        [Item(text=None), Item(text="Alice."), Item(text="\n  "), Item(text="x" * 25)], labels=["person"]
    )
    assert output.errors is not None
    assert [error.code if error is not None else None for error in output.errors] == [
        "INVALID_INPUT",
        None,
        "INVALID_INPUT",
        "INPUT_TOO_LONG",
    ]
    assert output.input_token_counts == [0, 6, 0, 0]
    assert model.calls[0][0] == ["Alice."]


@pytest.mark.parametrize(
    ("text", "surfaces"),
    [
        ("İstanbul met Alice and Alice.", ["İstanbul", "Alice", "Alice"]),
        ("Élodie and E\u0301lodie 🙂 東京 李雷.", ["Élodie", "E\u0301lodie", "🙂", "東京", "李雷"]),
        ("Mail A.B@Example.CO.uk or https://Example.com/x?y=1, @Team_1 ...", ["A.B@Example.CO.uk", "@Team_1"]),
    ],
)
def test_unicode_repetition_and_exact_offsets(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], text: str, surfaces: list[str]
) -> None:
    adapter, model = loaded
    offset = 0
    spans = []
    for surface in surfaces:
        start = text.index(surface, offset)
        spans.append(span(text, start, start + len(surface)))
        offset = start + len(surface)
    model.results[text] = {"entities": {"Exact Label": spans[::-1]}}
    output = adapter.extract([Item(text=text)], labels=["Exact Label"])
    assert output.errors is None
    assert [entity["text"] for entity in output.entities[0]] == surfaces
    assert all(text[entity["start"] : entity["end"]] == entity["text"] for entity in output.entities[0])
    assert model.calls[0][0] == [text]


def test_tokenizer_empty_word_keeps_source_offsets_and_actual_count(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel],
) -> None:
    adapter, model = loaded
    text = "\u2060 Alice."
    model.results[text] = {"entities": {"person": [span(text, 2, 7)]}}
    output = adapter.extract([Item(text=text)], labels=["person"])
    assert output.errors is None
    assert output.input_token_counts == [6]
    assert output.entities[0][0]["start"] == 2


def test_appended_punctuation_is_in_the_native_limit_and_usage(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    output = adapter.extract([Item(text="Alice"), Item(text="Alice.")], labels=["person"])
    assert output.input_token_counts == [6, 6]
    assert model.calls[0][0] == ["Alice", "Alice."]


@pytest.mark.parametrize(
    "bad_span",
    [
        {"text": "Alice", "start": -1, "end": 4, "confidence": 0.9},
        {"text": "Alice", "start": True, "end": 5, "confidence": 0.9},
        {"text": "Alice", "start": 0.0, "end": 5, "confidence": 0.9},
        {"text": "Alice", "start": 0, "end": 6, "confidence": 0.9},
        {"text": "Alice", "start": 5, "end": 0, "confidence": 0.9},
        {"text": "ALICE", "start": 0, "end": 5, "confidence": 0.9},
        {"text": "Alice.", "start": 0, "end": 6, "confidence": 0.9},
        {"start": 0, "end": 5, "confidence": 0.9},
    ],
)
def test_malformed_spans_fail_without_repair_or_billing(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], bad_span: dict[str, Any]
) -> None:
    adapter, model = loaded
    model.results["Alice"] = {"entities": {"person": [bad_span]}}
    output = adapter.extract([Item(text="Alice"), Item(text="Bob.")], labels=["person"])
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[0].code == "INFERENCE_ERROR"
    assert output.errors[1] is None
    assert output.entities == [[], []]
    assert output.input_token_counts == [0, 4]


@pytest.mark.parametrize(
    "confidence",
    [None, True, "0.9", float("nan"), float("inf"), -0.1, 1.1, pytest.param(10**400, id="overflowing-integer")],
)
def test_invalid_native_confidence_is_per_item_failure(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], confidence: Any
) -> None:
    adapter, model = loaded
    model.results["Alice."] = {"entities": {"person": [span("Alice.", 0, 5, confidence=confidence)]}}
    output = adapter.extract([Item(text="Alice.")], labels=["person"])
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[0].code == "INFERENCE_ERROR"
    assert output.input_token_counts == [0]


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {},
        {"entities": []},
        {"entities": {"unknown": []}},
        {"entities": {"person": "Alice"}},
        {"entities": {"person": ["Alice"]}},
    ],
)
def test_invalid_native_results_are_rejected(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], result: Any) -> None:
    adapter, model = loaded
    model.results["Alice."] = result
    output = adapter.extract([Item(text="Alice.")], labels=["person"])
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[0].code == "INFERENCE_ERROR"
    assert output.input_token_counts == [0]


@pytest.mark.parametrize("batch_result", [[], {}, [{"entities": {}}, {"entities": {}}]])
def test_wrong_batch_result_size_fails_the_group(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], batch_result: Any
) -> None:
    adapter, model = loaded
    model.batch_result = batch_result
    output = adapter.extract([Item(text="Alice.")], labels=["person"])
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[0].code == "INFERENCE_ERROR"
    assert output.input_token_counts == [0]


@pytest.mark.parametrize(
    "threshold",
    [True, "0.5", None, float("nan"), float("inf"), -0.1, 1.1, pytest.param(10**400, id="overflowing-integer")],
)
def test_threshold_requires_a_finite_native_probability(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], threshold: Any
) -> None:
    adapter, model = loaded
    with pytest.raises(InvalidInputError, match="finite number"):
        adapter.extract([Item(text="Alice.")], labels=["person"], options={"threshold": threshold})
    assert not model.calls


@pytest.mark.parametrize("threshold", [0, 0.4, 0.6, 0.8, 1])
def test_runtime_threshold_is_request_local(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], threshold: float) -> None:
    adapter, model = loaded
    adapter.extract([Item(text="Alice.")], labels=["person"], options={"threshold": threshold})
    adapter.extract([Item(text="Alice.")], labels=["person"])
    assert model.calls[0][2]["threshold"] == threshold
    assert model.calls[1][2]["threshold"] == 0.5
    assert adapter._threshold == 0.5


@pytest.mark.parametrize(
    "options",
    [
        {"max_len": 2},
        {"overflow_policy": "truncate"},
        {"batch_size": 1},
        {"classification_task": "x"},
        {"quantize": True},
    ],
)
def test_unsupported_runtime_options_are_rejected(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], options: dict[str, Any]
) -> None:
    adapter, model = loaded
    with pytest.raises(InvalidInputError, match="only the threshold"):
        adapter.extract([Item(text="Alice.")], labels=["person"], options=options)
    assert not model.calls


@pytest.mark.parametrize("kwargs", [{"instruction": "find people"}, {"instruction": ""}, {"output_schema": {}}])
def test_unsupported_extraction_modes_are_rejected(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], kwargs: dict[str, Any]
) -> None:
    adapter, model = loaded
    with pytest.raises(InvalidInputError, match="without output_schema or instruction"):
        adapter.extract([Item(text="Alice.")], labels=["person"], **kwargs)
    assert not model.calls


@pytest.mark.parametrize("labels", [None, [], "person", ["person", "person"], ["person", ""], [None], ["  "]])
def test_invalid_labels_fail_before_native_inference(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel], labels: Any
) -> None:
    adapter, model = loaded
    with pytest.raises(InvalidInputError, match="label"):
        adapter.extract([Item(text="Alice.")], labels=labels)
    assert not model.calls


def test_configured_labels_are_copied_and_exact(native_loader: tuple[Mock, Mock, ToyModel]) -> None:
    labels = [" Person "]
    adapter = GLiNER2EntitiesAdapter("publisher/model", default_labels=labels)
    labels.clear()
    adapter.load("cpu")
    adapter.extract([Item(text="Alice.")])
    adapter.extract([Item(text="Alice.")], labels=["person"])
    adapter.extract([Item(text="Alice.")])
    assert [call[1] for call in native_loader[2].calls] == [[" Person "], ["person"], [" Person "]]


@pytest.mark.parametrize("name", ["max_seq_length", "max_prompt_tokens", "batch_size"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "10"])
def test_invalid_loadtime_limits_are_rejected(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GLiNER2EntitiesAdapter("publisher/model", **{name: value})


def test_unknown_loadtime_options_and_precision_are_rejected() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        GLiNER2EntitiesAdapter("publisher/model", quantize=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="compute_precision"):
        GLiNER2EntitiesAdapter("publisher/model", compute_precision="auto")  # type: ignore[arg-type]


def test_native_processor_may_not_silently_drop_source_words(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    model.processor.truncate_words = 1
    with pytest.raises(RuntimeError, match="complete source"):
        adapter.extract([Item(text="Alice met Bob.")], labels=["person"])
    assert not model.calls


def test_native_processor_counts_must_match_mappings(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    model.processor.corrupt_length = True
    with pytest.raises(RuntimeError, match="invalid encoded token mappings"):
        adapter.extract([Item(text="Alice.")], labels=["person"])
    assert not model.calls


def test_word_cache_is_bounded_and_does_not_retain_long_strings(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel],
) -> None:
    adapter, model = loaded
    tokenize = model.processor._tokenize_cached
    long_word = "x" * (entities_module._CACHED_WORD_CHARS + 1)
    tokenize(long_word)
    tokenize(long_word)
    assert model.processor.tokenizer.calls.count(long_word) == 2
    tokenize("short")
    tokenize("short")
    assert model.processor.tokenizer.calls.count("short") == 1
    for index in range(entities_module._WORD_CACHE_SIZE + 1):
        tokenize(f"word-{index}")
    assert adapter._word_cache.cache_info().currsize == entities_module._WORD_CACHE_SIZE
    assert tokenize(long_word) == model.processor.tokenizer.tokenize(long_word)


def test_attention_budget_plans_separate_forwards_and_restores_order(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel],
) -> None:
    adapter, model = loaded
    texts = ["x" * 3900 + ".", "Alice.", "y" * 3900 + "."]
    for text in texts:
        model.results[text] = {"entities": {"person": [span(text, 0, 1)]}}
    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])
    assert output.errors is None
    assert [entity[0]["text"] for entity in output.entities] == ["x", "A", "y"]
    assert output.input_token_counts == [3901, 6, 3901]
    assert all(len(call[0]) == 1 for call in model.calls)
    assert {call[0][0] for call in model.calls} == set(texts)


def test_native_batch_size_bounds_small_row_groups(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    adapter._batch_size = 2
    output = adapter.extract([Item(text="Alice.") for _ in range(5)], labels=["person"])
    assert output.errors is None
    assert [call[2]["batch_size"] for call in model.calls] == [2, 2, 1]


def test_empty_batch_does_not_forward(loaded: tuple[GLiNER2EntitiesAdapter, ToyModel]) -> None:
    adapter, model = loaded
    output = adapter.extract([], labels=["person"])
    assert output.entities == []
    assert output.input_token_counts == []
    assert output.errors is None
    assert not model.calls


def test_unload_clears_native_objects_and_counts_stay_request_qualified(
    loaded: tuple[GLiNER2EntitiesAdapter, ToyModel],
) -> None:
    adapter, model = loaded
    model.processor._tokenize_cached("short")
    cache = adapter._word_cache
    assert adapter.count_input_tokens([Item(text="Alice.")]) is None
    adapter.unload()
    assert cache.cache_info().currsize == 0
    assert all(getattr(adapter, field) is None for field in adapter.spec.unload_fields)
    with pytest.raises(RuntimeError, match="loaded"):
        adapter.extract([Item(text="Alice.")], labels=["person"])


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("fastino/gliner2.5-base-v1", "ca906247640776a07753514055be9726f9080ead"),
        ("fastino/gliner2.5-small-v1", "7132dc4561c3f94563c6147e75ffa8ef34c4964a"),
        ("fastino/gliner2-privacy-filter-PII-multi", "1cb4166094dc58fa8d836429f060d6c95f62b495"),
    ],
)
def test_descriptors_pin_native_identity_and_explicit_row_budget(model_id: str, revision: str) -> None:
    models = Path(__file__).resolve().parents[2] / "models"
    descriptor = yaml.safe_load((models / f"{model_id.replace('/', '__')}.yaml").read_text())
    assert descriptor["sie_id"] == descriptor["hf_id"] == model_id
    assert descriptor["hf_revision"] == revision
    assert descriptor["max_sequence_length"] == 4096
    profile = descriptor["profiles"]["default"]
    assert profile["adapter_path"] == "sie_server.adapters.gliner2.entities:GLiNER2EntitiesAdapter"
    assert profile["compute_precision"] == "float32"
    assert profile["adapter_options"]["loadtime"] == {"max_prompt_tokens": 2048}
