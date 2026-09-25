"""Long and pathological documents in the GLiNER2 adapter (gliner2 1.x).

gliner2 lowercases a document, splits all of it into words with a regex that
takes quadratic time on runs like ``"...."``, and keeps the first ``max_len``
words. The adapter splits in linear time and hands gliner2 only the prefix
holding those words. These tests use gliner2's real processor with a stand-in
tokenizer: the prefix must give gliner2 exactly the input the whole text gives.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
from sie_server.adapters.gliner2.classification import GLiNER2ClassificationAdapter
from sie_server.adapters.gliner2.words import PACKAGE_PATTERN, LinearWordSplitter
from sie_server.types.inputs import Item

gliner2_processor = pytest.importorskip("gliner2.processor")
gliner2_engine = pytest.importorskip("gliner2.inference.engine")

MiB = 1024 * 1024
_PIECE = re.compile(r"\S{1,3}")


class FakeTokenizer:
    """Pieces of up to three non-space characters; ids are stable per piece."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def add_special_tokens(self, tokens: dict[str, list[str]]) -> None:
        for token in tokens["additional_special_tokens"]:
            self.vocab.setdefault(token, len(self.vocab))

    def tokenize(self, text: str) -> list[str]:
        if text in self.vocab:
            return [text]
        return _PIECE.findall(text)

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.vocab.setdefault(tokens, len(self.vocab))
        return [self.vocab.setdefault(token, len(self.vocab)) for token in tokens]

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        def encode(one: str) -> list[int]:
            ids = [0, *range(1, len(_PIECE.findall(one)) + 1), 0]
            return ids[:max_length] if truncation and max_length is not None else ids

        return {"input_ids": [encode(one) for one in text] if isinstance(text, list) else encode(text)}


def make_processor() -> Any:
    processor = gliner2_processor.SchemaTransformer(tokenizer=FakeTokenizer())
    processor.change_mode(is_training=False)
    return processor


class PackageModel:
    """gliner2's preprocessing for each extract method, without the encoder."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.inputs: list[Any] = []

    def _collate(self, texts: list[str], schema: Any, max_len: int | None) -> None:
        self.inputs.append(self.processor.collate_fn_inference([(text, schema) for text in texts], max_len=max_len))

    def extract_entities(self, text: str, labels: list[str], *, max_len: int | None, **_: Any) -> dict[str, Any]:
        self._collate([text], gliner2_engine.Schema().entities(labels), max_len)
        return {"entities": {}}

    def batch_extract_entities(
        self, texts: list[str], labels: list[str], *, max_len: int | None, **_: Any
    ) -> list[Any]:
        self._collate(texts, gliner2_engine.Schema().entities(labels), max_len)
        return [{"entities": {}} for _ in texts]

    def classify_text(self, text: str, tasks: dict[str, Any], *, max_len: int | None, **_: Any) -> dict[str, Any]:
        return self.batch_classify_text([text], tasks, max_len=max_len)[0]

    def batch_classify_text(
        self, texts: list[str], tasks: dict[str, Any], *, max_len: int | None, **_: Any
    ) -> list[Any]:
        ((name, config),) = tasks.items()
        self._collate(texts, gliner2_engine.Schema().classification(name, config["labels"]), max_len)
        return [{name: {"label": config["labels"][0], "confidence": 0.9}} for _ in texts]


def make_adapter(cls: type[GLiNER2Adapter] = GLiNER2Adapter, **kwargs: Any) -> tuple[GLiNER2Adapter, PackageModel]:
    adapter = cls("fake/gliner2", max_seq_length=512, **kwargs)
    model = PackageModel(make_processor())
    adapter._use_linear_word_splitter(model.processor)
    adapter._model = model
    return adapter, model


def collated(processor: Any, text: str, max_len: int) -> tuple[Any, ...]:
    batch = processor.collate_fn_inference([(text, gliner2_engine.Schema().entities(["person"]))], max_len=max_len)
    return (
        batch.input_ids.tolist(),
        batch.text_tokens,
        batch.start_mappings,
        batch.end_mappings,
    )


LONG_TEXTS = [
    " ".join(f"Sentence {i} mentions Dr. Priya Raman of Novartis in Basel." for i in range(40)),
    "\u8bf7\u5e2e\u6211\u53d6\u6d88\u8ba2\u5355\uff0c\u6211\u4e0d\u60f3\u8981\u4e86\u3002" * 40,
    "\u0130stanbul " * 30 + "Ankara " * 30,
    "\u039f\u0394\u039f\u03a3'\u0391 " * 40,
    "\u0391\u03a3'\u0391\u03a3'\u0391\u03a3'" * 40,
    "mail a.b@example.com, see https://example.com/x?y=z and @team; " * 20,
    "x" * 5000 + " tail words here",
    ". " * 400 + "end",
    "." * 5000,
    "a." * 3000,
    "see http://example.com/a?b=c " * 60,
    "www.example.org\tthen text\n" * 60,
]
LONG_TEXT_IDS = [
    "prose",
    "cjk",
    "dotted-capital-i",
    "final-sigma",
    "sigma-apostrophes",
    "email-url",
    "long-word",
    "spaced-dots",
    "dots",
    "a-dots",
    "url-at-cut",
    "www-at-cut",
]


class Gliner2V2Splitter:
    """gliner2 2.x's splitter: match the text as given, lowercase each word."""

    _PATTERN = PACKAGE_PATTERN

    def __call__(self, text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
        for match in self._PATTERN.finditer(text):
            word = match.group()
            yield (word.lower() if lower else word), match.start(), match.end()


Gliner2V2Splitter.__name__ = "WhitespaceTokenSplitter"


@pytest.mark.parametrize("splitting", ["gliner2-1.x", "gliner2-2.x"])
@pytest.mark.parametrize("max_len", [7, 64])
@pytest.mark.parametrize("text", LONG_TEXTS, ids=LONG_TEXT_IDS)
def test_the_prefix_gives_gliner2_the_same_input_as_the_whole_text(text: str, max_len: int, splitting: str) -> None:
    adapter = GLiNER2Adapter("fake/gliner2", max_seq_length=max_len)
    processor = make_processor()
    if splitting == "gliner2-2.x":
        processor.word_splitter = Gliner2V2Splitter()
    reference = collated(processor, text, max_len)  # gliner2's own splitter, on the whole text

    adapter._use_linear_word_splitter(processor)
    prefix = adapter._model_text(text)

    assert text.startswith(prefix)
    assert collated(processor, prefix, max_len) == reference
    assert collated(processor, text, max_len) == reference  # and the linear splitter on the whole text


@pytest.mark.parametrize("max_len", [1, 2])
def test_a_url_at_the_cut_keeps_its_separator(max_len: int) -> None:
    # gliner2 ends a text without a sentence end with ".", which a URL word would absorb.
    text = "http://example.com rest of the text"
    adapter = GLiNER2Adapter("fake/gliner2", max_seq_length=max_len)
    processor = make_processor()
    reference = collated(processor, text, max_len)
    adapter._use_linear_word_splitter(processor)
    assert collated(processor, adapter._model_text(text), max_len) == reference


def test_prefixes_are_short_for_long_documents() -> None:
    adapter, _ = make_adapter()
    assert len(adapter._model_text("hello world. " * 100_000)) < 4000
    assert len(adapter._model_text("." * MiB)) == 512
    assert adapter._model_text("short text") == "short text"


@pytest.mark.parametrize(
    "text",
    [
        "word " * 5000,
        "\u8bf7\u5e2e\u6211\u53d6\u6d88\u8ba2\u5355\uff0c" * 3000,
        "a" * 20_000,
        "x " * 100,
        ("prose words " * 2000) + "." * 50_000,
    ],
    ids=["words", "cjk", "one-run", "short", "prose-then-dots"],
)
def test_metering_long_documents_counts_what_tokenizing_all_of_them_counts(text: str) -> None:
    tokenizer = FakeTokenizer()
    full = len(tokenizer(text, add_special_tokens=True, truncation=True, max_length=512)["input_ids"])
    assert GLiNER2Adapter._long_doc_input_tokens(tokenizer, text, 512) == full


@pytest.mark.parametrize("cls", [GLiNER2Adapter, GLiNER2ClassificationAdapter])
@pytest.mark.parametrize(
    "text",
    ["." * (2 * MiB), "a." * MiB, "hello world. " * (2 * MiB // 13), "%" * (2 * MiB - 2) + "@x"],
    ids=["dots", "a-dots", "prose", "local-run-then-at"],
)
def test_pathological_documents_run_in_bounded_time(cls: type[GLiNER2Adapter], text: str) -> None:
    adapter, model = make_adapter(cls, classification_task="prompt_safety", default_labels=["safe", "unsafe"])
    started = time.perf_counter()
    output = adapter.extract([Item(text=text)])
    classify_seconds = time.perf_counter() - started
    started = time.perf_counter()
    entities = adapter.extract([Item(text=text), Item(text="Priya Raman works at Novartis")], labels=["person"])
    extract_seconds = time.perf_counter() - started

    assert output.input_token_counts == [512]
    assert entities.input_token_counts is not None
    assert entities.input_token_counts[0] == 512
    assert len(model.inputs[-1].text_tokens[0]) == 512
    # gliner2's own splitter takes minutes to hours on these texts.
    assert classify_seconds < 1.0
    assert extract_seconds < 1.0


def test_load_refuses_a_word_splitter_it_has_no_equivalent_for() -> None:
    processor = SimpleNamespace(word_splitter=MagicMock())
    with pytest.raises(RuntimeError, match="linear-time equivalent"):
        GLiNER2Adapter("fake/gliner2")._use_linear_word_splitter(processor)


def test_the_processor_gets_the_linear_splitter() -> None:
    _, model = make_adapter()
    assert isinstance(model.processor.word_splitter, LinearWordSplitter)
    assert model.processor.word_splitter.lower_text_first
