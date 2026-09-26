"""Long words in the GLiNER, GLiNER bi-encoder and GLiREL adapters.

These models keep a document's first ``max_len`` words and encode every
subword of each, so one long unbroken run made the encoder input unbounded.
The adapters now read words through ``_word_window``. These tests use gliner's
real processor with an in-memory tokenizer that gives one subword per
character, as DeBERTa tokenizers split a run of accented letters.
"""

from __future__ import annotations

import sys
import time
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters._word_window import (
    ATTENTION_BUDGET,
    MAX_WORD_CHARS,
    WindowedSplitter,
    subword_budget,
)
from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.adapters.gliner_bi import GLiNERBiAdapter
from sie_server.adapters.glirel import _SUBWORDS_PER_WORD as GLIREL_SUBWORDS_PER_WORD
from sie_server.adapters.glirel import GLiRELAdapter
from sie_server.types.inputs import Item

gliner_config = pytest.importorskip("gliner.config")
gliner_processor = pytest.importorskip("gliner.data_processing.processor")
gliner_model = pytest.importorskip("gliner.model")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

MiB = 1024 * 1024
MAX_LEN = 64
DEBERTA = {"model_type": "deberta-v2"}
MODERNBERT = {"model_type": "modernbert"}
BERT = {"model_type": "bert", "max_position_embeddings": 512}


def char_tokenizer() -> Any:
    """A fast tokenizer that encodes each character of a word as one subword."""
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab = {token: index for index, token in enumerate(specials)}
    for char in "abcdefghijklmnopqrstuvwxyz0123456789.,;:!?-_'\"()@/":
        vocab.setdefault(char, len(vocab))
    unknown, padding, first, last = "[UNK]", "[PAD]", "[CLS]", "[SEP]"
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token=unknown))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.WhitespaceSplit(),
            tokenizers.pre_tokenizers.Split(tokenizers.Regex("."), "isolated"),
        ]
    )
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token=unknown, pad_token=padding, cls_token=first, sep_token=last
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<<ENT>>", "<<SEP>>"]})
    return tokenizer


class FakeGLiNER:
    """gliner's preprocessing (word splitting, truncation, prompt, tokenization) around a stand-in encoder."""

    prepare_inputs = gliner_model.BaseEncoderGLiNER.prepare_inputs
    prepare_base_input = gliner_model.BaseEncoderGLiNER.prepare_base_input

    def __init__(
        self, encoder_config: dict[str, Any] | None, *, bi_encoder: bool = False, max_len: int = MAX_LEN
    ) -> None:
        self.config = gliner_config.GLiNERConfig(
            model_name="fake", max_len=max_len, encoder_config=dict(encoder_config) if encoder_config else None
        )
        self.config.relations_layer = None
        if bi_encoder:
            self.data_processor = gliner_processor.BiEncoderSpanProcessor(
                self.config, char_tokenizer(), None, char_tokenizer()
            )
        else:
            self.data_processor = gliner_processor.UniEncoderSpanProcessor(self.config, char_tokenizer(), None)
        self.calls: list[list[str]] = []

    def to(self, *args: Any, **kwargs: Any) -> FakeGLiNER:
        return self

    def inference(self, texts: list[str], labels: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(list(texts))
        return [[] for _ in texts]

    def encode_labels(self, labels: list[str], batch_size: int = 8) -> Any:
        return object()

    def batch_predict_with_embeds(self, texts: list[str], *args: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        return self.inference(texts, [])


def load(adapter_class: type[Any], model: FakeGLiNER) -> Any:
    module = ModuleType("gliner")
    module.GLiNER = MagicMock()  # ty:ignore[unresolved-attribute]
    module.GLiNER.from_pretrained.return_value = model  # ty:ignore[unresolved-attribute]
    adapter = adapter_class("fake/gliner")
    with patch.dict(sys.modules, {"gliner": module}):
        adapter.load("cpu")
    return adapter


def encoded(model: FakeGLiNER, texts: list[str], labels: list[str]) -> list[list[int]]:
    """The token ids gliner would give the encoder: the label prompt, then the document."""
    split, _, _ = model.prepare_inputs(texts)
    batch = model.data_processor.collate_raw_batch(model.prepare_base_input(split), entity_types=labels)
    encoding = model.data_processor.tokenize_inputs(batch["tokens"], batch["classes_to_id"])
    return encoding["input_ids"].tolist()


def encoded_rows(model: FakeGLiNER, texts: list[str], labels: list[str]) -> list[int]:
    """Tokens of each row gliner would give the encoder: the label prompt, then the document."""
    split, _, _ = model.prepare_inputs(texts)
    batch = model.data_processor.collate_raw_batch(model.prepare_base_input(split), entity_types=labels)
    encoded = model.data_processor.tokenize_inputs(batch["tokens"], batch["classes_to_id"])
    return [int(sum(mask)) for mask in encoded["attention_mask"].tolist()]


ORDINARY = [
    "Priya Raman works at Novartis in Basel.",
    " ".join(f"Sentence {i} mentions Dr. Priya Raman of Novartis." for i in range(20)),
    "mail a.b@example.com, see https://example.com/x?y=z and @team; " * 10,
    "\u8bf7\u5e2e\u6211\u53d6\u6d88\u8ba2\u5355\uff0c\u6211\u4e0d\u60f3\u8981\u4e86\u3002" * 20,
]
LONG = [
    "x" * (2 * MiB),
    "\u00e9" * MiB,
    "Priya Raman works at Novartis. " + "7" * MiB,
    ("b" * 200 + " ") * 5000,
]
LONG_IDS = ["letters", "accents", "digits-after-prose", "spaced-long-words"]


@pytest.mark.parametrize("text", ORDINARY, ids=["short", "prose", "email-url", "cjk"])
def test_gliner_reads_ordinary_text_as_before(text: str) -> None:
    reference_model = FakeGLiNER(DEBERTA)  # gliner's own word splitter
    reference_adapter = GLiNERAdapter("fake/gliner")
    reference_adapter._model = reference_model
    reference_counts = reference_adapter._doc_input_token_counts([text], ["person"])
    model = FakeGLiNER(DEBERTA)
    adapter = load(GLiNERAdapter, model)

    output = adapter.extract([Item(text=text)], labels=["person"])

    assert encoded(model, [text], ["person"]) == encoded(reference_model, [text], ["person"])
    assert output.input_token_counts == reference_counts
    assert model.calls == [[text]]


@pytest.mark.parametrize("encoder_config", [DEBERTA, MODERNBERT], ids=["deberta", "modernbert"])
@pytest.mark.parametrize("text", LONG, ids=LONG_IDS)
def test_gliner_reads_long_words_within_the_subword_budget(text: str, encoder_config: dict[str, Any]) -> None:
    model = FakeGLiNER(encoder_config)
    adapter = load(GLiNERAdapter, model)
    budget = subword_budget(MAX_LEN, encoder_config)

    started = time.perf_counter()
    output = adapter.extract([Item(text=text), Item(text="Priya Raman works at Novartis.")], labels=["person"])
    seconds = time.perf_counter() - started

    (split,), starts, ends = model.prepare_inputs([text])
    assert isinstance(model.data_processor.words_splitter, WindowedSplitter)
    assert all(end - start <= MAX_WORD_CHARS for start, end in zip(starts[0], ends[0], strict=True))
    assert sum(len(word) for word in split) <= budget
    # The meter counts the document subwords gliner encodes (with [CLS] and [SEP]).
    assert output.input_token_counts == [sum(len(word) for word in split) + 2, 28]
    assert max(encoded_rows(model, [text], ["person"])) <= budget + 16
    assert seconds < 2.0


def test_gliner_plans_forward_passes_for_long_rows_on_deberta() -> None:
    model = FakeGLiNER(DEBERTA, max_len=512)
    adapter = load(GLiNERAdapter, model)
    long_text = ("w" * 200 + " ") * 40
    texts = [long_text] * 8 + ["Priya Raman works at Novartis."] * 4
    rows = encoded_rows(model, texts, ["person"])

    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])

    assert len(output.entities) == len(texts)
    assert sorted(text for call in model.calls for text in call) == sorted(texts)
    assert len(model.calls) > 2
    for call in model.calls:
        longest = max(rows[texts.index(text)] for text in call)
        assert len(call) <= 8
        assert len(call) == 1 or len(call) * longest**2 <= ATTENTION_BUDGET


def test_gliner_keeps_one_call_for_long_rows_on_other_encoders() -> None:
    model = FakeGLiNER(MODERNBERT, max_len=512)
    adapter = load(GLiNERAdapter, model)
    texts = [("w" * 200 + " ") * 40] * 8

    adapter.extract([Item(text=text) for text in texts], labels=["person"])

    assert model.calls == [texts]


def test_gliner_caps_an_absolute_position_encoder_at_its_position_table() -> None:
    model = FakeGLiNER(BERT)
    adapter = load(GLiNERAdapter, model)
    text = " ".join(["supercalifragilistic"] * MAX_LEN)

    output = adapter.extract([Item(text=text)], labels=["person"])

    assert model.data_processor.transformer_tokenizer.model_max_length == 512
    assert output.input_token_counts is not None
    assert output.input_token_counts[0] <= subword_budget(MAX_LEN, BERT) + 2
    assert max(encoded_rows(model, [text] * 2, ["person"] * 200)) <= 512


@pytest.mark.parametrize("text", LONG, ids=LONG_IDS)
def test_gliner_bi_encoder_reads_long_words_within_the_subword_budget(text: str) -> None:
    model = FakeGLiNER(DEBERTA, bi_encoder=True)
    adapter = load(GLiNERBiAdapter, model)

    output = adapter.extract([Item(text=text), Item(text="Priya Raman works at Novartis.")], labels=["person"])

    (split,), _, _ = model.prepare_inputs([text])
    assert sum(len(word) for word in split) <= subword_budget(MAX_LEN, DEBERTA)
    assert output.input_token_counts is not None
    assert output.input_token_counts[1] == 28


def test_gliner_bi_encoder_plans_forward_passes_for_long_rows_on_deberta() -> None:
    model = FakeGLiNER(DEBERTA, bi_encoder=True, max_len=512)
    adapter = load(GLiNERBiAdapter, model)
    texts = [("w" * 200 + " ") * 40] * 8 + ["Priya Raman works at Novartis."] * 4

    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])

    assert len(output.entities) == len(texts)
    assert len(model.calls) > 2
    assert sorted(text for call in model.calls for text in call) == sorted(texts)


def glirel_adapter(max_words: int = MAX_LEN) -> tuple[GLiRELAdapter, MagicMock]:
    adapter = GLiRELAdapter("fake/glirel")
    adapter._model = MagicMock()
    adapter._model.predict_relations.return_value = []
    adapter._bound_words(char_tokenizer(), max_words, DEBERTA)
    return adapter, adapter._model.predict_relations


def test_glirel_reads_long_words_within_the_subword_budget() -> None:
    adapter, predict = glirel_adapter()
    text = "Tim Cook leads Apple. " + "z" * MiB + " Later text."
    entities = [
        {"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8},
        {"text": "Apple", "label": "ORG", "start": 15, "end": 20},
    ]

    adapter.extract([Item(text=text, metadata={"entities": entities})], labels=["ceo_of"])

    tokens = predict.call_args.kwargs["text"]
    assert tokens[:5] == ["Tim", "Cook", "leads", "Apple", "."]
    assert all(len(token) <= MAX_WORD_CHARS for token in tokens)
    assert sum(len(token) for token in tokens) <= subword_budget(MAX_LEN, DEBERTA, per_word=GLIREL_SUBWORDS_PER_WORD)
    assert predict.call_args.kwargs["ner"] == [[0, 1, "PERSON", "Tim Cook"], [3, 3, "ORG", "Apple"]]


def test_glirel_leaves_out_entities_past_the_words_it_reads() -> None:
    adapter, predict = glirel_adapter(max_words=4)
    text = "Tim Cook led Apple and then Microsoft hires Satya"
    entities = [
        {"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8},
        {"text": "Apple", "label": "ORG", "start": 13, "end": 18},
        {"text": "Microsoft", "label": "ORG", "start": 28, "end": 37},
    ]
    predict.return_value = [
        {
            "head_pos": [0, 2],
            "tail_pos": [3, 4],
            "head_text": ["Tim", "Cook"],
            "tail_text": ["Apple"],
            "label": "ceo_of",
            "score": 0.9,
        }
    ]

    output = adapter.extract([Item(text=text, metadata={"entities": entities})], labels=["ceo_of"])

    assert predict.call_args.kwargs["text"] == ["Tim", "Cook", "led", "Apple"]
    assert predict.call_args.kwargs["ner"] == [[0, 1, "PERSON", "Tim Cook"], [3, 3, "ORG", "Apple"]]
    assert output.relations == [[{"head": "Tim Cook", "tail": "Apple", "relation": "ceo_of", "score": 0.9}]]
    assert [entity["text"] for entity in output.entities[0]] == ["Tim Cook", "Apple", "Microsoft"]


def test_glirel_skips_the_model_when_no_entity_is_read() -> None:
    adapter, predict = glirel_adapter(max_words=2)
    text = "Some words first, then Microsoft hires Satya"
    entities = [
        {"text": "Microsoft", "label": "ORG", "start": 23, "end": 32},
        {"text": "Satya", "label": "PERSON", "start": 39, "end": 44},
    ]

    output = adapter.extract([Item(text=text, metadata={"entities": entities})], labels=["ceo_of"])

    predict.assert_not_called()
    assert output.relations == [[]]
