"""Input-token usage and the label limit of NLI zero-shot classification.

The flash adapter runs one (text, hypothesis) row per label, so an item's
``input_token_counts`` entry is the sum of those rows' tokens after truncation,
with special tokens and without padding. The expected counts come from the
same tokenizer called on one pair at a time, so they never see padding.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sie_server.adapters.nli_classification_flash import NLIClassificationFlashAdapter
from sie_server.core.extract_cost import MAX_EXTRACT_LABELS, adapter_extract_item_costs, build_extract_prepared_items
from sie_server.core.loader import load_model_configs
from sie_server.types.inputs import InvalidInputError, Item
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

_MAX_LENGTH = 32
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_TEMPLATE = "This text is about {}."
_WORDS = [
    "this",
    "text",
    "is",
    "about",
    "the",
    "app",
    "crashes",
    "on",
    "startup",
    "i",
    "was",
    "charged",
    "twice",
    "bug",
    "billing",
    "feature",
    "request",
    "report",
    "when",
    "config",
    "file",
    "missing",
]


def _tokenizer() -> PreTrainedTokenizerFast:
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab = {token: index for index, token in enumerate([*specials, *_WORDS, "."])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))  # noqa: S106 -- a vocabulary entry, not a secret
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",  # noqa: S106
        unk_token="[UNK]",  # noqa: S106
        cls_token="[CLS]",  # noqa: S106
        sep_token="[SEP]",  # noqa: S106
        model_max_length=_MAX_LENGTH,
    )


class _EntailmentModel:
    """Returns one (entailment, neutral, contradiction) logit row per input row and keeps the inputs."""

    def __init__(self) -> None:
        self.inputs: dict[str, torch.Tensor] = {}

    def __call__(self, **inputs: torch.Tensor) -> SimpleNamespace:
        self.inputs = inputs
        rows = inputs["input_ids"].shape[0]
        return SimpleNamespace(logits=torch.arange(rows * 3, dtype=torch.float32).view(rows, 3))


def _adapter(tokenizer: PreTrainedTokenizerFast) -> tuple[NLIClassificationFlashAdapter, _EntailmentModel]:
    adapter = NLIClassificationFlashAdapter("test-model", hypothesis_template=_TEMPLATE, max_length=_MAX_LENGTH)
    model = _EntailmentModel()
    adapter._tokenizer = tokenizer
    adapter._model = model
    adapter._device = "cpu"
    adapter._entailment_idx = 0
    return adapter, model


def _pair_tokens(tokenizer: PreTrainedTokenizerFast, text: str, labels: list[str]) -> int:
    return sum(
        len(tokenizer(text, _TEMPLATE.format(label), truncation=True, max_length=_MAX_LENGTH)["input_ids"])
        for label in labels
    )


def test_counts_every_pair_of_a_multi_label_item_without_padding() -> None:
    tokenizer = _tokenizer()
    adapter, model = _adapter(tokenizer)
    texts = ["the app crashes on startup when config file is missing", "i was charged twice"]
    labels = ["bug report", "billing", "feature request"]

    output = adapter.extract([Item(text=text) for text in texts], labels=labels)

    expected = [_pair_tokens(tokenizer, text, labels) for text in texts]
    assert output.input_token_counts == expected
    # Rows of different lengths were padded to one width in a single forward,
    # and the padding is not counted.
    rows, width = model.inputs["input_ids"].shape
    assert rows == len(texts) * len(labels)
    assert rows * width > sum(expected)
    assert sum(expected) == int(model.inputs["attention_mask"].sum())


def test_counts_a_truncated_long_item_at_the_window() -> None:
    tokenizer = _tokenizer()
    adapter, _ = _adapter(tokenizer)
    long_text = " ".join(["the app crashes on startup"] * 40)
    labels = ["bug", "billing"]

    output = adapter.extract([Item(text=long_text), Item(text="i was charged twice")], labels=labels)

    assert output.input_token_counts is not None
    assert output.input_token_counts[0] == _pair_tokens(tokenizer, long_text, labels)
    assert output.input_token_counts[0] == _MAX_LENGTH * len(labels)
    assert output.input_token_counts[1] == _pair_tokens(tokenizer, "i was charged twice", labels)


def test_count_is_positional_per_item() -> None:
    tokenizer = _tokenizer()
    adapter, _ = _adapter(tokenizer)
    labels = ["bug", "billing", "feature request", "report"]

    one = adapter.extract([Item(text="i was charged twice")], labels=labels)
    two = adapter.extract([Item(text="the app crashes on startup"), Item(text="i was charged twice")], labels=labels)

    assert one.input_token_counts is not None
    assert two.input_token_counts is not None
    assert two.input_token_counts[1] == one.input_token_counts[0]


def _labels(count: int) -> list[str]:
    return [f"label {index}" for index in range(count)]


def test_flash_adapter_accepts_the_label_limit() -> None:
    adapter, _ = _adapter(_tokenizer())

    output = adapter.extract([Item(text="i was charged twice")], labels=_labels(MAX_EXTRACT_LABELS))

    assert output.classifications is not None
    assert len(output.classifications[0]) == MAX_EXTRACT_LABELS


def test_flash_adapter_refuses_more_labels_than_the_extract_limit() -> None:
    adapter = NLIClassificationFlashAdapter("test-model")

    with pytest.raises(InvalidInputError, match=f"at most {MAX_EXTRACT_LABELS} labels"):
        adapter.extract([Item(text="i was charged twice")], labels=_labels(MAX_EXTRACT_LABELS + 1))


def test_batch_cost_counts_every_row_an_item_runs() -> None:
    adapter = NLIClassificationFlashAdapter("test-model", hypothesis_template=_TEMPLATE, max_length=_MAX_LENGTH)
    texts = ["i was charged twice", "the app crashes on startup"]
    labels = ["bug", "billing", "feature request"]

    costs = adapter.extract_item_costs([Item(text=text) for text in texts], labels=labels)

    assert costs == [sum(len(text) + len(_TEMPLATE) + len(label) for label in labels) for text in texts]
    custom = adapter.extract_item_costs(
        [Item(text=texts[0])], labels=labels, options={"hypothesis_template": "It is {}"}
    )
    assert custom == [sum(len(texts[0]) + len("It is {}") + len(label) for label in labels)]


def test_batch_cost_caps_each_row_at_the_window() -> None:
    adapter = NLIClassificationFlashAdapter("test-model", max_length=_MAX_LENGTH)
    long_text = "the app crashes on startup " * 200

    (cost,) = adapter.extract_item_costs([Item(text=long_text)], labels=["bug", "billing"]) or [0]

    assert cost == 2 * _MAX_LENGTH * 4


def test_batch_cost_splits_many_label_items_the_character_count_would_pack() -> None:
    config = load_model_configs(_MODELS_DIR)["MoritzLaurer/deberta-v3-large-zeroshot-v2.0"]
    budget = config.profiles["default"].max_batch_tokens
    adapter = NLIClassificationFlashAdapter("test-model")
    items = [Item(text="I still have not received my new card, I ordered over a week ago.") for _ in range(16)]
    labels = [f"intent number {index}" for index in range(77)]

    default = build_extract_prepared_items(items)
    paired = build_extract_prepared_items(items, item_costs=adapter_extract_item_costs(adapter, items, labels=labels))

    assert sum(item.cost for item in default) <= budget
    assert sum(item.cost for item in paired) > budget
    assert paired[0].cost == 77 * default[0].cost + sum(len("This text is about {}.") + len(label) for label in labels)


def test_batch_cost_defers_to_extract_for_requests_it_refuses() -> None:
    adapter = NLIClassificationFlashAdapter("test-model")
    items = [Item(text="i was charged twice")]

    assert adapter.extract_item_costs(items, labels=None) is None
    assert adapter.extract_item_costs(items, labels=_labels(MAX_EXTRACT_LABELS + 1)) is None
