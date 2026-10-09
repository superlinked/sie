"""SequenceClassificationAdapter on tiny random checkpoints, against transformers' text-classification pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sie_server.adapters.sequence_classification.adapter import SequenceClassificationAdapter
from sie_server.api.extract import _build_response
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.types.inputs import InvalidInputError, Item
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    PreTrainedTokenizerFast,
    pipeline,
)

WORDS = ["good", "bad", "stock", "price", "rose", "fell", "the", "a", "market", "today", "very", "not", "."]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
WINDOW = 16
TEXTS = [
    "the stock price rose today .",
    "the market fell very bad",
    "good",
    "",
    "not good not bad " * 20,
]


def _checkpoint(
    root: Path,
    labels: list[str],
    *,
    problem_type: str | None = None,
    seed: int = 0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(VOCAB) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True, model_max_length=WINDOW)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        problem_type=problem_type,
    )
    torch.manual_seed(seed)
    model = BertForSequenceClassification(config).eval()
    with torch.no_grad():
        model.classifier.weight.normal_(0.0, 2.0)
    model.save_pretrained(root)
    tokenizer.save_pretrained(root)
    return root


@pytest.fixture(scope="module")
def single(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _checkpoint(tmp_path_factory.mktemp("single"), ["positive", "negative", "neutral"])


@pytest.fixture(scope="module")
def multi(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _checkpoint(
        tmp_path_factory.mktemp("multi"),
        ["toxic", "insult", "threat", "obscene"],
        problem_type="multi_label_classification",
        seed=1,
    )


def _adapter(path: Path, **kwargs) -> SequenceClassificationAdapter:
    adapter = SequenceClassificationAdapter(path, max_seq_length=kwargs.pop("max_seq_length", WINDOW), **kwargs)
    adapter.load("cpu")
    return adapter


def _reference(path: Path, texts: list[str]) -> list[dict[str, float]]:
    pipe = pipeline("text-classification", model=str(path), device="cpu")
    rows = pipe(texts, top_k=None, truncation=True, max_length=WINDOW)
    return [{entry["label"]: entry["score"] for entry in row} for row in rows]


def _scores(output, i: int) -> dict[str, float]:
    return {entry["label"]: entry["score"] for entry in output.classifications[i]}


@pytest.mark.parametrize("fixture", ["single", "multi"])
def test_scores_match_the_text_classification_pipeline(fixture: str, request: pytest.FixtureRequest) -> None:
    path = request.getfixturevalue(fixture)
    adapter = _adapter(path)
    output = adapter.extract([Item(text=text) for text in TEXTS])
    reference = _reference(path, TEXTS)
    tokenizer = BertTokenizerFast.from_pretrained(path)
    for i, expected in enumerate(reference):
        actual = _scores(output, i)
        assert actual.keys() == expected.keys()
        for label, score in expected.items():
            assert actual[label] == pytest.approx(score, abs=1e-6)
        ranked = [entry["score"] for entry in output.classifications[i]]
        assert ranked == sorted(ranked, reverse=True)
    assert output.input_token_counts == [
        len(tokenizer(text, truncation=True, max_length=WINDOW)["input_ids"]) for text in TEXTS
    ]
    assert output.errors is None
    assert output.entities == [[] for _ in TEXTS]


def test_single_label_softmax_and_multi_label_sigmoid(single: Path, multi: Path) -> None:
    softmax = _adapter(single).extract([Item(text="the stock rose")])
    assert sum(_scores(softmax, 0).values()) == pytest.approx(1.0, abs=1e-6)
    sigmoid = _adapter(multi).extract([Item(text="the stock rose")])
    assert sum(_scores(sigmoid, 0).values()) != pytest.approx(1.0, abs=1e-3)


def test_multi_label_option_overrides_the_checkpoint(single: Path, multi: Path) -> None:
    text = [Item(text="the market fell")]
    one = _adapter(single)
    as_sigmoid = _scores(one.extract(text, options={"multi_label": True}), 0)
    logits = one._forward(one._encode(["the market fell"])[0])[0]
    for i, label in enumerate(one._labels):
        assert as_sigmoid[label] == pytest.approx(torch.sigmoid(logits[i]).item(), abs=1e-6)
    many = _adapter(multi)
    as_softmax = _scores(many.extract(text, options={"multi_label": False}), 0)
    assert sum(as_softmax.values()) == pytest.approx(1.0, abs=1e-6)


def test_one_class_head_is_always_a_sigmoid(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "one", ["relevant"])
    adapter = _adapter(path)
    for options in (None, {"multi_label": False}):
        score = adapter.extract([Item(text="good stock")], options=options).classifications[0][0]["score"]
        assert 0.0 < score < 1.0
        assert score == pytest.approx(_reference(path, ["good stock"])[0]["relevant"], abs=1e-6)


def test_regression_heads_and_duplicate_names_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Regression"):
        _adapter(_checkpoint(tmp_path / "reg", ["score"], problem_type="regression"))
    with pytest.raises(ValueError, match="unique"):
        _adapter(_checkpoint(tmp_path / "dup", ["same", "same"]))


def test_labels_select_classes_without_renormalizing(single: Path) -> None:
    adapter = _adapter(single)
    items = [Item(text="the price rose")]
    full = _scores(adapter.extract(items), 0)
    subset = adapter.extract(items, labels=["neutral", "positive"])
    assert {entry["label"] for entry in subset.classifications[0]} == {"neutral", "positive"}
    for entry in subset.classifications[0]:
        assert entry["score"] == full[entry["label"]]


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        (["bullish"], "not one of this model's classes"),
        (["positive", "positive"], "unique"),
        ([], "non-empty"),
        ([7], "not one of this model's classes"),
    ],
)
def test_invalid_labels_are_refused(single: Path, labels: list, match: str) -> None:
    with pytest.raises(InvalidInputError, match=match):
        _adapter(single).extract([Item(text="good")], labels=labels)


def test_unknown_label_message_is_bounded(single: Path) -> None:
    with pytest.raises(InvalidInputError) as error:
        _adapter(single).extract([Item(text="good")], labels=["x" * 10_000])
    assert len(str(error.value)) < 400


def test_top_k_and_threshold(single: Path, multi: Path) -> None:
    adapter = _adapter(multi)
    items = [Item(text="the market fell very bad")]
    scores = sorted(_scores(adapter.extract(items), 0).values(), reverse=True)
    top = adapter.extract(items, options={"top_k": 2}).classifications[0]
    assert [entry["score"] for entry in top] == scores[:2]
    cut = scores[1]
    kept = adapter.extract(items, options={"threshold": cut}).classifications[0]
    assert [entry["score"] for entry in kept] == [s for s in scores if s >= cut]
    assert adapter.extract(items, options={"threshold": 1.0}).classifications[0] == []
    both = _adapter(single).extract(items, labels=["neutral"], options={"top_k": 3, "threshold": 0.0})
    assert [entry["label"] for entry in both.classifications[0]] == ["neutral"]


@pytest.mark.parametrize(
    "options",
    [
        {"top_k": 0},
        {"top_k": True},
        {"top_k": 1.5},
        {"threshold": -0.1},
        {"threshold": 1.5},
        {"threshold": float("nan")},
        {"threshold": "0.5"},
        {"multi_label": "yes"},
        {"hypothesis_template": "This text is about {}."},
    ],
)
def test_invalid_options_are_refused(single: Path, options: dict) -> None:
    with pytest.raises(InvalidInputError):
        _adapter(single).extract([Item(text="good")], options=options)


def test_schema_and_instruction_are_refused(single: Path) -> None:
    adapter = _adapter(single)
    with pytest.raises(InvalidInputError):
        adapter.extract([Item(text="good")], instruction="Classify the sentiment.")
    with pytest.raises(InvalidInputError):
        adapter.extract([Item(text="good")], output_schema={"type": "object"})


def test_long_text_is_truncated_like_the_pipeline(single: Path) -> None:
    adapter = _adapter(single)
    text = "the stock price rose today and the market fell " * 40
    output = adapter.extract([Item(text=text)], options={"overflow_policy": "truncate_text"})
    assert output.input_token_counts == [WINDOW]
    expected = _reference(single, [text])[0]
    for label, score in _scores(output, 0).items():
        assert score == pytest.approx(expected[label], abs=1e-6)
    default = adapter.extract([Item(text=text)])
    assert default.classifications == output.classifications


def test_error_overflow_policy_fails_only_the_long_item(single: Path) -> None:
    adapter = _adapter(single)
    fits = "the stock price rose today ."  # 6 words + 2 special tokens
    exact = " ".join(["good"] * (WINDOW - 2))
    over = " ".join(["good"] * (WINDOW - 1))
    output = adapter.extract([Item(text=fits), Item(text=over), Item(text=exact)], options={"overflow_policy": "error"})
    assert output.errors is not None
    assert output.errors[0] is None
    assert output.errors[1].code == "INPUT_TOO_LONG"
    assert output.errors[2] is None
    assert output.classifications[1] == []
    assert output.input_token_counts == [8, 0, WINDOW]


def test_item_without_text_fails_alone(single: Path) -> None:
    output = _adapter(single).extract([Item(text="good"), Item()])
    assert output.errors is not None
    assert output.errors[0] is None
    assert output.errors[1].code == "INVALID_INPUT"
    assert output.classifications[1] == []
    assert output.input_token_counts[1] == 0
    assert output.classifications[0]


def test_cut_text_tokenizes_to_the_whole_texts_prefix(single: Path) -> None:
    adapter = _adapter(single)
    tokenizer = BertTokenizerFast.from_pretrained(single)
    texts = [
        "the market rose . " * 400,  # cut once
        ("\n \t  " * 30 + "good ") * 200,  # whitespace runs: the cut is widened
        "x" * 5000,  # no whitespace: read whole
        "good " * 60 + "y" * 5000,  # a giant word past the window
        "good bad",
    ]
    ids, overflowed = adapter._encode(texts)
    for text, row, overflow in zip(texts, ids, overflowed, strict=True):
        assert row == tokenizer(text, truncation=True, max_length=WINDOW)["input_ids"]
        assert overflow == (len(tokenizer(text)["input_ids"]) > WINDOW)
    costs = adapter.extract_item_costs([Item(text=text) for text in texts])
    limit = WINDOW * 16
    assert costs is not None
    assert costs[0] <= limit
    assert costs[2] == 5000
    assert costs[4] == len(texts[4])


def test_tokenizers_that_do_not_split_at_whitespace_read_whole_texts(tmp_path: Path) -> None:
    unk, pad = "[UNK]", "[PAD]"
    backend = Tokenizer(models.WordLevel({unk: 0, pad: 1}, unk_token=unk))
    backend.pre_tokenizer = pre_tokenizers.Metaspace(split=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=unk, pad_token=pad)
    assert not SequenceClassificationAdapter._splits_at_whitespace(tokenizer)
    bert = BertTokenizerFast(vocab_file=str(_checkpoint(tmp_path / "vocab", ["a", "b"]) / "vocab.txt"))
    assert SequenceClassificationAdapter._splits_at_whitespace(bert)


def test_forward_chunks_do_not_change_scores(single: Path) -> None:
    items = [Item(text=text) for text in TEXTS * 3]
    whole = _adapter(single, max_forward_tokens=1 << 20).extract(items)
    chunked = _adapter(single, max_forward_tokens=WINDOW).extract(items)
    for a, b in zip(whole.classifications, chunked.classifications, strict=True):
        assert [entry["label"] for entry in a] == [entry["label"] for entry in b]
        for x, y in zip(a, b, strict=True):
            assert x["score"] == pytest.approx(y["score"], abs=1e-5)
    assert whole.input_token_counts == chunked.input_token_counts


def test_max_seq_length_is_clamped_to_the_tokenizer(single: Path) -> None:
    adapter = _adapter(single, max_seq_length=10_000)
    assert adapter._max_length == WINDOW
    assert _adapter(single, max_seq_length=8)._max_length == 8
    with pytest.raises(ValueError, match="max_seq_length"):
        SequenceClassificationAdapter(single, max_seq_length=1)
    with pytest.raises(ValueError, match="max_forward_tokens"):
        SequenceClassificationAdapter(single, max_forward_tokens=0)


def test_costs_and_extract_require_a_loaded_model(single: Path) -> None:
    adapter = SequenceClassificationAdapter(single, max_seq_length=WINDOW)
    assert adapter.extract_item_costs([Item(text="good")]) is None
    with pytest.raises(RuntimeError):
        adapter.extract([Item(text="good")])
    adapter.load("cpu")
    adapter.unload()
    with pytest.raises(RuntimeError):
        adapter.extract([Item(text="good")])


def test_wire_response_carries_classifications_errors_and_usage(single: Path) -> None:
    adapter = _adapter(single)
    items = [Item(id="a", text="the stock rose"), Item(id="b", text="good " * 40)]
    output = adapter.extract(items, options={"overflow_policy": "error", "top_k": 1})
    wire = _build_response("acme/classifier", items, ExtractHandler.format_output(output))
    first, second = wire["items"]
    assert len(first["classifications"]) == 1
    assert first["classifications"][0]["label"] in {"positive", "negative", "neutral"}
    assert second["error"]["code"] == "INPUT_TOO_LONG"


def test_capacity_reads_the_tokenizer_and_the_position_table() -> None:
    from types import SimpleNamespace

    capacity = SequenceClassificationAdapter._capacity
    sentinel = SimpleNamespace(model_max_length=int(1e30))
    roberta = SimpleNamespace(model_type="roberta", max_position_embeddings=514, pad_token_id=1)
    assert capacity(sentinel, roberta) == 512
    assert capacity(SimpleNamespace(model_max_length=128), roberta) == 128
    bert = SimpleNamespace(model_type="bert", max_position_embeddings=512, pad_token_id=0)
    assert capacity(sentinel, bert) == 512
    assert capacity(sentinel, SimpleNamespace(model_type="custom")) > 1 << 30


def test_default_window_comes_from_the_checkpoint(single: Path) -> None:
    adapter = SequenceClassificationAdapter(single)
    adapter.load("cpu")
    assert adapter._max_length == WINDOW


def test_forward_passes_stay_within_budget_and_bound_padding() -> None:
    from sie_server.adapters.sequence_classification.adapter import _chunks

    assert _chunks([512] * 64, 16384) == [(0, 32), (32, 32)]
    assert _chunks([20] * 64, 16384) == [(0, 64)]
    assert _chunks([], 16384) == []
    assert _chunks([600], 100) == [(0, 1)]
    lengths = [512] * 7 + [418] + [210] * 6 + [106] * 5 + [54] * 6 + [28] * 4 + [15] * 3
    chunks = _chunks(lengths, 16384)
    assert sum(count for _, count in chunks) == len(lengths)
    assert all(lengths[start] * count <= 16384 for start, count in chunks)
    padded = sum(lengths[start] * count for start, count in chunks)
    assert padded < 0.7 * 512 * len(lengths)


def test_metering_fallback_counts_the_effective_window(single: Path) -> None:
    adapter = _adapter(single, max_seq_length=10_000)
    texts = ["good " * 100, "bad"]
    counted = adapter.count_input_tokens([Item(text=text) for text in texts])
    assert counted == adapter.extract([Item(text=text) for text in texts]).input_token_counts
