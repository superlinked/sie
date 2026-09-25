"""GLiNER2.5-Decide adapter: request contract, bounds, errors, metering, and the ModernBERT RoPE check.

The adapter's model and processor are replaced by fakes that follow gliner2
2.0's prompt layout; the real processor and weights are covered by the
``model``-marked parity tests in ``test_gliner2_decide_parity.py``.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import transformers
import yaml
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.gliner2 import decide as decide_module
from sie_server.adapters.gliner2.decide import (
    GLiNER2DecideAdapter,
    declared_rope_thetas,
    loadable_checkpoint,
    loaded_rope_thetas,
    transformers4_encoder_config,
    transformers4_tokenizer_config,
    verify_encoder_rope,
)
from sie_server.adapters.gliner2.decisions import (
    MAX_LABELS,
    MAX_LABELS_PER_TASK,
    MAX_SCHEMA_CHARS,
    MAX_TASKS,
    answer,
    parse_request,
    probabilities,
)
from sie_server.adapters.gliner2.words import PACKAGE_PATTERN, LinearWordSplitter
from sie_server.types.inputs import InvalidInputError, Item

SIE_SERVER = Path(__file__).resolve().parents[2]
SPECIAL = ("[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]", "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]")

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {"refund": "wants money back", "cancel": None, "other": ""},
    },
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
    "needs_human": {"type": "noul", "instructions": "Must a person act?", "criteria": {"true": "escalate"}},
}


# ---------------------------------------------------------------------------
# Fakes following gliner2 2.0's processor layout
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Pieces of at most three characters (markers stay whole); ids are stable per piece."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {token: 1000 + i for i, token in enumerate(SPECIAL)}
        self.tokenized_chars = 0

    def tokenize(self, text: str) -> list[str]:
        self.tokenized_chars += len(text)
        pieces: list[str] = []
        for part in re.split(r"(\[[A-Z_]+\])", text):
            if part in self.vocab and part in SPECIAL:
                pieces.append(part)
            else:
                pieces.extend(part[i : i + 3] for i in range(0, len(part), 3) if part[i : i + 3].strip())
        return pieces

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.setdefault(token, 2000 + len(self.vocab)) for token in tokens]


class WhitespaceTokenSplitter:
    """gliner2 2.0.0's word splitter (the package's regex, each word lowercased): the reference rows use it."""

    _PATTERN = PACKAGE_PATTERN

    def __call__(self, text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
        for match in self._PATTERN.finditer(text):
            word = match.group()
            yield (word.lower() if lower else word), match.start(), match.end()


class FakeProcessor:
    """``transform_and_format`` and ``collate`` (one row) as gliner2 2.0's ``SchemaTransformer`` builds them."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.word_splitter = WhitespaceTokenSplitter()
        self.reference_splitter = WhitespaceTokenSplitter()
        self._tokenize_cached = lru_cache(maxsize=100)(self.tokenizer.tokenize)

    def change_mode(self, is_training: bool) -> None:
        assert not is_training

    @staticmethod
    def _schemas(schema: dict[str, Any]) -> list[list[str]]:
        schemas = []
        for entry in schema["classifications"]:
            prompt = entry["task"] + (f": {entry['prompt']}" if entry.get("prompt") else "")
            for label, description in entry.get("label_descriptions", {}).items():
                prompt += f" [DESCRIPTION] {label}: {description}"
            labels = [token for label in entry["labels"] for token in ("[L]", label)]
            schemas.append(["(", "[P]", prompt, "(", *labels, ")", ")"])
        return schemas

    def transform_and_format(self, text: str, schema: dict[str, Any]) -> SimpleNamespace:
        words = [word for word, _, _ in self.reference_splitter(text, lower=True)]
        return self._format(self._schemas(schema), words)

    def collate_row(self, text: str, schema: dict[str, Any], max_len: int | None) -> list[int]:
        """``collate_fn_inference([(text, schema)], max_len)``'s input ids."""
        if not text.endswith((".", "!", "?")):
            text += "."
        words = [word for word, _, _ in self.reference_splitter(text, lower=True)]
        if max_len is not None:
            words = words[:max_len]
        return self._format(self._schemas(schema), words).input_ids

    def _format(self, schemas: list[list[str]], words: list[str]) -> SimpleNamespace:
        combined: list[str] = []
        for struct in schemas:
            combined.extend([*struct, "[SEP_STRUCT]"])
        combined.pop()
        combined.append("[SEP_TEXT]")
        combined.extend(words)
        markers: set[int] = set()
        offset = 0
        for struct in schemas:
            markers.add(offset + 1)
            markers.update(offset + index for index in range(4, len(struct) - 2, 2))
            offset += len(struct) + 1
        subwords: list[str] = []
        positions: list[list[int]] = [[] for _ in schemas]
        first_positions: list[int] = []
        schema_index, in_text = 0, False
        for index, element in enumerate(combined):
            position = len(subwords)
            subwords.extend(self._tokenize_cached(element))
            if element == "[SEP_TEXT]":
                in_text = True
            elif in_text:
                first_positions.append(position)
            elif element == "[SEP_STRUCT]":
                schema_index += 1
            elif index in markers:
                positions[schema_index].append(position)
        return SimpleNamespace(
            input_ids=self.tokenizer.convert_tokens_to_ids(subwords),
            text_word_first_positions=first_positions,
            schema_special_positions=positions,
        )


class FakeEncoder(torch.nn.Module):
    """Hidden state = (position, sum of the row's ids): label logits depend on marker and row."""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        position = torch.arange(input_ids.shape[1]).expand_as(input_ids).float()
        total = (input_ids * attention_mask).sum(dim=1, keepdim=True).float().expand_as(position)
        return SimpleNamespace(last_hidden_state=torch.stack([position, total], dim=-1))


def fake_classifier(hidden: torch.Tensor) -> torch.Tensor:
    return hidden[..., :1] * 0.01 + hidden[..., 1:2] * 1e-7


def expected_logits(row: list[int], positions: list[int]) -> list[float]:
    return [position * 0.01 + sum(row) * 1e-7 for position in positions]


def make_adapter(window: int = 256, **kwargs: Any) -> tuple[GLiNER2DecideAdapter, FakeProcessor]:
    adapter = GLiNER2DecideAdapter("fake/decide", max_seq_length=window, **kwargs)
    processor = FakeProcessor()
    adapter._attach(SimpleNamespace(encoder=FakeEncoder(), classifier=fake_classifier), processor, "cpu")
    return adapter, processor


def spy_rows(adapter: GLiNER2DecideAdapter, monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Record the rows the adapter scores."""
    seen: list[list[int]] = []
    original = adapter._score

    def score(rows: list[list[int]], positions: list[int]) -> np.ndarray:
        seen.extend(rows)
        return original(rows, positions)

    monkeypatch.setattr(adapter, "_score", score)
    return seen


# ---------------------------------------------------------------------------
# Request contract
# ---------------------------------------------------------------------------


def test_laya_questions_map_to_decide_tasks() -> None:
    request = parse_request(labels=None, output_schema=QUESTIONS, instruction=None, options={})

    intent, urgency, noul = request.tasks
    assert request.mode == "questions"
    assert intent.model_entry() == {
        "task": "intent",
        "labels": ["refund", "cancel", "other"],
        "true_label": ["N/A"],
        "multi_label": False,
        "cls_threshold": 0.5,
        "class_act": "auto",
        "prompt": "What does the customer want?",
        "label_descriptions": {"refund": "wants money back"},
    }
    assert urgency.labels == ("0", "1", "2")
    assert urgency.model_entry()["label_descriptions"] == {"0": "low", "1": "medium", "2": "high"}
    assert noul.labels == ("yes", "no")
    assert noul.model_entry()["label_descriptions"] == {"yes": "escalate"}
    assert request.model_schema()["classifications"] == [task.model_entry() for task in request.tasks]


def test_noul_labels_wording_and_numeric_score_levels() -> None:
    request = parse_request(
        labels=None,
        output_schema={
            "handoff": {"type": "noul", "instructions": "Hand off?", "labels": {"false": "keep", "true": "handoff"}},
            "rating": {"type": "score", "instructions": "Rate it", "criteria": [str(i) for i in range(11)]},
            "tone": {"type": "score", "instructions": "", "criteria": [{"calm": True}, "angry"]},
        },
        instruction=None,
        options={},
    )
    handoff, rating, tone = request.tasks
    assert handoff.labels == ("handoff", "keep")
    assert "label_descriptions" not in rating.model_entry()  # levels that repeat their index
    assert tone.prompt is None
    assert tone.descriptions == ('{"calm": true}', "angry")  # structured criteria render as Laya renders them


def test_label_groups_and_labels_modes() -> None:
    groups = parse_request(
        labels=None,
        output_schema=None,
        instruction="Triage the ticket.",
        options={"label_groups": {"topic": ["billing", "bug"], "urgency": [" low ", "high"]}},
    )
    assert groups.mode == "groups"
    assert [task.prompt for task in groups.tasks] == ["Triage the ticket.", "Triage the ticket."]
    assert groups.tasks[1].labels == ("low", "high")  # the model reads stripped labels...
    assert groups.tasks[1].keys == (" low ", "high")  # ...answers echo the caller's

    labels = parse_request(
        labels=["spam", "ham"],
        output_schema=None,
        instruction=None,
        options={"classification_type": "multi-label", "classification_task": "verdict"},
    )
    assert labels.mode == "labels"
    assert labels.tasks[0].name == "verdict"
    assert labels.tasks[0].multi_label
    default = parse_request(labels=["a", "b"], output_schema=None, instruction=None, options={})
    assert default.tasks[0].name == "label"
    assert default.tasks[0].prompt is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"labels": ["a"], "output_schema": QUESTIONS}, "either output_schema"),
        ({"output_schema": QUESTIONS, "options": {"label_groups": {"g": ["a"]}}}, "either output_schema"),
        ({"output_schema": QUESTIONS, "instruction": "x"}, "own 'instructions'"),
        ({"labels": ["a"], "options": {"label_groups": {"g": ["a"]}}}, "either labels or options.label_groups"),
        ({}, "requires typed questions"),
        ({"output_schema": {}}, "requires typed questions"),
        ({"output_schema": ["x"]}, "must map question ids"),
        ({"output_schema": {"q": {"type": "maybe", "instructions": "x"}}}, "unknown type"),
        ({"output_schema": {"q": {"type": "choice", "instructions": "x", "criteria": []}}}, "at least one criterion"),
        ({"output_schema": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", " a"]}}}, "unique"),
        ({"output_schema": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", " "]}}}, "non-empty"),
        ({"output_schema": {" ": {"type": "noul", "instructions": "x"}}}, "question id must be a non-empty"),
        (
            {
                "output_schema": {
                    "a": {"type": "noul", "instructions": "x"},
                    " a": {"type": "noul", "instructions": "y"},
                }
            },
            "must be unique",
        ),
        ({"labels": ["a", "b"], "options": {"group_encoding": "separate"}}, "group_encoding"),
        ({"labels": ["a", "b"], "options": {"examples": [{"text": "x", "labels": ["a"]}]}}, "few-shot"),
        ({"labels": ["a", "b"], "options": {"threshold": 1.5}}, "threshold"),
        ({"labels": ["a", "b"], "options": {"threshold": True}}, "threshold"),
        ({"labels": ["a", "b"], "options": {"classification_type": "multi"}}, "classification_type"),
        (
            {"labels": ["a", "b"], "options": {"classification_type": "multi-label", "multi_label": False}},
            "contradicts",
        ),
        ({"labels": "a"}, "must be a list"),
        ({"labels": ["a", 3]}, "non-empty string"),
        ({"options": {"label_groups": {"g": []}}}, "non-empty list"),
        ({"options": {"label_groups": []}}, "non-empty object"),
        ({"labels": ["a"], "options": {"classification_task": " "}}, "classification_task"),
    ],
)
def test_malformed_requests_are_invalid_input(kwargs: dict[str, Any], message: str) -> None:
    arguments = {"labels": None, "output_schema": None, "instruction": None, "options": {}, **kwargs}
    with pytest.raises(InvalidInputError, match=re.escape(message)):
        parse_request(**arguments)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels": ["a", "b[L]"]},
        {"labels": ["a", "b"], "instruction": "Pick [DESCRIPTION] one"},
        {"output_schema": {"q[P]": {"type": "noul", "instructions": "x"}}},
        {"output_schema": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "[SEP_TEXT] b"}}}},
        {"options": {"label_groups": {"g[E]": ["a", "b"]}}},
    ],
)
def test_prompt_markers_are_refused(kwargs: dict[str, Any]) -> None:
    arguments = {"labels": None, "output_schema": None, "instruction": None, "options": {}, **kwargs}
    with pytest.raises(InvalidInputError, match="structural token"):
        parse_request(**arguments)


def test_bounds_are_checked_before_tokenizing() -> None:
    def parse(**kwargs: Any) -> None:
        parse_request(**{"labels": None, "output_schema": None, "instruction": None, "options": {}, **kwargs})

    many = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(MAX_TASKS + 1)}
    with pytest.raises(InvalidInputError, match=f"at most {MAX_TASKS} questions"):
        parse(output_schema=many)
    wide = {"q": {"type": "choice", "instructions": "x", "criteria": [f"l{i}" for i in range(MAX_LABELS_PER_TASK + 1)]}}
    with pytest.raises(InvalidInputError, match=f"at most {MAX_LABELS_PER_TASK} options"):
        parse(output_schema=wide)
    with pytest.raises(InvalidInputError, match=f"at most {MAX_LABELS_PER_TASK} labels"):
        parse(labels=[f"l{i}" for i in range(MAX_LABELS_PER_TASK + 1)])
    groups = {f"g{i}": [f"l{j}" for j in range(40)] for i in range(MAX_LABELS // 40 + 1)}
    with pytest.raises(InvalidInputError, match=f"at most {MAX_LABELS} labels"):
        parse(options={"label_groups": groups})
    with pytest.raises(InvalidInputError, match="at most 256 characters"):
        parse(labels=["a", "b" * 257])
    with pytest.raises(InvalidInputError, match="at most 2048 characters"):
        parse(labels=["a", "b"], instruction="x" * 2049)
    long_criteria = {
        f"q{i}": {"type": "choice", "instructions": "x" * 2000, "criteria": {"a": "d" * 2000}} for i in range(17)
    }
    with pytest.raises(InvalidInputError, match=f"total at most {MAX_SCHEMA_CHARS}"):
        parse(output_schema=long_criteria)


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def test_answers_have_every_probability_and_laya_confidence() -> None:
    request = parse_request(labels=None, output_schema=QUESTIONS, instruction=None, options={})
    intent, urgency, noul = request.tasks

    p = probabilities(intent, np.array([2.0, 1.0, 0.0], dtype=np.float32))
    choice = answer(intent, p)
    assert choice["choice"] == "refund"
    assert list(choice["probabilities"]) == ["refund", "cancel", "other"]
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0)
    entropy = -sum(v * math.log(v) for v in p)
    assert choice["confidence"] == pytest.approx(1 - entropy / math.log(3))

    score = answer(urgency, probabilities(urgency, np.array([0.0, 0.0, 0.0])))
    assert score["score"] == pytest.approx(1.0)
    assert score["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert score["probabilities"] == pytest.approx({"0": 1 / 3, "1": 1 / 3, "2": 1 / 3})
    assert score["confidence"] == pytest.approx(0.0)

    verdict = answer(noul, probabilities(noul, np.array([0.0, math.log(3.0)])))  # labels ("yes", "no")
    assert verdict == {"type": "noul", "noul": pytest.approx(0.25), "answer": False, "confidence": pytest.approx(0.75)}


def test_multi_label_groups_use_sigmoid_and_threshold() -> None:
    request = parse_request(
        labels=None,
        output_schema=None,
        instruction=None,
        options={"label_groups": {"topics": ["hvac", "billing"]}, "classification_type": "multi-label"},
    )
    task = request.tasks[0]
    p = probabilities(task, np.array([2.0, -1.0]))
    assert p.tolist() == pytest.approx([1 / (1 + math.exp(-2.0)), 1 / (1 + math.exp(1.0))])
    assert answer(task, p) == {"labels": ["hvac"], "probabilities": {"hvac": p[0], "billing": p[1]}}
    assert answer(task, p, threshold=0.2)["labels"] == ["hvac", "billing"]


# ---------------------------------------------------------------------------
# Adapter: rows, outputs, metering, errors
# ---------------------------------------------------------------------------


def test_rows_equal_the_processor_rows_and_logits_come_from_label_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, processor = make_adapter()
    rows = spy_rows(adapter, monkeypatch)
    texts = ["Hi, we were billed twice for March. Refund please", "Login broken since update!"]

    output = adapter.extract([Item(text=text) for text in texts], output_schema=QUESTIONS)

    request = parse_request(labels=None, output_schema=QUESTIONS, instruction=None, options={})
    schema = request.model_schema()
    assert rows == [processor.collate_row(text, schema, None) for text in texts]
    prefix = processor.transform_and_format(".", schema)
    positions = [p for task in prefix.schema_special_positions for p in task[1:]]
    for row, data in zip(rows, output.data or [], strict=True):
        logits = expected_logits(row, positions)
        p = probabilities(request.tasks[0], np.array(logits[:3], dtype=np.float32))
        assert [data["intent"]["probabilities"][key] for key in ("refund", "cancel", "other")] == pytest.approx(p)
    assert output.classifications is None
    assert output.errors is None
    assert list((output.data or [{}])[0]) == ["intent", "urgency", "needs_human"]


def test_metering_bills_document_and_free_text_not_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, processor = make_adapter()
    tokenizer = processor.tokenizer
    text = "the payroll file is late"

    output = adapter.extract([Item(text=text), Item(text=None)], output_schema=QUESTIONS)

    document = sum(len(tokenizer.tokenize(word)) for word, _, _ in WhitespaceTokenSplitter()(text))
    free_text = sum(
        len(tokenizer.tokenize(value))
        for value in (
            "What does the customer want?",
            "wants money back",
            "How urgent is this?",
            "low",
            "medium",
            "high",
            "Must a person act?",
            "escalate",
        )
    )
    assert output.input_token_counts == [document + free_text, 0]  # the processor's appended "." is not billed
    assert output.errors is not None
    assert output.errors[1] is not None
    assert output.errors[1].code == "INVALID_INPUT"
    assert output.data is not None
    assert output.data[1] == {}

    labels_only = adapter.extract([Item(text=text)], labels=["a very long label name", "b"])
    assert labels_only.input_token_counts == [document]


def test_long_documents_are_cut_to_whole_words_in_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, processor = make_adapter(window=96)
    rows = spy_rows(adapter, monkeypatch)
    text = " ".join(f"word{i:04d}" for i in range(500))  # 3 pieces per word

    output = adapter.extract([Item(text=text)], labels=["yes", "no"])

    schema = parse_request(labels=["yes", "no"], output_schema=None, instruction=None, options={}).model_schema()
    prefix_len = processor.transform_and_format(".", schema).text_word_first_positions[0]
    kept = (96 - prefix_len) // 3
    assert rows == [processor.collate_row(text, schema, kept)]
    assert len(rows[0]) <= 96
    assert output.input_token_counts == [3 * kept]
    # Work stops at the window: only the kept words and the one that did not fit were tokenized.
    assert processor.tokenizer.tokenized_chars < 20 * 96


def test_caches_keep_neither_long_words_nor_task_prompts() -> None:
    adapter, processor = make_adapter(window=256)
    long_words = [f"{i:04d}" + "x" * 60 for i in range(20)]

    output = adapter.extract([Item(text=" ".join(["short", "words", *long_words]))], labels=["yes", "no"])

    assert output.input_token_counts is not None
    assert output.input_token_counts[0] > 100  # long words were read...
    assert adapter._word_cache is not None
    assert adapter._word_cache.cache_info().currsize == 2  # ...but only "short" and "words" were kept
    assert processor._tokenize_cached.cache_info().currsize == 0


def test_overflow_policy_error_and_unreadable_items_fail_alone() -> None:
    adapter, _ = make_adapter(window=64)
    items = [
        Item(text="short text"),
        Item(text=" ".join(["word"] * 200)),
        Item(text="x" * 5000),  # one word longer than the adapter reads
        Item(text="   "),
        Item(text="a", metadata={"state": "b"}),
        Item(metadata={"state": 3}),
    ]

    output = adapter.extract(items, labels=["yes", "no"], options={"overflow_policy": "error"})

    codes = [None if error is None else error.code for error in output.errors or []]
    assert codes == [None, "INPUT_TOO_LONG", "INPUT_TOO_LONG", "INVALID_INPUT", "INVALID_INPUT", "INVALID_INPUT"]
    assert output.input_token_counts is not None
    assert output.input_token_counts[0] > 0
    assert output.input_token_counts[1:] == [0, 0, 0, 0, 0]
    assert output.classifications is not None
    assert output.classifications[0]
    assert not any(output.classifications[1:])

    truncated = adapter.extract(items[:2], labels=["yes", "no"], options={"overflow_policy": "truncate_text"})
    assert truncated.errors is None


def test_a_document_missing_only_the_processors_period_counts_as_whole() -> None:
    adapter, processor = make_adapter(window=64)
    schema = parse_request(labels=["yes", "no"], output_schema=None, instruction=None, options={}).model_schema()
    room = 64 - processor.transform_and_format(".", schema).text_word_first_positions[0]
    text = " ".join(["abc"] * room)  # one token per word: the words fill the room exactly

    output = adapter.extract([Item(text=text)], labels=["yes", "no"], options={"overflow_policy": "error"})

    assert output.errors is None
    assert output.input_token_counts == [room]
    longer = adapter.extract([Item(text=text + " abc")], labels=["yes", "no"], options={"overflow_policy": "error"})
    assert longer.errors is not None
    assert longer.errors[0] is not None
    assert longer.errors[0].code == "INPUT_TOO_LONG"


def test_states_render_like_laya_and_conversations_keep_the_newest_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, processor = make_adapter(window=64)
    rows = spy_rows(adapter, monkeypatch)
    turns = [{"role": "user", "content": f"turn {i} " + "blah " * 10} for i in range(30)]

    output = adapter.extract(
        [Item(metadata={"state": {"subject": "Login", "body": "Error 500"}}), Item(metadata={"state": turns})],
        labels=["yes", "no"],
    )

    assert output.errors is None
    schema = parse_request(labels=["yes", "no"], output_schema=None, instruction=None, options={}).model_schema()
    assert rows[0] == processor.collate_row(json.dumps({"subject": "Login", "body": "Error 500"}), schema, None)
    inverse = {token_id: token for token, token_id in processor.tokenizer.vocab.items()}
    prefix_len = processor.transform_and_format(".", schema).text_word_first_positions[0]
    document = [inverse[token_id] for token_id in rows[1][prefix_len:]]
    assert "29" in document  # the newest turn is read...
    assert "0" not in document  # ...the oldest is not
    assert document[-1] == "."
    assert len(rows[1]) <= 64


def test_the_adapter_splits_words_in_linear_time_like_the_package() -> None:
    adapter, processor = make_adapter()
    assert isinstance(processor.word_splitter, LinearWordSplitter)
    assert adapter._word_splitter is processor.word_splitter
    goldens = sorted((Path(__file__).parent / "goldens" / "gliner2_decide").glob("*.json"))
    assert goldens
    texts = [
        text
        for path in goldens
        for text in (*json.loads(path.read_text())["texts"], json.loads(path.read_text())["long_text"])
    ]
    for text in texts:
        assert list(adapter._word_splitter(text)) == list(processor.reference_splitter(text))


def test_an_unknown_word_splitter_is_refused() -> None:
    adapter = GLiNER2DecideAdapter("fake/decide", max_seq_length=256)
    processor = FakeProcessor()
    processor.word_splitter = lambda text, lower=True: iter(())
    with pytest.raises(RuntimeError, match="linear-time equivalent"):
        adapter._attach(SimpleNamespace(encoder=FakeEncoder(), classifier=fake_classifier), processor, "cpu")


@pytest.mark.parametrize(
    "item",
    [
        Item(text="." * (2 * 1024 * 1024)),
        Item(text="a." * (1024 * 1024)),
        Item(text="hello world. " * (2 * 1024 * 1024 // 13)),
        Item(metadata={"state": ["." * (64 * 1024)]}),
        Item(metadata={"state": [{"role": "user", "content": "a." * (32 * 1024)}] * 3}),
    ],
    ids=["dots-2MiB", "a-dots-2MiB", "prose-2MiB", "dots-state-64KiB", "a-dots-turns"],
)
def test_pathological_documents_are_read_in_bounded_time(item: Item) -> None:
    adapter, _ = make_adapter(window=2048)
    adapter.extract([Item(text="warm up")], labels=["yes", "no"])
    started = time.perf_counter()
    output = adapter.extract([item], labels=["yes", "no"])
    elapsed = time.perf_counter() - started
    assert output.errors is None
    assert output.input_token_counts is not None
    assert 0 < output.input_token_counts[0] <= 2048
    # gliner2's own splitter takes 15 to 30 seconds on these; the adapter takes tens of milliseconds.
    assert elapsed < 1.0


def test_reading_from_the_end_keeps_the_tail_of_an_overlong_run(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, processor = make_adapter(window=64)
    rows = spy_rows(adapter, monkeypatch)
    turns = ["older turn " * 20, "x" + "." * 10_000 + "tail"]

    output = adapter.extract([Item(metadata={"state": turns})], labels=["yes", "no"])

    assert output.errors is None
    inverse = {token_id: token for token, token_id in processor.tokenizer.vocab.items()}
    prefix_len = processor.transform_and_format(
        ".", parse_request(labels=["yes", "no"], output_schema=None, instruction=None, options={}).model_schema()
    ).text_word_first_positions[0]
    document = [inverse[token_id] for token_id in rows[0][prefix_len:]]
    # The newest turn's run ('"x....tail"]' and the processor's ".") is longer than the
    # adapter reads, so only its tail is read, and nothing of the older turn.
    assert document[-5:] == ["tai", "l", '"', "]", "."]
    assert set(document[:-5]) == {"."}
    assert "old" not in document


def test_a_conversation_cut_inside_an_overlong_run_is_not_whole() -> None:
    adapter, _ = make_adapter(window=2048)
    item = Item(metadata={"state": ["older turn", "x" * 5000]})  # the newest run's tail fits in the room

    read = adapter.extract([item], labels=["yes", "no"])
    strict = adapter.extract([item], labels=["yes", "no"], options={"overflow_policy": "error"})

    assert read.errors is None
    assert strict.errors is not None
    assert strict.errors[0] is not None
    assert strict.errors[0].code == "INPUT_TOO_LONG"


def test_label_groups_answer_like_gliclass(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = make_adapter()
    output = adapter.extract(
        [Item(text="AC broken, move me tonight")],
        options={"label_groups": {"intent": ["room_change", "billing"], "priority": ["low", "high"]}, "threshold": 0.0},
    )
    data = (output.data or [{}])[0]
    assert set(data) == {"intent", "priority"}
    assert data["intent"]["type"] == "choice"
    assert set(data["intent"]["probabilities"]) == {"room_change", "billing"}
    labels = [c["label"] for c in (output.classifications or [[]])[0]]
    assert sorted(labels) == ["intent.billing", "intent.room_change", "priority.high", "priority.low"]
    scores = [c["score"] for c in (output.classifications or [[]])[0]]
    assert scores == sorted(scores, reverse=True)


def test_labels_mode_returns_every_label_sorted_and_threshold_filters() -> None:
    adapter, _ = make_adapter()
    output = adapter.extract([Item(text="hello there")], labels=["a", "b", "c"])
    assert output.data is None
    assert output.classifications is not None
    ranked = output.classifications[0]
    assert [c["label"] for c in ranked] == ["c", "b", "a"]  # later markers get higher fake logits
    assert sum(c["score"] for c in ranked) == pytest.approx(1.0)
    filtered = adapter.extract([Item(text="hello there")], labels=["a", "b", "c"], options={"threshold": 0.335})
    assert [c["label"] for c in (filtered.classifications or [[]])[0]] == ["c"]


def test_prompt_over_budget_is_input_too_long() -> None:
    adapter, _ = make_adapter(window=64, max_prompt_tokens=32)
    with pytest.raises(InputTooLongError, match="takes at most 32"):
        adapter.extract([Item(text="x")], labels=[f"label number {i}" for i in range(10)])


def test_scoring_chunks_preserve_row_order() -> None:
    adapter, _ = make_adapter(window=64, inference_batch_tokens=64)
    rows = [[5] * n for n in (3, 40, 7, 64, 1)]
    positions = [0, 1, 2]
    out = adapter._score(rows, positions)
    np.testing.assert_allclose(out, np.array([expected_logits(row, positions) for row in rows]), rtol=1e-5, atol=1e-9)


def test_non_finite_scores_fail_their_item(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = make_adapter()

    def score(rows: list[list[int]], positions: list[int]) -> np.ndarray:
        out = np.zeros((len(rows), len(positions)), dtype=np.float32)
        out[1, 0] = np.nan
        return out

    monkeypatch.setattr(adapter, "_score", score)
    output = adapter.extract([Item(text="a"), Item(text="b")], labels=["x", "y"])
    assert output.errors is not None
    assert output.errors[1] is not None
    assert output.errors[1].code == "INFERENCE_ERROR"
    assert output.input_token_counts is not None
    assert output.input_token_counts[1] == 0


def test_batching_cost_is_capped_at_the_window() -> None:
    adapter, _ = make_adapter(window=100)
    costs = adapter.extract_item_costs(
        [Item(text="x" * 10_000), Item(text="short"), Item(metadata={"state": {"a": "b"}})],
        labels=["yes", "no"],
    )
    assert costs is not None
    assert costs[0] <= 2 * 100 * 4
    assert costs[1] < costs[0]
    assert adapter.extract_item_costs([Item(text="x")], output_schema={"q": object()}) is not None


@pytest.mark.parametrize("policy", ["drop", [], {"a": 1}, 3])
def test_invalid_overflow_policy_is_invalid_input(policy: object) -> None:
    adapter, _ = make_adapter()
    with pytest.raises(InvalidInputError, match="overflow_policy"):
        adapter.extract([Item(text="x")], labels=["a", "b"], options={"overflow_policy": policy})


def test_constructor_bounds() -> None:
    assert GLiNER2DecideAdapter("x", max_seq_length=2048)._max_prompt_tokens == 512
    assert GLiNER2DecideAdapter("x", max_seq_length=256)._max_prompt_tokens == 128
    assert GLiNER2DecideAdapter("x", max_seq_length=2048, max_prompt_tokens=1024)._max_prompt_tokens == 1024
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        GLiNER2DecideAdapter("x", max_seq_length=64, max_prompt_tokens=64)
    with pytest.raises(ValueError, match="inference_batch_tokens"):
        GLiNER2DecideAdapter("x", max_seq_length=64, inference_batch_tokens=32)


# ---------------------------------------------------------------------------
# ModernBERT RoPE (GLiNER2.5-Decide-1B) and transformers 4 compatibility
# ---------------------------------------------------------------------------

ETTIN_ROPE = {
    "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 160000.0, "rope_type": "default"},
}


def tiny_ettin_config() -> dict[str, Any]:
    """GLiNER2.5-Decide-1B's encoder config (transformers 5 form), shrunk to a few small layers."""
    return {
        "model_type": "modernbert",
        "architectures": ["ModernBertForMaskedLM"],
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_attention_heads": 2,
        "num_hidden_layers": 3,
        "global_attn_every_n_layers": 3,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
        "local_attention": 16,
        "max_position_embeddings": 128,
        "vocab_size": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "cls_token_id": 1,
        "sep_token_id": 2,
        "rope_parameters": ETTIN_ROPE,
    }


def build_encoder(config: dict[str, Any]) -> torch.nn.Module:
    fields = {key: value for key, value in config.items() if key not in ("model_type", "architectures")}
    return transformers.AutoModel.from_config(transformers.ModernBertConfig(**fields), attn_implementation="eager")


def test_declared_rope_thetas() -> None:
    assert declared_rope_thetas({"model_type": "deberta-v2"}) is None
    assert declared_rope_thetas({"model_type": "modernbert", "rope_parameters": ETTIN_ROPE}) == {
        "full_attention": 160000.0,
        "sliding_attention": 160000.0,
    }
    assert declared_rope_thetas({"model_type": "modernbert", "local_rope_theta": 5000.0}) == {
        "full_attention": 160000.0,
        "sliding_attention": 5000.0,
    }
    with pytest.raises(ValueError, match="Unsupported"):
        declared_rope_thetas(
            {"model_type": "modernbert", "rope_parameters": {"full_attention": {"rope_type": "yarn", "rope_theta": 1}}}
        )


def test_transformers4_silently_misreads_ettin_rope_and_the_fix_restores_it() -> None:
    config = tiny_ettin_config()
    if int(transformers.__version__.split(".")[0]) >= 5:
        assert transformers4_encoder_config(config) is None
        assert verify_encoder_rope(build_encoder(config), config) == {
            "full_attention": 160000.0,
            "sliding_attention": 160000.0,
        }
        return
    # The trap: transformers 4 ignores rope_parameters and runs sliding layers at 10000.
    unpatched = build_encoder(config)
    observed = loaded_rope_thetas(unpatched, 3)
    assert observed["sliding_attention"] == pytest.approx([10000.0, 10000.0], rel=1e-3)
    with pytest.raises(RuntimeError, match=r"sliding_attention layers run RoPE base 10000\.0.*declares 160000\.0"):
        verify_encoder_rope(unpatched, config)
    # The fix: write the bases where transformers 4 reads them.
    patched = transformers4_encoder_config(config)
    assert patched is not None
    assert (patched["global_rope_theta"], patched["local_rope_theta"]) == (160000.0, 160000.0)
    fixed = build_encoder(patched)
    assert verify_encoder_rope(fixed, config) == {"full_attention": 160000.0, "sliding_attention": 160000.0}
    assert loaded_rope_thetas(fixed, 3)["sliding_attention"] == pytest.approx([160000.0, 160000.0], rel=1e-3)


def test_rope_tables_are_read_by_layer_type_in_both_transformers_layouts() -> None:
    def inv_freq(theta: float, n: int = 8) -> torch.Tensor:
        return 1.0 / (theta ** (torch.arange(0, 2 * n, 2).float() / (2 * n)))

    v5 = torch.nn.Module()
    v5.rotary_emb = torch.nn.Module()
    v5.rotary_emb.register_buffer("full_attention_inv_freq", inv_freq(160000.0))
    v5.rotary_emb.register_buffer("sliding_attention_inv_freq", inv_freq(10000.0))
    v5.rotary_emb.register_buffer("sliding_attention_original_inv_freq", inv_freq(1.0))
    assert loaded_rope_thetas(v5, 3) == {
        "full_attention": [pytest.approx(160000.0, rel=1e-4)],
        "sliding_attention": [pytest.approx(10000.0, rel=1e-4)],
    }
    with pytest.raises(RuntimeError, match="sliding_attention"):
        verify_encoder_rope(v5, {"model_type": "modernbert", "rope_parameters": ETTIN_ROPE})
    with pytest.raises(RuntimeError, match="Cannot find"):
        verify_encoder_rope(torch.nn.Linear(2, 2), {"model_type": "modernbert", "rope_parameters": ETTIN_ROPE})
    assert verify_encoder_rope(torch.nn.Linear(2, 2), {"model_type": "deberta-v2"}) is None


def test_transformers4_tokenizer_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decide_module, "_transformers_major", lambda: 4)
    config = {"tokenizer_class": "TokenizersBackend", "extra_special_tokens": ["[P]"]}
    assert transformers4_tokenizer_config(config) == {**config, "tokenizer_class": "PreTrainedTokenizerFast"}
    assert transformers4_tokenizer_config({"tokenizer_class": "DebertaV2Tokenizer"}) is None
    monkeypatch.setattr(decide_module, "_transformers_major", lambda: 5)
    assert transformers4_tokenizer_config(config) is None


def test_transformers4_layer_layout_must_follow_global_attention_period(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decide_module, "_transformers_major", lambda: 4)
    config = {**tiny_ettin_config(), "layer_types": ["sliding_attention", "full_attention", "sliding_attention"]}
    with pytest.raises(ValueError, match="layer_types"):
        transformers4_encoder_config(config)


def test_loadable_checkpoint_overlays_only_the_patched_configs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(decide_module, "_transformers_major", lambda: 4)
    (tmp_path / "encoder_config").mkdir()
    encoder = tiny_ettin_config()
    (tmp_path / "encoder_config" / "config.json").write_text(json.dumps(encoder))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    with loadable_checkpoint(tmp_path) as overlay:
        assert overlay != tmp_path
        assert (overlay / "model.safetensors").is_symlink()
        assert (overlay / "model.safetensors").read_bytes() == b"weights"
        assert json.loads((overlay / "encoder_config" / "config.json").read_text())["local_rope_theta"] == 160000.0
        assert (
            json.loads((overlay / "tokenizer_config.json").read_text())["tokenizer_class"] == "PreTrainedTokenizerFast"
        )
    assert not overlay.exists()
    assert json.loads((tmp_path / "encoder_config" / "config.json").read_text()) == encoder  # cache untouched

    monkeypatch.setattr(decide_module, "_transformers_major", lambda: 5)
    with loadable_checkpoint(tmp_path) as path:
        assert path == tmp_path


# ---------------------------------------------------------------------------
# Catalog and bundle
# ---------------------------------------------------------------------------

DECIDE_MODELS = {
    "fastino/GLiNER2.5-Decide": ("7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6", 512),
    "fastino/GLiNER2.5-multi-Decide": ("6bc1d43d201b0691e733626389af8c57eea3ea68", 2048),
    "fastino/GLiNER2.5-Decide-1B": ("52c94d3b698bf6d2619df9d898bdc1523ea3f1ca", 2048),
}


@pytest.mark.parametrize(("sie_id", "pinned"), DECIDE_MODELS.items())
def test_decide_model_configs(sie_id: str, pinned: tuple[str, int]) -> None:
    path = SIE_SERVER / "models" / f"{sie_id.replace('/', '__')}.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["sie_id"] == config["hf_id"] == sie_id
    assert (config["hf_revision"], config["max_sequence_length"]) == pinned
    profile = config["profiles"]["default"]
    assert profile["adapter_path"] == "sie_server.adapters.gliner2.decide:GLiNER2DecideAdapter"
    assert profile["compute_precision"] == "float16"


def test_decide_is_served_from_the_bundle_that_pins_gliner2_2() -> None:
    bundles = SIE_SERVER / "bundles"
    transformers5 = yaml.safe_load((bundles / "transformers5.yaml").read_text())
    default = yaml.safe_load((bundles / "default.yaml").read_text())
    assert "sie_server.adapters.gliner2.decide" in transformers5["adapters"]
    assert transformers5["deps"]["gliner2"] == "==2.0.0"
    assert "sie_server.adapters.gliner2.decide" not in default["adapters"]
    assert default["deps"]["gliner2"] == ">=1.3.1,<2"
