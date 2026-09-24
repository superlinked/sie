"""GLiClass request fields: instruction, examples, classification_type, label_groups, group_encoding."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.types.inputs import InvalidInputError, Item

_CLASS_TOKEN = 7
_SPECIAL_IDS = {"<<LABEL>>": _CLASS_TOKEN, "<<SEP>>": 8, "<<EXAMPLE>>": 9}
_TOKEN_RE = re.compile(r"<<LABEL>>|<<SEP>>|<<EXAMPLE>>|(?:(?!<<LABEL>>|<<SEP>>|<<EXAMPLE>>)\S)+")


class _MarkerAwareTokenizer:
    """One token per word or marker; CLS/SEP wrap adds two."""

    truncation_side = "right"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(
        self,
        texts: str | list[str],
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        batch = [texts] if isinstance(texts, str) else texts
        self.calls.append(list(batch))
        encoded = []
        for text in batch:
            ids = [_SPECIAL_IDS.get(token, 1) for token in _TOKEN_RE.findall(text)]
            specials = 2 if add_special_tokens else 0
            if truncation and max_length is not None:
                ids = ids[: max_length - specials]  # like HF: keep room for CLS/SEP
            if add_special_tokens:
                ids = [0, *ids, 2]
            encoded.append(ids)
        return {"input_ids": encoded[0] if isinstance(texts, str) else encoded}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join("w" for _ in ids)


class _StubPipe:
    """The attributes of a gliclass uni-encoder pipe that request checks read."""

    label_token = "<<LABEL>>"  # noqa: S105 -- a model marker, not a secret
    sep_token = "<<SEP>>"  # noqa: S105 -- a model marker, not a secret

    def __init__(self, *, prompt_first: bool, max_length: int = 512) -> None:
        self.max_length = max_length
        self.model: Any = SimpleNamespace(
            config=SimpleNamespace(
                architecture_type="uni-encoder", class_token_index=_CLASS_TOKEN, prompt_first=prompt_first
            )
        )

    def _format_examples_for_input(self, examples: list[dict[str, Any]]) -> str:
        parts = [f"<<EXAMPLE>>{e['text']} \nLabels:\n {', '.join(e['labels'])}" for e in examples]
        return "".join(parts) + "<<SEP>>"

    def prepare_input(
        self, text: str, labels: list[str], examples: list[dict[str, Any]] | None = None, prompt: str | None = None
    ) -> str:
        label_part = "".join(f"{self.label_token}{label}" for label in labels) + self.sep_token + (prompt or "")
        tail = self._format_examples_for_input(examples) if examples else ""
        if self.model.config.prompt_first:
            return label_part + text + tail
        return text + label_part + tail


class _SpaceBeforeMarkerTokenizer(_MarkerAwareTokenizer):
    """Like DeBERTa: whitespace right before a marker becomes its own token."""

    def __call__(self, texts: str | list[str], **kwargs: Any) -> dict[str, Any]:
        batch = [texts] if isinstance(texts, str) else texts
        marked = [re.sub(r"\s+(?=<<)", " SPACE ", text) for text in batch]
        return super().__call__(marked[0] if isinstance(texts, str) else marked, **kwargs)


Logit = Callable[[str, str], float]


class _ScoringPipe(_StubPipe):
    """A gliclass uni-encoder pipe whose model scores each (text, label) with ``logit``.

    Class slots past a row's own labels (a batch of rows with different label
    counts) get ``filler``, so a softmax that leaked across them would show.
    """

    def __init__(
        self,
        logit: Logit,
        *,
        prompt_first: bool = True,
        max_length: int = 512,
        filler: float = 50.0,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(prompt_first=prompt_first, max_length=max_length)
        self.logit = logit
        self.filler = filler
        self.error = error
        self.prepare_calls: list[dict[str, Any]] = []
        self.model = _ScoringModel(self, prompt_first=prompt_first)

    def prepare_inputs(
        self,
        texts: list[str],
        labels: list[str] | list[list[str]],
        same_labels: bool = False,
        examples: Any = None,
        prompt: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.error is not None:
            raise self.error
        row_labels = [list(labels)] * len(texts) if same_labels else [list(row) for row in labels]
        self.prepare_calls.append(
            {
                "texts": list(texts),
                "labels": row_labels,
                "same_labels": same_labels,
                "examples": examples,
                "prompt": prompt,
            }
        )
        width = max(len(row) for row in row_labels) + 2
        return {
            "input_ids": torch.ones(len(texts), width, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), width, dtype=torch.long),
        }

    def _resolve_max_num_classes(self, labels: list[str] | list[list[str]], same_labels: bool) -> int:
        return len(labels) if same_labels else max(len(row) for row in labels)


class _ScoringModel:
    def __init__(self, pipe: _ScoringPipe, *, prompt_first: bool) -> None:
        self._pipe = pipe
        self.config = SimpleNamespace(
            class_token_index=_CLASS_TOKEN, architecture_type="uni-encoder", prompt_first=prompt_first
        )
        self.forward_kwargs: list[dict[str, Any]] = []

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.forward_kwargs.append(kwargs)
        call = self._pipe.prepare_calls[-1]
        width = kwargs["max_num_classes"]
        rows = [
            [self._pipe.logit(text, label) for label in labels] + [self._pipe.filler] * (width - len(labels))
            for text, labels in zip(call["texts"], call["labels"], strict=True)
        ]
        return SimpleNamespace(logits=torch.tensor(rows, dtype=torch.float32))


def _adapter(
    logit: Logit,
    *,
    classification_type: str = "single-label",
    prompt_first: bool = True,
    max_length: int = 512,
    tokenizer: Any = None,
    error: BaseException | None = None,
) -> tuple[GLiClassAdapter, _ScoringPipe]:
    adapter = GLiClassAdapter("test-model", max_seq_length=max_length, classification_type=classification_type)  # ty:ignore[invalid-argument-type]
    pipe = _ScoringPipe(logit, prompt_first=prompt_first, max_length=max_length, error=error)
    adapter._attach(pipe, tokenizer or _MarkerAwareTokenizer())  # ty:ignore[invalid-argument-type]
    return adapter, pipe


_LABELS = ["billing", "bug report", "feature request"]
_SCORES = {"billing": 0.2, "bug report": 0.7, "feature request": 0.1}


def _score_logit(text: str, label: str) -> float:
    """Log-probabilities: a softmax over all three labels gives back ``_SCORES``."""
    return math.log(_SCORES[label])


def _flat_adapter(**kwargs: Any) -> tuple[GLiClassAdapter, _ScoringPipe]:
    return _adapter(_score_logit, **kwargs)


def _softmax(values: list[float]) -> list[float]:
    exps = [math.exp(v) for v in values]
    return [e / sum(exps) for e in exps]


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


class TestUnchangedRequests:
    """A request that sets none of the newer fields builds the historical model input."""

    def test_model_input_and_output_are_unchanged(self) -> None:
        adapter, pipe = _flat_adapter()

        output = adapter.extract([Item(text="The app crashes")], labels=list(_LABELS))

        assert pipe.prepare_calls == [
            {"texts": ["The app crashes"], "labels": [_LABELS], "same_labels": True, "examples": None, "prompt": None}
        ]
        assert pipe.model.forward_kwargs == [{"max_num_classes": 3}]
        assert output.classifications is not None
        assert [c["label"] for c in output.classifications[0]] == ["bug report", "billing", "feature request"]
        assert [c["score"] for c in output.classifications[0]] == pytest.approx([0.7, 0.2, 0.1], abs=1e-6)
        assert output.data is None
        assert output.entities == [[]]

    def test_threshold_filter_is_unchanged(self) -> None:
        adapter, _ = _flat_adapter()

        output = adapter.extract([Item(text="x")], labels=list(_LABELS), options={"threshold": 0.15})

        assert output.classifications is not None
        assert [c["label"] for c in output.classifications[0]] == ["bug report", "billing"]

    def test_blank_instruction_and_empty_examples_are_ignored(self) -> None:
        adapter, pipe = _flat_adapter()

        adapter.extract([Item(text="x")], labels=list(_LABELS), instruction="  ", options={"examples": []})

        assert pipe.prepare_calls[0]["prompt"] is None
        assert pipe.prepare_calls[0]["examples"] is None

    def test_items_run_in_forward_passes_of_eight(self) -> None:
        adapter, pipe = _flat_adapter()
        texts = [f"text {i}" for i in range(10)]

        output = adapter.extract([Item(text=t) for t in texts], labels=list(_LABELS))

        assert [call["texts"] for call in pipe.prepare_calls] == [texts[:8], texts[8:]]
        assert output.classifications is not None
        assert len(output.classifications) == 10

    def test_a_request_tokenizes_each_document_and_its_labels_once(self) -> None:
        tokenizer = _MarkerAwareTokenizer()
        adapter, _ = _flat_adapter(prompt_first=False, tokenizer=tokenizer)
        items = [Item(text="The app crashes"), Item(text="Charged twice")]

        adapter.extract(items, labels=list(_LABELS), options={"overflow_policy": "truncate_text"})
        first = list(tokenizer.calls)
        tokenizer.calls.clear()
        adapter.extract(items, labels=list(_LABELS), options={"overflow_policy": "truncate_text"})

        # One call for the label prompt, which the fit check and the overflow
        # policy both measure, and one for the documents; the model input
        # itself is tokenized by the pipe. The overflow policy, fit check and
        # metering share the document tokens.
        label_prompt = "<<LABEL>>billing<<LABEL>>bug report<<LABEL>>feature request<<SEP>>"
        assert first == [[label_prompt], ["The app crashes", "Charged twice"]]
        # Nothing carries over from one request to the next.
        assert tokenizer.calls == first


class TestInstructionAndExamples:
    def test_instruction_becomes_the_model_prompt(self) -> None:
        adapter, pipe = _flat_adapter()

        adapter.extract([Item(text="x")], labels=list(_LABELS), instruction="Classify the support ticket.")

        assert pipe.prepare_calls[0]["prompt"] == "Classify the support ticket."
        assert pipe.prepare_calls[0]["examples"] is None

    def test_examples_are_normalized_and_forwarded(self) -> None:
        adapter, pipe = _flat_adapter()
        examples = [
            {"text": "I was charged twice", "labels": [" billing "]},
            {"text": "Nothing applies here", "labels": []},
        ]

        adapter.extract([Item(text="x")], labels=list(_LABELS), options={"examples": examples})

        assert pipe.prepare_calls[0]["examples"] == [
            {"text": "I was charged twice", "labels": ["billing"]},
            {"text": "Nothing applies here", "labels": []},
        ]
        assert pipe.prepare_calls[0]["prompt"] is None

    @pytest.mark.parametrize(
        ("examples", "match"),
        [
            ("not a list", "must be a list"),
            ([{"text": "x"}], "exactly 'text' and 'labels'"),
            ([{"text": "x", "labels": ["billing"], "label": "billing"}], "exactly 'text' and 'labels'"),
            ([{"text": " ", "labels": ["billing"]}], "text must be a non-empty string"),
            ([{"text": "x", "labels": "billing"}], "must be a list of labels"),
            ([{"text": "x", "labels": [1]}], "must be a list of strings"),
            ([{"text": "x", "labels": ["refund"]}], "outside the requested label set"),
            ([{"text": "x", "labels": {"topic": "billing"}}], "only when label_groups is set"),
            ([{"text": "x", "labels": []}] * 33, "at most 32"),
            ([{"text": "x", "labels": ["billing"] * 4}], "more labels than the request has"),
            ([{"text": "x" * 2049, "labels": []}], "at most 2048 characters"),
            ([{"text": "x" * 2000, "labels": []}] * 5, "total at most 8192 characters"),
        ],
    )
    def test_malformed_examples_are_rejected(self, examples: object, match: str) -> None:
        adapter, _ = _flat_adapter()

        with pytest.raises(InvalidInputError, match=match):
            adapter.extract([Item(text="x")], labels=list(_LABELS), options={"examples": examples})

    def test_repeated_example_labels_are_sent_once(self) -> None:
        adapter, pipe = _flat_adapter()

        adapter.extract(
            [Item(text="x")],
            labels=list(_LABELS),
            options={"examples": [{"text": "charged twice", "labels": ["billing", " billing"]}]},
        )

        assert pipe.prepare_calls[0]["examples"] == [{"text": "charged twice", "labels": ["billing"]}]

    def test_label_text_that_cannot_fit_the_window_is_refused_before_tokenizing(self) -> None:
        adapter, _ = _flat_adapter()
        calls: list[object] = []
        tokenizer = adapter._tokenizer

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return tokenizer(*args, **kwargs)  # ty:ignore[call-non-callable]

        adapter._tokenizer = counting  # ty:ignore[invalid-assignment]
        adapter._max_token_chars = 4
        adapter._max_seq_length = 8  # 4 x 8 = 32 characters, exactly the length of _LABELS

        adapter.extract([Item(text="x")], labels=list(_LABELS))
        accepted = len(calls)
        with pytest.raises(InvalidInputError, match="at most 32 can fit"):
            adapter.extract([Item(text="x")], labels=[*_LABELS[:2], "x" * 2_000_000])

        assert accepted > 0
        assert len(calls) == accepted  # the refused request tokenized nothing

    def test_label_limit_ignores_vocabularies_with_very_long_tokens(self) -> None:
        adapter, _ = _flat_adapter()
        adapter._max_token_chars = 512  # e.g. a vocabulary holding a long whitespace run
        adapter._max_seq_length = 8  # 16 characters per token x 8 = 128

        with pytest.raises(InvalidInputError, match="at most 128 can fit"):
            adapter.extract([Item(text="x")], labels=[*_LABELS[:2], "x" * 200])

    def test_overlong_instruction_is_rejected(self) -> None:
        adapter, _ = _flat_adapter()

        with pytest.raises(InvalidInputError, match="at most 2048 characters"):
            adapter.extract([Item(text="x")], labels=list(_LABELS), instruction="x" * 2049)

    def test_item_without_text_is_invalid_input(self) -> None:
        adapter, _ = _flat_adapter()

        with pytest.raises(InvalidInputError, match="must have text"):
            adapter.extract([Item(text="")], labels=list(_LABELS))

    def test_non_string_instruction_is_rejected(self) -> None:
        adapter, _ = _flat_adapter()

        with pytest.raises(InvalidInputError, match="instruction must be a string"):
            adapter.extract([Item(text="x")], labels=list(_LABELS), instruction=3)  # ty:ignore[invalid-argument-type]


class TestClassificationType:
    def test_request_can_switch_to_multi_label(self) -> None:
        adapter, _ = _flat_adapter()

        output = adapter.extract([Item(text="x")], labels=list(_LABELS), options={"classification_type": "multi-label"})

        assert output.classifications is not None
        scores = {c["label"]: c["score"] for c in output.classifications[0]}
        expected = {label: _sigmoid(math.log(p)) for label, p in _SCORES.items()}
        assert scores == pytest.approx(expected, abs=1e-6)

    def test_multi_label_model_can_switch_to_single_label(self) -> None:
        adapter, _ = _flat_adapter(classification_type="multi-label")

        output = adapter.extract(
            [Item(text="x")], labels=list(_LABELS), options={"classification_type": "single-label"}
        )

        assert output.classifications is not None
        assert output.classifications[0][0]["label"] == "bug report"
        assert output.classifications[0][0]["score"] == pytest.approx(0.7, abs=1e-6)

    @pytest.mark.parametrize("value", ["multi_label", "binary", None, 1])
    def test_unknown_classification_type_is_rejected(self, value: object) -> None:
        adapter, _ = _flat_adapter()

        with pytest.raises(InvalidInputError, match="classification_type"):
            adapter.extract([Item(text="x")], labels=list(_LABELS), options={"classification_type": value})

    def test_unknown_load_time_classification_type_is_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="classification_type"):
            GLiClassAdapter("test-model", classification_type="multilabel")  # ty:ignore[invalid-argument-type]


_GROUPS = {"urgency": ["low", "high"], "topic": ["billing", "bug", "feature"]}
_LOGITS = {
    "Charged twice, fix today": {"low": -1.0, "high": 2.0, "billing": 3.0, "bug": 0.5, "feature": -2.0},
    "Dark mode someday?": {"low": 1.5, "high": -1.0, "billing": -3.0, "bug": -2.0, "feature": 4.0},
}


def _table_logit(text: str, label: str) -> float:
    """Scores a plain label ("low") and its joint form ("urgency.low") alike."""
    return _LOGITS[text][label.rsplit(".", 1)[-1]]


def _group_adapter(**kwargs: Any) -> tuple[GLiClassAdapter, _ScoringPipe]:
    return _adapter(_table_logit, **kwargs)


def _group_options(**options: Any) -> dict[str, Any]:
    return {"label_groups": _GROUPS, **options}


@pytest.fixture(params=["separate", "joint"])
def group_encoding(request: pytest.FixtureRequest) -> str:
    return request.param


class TestLabelGroups:
    """What both group encodings share: answer shape, normalization, validation."""

    def test_single_label_scores_are_normalized_per_group(self, group_encoding: str) -> None:
        adapter, _ = _group_adapter()
        texts = list(_LOGITS)

        output = adapter.extract([Item(text=t) for t in texts], options=_group_options(group_encoding=group_encoding))

        assert output.data is not None
        for text, data in zip(texts, output.data, strict=True):
            logits = _LOGITS[text]
            assert list(data) == ["urgency", "topic"]
            urgency, topic = data["urgency"]["probabilities"], data["topic"]["probabilities"]
            assert list(urgency) == ["low", "high"]
            assert list(urgency.values()) == pytest.approx(_softmax([logits["low"], logits["high"]]), abs=1e-6)
            assert list(topic.values()) == pytest.approx(
                _softmax([logits["billing"], logits["bug"], logits["feature"]]), abs=1e-6
            )
        assert output.data[0]["topic"]["choice"] == "billing"
        assert output.data[1]["urgency"]["choice"] == "low"

    def test_single_label_groups_answer_like_choice_questions(self, group_encoding: str) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract(
            [Item(text="Charged twice, fix today")], options=_group_options(group_encoding=group_encoding)
        )

        assert output.data is not None
        answer = output.data[0]["topic"]
        assert set(answer) == {"type", "choice", "probabilities", "confidence"}
        assert answer["type"] == "choice"
        probabilities = list(answer["probabilities"].values())
        entropy = -sum(p * math.log(p) for p in probabilities)
        assert answer["confidence"] == pytest.approx(1 - entropy / math.log(3))
        assert 0.0 < answer["confidence"] < 1.0

    def test_classifications_use_group_dot_label_and_honor_threshold(self, group_encoding: str) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract(
            [Item(text="Charged twice, fix today")],
            options=_group_options(group_encoding=group_encoding, threshold=0.5),
        )

        assert output.classifications is not None
        assert [c["label"] for c in output.classifications[0]] == ["urgency.high", "topic.billing"]
        assert output.data is not None
        assert len(output.data[0]["topic"]["probabilities"]) == 3  # never threshold-filtered

    def test_multi_label_uses_independent_sigmoids(self, group_encoding: str) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract(
            [Item(text="Charged twice, fix today")],
            options=_group_options(group_encoding=group_encoding, classification_type="multi-label"),
        )

        assert output.data is not None
        expected = [_sigmoid(v) for v in _LOGITS["Charged twice, fix today"].values()]
        urgency, topic = output.data[0]["urgency"], output.data[0]["topic"]
        assert set(urgency) == {"labels", "probabilities"}
        actual = [*urgency["probabilities"].values(), *topic["probabilities"].values()]
        assert actual == pytest.approx(expected, abs=1e-6)
        # Without a request threshold, labels at or above 0.5 are selected.
        assert urgency["labels"] == ["high"]
        assert topic["labels"] == ["billing", "bug"]

    def test_multi_label_selection_follows_the_request_threshold(self, group_encoding: str) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract(
            [Item(text="Charged twice, fix today")],
            options=_group_options(group_encoding=group_encoding, classification_type="multi-label", threshold=0.9),
        )

        assert output.data is not None
        assert output.data[0]["topic"]["labels"] == ["billing"]
        assert output.data[0]["urgency"]["labels"] == []

    def test_a_document_that_pushes_labels_out_fails_only_its_own_item(self, group_encoding: str) -> None:
        long_text = "attacker " * 600
        logits = {
            "Charged twice, fix today": _LOGITS["Charged twice, fix today"],
            long_text: _LOGITS["Dark mode someday?"],
        }
        adapter, pipe = _adapter(lambda text, label: logits[text][label.rsplit(".", 1)[-1]], prompt_first=False)

        output = adapter.extract(
            [Item(text="Charged twice, fix today"), Item(text=long_text)],
            options=_group_options(group_encoding=group_encoding),
        )

        assert {text for call in pipe.prepare_calls for text in call["texts"]} == {"Charged twice, fix today"}
        assert output.errors is not None
        assert output.errors[0] is None
        assert output.errors[1] is not None
        assert output.errors[1].code == "INPUT_TOO_LONG"
        assert output.data is not None
        assert output.data[0]["topic"]["choice"] == "billing"
        assert output.data[1] == {}
        assert output.classifications is not None
        assert output.classifications[1] == []
        assert output.input_token_counts is not None
        assert output.input_token_counts[1] == 0

    def test_labels_that_alone_overflow_the_window_are_refused(self, group_encoding: str) -> None:
        adapter, _ = _group_adapter(max_length=6)

        with pytest.raises(InputTooLongError):
            adapter.extract(
                [Item(text="Charged twice, fix today")], options=_group_options(group_encoding=group_encoding)
            )

    def test_a_marker_inside_the_document_is_not_an_overflow(self, group_encoding: str) -> None:
        text = "Charged twice, fix today"
        adapter, _ = _adapter(lambda _text, label: _table_logit(text, label))

        output = adapter.extract(
            [Item(text=f"{text} <<LABEL>>")], options=_group_options(group_encoding=group_encoding)
        )

        assert output.errors is None
        assert output.data is not None
        assert output.data[0]["topic"]["choice"] == "billing"

    @pytest.mark.parametrize(
        ("options", "labels", "match"),
        [
            ({"label_groups": _GROUPS}, ["billing"], "either labels or options.label_groups"),
            ({"label_groups": {}}, None, "non-empty object"),
            ({"label_groups": ["urgency"]}, None, "non-empty object"),
            ({"label_groups": {"urgency": []}}, None, "non-empty list"),
            ({"label_groups": {"urgency": "low"}}, None, "non-empty list"),
            ({"label_groups": {"urgency": ["low", "low"]}}, None, "unique"),
            ({"label_groups": {"urgency": ["low", "high"], " urgency ": ["x", "y"]}}, None, "names must be unique"),
            ({"label_groups": {"spam": ["yes"]}}, None, "at least two labels"),
            ({"label_groups": {"a.b": ["c", "d"], "a": ["b.c", "e"]}}, None, "repeat a label"),
            (
                {"label_groups": _GROUPS, "examples": [{"text": "x", "labels": {"size": "big"}}]},
                None,
                "unknown group",
            ),
            (
                {"label_groups": _GROUPS, "examples": [{"text": "x", "labels": {"topic": "refund"}}]},
                None,
                "not a label of group",
            ),
            (
                {"label_groups": _GROUPS, "examples": [{"text": "x", "labels": ["billing"]}]},
                None,
                "outside the requested label set",
            ),
            (
                {"label_groups": _GROUPS, "examples": [{"text": "x", "labels": {"urgency": ["low", "low", "high"]}}]},
                None,
                "more labels than the group has",
            ),
        ],
    )
    def test_malformed_groups_are_rejected(
        self, group_encoding: str, options: dict[str, Any], labels: list[str] | None, match: str
    ) -> None:
        adapter, _ = _group_adapter()

        with pytest.raises(InvalidInputError, match=match):
            adapter.extract([Item(text="x")], labels=labels, options={**options, "group_encoding": group_encoding})

    def test_single_label_group_of_one_is_allowed_in_multi_label_mode(self, group_encoding: str) -> None:
        adapter, _ = _adapter(lambda text, label: 0.3)

        output = adapter.extract(
            [Item(text="x")],
            options={
                "label_groups": {"spam": ["yes"]},
                "classification_type": "multi-label",
                "group_encoding": group_encoding,
            },
        )

        assert output.data is not None
        assert output.data[0]["spam"]["probabilities"]["yes"] == pytest.approx(_sigmoid(0.3), abs=1e-6)
        assert output.data[0]["spam"]["labels"] == ["yes"]

    @pytest.mark.parametrize("value", ["Joint", "per-group", 1, True, None, ["separate"]])
    def test_unknown_group_encoding_is_rejected(self, value: object) -> None:
        adapter, _ = _group_adapter()

        with pytest.raises(InvalidInputError, match="group_encoding must be 'separate' or 'joint'"):
            adapter.extract([Item(text="x")], options=_group_options(group_encoding=value))

    def test_group_encoding_without_label_groups_is_rejected(self) -> None:
        adapter, _ = _flat_adapter()

        with pytest.raises(InvalidInputError, match=r"only with options\.label_groups"):
            adapter.extract([Item(text="x")], labels=list(_LABELS), options={"group_encoding": "separate"})


class TestJointGroupEncoding:
    def test_all_groups_share_one_row_per_item(self) -> None:
        adapter, pipe = _group_adapter()
        texts = list(_LOGITS)

        adapter.extract([Item(text=t) for t in texts], options=_group_options(group_encoding="joint"))

        assert len(pipe.prepare_calls) == 1
        assert pipe.prepare_calls[0]["texts"] == texts
        assert pipe.prepare_calls[0]["same_labels"] is True
        assert pipe.prepare_calls[0]["labels"][0] == [
            "urgency.low",
            "urgency.high",
            "topic.billing",
            "topic.bug",
            "topic.feature",
        ]

    def test_prompt_examples_and_forward_kwargs_reach_the_model(self) -> None:
        adapter, pipe = _group_adapter()

        adapter.extract(
            [Item(text="Charged twice, fix today")],
            instruction="Triage the ticket.",
            options=_group_options(
                group_encoding="joint",
                examples=[
                    {"text": "Refund please", "labels": {"topic": "billing", "urgency": ["low"]}},
                    {"text": "App crashes", "labels": ["topic.bug"]},
                ],
            ),
        )

        call = pipe.prepare_calls[0]
        assert call["prompt"] == "Triage the ticket."
        assert call["examples"] == [
            {"text": "Refund please", "labels": ["topic.billing", "urgency.low"]},
            {"text": "App crashes", "labels": ["topic.bug"]},
        ]
        assert pipe.model.forward_kwargs == [{"max_num_classes": 5}]

    def test_batches_follow_the_pipeline_sub_batch_size(self) -> None:
        texts = [f"text {i}" for i in range(10)]
        adapter, pipe = _adapter(lambda text, label: 0.5)

        output = adapter.extract([Item(text=t) for t in texts], options=_group_options(group_encoding="joint"))

        assert [len(call["texts"]) for call in pipe.prepare_calls] == [8, 2]
        assert output.data is not None
        assert len(output.data) == 10

    def test_joint_groups_bill_the_document_once(self) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract(
            [Item(text="Charged twice, fix today")], options=_group_options(group_encoding="joint")
        )

        assert output.input_token_counts == [4 + 2]


class TestSeparateGroupEncoding:
    def test_separate_is_the_default(self) -> None:
        adapter, pipe = _group_adapter()

        adapter.extract([Item(text="Charged twice, fix today")], options=_group_options())

        assert pipe.prepare_calls[0]["same_labels"] is False
        assert sorted(pipe.prepare_calls[0]["labels"]) == [["billing", "bug", "feature"], ["low", "high"]]

    def test_rows_of_every_item_and_group_share_one_forward_pass(self) -> None:
        adapter, pipe = _group_adapter()
        texts = list(_LOGITS)

        adapter.extract([Item(text=t) for t in texts], options=_group_options())

        # One row per (item, group), the longest rows first.
        assert len(pipe.prepare_calls) == 1
        call = pipe.prepare_calls[0]
        assert list(zip(call["texts"], call["labels"], strict=True)) == [
            (texts[0], ["billing", "bug", "feature"]),
            (texts[1], ["billing", "bug", "feature"]),
            (texts[0], ["low", "high"]),
            (texts[1], ["low", "high"]),
        ]
        assert pipe.model.forward_kwargs == [{"max_num_classes": 3}]

    def test_many_short_rows_share_one_forward_pass(self) -> None:
        texts = [f"text {i}" for i in range(20)]
        adapter, pipe = _adapter(lambda text, label: 0.5)

        output = adapter.extract([Item(text=t) for t in texts], options=_group_options())

        assert [len(call["texts"]) for call in pipe.prepare_calls] == [40]
        assert output.data is not None
        assert len(output.data) == 20

    def test_rows_are_packed_into_passes_of_eight_full_windows(self) -> None:
        # Window 64, so a pass holds 8 x 64 = 512 padded tokens. Forty-word
        # documents make 49-token topic rows and 47-token urgency rows:
        # ten rows fit in the first pass.
        texts = [" ".join(f"w{i}x{j}" for j in range(40)) for i in range(6)]
        adapter, pipe = _adapter(lambda text, label: 0.5, max_length=64)

        output = adapter.extract([Item(text=t) for t in texts], options=_group_options())

        assert [len(call["texts"]) for call in pipe.prepare_calls] == [10, 2]
        assert [len(labels) for labels in pipe.prepare_calls[0]["labels"]] == [3] * 6 + [2] * 4
        assert output.data is not None
        assert all(list(answers) == ["urgency", "topic"] for answers in output.data)

    def test_each_group_scores_like_a_labels_request_with_its_labels(self) -> None:
        # The filler logit (50.0) sits in the class slot a two-label row gets
        # when it shares a forward pass with a three-label row. A softmax over
        # it would push every urgency probability to zero.
        adapter, _ = _group_adapter()
        items = [Item(text=t) for t in _LOGITS]

        grouped = adapter.extract(items, options=_group_options())

        assert grouped.data is not None
        for group, labels in _GROUPS.items():
            alone = adapter.extract(items, labels=labels)
            assert alone.classifications is not None
            for index, row in enumerate(alone.classifications):
                expected = {c["label"]: c["score"] for c in row}
                assert grouped.data[index][group]["probabilities"] == expected

    def test_each_row_gets_only_its_groups_example_labels(self) -> None:
        adapter, pipe = _group_adapter()

        adapter.extract(
            [Item(text="Charged twice, fix today")],
            instruction="Triage the ticket.",
            options=_group_options(
                examples=[
                    {"text": "Refund please", "labels": {"topic": "billing", "urgency": ["low"]}},
                    {"text": "App crashes", "labels": ["topic.bug"]},
                ],
            ),
        )

        call = pipe.prepare_calls[0]
        assert call["prompt"] == "Triage the ticket."
        examples_by_labels = {
            tuple(labels): examples for labels, examples in zip(call["labels"], call["examples"], strict=True)
        }
        assert examples_by_labels == {
            ("low", "high"): [{"text": "Refund please", "labels": ["low"]}, {"text": "App crashes", "labels": []}],
            ("billing", "bug", "feature"): [
                {"text": "Refund please", "labels": ["billing"]},
                {"text": "App crashes", "labels": ["bug"]},
            ],
        }

    def test_every_row_is_billed(self) -> None:
        adapter, _ = _group_adapter()
        items = [Item(text="Charged twice, fix today"), Item(text="Dark mode someday?")]

        plain = adapter.extract(items, options=_group_options())
        with_context = adapter.extract(
            items,
            instruction="Triage the ticket now",
            options=_group_options(examples=[{"text": "Refund please", "labels": ["topic.billing"]}]),
        )

        # Documents of 4 and 3 words plus CLS/SEP, once per group.
        assert plain.input_token_counts == [2 * (4 + 2), 2 * (3 + 2)]
        # Each row also encodes the instruction (4) and the example text (2).
        assert with_context.input_token_counts == [2 * (4 + 2 + 6), 2 * (3 + 2 + 6)]

    def test_truncate_text_cuts_each_row_to_its_own_budget(self) -> None:
        # max_length 14: the urgency label prompt is 5 tokens (two markers, two
        # labels, the separator) and topic's is 7. With CLS/SEP the ten-word
        # document keeps 7 tokens next to urgency and 5 next to topic.
        text = "one two three four five six seven eight nine ten"
        adapter, pipe = _adapter(lambda _text, label: 0.5, max_length=14)

        output = adapter.extract([Item(text=text)], options=_group_options(overflow_policy="truncate_text"))

        call = pipe.prepare_calls[0]
        texts_by_labels = {tuple(labels): text for labels, text in zip(call["labels"], call["texts"], strict=True)}
        assert texts_by_labels == {
            ("low", "high"): " ".join(["w"] * 7),
            ("billing", "bug", "feature"): " ".join(["w"] * 5),
        }
        assert output.input_token_counts == [(7 + 2) + (5 + 2)]

    def test_an_item_whose_labels_miss_any_row_fails_alone(self) -> None:
        # Text-first model keeping 10 content tokens: the six-word document leaves
        # room for both urgency markers but pushes the third topic marker out.
        adapter, pipe = _adapter(lambda _text, label: 0.5, prompt_first=False, max_length=12)
        items = [Item(text="one two three four five six"), Item(text="short")]

        output = adapter.extract(items, options=_group_options())

        assert output.errors is not None
        assert output.errors[0] is not None
        assert output.errors[0].code == "INPUT_TOO_LONG"
        assert output.errors[1] is None
        assert {text for call in pipe.prepare_calls for text in call["texts"]} == {"short"}
        assert output.input_token_counts is not None
        assert output.input_token_counts[0] == 0
        assert output.input_token_counts[1] == 2 * (1 + 2)

    def test_the_group_count_is_capped_by_rows_per_item(self) -> None:
        adapter, _ = _adapter(lambda _text, label: 0.5)
        groups = {f"q{i}": ["yes", "no"] for i in range(65)}

        with pytest.raises(InvalidInputError, match="at most 64 groups"):
            adapter.extract([Item(text="x")], options={"label_groups": groups})
        # A 1024-token window allows half as many rows per item.
        adapter._max_seq_length = 1024
        with pytest.raises(InvalidInputError, match="at most 32 groups"):
            adapter.extract([Item(text="x")], options={"label_groups": dict(list(groups.items())[:33])})
        # The joint encoding is one row per item and keeps its label cap only.
        output = adapter.extract([Item(text="x")], options={"label_groups": groups, "group_encoding": "joint"})
        assert output.data is not None
        assert len(output.data[0]) == 65

    def test_the_group_cap_is_checked_before_tokenizing(self) -> None:
        tokenizer = _MarkerAwareTokenizer()
        adapter, _ = _adapter(lambda _text, label: 0.5, tokenizer=tokenizer)

        with pytest.raises(InvalidInputError, match="groups"):
            adapter.extract([Item(text="x")], options={"label_groups": {f"q{i}": ["a", "b"] for i in range(65)}})

        assert tokenizer.calls == []

    def test_each_groups_label_text_is_checked_against_the_window(self) -> None:
        adapter, _ = _adapter(lambda _text, label: 0.5)
        adapter._max_seq_length = 8  # 16 characters per token x 8 = 128 per row

        # Every group fits alone although together they exceed one row.
        groups = {f"q{i}": ["x" * 60, "y" * 60] for i in range(3)}
        output = adapter.extract([Item(text="x")], options={"label_groups": groups})
        assert output.data is not None
        with pytest.raises(InvalidInputError, match="at most 128 can fit"):
            adapter.extract([Item(text="x")], options={"label_groups": {"q": ["x" * 70, "y" * 70]}})

    def test_an_item_that_fails_one_group_is_not_checked_against_the_rest(self) -> None:
        # Text-first model keeping 10 content tokens. Nine words land within the
        # fit margin of the urgency row, whose exact check refuses them; the
        # topic row is then skipped for that item.
        adapter, pipe = _adapter(lambda _text, label: 0.5, prompt_first=False, max_length=12)
        exact_checks: list[str] = []
        original = adapter._labels_survive

        def counting(text: str, *args: Any) -> bool:
            exact_checks.append(text)
            return original(text, *args)

        adapter._labels_survive = counting  # ty:ignore[invalid-assignment]
        failing = " ".join(f"w{index}" for index in range(9))

        output = adapter.extract([Item(text=failing), Item(text="short")], options=_group_options())

        assert exact_checks.count(failing) == 1
        assert output.errors is not None
        assert output.errors[0] is not None
        assert output.errors[0].code == "INPUT_TOO_LONG"
        assert output.errors[1] is None
        assert {text for call in pipe.prepare_calls for text in call["texts"]} == {"short"}

    def test_text_laid_out_with_whitespace_runs_is_read_at_64_groups(self) -> None:
        # Whitespace carries no tokens here, as with DeBERTa tokenizers, and
        # columns padded with spaces average about 42 characters per token:
        # the part the model reads spans about 23,000 characters.
        adapter, _ = _adapter(lambda _text, label: 0.5)
        layout = "".join(f"{f'item{index}':<20}{index:>22}\n" for index in range(700))
        groups = {f"q{index}": ["yes", "no"] for index in range(64)}

        output = adapter.extract([Item(text=layout)], options={"label_groups": groups})

        assert output.errors is None
        assert output.data is not None
        assert len(output.data[0]) == 64

    def test_a_document_the_groups_cannot_all_afford_fails_alone(self) -> None:
        # The part of this document the model reads spans 40,000 characters,
        # more than a row may tokenize (64 per window token: 32,768), though a
        # labels request reads it whole.
        adapter, _ = _adapter(lambda _text, label: 0.5)
        sparse = "start" + " " * 40_000 + "end"
        groups = {f"q{index}": ["yes", "no"] for index in range(64)}

        output = adapter.extract([Item(text=sparse), Item(text="short")], options={"label_groups": groups})

        assert output.errors is not None
        assert output.errors[0] is not None
        assert output.errors[0].code == "INPUT_TOO_LONG"
        assert "32768 characters" in output.errors[0].message
        assert output.errors[1] is None
        assert output.input_token_counts is not None
        assert output.input_token_counts[0] == 0
        assert adapter.extract([Item(text=sparse)], labels=["yes", "no"]).errors is None

    def test_few_groups_share_a_larger_per_item_budget(self) -> None:
        adapter, _ = _adapter(lambda _text, label: 0.5)
        sparse = "start" + " " * 40_000 + "end"

        # 524,288 characters per item among 4 groups is 131,072 per row.
        output = adapter.extract(
            [Item(text=sparse)], options={"label_groups": {f"q{i}": ["yes", "no"] for i in range(4)}}
        )

        assert output.errors is None

    def test_a_context_that_cannot_fit_is_refused_after_one_groups_worth_of_tokenizing(self) -> None:
        tokenizer = _MarkerAwareTokenizer()
        adapter, _ = _adapter(lambda _text, label: 0.5, max_length=64, tokenizer=tokenizer)
        instruction = " ".join(["word"] * 60)
        groups = {f"q{index}": ["yes", "no"] for index in range(64)}

        with pytest.raises(InvalidInputError, match="no room for the document"):
            adapter.extract([Item(text="short")], instruction=instruction, options={"label_groups": groups})

        tokenized = [text for call in tokenizer.calls for text in call]
        assert tokenized.count(instruction) == 1
        assert sum(len(text) for text in tokenized) < 2 * len(instruction)

    def test_each_groups_context_shares_one_tokenization_of_the_instruction_and_examples(self) -> None:
        tokenizer = _MarkerAwareTokenizer()
        adapter, _ = _adapter(lambda _text, label: 0.5, tokenizer=tokenizer)
        examples = [{"text": "Refund please", "labels": {"topic": "billing"}}]

        adapter.extract(
            [Item(text="Charged twice")],
            instruction="Triage the ticket.",
            options=_group_options(examples=examples, overflow_policy="truncate_text"),
        )

        tokenized = [text for call in tokenizer.calls for text in call]
        assert tokenized.count("Triage the ticket.") == 1
        assert tokenized.count("Refund please") == 1


class TestBatchingCost:
    @pytest.mark.parametrize("value", [None, "Joint", 1])
    def test_a_group_encoding_extract_refuses_is_never_costed_as_one_row(self, value: object) -> None:
        # The cost hook and extract read group_encoding in one place: a value
        # extract refuses gets no cost of its own, so it cannot be batched
        # cheaply and then run as separate rows.
        adapter, _ = _group_adapter()
        options = {"label_groups": _GROUPS, "group_encoding": value}

        assert adapter.extract_item_costs([Item(text="abcd")], options=options) is None
        with pytest.raises(InvalidInputError, match="group_encoding"):
            adapter.extract([Item(text="abcd")], options=options)

    def test_separate_groups_cost_every_rows_characters(self) -> None:
        adapter, _ = _group_adapter()
        items = [Item(text="abcd"), Item(text="abcdefgh")]
        label_chars = len("low" + "high" + "billing" + "bug" + "feature")

        assert adapter.extract_item_costs(items, options={"label_groups": _GROUPS}) == [
            2 * 4 + label_chars,
            2 * 8 + label_chars,
        ]
        # Each row also encodes the instruction and the example texts.
        costs = adapter.extract_item_costs(
            items,
            instruction="Triage",
            options={"label_groups": _GROUPS, "examples": [{"text": "Refund", "labels": ["topic.billing"]}]},
        )
        assert costs == [2 * (4 + 12) + label_chars, 2 * (8 + 12) + label_chars]

    @pytest.mark.parametrize(
        "options",
        [
            None,
            {},
            {"label_groups": _GROUPS, "group_encoding": "joint"},
            {"label_groups": "not a mapping"},
            {"label_groups": {f"q{i}": ["a", "b"] for i in range(65)}},
        ],
    )
    def test_other_requests_keep_the_default_cost(self, options: dict[str, Any] | None) -> None:
        adapter, _ = _group_adapter()

        assert adapter.extract_item_costs([Item(text="abcd")], labels=["a"], options=options) is None


class TestOverflowBudget:
    def test_prompt_and_examples_count_against_the_text_budget(self) -> None:
        adapter = GLiClassAdapter("test-model")
        adapter._max_seq_length = 20
        adapter._special_count = 2

        class _Tokenizer:
            def __call__(self, text: str | list[str], add_special_tokens: bool = False) -> dict[str, Any]:
                if isinstance(text, list):
                    return {"input_ids": [t.split() for t in text]}
                return {"input_ids": text.split()}

            def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
                return " ".join(ids)

        class _Pipe:
            def prepare_input(
                self,
                text: str,
                labels: list[str],
                examples: list[dict[str, Any]] | None = None,
                prompt: str | None = None,
            ) -> str:
                parts = [f"<L> {label}" for label in labels] + ["<S>"]
                if prompt:
                    parts.append(prompt)
                parts.append(text)
                for example in examples or []:
                    parts.append(f"<E> {example['text']} {' '.join(example['labels'])}")
                return " ".join(part for part in parts if part)

        adapter._tokenizer = _Tokenizer()  # ty:ignore[invalid-assignment]
        adapter._pipe = _Pipe()
        text = " ".join(f"w{i}" for i in range(12))

        plain = adapter._apply_overflow_policy([text], ["a"], "truncate_text")
        with_context = adapter._apply_overflow_policy(
            [text],
            ["a"],
            "truncate_text",
            prompt="Pick one label",
            examples=[{"text": "some example", "labels": ["a"]}],
        )

        # overhead: "<L> a <S>" = 3 (+2 special) leaves 15; the prompt adds 3 and
        # the example "<E> some example a" adds 4, leaving 8 document tokens.
        assert plain == [text]
        assert with_context == [" ".join(f"w{i}" for i in range(8))]


class _WordTokenizer:
    """One token per whitespace word; specials add two unless disabled."""

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, texts: list[str], add_special_tokens: bool = True, **kwargs: Any) -> dict[str, list[list[int]]]:
        extra = 2 if add_special_tokens else 0
        counts = [len(text.split()) + extra for text in texts]
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            counts = [min(count, kwargs["max_length"]) for count in counts]
        return {"input_ids": [list(range(count)) for count in counts]}


class TestMetering:
    def _adapter(self, max_seq_length: int = 512) -> GLiClassAdapter:
        adapter, _ = _flat_adapter(tokenizer=_WordTokenizer())
        adapter._max_seq_length = max_seq_length
        return adapter

    def test_requests_without_context_bill_the_document_only(self) -> None:
        adapter = self._adapter()

        output = adapter.extract([Item(text="two words"), Item(text="three whole words")], labels=list(_LABELS))

        assert output.input_token_counts == [4, 5]

    def test_instruction_and_example_texts_are_billed_per_item(self) -> None:
        adapter = self._adapter()
        examples = [
            {"text": "charged twice for one order", "labels": ["billing"]},
            {"text": "add dark mode", "labels": ["feature request", "bug report"]},
        ]

        output = adapter.extract(
            [Item(text="two words"), Item(text="three whole words")],
            labels=list(_LABELS),
            instruction="Classify the ticket",
            options={"examples": examples},
        )

        # instruction 3 + example texts 5 + 3 = 11 per item; example labels are not billed.
        assert output.input_token_counts == [4 + 11, 5 + 11]

    def test_billing_is_capped_at_the_model_window(self) -> None:
        adapter = self._adapter(max_seq_length=12)

        output = adapter.extract(
            [Item(text="two words")], labels=list(_LABELS), instruction="one two three four five six seven eight nine"
        )

        assert output.input_token_counts == [12]

    def test_long_documents_bill_at_most_the_model_window(self) -> None:
        adapter = self._adapter(max_seq_length=12)

        output = adapter.extract([Item(text=" ".join(["word"] * 40))], labels=list(_LABELS))

        assert output.input_token_counts == [12]


class TestFlatLabelOverflow:
    def _adapter(self, *, prompt_first: bool, max_length: int = 512) -> tuple[GLiClassAdapter, _ScoringPipe]:
        return _flat_adapter(prompt_first=prompt_first, max_length=max_length)

    def test_one_oversized_document_does_not_fail_the_batch(self) -> None:
        adapter, pipe = self._adapter(prompt_first=False)
        attacker = "attacker " * 600

        output = adapter.extract([Item(text="short victim text"), Item(text=attacker)], labels=list(_LABELS))

        assert [call["texts"] for call in pipe.prepare_calls] == [["short victim text"]]
        assert output.classifications is not None
        assert output.classifications[0][0]["label"] == "bug report"
        assert output.classifications[1] == []
        assert output.errors is not None
        assert output.errors[0] is None
        assert output.errors[1] is not None
        assert output.errors[1].code == "INPUT_TOO_LONG"
        assert output.input_token_counts == [5, 0]

    def test_only_oversized_documents_skip_inference(self) -> None:
        adapter, pipe = self._adapter(prompt_first=False)

        output = adapter.extract([Item(text="attacker " * 600)], labels=list(_LABELS))

        assert pipe.prepare_calls == []
        assert output.errors is not None
        assert output.errors[0] is not None

    def test_labels_first_models_are_not_affected_by_document_length(self) -> None:
        adapter, _ = self._adapter(prompt_first=True)

        output = adapter.extract([Item(text="attacker " * 600)], labels=list(_LABELS))

        assert output.errors is None

    def test_labels_that_alone_overflow_the_window_are_refused(self) -> None:
        adapter, _ = self._adapter(prompt_first=True, max_length=6)

        with pytest.raises(InputTooLongError):
            adapter.extract([Item(text="The app crashes")], labels=list(_LABELS))

    def test_a_marker_inside_the_document_is_not_an_overflow(self) -> None:
        adapter, _ = self._adapter(prompt_first=False)

        output = adapter.extract([Item(text="quoted <<LABEL>> marker")], labels=list(_LABELS))

        assert output.errors is None

    def test_context_that_leaves_no_room_for_the_document_is_refused(self) -> None:
        adapter, _ = self._adapter(prompt_first=True, max_length=24)

        with pytest.raises(InvalidInputError, match="no room for the document"):
            adapter.extract(
                [Item(text="The app crashes")],
                labels=list(_LABELS),
                instruction=" ".join(["word"] * 20),
            )

    @pytest.mark.parametrize(("words", "fits"), [(11, True), (12, False)])
    def test_the_fit_estimate_is_exact_at_the_window_edge(self, words: int, fits: bool) -> None:
        # Window of 18 non-special tokens; the last marker sits at index 5 of the
        # label prompt. Twelve words look like they fit (12 + 5 < 18), but the
        # trailing space becomes a token next to the first marker, so the last
        # marker falls out of the window.
        adapter, _ = _flat_adapter(prompt_first=False, max_length=20, tokenizer=_SpaceBeforeMarkerTokenizer())

        output = adapter.extract([Item(text="word " * words)], labels=list(_LABELS))

        if fits:
            assert output.errors is None
        else:
            assert output.errors is not None
            assert output.errors[0] is not None
            assert output.errors[0].code == "INPUT_TOO_LONG"

    def test_billing_with_context_is_capped_at_window_minus_labels(self) -> None:
        adapter, _ = self._adapter(prompt_first=True, max_length=40)
        text = " ".join(["doc"] * 30)

        output = adapter.extract([Item(text=text)], labels=list(_LABELS), instruction="one two three")

        # label prompt = 3 markers + 5 label words + SEP = 9 tokens; cap = 40 - 9 = 31.
        # The document count alone is 32 (30 words + CLS/SEP), so it stands.
        assert output.input_token_counts == [32]
        short = adapter.extract([Item(text="doc doc")], labels=list(_LABELS), instruction="one two three")
        assert short.input_token_counts == [4 + 3]

    def test_a_model_with_fewer_class_slots_than_labels_is_an_overflow(self) -> None:
        adapter, pipe = self._adapter(prompt_first=True)
        pipe._resolve_max_num_classes = lambda labels, same_labels: 2  # ty:ignore[invalid-assignment]

        with pytest.raises(InputTooLongError):
            adapter.extract([Item(text="The app crashes")], labels=list(_LABELS))
