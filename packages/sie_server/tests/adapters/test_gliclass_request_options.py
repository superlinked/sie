"""GLiClass request fields: instruction, examples, classification_type, label_groups."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

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

    def __call__(
        self,
        texts: str | list[str],
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        batch = [texts] if isinstance(texts, str) else texts
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


class _StubPipe:
    """The attributes of a gliclass uni-encoder pipe that request checks read."""

    label_token = "<<LABEL>>"  # noqa: S105 -- a model marker, not a secret
    sep_token = "<<SEP>>"  # noqa: S105 -- a model marker, not a secret

    def __init__(self, *, prompt_first: bool, max_length: int = 512) -> None:
        self.max_length = max_length
        self.model = SimpleNamespace(
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


def _pipeline(results: list[dict[str, float]]) -> MagicMock:
    pipeline = MagicMock()
    pipeline.return_value = results
    return pipeline


def _flat_adapter(
    single: list[dict[str, float]],
    multi: list[dict[str, float]] | None = None,
    *,
    classification_type: str = "single-label",
) -> GLiClassAdapter:
    adapter = GLiClassAdapter("test-model", max_seq_length=512, classification_type=classification_type)  # ty:ignore[invalid-argument-type]
    pipelines = {"single-label": _pipeline(single), "multi-label": _pipeline(multi or single)}
    adapter._pipelines = pipelines  # ty:ignore[invalid-assignment]
    adapter._pipeline = pipelines[classification_type]  # ty:ignore[invalid-assignment]
    adapter._tokenizer = _MarkerAwareTokenizer()  # ty:ignore[invalid-assignment]
    adapter._special_count = 2
    return adapter


_LABELS = ["billing", "bug report", "feature request"]
_SCORES = {"billing": 0.2, "bug report": 0.7, "feature request": 0.1}


class TestUnchangedRequests:
    """A request that sets none of the new fields runs the historical call."""

    def test_pipeline_call_and_output_are_unchanged(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        output = adapter.extract([Item(text="The app crashes")], labels=list(_LABELS))

        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        pipeline.assert_called_once_with(["The app crashes"], _LABELS, threshold=0.0, return_hierarchical=True)
        assert adapter._pipelines is not None
        other = adapter._pipelines["multi-label"]
        assert isinstance(other, MagicMock)
        other.assert_not_called()
        assert output.classifications == [
            [
                {"label": "bug report", "score": 0.7},
                {"label": "billing", "score": 0.2},
                {"label": "feature request", "score": 0.1},
            ]
        ]
        assert output.data is None
        assert output.entities == [[]]

    def test_threshold_filter_is_unchanged(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        output = adapter.extract([Item(text="x")], labels=list(_LABELS), options={"threshold": 0.15})

        assert output.classifications == [[{"label": "bug report", "score": 0.7}, {"label": "billing", "score": 0.2}]]

    def test_blank_instruction_and_empty_examples_are_ignored(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        adapter.extract([Item(text="x")], labels=list(_LABELS), instruction="  ", options={"examples": []})

        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        pipeline.assert_called_once_with(["x"], _LABELS, threshold=0.0, return_hierarchical=True)

    def test_default_classification_type_uses_the_load_time_pipeline(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        adapter.extract([Item(text="x")], labels=list(_LABELS), options={"classification_type": "single-label"})

        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        assert pipeline.call_count == 1


class TestInstructionAndExamples:
    def test_instruction_becomes_the_pipeline_prompt(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        adapter.extract([Item(text="x")], labels=list(_LABELS), instruction="Classify the support ticket.")

        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        assert pipeline.call_args.kwargs["prompt"] == "Classify the support ticket."
        assert "examples" not in pipeline.call_args.kwargs

    def test_examples_are_normalized_and_forwarded(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])
        examples = [
            {"text": "I was charged twice", "labels": [" billing "]},
            {"text": "Nothing applies here", "labels": []},
        ]

        adapter.extract([Item(text="x")], labels=list(_LABELS), options={"examples": examples})

        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        assert pipeline.call_args.kwargs["examples"] == [
            {"text": "I was charged twice", "labels": ["billing"]},
            {"text": "Nothing applies here", "labels": []},
        ]
        assert "prompt" not in pipeline.call_args.kwargs

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
        adapter = _flat_adapter([dict(_SCORES)])

        with pytest.raises(InvalidInputError, match=match):
            adapter.extract([Item(text="x")], labels=list(_LABELS), options={"examples": examples})

    def test_repeated_example_labels_are_sent_once(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        adapter.extract(
            [Item(text="x")],
            labels=list(_LABELS),
            options={"examples": [{"text": "charged twice", "labels": ["billing", " billing"]}]},
        )

        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        assert pipeline.call_args.kwargs["examples"] == [{"text": "charged twice", "labels": ["billing"]}]

    def test_label_text_that_cannot_fit_the_window_is_refused_before_tokenizing(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])
        calls: list[object] = []
        tokenizer = adapter._tokenizer

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return tokenizer(*args, **kwargs)  # ty:ignore[call-non-callable]

        adapter._tokenizer = counting  # ty:ignore[invalid-assignment]
        adapter._max_token_chars = 4
        adapter._max_seq_length = 8  # 4 x 8 = 32 characters, exactly the length of _LABELS

        adapter.extract([Item(text="x")], labels=list(_LABELS))
        with pytest.raises(InvalidInputError, match="at most 32 can fit"):
            adapter.extract([Item(text="x")], labels=[*_LABELS[:2], "x" * 2_000_000])

        assert len(calls) == 1  # only the accepted request tokenized anything

    def test_label_limit_ignores_vocabularies_with_very_long_tokens(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])
        adapter._max_token_chars = 512  # e.g. a vocabulary holding a long whitespace run
        adapter._max_seq_length = 8  # 16 characters per token x 8 = 128

        with pytest.raises(InvalidInputError, match="at most 128 can fit"):
            adapter.extract([Item(text="x")], labels=[*_LABELS[:2], "x" * 200])

    def test_overlong_instruction_is_rejected(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        with pytest.raises(InvalidInputError, match="at most 2048 characters"):
            adapter.extract([Item(text="x")], labels=list(_LABELS), instruction="x" * 2049)

    def test_item_without_text_is_invalid_input(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        with pytest.raises(InvalidInputError, match="must have text"):
            adapter.extract([Item(text="")], labels=list(_LABELS))

    def test_non_string_instruction_is_rejected(self) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        with pytest.raises(InvalidInputError, match="instruction must be a string"):
            adapter.extract([Item(text="x")], labels=list(_LABELS), instruction=3)  # ty:ignore[invalid-argument-type]


class TestClassificationType:
    def test_request_can_switch_to_multi_label(self) -> None:
        multi = {"billing": 0.9, "bug report": 0.8, "feature request": 0.05}
        adapter = _flat_adapter([dict(_SCORES)], [multi])

        output = adapter.extract([Item(text="x")], labels=list(_LABELS), options={"classification_type": "multi-label"})

        assert adapter._pipelines is not None
        single = adapter._pipelines["single-label"]
        assert isinstance(single, MagicMock)
        single.assert_not_called()
        assert output.classifications is not None
        assert [c["label"] for c in output.classifications[0]] == ["billing", "bug report", "feature request"]

    def test_multi_label_model_can_switch_to_single_label(self) -> None:
        multi = {"billing": 0.9, "bug report": 0.8, "feature request": 0.05}
        adapter = _flat_adapter([dict(_SCORES)], [multi], classification_type="multi-label")

        output = adapter.extract(
            [Item(text="x")], labels=list(_LABELS), options={"classification_type": "single-label"}
        )

        assert output.classifications is not None
        assert output.classifications[0][0] == {"label": "bug report", "score": 0.7}

    @pytest.mark.parametrize("value", ["multi_label", "binary", None, 1])
    def test_unknown_classification_type_is_rejected(self, value: object) -> None:
        adapter = _flat_adapter([dict(_SCORES)])

        with pytest.raises(InvalidInputError, match="classification_type"):
            adapter.extract([Item(text="x")], labels=list(_LABELS), options={"classification_type": value})

    def test_unknown_load_time_classification_type_is_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="classification_type"):
            GLiClassAdapter("test-model", classification_type="multilabel")  # ty:ignore[invalid-argument-type]


class _FakeGroupPipe(_StubPipe):
    """Stands in for the gliclass uni-encoder pipe used by the grouped path."""

    def __init__(
        self, logits_by_text: dict[str, list[float]], *, prompt_first: bool = True, max_length: int = 512
    ) -> None:
        super().__init__(prompt_first=prompt_first, max_length=max_length)
        self.logits_by_text = logits_by_text
        self.prepare_calls: list[dict[str, Any]] = []
        self.model = _FakeGroupModel(self, prompt_first=prompt_first)

    def prepare_inputs(
        self,
        texts: list[str],
        labels: list[str],
        same_labels: bool = False,
        examples: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
    ) -> dict[str, torch.Tensor]:
        self.prepare_calls.append(
            {
                "texts": list(texts),
                "labels": list(labels),
                "same_labels": same_labels,
                "examples": examples,
                "prompt": prompt,
            }
        )
        row = [1] + [_CLASS_TOKEN] * len(labels) + [2, 3, 4]
        return {
            "input_ids": torch.tensor([row] * len(texts)),
            "attention_mask": torch.ones(len(texts), len(row), dtype=torch.long),
        }

    def _resolve_max_num_classes(self, labels: list[str], same_labels: bool) -> int:
        assert same_labels is True
        return len(labels)


class _FakeGroupModel:
    def __init__(self, pipe: _FakeGroupPipe, *, prompt_first: bool) -> None:
        self._pipe = pipe
        self.config = SimpleNamespace(
            class_token_index=_CLASS_TOKEN, architecture_type="uni-encoder", prompt_first=prompt_first
        )
        self.forward_kwargs: list[dict[str, Any]] = []

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.forward_kwargs.append(kwargs)
        texts = self._pipe.prepare_calls[-1]["texts"]
        logits = torch.tensor([self._pipe.logits_by_text[text] for text in texts], dtype=torch.float16)
        return SimpleNamespace(logits=logits)


_GROUPS = {"urgency": ["low", "high"], "topic": ["billing", "bug", "feature"]}
# Flattened order: urgency.low, urgency.high, topic.billing, topic.bug, topic.feature
_LOGITS = {
    "Charged twice, fix today": [-1.0, 2.0, 3.0, 0.5, -2.0],
    "Dark mode someday?": [1.5, -1.0, -3.0, -2.0, 4.0],
}


def _group_adapter(
    logits: dict[str, list[float]] | None = None, *, prompt_first: bool = True, max_length: int = 512
) -> tuple[GLiClassAdapter, _FakeGroupPipe]:
    adapter = GLiClassAdapter("test-model", max_seq_length=max_length)
    pipe = _FakeGroupPipe(logits or _LOGITS, prompt_first=prompt_first, max_length=max_length)
    pipeline = MagicMock()
    pipeline.pipe = pipe
    adapter._pipeline = pipeline  # ty:ignore[invalid-assignment]
    adapter._pipelines = {"single-label": pipeline, "multi-label": MagicMock()}  # ty:ignore[invalid-assignment]
    adapter._tokenizer = _MarkerAwareTokenizer()  # ty:ignore[invalid-assignment]
    adapter._special_count = 2
    return adapter, pipe


def _softmax(values: list[float]) -> list[float]:
    exps = [math.exp(v) for v in values]
    return [e / sum(exps) for e in exps]


class TestLabelGroups:
    def test_single_label_scores_are_normalized_per_group(self) -> None:
        adapter, pipe = _group_adapter()
        texts = list(_LOGITS)

        output = adapter.extract([Item(text=t) for t in texts], options={"label_groups": _GROUPS})

        assert pipe.prepare_calls[0]["labels"] == [
            "urgency.low",
            "urgency.high",
            "topic.billing",
            "topic.bug",
            "topic.feature",
        ]
        assert output.data is not None
        for text, data in zip(texts, output.data, strict=True):
            logits = [float(torch.tensor(v, dtype=torch.float16)) for v in _LOGITS[text]]
            assert list(data) == ["urgency", "topic"]
            urgency, topic = data["urgency"]["probabilities"], data["topic"]["probabilities"]
            assert list(urgency) == ["low", "high"]
            assert list(urgency.values()) == pytest.approx(_softmax(logits[:2]), abs=1e-6)
            assert list(topic.values()) == pytest.approx(_softmax(logits[2:]), abs=1e-6)
            assert sum(urgency.values()) == pytest.approx(1.0)
            assert sum(topic.values()) == pytest.approx(1.0)
        assert output.data[0]["topic"]["choice"] == "billing"
        assert output.data[1]["urgency"]["choice"] == "low"

    def test_single_label_groups_answer_like_choice_questions(self) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract([Item(text="Charged twice, fix today")], options={"label_groups": _GROUPS})

        assert output.data is not None
        answer = output.data[0]["topic"]
        assert set(answer) == {"type", "choice", "probabilities", "confidence"}
        assert answer["type"] == "choice"
        probabilities = list(answer["probabilities"].values())
        entropy = -sum(p * math.log(p) for p in probabilities)
        assert answer["confidence"] == pytest.approx(1 - entropy / math.log(3))
        assert 0.0 < answer["confidence"] < 1.0

    def test_classifications_use_group_dot_label_and_honor_threshold(self) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract(
            [Item(text="Charged twice, fix today")], options={"label_groups": _GROUPS, "threshold": 0.5}
        )

        assert output.classifications is not None
        assert [c["label"] for c in output.classifications[0]] == ["urgency.high", "topic.billing"]
        assert output.data is not None
        assert len(output.data[0]["topic"]["probabilities"]) == 3  # never threshold-filtered

    def test_multi_label_uses_independent_sigmoids(self) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract(
            [Item(text="Charged twice, fix today")],
            options={"label_groups": _GROUPS, "classification_type": "multi-label"},
        )

        assert output.data is not None
        expected = [1 / (1 + math.exp(-v)) for v in _LOGITS["Charged twice, fix today"]]
        urgency, topic = output.data[0]["urgency"], output.data[0]["topic"]
        assert set(urgency) == {"labels", "probabilities"}
        actual = [*urgency["probabilities"].values(), *topic["probabilities"].values()]
        assert actual == pytest.approx(expected, abs=1e-3)
        # Without a request threshold, labels at or above 0.5 are selected.
        assert urgency["labels"] == ["high"]
        assert topic["labels"] == ["billing", "bug"]

    def test_multi_label_selection_follows_the_request_threshold(self) -> None:
        adapter, _ = _group_adapter()

        output = adapter.extract(
            [Item(text="Charged twice, fix today")],
            options={"label_groups": _GROUPS, "classification_type": "multi-label", "threshold": 0.9},
        )

        assert output.data is not None
        assert output.data[0]["topic"]["labels"] == ["billing"]
        assert output.data[0]["urgency"]["labels"] == []

    def test_prompt_examples_and_forward_kwargs_reach_the_model(self) -> None:
        adapter, pipe = _group_adapter()

        adapter.extract(
            [Item(text="Charged twice, fix today")],
            instruction="Triage the ticket.",
            options={
                "label_groups": _GROUPS,
                "examples": [
                    {"text": "Refund please", "labels": {"topic": "billing", "urgency": ["low"]}},
                    {"text": "App crashes", "labels": ["topic.bug"]},
                ],
            },
        )

        call = pipe.prepare_calls[0]
        assert call["prompt"] == "Triage the ticket."
        assert call["same_labels"] is True
        assert call["examples"] == [
            {"text": "Refund please", "labels": ["topic.billing", "urgency.low"]},
            {"text": "App crashes", "labels": ["topic.bug"]},
        ]
        assert pipe.model.forward_kwargs == [{"max_num_classes": 5}]

    def test_batches_follow_the_pipeline_sub_batch_size(self) -> None:
        texts = [f"text {i}" for i in range(10)]
        logits = {text: [0.0, 1.0, 0.0, 1.0, 2.0] for text in texts}
        adapter, pipe = _group_adapter(logits)

        output = adapter.extract([Item(text=t) for t in texts], options={"label_groups": _GROUPS})

        assert [len(call["texts"]) for call in pipe.prepare_calls] == [8, 2]
        assert output.data is not None
        assert len(output.data) == 10

    def test_a_document_that_pushes_labels_out_fails_only_its_own_item(self) -> None:
        long_text = "attacker " * 600
        logits = {"Charged twice, fix today": _LOGITS["Charged twice, fix today"], long_text: [0.0] * 5}
        adapter, pipe = _group_adapter(logits, prompt_first=False)

        output = adapter.extract(
            [Item(text="Charged twice, fix today"), Item(text=long_text)], options={"label_groups": _GROUPS}
        )

        assert [call["texts"] for call in pipe.prepare_calls] == [["Charged twice, fix today"]]
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

    def test_labels_that_alone_overflow_the_window_are_refused(self) -> None:
        adapter, _ = _group_adapter(max_length=6)

        with pytest.raises(InputTooLongError):
            adapter.extract([Item(text="Charged twice, fix today")], options={"label_groups": _GROUPS})

    def test_a_marker_inside_the_document_is_not_an_overflow(self) -> None:
        text = "Charged twice, fix today"
        adapter, _ = _group_adapter({f"{text} <<LABEL>>": _LOGITS[text]})

        output = adapter.extract([Item(text=f"{text} <<LABEL>>")], options={"label_groups": _GROUPS})

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
    def test_malformed_groups_are_rejected(self, options: dict[str, Any], labels: list[str] | None, match: str) -> None:
        adapter, _ = _group_adapter()

        with pytest.raises(InvalidInputError, match=match):
            adapter.extract([Item(text="x")], labels=labels, options=options)

    def test_single_label_group_of_one_is_allowed_in_multi_label_mode(self) -> None:
        logits = {"x": [0.3]}
        adapter, _ = _group_adapter(logits)

        output = adapter.extract(
            [Item(text="x")], options={"label_groups": {"spam": ["yes"]}, "classification_type": "multi-label"}
        )

        assert output.data is not None
        assert output.data[0]["spam"]["probabilities"]["yes"] == pytest.approx(1 / (1 + math.exp(-0.3)), abs=1e-3)
        assert output.data[0]["spam"]["labels"] == ["yes"]


class TestOverflowBudget:
    def test_prompt_and_examples_count_against_the_text_budget(self) -> None:
        adapter = GLiClassAdapter("test-model")
        adapter._max_seq_length = 20
        adapter._special_count = 2

        class _Tokenizer:
            def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[str]]:
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
        adapter._pipeline = SimpleNamespace(pipe=_Pipe())  # ty:ignore[invalid-assignment]
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

    def __call__(self, texts: list[str], add_special_tokens: bool = True, **kwargs: Any) -> dict[str, list[list[int]]]:
        extra = 2 if add_special_tokens else 0
        counts = [len(text.split()) + extra for text in texts]
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            counts = [min(count, kwargs["max_length"]) for count in counts]
        return {"input_ids": [list(range(count)) for count in counts]}


class TestMetering:
    def _adapter(self, max_seq_length: int = 512, items: int = 2) -> GLiClassAdapter:
        adapter = _flat_adapter([dict(_SCORES)] * items)
        adapter._max_seq_length = max_seq_length
        adapter._tokenizer = _WordTokenizer()  # ty:ignore[invalid-assignment]
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
        adapter = self._adapter(max_seq_length=12, items=1)

        output = adapter.extract(
            [Item(text="two words")], labels=list(_LABELS), instruction="one two three four five six seven eight nine"
        )

        assert output.input_token_counts == [12]


class TestFlatLabelOverflow:
    def _adapter(
        self, results: list[dict[str, float]], *, prompt_first: bool, max_length: int = 512
    ) -> GLiClassAdapter:
        adapter = _flat_adapter(results)
        adapter._max_seq_length = max_length
        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        pipeline.pipe = _StubPipe(prompt_first=prompt_first, max_length=max_length)
        return adapter

    def test_one_oversized_document_does_not_fail_the_batch(self) -> None:
        adapter = self._adapter([dict(_SCORES)], prompt_first=False)
        attacker = "attacker " * 600

        output = adapter.extract([Item(text="short victim text"), Item(text=attacker)], labels=list(_LABELS))

        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        pipeline.assert_called_once_with(["short victim text"], _LABELS, threshold=0.0, return_hierarchical=True)
        assert output.classifications is not None
        assert output.classifications[0][0] == {"label": "bug report", "score": 0.7}
        assert output.classifications[1] == []
        assert output.errors is not None
        assert output.errors[0] is None
        assert output.errors[1] is not None
        assert output.errors[1].code == "INPUT_TOO_LONG"
        assert output.input_token_counts == [5, 0]

    def test_only_oversized_documents_skip_inference(self) -> None:
        adapter = self._adapter([], prompt_first=False)

        output = adapter.extract([Item(text="attacker " * 600)], labels=list(_LABELS))

        pipeline = adapter._pipeline
        assert isinstance(pipeline, MagicMock)
        pipeline.assert_not_called()
        assert output.errors is not None
        assert output.errors[0] is not None

    def test_labels_first_models_are_not_affected_by_document_length(self) -> None:
        adapter = self._adapter([dict(_SCORES)], prompt_first=True)

        output = adapter.extract([Item(text="attacker " * 600)], labels=list(_LABELS))

        assert output.errors is None

    def test_labels_that_alone_overflow_the_window_are_refused(self) -> None:
        adapter = self._adapter([dict(_SCORES)], prompt_first=True, max_length=6)

        with pytest.raises(InputTooLongError):
            adapter.extract([Item(text="The app crashes")], labels=list(_LABELS))

    def test_a_marker_inside_the_document_is_not_an_overflow(self) -> None:
        adapter = self._adapter([dict(_SCORES)], prompt_first=False)

        output = adapter.extract([Item(text="quoted <<LABEL>> marker")], labels=list(_LABELS))

        assert output.errors is None

    def test_context_that_leaves_no_room_for_the_document_is_refused(self) -> None:
        adapter = self._adapter([dict(_SCORES)], prompt_first=True, max_length=24)

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
        adapter = self._adapter([dict(_SCORES)], prompt_first=False, max_length=20)
        adapter._tokenizer = _SpaceBeforeMarkerTokenizer()  # ty:ignore[invalid-assignment]

        output = adapter.extract([Item(text="word " * words)], labels=list(_LABELS))

        if fits:
            assert output.errors is None
        else:
            assert output.errors is not None
            assert output.errors[0] is not None
            assert output.errors[0].code == "INPUT_TOO_LONG"

    def test_billing_with_context_is_capped_at_window_minus_labels(self) -> None:
        adapter = self._adapter([dict(_SCORES)], prompt_first=True, max_length=40)
        text = " ".join(["doc"] * 30)

        output = adapter.extract([Item(text=text)], labels=list(_LABELS), instruction="one two three")

        # label prompt = 3 markers + 5 label words + SEP = 9 tokens; cap = 40 - 9 = 31.
        # The document count alone is 32 (30 words + CLS/SEP), so it stands.
        assert output.input_token_counts == [32]
        short = adapter.extract([Item(text="doc doc")], labels=list(_LABELS), instruction="one two three")
        assert short.input_token_counts == [4 + 3]
