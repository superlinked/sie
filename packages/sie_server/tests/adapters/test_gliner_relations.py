"""Relation extraction through the GLiNER adapter (joint entity-relation models)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from sie_server.adapters.gliner import GLiNERAdapter, _cap_relation_candidates
from sie_server.adapters.gliner_bi import GLiNERBiAdapter
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.types.inputs import InvalidInputError, Item

_TEXT = "Steve Jobs founded Apple in Cupertino."
_LABELS = ["person", "organization", "location"]
_ENTITIES = [
    {"start": 0, "end": 10, "text": "Steve Jobs", "label": "person", "score": 0.91},
    {"start": 19, "end": 24, "text": "Apple", "label": "organization", "score": 0.88},
    {"start": 28, "end": 37, "text": "Cupertino", "label": "location", "score": 0.8},
]


def _endpoint(entity: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "start": entity["start"],
        "end": entity["end"],
        "text": entity["text"],
        "type": entity["label"],
        "entity_idx": index,
    }


_RELATIONS = [
    {"head": _endpoint(_ENTITIES[0], 0), "tail": _endpoint(_ENTITIES[1], 1), "relation": "founded", "score": 0.72},
    {"head": _endpoint(_ENTITIES[1], 1), "tail": _endpoint(_ENTITIES[2], 2), "relation": "located in", "score": 0.93},
]


def _adapter(result: Any, *, relex: bool) -> GLiNERAdapter:
    adapter = GLiNERAdapter("test-model")
    model = MagicMock()
    model.data_processor = None  # no metering in these tests
    model.inference.return_value = result
    adapter._model = model
    adapter._device = "cpu"
    adapter._extracts_relations = relex
    return adapter


def _inference(adapter: GLiNERAdapter) -> MagicMock:
    inference = adapter._model.inference
    assert isinstance(inference, MagicMock)
    return inference


def test_models_without_relations_keep_their_inference_call() -> None:
    adapter = _adapter([list(_ENTITIES)], relex=False)

    output = adapter.extract([Item(text=_TEXT)], labels=list(_LABELS))

    _inference(adapter).assert_called_once_with([_TEXT], _LABELS, threshold=0.5, flat_ner=True, multi_label=False)
    assert [entity["text"] for entity in output.entities[0]] == ["Steve Jobs", "Apple", "Cupertino"]
    assert output.relations is None


def test_relex_model_without_relation_labels_returns_entities_only() -> None:
    adapter = _adapter([list(_ENTITIES)], relex=True)

    output = adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"relation_threshold": 0.9})

    _inference(adapter).assert_called_once_with(
        [_TEXT],
        _LABELS,
        threshold=0.5,
        flat_ner=True,
        multi_label=False,
        relations=[],
        return_relations=False,
        adjacency_threshold=0.5,
    )
    assert len(output.entities[0]) == 3
    assert output.relations is None


def test_relation_labels_return_relations_between_found_entities() -> None:
    adapter = _adapter(([list(_ENTITIES)], [list(_RELATIONS)]), relex=True)

    output = adapter.extract(
        [Item(text=_TEXT)],
        labels=list(_LABELS),
        options={"relation_labels": [" founded ", "located in"], "relation_threshold": 0.6, "threshold": 0.4},
    )

    _inference(adapter).assert_called_once_with(
        [_TEXT],
        _LABELS,
        threshold=0.4,
        flat_ner=True,
        multi_label=False,
        relations=["founded", "located in"],
        return_relations=True,
        adjacency_threshold=0.5,
        relation_threshold=0.6,
    )
    assert output.relations == [
        [
            {"head": "Apple", "tail": "Cupertino", "relation": "located in", "score": 0.93},
            {"head": "Steve Jobs", "tail": "Apple", "relation": "founded", "score": 0.72},
        ]
    ]
    assert len(output.entities[0]) == 3


def test_relation_threshold_defaults_to_the_config_then_the_library() -> None:
    adapter = _adapter(([[]], [[]]), relex=True)

    adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"relation_labels": ["founded"]})
    assert "relation_threshold" not in _inference(adapter).call_args.kwargs

    adapter._relation_threshold = 0.75
    adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"relation_labels": ["founded"]})
    assert _inference(adapter).call_args.kwargs["relation_threshold"] == 0.75


def test_relation_labels_batch_by_order_and_reach_the_adapter_as_sent() -> None:
    handler = ExtractHandler()

    def metadata(relation_labels: list[str]) -> MagicMock:
        meta = MagicMock()
        meta.labels = list(_LABELS)
        meta.output_schema = None
        meta.instruction = None
        meta.options = {"relation_labels": relation_labels, "relation_threshold": 0.7}
        return meta

    founded_first = metadata(["founded", "located in"])
    same = metadata(["founded", "located in"])
    located_first = metadata(["located in", "founded"])
    assert handler.make_config_key(founded_first) == handler.make_config_key(same)
    assert handler.make_config_key(founded_first) != handler.make_config_key(located_first)

    adapter = _adapter(([[], []], [[], []]), relex=True)
    output = handler.run_inference(
        adapter,
        [Item(text=_TEXT), Item(text=_TEXT)],
        handler.make_config_key(located_first),
        None,
        [located_first, located_first],
    )

    assert _inference(adapter).call_args.kwargs["relations"] == ["located in", "founded"]
    assert output.relations == [[], []]


def test_relation_labels_need_a_relation_model() -> None:
    adapter = _adapter([[]], relex=False)

    with pytest.raises(InvalidInputError, match="does not extract relations"):
        adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"relation_labels": ["founded"]})

    _inference(adapter).assert_not_called()


@pytest.mark.parametrize(
    ("relation_labels", "match"),
    [
        ("founded", "must be a list"),
        ([""], "non-empty strings"),
        ([3], "non-empty strings"),
        (["founded", " founded"], "unique"),
        ([f"relation {index}" for index in range(998)], "at most 1000"),
    ],
)
def test_malformed_relation_labels_are_rejected(relation_labels: object, match: str) -> None:
    adapter = _adapter(([[]], [[]]), relex=True)

    with pytest.raises(InvalidInputError, match=match):
        adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"relation_labels": relation_labels})


@pytest.mark.parametrize("threshold", [True, "0.5", -0.1, 1.5, float("nan"), 10**1000])
def test_malformed_relation_threshold_is_rejected(threshold: object) -> None:
    adapter = _adapter(([[]], [[]]), relex=True)

    with pytest.raises(InvalidInputError, match="relation_threshold"):
        adapter.extract(
            [Item(text=_TEXT)],
            labels=list(_LABELS),
            options={"relation_labels": ["founded"], "relation_threshold": threshold},
        )


def test_relation_count_mismatch_is_an_internal_error() -> None:
    adapter = _adapter(([[]], []), relex=True)

    with pytest.raises(ValueError, match="different number of items") as raised:
        adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"relation_labels": ["founded"]})
    assert not isinstance(raised.value, InvalidInputError)


def test_bi_encoder_models_reject_relation_labels() -> None:
    adapter = GLiNERBiAdapter("test-model")
    adapter._model = MagicMock()
    adapter._device = "cpu"

    with pytest.raises(InvalidInputError, match="do not extract relations"):
        adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"relation_labels": ["founded"]})

    adapter._model.inference.assert_not_called()
    adapter._model.batch_predict_with_embeds.assert_not_called()


class _Encoding:
    def __init__(self, word_ids: list[list[int | None]], words_masks: list[list[int]]) -> None:
        self._word_ids = word_ids
        self._tensors = {
            "attention_mask": [torch.ones(len(ids), dtype=torch.long) for ids in word_ids],
            "words_mask": [torch.tensor(mask, dtype=torch.long) for mask in words_masks],
        }

    def word_ids(self, batch_index: int) -> list[int | None]:
        return self._word_ids[batch_index]

    def __getitem__(self, key: str) -> list[torch.Tensor]:
        return self._tensors[key]


class _RelexProcessor:
    """Prompt = entity markers, SEP, relation markers, SEP (as gliner's relex processors build it)."""

    def __init__(self) -> None:
        self.collate_calls: list[dict[str, Any]] = []
        self.tokenize_calls: list[Any] = []

    def collate_raw_batch(
        self, batch: list[dict[str, Any]], *, entity_types: list[str], relation_types: list[str]
    ) -> dict[str, Any]:
        self.collate_calls.append({"entity_types": entity_types, "relation_types": relation_types})
        return {
            "tokens": [item["tokenized_text"] for item in batch],
            "classes_to_id": {label: index for index, label in enumerate(entity_types, start=1)},
            "rel_class_to_ids": {label: index for index, label in enumerate(relation_types, start=1)},
        }

    def tokenize_inputs(
        self, texts: list[list[str]], entity_mappings: dict[str, int], relations: dict[str, int]
    ) -> _Encoding:
        self.tokenize_calls.append(relations)
        prompt = len(entity_mappings) * 2 + 1 + len(relations) * 2 + 1
        word_ids: list[list[int | None]] = []
        words_masks: list[list[int]] = []
        for words in texts:
            ids: list[int | None] = [None, *range(prompt)]
            mask = [0] * (prompt + 1)
            for position, _word in enumerate(words):
                ids.append(prompt + position)
                mask.append(position + 1)
            ids.append(None)
            mask.append(0)
            word_ids.append(ids)
            words_masks.append(mask)
        return _Encoding(word_ids, words_masks)


def test_relex_metering_passes_relation_types_to_the_processor() -> None:
    processor = _RelexProcessor()
    adapter = _adapter(([list(_ENTITIES)], [list(_RELATIONS)]), relex=True)
    model = adapter._model
    model.data_processor = processor
    model.prepare_inputs.side_effect = lambda texts: ([text.split() for text in texts], [[]], [[]])
    model.prepare_base_input.side_effect = lambda texts: [{"tokenized_text": words, "ner": None} for words in texts]

    output = adapter.extract(
        [Item(text=_TEXT)], labels=list(_LABELS), options={"relation_labels": ["founded", "located in"]}
    )

    assert processor.collate_calls == [{"entity_types": _LABELS, "relation_types": ["founded", "located in"]}]
    assert processor.tokenize_calls == [{"founded": 1, "located in": 2}]
    # 6 document words plus CLS and SEP; the entity and relation prompt is not billed.
    assert output.input_token_counts == [8]


@pytest.mark.parametrize("threshold", [True, "0.3", -0.01, 1.01, float("nan"), 10**1000])
def test_entity_threshold_is_validated(threshold: object) -> None:
    adapter = _adapter([[]], relex=False)

    with pytest.raises(InvalidInputError, match="GLiNER threshold"):
        adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"threshold": threshold})

    _inference(adapter).assert_not_called()


def test_a_low_entity_threshold_keeps_the_adjacency_floor() -> None:
    adapter = _adapter(([[]], [[]]), relex=True)

    adapter.extract(
        [Item(text=_TEXT)], labels=list(_LABELS), options={"threshold": 0.2, "relation_labels": ["founded"]}
    )

    kwargs = _inference(adapter).call_args.kwargs
    assert kwargs["threshold"] == 0.2
    assert kwargs["adjacency_threshold"] == 0.5


@pytest.mark.parametrize("options", [{"threshold": 0.0}, {"threshold": 0.05, "relation_labels": ["founded"]}])
def test_relex_models_refuse_a_near_zero_entity_threshold(options: dict[str, Any]) -> None:
    adapter = _adapter(([[]], [[]]), relex=True)

    with pytest.raises(InvalidInputError, match=r"at least 0\.1"):
        adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options=options)

    _inference(adapter).assert_not_called()


def test_other_gliner_models_accept_a_zero_threshold() -> None:
    adapter = _adapter([[]], relex=False)

    adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"threshold": 0.0})

    assert _inference(adapter).call_args.kwargs["threshold"] == 0.0


@pytest.mark.parametrize(
    ("items", "labels", "match"),
    [
        ([Item(text="Ada")], [], "requires labels"),
        ([Item(text="   ")], ["person"], "non-blank text"),
        ([Item()], ["person"], "requires"),
    ],
)
def test_caller_errors_are_invalid_input(items: list[Item], labels: list[str], match: str) -> None:
    adapter = _adapter([[]], relex=False)

    with pytest.raises(InvalidInputError, match=match):
        adapter.extract(items, labels=labels)


class _SpanModel:
    """Minimal stand-in for gliner's relex model: candidates packed at the front."""

    def __init__(self, valid_per_row: list[int], width: int, dim: int = 4) -> None:
        self.valid_per_row = valid_per_row
        self.width = width
        self.dim = dim

    def represent_spans(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        rows = [[1] * n + [0] * (self.width - n) for n in self.valid_per_row]
        mask = torch.tensor(rows, dtype=torch.long)
        reps = torch.arange(len(rows) * self.width * self.dim, dtype=torch.float32).view(
            len(rows), self.width, self.dim
        )
        spans = torch.zeros(len(rows), self.width, 2, dtype=torch.long)
        scores = torch.zeros(len(rows), self.width, 3)  # size(1) == width too; must stay whole
        return (scores, reps, mask, spans)


def test_relation_candidates_are_sliced_per_item() -> None:
    model = _SpanModel([150, 3], width=150)
    _, full_reps, _, _ = model.represent_spans()
    _cap_relation_candidates(model, 100)

    scores, reps, mask, spans = model.represent_spans("words", "mask", "prompts")

    assert tuple(scores.shape) == (2, 150, 3)
    assert tuple(reps.shape) == (2, 100, 4)
    assert tuple(mask.shape) == (2, 100)
    assert tuple(spans.shape) == (2, 100, 2)
    assert torch.equal(reps, full_reps[:, :100])
    assert mask.sum(dim=1).tolist() == [100, 3]


def test_relation_candidates_under_the_cap_are_untouched() -> None:
    model = _SpanModel([5, 3], width=5)
    original = model.represent_spans("x")
    _cap_relation_candidates(model, 100)

    capped = model.represent_spans("x")
    assert all(torch.equal(a, b) for a, b in zip(original, capped, strict=True))


@pytest.mark.parametrize(
    "outputs",
    [
        (torch.zeros(1, 150, 3), torch.zeros(1, 150, 4), None, None),
        (torch.zeros(1, 150, 3), torch.zeros(1, 150, 4), torch.ones(1, 150)),
        (torch.zeros(1, 150, 3), torch.zeros(1, 120, 4), torch.ones(1, 150), torch.zeros(1, 150, 2)),
        [torch.zeros(1, 150, 3), torch.zeros(1, 150, 4), torch.ones(1, 150), torch.zeros(1, 150, 2)],
    ],
)
def test_an_unknown_candidate_layout_fails_closed(outputs: Any) -> None:
    model = MagicMock()
    model.represent_spans.return_value = outputs
    _cap_relation_candidates(model, 100)

    with pytest.raises(RuntimeError, match="candidate layout"):
        model.represent_spans("words", "mask", "prompts")


def test_bi_encoder_caller_errors_are_invalid_input() -> None:
    adapter = GLiNERBiAdapter("test-model")
    adapter._model = MagicMock()
    adapter._device = "cpu"

    with pytest.raises(InvalidInputError, match="requires labels"):
        adapter.extract([Item(text="Ada")], labels=[])
    with pytest.raises(InvalidInputError):
        adapter.extract([Item()], labels=["person"])


@pytest.mark.parametrize("threshold", ["0.3", True, -0.1, 1.5, float("nan")])
def test_bi_encoder_threshold_is_validated(threshold: object) -> None:
    adapter = GLiNERBiAdapter("test-model")
    adapter._model = MagicMock()
    adapter._device = "cpu"

    with pytest.raises(InvalidInputError, match="GLiNER-bi threshold"):
        adapter.extract([Item(text="Ada")], labels=["person"], options={"threshold": threshold})

    adapter._model.inference.assert_not_called()
