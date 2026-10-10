"""GLiNER opt-in output filters: options.exclude_labels and options.require_uppercase."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.types.inputs import InvalidInputError, Item

_LABELS = ["person name", "organization name", "location name"]
_EXCLUDE = [
    "software or product name",
    "programme, fund or event name",
    "standard or protocol name",
    "website or domain",
    "job title or role",
]
_TEXT = (
    "The government asked NIST and the Office of Weights and Measures to test Firefox on gpg.fail, "
    "said Ada Lovelace, a Moderator in Zürich working under ISO/IEC 17043 with the military."
)


def _span(text: str, label: str, score: float, occurrence: int = 0) -> dict[str, Any]:
    start = -1
    for _ in range(occurrence + 1):
        start = _TEXT.index(text, start + 1)
    return {"start": start, "end": start + len(text), "text": text, "label": label, "score": score}


# Model output as GLiNER returns it after flat_ner: every label, including the competitor labels.
_MODEL_SPANS = [
    _span("government", "organization name", 0.91),
    _span("NIST", "organization name", 0.97),
    _span("Office of Weights and Measures", "organization name", 0.95),
    _span("Firefox", "software or product name", 0.94),
    _span("gpg.fail", "website or domain", 0.9),
    _span("Ada Lovelace", "person name", 0.99),
    _span("Moderator", "job title or role", 0.88),
    _span("Zürich", "location name", 0.96),
    _span("ISO/IEC 17043", "standard or protocol name", 0.93),
    _span("military", "organization name", 0.87),
]


def _adapter(result: Any) -> GLiNERAdapter:
    adapter = GLiNERAdapter("test-model")
    model = MagicMock()
    model.data_processor = None  # no metering in these tests
    model.inference.return_value = result
    adapter._model = model
    adapter._device = "cpu"
    adapter._extracts_relations = False
    return adapter


def _inference(adapter: GLiNERAdapter) -> MagicMock:
    inference = adapter._model.inference
    assert isinstance(inference, MagicMock)
    return inference


def _r1u_reference(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The frozen R1u client filter (r1u_filter.py, sha256 d39f31bf36f2d10d1f28f73cbd08dd5f362b0b337827eff6439da0ec99e584b4).

    Keep spans whose label is an inclusion label, then drop spans whose text has no uppercase character.
    """
    kept = []
    for entity in entities:
        if entity["label"] not in _LABELS:
            continue
        if not any(character.isupper() for character in entity["text"]):
            continue
        kept.append(entity)
    return kept


def test_without_options_the_inference_call_and_reply_are_unchanged() -> None:
    adapter = _adapter([list(_MODEL_SPANS)])

    output = adapter.extract([Item(text=_TEXT)], labels=list(_LABELS))

    _inference(adapter).assert_called_once_with(
        [_TEXT], _LABELS, batch_size=32, threshold=0.5, flat_ner=True, multi_label=False
    )
    assert [entity["text"] for entity in output.entities[0]] == [span["text"] for span in _MODEL_SPANS]


def test_exclude_labels_compete_in_the_model_and_their_spans_are_removed() -> None:
    adapter = _adapter([list(_MODEL_SPANS)])

    output = adapter.extract(
        [Item(text=_TEXT)], labels=list(_LABELS), options={"threshold": 0.85, "exclude_labels": list(_EXCLUDE)}
    )

    _inference(adapter).assert_called_once_with(
        [_TEXT], [*_LABELS, *_EXCLUDE], batch_size=32, threshold=0.85, flat_ner=True, multi_label=False
    )
    labels = {entity["label"] for entity in output.entities[0]}
    assert labels <= set(_LABELS)
    assert "Firefox" not in [entity["text"] for entity in output.entities[0]]
    assert "government" in [entity["text"] for entity in output.entities[0]]


def test_require_uppercase_drops_spans_without_an_uppercase_character() -> None:
    adapter = _adapter([list(_MODEL_SPANS)])

    output = adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options={"require_uppercase": True})

    texts = [entity["text"] for entity in output.entities[0]]
    assert "government" not in texts
    assert "military" not in texts
    assert "Zürich" in texts
    assert "gpg.fail" not in texts


def test_options_reproduce_the_frozen_r1u_client_filter() -> None:
    adapter = _adapter([list(_MODEL_SPANS)])

    output = adapter.extract(
        [Item(text=_TEXT)],
        labels=list(_LABELS),
        options={"threshold": 0.85, "exclude_labels": list(_EXCLUDE), "require_uppercase": True},
    )

    expected = _r1u_reference(_MODEL_SPANS)
    got = output.entities[0]
    assert [(e["text"], e["label"], e["start"], e["end"]) for e in got] == [
        (e["text"], e["label"], e["start"], e["end"]) for e in expected
    ]
    assert [e["text"] for e in got] == ["NIST", "Office of Weights and Measures", "Ada Lovelace", "Zürich"]
    for entity in got:
        assert _TEXT[entity["start"] : entity["end"]] == entity["text"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"exclude_labels": "software"}, "must be a list"),
        ({"exclude_labels": ["software", ""]}, "non-empty strings"),
        ({"exclude_labels": ["software", "software"]}, "must be unique"),
        ({"exclude_labels": ["person name"]}, "must not repeat"),
        ({"require_uppercase": "yes"}, "must be a boolean"),
    ],
)
def test_malformed_options_are_rejected(options: dict[str, Any], message: str) -> None:
    adapter = _adapter([list(_MODEL_SPANS)])

    with pytest.raises(InvalidInputError, match=message):
        adapter.extract([Item(text=_TEXT)], labels=list(_LABELS), options=options)
    _inference(adapter).assert_not_called()


def test_exclude_labels_cannot_be_combined_with_relation_labels() -> None:
    adapter = _adapter([list(_MODEL_SPANS)])
    adapter._extracts_relations = True

    with pytest.raises(InvalidInputError, match="cannot be combined with relation_labels"):
        adapter.extract(
            [Item(text=_TEXT)],
            labels=list(_LABELS),
            options={"exclude_labels": ["software"], "relation_labels": ["works at"]},
        )
