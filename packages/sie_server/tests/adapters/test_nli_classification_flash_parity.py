"""NLI zero-shot classification parity with transformers' pipeline, on real weights.

Every shipped NLI config runs the flash adapter, which reimplements the
pipeline's scoring. This pins it to ``pipeline("zero-shot-classification")``
in the three modes users reach: one label, several mutually exclusive labels,
and ``multi_label``. Runs on CPU in float32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sie_server.core.loader import load_adapter, load_model_configs
from sie_server.types.inputs import Item

pytestmark = pytest.mark.model

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
TEMPLATE = "This text is about {}."
TEXTS = [
    "The Lakers beat the Celtics 112-104 last night behind a late run from their bench.",
    "Preheat the oven to 180 degrees, knead the dough and bake the bread for forty minutes.",
    "The city council voted 7-2 to spend public money on a new stadium for the local football club.",
]


@pytest.fixture(scope="module")
def adapter():
    config = load_model_configs(MODELS_DIR)[MODEL]
    adapter = load_adapter(config, MODELS_DIR, device="cpu")
    adapter.load("cpu")
    return adapter


@pytest.fixture(scope="module")
def pipeline():
    from transformers import pipeline as hf_pipeline

    config = load_model_configs(MODELS_DIR)[MODEL]
    return hf_pipeline("zero-shot-classification", model=config.hf_id, revision=config.hf_revision, device=-1)


def _sie(adapter, labels: list[str], multi_label: bool) -> np.ndarray:
    output = adapter.extract(
        [Item(text=text) for text in TEXTS],
        labels=labels,
        options={"hypothesis_template": TEMPLATE, "multi_label": multi_label},
    )
    return np.array(
        [[{c["label"]: c["score"] for c in row}[label] for label in labels] for row in output.classifications]
    )


def _reference(pipeline, labels: list[str], multi_label: bool) -> np.ndarray:
    rows = []
    for text in TEXTS:
        result = pipeline(text, candidate_labels=labels, hypothesis_template=TEMPLATE, multi_label=multi_label)
        by_label = dict(zip(result["labels"], result["scores"], strict=True))
        rows.append([by_label[label] for label in labels])
    return np.array(rows)


@pytest.mark.parametrize(
    ("labels", "multi_label"),
    [
        (["sports"], False),
        (["sports", "politics", "cooking"], False),
        (["sports", "politics", "cooking"], True),
    ],
    ids=["one-label", "exclusive", "multi-label"],
)
def test_scores_match_the_pipeline(adapter, pipeline, labels: list[str], multi_label: bool) -> None:
    # The adapter runs every (text, hypothesis) pair in one padded batch and the
    # pipeline does not, so float32 scores differ by up to about 3e-4. The scoring
    # bugs this guards against were off by 0.18 (multi_label) and 1.0 (one label).
    np.testing.assert_allclose(
        _sie(adapter, labels, multi_label), _reference(pipeline, labels, multi_label), rtol=0, atol=2e-3
    )


def test_one_label_scores_follow_the_text(adapter) -> None:
    scores = _sie(adapter, ["sports"], multi_label=False)[:, 0]

    assert scores[0] > 0.99
    assert scores[1] < 0.01
