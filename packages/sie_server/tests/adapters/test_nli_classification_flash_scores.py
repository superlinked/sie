"""Score normalisation of NLI zero-shot classification.

The flash adapter must score labels the way transformers'
ZeroShotClassificationPipeline does:

- one label, or ``multi_label``: every label on its own, a softmax over that
  label's [contradiction, entailment] logits;
- two or more labels otherwise: a softmax of the entailment logits across labels.

A softmax over one label's entailment logit is 1.0 whatever the text, and a
sigmoid of the entailment logit ignores the contradiction class, so both are
pinned here against the pipeline's formula on fixed logits.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from sie_server.adapters.nli_classification_flash import NLIClassificationFlashAdapter
from sie_server.types.inputs import Item

# Heads of the shipped configs: MoritzLaurer zeroshot-v2.0 (binary),
# cross-encoder/nli-deberta-v3-base, and facebook/bart-large-mnli.
_BINARY = 0  # {0: entailment, 1: not_entailment}
_CROSS_ENCODER = 1  # {0: contradiction, 1: entailment, 2: neutral}
_MNLI = 2  # {0: contradiction, 1: neutral, 2: entailment}


class _Tokenizer:
    """One row of inputs per (text, hypothesis) pair."""

    def __call__(self, texts: list[str], hypotheses: list[str], **_: Any) -> dict[str, torch.Tensor]:
        rows = len(texts)
        return {
            "input_ids": torch.zeros(rows, 4, dtype=torch.long),
            "attention_mask": torch.ones(rows, 4, dtype=torch.long),
        }


class _Model:
    """Returns fixed logits, one row per (text, hypothesis) pair in request order."""

    def __init__(self, logits: list[list[float]]) -> None:
        self._logits = torch.tensor(logits, dtype=torch.float32)

    def __call__(self, **inputs: torch.Tensor) -> SimpleNamespace:
        assert inputs["input_ids"].shape[0] == self._logits.shape[0]
        return SimpleNamespace(logits=self._logits)


def _adapter(
    logits: list[list[float]], entailment_idx: int, *, multi_label: bool = False
) -> NLIClassificationFlashAdapter:
    adapter = NLIClassificationFlashAdapter("test-model", multi_label=multi_label)
    adapter._tokenizer = cast("Any", _Tokenizer())
    adapter._model = _Model(logits)
    adapter._device = "cpu"
    adapter._entailment_idx = entailment_idx
    return adapter


def _pipeline_scores(
    logits: list[list[float]], n_texts: int, n_labels: int, entailment_idx: int, *, multi_label: bool
) -> np.ndarray:
    """Transformers ZeroShotClassificationPipeline.postprocess, on raw logits."""
    reshaped = np.asarray(logits, dtype=np.float64).reshape(n_texts, n_labels, -1)
    if multi_label or n_labels == 1:
        contradiction_idx = -1 if entailment_idx == 0 else 0
        pair = np.exp(reshaped[..., [contradiction_idx, entailment_idx]])
        return (pair / pair.sum(-1, keepdims=True))[..., 1]
    entail = np.exp(reshaped[..., entailment_idx])
    return entail / entail.sum(-1, keepdims=True)


def _scores(adapter: NLIClassificationFlashAdapter, texts: list[str], labels: list[str], **kwargs: Any) -> np.ndarray:
    output = adapter.extract([Item(text=text) for text in texts], labels=labels, **kwargs)
    assert output.classifications is not None
    return np.array(
        [[{c["label"]: c["score"] for c in row}[label] for label in labels] for row in output.classifications]
    )


@pytest.mark.parametrize(
    ("entailment_idx", "says_yes", "says_no"),
    [
        # The "no" row is the real [entailment, not_entailment] pair of
        # MoritzLaurer/deberta-v3-base-zeroshot-v2.0 for "This text is about sports."
        # against a bread recipe.
        (_BINARY, [5.6, -6.1], [-4.854, 5.337]),
        (_CROSS_ENCODER, [-5.2, 6.0, -1.1], [6.647, -4.061, -2.0]),
        (_MNLI, [-3.0, -0.5, 4.2], [4.0, 0.2, -3.5]),
    ],
)
def test_one_label_is_scored_on_its_own(entailment_idx: int, says_yes: list[float], says_no: list[float]) -> None:
    logits = [says_yes, says_no]
    adapter = _adapter(logits, entailment_idx)

    scores = _scores(adapter, ["The Lakers beat the Celtics.", "Knead the dough and bake."], ["sports"])

    expected = _pipeline_scores(logits, 2, 1, entailment_idx, multi_label=False)
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-6)
    # Before, a softmax over the single label made both scores 1.0.
    assert scores[0, 0] > 0.99
    assert scores[1, 0] < 0.01


@pytest.mark.parametrize("entailment_idx", [_BINARY, _CROSS_ENCODER, _MNLI])
def test_multi_label_weighs_entailment_against_contradiction(entailment_idx: int) -> None:
    rng = np.random.default_rng(entailment_idx)
    width = 2 if entailment_idx == _BINARY else 3
    logits = rng.normal(0.0, 3.0, size=(2 * 3, width)).round(3).tolist()
    adapter = _adapter(logits, entailment_idx)

    scores = _scores(
        adapter, ["first text", "second text"], ["politics", "health", "economy"], options={"multi_label": True}
    )

    np.testing.assert_allclose(
        scores, _pipeline_scores(logits, 2, 3, entailment_idx, multi_label=True), rtol=0, atol=1e-6
    )


def test_multi_label_reads_the_contradiction_logit() -> None:
    # cross-encoder/nli-deberta-v3-base on a stadium-funding text with "politics":
    # contradiction -4.341, entailment 0.095, neutral 3.643. The pipeline gives 0.988;
    # a sigmoid of the entailment logit alone gave 0.524.
    logits = [[-4.341, 0.095, 3.643]]
    adapter = _adapter(logits, _CROSS_ENCODER, multi_label=True)

    (score,) = _scores(adapter, ["The council voted to fund a stadium."], ["politics"])[0]

    assert score == pytest.approx(1.0 / (1.0 + np.exp(-4.341 - 0.095)), abs=1e-6)
    assert score > 0.98


@pytest.mark.parametrize("entailment_idx", [_BINARY, _CROSS_ENCODER, _MNLI])
def test_two_or_more_labels_share_one_distribution(entailment_idx: int) -> None:
    rng = np.random.default_rng(10 + entailment_idx)
    width = 2 if entailment_idx == _BINARY else 3
    logits = rng.normal(0.0, 3.0, size=(2 * 3, width)).round(3).tolist()
    adapter = _adapter(logits, entailment_idx)

    scores = _scores(adapter, ["first text", "second text"], ["sports", "politics", "cooking"])

    np.testing.assert_allclose(
        scores, _pipeline_scores(logits, 2, 3, entailment_idx, multi_label=False), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=0, atol=1e-6)


def test_request_option_overrides_the_profile_mode() -> None:
    logits = [[1.0, -1.0], [-2.0, 2.0]]
    single = _adapter(logits, _BINARY, multi_label=True)
    multi = _adapter(logits, _BINARY, multi_label=False)

    exclusive = _scores(single, ["text"], ["a", "b"], options={"multi_label": False})
    independent = _scores(multi, ["text"], ["a", "b"], options={"multi_label": True})

    np.testing.assert_allclose(exclusive, _pipeline_scores(logits, 1, 2, _BINARY, multi_label=False), atol=1e-6)
    np.testing.assert_allclose(independent, _pipeline_scores(logits, 1, 2, _BINARY, multi_label=True), atol=1e-6)
