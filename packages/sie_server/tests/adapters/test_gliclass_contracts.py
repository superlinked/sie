from unittest.mock import MagicMock

import pytest
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.types.inputs import Item


class _CountingTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int | None,
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens is True
        counts = [len(text.split()) + 2 for text in texts]
        if truncation:
            assert max_length is not None
            counts = [min(count, max_length) for count in counts]
        return {"input_ids": [list(range(count)) for count in counts]}


def _adapter(results: object) -> GLiClassAdapter:
    adapter = GLiClassAdapter("test-model", max_seq_length=512)
    pipeline = MagicMock()
    pipeline.return_value = results
    adapter._pipeline = pipeline
    adapter._tokenizer = _CountingTokenizer()  # type: ignore[assignment]
    return adapter


def test_exact_document_token_counts_align_with_classification_batch() -> None:
    adapter = _adapter(
        [
            {"positive": 0.8, "negative": 0.2},
            {"positive": 0.1, "negative": 0.9},
        ]
    )

    output = adapter.extract(
        [Item(text="two words"), Item(text="three whole words")],
        labels=["positive", "negative"],
    )

    assert output.input_token_counts == [4, 5]
    assert output.classifications is not None
    assert len(output.classifications) == 2


@pytest.mark.parametrize("threshold", [True, "0.5", None, -0.1, 1.1, float("nan"), float("inf"), 10**1000])
def test_threshold_rejects_non_finite_and_non_numeric_values(threshold: object) -> None:
    adapter = _adapter([{"positive": 0.8, "negative": 0.2}])

    with pytest.raises(ValueError, match="finite number between 0 and 1"):
        adapter.extract(
            [Item(text="hello")],
            labels=["positive", "negative"],
            options={"threshold": threshold},
        )


@pytest.mark.parametrize("labels", [[""], ["positive", "positive"], ["   "]])
def test_labels_must_be_non_empty_and_unique(labels: list[str]) -> None:
    adapter = _adapter([{}])

    with pytest.raises(ValueError, match="labels"):
        adapter.extract([Item(text="hello")], labels=labels)


@pytest.mark.parametrize("score", [True, "0.8", -0.1, 1.1, float("nan"), float("inf"), 10**1000])
def test_pipeline_scores_must_be_finite_probabilities(score: object) -> None:
    adapter = _adapter([{"positive": score, "negative": 0.2}])

    with pytest.raises(ValueError, match="invalid classification score"):
        adapter.extract(
            [Item(text="hello")],
            labels=["positive", "negative"],
        )


@pytest.mark.parametrize(
    "result",
    [
        {"positive": 0.8},
        {"positive": 0.8, "negative": 0.1, "unexpected": 0.1},
    ],
)
def test_pipeline_label_set_must_match_requested_labels(result: dict[str, float]) -> None:
    adapter = _adapter([result])

    with pytest.raises(ValueError, match="requested label set"):
        adapter.extract(
            [Item(text="hello")],
            labels=["positive", "negative"],
        )
