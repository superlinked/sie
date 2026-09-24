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


class TestTokenizerLoading:
    def test_transformers5_tokenizer_class_falls_back_to_fast_tokenizer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import transformers

        calls: list[tuple[str, dict[str, object]]] = []

        def auto(name: str, **kwargs: object) -> object:
            raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")

        def fast(name: str, **kwargs: object) -> str:
            calls.append((name, kwargs))
            return "fast-tokenizer"

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", auto)
        monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", fast)
        adapter = GLiClassAdapter("org/model", revision="abc123")

        assert adapter._load_tokenizer({"revision": "abc123"}) == "fast-tokenizer"
        assert calls == [("org/model", {"revision": "abc123"})]

    def test_other_tokenizer_errors_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import transformers

        def auto(name: str, **kwargs: object) -> object:
            raise ValueError("Unrecognized model identifier")

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", auto)

        with pytest.raises(ValueError, match="Unrecognized model identifier"):
            GLiClassAdapter("org/model")._load_tokenizer({})


class TestModernBertRopeCompatibility:
    """Checkpoints saved with transformers 5 keep ModernBERT RoPE bases in rope_parameters."""

    @staticmethod
    def _config(**overrides: object) -> object:
        from transformers import ModernBertConfig

        fields: dict[str, object] = {"num_hidden_layers": 4, "global_attn_every_n_layers": 3}
        fields.update(overrides)
        return ModernBertConfig(**fields)

    def test_rope_parameters_fill_the_transformers4_fields(self) -> None:
        from sie_server.adapters.gliclass import _restore_modernbert_rope_fields

        config = self._config(
            rope_parameters={
                "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            },
            layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        )
        assert config.local_rope_theta == 10000.0  # the transformers 4 default a v5 checkpoint would get

        assert _restore_modernbert_rope_fields(config) is True
        assert config.local_rope_theta == 160000.0
        assert config.global_rope_theta == 160000.0

    def test_transformers4_checkpoints_are_left_alone(self) -> None:
        from sie_server.adapters.gliclass import _restore_modernbert_rope_fields

        config = self._config(local_rope_theta=10000.0, global_rope_theta=160000.0)

        assert _restore_modernbert_rope_fields(config) is False
        assert config.local_rope_theta == 10000.0

    def test_non_modernbert_encoders_are_left_alone(self) -> None:
        from sie_server.adapters.gliclass import _restore_modernbert_rope_fields
        from transformers import DebertaV2Config

        assert _restore_modernbert_rope_fields(DebertaV2Config()) is False

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"rope_parameters": {"sliding_attention": {"rope_theta": 1.0, "rope_type": "yarn"}}}, "not supported"),
            ({"rope_parameters": {"rope_theta": 1.0, "rope_type": "default"}}, "keys"),
            (
                {
                    "rope_parameters": {"full_attention": {"rope_theta": 1.0}},
                    "layer_types": ["full_attention", "full_attention", "sliding_attention", "full_attention"],
                },
                "layer_types",
            ),
        ],
    )
    def test_layouts_transformers4_cannot_express_are_rejected(self, overrides: dict[str, object], match: str) -> None:
        from sie_server.adapters.gliclass import _restore_modernbert_rope_fields

        with pytest.raises(ValueError, match=match):
            _restore_modernbert_rope_fields(self._config(**overrides))
