from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from sie_server.adapters.jina_flash_cross_encoder import JinaFlashCrossEncoderAdapter
from sie_server.types.inputs import Item


class _StubTokenizer:
    """Minimal stand-in for a HF tokenizer that records call kwargs."""

    def __init__(self, *, model_max_length: int) -> None:
        self.model_max_length = model_max_length
        self.last_call_kwargs: dict[str, Any] | None = None

    def __call__(
        self,
        a: list[str],
        b: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        self.last_call_kwargs = {"a": a, "b": b, **kwargs}
        batch = len(a)
        # Return tensors that match the requested max_length so downstream
        # plumbing keeps shape sanity even though we replace the model output.
        seq = int(kwargs.get("max_length", 8))
        return {
            "input_ids": torch.zeros((batch, seq), dtype=torch.long),
            "attention_mask": torch.ones((batch, seq), dtype=torch.long),
        }


class _StubModel:
    """Stand-in HF model returning constant logits."""

    def __init__(self, *, num_labels: int = 1, max_position_embeddings: int = 1024) -> None:
        self.config = SimpleNamespace(
            num_labels=num_labels,
            max_position_embeddings=max_position_embeddings,
        )

    def __call__(self, **encodings: torch.Tensor) -> Any:
        batch = next(iter(encodings.values())).shape[0]
        return SimpleNamespace(logits=torch.zeros((batch, 1)))


def _make_loaded_adapter(
    *,
    configured_max_seq_length: int = 8192,
    tokenizer_max_length: int = 1024,
    model_pos_embed: int = 1024,
) -> JinaFlashCrossEncoderAdapter:
    """Build an adapter as if load() had run successfully on CPU."""
    adapter = JinaFlashCrossEncoderAdapter(
        model_name_or_path="stub/model",
        max_seq_length=configured_max_seq_length,
    )
    adapter._tokenizer = _StubTokenizer(model_max_length=tokenizer_max_length)
    adapter._model = _StubModel(max_position_embeddings=model_pos_embed)
    adapter._device = "cpu"
    adapter._dtype = torch.float32
    # Apply the same clamp load() would do.
    adapter._max_seq_length = adapter._resolve_tokenizer_ceiling(
        adapter._tokenizer,
        adapter._model,
        adapter._max_seq_length,
    )
    adapter._tokenizer_max_length = adapter._max_seq_length
    return adapter


class TestJinaLoadTimeClamp:
    """The adapter must lower its configured max_seq_length to the model's ceiling."""

    def test_clamps_when_yaml_oversells_capacity(self) -> None:
        adapter = _make_loaded_adapter(
            configured_max_seq_length=8192,
            tokenizer_max_length=1024,
            model_pos_embed=1024,
        )
        assert adapter._max_seq_length == 1024
        assert adapter._tokenizer_max_length == 1024

    def test_no_clamp_when_already_within_capacity(self) -> None:
        adapter = _make_loaded_adapter(
            configured_max_seq_length=512,
            tokenizer_max_length=1024,
            model_pos_embed=1024,
        )
        assert adapter._max_seq_length == 512
        assert adapter._tokenizer_max_length == 512


class TestJinaScoreTimeClamp:
    """Runtime overrides must never push past the resolved ceiling."""

    def test_score_pairs_truncates_to_clamped_length(self) -> None:
        adapter = _make_loaded_adapter()
        # Caller asks for 4096 via runtime options — must still be capped at 1024.
        out = adapter.score_pairs(
            queries=[Item(text="q")],
            docs=[Item(text="d")],
            options={"max_seq_length": 4096},
        )
        assert isinstance(out.scores, np.ndarray)
        kwargs = adapter._tokenizer.last_call_kwargs
        assert kwargs is not None
        assert kwargs["max_length"] == 1024
        assert kwargs["truncation"] is True

    def test_score_truncates_to_clamped_length(self) -> None:
        adapter = _make_loaded_adapter()
        scores = adapter.score(query=Item(text="q"), items=[Item(text="d")])
        assert len(scores) == 1
        kwargs = adapter._tokenizer.last_call_kwargs
        assert kwargs is not None
        assert kwargs["max_length"] == 1024

    def test_score_pairs_uses_clamp_when_no_runtime_override(self) -> None:
        adapter = _make_loaded_adapter()
        adapter.score_pairs(queries=[Item(text="q")], docs=[Item(text="d")])
        kwargs = adapter._tokenizer.last_call_kwargs
        assert kwargs is not None
        assert kwargs["max_length"] == 1024

    def test_score_pairs_ignores_none_runtime_override(self) -> None:
        adapter = _make_loaded_adapter()
        adapter.score_pairs(
            queries=[Item(text="q")],
            docs=[Item(text="d")],
            options={"max_seq_length": None},
        )
        kwargs = adapter._tokenizer.last_call_kwargs
        assert kwargs is not None
        assert kwargs["max_length"] == 1024
        assert kwargs["truncation"] is True

    def test_score_pairs_coerces_string_runtime_override(self) -> None:
        adapter = _make_loaded_adapter()
        # A numeric string asking for 4096 must still be capped at the
        # load-time ceiling of 1024.
        adapter.score_pairs(
            queries=[Item(text="q")],
            docs=[Item(text="d")],
            options={"max_seq_length": "4096"},
        )
        kwargs = adapter._tokenizer.last_call_kwargs
        assert kwargs is not None
        assert kwargs["max_length"] == 1024
        assert kwargs["truncation"] is True

    def test_score_pairs_ignores_negative_runtime_override(self) -> None:
        adapter = _make_loaded_adapter()
        adapter.score_pairs(
            queries=[Item(text="q")],
            docs=[Item(text="d")],
            options={"max_seq_length": -1},
        )
        kwargs = adapter._tokenizer.last_call_kwargs
        assert kwargs is not None
        assert kwargs["max_length"] == 1024
        assert kwargs["truncation"] is True

    def test_score_ignores_runtime_override_options(self) -> None:
        # ``score()`` does not consult ``options`` at all — it always uses
        # the load-time ceiling. Pass a malformed override to confirm it
        # never reaches the tokenizer.
        adapter = _make_loaded_adapter()
        adapter.score(
            query=Item(text="q"),
            items=[Item(text="d")],
            options={"max_seq_length": "4096"},  # type: ignore[arg-type]
        )
        kwargs = adapter._tokenizer.last_call_kwargs
        assert kwargs is not None
        assert kwargs["max_length"] == 1024


class TestJinaBundledConfig:
    """Sanity-check the bundled YAML so the regression doesn't reappear.

    jinaai/jina-reranker-v2-base-multilingual is an XLM-RoBERTa model with
    absolute position embeddings capped at 1024 tokens. The bundled YAML
    used to advertise 8192, which bypassed tokenizer truncation and
    crashed the CUDA worker on long inputs (issue #718).
    """

    def test_bundled_yaml_max_sequence_length_matches_model_capacity(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "models" / "jinaai__jina-reranker-v2-base-multilingual.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert data["max_sequence_length"] == 1024
