from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.core.inference_output import EncodeOutput
from sie_server.types.inputs import Item


class _StubFlashAdapter(FlashBaseAdapter):
    """Minimal concrete FlashBaseAdapter used to exercise shared helpers."""

    spec = AdapterSpec(inputs=("text",), outputs=("dense",), unload_fields=())

    def load(self, device: str) -> None:  # pragma: no cover - not exercised
        pass

    def encode(  # pragma: no cover - not exercised
        self,
        items: list[Item],
        output_types: list[str],
        **kwargs: Any,
    ) -> EncodeOutput:
        return EncodeOutput(batch_size=len(items))


def _tok(model_max_length: Any) -> Any:
    return SimpleNamespace(model_max_length=model_max_length)


def _model(max_position_embeddings: Any) -> Any:
    return SimpleNamespace(config=SimpleNamespace(max_position_embeddings=max_position_embeddings))


class TestResolveTokenizerCeiling:
    """Tests for FlashBaseAdapter._resolve_tokenizer_ceiling."""

    def test_clamps_when_requested_exceeds_tokenizer_cap(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        adapter = _StubFlashAdapter()
        with caplog.at_level(logging.WARNING):
            result = adapter._resolve_tokenizer_ceiling(
                _tok(1024),
                _model(1024),
                requested=8192,
            )
        assert result == 1024
        # Warning fired and names the adapter
        assert any("clamping to 1024" in rec.message for rec in caplog.records)
        assert any("_StubFlashAdapter" in rec.message for rec in caplog.records)

    def test_no_clamp_when_requested_fits(self) -> None:
        adapter = _StubFlashAdapter()
        result = adapter._resolve_tokenizer_ceiling(
            _tok(8192),
            _model(8192),
            requested=512,
        )
        assert result == 512

    def test_uses_min_of_tokenizer_and_model_caps(self) -> None:
        adapter = _StubFlashAdapter()
        # tokenizer says 4096, model says 1024 — tighter wins
        result = adapter._resolve_tokenizer_ceiling(
            _tok(4096),
            _model(1024),
            requested=8192,
        )
        assert result == 1024

    def test_ignores_hf_sentinel_model_max_length(self) -> None:
        adapter = _StubFlashAdapter()
        # HuggingFace sentinel int(1e30) — should be treated as "unknown"
        result = adapter._resolve_tokenizer_ceiling(
            _tok(int(1e30)),
            _model(2048),
            requested=8192,
        )
        assert result == 2048

    def test_ignores_missing_max_position_embeddings(self) -> None:
        adapter = _StubFlashAdapter()
        result = adapter._resolve_tokenizer_ceiling(
            _tok(1024),
            _model(None),
            requested=8192,
        )
        assert result == 1024

    def test_returns_requested_when_no_caps_available(self) -> None:
        adapter = _StubFlashAdapter()
        result = adapter._resolve_tokenizer_ceiling(
            _tok(int(1e30)),
            _model(None),
            requested=8192,
        )
        assert result == 8192

    def test_handles_none_model(self) -> None:
        """Used by adapters that load weights from raw safetensors (e.g. nomic_flash)."""
        adapter = _StubFlashAdapter()
        result = adapter._resolve_tokenizer_ceiling(
            _tok(2048),
            None,
            requested=8192,
        )
        assert result == 2048

    def test_handles_none_tokenizer(self) -> None:
        adapter = _StubFlashAdapter()
        result = adapter._resolve_tokenizer_ceiling(
            None,
            _model(1024),
            requested=8192,
        )
        assert result == 1024

    def test_handles_zero_or_negative_caps(self) -> None:
        """Defensive: caps must be positive ints to count."""
        adapter = _StubFlashAdapter()
        result = adapter._resolve_tokenizer_ceiling(
            _tok(0),
            _model(-1),
            requested=8192,
        )
        assert result == 8192
