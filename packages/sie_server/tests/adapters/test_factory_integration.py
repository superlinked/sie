from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    ModelConfig,
    ProfileConfig,
    ScoreTask,
    Tasks,
)
from sie_server.core.loader import load_adapter


def _make_config(
    sie_id: str,
    hf_id: str,
    adapter_path: str,
    dense_dim: int | None = None,
    sparse_dim: int | None = None,
    multivector_dim: int | None = None,
    score: bool = False,
) -> ModelConfig:
    encode = EncodeTask(
        dense=EmbeddingDim(dim=dense_dim) if dense_dim else None,
        sparse=EmbeddingDim(dim=sparse_dim) if sparse_dim else None,
        multivector=EmbeddingDim(dim=multivector_dim) if multivector_dim else None,
    )
    return ModelConfig(
        sie_id=sie_id,
        hf_id=hf_id,
        tasks=Tasks(encode=encode, score=ScoreTask() if score else None),
        profiles={"default": ProfileConfig(adapter_path=adapter_path, max_batch_tokens=8192)},
    )


class TestAdapterFactoryIntegration:
    """Integration tests for factory method device-aware swapping."""

    def test_bert_flash_cross_encoder_on_cpu_returns_fallback(self, tmp_path: Path) -> None:
        """BERT flash cross-encoder returns CrossEncoderAdapter on CPU."""
        config = _make_config(
            "test-bert-flash",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            score=True,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "CrossEncoderAdapter"
        assert adapter.capabilities.outputs == ["score"]

    def test_bert_flash_on_mps_returns_sentence_transformer(self, tmp_path: Path) -> None:
        """BERT flash dense adapter returns SentenceTransformerDenseAdapter on MPS."""
        config = _make_config(
            "test-bert-flash",
            "intfloat/e5-base-v2",
            "sie_server.adapters.bert_flash:BertFlashAdapter",
            dense_dim=768,
        )

        adapter = load_adapter(config, tmp_path, device="mps")

        assert type(adapter).__name__ == "SentenceTransformerDenseAdapter"
        assert adapter.capabilities.outputs == ["dense"]

    def test_modernbert_flash_cross_encoder_on_cpu(self, tmp_path: Path) -> None:
        """ModernBERT flash cross-encoder falls back on CPU."""
        config = _make_config(
            "test-modernbert",
            "Alibaba-NLP/gte-reranker-modernbert-base",
            "sie_server.adapters.modernbert_flash_cross_encoder:ModernBertFlashCrossEncoderAdapter",
            score=True,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "CrossEncoderAdapter"

    def test_jina_flash_cross_encoder_on_mps(self, tmp_path: Path) -> None:
        """Jina flash cross-encoder falls back on MPS."""
        config = _make_config(
            "test-jina",
            "jinaai/jina-reranker-v2-base-multilingual",
            "sie_server.adapters.jina_flash_cross_encoder:JinaFlashCrossEncoderAdapter",
            score=True,
        )

        adapter = load_adapter(config, tmp_path, device="mps")

        assert type(adapter).__name__ == "CrossEncoderAdapter"

    def test_qwen2_flash_cross_encoder_on_cpu(self, tmp_path: Path) -> None:
        """Qwen2 flash cross-encoder falls back on CPU."""
        config = _make_config(
            "test-qwen2",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "sie_server.adapters.qwen2_flash_cross_encoder:Qwen2FlashCrossEncoderAdapter",
            score=True,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "CrossEncoderAdapter"

    def test_nomic_flash_on_cpu(self, tmp_path: Path) -> None:
        """Nomic flash falls back to SentenceTransformer on CPU."""
        config = _make_config(
            "test-nomic",
            "nomic-ai/nomic-embed-text-v1",
            "sie_server.adapters.nomic_flash:NomicFlashAdapter",
            dense_dim=768,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "SentenceTransformerDenseAdapter"

    def test_qwen2_flash_dense_on_mps(self, tmp_path: Path) -> None:
        """Qwen2 flash dense falls back on MPS."""
        config = _make_config(
            "test-qwen2-dense",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "sie_server.adapters.qwen2_flash:Qwen2FlashAdapter",
            dense_dim=1536,
        )

        adapter = load_adapter(config, tmp_path, device="mps")

        assert type(adapter).__name__ == "SentenceTransformerDenseAdapter"

    def test_rope_flash_on_cpu(self, tmp_path: Path) -> None:
        """RoPE flash falls back on CPU."""
        config = _make_config(
            "test-rope",
            "jinaai/jina-embeddings-v3",
            "sie_server.adapters.rope_flash:RoPEFlashAdapter",
            dense_dim=1024,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "SentenceTransformerDenseAdapter"

    def test_bge_m3_flash_on_cpu(self, tmp_path: Path) -> None:
        """BGE-M3 flash falls back to BGEM3Adapter on CPU."""
        config = _make_config(
            "test-bge-m3",
            "BAAI/bge-m3",
            "sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter",
            dense_dim=1024,
            sparse_dim=250002,
            multivector_dim=1024,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "BGEM3Adapter"

    def test_modernbert_flash_on_cpu_returns_fallback(self, tmp_path: Path) -> None:
        """ModernBERT flash dense falls back to SentenceTransformerDenseAdapter on CPU."""
        config = _make_config(
            "test-modernbert-dense",
            "Alibaba-NLP/gte-modernbert-base",
            "sie_server.adapters.modernbert_flash:ModernBERTFlashAdapter",
            dense_dim=768,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "SentenceTransformerDenseAdapter"

    def test_colbert_modernbert_flash_on_mps(self, tmp_path: Path) -> None:
        """ColBERT ModernBERT flash falls back to ColBERTAdapter on MPS."""
        config = _make_config(
            "test-colbert-modernbert",
            "answerdotai/ModernBERT-base",
            "sie_server.adapters.colbert_modernbert_flash:ColBERTModernBERTFlashAdapter",
            multivector_dim=128,
        )

        adapter = load_adapter(config, tmp_path, device="mps")

        assert type(adapter).__name__ == "ColBERTAdapter"

    def test_colbert_rotary_flash_on_cpu(self, tmp_path: Path) -> None:
        """ColBERT Rotary flash falls back to ColBERTAdapter on CPU."""
        config = _make_config(
            "test-colbert-rotary",
            "jinaai/jina-colbert-v2",
            "sie_server.adapters.colbert_rotary_flash:ColBERTRotaryFlashAdapter",
            multivector_dim=128,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "ColBERTAdapter"

    def test_splade_flash_on_cpu_returns_same_adapter(self, tmp_path: Path) -> None:
        """SPLADE flash returns SPLADEFlashAdapter on CPU (uses SDPA fallback)."""
        config = _make_config(
            "test-splade",
            "naver/splade-v3",
            "sie_server.adapters.splade_flash:SPLADEFlashAdapter",
            sparse_dim=30522,
        )

        adapter = load_adapter(config, tmp_path, device="cpu")

        assert type(adapter).__name__ == "SPLADEFlashAdapter"

    def test_bert_flash_cross_encoder_on_cuda_with_flash_attn(self, tmp_path: Path) -> None:
        """BERT flash cross-encoder returns flash adapter on CUDA when flash-attn is installed."""
        pytest.importorskip("flash_attn", reason="flash-attn not installed")

        config = _make_config(
            "test-bert-flash-cuda",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            score=True,
        )

        with patch("sie_server.core.inference.is_flash_attention_available", return_value=True):
            adapter = load_adapter(config, tmp_path, device="cuda:0")

        assert type(adapter).__name__ == "BertFlashCrossEncoderAdapter"

    def test_flash_adapter_logs_helpful_message_on_fallback(self, tmp_path: Path, caplog) -> None:
        """Flash adapters log helpful messages when falling back."""
        config = _make_config(
            "test-log",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "sie_server.adapters.bert_flash_cross_encoder:BertFlashCrossEncoderAdapter",
            score=True,
        )

        with caplog.at_level(logging.INFO):
            adapter = load_adapter(config, tmp_path, device="cpu")

        # Verify helpful log message (comes from base class helper now)
        assert "requires CUDA" in caplog.text or "flash-attn not installed" in caplog.text
        assert "CrossEncoderAdapter" in caplog.text
        assert type(adapter).__name__ == "CrossEncoderAdapter"
