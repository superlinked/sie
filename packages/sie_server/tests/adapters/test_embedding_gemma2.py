"""Text recipe for EmbeddingGemma2Adapter: prompts, mean pool, Matryoshka."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters.embedding_gemma2.adapter import (
    DOCUMENT_TEMPLATE,
    QUERY_TEMPLATE,
    EmbeddingGemma2Adapter,
    _mean_pool,
)
from sie_server.types.inputs import AudioInput, Item


def _adapter(**kwargs: Any) -> EmbeddingGemma2Adapter:
    dense_dim = int(kwargs.pop("dense_dim", 4))
    adapter = EmbeddingGemma2Adapter("google/embeddinggemma-2", revision="abc", dense_dim=dense_dim, **kwargs)
    adapter._device = "cpu"
    adapter._model = object()
    adapter._tokenizer = _FakeTokenizer()
    adapter._dense_dim = dense_dim
    return adapter


class _FakeTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] | None = None
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        self.texts = list(texts)
        self.kwargs = kwargs
        # One content token per word, plus a padding column on the shorter row.
        lengths = [max(1, len(text.split())) for text in texts]
        width = max(lengths)
        input_ids = []
        mask = []
        for length in lengths:
            input_ids.append([1] * length + [0] * (width - length))
            mask.append([1] * length + [0] * (width - length))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(mask),
        }


class _FakeModel:
    def __init__(self, last_hidden: torch.Tensor) -> None:
        self._last_hidden = last_hidden
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        batch, seq = kwargs["input_ids"].shape
        hidden = self._last_hidden
        if hidden.shape[0] != batch or hidden.shape[1] != seq:
            hidden = hidden[:, :1].expand(batch, seq, hidden.shape[-1]).clone()
        return SimpleNamespace(last_hidden_state=hidden)


def test_mean_pool_skips_padding() -> None:
    hidden = torch.tensor(
        [
            [[2.0, 0.0], [6.0, 0.0], [100.0, 0.0]],
            [[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = _mean_pool(hidden, mask)
    assert torch.allclose(pooled[0], torch.tensor([4.0, 0.0]))
    assert torch.allclose(pooled[1], torch.tensor([0.0, 3.0]))


def test_query_and_document_prompts_are_included_in_the_pool() -> None:
    adapter = _adapter()
    hidden = torch.tensor([[[3.0, 4.0]]])
    adapter._model = _FakeModel(hidden)  # ty: ignore[invalid-assignment]

    query = adapter.encode([Item(text="mars")], ["dense"], is_query=True)
    document = adapter.encode([Item(text="mars")], ["dense"], is_query=False)

    assert adapter._tokenizer.texts == [DOCUMENT_TEMPLATE.format(text="mars")]  # ty: ignore[unresolved-attribute]
    adapter.encode([Item(text="mars")], ["dense"], is_query=True)
    assert adapter._tokenizer.texts == [QUERY_TEMPLATE.format(text="mars")]  # ty: ignore[unresolved-attribute]
    # [3, 4] normalized is [0.6, 0.8]. Prompt tokens are part of the string the
    # tokenizer saw, so they are inside the pooled span.
    assert query.dense is not None
    assert document.dense is not None
    assert pytest.approx(query.dense[0].tolist(), abs=1e-5) == [0.6, 0.8]
    assert pytest.approx(document.dense[0].tolist(), abs=1e-5) == [0.6, 0.8]
    assert query.extra["input_token_counts"] == [len(QUERY_TEMPLATE.format(text="mars").split())]


def test_empty_template_disables_the_task_prompt() -> None:
    adapter = _adapter()
    adapter._model = _FakeModel(torch.tensor([[[1.0, 0.0]]]))  # ty: ignore[invalid-assignment]
    adapter.encode(
        [Item(text="mars")],
        ["dense"],
        is_query=True,
        options={"query_template": None},
    )
    assert adapter._tokenizer.texts == ["mars"]  # ty: ignore[unresolved-attribute]


def test_truncate_dim_slices_then_renormalizes() -> None:
    adapter = _adapter(dense_dim=768)
    adapter._dense_dim = 768
    hidden = torch.zeros(1, 1, 768)
    hidden[0, 0, 0] = 3.0
    hidden[0, 0, 1] = 4.0
    adapter._model = _FakeModel(hidden)  # ty: ignore[invalid-assignment]
    output = adapter.encode([Item(text="mars")], ["dense"], options={"truncate_dim": 128})
    assert output.dense is not None
    assert output.dense.shape == (1, 128)
    assert output.dense_dim == 128
    # The leading [3, 4] pair is re-normalized after the slice; the rest is zero
    # and does not change the direction of that pair.
    assert pytest.approx(output.dense[0, 0].item(), abs=1e-5) == 0.6
    assert pytest.approx(output.dense[0, 1].item(), abs=1e-5) == 0.8


def test_native_truncate_dim_keeps_the_full_vector() -> None:
    adapter = _adapter(dense_dim=768)
    adapter._dense_dim = 768
    adapter._model = _FakeModel(torch.ones(1, 1, 768))  # ty: ignore[invalid-assignment]
    output = adapter.encode([Item(text="mars")], ["dense"], options={"truncate_dim": 768})
    assert output.dense is not None
    assert output.dense.shape == (1, 768)


@pytest.mark.parametrize("value", [64, 384, 0, True, "256"])
def test_truncate_dim_rejects_untrained_widths(value: object) -> None:
    adapter = _adapter()
    adapter._model = _FakeModel(torch.ones(1, 1, 4))  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="truncate_dim"):
        adapter.encode([Item(text="mars")], ["dense"], options={"truncate_dim": value})


def test_non_mean_pooling_is_rejected() -> None:
    with pytest.raises(ValueError, match="mean"):
        EmbeddingGemma2Adapter("google/embeddinggemma-2", pooling="last")
    adapter = _adapter()
    adapter._model = _FakeModel(torch.ones(1, 1, 4))  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="mean"):
        adapter.encode([Item(text="mars")], ["dense"], options={"pooling": "cls"})


def test_non_text_modalities_are_refused() -> None:
    adapter = _adapter()
    image = Item(text="a cat", images=[{"data": b"not-an-image"}])
    audio = Item(audio=AudioInput(data=b"abc", format="wav"))
    video = Item(video={"data": b"mp4"})
    for item, modality in ((image, "image"), (audio, "audio"), (video, "video")):
        with pytest.raises(ValueError, match=modality):
            adapter.encode([item], ["dense"])


def test_encode_requires_a_loaded_model() -> None:
    adapter = EmbeddingGemma2Adapter("google/embeddinggemma-2")
    with pytest.raises(RuntimeError, match=re.escape(ERR_NOT_LOADED)):
        adapter.encode([Item(text="mars")], ["dense"])


def test_trust_remote_code_is_refused() -> None:
    with pytest.raises(ValueError, match="trust_remote_code"):
        EmbeddingGemma2Adapter("google/embeddinggemma-2", trust_remote_code=True)


def test_load_refuses_transformers_before_5_19(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.17.0", raising=False)
    adapter = EmbeddingGemma2Adapter("google/embeddinggemma-2", revision="f" * 40)
    with pytest.raises(RuntimeError, match=r"5\.19\.0"):
        adapter.load("cpu")


def test_load_drops_modality_towers_and_pins_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.19.0", raising=False)
    config = SimpleNamespace(vision_config={"keep": False}, audio_config={"keep": False}, text_config=None)
    model = MagicMock()
    model.config.text_config.embedding_dim = 768
    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = config
    auto_tokenizer = MagicMock()
    embedding_cls = MagicMock()
    embedding_cls.from_pretrained.return_value = model
    monkeypatch.setattr(transformers, "AutoConfig", auto_config, raising=False)
    monkeypatch.setattr(transformers, "AutoTokenizer", auto_tokenizer, raising=False)
    monkeypatch.setattr(transformers, "EmbeddingGemma2Model", embedding_cls, raising=False)

    revision = "f9972ac22e696a105c482202646979e33e4b5a1e"
    adapter = EmbeddingGemma2Adapter(
        "google/embeddinggemma-2",
        revision=revision,
        dense_dim=768,
        compute_precision="float16",
    )
    adapter.load("cpu")

    assert config.vision_config is None
    assert config.audio_config is None
    assert adapter._dense_dim == 768
    for loader in (auto_config, auto_tokenizer, embedding_cls):
        kwargs = loader.from_pretrained.call_args.kwargs
        assert kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == revision
    model_kwargs = embedding_cls.from_pretrained.call_args.kwargs
    assert model_kwargs["config"] is config
    assert model_kwargs["torch_dtype"] == torch.float32
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once()


def test_float16_on_cuda_upgrades_to_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.22.1", raising=False)
    config = SimpleNamespace(vision_config=object(), audio_config=object())
    model = MagicMock()
    model.config.text_config.embedding_dim = 768
    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = config
    monkeypatch.setattr(transformers, "AutoConfig", auto_config, raising=False)
    monkeypatch.setattr(transformers, "AutoTokenizer", MagicMock(), raising=False)
    embedding_cls = MagicMock()
    embedding_cls.from_pretrained.return_value = model
    monkeypatch.setattr(transformers, "EmbeddingGemma2Model", embedding_cls, raising=False)

    adapter = EmbeddingGemma2Adapter("google/embeddinggemma-2", compute_precision="float16", dense_dim=768)
    adapter.load("cuda:0")
    assert embedding_cls.from_pretrained.call_args.kwargs["torch_dtype"] == torch.bfloat16
