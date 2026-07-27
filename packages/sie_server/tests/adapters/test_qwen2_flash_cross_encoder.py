from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sie_server.adapters.qwen2_flash_cross_encoder.adapter import (
    QWEN3_CHAT_PREFIX,
    Qwen2FlashCrossEncoderAdapter,
)
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import InvalidInputError, InvalidMediaError, Item


class _CharTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text)))


def test_qwen3_chat_prefix_matches_checkpoint_template() -> None:
    assert QWEN3_CHAT_PREFIX == (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )


def test_position_ids_restart_without_scalar_syncs() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused")
    cu_seqlens = torch.tensor([0, 3, 4, 8], dtype=torch.int32)

    assert torch.equal(
        adapter._build_position_ids(cu_seqlens, batch_size=3),
        torch.tensor([0, 1, 2, 0, 0, 1, 2, 3]),
    )


def test_score_projection_matches_selected_full_vocabulary_logits() -> None:
    torch.manual_seed(0)
    adapter = Qwen2FlashCrossEncoderAdapter("unused")
    lm_head = torch.nn.Linear(4, 11, bias=True)
    token_ids = torch.tensor([7, 3])
    adapter._score_weight = lm_head.weight.index_select(0, token_ids)
    adapter._score_bias = lm_head.bias.index_select(0, token_ids)
    hidden = torch.randn(5, 4)

    expected = lm_head(hidden).index_select(1, token_ids)

    torch.testing.assert_close(adapter._project_score_logits(hidden), expected)


def test_score_delegates_to_score_pairs_with_instruction_and_options() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused")
    adapter._model = object()
    adapter._tokenizer = object()  # ty: ignore[invalid-assignment]
    query = Item(text="query")
    docs = [Item(text="first"), Item(text="second")]
    options = {"max_seq_length": 512}
    delegate = MagicMock(
        return_value=ScoreOutput(scores=np.array([0.25, 0.75], dtype=np.float32)),
    )
    adapter.score_pairs = delegate  # ty: ignore[invalid-assignment]

    scores = adapter.score(
        query,
        docs,
        instruction="instruction",
        options=options,
    )

    assert scores == [0.25, 0.75]
    assert isinstance(scores, list)
    delegate.assert_called_once_with(
        [query, query],
        docs,
        instruction="instruction",
        options=options,
    )


def test_score_preserves_empty_items_and_loaded_checks() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused")

    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.score(Item(text="query"), [])

    adapter._model = object()
    assert adapter.score(Item(text="query"), []) == []


def test_score_uses_score_pairs_tokenizer_check_for_nonempty_items() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused")
    adapter._model = object()

    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.score(Item(text="query"), [Item(text="document")])


def test_qwen3_score_pairs_surfaces_exact_truncated_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = Qwen2FlashCrossEncoderAdapter(
        "unused",
        max_seq_length=4096,
        input_format="qwen3",
        score_mode="log_softmax",
    )
    adapter._tokenizer = _CharTokenizer()  # ty: ignore[invalid-assignment]
    adapter._model = object()
    adapter._device = "cpu"
    adapter._pre_tokenize_templates()
    monkeypatch.setattr(
        adapter,
        "_forward_flash",
        lambda *_args, **_kwargs: torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )

    query = Item(text="query")
    docs = [Item(text="x" * 100), Item(text="short")]
    query_template_length = len(adapter._build_input_ids("query", "", max_length=4096))
    max_length = query_template_length + 7

    output = adapter.score_pairs(
        [query, query],
        docs,
        options={"max_seq_length": max_length},
    )

    assert output.input_token_counts == [max_length, query_template_length + len("short")]
    assert output.input_token_counts is not None
    assert all(count <= max_length for count in output.input_token_counts)


def test_qwen3_rejects_query_template_larger_than_model_window() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused", input_format="qwen3")
    adapter._tokenizer = _CharTokenizer()  # ty: ignore[invalid-assignment]
    adapter._pre_tokenize_templates()
    fixed_chat_length = len(adapter._chat_prefix_ids) + len(adapter._chat_suffix_ids)

    with pytest.raises(InvalidInputError, match="too small for the reranker instruction, query"):
        adapter._build_input_ids_qwen3(
            "query",
            "document",
            max_length=fixed_chat_length + 1,
        )


def test_text_reranker_runtime_max_length_is_strict_and_clamped() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused", max_seq_length=1024)

    assert adapter._runtime_max_length(None) == 1024
    assert adapter._runtime_max_length(2048) == 1024
    assert adapter._runtime_max_length(512) == 512
    for value in (True, 0, -1, "512"):
        with pytest.raises(InvalidInputError, match="positive integer"):
            adapter._runtime_max_length(value)


@pytest.mark.parametrize("text", [None, "", "   "])
def test_text_reranker_rejects_missing_or_blank_text(text: str | None) -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused")

    with pytest.raises(InvalidInputError, match="requires"):
        adapter._extract_text_only(Item(text=text))


def test_text_reranker_rejects_silently_ignored_media() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused")

    with pytest.raises(InvalidMediaError, match="text-only"):
        adapter._extract_text_only(Item(text="query", images=[{"data": b"image"}]))
