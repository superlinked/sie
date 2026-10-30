from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sie_server.adapters.qwen2_flash_cross_encoder.adapter import (
    QWEN3_CHAT_PREFIX,
    QWEN3_CHAT_SUFFIX,
    Qwen2FlashCrossEncoderAdapter,
)
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import InvalidInputError, InvalidMediaError, Item
from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast


class _CharTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text)))

    def __call__(self, texts: list[str], **_kwargs: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [self.encode(text) for text in texts]}


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


def test_bfloat16_logits_near_certainty_still_rank_apart() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused", score_mode="log_softmax")
    # [no, yes] logits whose P(yes) all round to 1.0 in bfloat16.
    logits = torch.tensor([[0.0, 6.0], [0.0, 7.0], [0.0, 9.0]], dtype=torch.bfloat16)

    scores = adapter._compute_scores(logits)

    assert scores.dtype == torch.float32
    assert scores[0] < scores[1] < scores[2]


def test_score_projection_runs_in_float32_for_bfloat16_states() -> None:
    torch.manual_seed(0)
    adapter = Qwen2FlashCrossEncoderAdapter("unused")
    lm_head = torch.nn.Linear(4, 11, bias=True).to(torch.bfloat16)
    token_ids = torch.tensor([7, 3])
    # As cached at load: the two score rows in float32.
    adapter._score_weight = lm_head.weight.index_select(0, token_ids).float()
    adapter._score_bias = lm_head.bias.index_select(0, token_ids).float()
    hidden = torch.randn(5, 4).to(torch.bfloat16)

    projected = adapter._project_score_logits(hidden)

    assert projected.dtype == torch.float32
    expected = torch.nn.functional.linear(hidden.float(), adapter._score_weight, adapter._score_bias)
    torch.testing.assert_close(projected, expected)


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


def _char_qwen3_adapter(monkeypatch: pytest.MonkeyPatch, pairs: int) -> Qwen2FlashCrossEncoderAdapter:
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
        lambda *_args, **_kwargs: torch.zeros((pairs, 2)),
    )
    return adapter


def test_qwen3_content_counts_exclude_the_template_and_default_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _char_qwen3_adapter(monkeypatch, pairs=2)
    query = Item(text="query")
    docs = [Item(text="x" * 100), Item(text="short")]
    template_length = len(adapter._build_input_ids("query", "", max_length=4096)) - len("query")
    max_length = template_length + len("query") + 7

    output = adapter.score_pairs([query, query], docs, options={"max_seq_length": max_length})

    assert output.content_token_counts == [len("query") + 7, len("query") + len("short")]
    assert output.input_token_counts == [template_length + count for count in output.content_token_counts]


def test_qwen3_content_counts_bill_a_supplied_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _char_qwen3_adapter(monkeypatch, pairs=2)
    queries = [Item(text="q1"), Item(text="query two")]
    docs = [Item(text="doc"), Item(text="document")]

    supplied = adapter.score_pairs(queries, docs, instruction="Find the rule")
    empty = adapter.score_pairs(queries, docs, instruction="")
    default = adapter.score_pairs(queries, docs)

    assert supplied.content_token_counts == [
        len("q1") + len("Find the rule") + len("doc"),
        len("query two") + len("Find the rule") + len("document"),
    ]
    assert (
        empty.content_token_counts
        == default.content_token_counts
        == [
            len("q1") + len("doc"),
            len("query two") + len("document"),
        ]
    )
    assert empty.input_token_counts == default.input_token_counts


def test_mxbai_format_reports_no_content_split(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused", max_seq_length=4096)
    adapter._tokenizer = _CharTokenizer()  # ty: ignore[invalid-assignment]
    adapter._model = object()
    adapter._device = "cpu"
    adapter._pre_tokenize_templates()
    monkeypatch.setattr(adapter, "_forward_flash", lambda *_args, **_kwargs: torch.zeros((1, 2)))

    output = adapter.score_pairs([Item(text="query")], [Item(text="doc")])

    assert output.input_token_counts is not None
    assert output.content_token_counts is None


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


# Qwen2's pre-tokenizer (tokenizer.json of Qwen/Qwen3-Reranker-4B). BPE merges never cross a
# pre-token boundary, so two texts whose pre-tokens differ get different ids from the real tokenizer.
_QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _qwen2_pre_tokenizer() -> pre_tokenizers.PreTokenizer:
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(_QWEN2_SPLIT), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


def _qwen2_word_tokenizer(texts: list[str]) -> PreTrainedTokenizerFast:
    """One id per Qwen2 pre-token of ``texts``: ids match exactly when the real tokenizer's would."""
    pre = _qwen2_pre_tokenizer()
    pieces = sorted({piece for text in texts for piece, _ in pre.pre_tokenize_str(text)})
    vocab = {token: index for index, token in enumerate(["<unk>", *pieces])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))  # noqa: S106 -- a vocabulary entry, not a secret
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize(
    "document",
    [
        "License Agreement: Spirit Airlines and Navitaire, dated February 28, 2007\n5.1 Term.",
        "5.1 Term. Unless otherwise terminated earlier, this Agreement shall continue.",
        "(g) Section 6.1 of the Marketing Agreement is hereby amended in its entirety.",
        '"Initial Term" means ten (10) years.',
        "\nSection 4B, Annual Payments, is hereby amended.",
        " a document that starts with a space",
    ],
)
def test_qwen3_rows_are_the_model_cards_token_ids(document: str, monkeypatch: pytest.MonkeyPatch) -> None:
    instruction = "Prefer the text that controls today: an amendment supersedes the clause it amends."
    query = "What is the initial term of the hosted services agreement?"
    # The model card: the whole user text tokenized as one string between the template's two ends.
    user = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    # The vocabulary also holds the pieces of the text before the document and of the document alone, so a row
    # that tokenizes them apart (a lone " " then "License") gets real ids that differ, not unknown ones.
    split_user = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "
    tokenizer = _qwen2_word_tokenizer([QWEN3_CHAT_PREFIX, QWEN3_CHAT_SUFFIX, user, split_user, document])

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    card = encode(QWEN3_CHAT_PREFIX) + encode(user) + encode(QWEN3_CHAT_SUFFIX)
    adapter = Qwen2FlashCrossEncoderAdapter(
        "unused", max_seq_length=4096, input_format="qwen3", score_mode="log_softmax"
    )
    adapter._tokenizer = tokenizer
    adapter._model = object()
    adapter._device = "cpu"
    adapter._pre_tokenize_templates()

    assert adapter._build_input_ids(query, document, instruction=instruction) == card

    # The batched path builds the same row and bills the caller's own text, not the template's space.
    rows: list[list[int]] = []

    def capture(batch: list[tuple[list[int], list[int]]], _lengths: list[int]) -> torch.Tensor:
        rows.extend(prefix + tail for prefix, tail in batch)
        return torch.zeros((len(batch), 2))

    monkeypatch.setattr(adapter, "_forward_rows", capture)
    output = adapter.score_pairs([Item(text=query)], [Item(text=document)], instruction=instruction)

    assert rows == [card]
    # The document's tokens as the row holds them, less the separating space where it is a token of its own.
    # That is the document tokenized alone, except where its first character joins the space (" (" "g" for "(g").
    in_row = encode(" " + document)
    document_tokens = len(in_row) - int(in_row[:1] == encode(" "))
    assert output.content_token_counts == [len(encode(query)) + len(encode(instruction)) + document_tokens]
    if document[:1].isalnum():
        assert document_tokens == len(encode(document))
