from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sie_server.adapters.qwen2_flash_cross_encoder.adapter import (
    QWEN3_CHAT_PREFIX,
    QWEN3_CHAT_SUFFIX,
    Qwen2CrossEncoderAdapter,
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
    assert output.content_token_counts is not None
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


class _TextTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 4096

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        special = {"yes": [7], "no": [3], "1": [17], "0": [12]}
        return special.get(text, [ord(char) for char in text])


class _TinyCausalBody(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, *, input_ids, attention_mask, position_ids, use_cache, return_dict):
        assert torch.is_inference_mode_enabled()
        assert not use_cache
        assert return_dict
        self.calls.append((input_ids.clone(), attention_mask.clone(), position_ids.clone()))
        attended = input_ids * attention_mask
        hidden = torch.stack(
            (
                attended.cumsum(dim=1).float() / 1000,
                input_ids.float() / 1000,
                position_ids.float() / 1000,
                torch.ones_like(input_ids, dtype=torch.float32),
            ),
            dim=-1,
        )
        # A padded position would yield an observably different score.
        hidden = torch.where(attention_mask.unsqueeze(-1), hidden, -1000.0)
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyCausalModel(torch.nn.Module):
    def __init__(self, *, bias: bool = True) -> None:
        super().__init__()
        self.model = _TinyCausalBody()
        self.lm_head = torch.nn.Linear(4, 32, bias=bias)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(128).reshape(32, 4) / 1000)
            if self.lm_head.bias is not None:
                self.lm_head.bias.copy_(torch.linspace(-0.5, 0.5, 32))
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_key_value_heads=1,
            hidden_size=4,
            head_dim=4,
            num_hidden_layers=1,
            max_position_embeddings=4096,
        )


def _load_native(monkeypatch, *, input_format="qwen3", score_mode="log_softmax", bias=True, default_instruction=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = _TextTokenizer()
    model = _TinyCausalModel(bias=bias)
    tokenizer_loader = MagicMock(return_value=tokenizer)
    model_loader = MagicMock(return_value=model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", model_loader)
    positive, negative = ("yes", "no") if input_format == "qwen3" else ("1", "0")
    adapter = Qwen2CrossEncoderAdapter(
        "public/test-checkpoint",
        revision="a" * 40,
        max_seq_length=8192,
        compute_precision="bfloat16",
        trust_remote_code=False,
        yes_token=positive,
        no_token=negative,
        input_format=input_format,
        score_mode=score_mode,
        default_instruction=default_instruction,
    )
    adapter.load("cpu")
    return adapter, model, tokenizer_loader, model_loader


def test_native_load_preserves_source_and_uses_cpu_safe_sdpa(monkeypatch) -> None:
    adapter, model, tokenizer_loader, model_loader = _load_native(monkeypatch)

    tokenizer_loader.assert_called_once_with("public/test-checkpoint", trust_remote_code=False, revision="a" * 40)
    model_loader.assert_called_once_with(
        "public/test-checkpoint",
        trust_remote_code=False,
        revision="a" * 40,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    assert adapter._dtype == torch.float32
    assert adapter._max_seq_length == 4096
    assert not model.training
    assert not model.model.training
    assert model.lm_head.weight.device.type == "cpu"
    torch.testing.assert_close(adapter._score_weight, model.lm_head.weight[[3, 7]])
    torch.testing.assert_close(adapter._score_bias, model.lm_head.bias[[3, 7]])


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [
        (None, "float16", torch.float16),
        (None, "bfloat16", torch.bfloat16),
        (None, "float32", torch.float32),
        ("cpu", "bfloat16", torch.float32),
        ("cpu:0", "float16", torch.float32),
        ("cpu:0", "bfloat16", torch.float32),
        ("cpu:0", "float32", torch.float32),
        ("mps", "bfloat16", torch.float16),
        ("mps", "float32", torch.float32),
        ("mps:0", "float16", torch.float16),
        ("mps:0", "bfloat16", torch.float16),
        ("mps:0", "float32", torch.float32),
        ("cuda", "bfloat16", torch.bfloat16),
        ("cuda:0", "float16", torch.float16),
        ("cuda:0", "bfloat16", torch.bfloat16),
        ("cuda:1", "float32", torch.float32),
    ],
)
def test_native_precision_resolution_without_hardware(device, precision, dtype) -> None:
    adapter = Qwen2CrossEncoderAdapter("unused", compute_precision=precision)
    adapter._device = device
    assert adapter._resolve_dtype() == dtype


@pytest.mark.parametrize("instruction", [None, "", "A caller rule"])
def test_native_qwen3_pair_preserves_default_and_request_instructions(monkeypatch, instruction) -> None:
    adapter, _model, _tokenizer_loader, _model_loader = _load_native(
        monkeypatch, default_instruction="A configured rule"
    )
    effective = instruction or "A configured rule"
    expected = QWEN3_CHAT_PREFIX + f"<Instruct>: {effective}\n<Query>: query\n<Document>: document" + QWEN3_CHAT_SUFFIX

    assert adapter._build_input_ids("query", "document", instruction=instruction) == [ord(char) for char in expected]


def test_native_mxbai_pair_preserves_publisher_prompt_and_ignores_instruction(monkeypatch) -> None:
    adapter, _model, _tokenizer_loader, _model_loader = _load_native(monkeypatch, input_format="mxbai")
    expected = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nquery: query\ndocument: document\n"
        "You are a search relevance expert who evaluates how well documents match search queries. "
        "For each query-document pair, carefully analyze the semantic relationship between them, "
        "then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:"
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    assert adapter._build_input_ids("query", "document", instruction="unused rule") == [ord(char) for char in expected]


@pytest.mark.parametrize("input_format", ["mxbai", "qwen3"])
@pytest.mark.parametrize("score_mode", ["logit_diff", "log_softmax"])
@pytest.mark.parametrize("bias", [False, True])
def test_native_padding_projection_chunks_and_order_match_full_head(
    monkeypatch, input_format, score_mode, bias
) -> None:
    adapter, model, _tokenizer_loader, _model_loader = _load_native(
        monkeypatch, input_format=input_format, score_mode=score_mode, bias=bias
    )
    pairs = [[9, 4, 8, 6, 2], [1], [7, 8, 9], [3, 4], [1, 2, 3, 4, 5, 6, 7, 8]]
    expected_hidden = torch.tensor([[sum(ids) / 1000, ids[-1] / 1000, (len(ids) - 1) / 1000, 1] for ids in pairs])
    positive, negative = (7, 3) if input_format == "qwen3" else (17, 12)
    full_logits = model.lm_head(expected_hidden)[:, [negative, positive]]
    expected_scores = (
        full_logits[:, 1] - full_logits[:, 0]
        if score_mode == "logit_diff"
        else torch.softmax(full_logits, dim=-1)[:, 1]
    )

    with torch.inference_mode():
        logits = adapter._forward_pairs(pairs)
        scores = adapter._compute_scores(logits)
        singles = torch.cat([adapter._forward_pairs([ids]) for ids in pairs])

    torch.testing.assert_close(logits, full_logits)
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(logits, singles)
    chunks = model.model.calls[:2]
    assert [tuple(ids.shape) for ids, _mask, _positions in chunks] == [(3, 3), (2, 8)]
    assert all(ids.numel() <= sum(map(len, pairs)) for ids, _mask, _positions in chunks)
    assert chunks[0][1].tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    assert chunks[1][1].sum(dim=1).tolist() == [5, 8]
    for ids, mask, positions in chunks:
        assert torch.equal(positions, torch.arange(ids.shape[1]).expand(ids.shape[0], -1))
        assert torch.all(ids[~mask] == 0)


def test_native_uses_existing_eos_when_pad_token_is_missing(monkeypatch) -> None:
    adapter, model, _tokenizer_loader, _model_loader = _load_native(monkeypatch)
    adapter._tokenizer.pad_token_id = None

    with torch.inference_mode():
        adapter._forward_pairs([[4], [5, 6], [7, 8, 9]])

    ids, mask, _positions = model.model.calls[0]
    assert (~mask).any()
    assert torch.all(ids[~mask] == 1)
    assert adapter._tokenizer.pad_token_id is None


def test_native_score_pairs_preserves_admission_and_instruction_accounting(monkeypatch) -> None:
    adapter, model, _tokenizer_loader, _model_loader = _load_native(monkeypatch)
    queries = [Item(text="query"), Item(text="q2")]
    docs = [Item(text="x" * 100), Item(text="short")]
    instruction = "Keep this rule"
    template_length = len(adapter._build_input_ids("query", "", instruction=instruction)) - len("query")
    limit = template_length + len("query") + 7

    result = adapter.score_pairs(queries, docs, instruction=instruction, options={"max_seq_length": limit})

    assert result.input_token_counts == [limit, template_length + len("q2") + len("short")]
    assert result.content_token_counts == [
        len("query") + len(instruction) + 7,
        len("q2") + len(instruction) + len("short"),
    ]
    assert result.scores.dtype == np.float32
    assert len(result.scores) == 2
    assert sum(mask.sum().item() for _ids, mask, _positions in model.model.calls) == sum(result.input_token_counts)
    assert adapter.score_pairs([], []).scores.size == 0
    with pytest.raises(ValueError, match="equal-length"):
        adapter.score_pairs(queries, docs[:1])
    with pytest.raises(InvalidInputError, match="too small"):
        adapter.score_pairs(queries, docs, options={"max_seq_length": 1})


@pytest.mark.parametrize("architecture", ["qwen2", "qwen3"])
def test_native_sdpa_matches_tiny_causal_lm_on_unpadded_pairs(monkeypatch, architecture) -> None:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    torch.manual_seed(0)
    config_class, model_class = (
        (Qwen2Config, Qwen2ForCausalLM) if architecture == "qwen2" else (Qwen3Config, Qwen3ForCausalLM)
    )
    config = config_class(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = model_class(config)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", MagicMock(return_value=_TextTokenizer()))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", MagicMock(return_value=model))
    adapter = Qwen2CrossEncoderAdapter("unused", yes_token="yes", no_token="no", input_format="qwen3")  # noqa: S106
    adapter.load("cpu")
    pairs = [[9, 4, 8, 6, 2], [1], [7, 8, 9], [3, 4], [1, 2, 3, 4, 5, 6, 7, 8]]

    with torch.inference_mode():
        actual = adapter._forward_pairs(pairs)
        expected = torch.stack(
            [
                model(
                    input_ids=torch.tensor([ids]),
                    attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                    use_cache=False,
                ).logits[0, -1, [3, 7]]
                for ids in pairs
            ]
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
