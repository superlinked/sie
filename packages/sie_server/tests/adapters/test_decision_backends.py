"""Independent attention math and engine-boundary checks for typed decisions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import torch
from sie_server.adapters._generation_base import GenerationAdapter
from sie_server.adapters.decision.base import Row
from sie_server.adapters.decision.questions import parse_questions
from sie_server.adapters.decision.qwen import QwenDecisionAdapter, prefix_attention
from sie_server.adapters.sglang.decision import SGLangDecisionAdapter
from sie_server.types.inputs import InvalidInputError, Item


def test_prefix_attention_matches_dense_block_mask_with_padding_and_grouped_heads():
    torch.manual_seed(91)
    query = torch.randn(2, 4, 9, 8, dtype=torch.float64)
    key, value = (torch.randn(2, 2, 9, 8, dtype=torch.float64) for _ in range(2))
    bounds = ((2, 5), (0, 3))
    actual, weights = prefix_attention(None, query, key, value, bounds, scaling=0.25)
    dense = torch.zeros(2, 1, 9, 9, dtype=torch.bool)
    for row, (first, end) in enumerate(bounds):
        dense[row, :, first:end, first:end] = True
        dense[row, :, end:, first:] = True
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key.repeat_interleave(2, 1), value.repeat_interleave(2, 1), attn_mask=dense, scale=0.25
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert weights is None


def test_d3_rejects_ambiguous_or_oversized_score_levels():
    adapter = QwenDecisionAdapter("unused", decision_format="d3")
    assert adapter._parse_questions({"score": {"type": "score", "criteria": ["low", "high"]}})
    with pytest.raises(InvalidInputError, match="distinct"):
        adapter._parse_questions({"score": {"type": "score", "criteria": ["low", "low"]}})
    with pytest.raises(InvalidInputError, match="10 levels"):
        adapter._parse_questions({"score": {"type": "score", "criteria": [str(i) for i in range(11)]}})


def scoring_adapter():
    adapter = SGLangDecisionAdapter("unused")
    adapter._engine = SimpleNamespace(server_url="http://localhost:30000", _check_loaded=Mock())
    adapter._tokenizer = object()
    adapter._session = Mock()
    adapter._code_ids = (10, 20, 30)
    return adapter


def test_sglang_scoring_reuses_engine_without_advertising_generation():
    adapter = scoring_adapter()
    assert not isinstance(adapter, GenerationAdapter)
    assert adapter.spec.outputs == ("json",)
    questions = parse_questions(
        {
            "choice": {"type": "choice", "criteria": {"left": None, "right": None, "center": None}},
            "yesno": {"type": "noul"},
        }
    )
    rows = [Row(0, (question,), {"input_ids": torch.tensor([4, 5, 6])}) for question in questions]
    adapter._session.post.return_value.json.return_value = [
        {"meta_info": {"prompt_tokens": 3, "output_token_ids_logprobs": [[[-2.0, 30], [-3.0, 10], [-1.0, 20]]]}},
        {"meta_info": {"prompt_tokens": 3, "output_token_ids_logprobs": [[[-1.0, 10], [-2.0, 20]]]}},
    ]
    actual = adapter._forward(rows)
    assert actual[0] == pytest.approx(torch.tensor([-3.0, -1.0, -2.0]).softmax(0).tolist())
    assert actual[1] == pytest.approx(torch.tensor([-2.0, -1.0]).softmax(0).tolist())
    payload = adapter._session.post.call_args.kwargs["json"]
    assert payload["input_ids"] == [[4, 5, 6], [4, 5, 6]]
    assert payload["token_ids_logprob"] == [[10, 20, 30], [10, 20]]
    assert payload["sampling_params"]["max_new_tokens"] == 1
    assert payload["sampling_params"]["presence_penalty"] == 0.0
    assert adapter._answer(questions[0], [1 / 3] * 3)["confidence"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "entries", [[[0.0, 10]], [[0.0, 10], [0.0, 10]], [[None, 10], [0.0, 20]], [[float("nan"), 10], [0.0, 20]]]
)
def test_sglang_rejects_missing_duplicate_or_nonfinite_selected_labels(entries):
    adapter = scoring_adapter()
    question = parse_questions({"q": {"type": "noul"}})[0]
    row = Row(0, (question,), {"input_ids": torch.tensor([4])})
    adapter._session.post.return_value.json.return_value = [
        {"meta_info": {"prompt_tokens": 1, "output_token_ids_logprobs": [entries]}}
    ]
    with pytest.raises(ValueError, match=r"selected label|nonfinite"):
        adapter._forward([row])


def test_sglang_timeout_aborts_only_its_own_decision_requests():
    adapter = scoring_adapter()
    question = parse_questions({"q": {"type": "noul"}})[0]
    rows = [Row(0, (question,), {"input_ids": torch.tensor([4])}) for _ in range(2)]
    adapter._session.post.side_effect = [requests.Timeout("deadline"), Mock()]
    with pytest.raises(requests.Timeout, match="deadline"):
        adapter._forward(rows)
    generate, abort = adapter._session.post.call_args_list
    identifiers = generate.kwargs["json"]["rid"]
    assert len(identifiers) == len(set(identifiers)) == 2
    assert abort.args == ("http://localhost:30000/abort_request",)
    assert abort.kwargs["json"] == {"rid": identifiers}


def test_decision_admission_is_atomic_per_item_and_preserves_order():
    adapter = scoring_adapter()
    adapter._max_seq_length, adapter._max_forward_tokens = 4, 8
    question = parse_questions({"q": {"type": "noul"}})[0]
    adapter._item_rows = lambda index, item, _: [
        Row(index, (question,), {"input_ids": torch.ones(n, dtype=torch.long)})
        for n in ([2, 5] if index == 1 else [index + 1])
    ]
    forwards = []

    def forward(rows):
        forwards.extend(row.item for row in rows)
        return [[0.25, 0.75] for _ in rows]

    adapter._forward = forward
    output = adapter.extract(
        [Item(text="first"), Item(text="oversized"), Item(text="last")], output_schema={"q": {"type": "noul"}}
    )
    assert forwards == [2, 0]
    assert output.errors[1].code == "INPUT_TOO_LONG"
    assert output.data[1] == {}
    assert output.data[0]["q"]["noul"] == output.data[2]["q"]["noul"] == 0.75
    assert output.input_token_counts == [1, 0, 3]


def test_knowline_nested_state_is_an_item_error_without_losing_healthy_neighbors():
    adapter = scoring_adapter()
    adapter._codes = ("A", "B")
    adapter._tokenizer = SimpleNamespace(
        all_special_tokens=(),
        apply_chat_template=lambda turns, **_: "".join(str(turn["content"]) for turn in turns),
        encode=lambda text, **_: list(map(ord, text)),
    )
    adapter._forward = lambda rows: [[0.25, 0.75] for _ in rows]
    nested = "x"
    for _ in range(200):
        nested = [nested]
    output = adapter.extract(
        [Item(text="first"), Item(metadata={"state": nested}), Item(text="last")],
        output_schema={"q": {"type": "noul"}},
    )
    assert output.errors[1].code == "INPUT_TOO_LONG"
    assert output.errors[0] is output.errors[2] is None
    assert output.data[1] == {}
    assert output.data[0]["q"]["noul"] == output.data[2]["q"]["noul"] == 0.75
