"""The label contract is shared by NLI, Qwen and engine-backed decisions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sie_server.adapters.decision import d3, qev
from sie_server.adapters.decision.base import MAX_ITEM_TOKENS, Row
from sie_server.adapters.decision.qwen import QwenDecisionAdapter
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.nli_classification_flash import NLIClassificationFlashAdapter
from sie_server.adapters.sglang.decision import SGLangDecisionAdapter
from sie_server.types.inputs import InvalidInputError, Item


@pytest.fixture(params=["qev", "d3", "sglang"])
def decision(request):
    adapter = (
        SGLangDecisionAdapter("unused")
        if request.param == "sglang"
        else QwenDecisionAdapter("unused", decision_format=request.param)
    )
    adapter._check_decision_loaded = Mock()
    adapter._max_seq_length = adapter._max_forward_tokens = 64
    adapter._item_rows = Mock(
        side_effect=lambda index, item, questions: [
            Row(index, (question,), {"input_ids": torch.ones(len(item.text), dtype=torch.long)})
            for question in questions
        ]
    )
    adapter._forward = Mock(side_effect=lambda rows: [[0.25, 0.75] for row in rows for _ in row.questions])
    return adapter


def test_same_boolean_request_and_response_contract_as_nli(decision):
    nli = NLIClassificationFlashAdapter("unused")
    nli._device = "cpu"
    nli._entailment_idx = 0
    nli._tokenizer = Mock(
        return_value={"input_ids": torch.ones(1, 7, dtype=torch.long), "attention_mask": torch.ones(1, 7)}
    )
    nli._model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[1.0986123, 0.0]])))
    request = {"labels": ["Is this spam?"], "options": {"hypothesis_template": "{}", "multi_label": True}}
    for adapter in (nli, decision):
        output = adapter.extract([Item(text="message")], **request)
        assert output.classifications == [[{"label": "Is this spam?", "score": pytest.approx(0.75)}]]
        assert output.entities == [[]]
        assert output.data is None
        assert output.input_token_counts == [7]
    question = decision._item_rows.call_args.args[2][0]
    assert question.kind == "noul"
    assert question.instructions == "Is this spam?"


def test_single_label_keeps_independent_probability_without_multi_label_option(decision):
    output = decision.extract([Item(text="hello")], labels=["spam"])
    assert output.classifications == [[{"label": "spam", "score": 0.75}]]
    assert decision._item_rows.call_args.args[2][0].instructions == "This text is about spam."


def test_exclusive_labels_return_one_distribution_in_score_order(decision):
    output = decision.extract([Item(text="hello")], labels=["support", "billing"])
    assert output.classifications == [[{"label": "billing", "score": 0.75}, {"label": "support", "score": 0.25}]]
    assert output.input_token_counts == [5]
    question = decision._item_rows.call_args.args[2][0]
    assert question.kind == "choice"
    assert question.descriptions == ("0: This text is about support.", "1: This text is about billing.")


def test_independent_labels_preserve_partial_errors_and_authoritative_meters(decision):
    output = decision.extract(
        [Item(text="one"), Item(text="x" * 65), Item(text="last")],
        labels=["positive", "helpful"],
        options={"multi_label": True, "hypothesis_template": "It is {0}"},
    )
    assert output.classifications == [
        [{"label": "positive", "score": 0.75}, {"label": "helpful", "score": 0.75}],
        [],
        [{"label": "positive", "score": 0.75}, {"label": "helpful", "score": 0.75}],
    ]
    assert output.errors[1].code == "INPUT_TOO_LONG"
    assert output.errors[0] is output.errors[2] is None
    assert output.input_token_counts == [6, 0, 8]
    assert [row.item for call in decision._forward.call_args_list for row in call.args[0]] == [2, 2, 0, 0]


@pytest.mark.parametrize(
    "arguments",
    [
        {"labels": []},
        {"labels": [""]},
        {"labels": [1]},
        {"labels": "spam"},
        {"labels": ["spam"], "output_schema": {"q": {"type": "noul"}}},
        {"labels": ["spam"], "instruction": "extra"},
        {"labels": ["spam"], "options": {"multi_label": "false"}},
        {"labels": ["spam"], "options": {"hypothesis_template": "{0:1000000000}"}},
        {"labels": ["spam"], "options": {"hypothesis_template": "{0.__class__}"}},
        {"labels": ["spam"], "options": {"hypothesis_template": "{label}"}},
        {"labels": ["spam"], "options": {"hypothesis_template": "no placeholder"}},
        {"labels": ["spam"], "options": {"hypothesis_template": "{"}},
        {"labels": ["x" * 65536]},
        {"labels": ["x"] * 33, "options": {"multi_label": True}},
        {"labels": ["spam"], "options": {"overflow_policy": "truncate"}},
        {"labels": ["spam"], "options": {"temperature": 0.5}},
        {"output_schema": {"q": {"type": "noul"}}, "options": {"multi_label": True}},
    ],
)
def test_invalid_classification_fails_before_preparation_or_inference(decision, arguments):
    with pytest.raises(InvalidInputError):
        decision.extract([Item(text="hello")], **arguments)
    assert decision.extract_item_costs([Item(text="hello")], **arguments) is None
    decision._item_rows.assert_not_called()
    decision._forward.assert_not_called()


def test_costs_account_for_all_independent_hypotheses(decision):
    items = [Item(text="a message")]
    one = decision.extract_item_costs(items, labels=["spam"], options={"multi_label": True})
    two = decision.extract_item_costs(items, labels=["spam", "spam"], options={"multi_label": True})
    assert two == [2 * one[0]]


def test_typed_questions_keep_their_existing_output(decision):
    output = decision.extract([Item(text="hello")], output_schema={"spam": {"type": "noul"}})
    assert output.classifications is None
    assert output.data == [{"spam": {"type": "noul", "noul": 0.75, "answer": True, "confidence": 0.75}}]


def test_loaded_costs_are_the_unpadded_tokens_of_the_rendered_rows(decision):
    items = [Item(text="a message"), Item(text="x" * 65)]
    costs = decision.extract_item_costs(items, labels=["spam", "ham"], options={"multi_label": True})
    assert costs == [18, 130]
    assert [call.args[1] for call in decision._item_rows.call_args_list] == items


def test_unloaded_or_unrenderable_costs_use_a_bounded_character_estimate(decision):
    items = [Item(text="a message about an invoice")]
    exact = decision.extract_item_costs(items, labels=["spam"])
    decision._check_decision_loaded.side_effect = RuntimeError("not loaded")
    unloaded = decision.extract_item_costs(items, labels=["spam"])
    decision._check_decision_loaded.side_effect = None
    decision._item_rows.side_effect = ValueError("template")
    unrenderable = decision.extract_item_costs(items, labels=["spam"])
    question = decision._extract_questions(["spam"], None, None, None)[0]
    chars = len(items[0].text) + len(question.instructions) + sum(map(len, question.descriptions))
    assert exact == [len(items[0].text)]
    assert unloaded == unrenderable == [chars + 8 * len(question.keys) + 384]
    # Under half of the previous estimate, 2 * (characters + 512 per question).
    assert unloaded[0] < chars + 8 * len(question.keys) + 512


def test_costs_of_items_extract_rejects_stay_at_the_limit(decision):
    decision._item_rows.side_effect = InputTooLongError("too long")
    assert decision.extract_item_costs([Item(text="hello")], labels=["spam"]) == [MAX_ITEM_TOKENS * 4]
    assert decision.extract_item_costs([Item(text="a", images=[{"data": b"x"}])], labels=["spam"]) == [
        MAX_ITEM_TOKENS * 4
    ]


class _CharTokenizer:
    all_special_tokens = ("<|im_start|>", "<|im_end|>")
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        _ = kwargs
        return (
            "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages) + "<|im_start|>assistant\n"
        )

    def encode(self, text, **kwargs):
        _ = kwargs
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        _ = kwargs
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


@pytest.mark.parametrize("decision_format", ["d3", "qev"])
def test_qwen_costs_match_the_prompts_extract_tokenizes(decision_format):
    adapter = QwenDecisionAdapter("unused", decision_format=decision_format)
    adapter._model, adapter._tokenizer, adapter._codes = Mock(), _CharTokenizer(), ("A", "B", "C")
    schema = {
        "topic": {"type": "choice", "instructions": "Topic?", "criteria": {"world": None, "sports": None}},
        "spam": {"type": "noul"},
    }
    state = "Stocks rallied after the earnings report."
    questions = adapter._parse_questions(schema)
    rendered = (
        d3.prompts(adapter._tokenizer, state, questions, adapter._codes)
        if decision_format == "d3"
        else qev.prompts(adapter._tokenizer, state, questions)
    )
    (cost,) = adapter.extract_item_costs([Item(text=state)], output_schema=schema)
    assert cost == sum(len(ids) for ids, _ in rendered)
