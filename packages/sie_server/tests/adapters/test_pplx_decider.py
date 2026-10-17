"""Decision request bounds, faithful output semantics and native head behavior."""

import json
from types import SimpleNamespace

import pytest
import torch
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.pplx_decider.adapter import PerplexityDecisionAdapter
from sie_server.adapters.pplx_decider.model import DecisionModel, checkpoint_config
from sie_server.adapters.pplx_decider.questions import answer, describe, messages, parse_questions
from sie_server.core.batcher import BatchConfig, BatchFormer
from sie_server.core.extract_cost import adapter_extract_item_costs, build_extract_prepared_items
from sie_server.types.inputs import InvalidInputError, Item


def choice(**changes):
    return {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": "second"}, **changes}


@pytest.mark.parametrize(
    "definition",
    [
        {},
        {"type": "other"},
        choice(criteria={}),
        choice(criteria={str(i): None for i in range(256)}),
        choice(criteria={"": None}),
        choice(criteria={1: None}),
        choice(images=[]),
        {"type": "score", "criteria": ["only"]},
        {"type": "score", "criteria": ["a", None]},
        {"type": "noul", "criteria": {"other": "bad"}},
        {"type": "noul", "criteria": []},
    ],
)
def test_invalid_questions_fail_before_tokenization(definition):
    with pytest.raises(InvalidInputError):
        parse_questions({"q": definition})


@pytest.mark.parametrize("value", [False, 0])
def test_falsey_non_content_is_not_treated_as_an_omitted_instruction(value):
    with pytest.raises(InvalidInputError, match="instructions"):
        parse_questions({"q": choice(instructions=value)})
    for key in ("false", "true"):
        with pytest.raises(InvalidInputError, match="descriptions"):
            parse_questions({"q": {"type": "noul", "criteria": {key: value}}})


def test_question_limits_are_aggregate():
    with pytest.raises(InvalidInputError, match="1 to 32"):
        parse_questions({str(i): choice() for i in range(33)})
    with pytest.raises(InvalidInputError, match="1024-option"):
        parse_questions({str(i): choice(criteria={str(j): None for j in range(255)}) for i in range(5)})
    with pytest.raises(InvalidInputError, match="character"):
        parse_questions({str(i): choice(instructions="a" * 40_000) for i in range(2)})


def test_reference_prompt_preserves_json_and_option_order():
    question = parse_questions({"q": {"type": "choice", "criteria": {"z": {"label": "é"}, "a": None}}})[0]
    result = messages(describe({"state": ["é", 1]}, 100), question, ("A", "B"))
    assert result[0]["content"].startswith("Classify the supplied state")
    assert result[1]["content"] == [
        {
            "type": "text",
            "text": (
                'State:\n{"state": ["é", 1]}\n\nQuestion:\nChoose the best matching option.\n\nOptions:\n'
                'A: z: {"label": "é"}\nB: a\n\nReturn only the letter code of the best option.'
            ),
        }
    ]


@pytest.mark.parametrize("value", ["abcdef", {"long": "abcdef"}, ["abcdef"]])
def test_character_budget_rejects_without_truncating(value):
    with pytest.raises(InputTooLongError):
        describe(value, 3)


def test_non_json_and_cycles_are_refused():
    for value in ({"bad": float("nan")}, {"bad": object()}):
        with pytest.raises(InvalidInputError):
            describe(value, 100)
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(InvalidInputError):
        describe(cyclic, 100)


def test_all_answer_types_and_reference_confidence():
    questions = parse_questions(
        {
            "choice": {"type": "choice", "criteria": {"first": None, "second": None, "third": None}},
            "noul": {"type": "noul"},
            "score": {"type": "score", "criteria": ["low", "medium", "high"]},
        }
    )
    picked = answer(questions[0], [0.1, 0.8, 0.1])
    assert picked["choice"] == "second"
    assert picked["confidence"] == pytest.approx(0.7)
    assert answer(questions[1], [0.2, 0.8]) == {"type": "noul", "noul": 0.8, "answer": True, "confidence": 0.8}
    scored = answer(questions[2], [0.1, 0.1, 0.8])
    assert scored["score"] == pytest.approx(1.7)
    assert scored["confidence"] == pytest.approx(0.55)
    assert scored["legend"] == {"0": "low", "1": "medium", "2": "high"}
    only = parse_questions({"only": {"type": "choice", "criteria": {"sole": None}}})[0]
    assert answer(only, [1.0])["confidence"] == 1.0


@pytest.mark.parametrize("values", [[0, 0], [-1, 2], [float("nan"), 1], [float("inf"), 0], [1]])
def test_invalid_model_probabilities_fail(values):
    with pytest.raises(ValueError, match="Decision model"):
        answer(parse_questions({"q": choice()})[0], values)


class Backbone(torch.nn.Module):
    def forward(self, input_ids, attention_mask, *, use_cache):
        assert use_cache is False
        # Deterministic full-row representation; sensitive to padding and text.
        values = input_ids.float() * attention_mask
        hidden = torch.stack((values, values.cumsum(-1)), -1)
        return SimpleNamespace(last_hidden_state=hidden)


def model(pooling="last", temperature=2.0):
    head = torch.nn.Linear(2, 255, bias=False)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 0] = 0.1
        head.weight[1, 1] = 0.1
    return DecisionModel(Backbone(), head, {"pooling": pooling, "temperature": temperature})


def test_native_head_uses_saved_temperature_once_and_masks_unavailable_options():
    inputs = {"input_ids": torch.tensor([[0, 1, 3], [1, 3, 2]]), "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])}
    actual = model()(inputs, [2, 1])
    expected = torch.tensor([0.3, 0.4]).div(2).softmax(-1)
    torch.testing.assert_close(actual[0, :2], expected)
    assert actual[0, 2:].count_nonzero() == 0
    assert actual[1, 0] == 1
    assert actual[1, 1:].count_nonzero() == 0


def test_mean_pooling_excludes_left_padding():
    inputs = {"input_ids": torch.tensor([[99, 1, 3]]), "attention_mask": torch.tensor([[0, 1, 1]])}
    expected = torch.tensor([0.2, 0.25]).div(2).softmax(-1)
    torch.testing.assert_close(model("mean")(inputs, [2])[0, :2], expected)


def test_nonfinite_logits_fail_before_answer():
    broken = model()
    with torch.no_grad():
        broken.readout.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="logits"):
        broken(
            {"input_ids": torch.ones(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2, dtype=torch.long)}, [2]
        )


class Processor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, chat, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return chat[1]["content"][0]["text"]

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        assert images is None
        assert padding is True
        assert truncation is False
        assert return_tensors == "pt"
        assert len(text) == 1
        ids = torch.tensor([[1 + ord(c) % 4 for c in text[0]]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def adapter(window=512, budget=512):
    result = PerplexityDecisionAdapter("unused", max_seq_length=window, max_forward_tokens=budget)
    result._model = model()
    result._processor = Processor()
    result._tokenizer = result._processor.tokenizer
    result._codes = ("A", "B")
    result._device = "cpu"
    return result


@pytest.mark.asyncio
async def test_normal_text_and_structured_states_share_worker_batch_with_exact_forward_bounds():
    instance = adapter()
    schema = {"choice": choice(), "noul": {"type": "noul"}}
    items = [Item(text="refund please"), Item(metadata={"state": {"message": "refund please"}})]
    costs = adapter_extract_item_costs(instance, items, output_schema=schema)
    batcher = BatchFormer(BatchConfig(max_batch_tokens=8192, max_batch_wait_ms=0))
    for prepared in build_extract_prepared_items(items, item_costs=costs):
        await batcher.submit(prepared, prepared.original_index)
    batch = batcher.try_get_batch()
    assert batch is not None
    assert batch.size == 2
    assert batcher.pending_count == 0
    forwarded = []
    original = instance._forward

    def observe(rows):
        forwarded.append(rows)
        return original(rows)

    instance._forward = observe
    result = instance.extract([items[i] for i in batch.metadata], output_schema=schema)
    assert result.errors == [None, None]
    assert any(len(rows) > 1 for rows in forwarded)
    assert all(len(rows) * max(row.length for row in rows) <= 512 for rows in forwarded)
    assert result.input_token_counts == [
        sum(row.length for rows in forwarded for row in rows if row.item == i) for i in range(2)
    ]
    assert result.input_token_counts != [costs[i] for i in batch.metadata]


def test_cost_estimates_account_for_repeated_questions_and_preserve_invalid_item_isolation():
    instance = adapter()
    schema = {"q": choice()}
    items = [Item(text="ok"), Item(metadata={"state": {"bad": float("nan")}}), Item(text="x" * 40_000)]
    costs = instance.extract_item_costs(items, output_schema=schema)
    assert costs[1:] == [32_768 * 4] * 2
    assert instance.extract_item_costs(items[:1], output_schema={"q": choice(), "r": choice()}) == [2 * costs[0]]
    assert instance.extract_item_costs(items[:1], output_schema={"q": choice(instructions="x" * 1000)})[0] > costs[0]
    result = instance.extract(items, output_schema=schema)
    assert result.errors[0] is None
    assert [error.code for error in result.errors[1:]] == ["INVALID_INPUT", "INPUT_TOO_LONG"]
    assert result.input_token_counts[1:] == [0, 0]


def test_extract_sorts_and_splits_complete_rows_preserving_item_question_order_and_usage():
    instance = adapter(budget=512)
    schema = {"later": choice(), "earlier": {"type": "noul", "instructions": "?"}}
    items = [Item(id="a", text="a"), Item(id="b", text="b" * 60), Item(id="c", metadata={"state": ["c"]})]
    forwarded = []
    original = instance._forward

    def observe(rows):
        forwarded.append([(row.item, row.question.id, row.length) for row in rows])
        return original(rows)

    instance._forward = observe
    output = instance.extract(items, output_schema=schema)
    assert output.errors == [None] * 3
    assert all(list(data) == ["later", "earlier"] for data in output.data)
    assert len(forwarded) > 1
    assert all(len(batch) <= 8 and len(batch) * max(row[2] for row in batch) <= 512 for batch in forwarded)
    assert output.input_token_counts == [
        sum(row[2] for batch in forwarded for row in batch if row[0] == i) for i in range(3)
    ]
    for index, item in enumerate(items):
        individual = instance.extract([item], output_schema=schema)
        assert output.data[index] == individual.data[0]


def test_an_oversized_later_question_runs_none_of_that_items_rows():
    instance = adapter(window=200)
    schema = {"short": choice(), "long": choice(instructions="x" * 120)}
    result = instance.extract([Item(text="x" * 100), Item(text="")], output_schema=schema)
    assert result.errors[0].code == "INPUT_TOO_LONG"
    assert result.input_token_counts[0] == 0
    assert result.data[0] == {}


@pytest.mark.parametrize(
    "options", [{"temperature": 1}, {"lora": "a"}, {"overflow_policy": "truncate"}, {"unknown": False}]
)
def test_unsupported_options_are_not_ignored(options):
    with pytest.raises(InvalidInputError):
        adapter().extract([Item(text="x")], output_schema={"q": choice()}, options=options)


@pytest.mark.parametrize("options", [False, 0, [], ""])
def test_falsey_non_object_options_are_rejected(options):
    with pytest.raises(InvalidInputError, match="object"):
        adapter().extract([Item(text="x")], output_schema={"q": choice()}, options=options)


def test_invalid_items_are_isolated_from_valid_states():
    instance = adapter()
    out = instance.extract(
        [Item(text="ok"), Item(text="text", metadata={"state": "ambiguous"}), Item(metadata={"state": 1})],
        output_schema={"q": choice()},
    )
    assert out.errors[0] is None
    assert [error.code for error in out.errors[1:]] == ["INVALID_INPUT", "INVALID_INPUT"]
    assert out.input_token_counts[0] > 0
    assert out.input_token_counts[1:] == [0, 0]


def test_unload_releases_all_owned_components():
    instance = adapter()
    instance.unload()
    assert instance._model is instance._processor is instance._tokenizer is instance._device is None


def checkpoint():
    return {
        "format_version": 1,
        "revision": "a" * 40,
        "codes": [str(i) for i in range(255)],
        "token_ids": list(range(255)),
        "temperature": 1.2,
        "pooling": "last",
        "attention_mode": "noncausal_full_attention",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"format_version": True},
        {"revision": "main"},
        {"codes": ["A"] * 255},
        {"token_ids": [0] * 255},
        {"temperature": False},
        {"temperature": float("nan")},
        {"temperature": 0},
        {"pooling": "first"},
        {"attention_mode": "flash"},
    ],
)
def test_invalid_checkpoint_semantics_are_rejected_before_weights(tmp_path, changes):
    (tmp_path / "decision_config.json").write_text(json.dumps(checkpoint() | changes))
    with pytest.raises(ValueError, match=r"checkpoint|Checkpoint"):
        checkpoint_config(tmp_path)
