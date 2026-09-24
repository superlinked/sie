"""Laya adapter: parity with the reference implementation, without model weights.

The fixtures under ``fixtures/laya/`` were recorded from the reference
implementation (the ``laya`` 0.3.11 package, CPU float32) at each model's pinned
revision. For every (state, question) row they hold the sha256 and length of the
exact ``input_ids``, the option-marker positions, and the raw decision-head
logits; per state, the reference's final answers and token usage; and every
string the reference tokenized, with its token ids.

That last map stands in for the real tokenizer here: it answers only for strings
the reference tokenized, so these tests fail if the adapter tokenizes anything
different, and they check sequence assembly, truncation, calibration, decoding,
labels mode, and metering against the reference with no network access.

``packages/sie_server/scripts/generate_laya_fixtures.py`` regenerates the
fixtures (outside CI, with the reference package installed); re-run it when a
model config's ``hf_revision`` changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sie_server.adapters.laya import questions as laya_questions
from sie_server.adapters.laya.adapter import DEFAULT_LABELS_INSTRUCTION, MAX_QUESTIONS, LayaAdapter
from sie_server.adapters.laya.model import apply_rope_parameters, split_checkpoint
from sie_server.adapters.laya.questions import Calibration, build_prefix, calibrated_probs, parse_questions
from sie_server.core.extract_cost import adapter_extract_item_costs, build_extract_prepared_items
from sie_server.types.inputs import InvalidInputError, Item

FIXTURES = Path(__file__).parent / "fixtures" / "laya"
VARIANTS = ("laya", "laya-multilingual", "laya-typed-decisions")
# Reference answers are rounded to 4 decimals.
ROUNDED_ATOL = 5.1e-5


def _golden(variant: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{variant}.json").read_text(encoding="utf-8"))


def _sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


class RecordedTokenizer:
    """Answers exactly the tokenizations the reference performed; anything else is a KeyError."""

    def __init__(self, golden: dict[str, Any]) -> None:
        spec = golden["tokenizer"]
        self.mask_token = spec["mask_token"]
        self.mask_token_id = spec["mask_token_id"]
        self.cls_token_id = spec["cls_token_id"]
        self.sep_token_id = spec["sep_token_id"]
        self.pad_token_id = spec["pad_token_id"]
        self._ids: dict[str, list[int]] = golden["tokenizations"]

    def __call__(self, text: str | list[str], add_special_tokens: bool = True) -> dict[str, Any]:
        assert add_special_tokens is False
        if isinstance(text, list):
            return {"input_ids": [list(self._ids[t]) for t in text]}
        return {"input_ids": list(self._ids[text])}


class GoldenLogitsModel:
    """Stands in for the network: returns the reference logits of each row, keyed by its exact ids."""

    use_flash = False

    def __init__(self, golden: dict[str, Any]) -> None:
        self.by_row: dict[str, list[float]] = {}
        for case in [*golden["cases"], *golden["override"]["cases"]]:
            for row in case["rows"]:
                self.by_row[row["sha256"]] = row["logits"]
        self.calls: list[int] = []

    def forward_padded(
        self, rows: list[list[int]], marker_pos: torch.Tensor, qtype: torch.Tensor, device: str, pad_token_id: int
    ) -> torch.Tensor:
        _ = (qtype, device, pad_token_id)
        self.calls.append(len(rows))
        out = torch.zeros(marker_pos.shape, dtype=torch.float32)
        for i, row in enumerate(rows):
            logits = self.by_row[_sha(row)]
            out[i, : len(logits)] = torch.tensor(logits)
        return out


def _adapter(golden: dict[str, Any], *, max_forward_tokens: int = 16384) -> LayaAdapter:
    adapter = LayaAdapter(golden["repo"], revision=golden["revision"], max_forward_tokens=max_forward_tokens)
    adapter._tokenizer = RecordedTokenizer(golden)  # ty: ignore[invalid-assignment]
    adapter._model = GoldenLogitsModel(golden)  # ty: ignore[invalid-assignment]
    adapter._device = "cpu"
    adapter._max_len = golden["max_len"]
    adapter._head_max_len = golden["head_max_len"]
    adapter._calibration = Calibration.from_config(golden["rl_agent_config"])
    return adapter


def _item(state: Any) -> Item:
    return Item(text=state) if isinstance(state, str) else Item(metadata={"state": state})


def _assert_answer(mine: dict[str, Any], ref: dict[str, Any]) -> None:
    assert mine["type"] == ref["type"]
    assert mine["confidence"] == pytest.approx(ref["confidence"], abs=ROUNDED_ATOL)
    if ref["type"] == "noul":
        assert mine["noul"] == pytest.approx(ref["noul"], abs=ROUNDED_ATOL)
        assert mine["answer"] is (mine["noul"] > 0.5)
        return
    assert mine["probabilities"].keys() == ref["probabilities"].keys()
    for key, value in ref["probabilities"].items():
        assert mine["probabilities"][key] == pytest.approx(value, abs=ROUNDED_ATOL)
    if ref["type"] == "choice":
        assert mine["choice"] == ref["choice"]
    else:
        assert mine["score"] == pytest.approx(ref["score"], abs=ROUNDED_ATOL)
        assert mine["legend"] == ref["legend"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_rows_match_reference(variant: str) -> None:
    """Every row's input_ids and marker positions equal the reference's, byte for byte."""
    golden = _golden(variant)
    adapter = _adapter(golden)
    for section, max_len, head_max_len in (
        (golden, golden["max_len"], golden["head_max_len"]),
        (golden["override"], golden["override"]["max_len"], golden["override"]["head_max_len"]),
    ):
        questions = parse_questions(section["questions"])
        states = [adapter._item_state(_item(case["state"])) for case in section["cases"]]
        prefixes, rows = adapter._build_rows(states, questions, max_len=max_len, head_max_len=head_max_len)
        expected = [row for case in section["cases"] for row in case["rows"]]
        assert len(rows) == len(expected)
        for n, (row, ref) in enumerate(zip(rows, expected, strict=True)):
            prefix = prefixes[n % len(prefixes)]
            assert prefix.question.qid == ref["qid"]
            assert (len(row), _sha(row)) == (ref["len"], ref["sha256"]), ref["qid"]
            assert prefix.markers == ref["markers"]
            assert laya_questions.QTYPES[prefix.question.qtype] == ref["qtype"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_rows_cover_truncation_paths(variant: str) -> None:
    """The fixtures exercise full-length rows, left truncation, and the option-budget squeeze."""
    golden = _golden(variant)
    by_name = {case["name"]: case for case in golden["cases"]}
    assert {row["len"] for row in by_name["long_truncated"]["rows"]} == {golden["max_len"]}
    assert {row["len"] for row in by_name["long_conversation_left_truncated"]["rows"]} == {golden["max_len"]}
    squeeze = next(row for row in by_name["plain_email"]["rows"] if row["qid"] == "budget_squeeze")
    assert len(squeeze["markers"]) == 30


@pytest.mark.parametrize("variant", VARIANTS)
def test_extract_matches_reference_answers(variant: str) -> None:
    """Typed answers and usage equal the reference's, through the full extract() path."""
    golden = _golden(variant)
    # A small token budget forces several forward chunks.
    adapter = _adapter(golden, max_forward_tokens=2 * golden["max_len"])
    items = [_item(case["state"]) for case in golden["cases"]]
    output = adapter.extract(items, output_schema=golden["questions"])

    assert len(cast("GoldenLogitsModel", adapter._model).calls) > 1
    assert output.entities == [[] for _ in items]
    assert output.classifications is None
    assert output.errors is None
    assert output.input_token_counts == [case["input_tokens"] for case in golden["cases"]]
    assert output.data is not None
    for data, case in zip(output.data, golden["cases"], strict=True):
        assert list(data) == list(golden["questions"])
        for qid, ref in case["answers"].items():
            _assert_answer(data[qid], ref)


@pytest.mark.parametrize("variant", VARIANTS)
def test_extract_token_budget_overrides_match_reference(variant: str) -> None:
    golden = _golden(variant)
    override = golden["override"]
    adapter = _adapter(golden)
    output = adapter.extract(
        [_item(case["state"]) for case in override["cases"]],
        output_schema=override["questions"],
        options={"max_len": override["max_len"], "head_max_len": override["head_max_len"]},
    )
    assert output.input_token_counts == [case["input_tokens"] for case in override["cases"]]
    assert output.data is not None
    for data, case in zip(output.data, override["cases"], strict=True):
        for qid, ref in case["answers"].items():
            _assert_answer(data[qid], ref)


@pytest.mark.parametrize("variant", VARIANTS)
def test_labels_mode_matches_reference_choice(variant: str) -> None:
    """labels=[...] is one choice question over the labels, returned as sorted classifications."""
    golden = _golden(variant)
    label_question = golden["questions"]["label"]
    adapter = _adapter(golden)
    items = [_item(case["state"]) for case in golden["cases"]]
    output = adapter.extract(items, labels=list(label_question["criteria"]))

    assert output.data is None
    assert output.classifications is not None
    for classifications, case in zip(output.classifications, golden["cases"], strict=True):
        ref = case["answers"]["label"]["probabilities"]
        assert [c["label"] for c in classifications] == sorted(ref, key=lambda k: -ref[k])
        for c in classifications:
            assert c["score"] == pytest.approx(ref[c["label"]], abs=ROUNDED_ATOL)
        assert sum(c["score"] for c in classifications) == pytest.approx(1.0, abs=1e-5)
    assert output.input_token_counts == [
        next(row["len"] for row in case["rows"] if row["qid"] == "label") for case in golden["cases"]
    ]


def test_labels_mode_echoes_caller_labels() -> None:
    """The model reads each label stripped; classifications carry the caller's label unchanged."""
    golden = _golden("laya")
    adapter = _adapter(golden)
    labels = list(golden["questions"]["label"]["criteria"])
    padded = [f" {label}\t" if n % 2 else f"{label} " for n, label in enumerate(labels)]
    case = golden["cases"][0]
    output = adapter.extract([_item(case["state"])], labels=padded)
    assert output.classifications is not None
    ref = case["answers"]["label"]["probabilities"]
    by_label = {c["label"]: c["score"] for c in output.classifications[0]}
    assert set(by_label) == set(padded)
    for label in padded:
        assert by_label[label] == pytest.approx(ref[label.strip()], abs=ROUNDED_ATOL)


def test_labels_mode_instruction_and_threshold() -> None:
    golden = _golden("laya")
    adapter = _adapter(golden)
    (question,), caller_labels = adapter._resolve_questions([" a ", "b"], None, "Which team?")
    assert caller_labels == [" a ", "b"]
    assert (question.qtype, question.instructions, list(question.criteria)) == ("choice", "Which team?", ["a", "b"])
    label_question = golden["questions"]["label"]
    case = golden["cases"][0]
    output = adapter.extract(
        [_item(case["state"])],
        labels=list(label_question["criteria"]),
        instruction=label_question["instructions"],
        options={"threshold": 0.2},
    )
    assert output.classifications is not None
    ref = case["answers"]["label"]["probabilities"]
    assert [c["label"] for c in output.classifications[0]] == [k for k, v in ref.items() if v >= 0.2]


@pytest.mark.parametrize("variant", VARIANTS)
def test_calibration_matches_reference(variant: str) -> None:
    """Temperatures are the shipped buckets, clamped to [0.5, 5] like laya 0.3.11."""
    golden = _golden(variant)
    calibration = Calibration.from_config(golden["rl_agent_config"])
    assert list(calibration.temperature) == pytest.approx(golden["temperature_applied"])
    assert calibration.temperature_by_options == pytest.approx(golden["temperature_by_options_applied"])
    if variant == "laya-multilingual":
        assert calibration.temperature == (1.0, 1.0, 1.0)
        assert calibration.temperature_by_options == {}
    else:
        # The shipped 12-option bucket sharpens by 10x; it is applied clamped.
        assert golden["rl_agent_config"]["temperature_by_options"]["choice:11+"] < 0.5
        assert calibration.temperature_for("choice", 12) == 0.5


def test_calibration_rejects_non_finite_and_malformed() -> None:
    calibration = Calibration.from_config(
        {"temperature": [float("nan"), "x", 9.0], "temperature_by_options": {"noul:2": float("inf")}}
    )
    assert calibration.temperature == (1.0, 1.0, 5.0)
    assert calibration.temperature_by_options == {"noul:2": 1.0}
    assert Calibration.from_config({"temperature": [2.0]}).temperature == (1.0, 1.0, 1.0)
    probs = calibrated_probs(np.array([1.0, 2.0, 3.0], dtype=np.float32), "choice", calibration)
    assert probs.dtype == np.float32
    assert float(probs.sum()) == pytest.approx(1.0)


def test_options_overflow_error_matches_reference() -> None:
    golden = _golden("laya")
    tokenizer = RecordedTokenizer(golden)
    (question,) = parse_questions(
        {"too_many": {"type": "choice", "instructions": "pick", "criteria": [f"option {i}" for i in range(80)]}}
    )
    with pytest.raises(InvalidInputError, match="options exceed head_max_len") as excinfo:
        build_prefix(tokenizer, question, max_len=96, head_max_len=64)
    assert str(excinfo.value) == golden["overflow_error"]


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ({"q": "not a dict"}, "question 'q': definition must be a dict, got str"),
        ({"q": {"type": "rank", "instructions": "x"}}, "question 'q': unknown type 'rank'"),
        ({"q": {"type": "noul"}}, "question 'q': no 'instructions'"),
        ({"q": {"type": "choice", "instructions": "x"}}, "a choice question takes 'criteria' as a dict"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": []}}, "needs at least one criterion"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": [1, 2]}}, "list criteria must be strings"),
        ({"q": {"type": "score", "instructions": "x", "criteria": {"a": 1}}}, "a score question takes 'criteria'"),
        ({"q": {"type": "score", "instructions": "x", "criteria": []}}, "needs at least one level"),
        ({"q": {"type": "noul", "instructions": "x", "criteria": ["yes"]}}, "a noul question takes 'criteria'"),
        ({"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}}, "keyed only 'true'/'false'"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": ["a"], "labels": {}}}, "only supported for noul"),
        ({"q": {"type": "noul", "instructions": "x", "labels": {"false": "a", "true": "a"}}}, "distinct non-empty"),
        ({"q": {"type": "noul", "instructions": {"raw": b"\x00"}}}, "'instructions' must be a string or JSON"),
    ],
)
def test_malformed_questions_are_rejected(questions: dict[str, Any], message: str) -> None:
    adapter = _adapter(_golden("laya"))
    with pytest.raises(InvalidInputError, match=message.replace("(", r"\(").replace(")", r"\)")):
        adapter.extract([Item(text="x")], output_schema=questions)


def test_request_shape_errors() -> None:
    """Caller mistakes raise InvalidInputError, which both HTTP and the queue path report as INVALID_INPUT."""
    adapter = _adapter(_golden("laya"))
    item = [Item(text="x")]
    with pytest.raises(InvalidInputError, match="output_schema"):
        adapter.extract(item)
    with pytest.raises(InvalidInputError, match="not both"):
        adapter.extract(item, labels=["a"], output_schema={"q": {"type": "noul", "instructions": "x"}})
    with pytest.raises(InvalidInputError, match="applies only to labels"):
        adapter.extract(item, instruction="x", output_schema={"q": {"type": "noul", "instructions": "x"}})
    with pytest.raises(InvalidInputError, match="unique"):
        adapter.extract(item, labels=["a", " a "])
    with pytest.raises(InvalidInputError, match="non-empty strings"):
        adapter.extract(item, labels=["a", " "])
    with pytest.raises(InvalidInputError, match="must map question ids"):
        adapter.extract(item, output_schema=["q"])  # ty: ignore[invalid-argument-type]
    with pytest.raises(InvalidInputError, match=f"at most {MAX_QUESTIONS} questions"):
        adapter.extract(
            item, output_schema={f"q{i}": {"type": "noul", "instructions": "x"} for i in range(MAX_QUESTIONS + 1)}
        )
    one = {"q": {"type": "noul", "instructions": "x"}}
    with pytest.raises(InvalidInputError, match="max_len must be at most 512"):
        adapter.extract(item, output_schema=one, options={"max_len": 513})
    with pytest.raises(InvalidInputError, match="positive integer"):
        adapter.extract(item, output_schema=one, options={"head_max_len": "64"})
    with pytest.raises(InvalidInputError, match=r"head_max_len must be at most max_len \(96\)"):
        adapter.extract(item, output_schema=one, options={"max_len": 96, "head_max_len": 97})
    with pytest.raises(InvalidInputError, match="threshold"):
        adapter.extract(item, labels=["a"], options={"threshold": 2})


def test_empty_question_mapping_is_rejected() -> None:
    adapter = _adapter(_golden("laya"))
    with pytest.raises(InvalidInputError, match="requires typed questions"):
        adapter.extract([Item(text="x"), Item(text="y")], output_schema={})
    assert cast("GoldenLogitsModel", adapter._model).calls == []


def test_default_head_budget_is_kept_under_a_smaller_max_len() -> None:
    """Only a head_max_len the caller sets is bounded by max_len; the checkpoint default is used as is."""
    adapter = _adapter(_golden("laya"))
    assert adapter._token_budget({"max_len": 96}) == (96, adapter._head_max_len)
    assert adapter._token_budget({"max_len": 96, "head_max_len": 96}) == (96, 96)


def test_invalid_states_fail_only_their_item() -> None:
    golden = _golden("laya")
    adapter = _adapter(golden)
    case = golden["cases"][0]
    items = [
        Item(),
        _item(case["state"]),
        Item(text="x", metadata={"state": "y"}),
        Item(metadata={"state": 42}),
        Item(metadata={"state": {"blob": b"\x00"}}),  # msgpack bytes cannot be serialized
    ]
    output = adapter.extract(items, output_schema=golden["questions"])
    assert output.errors is not None
    assert [e is None for e in output.errors] == [False, True, False, False, False]
    assert {e.code for e in output.errors if e is not None} == {"INVALID_INPUT"}
    assert output.errors[4] is not None
    assert "JSON-serializable" in output.errors[4].message
    assert output.data is not None
    assert output.data[0] == output.data[2] == output.data[3] == output.data[4] == {}
    for qid, ref in case["answers"].items():
        _assert_answer(output.data[1][qid], ref)
    assert output.input_token_counts == [0, case["input_tokens"], 0, 0, 0]


def test_item_states_serialize_like_the_reference() -> None:
    state = {"b": "é", "a": [1, 2]}
    assert LayaAdapter._item_state(Item(metadata={"state": state})) == (json.dumps(state, ensure_ascii=False), False)
    turns = [{"role": "user", "content": "hi"}]
    assert LayaAdapter._item_state(Item(metadata={"state": turns})) == (json.dumps(turns), True)
    assert LayaAdapter._item_state(Item(metadata={"state": "plain"})) == ("plain", False)
    assert LayaAdapter._item_state(Item(text="plain", metadata={"other": 1})) == ("plain", False)


def test_extract_item_costs_scale_with_questions() -> None:
    """Each question re-encodes the state, so an item costs its state once per question plus the questions."""
    adapter = LayaAdapter("convaiinnovations/laya")
    schema = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(4)}
    ticket = {"subject": "Login broken", "body": "Since the update I can't log in; error 500.", "prior_tickets": 3}
    items = [Item(text="a" * 100), Item(metadata={"state": ticket}), Item()]
    costs = adapter.extract_item_costs(items, output_schema=schema)
    assert costs is not None
    question_chars = costs[2]  # an item with no state costs only the question text
    assert costs[0] == 4 * 100 + question_chars
    state_chars = (costs[1] - question_chars) / 4
    assert state_chars == pytest.approx(len(json.dumps(ticket)), rel=0.25)
    assert question_chars == pytest.approx(len(json.dumps(schema)), rel=0.25)
    label_costs = adapter.extract_item_costs([Item(text="abc")], labels=["x", "yy"])
    assert label_costs == [3 + 3 + len(DEFAULT_LABELS_INSTRUCTION)]

    prepared = build_extract_prepared_items(
        items, item_costs=adapter_extract_item_costs(adapter, items, output_schema=schema)
    )
    assert [p.cost for p in prepared] == costs
    # Adapters without the hook (or with a malformed answer) keep the character count.
    assert [p.cost for p in build_extract_prepared_items(items, item_costs=None)] == [100, 0, 0]
    assert [p.cost for p in build_extract_prepared_items(items, item_costs=[1, -2, 3])] == [100, 0, 0]
    assert adapter_extract_item_costs(MagicMock(), items) is None


def test_extract_item_costs_are_bounded_and_never_serialize() -> None:
    """The estimate runs on the event loop: it walks at most one row's worth of a state and never serializes it."""
    adapter = LayaAdapter("convaiinnovations/laya")
    row_chars = 4 * adapter._max_len
    turns = [{"role": "user", "content": "x" * 50} for _ in range(100_000)]
    items = [
        Item(metadata={"state": turns}),
        Item(metadata={"state": {"history": turns}}),
        Item(metadata={"state": "y" * 1_000_000}),
        Item(text="z" * 1_000_000),
    ]
    schema = {"q": {"type": "noul", "instructions": "x"}}
    with patch("json.dumps", side_effect=AssertionError("serialized")):
        costs = adapter.extract_item_costs(items, output_schema=schema)
        question_chars = adapter.extract_item_costs([Item()], output_schema=schema)
    assert costs is not None
    assert question_chars is not None
    assert costs == [row_chars + question_chars[0]] * len(items)


def test_rope_parameters_override_transformers4_defaults() -> None:
    """The mmBERT encoder declares 160000 for both layer kinds; transformers 4.x defaults sliding layers to 10000."""
    config = MagicMock(spec=["rope_parameters", "global_rope_theta", "local_rope_theta"])
    config.rope_parameters = {
        "full_attention": {"rope_theta": 160000, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 160000, "rope_type": "default"},
    }
    config.global_rope_theta, config.local_rope_theta = 160000.0, 10000.0
    apply_rope_parameters(config)
    assert (config.global_rope_theta, config.local_rope_theta) == (160000.0, 160000.0)


def test_split_checkpoint_drops_unused_heads() -> None:
    tensor = torch.zeros(1)
    encoder, head = split_checkpoint(
        {
            "encoder.layers.0.attn.Wqkv.weight": tensor,
            "head.layers.0.linear1.weight": tensor,
            "type_emb.weight": tensor,
            "scorer.0.weight": tensor,
            "act_head.0.weight": tensor,
            "temperature": tensor,
        }
    )
    assert list(encoder) == ["layers.0.attn.Wqkv.weight"]
    assert list(head) == ["layers.0.linear1.weight", "type_emb.weight", "scorer.0.weight"]
    with pytest.raises(ValueError, match="Unexpected tensor") as excinfo:
        split_checkpoint({"encoder.x": tensor, "head.y": tensor, "classifier.weight": tensor})
    # A bad checkpoint is a server fault, not a caller error.
    assert not isinstance(excinfo.value, InvalidInputError)


def test_checkpoint_download_is_pinned_and_filtered(tmp_path: Path) -> None:
    adapter = LayaAdapter("convaiinnovations/laya", revision="0123abc")
    with patch("sie_server.adapters.laya.adapter.snapshot_download", return_value=str(tmp_path)) as download:
        assert adapter._checkpoint_dir() == tmp_path
    kwargs = download.call_args.kwargs
    assert kwargs["repo_id"] == "convaiinnovations/laya"
    assert kwargs["revision"] == "0123abc"
    assert set(kwargs["allow_patterns"]) == {"rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"}
    local = LayaAdapter(str(tmp_path))
    with patch("sie_server.adapters.laya.adapter.snapshot_download") as download:
        assert local._checkpoint_dir() == tmp_path
    download.assert_not_called()


def test_tokenizer_list_special_tokens_are_passed_as_mapping(tmp_path: Path) -> None:
    """A list-valued extra_special_tokens (unreadable on transformers 4.x) is passed as a mapping, not rewritten."""
    config_file = tmp_path / "tokenizer_config.json"
    config_file.write_text(json.dumps({"extra_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}))
    before = config_file.read_text()
    with patch("transformers.PreTrainedTokenizerFast.from_pretrained") as from_pretrained:
        LayaAdapter._load_tokenizer(tmp_path)
    assert from_pretrained.call_args.kwargs == {
        "extra_special_tokens": {"extra_0": "<start_of_turn>", "extra_1": "<end_of_turn>"}
    }
    assert config_file.read_text() == before
