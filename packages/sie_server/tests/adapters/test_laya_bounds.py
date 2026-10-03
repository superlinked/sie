"""Laya adapter: request bounds, and long inputs tokenized only as far as a row can read.

Count limits fail before any tokenizer call. Long states, instructions, and
options are cut to a budget of UTF-8 bytes per token a row can use before
tokenizing, and structured states are rendered only that far; these tests check
the cuts leave every row as the whole texts would produce it, in any script.
Error messages quote caller data only in bounded form.
"""

from __future__ import annotations

import json
import math
import random
import re
import zlib
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters.laya import adapter as laya_adapter
from sie_server.adapters.laya import questions as laya_questions
from sie_server.adapters.laya.adapter import MAX_ITEM_TOKENS, MAX_OPTIONS, MAX_QUESTIONS, LayaAdapter
from sie_server.adapters.laya.questions import (
    STATE_BYTES_PER_TOKEN,
    clip_text,
    parse_questions,
    render_criterion,
    serialize_state,
)
from sie_server.types.inputs import InvalidInputError, Item

MAX_LEN = 512
HEAD_MAX_LEN = 192


class WordTokenizer:
    """Deterministic stand-in tokenizer: an ASCII word, a whitespace run, or any other character is one token."""

    mask_token = "[MASK]"  # noqa: S105 — tokenizer attribute, not a secret
    mask_token_id = 4
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def __call__(self, text: str | list[str], add_special_tokens: bool = True) -> dict[str, Any]:
        assert add_special_tokens is False
        texts = text if isinstance(text, list) else [text]
        self.inputs.extend(texts)
        pieces = (re.findall(r"\s+|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", s) for s in texts)
        ids = [[5 + zlib.crc32(t.encode("utf-8", "surrogatepass")) % 50_000 for t in piece] for piece in pieces]
        return {"input_ids": ids if isinstance(text, list) else ids[0]}


class NoCallTokenizer(WordTokenizer):
    def __call__(self, text: str | list[str], add_special_tokens: bool = True) -> dict[str, Any]:
        raise AssertionError("tokenized before the request was validated")


def _adapter(tokenizer: WordTokenizer, *, max_len: int = MAX_LEN) -> LayaAdapter:
    adapter = LayaAdapter("convaiinnovations/laya")
    adapter._tokenizer = tokenizer  # ty: ignore[invalid-assignment]
    adapter._model = MagicMock()  # ty: ignore[invalid-assignment]
    adapter._device = "cpu"
    adapter._max_len = max_len
    adapter._head_max_len = HEAD_MAX_LEN
    return adapter


# ---------------------------------------------------------------------------- counts


@pytest.mark.parametrize(
    ("request_kwargs", "message"),
    [
        ({"labels": [f"l{i}" for i in range(MAX_OPTIONS + 1)]}, f"at most {MAX_OPTIONS} labels"),
        (
            {"output_schema": {"q": {"type": "choice", "instructions": "x", "criteria": list(range(100_000))}}},
            f"at most {MAX_OPTIONS} answer options",
        ),
        (
            {
                "output_schema": {
                    f"q{i}": {"type": "choice", "instructions": "x", "criteria": [f"o{j}" for j in range(40)]}
                    for i in range(MAX_OPTIONS // 40 + 1)
                }
            },
            f"at most {MAX_OPTIONS} answer options",
        ),
        (
            {"output_schema": {"q": {"type": "choice", "instructions": "x", "criteria": [f"o{j}" for j in range(95)]}}},
            "question 'q' options exceed head_max_len=40",
        ),
        (
            {"output_schema": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(MAX_QUESTIONS + 1)}},
            f"at most {MAX_QUESTIONS} questions",
        ),
    ],
)
def test_oversized_requests_fail_before_tokenizing(request_kwargs: dict[str, Any], message: str) -> None:
    adapter = _adapter(NoCallTokenizer())
    options = {"max_len": 96, "head_max_len": 40}  # 95 options cannot all be marked in 96 positions
    with pytest.raises(InvalidInputError, match=re.escape(message)):
        adapter.extract([Item(text="x")], options=options, **request_kwargs)
    assert adapter._model.mock_calls == []  # ty: ignore[unresolved-attribute]


def test_item_token_cap_limits_questions_by_max_len() -> None:
    """Questions x max_len stay within MAX_ITEM_TOKENS: 32 questions at max_len 1024, all 64 at 512."""
    adapter = _adapter(NoCallTokenizer(), max_len=1024)
    schema = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(MAX_ITEM_TOKENS // 1024 + 1)}
    with pytest.raises(InvalidInputError, match="send at most 32 questions, or a smaller max_len option"):
        adapter.extract([Item(text="x")], output_schema=schema)
    LayaAdapter._check_item_tokens(MAX_ITEM_TOKENS // 1024, 1024)
    LayaAdapter._check_item_tokens(MAX_QUESTIONS, 512)
    # A smaller max_len option admits more questions on a 1024-token model.
    with pytest.raises(AssertionError, match="tokenized before"):
        adapter.extract([Item(text="x")], output_schema=schema, options={"max_len": 512})


def test_count_check_is_exact_at_the_marker_limit() -> None:
    """max_len - 2 options can still be marked (one [MASK] each after [CLS] [SEP]); one more cannot."""
    tokenizer = WordTokenizer()
    adapter = _adapter(tokenizer)
    for n, fits in ((94, True), (95, False)):
        question = parse_questions(
            {"q": {"type": "choice", "instructions": "", "criteria": {f"{j}": None for j in range(n)}}}
        )[0]
        if fits:
            laya_questions.check_options_fit(question, max_len=96, head_max_len=40)
        else:
            with pytest.raises(InvalidInputError, match="options exceed head_max_len=40"):
                adapter._build_rows([("x", False)], [question], max_len=96, head_max_len=40)
    assert tokenizer.inputs == []


# ---------------------------------------------------------------------------- long inputs

LONG_TEXT = " ".join(f"word{i % 997} and some more text, number {i}." for i in range(8_000))  # ~330k chars
LONG_CJK = "二重に請求されました。返金してください。注文番号は一二三四です。" * 5_000  # 3 bytes per character
LONG_ASTRAL = "𠀀𠀁😀🎉" * 50_000  # 4 bytes per character


def _long_inputs() -> tuple[list[Item], dict[str, Any]]:
    items = [
        Item(text=LONG_TEXT),
        Item(metadata={"state": {"subject": "a long ticket", "body": LONG_TEXT, "tags": list(range(5_000))}}),
        Item(metadata={"state": [{"role": "user", "content": f"turn {i}: {LONG_TEXT[:200]}"} for i in range(3_000)]}),
        Item(metadata={"state": LONG_TEXT[:500]}),  # short: within bounds
        Item(text=LONG_CJK),
        Item(metadata={"state": [{"role": "user", "content": LONG_ASTRAL}]}),
    ]
    schema = {
        "long_instructions": {"type": "noul", "instructions": LONG_TEXT},
        "long_options": {
            "type": "choice",
            "instructions": "Which fits?",
            "criteria": {"a": LONG_TEXT, "b": {"detail": LONG_TEXT, "n": 1}, "c": None},
        },
        "score": {"type": "score", "instructions": "How much?", "criteria": ["low", {"desc": LONG_TEXT}, "high"]},
        "many": {"type": "choice", "instructions": "Pick", "criteria": [f"option {j}" for j in range(40)]},
        "astral": {"type": "choice", "instructions": LONG_ASTRAL, "criteria": {"喜": LONG_CJK, "😀": LONG_ASTRAL}},
    }
    return items, schema


def _rows(adapter: LayaAdapter, items: list[Item], schema: dict[str, Any]) -> tuple[list[Any], list[list[int]]]:
    limit = adapter._max_len * laya_adapter.STATE_BYTES_PER_TOKEN
    states = [adapter._item_state(item, limit) for item in items]
    prefixes, rows = adapter._build_rows(states, parse_questions(schema), max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
    return [(p.ids, p.markers) for p in prefixes], rows


def test_long_inputs_are_cut_before_tokenizing() -> None:
    """extract() renders each state only as far as a row reads, and tokenizes only cut texts."""
    tokenizer = WordTokenizer()
    adapter = _adapter(tokenizer)
    adapter._run_rows = lambda rows, prefixes: [np.zeros(len(p.markers), dtype=np.float32) for p in prefixes]  # ty: ignore[invalid-assignment]
    items, schema = _long_inputs()
    state_bytes = MAX_LEN * STATE_BYTES_PER_TOKEN
    with patch.object(laya_adapter, "serialize_state", wraps=laya_adapter.serialize_state) as render:
        output = adapter.extract(items, output_schema=schema)
    assert output.errors is None
    assert [call.args[1] for call in render.call_args_list] == [state_bytes] * 4
    sizes = [len(t.encode()) for t in tokenizer.inputs]
    assert max(sizes) <= state_bytes
    assert sum(sizes) < 12 * state_bytes


def test_row_building_cuts_whole_texts() -> None:
    """_build_rows cuts states it is given whole (as the parity tests pass them)."""
    tokenizer = WordTokenizer()
    adapter = _adapter(tokenizer)
    items, schema = _long_inputs()
    states = [adapter._item_state(item) for item in items]
    adapter._build_rows(states, parse_questions(schema), max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
    assert max(len(t.encode()) for t in tokenizer.inputs) <= MAX_LEN * STATE_BYTES_PER_TOKEN


def test_cut_inputs_keep_the_rows_of_whole_inputs() -> None:
    """Every prefix and row equals the one built from the whole, uncut texts."""
    items, schema = _long_inputs()
    cut = _rows(_adapter(WordTokenizer()), items, schema)
    with (
        patch.object(laya_adapter, "STATE_BYTES_PER_TOKEN", 10**9),
        patch.object(laya_questions, "QUESTION_BYTES_PER_TOKEN", 10**9),
    ):
        whole_tokenizer = WordTokenizer()
        whole = _rows(_adapter(whole_tokenizer), items, schema)
    assert max(len(t) for t in whole_tokenizer.inputs) > len(LONG_TEXT)  # the comparison did tokenize whole texts
    assert cut == whole
    # The conversation keeps its newest turn; the other states their beginning.
    _, rows = cut
    assert all(len(row) == MAX_LEN for row in rows[: 3 * len(schema)])
    assert all(len(row) == MAX_LEN for row in rows[4 * len(schema) :])


# ---------------------------------------------------------------------------- bounded JSON


def _random_json(rng: random.Random, depth: int = 0) -> Any:
    kind = rng.randrange(9 if depth < 4 else 5)
    if kind == 0:
        return rng.choice(["", "plain", 'quote " and \\ back', "tab\tnew\nline\x01", "é ü 二重 😀", "x" * 50])
    if kind == 1:
        return rng.choice([0, -7, 2**70, 10**300])
    if kind == 2:
        return rng.choice([0.5, -1e-300, 1e300, math.inf, -math.inf, math.nan, 3.141592653589793])
    if kind == 3:
        return rng.choice([True, False, None])
    if kind == 4:
        return "y" * rng.randrange(0, 200)
    if kind in (5, 6):
        keys = ["k", "", "ключ", 1, 2.5, True, None, math.nan, 'q"k']
        return {rng.choice(keys) if rng.random() < 0.3 else f"key{i}": _random_json(rng, depth + 1) for i in range(4)}
    return [_random_json(rng, depth + 1) for _ in range(rng.randrange(0, 5))]


def _utf8_head(text: str, limit: int) -> str:
    return text.encode()[:limit].decode("utf-8", "ignore")


def _utf8_tail(text: str, limit: int) -> str:
    return text.encode()[-limit:].decode("utf-8", "ignore")


def test_bounded_rendering_matches_json_dumps() -> None:
    rng = random.Random(0)  # noqa: S311 — deterministic test data
    for _ in range(300):
        state = {"root": _random_json(rng)} if rng.random() < 0.5 else [_random_json(rng) for _ in range(3)]
        full = json.dumps(state, ensure_ascii=False)
        criterion_full = json.dumps(state, ensure_ascii=False, separators=(", ", ": "), default=str)
        size = len(full.encode())
        for limit in sorted({1, 2, 3, 7, 64, size // 2 or 1, size - 1 or 1, size, size + 3}):
            assert serialize_state(state, limit) == _utf8_head(full, limit)
            assert serialize_state(state, limit, from_end=True) == _utf8_tail(full, limit)
            assert render_criterion(state, limit) == _utf8_head(criterion_full, limit)


def test_clip_text_cuts_utf8_bytes_at_character_boundaries() -> None:
    text = "a€😀b" * 1000  # 1, 3, 4, and 1 bytes per character
    for limit in range(1, 40):
        head, tail = clip_text(text, limit), clip_text(text, limit, from_end=True)
        assert head == _utf8_head(text, limit)
        assert tail == _utf8_tail(text, limit)
        assert len(head.encode()) <= limit >= len(tail.encode())
    assert clip_text(text, 10**6) is text
    assert clip_text("x" * 100, 40) == "x" * 40
    lone = "a\ud800b" * 10  # a lone surrogate (possible from JSON escapes) survives a cut
    assert clip_text(lone, 12) == "a\ud800b" * 2 + "a"


def test_bounded_rendering_encodes_like_json_dumps_for_unusual_values() -> None:
    value = {"blob": b"\x00", "set": {1}, "nested": [{"t": (1, 2)}]}
    assert render_criterion(value, 1000) == json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
    for bad in ({"blob": b"\x00"}, {(1, 2): "tuple key"}, [{b"k": 1}]):
        with pytest.raises(TypeError):
            serialize_state(bad, 1000)
        with pytest.raises(TypeError):
            json.dumps(bad)


def test_bounded_rendering_stops_early() -> None:
    """A huge state renders in bounded work: the walk stops once the limit is reached."""
    huge = {"items": [{"n": i, "text": "abc"} for i in range(200_000)], "tail": "z" * 5_000_000}
    with patch.object(laya_questions, "_json_string", wraps=laya_questions._json_string) as encode:
        assert serialize_state(huge, 100) == json.dumps(huge)[:100]
        assert serialize_state(huge, 100, from_end=True) == json.dumps(huge)[-100:]
    assert encode.call_count < 60


# ---------------------------------------------------------------------------- caller data errors


def test_unrenderable_states_are_invalid_input() -> None:
    adapter = _adapter(WordTokenizer())
    for state in ({(1, 2): "tuple key"}, {"blob": b"\x00"}, [{b"k": "bytes key"}]):
        with pytest.raises(InvalidInputError, match="JSON-serializable"):
            adapter._item_state(Item(metadata={"state": state}), MAX_LEN * STATE_BYTES_PER_TOKEN)


@pytest.mark.parametrize(
    ("criteria", "message"),
    [
        ({"a": {b"key": 1}}, "criteria must be JSON-serializable"),
        ({"a": {(1, 2): 1}}, "criteria must be JSON-serializable"),
        ({1: "label from msgpack"}, "criteria labels must be strings"),
    ],
)
def test_unrenderable_criteria_are_invalid_input(criteria: dict[Any, Any], message: str) -> None:
    adapter = _adapter(WordTokenizer())
    with pytest.raises(InvalidInputError, match=message):
        questions = parse_questions({"q": {"type": "choice", "instructions": "x", "criteria": criteria}})
        adapter._build_rows([("x", False)], questions, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)


@pytest.mark.parametrize(
    "criteria",
    [
        {f"k{i}": i for i in range(300_000)},
        {"true": "x", "false": "y", "other": "z"},
        {"t" * 5_000_000: "a key far too long to be 'true'"},
    ],
)
def test_noul_criteria_keys_are_checked_without_reading_every_key(criteria: dict[str, Any]) -> None:
    """More than two keys fail on the count alone; long keys are never lowercased or echoed whole."""
    adapter = _adapter(NoCallTokenizer())
    with (
        patch.object(laya_questions, "_noul_key", wraps=laya_questions._noul_key) as read_key,
        pytest.raises(InvalidInputError, match="keyed only 'true'/'false'") as excinfo,
    ):
        adapter.extract(
            [Item(text="x")], output_schema={"q": {"type": "noul", "instructions": "x", "criteria": criteria}}
        )
    assert read_key.call_count == (0 if len(criteria) > 2 else len(criteria))
    assert len(str(excinfo.value)) < 700
    if len(criteria) > 3:
        assert f"({len(criteria)} keys)" in str(excinfo.value)


def test_noul_labels_mapping_is_checked_by_size_first() -> None:
    adapter = _adapter(NoCallTokenizer())
    labels = {f"k{i}": "x" for i in range(1_000_000)}
    with pytest.raises(InvalidInputError, match="noul labels must map exactly"):
        adapter.extract([Item(text="x")], output_schema={"q": {"type": "noul", "instructions": "x", "labels": labels}})


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"q" * 1_000_000: {"type": "rank", "instructions": "x"}}, "unknown type 'rank'"),
        ({"q": {"type": "r" * 1_000_000, "instructions": "x"}}, "unknown type 'rrrr"),
        ({"q": {"type": ["choice"] * 1_000, "instructions": "x"}}, "unknown type <list>"),
        ({"q" * 1_000_000: {"type": "noul"}}, "no 'instructions'"),
        (
            {"q" * 1_000_000: {"type": "choice", "instructions": "x", "criteria": [f"o{j}" for j in range(95)]}},
            "options exceed head_max_len=40",
        ),
    ],
)
def test_error_messages_quote_caller_data_briefly(schema: dict[str, Any], message: str) -> None:
    adapter = _adapter(NoCallTokenizer())
    with pytest.raises(InvalidInputError, match=re.escape(message)) as excinfo:
        adapter.extract([Item(text="x")], output_schema=schema, options={"max_len": 96, "head_max_len": 40})
    assert len(str(excinfo.value)) < 400


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_len": 10**5000}, "max_len must be at most 512 for this model, got <int>"),
        ({"max_len": 96, "head_max_len": 10**5000}, "head_max_len must be at most max_len (96), got <int>"),
        ({"threshold": 10**400}, "threshold must be a finite number"),
    ],
)
def test_oversized_numbers_are_invalid_input(options: dict[str, Any], message: str) -> None:
    adapter = _adapter(NoCallTokenizer())
    kwargs: dict[str, Any] = {"labels": ["a", "b"]} if "threshold" in options else {"output_schema": ONE_QUESTION}
    with pytest.raises(InvalidInputError, match=re.escape(message)):
        adapter.extract([Item(text="x")], options=options, **kwargs)


ONE_QUESTION = {"q": {"type": "noul", "instructions": "x"}}


# ---------------------------------------------------------------------------- cost hook


def test_cost_hook_is_bounded_for_oversized_requests() -> None:
    adapter = LayaAdapter("convaiinnovations/laya")
    too_many = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(MAX_QUESTIONS + 1)}
    assert adapter.extract_item_costs([Item(text="abc")], output_schema=too_many) is None
    row_chars = 4 * adapter._max_len
    costs = adapter.extract_item_costs([Item(text="abc")], labels=["x" * 10] * 1_000_000)
    assert costs == [3 + row_chars]
