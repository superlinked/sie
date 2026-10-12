"""The SGLang child's thinking-budget logit processor."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sie_server.adapters.sglang._compat import sie_thinking_budget as budget_module
from sie_server.adapters.sglang._compat.sie_thinking_budget import (
    PARAM,
    PROCESSOR_REFERENCE,
    ThinkingBudgetLogitProcessor,
    forced_token,
)

OPEN, CLOSE, NEWLINE = 100, 101, 107
SPEC = {"budget": 4, "open": OPEN, "close": CLOSE, "newline": NEWLINE, "open_in_prompt": False}
COMPAT_DIR = Path(budget_module.__file__).parent


def _run(spec: dict, steps: list[list[int]]) -> list[int | None]:
    state: dict = {}
    return [forced_token(spec, state, output) for output in steps]


def test_nothing_is_forced_before_the_block_opens_or_inside_the_budget() -> None:
    assert _run(SPEC, [[], [5], [5, 6]]) == [None, None, None]
    assert _run(SPEC, [[OPEN], [OPEN, 1], [OPEN, 1, 2, 3]]) == [None, None, None]


def test_the_budget_forces_a_newline_then_the_closing_token() -> None:
    assert _run(SPEC, [[OPEN, 1, 2, 3, 4]]) == [NEWLINE]
    assert _run(SPEC, [[OPEN, 1, 2, 3, NEWLINE]]) == [CLOSE]
    assert _run({**SPEC, "newline": None}, [[OPEN, 1, 2, 3, 4]]) == [CLOSE]


def test_the_answer_after_the_closing_token_is_left_to_the_grammar() -> None:
    # Once the block closes, the processor never touches the logits again, so a
    # reasoning-aware grammar constrains the answer as it does without a budget.
    steps = [[OPEN, 1, 2, 3, NEWLINE], [OPEN, 1, 2, 3, NEWLINE, CLOSE]]
    steps += [[*steps[-1], *range(200, 200 + n)] for n in (1, 10, 50)]
    assert _run(SPEC, steps) == [CLOSE, None, None, None, None]


def test_a_block_the_model_closes_on_its_own_is_left_alone() -> None:
    assert _run(SPEC, [[OPEN, 1, CLOSE], [OPEN, 1, CLOSE, 9, 9, 9, 9, 9, 9]]) == [None, None]


def test_a_block_opened_by_the_prompt_counts_from_the_first_generated_token() -> None:
    spec = {**SPEC, "open_in_prompt": True}
    assert _run(spec, [[1, 2, 3], [1, 2, 3, 4]]) == [None, NEWLINE]


def test_a_shorter_output_resets_the_scan() -> None:
    state: dict = {}
    assert forced_token(SPEC, state, [OPEN, 1, 2]) is None
    assert forced_token(SPEC, state, [5]) is None
    assert state["open_at"] is None


@pytest.mark.parametrize("spec", [None, {}, {"budget": "x", "open": OPEN, "close": CLOSE}])
def test_a_missing_or_malformed_spec_forces_nothing(spec: object) -> None:
    assert forced_token(spec, {}, [OPEN, 1, 2, 3, 4, 5]) is None


def test_the_processor_rewrites_only_the_rows_at_their_budget() -> None:
    logits = torch.zeros(3, 128)
    params: list[dict[str, Any] | None] = [
        {PARAM: SPEC, "__req__": SimpleNamespace(output_ids=[OPEN, 1, 2, 3, NEWLINE])},
        {PARAM: SPEC, "__req__": SimpleNamespace(output_ids=[OPEN, 1])},
        None,
    ]

    out = ThinkingBudgetLogitProcessor()(logits, params)

    assert out[0, CLOSE] == 0.0
    assert torch.isinf(out[0, :CLOSE]).all()
    assert torch.isinf(out[0, CLOSE + 1 :]).all()
    assert torch.equal(out[1], torch.zeros(128))
    assert torch.equal(out[2], torch.zeros(128))


def test_the_processor_keeps_per_request_state_between_steps() -> None:
    req = SimpleNamespace(output_ids=[OPEN, 1])
    state: dict[str, Any] = {PARAM: SPEC, "__req__": req}
    params: list[dict[str, Any] | None] = [state]
    processor = ThinkingBudgetLogitProcessor()
    processor(torch.zeros(1, 128), params)
    req.output_ids = [OPEN, 1, CLOSE, 7]
    processor(torch.zeros(1, 128), params)

    assert state["_sie_thinking_budget_state"]["closed"] is True


def test_the_reference_names_the_module_the_sglang_child_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    # The adapter sends a pickled module-level reference, never code: the child
    # resolves it from the compat directory on its PYTHONPATH.
    monkeypatch.syspath_prepend(str(COMPAT_DIR))
    monkeypatch.delitem(sys.modules, "sie_thinking_budget", raising=False)
    payload = bytes.fromhex(json.loads(PROCESSOR_REFERENCE)["callable"])

    loaded = pickle.loads(payload)  # noqa: S301 - a fixed, code-owned module reference

    assert loaded.__module__ == "sie_thinking_budget"
    assert loaded.__name__ == ThinkingBudgetLogitProcessor.__name__
    assert set(payload) <= set(range(128))
