"""Close a model's private reasoning block once it reaches a token budget.

The SGLang generation child imports this module by name: its directory is on
the child's ``PYTHONPATH``, and the adapter sends ``PROCESSOR_REFERENCE``, a
pickle that names :class:`ThinkingBudgetLogitProcessor` instead of carrying
code. SGLang resolves it when the launch sets ``--enable-custom-logit-processor``.

Each request carries its budget and the tokenizer's reasoning control tokens
in ``custom_params[PARAM]``. Once the open block has produced ``budget`` tokens,
the processor forces a newline (when the tokenizer has a single newline token)
and then the closing token, the way SGLang's own ``ThinkingBudgetLogitProcessor``
does. After the block closes it leaves the logits alone, so a reasoning-aware
grammar constrains the answer exactly as it does without a budget.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping, Sequence
from typing import Any

PARAM = "sie_thinking_budget"
_STATE = "_sie_thinking_budget_state"
# Protocol-0 pickle of a module-level reference: no code travels with the request.
PROCESSOR_REFERENCE = json.dumps({"callable": b"csie_thinking_budget\nThinkingBudgetLogitProcessor\n.".hex()})


def forced_token(spec: Any, state: MutableMapping[str, Any], output_ids: Sequence[int]) -> int | None:
    """Return the token to force at the next step, or ``None`` to leave the logits alone.

    ``spec`` holds ``budget``, the ``open`` and ``close`` token ids, an optional
    ``newline`` id, and ``open_in_prompt`` when the rendered prompt already
    opened the block. ``state`` persists per request between steps, so each
    step reads only the tokens generated since the previous one.
    """
    if not isinstance(spec, dict) or state.get("closed"):
        return None
    try:
        budget = int(spec["budget"])
        open_id = int(spec["open"])
        close_id = int(spec["close"])
    except (KeyError, TypeError, ValueError):
        return None
    newline_id = spec.get("newline")
    scanned = int(state.get("scanned", 0))
    if scanned > len(output_ids):
        state.clear()
        scanned = 0
    if "open_at" not in state:
        state["open_at"] = -1 if spec.get("open_in_prompt") else None
    open_at = state["open_at"]
    for index in range(scanned, len(output_ids)):
        token = output_ids[index]
        if token == close_id and open_at is not None:
            state["closed"] = True
            state["scanned"] = index + 1
            return None
        if token == open_id:
            open_at = index
    state["open_at"] = open_at
    state["scanned"] = len(output_ids)
    if open_at is None or len(output_ids) - open_at - 1 < budget:
        return None
    if isinstance(newline_id, int) and (not output_ids or output_ids[-1] != newline_id):
        return newline_id
    return close_id


class ThinkingBudgetLogitProcessor:
    """SGLang custom logit processor that enforces each request's thinking budget."""

    def __call__(self, logits: Any, custom_param_list: list[dict[str, Any] | None] | None = None) -> Any:
        if not custom_param_list:
            return logits
        for row, params in enumerate(custom_param_list):
            if not params:
                continue
            req = params.get("__req__")
            if req is None:
                continue
            token = forced_token(params.get(PARAM), params.setdefault(_STATE, {}), req.output_ids)
            if token is not None:
                logits[row, :] = float("-inf")
                logits[row, token] = 0.0
        return logits
