"""System One typed decisions on top of ``extract``.

``SIEClient.system_one`` answers one state against typed ``noul``, ``choice``
and ``score`` questions with a decision model, and returns the System One
response shape (``{model, answers, usage}``) that SIE's ``POST /v1/systemone``
serves. It runs as an ``extract`` call, so it waits for capacity and model
loading like every other SDK call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sie_sdk.client.errors import RequestError
from sie_sdk.types import ExtractResult, Item, SystemOneResult

# Score levels per question, as the System One API accepts.
MAX_SCORE_LEVELS = 10

# Options every System One call sends: input that does not fit the model's row
# is refused instead of being cut.
SYSTEM_ONE_OPTIONS: dict[str, Any] = {"overflow_policy": "error"}


def system_one_request(state: Any, questions: Mapping[str, Any]) -> tuple[Item, dict[str, Any]]:
    """The extract item and ``output_schema`` for a System One call.

    A string state is the item's text; an object or a list of turns rides in
    ``metadata.state``. A question may omit ``instructions``; a score question
    takes 1 to 10 levels, none of them ``None``.

    Raises:
        RequestError: The state or a question is malformed.
    """
    item: Item
    if isinstance(state, str):
        item = Item(text=state)
    elif isinstance(state, (Mapping, list)):
        item = Item(metadata={"state": state})
    else:
        raise RequestError("state must be a string, a mapping, or a list", code="INVALID_INPUT")
    if not isinstance(questions, Mapping) or not questions:
        raise RequestError("questions must be a non-empty mapping of questions by id", code="INVALID_INPUT")
    schema: dict[str, Any] = {}
    for qid, question in questions.items():
        if not isinstance(question, Mapping):
            raise RequestError(f"question {qid!r} must be a mapping", code="INVALID_INPUT")
        normalized = dict(question)
        if normalized.get("instructions") is None:
            normalized["instructions"] = ""
        levels = normalized.get("criteria")
        if normalized.get("type") == "score" and isinstance(levels, list):
            if not levels:
                raise RequestError(f"score question {qid!r} has no levels", code="INVALID_INPUT")
            if len(levels) > MAX_SCORE_LEVELS:
                raise RequestError(
                    f"score question {qid!r} has {len(levels)} levels; at most {MAX_SCORE_LEVELS} are allowed",
                    code="INVALID_INPUT",
                )
            if any(level is None for level in levels):
                raise RequestError(f"score question {qid!r} has a None level", code="INVALID_INPUT")
        schema[str(qid)] = normalized
    return item, schema


def choice_confidence(probabilities: list[float]) -> float:
    """System One choice confidence: ``(p_max - 1/n) / (1 - 1/n)``, within [0, 1]."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    floor = 1.0 / n
    return min(1.0, max(0.0, (max(probabilities) - floor) / (1.0 - floor)))


def score_confidence(probabilities: list[float]) -> float:
    """System One score confidence: ``max(0, 1 - spread / even)``.

    ``spread`` is the expected distance from the most likely level and
    ``even`` that of a uniform distribution from the middle level.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    mode = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return min(1.0, max(0.0, 1.0 - spread / even))


def _answer(qid: str, answer: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(answer)
    probabilities = result.get("probabilities")
    try:
        if result.get("type") == "choice" and isinstance(probabilities, Mapping):
            result["confidence"] = choice_confidence([float(p) for p in probabilities.values()])
        elif result.get("type") == "score" and isinstance(probabilities, Mapping):
            ordered = [p for _, p in sorted((int(key), float(p)) for key, p in probabilities.items())]
            result["confidence"] = score_confidence(ordered)
    except (TypeError, ValueError) as e:
        raise RequestError(f"unreadable answer to question {qid!r}: {e}", code="INVALID_RESPONSE") from e
    return result


def system_one_result(model: str, result: ExtractResult, questions: Mapping[str, Any]) -> SystemOneResult:
    """An extract result in the System One response shape.

    Raises:
        RequestError: The model refused the state (for example, it does not fit
            the model's window), or answered no question of the request.
    """
    error = result.get("error")
    if error:
        raise RequestError(
            str(error.get("message", "the state could not be read")),
            code=str(error.get("code", "INVALID_INPUT")),
            status_code=422,
            param="state",
        )
    data = result.get("data") or {}
    missing = [qid for qid in questions if not isinstance(data.get(qid), Mapping)]
    if missing:
        raise RequestError(
            f"model {model!r} returned no answer for questions {missing}; use a decision model "
            "such as fastino/GLiNER2.5-Decide",
            code="INVALID_RESPONSE",
        )
    usage = result.get("request", {}).get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    return {
        "model": model,
        "answers": {qid: _answer(qid, data[qid]) for qid in questions},
        "usage": {"input_tokens": input_tokens if isinstance(input_tokens, int) else 0, "output_tokens": 0},
    }
