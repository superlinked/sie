"""Check that the TypeSafe SDK works against SIE, unchanged.

Points the official TypeSafe Python SDK at SIE's System One-compatible API
through the SDK's own environment variables and runs every SDK feature that
reaches the server: ``system_one`` with each question type and state shape,
``models.list``, the async client, and the error classes. Each case is
reported as pass or fail with what it saw, and the run exits non-zero if any
case fails.

    TYPESAFE_BASE_URL=http://localhost:8080 \\
    TYPESAFE_DEFAULT_MODEL=fastino/GLiNER2.5-Decide \\
    TYPESAFE_API_KEY=local \\
    uv run conformance.py [--gateway] [--check-auth] [--json results.json]

``--check-auth`` adds the bad-key case, for a deployment that checks keys
(a cluster gateway or SIE Serverless); a local ``sie-server`` checks none.
``--gateway`` accepts its current zero input-token usage placeholder. Direct
server checks require a positive input-token count.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    Score,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeNotFoundError,
    TypeSafeUnprocessableEntityError,
)

TICKET = "I was charged twice for my March invoice. Please refund one of the charges today."
CRITERIA = {"billing": "payments, invoices or refunds", "technical": "something is broken", "other": None}
LEVELS = ["can wait", "this week", "today"]
# Far more text than any decision model's row holds (512 to 2,048 tokens).
TOO_LONG = " ".join(["refund"] * 20_000)


@dataclass
class Case:
    name: str
    passed: bool
    detail: str


def _probability(value: float) -> bool:
    return 0.0 <= value <= 1.0


def case_models_list(client: TypeSafeClient, model: str) -> str:
    listed = client.models.list()
    names = [entry.name for entry in listed.models]
    assert model in names, f"{model} not in {names[:5]}..."
    entry = next(entry for entry in listed.models if entry.name == model)
    return f"{len(names)} models; {model}: release_date {entry.release_date!r}"


def case_noul(client: TypeSafeClient, model: str) -> str:
    answer = client.system_one(state=TICKET, questions={"spam": Noul(instructions="Is this spam?")}).nouls["spam"]
    assert _probability(answer.noul), answer
    return f"noul={answer.noul:.3f}"


def case_noul_criteria_only(client: TypeSafeClient, model: str) -> str:
    question = Noul(criteria={"true": "asks for money back", "false": "anything else"})
    answer = client.system_one(state=TICKET, questions={"refund": question}).nouls["refund"]
    assert _probability(answer.noul), answer
    return f"no instructions sent; noul={answer.noul:.3f}"


def case_choice(client: TypeSafeClient, model: str) -> str:
    question = Choice(instructions="What does the customer want?", criteria=CRITERIA)
    answer = client.system_one(state=TICKET, questions={"intent": question}).choices["intent"]
    assert answer.choice in CRITERIA, answer
    assert set(answer.probabilities) == set(CRITERIA), answer
    assert _probability(answer.confidence), answer
    return f"choice={answer.choice} confidence={answer.confidence:.3f}"


def case_score(client: TypeSafeClient, model: str) -> str:
    question = Score(instructions="How urgent is this?", criteria=LEVELS)
    answer = client.system_one(state=TICKET, questions={"urgency": question}).scores["urgency"]
    assert 0.0 <= answer.score <= len(LEVELS) - 1, answer
    assert len(answer.legend) == len(answer.probabilities) == len(LEVELS), answer
    assert _probability(answer.confidence), answer
    return f"score={answer.score:.3f} confidence={answer.confidence:.3f}"


def case_dict_questions(client: TypeSafeClient, model: str) -> str:
    response = client.system_one(
        state=TICKET,
        questions={
            "spam": {"type": "noul", "instructions": "Is this spam?"},
            "intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": CRITERIA},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": LEVELS},
        },
    )
    assert set(response.answers) == {"spam", "intent", "urgency"}, response.answers
    return "noul, choice and score in one call"


def case_object_state(client: TypeSafeClient, model: str) -> str:
    state = {"subject": "Double charge", "body": TICKET, "plan": "pro"}
    answer = client.system_one(state=state, questions={"spam": Noul(instructions="Is this spam?")}).nouls["spam"]
    assert _probability(answer.noul), answer
    return f"object state; noul={answer.noul:.3f}"


def case_conversation_state(client: TypeSafeClient, model: str) -> str:
    state = [{"role": "user", "content": "Hi"}, {"role": "user", "content": TICKET}]
    question = Noul(instructions="Does the customer ask for a refund?")
    answer = client.system_one(state=state, questions={"refund": question}).nouls["refund"]
    assert _probability(answer.noul), answer
    return f"conversation state; noul={answer.noul:.3f}"


def case_extra_body(client: TypeSafeClient, model: str) -> str:
    response = client.system_one(
        state=TICKET,
        questions={"spam": Noul(instructions="Is this spam?")},
        extra_body={"metadata": {"user": "u1"}},
    )
    assert "spam" in response.answers
    return "unknown request fields ignored"


def case_envelope(client: TypeSafeClient, model: str, *, gateway: bool = False) -> str:
    response = client.system_one(state=TICKET, questions={"spam": Noul(instructions="Is this spam?")})
    assert response.model == model, response.model
    minimum_input_tokens = 0 if gateway else 1
    assert response.usage.input_tokens is not None and response.usage.input_tokens >= minimum_input_tokens, (
        response.usage
    )
    assert response.usage.output_tokens == 0, response.usage
    return f"model={response.model} input_tokens={response.usage.input_tokens}"


def case_async(client: TypeSafeClient, model: str) -> str:
    async def run() -> float:
        async with AsyncTypeSafeClient() as async_client:
            response = await async_client.system_one(state=TICKET, questions={"spam": Noul(instructions="Spam?")})
            return response.nouls["spam"].noul

    noul = asyncio.run(run())
    assert _probability(noul), noul
    return f"AsyncTypeSafeClient noul={noul:.3f}"


def case_unknown_model(client: TypeSafeClient, model: str) -> str:
    try:
        client.system_one(state=TICKET, questions={"q": Noul(instructions="?")}, model="acme/no-such-model")
    except TypeSafeNotFoundError as error:
        return f"TypeSafeNotFoundError: {error}"
    raise AssertionError("an unknown model was answered")


def case_invalid_question(client: TypeSafeClient, model: str) -> str:
    try:
        client.system_one(state=TICKET, questions={"q": {"type": "maybe", "instructions": "?"}})  # type: ignore[dict-item]
    except TypeSafeUnprocessableEntityError as error:
        return f"TypeSafeUnprocessableEntityError: {error}"
    raise AssertionError("an unknown question type was answered")


def case_too_many_levels(client: TypeSafeClient, model: str) -> str:
    try:
        client.system_one(state=TICKET, questions={"q": Score(criteria=[str(i) for i in range(11)])})
    except TypeSafeUnprocessableEntityError as error:
        return f"TypeSafeUnprocessableEntityError: {error}"
    raise AssertionError("11 score levels were answered")


def case_state_too_long(client: TypeSafeClient, model: str) -> str:
    try:
        client.system_one(state=TOO_LONG, questions={"q": Noul(instructions="Is this spam?")})
    except TypeSafeUnprocessableEntityError as error:
        return f"TypeSafeUnprocessableEntityError: {error}"
    raise AssertionError("a state longer than the model's window was answered, not refused")


def case_bad_key(client: TypeSafeClient, model: str) -> str:
    with TypeSafeClient(api_key="not-a-valid-key") as bad:
        try:
            bad.system_one(state=TICKET, questions={"q": Noul(instructions="?")})
        except TypeSafeAuthenticationError as error:
            return f"TypeSafeAuthenticationError: {error}"
    raise AssertionError("a bad key was accepted")


CASES: list[Callable[[TypeSafeClient, str], str]] = [
    case_models_list,
    case_noul,
    case_noul_criteria_only,
    case_choice,
    case_score,
    case_dict_questions,
    case_object_state,
    case_conversation_state,
    case_extra_body,
    case_envelope,
    case_async,
    case_unknown_model,
    case_invalid_question,
    case_too_many_levels,
    case_state_too_long,
]


def run(*, check_auth: bool = False, gateway: bool = False) -> list[Case]:
    model = os.environ.get("TYPESAFE_DEFAULT_MODEL", "")
    if not model:
        raise SystemExit("Set TYPESAFE_DEFAULT_MODEL to an SIE decision model, e.g. fastino/GLiNER2.5-Decide")
    cases = [*CASES, case_bad_key] if check_auth else CASES
    results: list[Case] = []
    with TypeSafeClient() as client:
        for case in cases:
            name = case.__name__.removeprefix("case_")
            try:
                detail = case_envelope(client, model, gateway=gateway) if case is case_envelope else case(client, model)
                results.append(Case(name, True, detail))
            except Exception as error:  # noqa: BLE001 - every failure is reported, not raised
                results.append(Case(name, False, f"{type(error).__name__}: {error}"))
    return results


def report(results: list[Case]) -> str:
    width = max(len(result.name) for result in results)
    lines = [f"{'PASS' if r.passed else 'FAIL'}  {r.name:<{width}}  {r.detail}" for r in results]
    passed = sum(result.passed for result in results)
    lines.append(f"{passed}/{len(results)} passed")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-auth", action="store_true", help="also check that a bad key is refused")
    parser.add_argument("--gateway", action="store_true", help="allow the gateway's zero input-token usage placeholder")
    parser.add_argument("--json", type=Path, help="write the results to this file")
    args = parser.parse_args(argv)
    results = run(check_auth=args.check_auth, gateway=args.gateway)
    print(report(results))
    if args.json:
        payload: dict[str, Any] = {
            "base_url": os.environ.get("TYPESAFE_BASE_URL"),
            "model": os.environ.get("TYPESAFE_DEFAULT_MODEL"),
            "gateway": args.gateway,
            "cases": [asdict(result) for result in results],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
