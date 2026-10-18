"""Exercise the official OpenAI SDK against a running SIE decision model."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import openai
from openai import APIStatusError, AsyncOpenAI, BadRequestError, NotFoundError, OpenAI
from openai.types.decision import Decision

TEXT = "Please refund the duplicate charge today."
PREDICATE = {"type": "predicate", "instructions": "Does the customer request a refund?"}
CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "choices": [{"value": "billing"}, {"value": "technical"}],
}
SCORE = {
    "type": "score",
    "instructions": "How urgent is the request?",
    "levels": [{"label": "can wait"}, {"label": "today"}],
}


def check_response(response: Decision, kinds: list[str]) -> None:
    # The SDK normally constructs responses without strict schema validation.
    # Validate again so missing required fields cannot pass unnoticed.
    response = Decision.model_validate(response.model_dump())
    assert [answer.type for answer in response.answers] == kinds
    for answer in response.answers:
        if answer.type == "predicate":
            assert math.isfinite(answer.probability) and 0 <= answer.probability <= 1
        elif answer.type in ("choice", "score"):
            assert 0 <= answer.confidence <= 1
            probabilities = [entry.probability for entry in answer.probabilities]
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            assert math.isclose(sum(probabilities), 1, abs_tol=1e-4)
            if answer.type == "score":
                assert math.isclose(answer.score, sum(i * p for i, p in enumerate(probabilities)), abs_tol=1e-6)
    usage = response.usage
    assert usage.input_tokens >= 0
    assert usage.total_tokens == usage.input_tokens
    assert usage.output_tokens == usage.output_tokens_details.reasoning_tokens == 0
    assert usage.input_tokens_details.cached_tokens == usage.input_tokens_details.cache_write_tokens == 0


def run() -> list[dict[str, str]]:
    model = os.environ.get("SIE_DECISION_MODEL", "fastino/GLiNER2.5-Decide")
    base_url = os.environ.get("SIE_OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("SIE_API_KEY", "local")
    results: list[dict[str, str]] = []

    def record(name: str, check: Callable[[], None]) -> None:
        try:
            check()
        except Exception as exc:
            results.append({"name": name, "status": "fail", "detail": f"{type(exc).__name__}: {exc}"})
        else:
            results.append({"name": name, "status": "pass"})

    with OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=60) as client:

        def warmup() -> None:
            deadline = time.monotonic() + 300
            while True:
                try:
                    response = client.decisions.create(model=model, input=TEXT, questions=[PREDICATE])
                except APIStatusError as exc:
                    if (
                        exc.status_code != 503
                        or not isinstance(exc.body, dict)
                        or exc.body.get("code") != "MODEL_LOADING"
                        or time.monotonic() >= deadline
                    ):
                        raise
                    time.sleep(5)
                else:
                    check_response(response, ["predicate"])
                    return

        record("warmup", warmup)
        if results[-1]["status"] != "pass":
            return results

        for question in (PREDICATE, CHOICE, SCORE):
            record(
                question["type"],
                lambda question=question: check_response(
                    client.decisions.create(model=model, input=TEXT, questions=[question]), [question["type"]]
                ),
            )

        def ordered() -> None:
            response = client.decisions.create(
                model=model,
                input=TEXT,
                questions=[PREDICATE, dict(CHOICE, name="same"), dict(SCORE, name="same")],
            )
            check_response(response, ["predicate", "choice", "score"])
            assert [answer.name for answer in response.answers] == [None, "same", "same"]

        record("ordered_unnamed_and_repeated_names", ordered)

        def typed_choice() -> None:
            response = client.decisions.create(
                model=model,
                input="true",
                questions=[dict(CHOICE, choices=[{"value": True}, {"value": "true"}])],
            )
            check_response(response, ["choice"])
            answer = response.answers[0]
            assert answer.type == "choice"
            assert type(answer.probabilities[0].value) is bool
            assert type(answer.probabilities[1].value) is str

        record("typed_choice_values", typed_choice)
        record(
            "text_messages",
            lambda: check_response(
                client.decisions.create(
                    model=model,
                    input=[{"role": "user", "content": [{"type": "input_text", "text": TEXT}]}],
                    questions=[PREDICATE],
                ),
                ["predicate"],
            ),
        )

        def rejected(name: str, request: dict, expected: type[Exception], status: int) -> None:
            def check() -> None:
                try:
                    client.decisions.create(**{"model": model, "input": TEXT, "questions": [PREDICATE], **request})
                except expected as exc:
                    assert exc.status_code == status
                    assert isinstance(exc.body, dict) and isinstance(exc.body["message"], str)
                else:
                    raise AssertionError("request unexpectedly succeeded")

            record(name, check)

        rejected("unknown_model", {"model": "acme/no-such-decision-model"}, NotFoundError, 404)
        rejected("unsupported_field", {"extra_body": {"temperature": 0}}, BadRequestError, 400)
        rejected(
            "unsupported_image",
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/x"}]}]},
            BadRequestError,
            400,
        )
        rejected("input_overflow", {"input": "word " * 10000}, BadRequestError, 400)

    async def async_check() -> None:
        async with AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=60) as client:
            response = await client.decisions.create(model=model, input=TEXT, questions=[PREDICATE])
            check_response(response, ["predicate"])

    record("async_client", lambda: asyncio.run(async_check()))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write the results to this file")
    args = parser.parse_args()
    results = run()
    report = {"openai_sdk": openai.__version__, "cases": results}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n")
    if any(case["status"] != "pass" for case in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
