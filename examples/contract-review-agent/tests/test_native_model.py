from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agents import (
    Agent,
    ModelBehaviorError,
    ModelSettings,
    ModelTracing,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from pydantic import BaseModel

from contract_review_agent.native_model import SIENativeModel
from contract_review_agent.runtime import AppContext, Ledger, instruct_once

set_tracing_disabled(True)


class FakeSIE:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "prompt": prompt, "kwargs": kwargs})
        return self.responses.pop(0)


def generated(text: str, request_id: str) -> dict[str, Any]:
    return {
        "text": text,
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
        "request": {"id": request_id},
    }


@pytest.mark.asyncio
async def test_agents_runner_executes_native_tool_turn_then_finishes() -> None:
    client = FakeSIE(
        [
            generated(
                json.dumps(
                    {
                        "kind": "tool_call",
                        "name": "echo",
                        "arguments": {"value": "clause"},
                    }
                ),
                "request-tool",
            ),
            generated(
                json.dumps(
                    {
                        "kind": "final",
                        "output": "Grounded result: CLAUSE",
                    }
                ),
                "request-final",
            ),
        ]
    )

    @function_tool
    async def echo(value: str) -> str:
        """Uppercase one value."""
        return value.upper()

    agent = Agent(
        name="native-test",
        instructions="Use the echo tool, then report its exact result.",
        model=SIENativeModel(
            "Qwen/Qwen3.5-4B",
            client,  # type: ignore[arg-type]
            provision_timeout_s=30,
        ),
        tools=[echo],
    )

    result = await Runner.run(agent, "Process clause")

    assert result.final_output == "Grounded result: CLAUSE"
    assert len(client.calls) == 2
    first_schema = client.calls[0]["kwargs"]["grammar"]["json_schema"]
    assert len(first_schema["oneOf"]) == 2
    assert first_schema["oneOf"][0]["properties"]["name"]["const"] == "echo"
    assert "[tool echo]\nCLAUSE" in client.calls[1]["prompt"]
    assert client.calls[1]["kwargs"]["wait_for_capacity"] is True


class Risk(BaseModel):
    clause: str
    severity: str


class Review(BaseModel):
    recommendation: str
    executed: bool
    risks: list[Risk]


@pytest.mark.asyncio
async def test_agents_runner_validates_native_structured_output() -> None:
    client = FakeSIE(
        [
            generated(
                json.dumps(
                    {
                        "kind": "final",
                        "output": {
                            "recommendation": "Review renewal clause",
                            "executed": True,
                            "risks": [{"clause": "8.2", "severity": "high"}],
                        },
                    }
                ),
                "request-structured",
            )
        ]
    )
    agent = Agent(
        name="structured-test",
        instructions="Return the review.",
        model=SIENativeModel(
            "Qwen/Qwen3.6-27B",
            client,  # type: ignore[arg-type]
            provision_timeout_s=30,
        ),
        output_type=Review,
    )

    result = await Runner.run(agent, "Review this contract")

    assert result.final_output_as(Review) == Review(
        recommendation="Review renewal clause",
        executed=True,
        risks=[Risk(clause="8.2", severity="high")],
    )
    schema = client.calls[0]["kwargs"]["grammar"]["json_schema"]
    output = schema["properties"]["output"]
    assert "$defs" not in output
    assert output["properties"]["risks"]["items"]["$ref"] == ("#/$defs/output__Risk")
    assert schema["$defs"]["output__Risk"]["properties"]["severity"]["type"] == "string"
    assert client.calls[0]["kwargs"]["provision_timeout_s"] == 30


@pytest.mark.asyncio
async def test_instruction_helper_uses_native_multimodal_generate() -> None:
    client = FakeSIE([generated("signed", "request-vision")])
    app = AppContext(
        sie=client,  # type: ignore[arg-type]
        cfg={"cluster": {"provision_timeout_s": 42}},
        ledger=Ledger(),
        contract_text="contract",
        scan_path="scan.png",
        db_path="obligations.db",
    )

    result = await instruct_once(
        app,
        "Qwen/Qwen3.5-4B",
        [
            {"role": "system", "content": "Read only the image."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is it signed?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,YQ=="},
                    },
                ],
            },
        ],
        max_tokens=64,
    )

    call = client.calls[0]
    assert call["prompt"] == "SYSTEM\nRead only the image.\n\nUSER\nIs it signed?"
    assert call["kwargs"]["images"] == [{"data": b"a", "format": "png"}]
    assert call["kwargs"]["max_new_tokens"] == 64
    assert call["kwargs"]["provision_timeout_s"] == 42
    assert result.text == "signed"
    assert result.request_id == "request-vision"


async def _get_response(
    model: SIENativeModel,
    *,
    tools: list[Any] | None = None,
    handoffs: list[Any] | None = None,
    prompt: Any = None,
) -> Any:
    return await model.get_response(
        None,
        "Process clause",
        ModelSettings(),
        tools or [],  # type: ignore[arg-type]
        None,
        handoffs or [],  # type: ignore[arg-type]
        ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=prompt,
    )


@pytest.mark.asyncio
async def test_native_model_rejects_handoffs_and_stored_prompts() -> None:
    model = SIENativeModel(
        "Qwen/Qwen3.5-4B",
        FakeSIE([]),  # type: ignore[arg-type]
        provision_timeout_s=30,
    )

    with pytest.raises(ModelBehaviorError, match="does not support handoffs"):
        await _get_response(model, handoffs=[object()])
    with pytest.raises(ModelBehaviorError, match="does not support stored prompts"):
        await _get_response(model, prompt=object())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "match"),
    [
        ({"text": None}, "returned no text"),
        (generated("not-json", "request-invalid"), "invalid agent JSON"),
    ],
)
async def test_native_model_rejects_invalid_response_text(
    response: dict[str, Any],
    match: str,
) -> None:
    model = SIENativeModel(
        "Qwen/Qwen3.5-4B",
        FakeSIE([response]),  # type: ignore[arg-type]
        provision_timeout_s=30,
    )

    with pytest.raises(ModelBehaviorError, match=match):
        await _get_response(model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn", "match"),
    [
        (
            {"kind": "tool_call", "name": "unknown", "arguments": {}},
            "selected unknown tool",
        ),
        (
            {"kind": "tool_call", "name": "echo", "arguments": []},
            "tool arguments must be an object",
        ),
    ],
)
async def test_native_model_rejects_invalid_tool_calls(
    turn: dict[str, Any],
    match: str,
) -> None:
    @function_tool
    async def echo(value: str) -> str:
        """Return one value."""
        return value

    model = SIENativeModel(
        "Qwen/Qwen3.5-4B",
        FakeSIE([generated(json.dumps(turn), "request-invalid-tool")]),  # type: ignore[arg-type]
        provision_timeout_s=30,
    )

    with pytest.raises(ModelBehaviorError, match=match):
        await _get_response(model, tools=[echo])


@pytest.mark.asyncio
async def test_instruction_timeout_bounds_the_full_model_call() -> None:
    class BlockingSIE:
        async def generate(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(60)
            raise AssertionError("unreachable")

    app = AppContext(
        sie=BlockingSIE(),  # type: ignore[arg-type]
        cfg={"cluster": {}},
        ledger=Ledger(),
        contract_text="contract",
        scan_path="scan.png",
        db_path="obligations.db",
    )

    with pytest.raises(TimeoutError):
        await instruct_once(
            app,
            "Qwen/Qwen3.5-4B",
            [{"role": "user", "content": "Bound this call."}],
            timeout_s=0.01,
        )
