from __future__ import annotations

import json
from typing import Any

import pytest
from agents import Agent, Runner, function_tool, set_tracing_disabled
from pydantic import BaseModel

from contract_review_agent.native_model import SIENativeModel
from contract_review_agent.runtime import AppContext, Ledger, chat_once

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


class Review(BaseModel):
    recommendation: str
    executed: bool


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
    )
    schema = client.calls[0]["kwargs"]["grammar"]["json_schema"]
    assert (
        schema["properties"]["output"]["properties"]["recommendation"]["type"]
        == "string"
    )
    assert client.calls[0]["kwargs"]["provision_timeout_s"] == 30


@pytest.mark.asyncio
async def test_chat_helper_uses_native_multimodal_generate() -> None:
    client = FakeSIE([generated("signed", "request-vision")])
    app = AppContext(
        sie=client,  # type: ignore[arg-type]
        cfg={"cluster": {"provision_timeout_s": 42}},
        ledger=Ledger(),
        contract_text="contract",
        scan_path="scan.png",
        db_path="obligations.db",
    )

    result = await chat_once(
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
