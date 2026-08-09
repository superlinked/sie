from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
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
from agents.tool_context import ToolContext
from pydantic import BaseModel

from contract_review_agent import tools as contract_tools
from contract_review_agent.guardrails import _unsafe_verdict, safety_guardrail
from contract_review_agent.native_model import SIENativeModel, _next_required_tool
from contract_review_agent.runtime import AppContext, GenResult, Ledger, instruct_once

set_tracing_disabled(True)


def test_safety_guardrail_blocks_investigator_start() -> None:
    assert safety_guardrail.run_in_parallel is False


@pytest.mark.parametrize("verdict", ["yes", "YES", "unexpected", "No_of_turn>", ""])
def test_safety_guardrail_fails_closed_unless_verdict_is_exact_no(
    verdict: str,
) -> None:
    assert _unsafe_verdict(verdict) is True


def test_safety_guardrail_accepts_unambiguous_no() -> None:
    assert _unsafe_verdict(" no \n") is False


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
                        "kind": "final",
                        "output": "Grounded result: CLAUSE",
                    }
                ),
                "request-final",
            ),
        ]
    )

    @function_tool
    async def echo(query: str) -> str:
        """Uppercase one value."""
        return query.upper()

    agent = Agent(
        name="native-test",
        instructions="Use the echo tool, then report its exact result.",
        model=SIENativeModel(
            "Qwen/Qwen3.5-4B",
            client,  # type: ignore[arg-type]
            stage="test_agent",
            provision_timeout_s=30,
            required_tool_sequence=(("echo", "clause"),),
        ),
        tools=[echo],
    )

    result = await Runner.run(agent, "Process clause")

    assert result.final_output == "Grounded result: CLAUSE"
    assert len(client.calls) == 1
    schema = client.calls[0]["kwargs"]["grammar"]["json_schema"]
    assert len(schema["oneOf"]) == 2
    assert "[tool echo]\nCLAUSE" in client.calls[0]["prompt"]
    assert client.calls[0]["kwargs"]["wait_for_capacity"] is True


def test_required_search_steps_only_advance_on_the_expected_query() -> None:
    sequence = (
        ("search_clauses", "automatic renewal"),
        ("search_clauses", "termination"),
        ("analyze_clause_risks", None),
    )
    repeated_query = [
        {
            "type": "function_call",
            "name": "search_clauses",
            "arguments": json.dumps({"query": "automatic renewal"}),
        },
        {
            "type": "function_call",
            "name": "search_clauses",
            "arguments": json.dumps({"query": "automatic renewal"}),
        },
    ]
    distinct_queries = [
        repeated_query[0],
        {
            "type": "function_call",
            "name": "search_clauses",
            "arguments": json.dumps({"query": "termination"}),
        },
    ]

    assert _next_required_tool(sequence, repeated_query) == "search_clauses"
    assert _next_required_tool(sequence, distinct_queries) == "analyze_clause_risks"


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
            stage="test_agent",
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
        stage="read_signature_page",
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
async def test_required_query_is_emitted_without_generation() -> None:
    @function_tool
    async def search_clauses(query: str) -> str:
        """Find clauses for one exact query."""
        return query

    client = FakeSIE([])
    model = SIENativeModel(
        "Qwen/Qwen3.6-27B",
        client,  # type: ignore[arg-type]
        stage="test_agent",
        provision_timeout_s=30,
        required_tool_sequence=(("search_clauses", "termination"),),
    )

    response = await _get_response(model, tools=[search_clauses])

    assert client.calls == []
    assert response.output[0].name == "search_clauses"
    assert json.loads(response.output[0].arguments) == {"query": "termination"}


@pytest.mark.asyncio
async def test_required_question_is_emitted_without_generation() -> None:
    @function_tool
    async def query_obligations_db(question: str) -> str:
        """Query obligations with one exact question."""
        return question

    client = FakeSIE([])
    model = SIENativeModel(
        "Qwen/Qwen3.6-27B",
        client,  # type: ignore[arg-type]
        stage="test_agent",
        provision_timeout_s=30,
        required_tool_sequence=(("query_obligations_db", "upcoming obligations"),),
    )

    response = await _get_response(model, tools=[query_obligations_db])

    assert client.calls == []
    assert response.output[0].name == "query_obligations_db"
    assert json.loads(response.output[0].arguments) == {
        "question": "upcoming obligations"
    }


@pytest.mark.asyncio
async def test_required_text_rejects_non_string_tool_argument() -> None:
    @function_tool
    async def search_clauses(query: int) -> str:
        """Reject a required text query bound to an integer field."""
        return str(query)

    model = SIENativeModel(
        "Qwen/Qwen3.6-27B",
        FakeSIE([]),  # type: ignore[arg-type]
        stage="test_agent",
        provision_timeout_s=30,
        required_tool_sequence=(("search_clauses", "termination"),),
    )

    with pytest.raises(ModelBehaviorError, match="cannot bind text argument"):
        await _get_response(model, tools=[search_clauses])


@pytest.mark.asyncio
async def test_required_text_accepts_union_string_tool_argument() -> None:
    @function_tool
    async def search_clauses(query: str | None) -> str:
        """Accept a required text query through a string union field."""
        return query or ""

    model = SIENativeModel(
        "Qwen/Qwen3.6-27B",
        FakeSIE([]),  # type: ignore[arg-type]
        stage="test_agent",
        provision_timeout_s=30,
        required_tool_sequence=(("search_clauses", "termination"),),
    )

    response = await _get_response(model, tools=[search_clauses])

    assert json.loads(response.output[0].arguments) == {"query": "termination"}


@pytest.mark.asyncio
async def test_native_model_rejects_handoffs_and_stored_prompts() -> None:
    model = SIENativeModel(
        "Qwen/Qwen3.5-4B",
        FakeSIE([]),  # type: ignore[arg-type]
        stage="test_agent",
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
        stage="test_agent",
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
        stage="test_agent",
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
            stage="test_timeout",
            timeout_s=0.01,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_helper"),
    [("instruct", "instruct"), ("prompt", "prompt")],
)
async def test_query_obligations_db_selects_configured_generation_helper(
    mode: str,
    expected_helper: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_instruct(*_args: Any, **_kwargs: Any) -> GenResult:
        calls.append("instruct")
        return GenResult("SELECT 7 AS value", 0.0, 0.1)

    async def fake_prompt(*_args: Any, **_kwargs: Any) -> GenResult:
        calls.append("prompt")
        return GenResult("SELECT 7 AS value", 0.0, 0.1)

    monkeypatch.setattr(contract_tools, "instruct_once", fake_instruct)
    monkeypatch.setattr(contract_tools, "prompt_once", fake_prompt)
    monkeypatch.setattr(
        contract_tools,
        "_run_select",
        lambda _db_path, _sql: (["value"], [(7,)]),
    )
    app = AppContext(
        sie=FakeSIE([]),  # type: ignore[arg-type]
        cfg={
            "cluster": {"provision_timeout_s": 30},
            "models": {"sql": "sql-model"},
            "sql": {"mode": mode},
        },
        ledger=Ledger(),
        contract_text="contract",
        scan_path="scan.png",
        db_path="obligations.db",
    )

    result = await contract_tools.query_obligations_db.on_invoke_tool(
        ToolContext(
            app,
            tool_name="query_obligations_db",
            tool_call_id="call-query",
            tool_arguments=json.dumps({"question": "Show one value"}),
        ),
        json.dumps({"question": "Show one value"}),
    )

    assert calls == [expected_helper]
    assert result.splitlines() == ["SQL: SELECT 7 AS value", "", "value", "7"]


@pytest.mark.asyncio
async def test_query_obligations_db_rejects_unknown_generation_mode() -> None:
    app = AppContext(
        sie=FakeSIE([]),  # type: ignore[arg-type]
        cfg={
            "cluster": {},
            "models": {"sql": "sql-model"},
            "sql": {"mode": "unknown"},
        },
        ledger=Ledger(),
        contract_text="contract",
        scan_path="scan.png",
        db_path="obligations.db",
    )

    with pytest.raises(ValueError, match="sql.mode must be 'instruct' or 'prompt'"):
        await contract_tools.query_obligations_db.on_invoke_tool(
            ToolContext(
                app,
                tool_name="query_obligations_db",
                tool_call_id="call-invalid",
                tool_arguments=json.dumps({"question": "Show one value"}),
            ),
            json.dumps({"question": "Show one value"}),
        )


@pytest.mark.asyncio
async def test_clause_risk_tool_reads_saved_searches_without_copy_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    async def fake_run(_agent: object, prompt: str, **_kwargs: Any) -> object:
        prompts.append(prompt)
        return SimpleNamespace(
            final_output="grounded risk analysis",
            context_wrapper=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=20, output_tokens=5)
            ),
        )

    monkeypatch.setattr(contract_tools.Runner, "run", fake_run)
    app = AppContext(
        sie=FakeSIE([]),  # type: ignore[arg-type]
        cfg={
            "cluster": {"provision_timeout_s": 30},
            "models": {"reasoning": "reasoning-model"},
        },
        ledger=Ledger(),
        contract_text="contract",
        scan_path="scan.png",
        db_path="obligations.db",
        reasoning_agent=object(),
        clause_cache={
            "search_results": {
                "automatic renewal": [],
                "termination": ["termination clause"],
            }
        },
    )

    result = await contract_tools.analyze_clause_risks.on_invoke_tool(
        ToolContext(
            app,
            tool_name="analyze_clause_risks",
            tool_call_id="call-risk",
            tool_arguments="{}",
        ),
        "{}",
    )

    assert contract_tools.analyze_clause_risks.params_json_schema["properties"] == {}
    assert prompts and "Topic: automatic renewal" not in prompts[0]
    assert "Topic: termination\n\ntermination clause" in prompts[0]
    assert result == "grounded risk analysis"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_results",
    [
        {"termination": "termination clause"},
        {1: ["termination clause"]},
        {"termination": [1]},
        {"": []},
        {"   ": []},
        {"termination": [""]},
        {"termination": ["   "]},
    ],
)
async def test_clause_risk_tool_rejects_malformed_saved_searches(
    search_results: dict[object, object],
) -> None:
    app = AppContext(
        sie=FakeSIE([]),  # type: ignore[arg-type]
        cfg={"cluster": {}, "models": {"reasoning": "reasoning-model"}},
        ledger=Ledger(),
        contract_text="contract",
        scan_path="scan.png",
        db_path="obligations.db",
        reasoning_agent=object(),
        clause_cache={"search_results": search_results},
    )

    with pytest.raises(RuntimeError, match="map string queries to lists of strings"):
        await contract_tools.analyze_clause_risks.on_invoke_tool(
            ToolContext(
                app,
                tool_name="analyze_clause_risks",
                tool_call_id="call-risk",
                tool_arguments="{}",
            ),
            "{}",
        )
