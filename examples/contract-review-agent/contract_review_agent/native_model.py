"""Agents SDK model adapter backed by SIE's native generate primitive."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any
from uuid import uuid4

from agents import (
    FunctionTool,
    Handoff,
    Model,
    ModelBehaviorError,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Tool,
    Usage,
)
from agents.agent_output import AgentOutputSchemaBase
from agents.items import TResponseInputItem, TResponseOutputItem
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from sie_sdk import SIEAsyncClient

_DEFAULT_MAX_NEW_TOKENS = 1200


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_unset=True)
        if isinstance(dumped, dict):
            return dumped
    raise ModelBehaviorError(
        f"Unsupported Agents SDK input item: {type(item).__name__}"
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, default=str)
    parts: list[str] = []
    for part in content:
        row = _as_dict(part)
        text = next(
            (
                row[key]
                for key in ("text", "input_text", "output_text", "refusal")
                if isinstance(row.get(key), str)
            ),
            None,
        )
        if text is not None:
            parts.append(text)
        else:
            parts.append(json.dumps(row, ensure_ascii=False, default=str))
    return "\n".join(parts)


def _conversation(input_items: str | list[TResponseInputItem]) -> str:
    if isinstance(input_items, str):
        return f"[user]\n{input_items}"

    rows = [_as_dict(item) for item in input_items]
    call_names = {
        row["call_id"]: row["name"]
        for row in rows
        if row.get("type") == "function_call"
        and isinstance(row.get("call_id"), str)
        and isinstance(row.get("name"), str)
    }
    turns: list[str] = []
    for row in rows:
        item_type = row.get("type")
        if item_type == "function_call":
            turns.append(
                "[assistant tool call]\n"
                f"name: {row.get('name')}\n"
                f"arguments: {row.get('arguments')}"
            )
        elif item_type == "function_call_output":
            call_id = row.get("call_id")
            name = call_names.get(call_id, call_id)
            turns.append(f"[tool {name}]\n{_content_text(row.get('output'))}")
        elif item_type in (None, "message"):
            role = row.get("role", "user")
            turns.append(f"[{role}]\n{_content_text(row.get('content', ''))}")
        else:
            turns.append(
                f"[{item_type}]\n{json.dumps(row, ensure_ascii=False, default=str)}"
            )
    return "\n\n".join(turns)


def _function_tools(tools: list[Tool]) -> list[FunctionTool]:
    function_tools: list[FunctionTool] = []
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            raise ModelBehaviorError(
                f"SIE native agent adapter supports function tools, got {type(tool).__name__}"
            )
        function_tools.append(tool)
    return function_tools


def _selected_tools(
    tools: list[FunctionTool],
    model_settings: ModelSettings,
) -> tuple[list[FunctionTool], bool]:
    choice = model_settings.tool_choice
    if choice in (None, "auto"):
        return tools, True
    if choice == "required":
        return tools, False
    if choice == "none":
        return [], True
    if isinstance(choice, str):
        selected = [tool for tool in tools if tool.name == choice]
        if not selected:
            raise ModelBehaviorError(f"Unknown required tool: {choice}")
        return selected, False
    raise ModelBehaviorError(f"Unsupported tool choice: {choice!r}")


def _namespace_schema_defs(
    schema: dict[str, Any],
    namespace: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move local definitions to collision-free root names."""
    normalized = deepcopy(schema)
    raw_defs = normalized.pop("$defs", None)
    if raw_defs is None:
        return normalized, {}
    if not isinstance(raw_defs, dict):
        raise ModelBehaviorError("JSON Schema $defs must be an object")

    names = {name: f"{namespace}__{name}" for name in raw_defs}
    refs = {f"#/$defs/{name}": f"#/$defs/{names[name]}" for name in raw_defs}

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: refs.get(item, item)
                if key == "$ref" and isinstance(item, str)
                else rewrite(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    return rewrite(normalized), {
        names[name]: rewrite(definition) for name, definition in raw_defs.items()
    }


def _turn_schema(
    tools: list[FunctionTool],
    output_schema: AgentOutputSchemaBase | None,
    *,
    allow_final: bool,
) -> dict[str, Any]:
    branches: list[dict[str, Any]] = []
    root_defs: dict[str, Any] = {}
    for index, tool in enumerate(tools):
        arguments, definitions = _namespace_schema_defs(
            tool.params_json_schema,
            f"tool_{index}",
        )
        root_defs.update(definitions)
        branches.append(
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "tool_call"},
                    "name": {"type": "string", "const": tool.name},
                    "arguments": arguments,
                },
                "required": ["kind", "name", "arguments"],
                "additionalProperties": False,
            }
        )

    if allow_final:
        final_schema: dict[str, Any]
        if output_schema is None or output_schema.is_plain_text():
            final_schema = {"type": "string"}
        else:
            final_schema, definitions = _namespace_schema_defs(
                output_schema.json_schema(),
                "output",
            )
            root_defs.update(definitions)
        branches.append(
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "final"},
                    "output": final_schema,
                },
                "required": ["kind", "output"],
                "additionalProperties": False,
            }
        )

    if not branches:
        raise ModelBehaviorError(
            "The turn permits neither a tool call nor a final response"
        )
    schema = branches[0] if len(branches) == 1 else {"oneOf": branches}
    if root_defs:
        schema["$defs"] = root_defs
    return schema


def _tool_catalog(tools: list[FunctionTool]) -> str:
    if not tools:
        return "(none)"
    return "\n\n".join(
        f"{tool.name}: {tool.description}\n"
        f"arguments JSON Schema: {json.dumps(tool.params_json_schema, ensure_ascii=False)}"
        for tool in tools
    )


def _prompt_for_turn(
    system_instructions: str | None,
    input_items: str | list[TResponseInputItem],
    tools: list[FunctionTool],
    *,
    allow_final: bool,
) -> str:
    protocol = (
        "Return one JSON object. To call a tool, set kind to tool_call, select exactly "
        "one listed tool, and supply its arguments. To finish, set kind to final and "
        "put the answer in output. Do not invent tool results."
        if allow_final
        else "Return one JSON object calling exactly one listed tool. Do not finish yet."
    )
    sections = [
        f"SYSTEM\n{system_instructions}" if system_instructions else "",
        f"TOOLS\n{_tool_catalog(tools)}",
        f"TURN PROTOCOL\n{protocol}",
        f"CONVERSATION\n{_conversation(input_items)}",
    ]
    return "\n\n".join(section for section in sections if section)


def _usage(result: dict[str, Any]) -> Usage:
    raw = result.get("usage")
    row = raw if isinstance(raw, dict) else {}

    def integer(key: str) -> int:
        value = row.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    return Usage(
        requests=1,
        input_tokens=integer("prompt_tokens"),
        output_tokens=integer("completion_tokens"),
        total_tokens=integer("total_tokens"),
    )


def _request_id(result: dict[str, Any]) -> str | None:
    request = result.get("request")
    if not isinstance(request, dict):
        return None
    value = request.get("id")
    return value if isinstance(value, str) else None


def _message(text: str, request_id: str | None) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=request_id or f"sie-message-{uuid4().hex}",
        content=[
            ResponseOutputText(
                text=text,
                type="output_text",
                annotations=[],
                logprobs=[],
            )
        ],
        role="assistant",
        status="completed",
        type="message",
    )


class SIENativeModel(Model):
    """Execute one Agents SDK model turn through SIEAsyncClient.generate."""

    def __init__(
        self,
        model: str,
        client: SIEAsyncClient,
        *,
        provision_timeout_s: float,
    ) -> None:
        self.model = model
        self._client = client
        self._provision_timeout_s = provision_timeout_s

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> ModelResponse:
        del tracing, previous_response_id, conversation_id
        if handoffs:
            raise ModelBehaviorError(
                "SIE native agent adapter does not support handoffs"
            )
        if prompt is not None:
            raise ModelBehaviorError(
                "SIE native agent adapter does not support stored prompts"
            )

        selected_tools, allow_final = _selected_tools(
            _function_tools(tools),
            model_settings,
        )
        schema = _turn_schema(selected_tools, output_schema, allow_final=allow_final)
        result = await self._client.generate(
            self.model,
            _prompt_for_turn(
                system_instructions,
                input,
                selected_tools,
                allow_final=allow_final,
            ),
            max_new_tokens=model_settings.max_tokens or _DEFAULT_MAX_NEW_TOKENS,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            frequency_penalty=model_settings.frequency_penalty,
            presence_penalty=model_settings.presence_penalty,
            grammar={
                "json_schema": schema,
                "label": "agent_turn",
                "strict": True,
            },
            wait_for_capacity=True,
            provision_timeout_s=self._provision_timeout_s,
        )
        text = result.get("text")
        if not isinstance(text, str):
            raise ModelBehaviorError("SIE native generate returned no text")
        try:
            turn = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelBehaviorError(
                "SIE native generate returned invalid agent JSON"
            ) from exc
        if not isinstance(turn, dict):
            raise ModelBehaviorError("SIE native agent turn must be a JSON object")

        request_id = _request_id(result)
        output: list[TResponseOutputItem]
        if turn.get("kind") == "tool_call":
            name = turn.get("name")
            arguments = turn.get("arguments")
            if not isinstance(name, str) or name not in {
                tool.name for tool in selected_tools
            }:
                raise ModelBehaviorError(
                    f"SIE native agent selected unknown tool: {name!r}"
                )
            if not isinstance(arguments, dict):
                raise ModelBehaviorError("SIE native tool arguments must be an object")
            call_id = f"sie-call-{uuid4().hex}"
            output = [
                ResponseFunctionToolCall(
                    id=request_id or call_id,
                    call_id=call_id,
                    arguments=json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                    name=name,
                    type="function_call",
                )
            ]
        elif turn.get("kind") == "final" and allow_final:
            final = turn.get("output")
            if output_schema is None or output_schema.is_plain_text():
                if not isinstance(final, str):
                    raise ModelBehaviorError("SIE native final text must be a string")
                final_text = final
            else:
                final_text = json.dumps(
                    final, ensure_ascii=False, separators=(",", ":")
                )
            output = [_message(final_text, request_id)]
        else:
            raise ModelBehaviorError(
                f"Invalid SIE native agent turn kind: {turn.get('kind')!r}"
            )

        return ModelResponse(
            output=output,
            usage=_usage(result),
            response_id=None,
            request_id=request_id,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> AsyncIterator[Any]:
        del (
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        raise ModelBehaviorError(
            "Streaming is not supported by the SIE native agent adapter"
        )
        # Unreachable: preserve async-generator behavior so this error is
        # raised on first iteration rather than when the method is called.
        if False:
            yield None
