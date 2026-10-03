"""Locally rendered upstream chat for the direct single-node ingress."""

from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationChunk,
    GenerationInvalidRequestError,
    GenerationUnsupportedFieldError,
    aclose_with_error_precedence,
    reasoning_starts_in_prompt,
    resolve_reasoning_format,
    suppress_thinking_blocks,
    thinking_blocks_must_be_hidden,
)
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote.openai import OpenAIUpstreamAdapter
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.config.model import ModelConfig
from sie_server.core.runtime_options import bound_generation, resolve_generation_timeouts
from sie_server.core.tokenizer import load_tokenizer
from sie_server.processors.remote_chat import _check_tools
from sie_server.processors.tool_call_grammar import normalize_tool_choice
from sie_server.processors.tool_call_parser import ToolCallFormat, parse_tool_call_stream

_MAX_PROMPT_BYTES = 4_000_000


@lru_cache(maxsize=64)
def _tokenizer(source: str, revision: str | None) -> Any:
    return load_tokenizer(source, trust_remote_code=True, revision=revision)


def _tool_format(config: ModelConfig) -> ToolCallFormat:
    parser = config.resolve_profile("default").loadtime.get("tool_call_parser")
    if not isinstance(parser, str):
        return "auto"
    parser = parser.lower()
    if parser == "qwen25" or "hermes" in parser:
        return "hermes_json"
    if parser.startswith("qwen"):
        return "qwen_xml"
    if parser.startswith("glm"):
        return "glm_xml"
    return "auto"


async def prepare_rendered_chat(
    adapter: GenerationAdapter,
    body: dict[str, Any],
    *,
    config: ModelConfig,
    requested_model: str,
    max_response_bytes: int,
) -> AsyncIterator[dict[str, Any]] | None:
    """Prefer a usable onboarded template; otherwise select declared chat."""
    body = dict(body)
    if body.get("parallel_tool_calls") is None:
        body.pop("parallel_tool_calls", None)
    mode, _ = normalize_tool_choice(body.get("tool_choice"))
    response_format = body.get("response_format") or {}
    if (
        config.remote_backed
        or not isinstance(adapter, (OpenAIUpstreamAdapter, SieUpstreamAdapter))
        or (isinstance(adapter, OpenAIUpstreamAdapter) and not adapter.supports_raw_completions)
        or mode in {"required", "named"}
        or (body.get("n") or 1) > 1
        or response_format.get("type", "text") != "text"
        or body.get("logprobs")
        or body.get("logit_bias")
        or any(
            body.get(field) is not None
            for field in ("min_p", "repetition_context_size", "role_mapping", "user", "safety_identifier")
        )
        or (mode != "none" and any(tool.get("function", {}).get("strict") is True for tool in body.get("tools") or []))
    ):
        return None
    source = config.hf_id or config.weights_path
    if not source:
        return None
    try:
        tokenizer = await asyncio.to_thread(_tokenizer, str(source), config.hf_revision if config.hf_id else None)
    except Exception:  # noqa: BLE001 - unavailable onboarded template selects chat
        return None
    if not getattr(tokenizer, "chat_template", None):
        return None
    messages = copy.deepcopy(body["messages"])
    for message in messages:
        if message["role"] == "developer":
            message["role"] = "system"
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "".join(part.get("text", "") for part in content)
        elif content is None:
            message["content"] = ""
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                try:
                    call["function"]["arguments"] = json.loads(arguments) if arguments.strip() else {}
                except ValueError:
                    call["function"]["arguments"] = {}
    tools = body.get("tools") if mode != "none" else None
    kwargs = dict(body.get("chat_template_kwargs") or {})
    assert config.tasks.generate is not None
    kwargs.update(config.tasks.generate.chat_template_kwargs or {})
    if tools:
        kwargs["tools"] = tools
    try:
        prompt = await asyncio.to_thread(
            tokenizer.apply_chat_template, messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except Exception:  # noqa: BLE001 - template code failures receive a fixed public error
        raise GenerationInvalidRequestError("messages", "failed to render the model-native message prompt") from None
    if not isinstance(prompt, str):
        raise GenerationInvalidRequestError("messages", "chat template did not return a string")
    if len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
        raise GenerationInvalidRequestError("messages", "rendered chat prompt exceeds the byte limit")
    parameters: dict[str, Any] = {
        "prompt": prompt,
        "max_new_tokens": body.get("max_completion_tokens") or body.get("max_tokens"),
        "temperature": 1.0 if body.get("temperature") is None else body["temperature"],
        "top_p": 1.0 if body.get("top_p") is None else body["top_p"],
        **{
            target: body[source]
            for source, target in (
                ("stop", "stop"),
                ("frequency_penalty", "frequency_penalty"),
                ("presence_penalty", "presence_penalty"),
                ("top_k", "top_k"),
                ("repetition_penalty", "repetition_penalty"),
                ("min_tokens", "min_new_tokens"),
                ("seed", "seed"),
            )
            if body.get(source) is not None
        },
    }
    if isinstance(parameters.get("stop"), str):
        parameters["stop"] = [parameters["stop"]]
    try:
        preflight = adapter.preflight_generate(parameters, stream=bool(body.get("stream")))
    except GenerationUnsupportedFieldError:
        # Selection is still local: no upstream request has been dispatched.
        return None
    chunks = adapter.generate_with_preflight(parameters, preflight)
    chunks = _bound_bytes(chunks, max_response_bytes)
    if thinking_blocks_must_be_hidden(config):
        reasoning = resolve_reasoning_format(config, adapter)
        chunks = suppress_thinking_blocks(
            chunks,
            start_inside=isinstance(adapter, OpenAIUpstreamAdapter) and reasoning_starts_in_prompt(prompt, reasoning),
            reasoning_format=reasoning,
        )
    if tools:
        chunks = parse_tool_call_stream(
            chunks,
            tool_call_format=_tool_format(config),
            parallel_tool_calls=body.get("parallel_tool_calls") is not False,
        )
    chunks = bound_generation(chunks, resolve_generation_timeouts(config, None))
    return _events(chunks, body, requested_model)


async def _bound_bytes(chunks: AsyncIterator[GenerationChunk], maximum: int) -> AsyncIterator[GenerationChunk]:
    received = 0
    try:
        async for chunk in chunks:
            received += len(chunk.text_delta.encode("utf-8"))
            if received > maximum:
                raise RemoteUpstreamError("upstream chat response exceeded the byte limit")
            yield chunk
    finally:
        await aclose_with_error_precedence(chunks, outcome_selected=True, context="rendered chat byte bound")


async def _events(
    chunks: AsyncIterator[GenerationChunk], body: dict[str, Any], model: str
) -> AsyncIterator[dict[str, Any]]:
    identifier, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    calls: dict[int, str] = {}
    terminal = False
    first = True
    try:
        async for chunk in chunks:
            if terminal:
                raise RemoteUpstreamError("upstream rendered chat emitted data after completion")
            if chunk.error_code or chunk.finish_reason == "cancelled":
                raise RemoteUpstreamError("upstream rendered chat did not finish successfully")
            delta: dict[str, Any] = {}
            if chunk.text_delta:
                delta["content"] = chunk.text_delta
            if chunk.tool_call_delta is not None:
                call = chunk.tool_call_delta
                calls[call.index] = calls.get(call.index, "") + (call.function_name or "")
                _check_tools(calls, body, finished=False)
                tool: dict[str, Any] = {"index": call.index, "function": {"arguments": call.arguments_delta}}
                if call.id:
                    tool.update(id=call.id, type="function")
                if call.function_name:
                    tool["function"]["name"] = call.function_name
                delta["tool_calls"] = [tool]
            if not delta and not chunk.done:
                continue
            if first:
                delta["role"] = "assistant"
                first = False
            event: dict[str, Any] = {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": chunk.finish_reason if chunk.done else None}],
            }
            if chunk.done:
                if chunk.prompt_tokens is None or chunk.completion_tokens is None or chunk.finish_reason is None:
                    raise RemoteUpstreamError("upstream rendered chat omitted exact usage")
                _check_tools(calls, body, finished=True)
                usage: dict[str, Any] = {
                    "prompt_tokens": chunk.prompt_tokens,
                    "completion_tokens": chunk.completion_tokens,
                    "total_tokens": chunk.prompt_tokens + chunk.completion_tokens,
                }
                if chunk.cached_tokens is not None:
                    usage["prompt_tokens_details"] = {"cached_tokens": chunk.cached_tokens}
                event["usage"] = usage
                terminal = True
            yield event
        if not terminal:
            raise RemoteUpstreamError("upstream rendered chat omitted its terminal chunk")
    finally:
        await aclose_with_error_precedence(chunks, outcome_selected=terminal, context="rendered remote chat")


async def collect_rendered_chat(events: AsyncIterator[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the same normalized events for a buffered chat request."""
    text: list[str] = []
    tools: dict[int, dict[str, Any]] = {}
    final = None
    try:
        async for event in events:
            choice = event["choices"][0]
            delta = choice["delta"]
            text.append(delta.get("content", ""))
            for fragment in delta.get("tool_calls") or []:
                tool = tools.setdefault(
                    fragment["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                tool["id"] += fragment.get("id", "")
                for field in ("name", "arguments"):
                    tool["function"][field] += fragment.get("function", {}).get(field, "")
            if choice["finish_reason"]:
                final = event
        if final is None:
            raise RemoteUpstreamError("upstream rendered chat omitted its terminal chunk")
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
        if tools:
            message["tool_calls"] = [tools[index] for index in sorted(tools)]
        return {
            **{key: final[key] for key in ("id", "created", "model", "usage")},
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": final["choices"][0]["finish_reason"]}],
        }
    finally:
        await aclose_with_error_precedence(events, outcome_selected=True, context="buffered rendered chat")
