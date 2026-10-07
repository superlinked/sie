"""Bridge normalized upstream chat responses onto the existing queue chunks."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from sie_server.adapters._generation_base import (
    FinishReason,
    GenerationAdapter,
    GenerationChunk,
    GenerationDrainingError,
    GenerationInputTooLongError,
    GenerationInvalidRequestError,
    ToolCallDelta,
    aclose_with_error_precedence,
)
from sie_server.adapters.errors import InputTooLongError, UpstreamUnavailableError
from sie_server.adapters.remote._http import RemoteUpstreamError, generation_error
from sie_server.config.upstreams import RemoteServingDisabledError
from sie_server.processors.tool_call_grammar import normalize_tool_choice
from sie_server.types.inputs import InvalidInputError

_MAX_RESPONSE_BYTES = 64 << 20


def _usage(payload: dict[str, Any], body: dict[str, Any], context_length: int) -> dict[str, Any] | None:
    usage = payload.get("usage")
    if usage is not None:
        prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
        if prompt > context_length or completion > min(body["max_tokens"], context_length - prompt) * body["n"]:
            raise RemoteUpstreamError("upstream chat usage exceeded the configured token bounds")
    return usage


def _logprobs(choice: dict[str, Any], body: dict[str, Any]) -> tuple[dict[str, Any], ...] | None:
    probabilities = choice.get("logprobs")
    if body.get("logprobs") and probabilities is not None:
        return tuple(probabilities.get("content") or ()) or None
    return None


def _check_tools(calls: dict[int, str], body: dict[str, Any], *, finished: bool) -> None:
    mode, name = normalize_tool_choice(body.get("tool_choice"))
    allowed = {tool["function"]["name"] for tool in body.get("tools", [])}
    if calls and (mode == "none" or not allowed or (not body.get("parallel_tool_calls", True) and len(calls) > 1)):
        raise RemoteUpstreamError("upstream chat did not honor the requested tools")
    permitted = {name} if name is not None and name in allowed else allowed if name is None else set()
    for value in calls.values():
        if (finished and value not in permitted) or (
            not finished and not any(candidate.startswith(value) for candidate in permitted)
        ):
            raise RemoteUpstreamError("upstream chat called an unrequested tool")
    if finished and mode in ("required", "named") and not calls:
        raise RemoteUpstreamError("upstream chat omitted the required tool call")


def _terminal(usage: dict[str, Any], reasons: list[str], **kwargs: Any) -> GenerationChunk:
    if "content_filter" in reasons:
        raise RemoteUpstreamError("upstream chat refused its output")
    reason = "tool_calls" if "tool_calls" in reasons else "length" if "length" in reasons else "stop"
    details = usage.get("prompt_tokens_details") or {}
    return GenerationChunk(
        text_delta="",
        done=True,
        finish_reason=cast("FinishReason", reason),
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        cached_tokens=details.get("cached_tokens"),
        **kwargs,
    )


def remote_chat_chunks(
    adapter: GenerationAdapter,
    body: dict[str, Any],
    *,
    requested_model: str,
    context_length: int,
    keep_reasoning: bool = False,
) -> AsyncIterator[GenerationChunk]:
    """Prepare chat without dispatch; the worker owns iteration and cancellation.

    With ``keep_reasoning`` the upstream's private reasoning text rides on
    ``GenerationChunk.reasoning_delta`` chunks, for the caller to count and drop.
    """
    if body["stream"]:
        # Constructing the adapter iterator checks the declared endpoint before
        # queue admission, but does not send the upstream request.
        iterator = adapter.chat_completion_stream(
            body,
            requested_model=requested_model,
            max_response_bytes=_MAX_RESPONSE_BYTES,
            keep_reasoning=keep_reasoning,
        )
        return _stream_chunks(iterator, body, context_length)
    return _buffered_chunks(adapter, body, requested_model, context_length, keep_reasoning)


async def _buffered_chunks(
    adapter: GenerationAdapter,
    body: dict[str, Any],
    requested_model: str,
    context_length: int,
    keep_reasoning: bool,
) -> AsyncIterator[GenerationChunk]:
    try:
        payload = await adapter.chat_completion(
            body,
            requested_model=requested_model,
            max_response_bytes=_MAX_RESPONSE_BYTES,
            keep_reasoning=keep_reasoning,
        )
        usage = _usage(payload, body, context_length)
        if usage is None:
            raise RemoteUpstreamError("upstream chat omitted exact usage")
        candidates: list[dict[str, Any]] = []
        reasoning: list[GenerationChunk] = []
        for choice in sorted(payload["choices"], key=lambda choice: choice["index"]):
            message = choice["message"]
            if message.get("reasoning_content"):
                reasoning.append(
                    GenerationChunk(
                        text_delta="", choice_index=choice["index"], reasoning_delta=message["reasoning_content"]
                    )
                )
            tools = message.get("tool_calls") or []
            _check_tools({index: tool["function"]["name"] for index, tool in enumerate(tools)}, body, finished=True)
            if choice["finish_reason"] == "content_filter" or message.get("refusal"):
                raise RemoteUpstreamError("upstream chat refused its output")
            candidates.append(
                {
                    "text": message.get("content") or "",
                    "finish_reason": choice["finish_reason"],
                    "logprobs": _logprobs(choice, body),
                    "tool_calls": tools or None,
                }
            )
        for chunk in reasoning:
            yield chunk
        if body["n"] > 1:
            yield _terminal(usage, [choice["finish_reason"] for choice in candidates], candidates=tuple(candidates))
            return
        candidate = candidates[0]
        if candidate["text"] or candidate["logprobs"]:
            yield GenerationChunk(text_delta=candidate["text"], is_first=True, logprobs=candidate["logprobs"])
        for index, tool in enumerate(candidate["tool_calls"] or []):
            yield GenerationChunk(
                text_delta="",
                tool_call_delta=ToolCallDelta(
                    index=index,
                    id=tool["id"],
                    function_name=tool["function"]["name"],
                    arguments_delta=tool["function"]["arguments"],
                ),
            )
        yield _terminal(usage, [candidate["finish_reason"]])
    except UpstreamUnavailableError as exc:
        raise generation_error(exc) from None
    except InputTooLongError:
        raise GenerationInputTooLongError("the upstream refused the chat input as too long") from None
    except InvalidInputError:
        raise GenerationInvalidRequestError("messages", "the upstream refused the chat input") from None
    except RemoteServingDisabledError:
        raise GenerationDrainingError("remote serving is unavailable", retry_after_s=1) from None


async def _stream_chunks(
    iterator: AsyncIterator[dict[str, Any]],
    body: dict[str, Any],
    context_length: int,
) -> AsyncIterator[GenerationChunk]:
    usage = None
    reasons: dict[int, str] = {}
    tools: dict[int, dict[int, str]] = {}
    visible = False
    terminal_selected = False
    try:
        async for payload in iterator:
            usage = _usage(payload, body, context_length) or usage
            for choice in payload["choices"]:
                index, delta = choice["index"], choice["delta"]
                reason = choice["finish_reason"]
                if reason == "content_filter" or delta.get("refusal"):
                    raise RemoteUpstreamError("upstream chat refused its output")
                if delta.get("reasoning_content"):
                    yield GenerationChunk(text_delta="", choice_index=index, reasoning_delta=delta["reasoning_content"])
                calls = tools.setdefault(index, {})
                for tool in delta.get("tool_calls") or []:
                    name = tool.get("function", {}).get("name")
                    calls.setdefault(tool["index"], "")
                    if name:
                        calls[tool["index"]] += name
                _check_tools(calls, body, finished=reason is not None)
                text = delta.get("content") or ""
                if text:
                    yield GenerationChunk(
                        text_delta=text,
                        is_first=not visible and bool(text),
                        choice_index=index,
                        logprobs=_logprobs(choice, body),
                    )
                    visible = visible or bool(text)
                for tool in delta.get("tool_calls") or []:
                    function = tool.get("function", {})
                    yield GenerationChunk(
                        text_delta="",
                        choice_index=index,
                        tool_call_delta=ToolCallDelta(
                            index=tool["index"],
                            id=tool.get("id"),
                            function_name=function.get("name"),
                            arguments_delta=function.get("arguments", ""),
                        ),
                    )
                    visible = True
                if reason:
                    reasons[index] = reason
                    # A finish frame may also carry final text/tool fragments.
                    # Publish those before closing this choice at the gateway.
                    yield GenerationChunk(text_delta="", choice_index=index, finish_reason=cast("FinishReason", reason))
        # Clean exhaustion certifies [DONE], finished choices, and exact usage
        # in the shared adapter parser. Never settle a truncated stream as stop.
        if usage is None or len(reasons) != body["n"]:
            raise RemoteUpstreamError("upstream chat did not finish with exact usage")
        terminal_selected = True
        yield _terminal(usage, list(reasons.values()))
    except UpstreamUnavailableError as exc:
        if visible:
            raise RemoteUpstreamError("the upstream failed during chat generation") from None
        raise generation_error(exc) from None
    except InputTooLongError:
        raise GenerationInputTooLongError("the upstream refused the chat input as too long") from None
    except InvalidInputError:
        raise GenerationInvalidRequestError("messages", "the upstream refused the chat input") from None
    except RemoteServingDisabledError:
        raise GenerationDrainingError("remote serving is unavailable", retry_after_s=1) from None
    finally:
        await aclose_with_error_precedence(iterator, outcome_selected=terminal_selected, context="queued remote chat")
