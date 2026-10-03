"""Validate untrusted OpenAI chat answers and expose only their wire fields."""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any, cast

from sie_server.adapters.remote._http import RemoteUpstreamError

_MAX_CHOICES = 128
_MAX_COUNT = 1 << 32
_MAX_EVENT_BYTES = 1 << 20
_MAX_RESPONSE_BYTES = 64 << 20
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter"})


def _invalid() -> RemoteUpstreamError:
    return RemoteUpstreamError("upstream returned an invalid chat response")


def _json(data: bytes, *, max_bytes: int) -> dict[str, Any]:
    if len(data) > max_bytes:
        raise _invalid()
    try:
        payload = json.loads(data)
    except (ValueError, RecursionError):
        raise _invalid() from None
    if not isinstance(payload, dict) or "error" in payload:
        raise _invalid()
    return payload


def _count(value: Any, *, maximum: int = _MAX_COUNT) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < maximum:
        raise _invalid()
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise _invalid()
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _invalid() from None
    return value


def _usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RemoteUpstreamError("upstream chat response omitted exact usage")
    prompt = _count(value.get("prompt_tokens"))
    completion = _count(value.get("completion_tokens"))
    total = _count(value.get("total_tokens"))
    if total != prompt + completion:
        raise _invalid()
    clean: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    details = value.get("prompt_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise _invalid()
        if details.get("cached_tokens") is not None:
            cached = _count(details["cached_tokens"])
            if cached > prompt:
                raise _invalid()
            clean["prompt_tokens_details"] = {"cached_tokens": cached}
    return clean


def _tool_calls(value: Any, *, stream: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > _MAX_CHOICES:
        raise _invalid()
    clean = []
    seen: set[int] = set()
    for position, raw_tool in enumerate(value):
        if not isinstance(raw_tool, dict):
            raise _invalid()
        tool = cast("dict[str, Any]", raw_tool)
        kind = tool.get("type")
        if kind != "function" and (kind is not None or not stream):
            raise _invalid()
        index = _count(tool.get("index") if stream else position, maximum=_MAX_CHOICES)
        if index in seen:
            raise _invalid()
        seen.add(index)
        result: dict[str, Any] = {"index": index} if stream else {}
        if tool.get("id") is not None or not stream:
            identifier = _text(tool.get("id"))
            if not identifier or len(identifier) > 256:
                raise _invalid()
            result["id"] = identifier
        if kind is not None:
            result["type"] = "function"
        function = tool.get("function")
        if function is None and stream:
            clean.append(result)
            continue
        if not isinstance(function, dict):
            raise _invalid()
        clean_function: dict[str, str] = {}
        for field in ("name", "arguments"):
            if field in function or not stream:
                text = function.get(field)
                if text is None and stream:
                    continue
                text = _text(text)
                if field == "name" and (not text or len(text) > 256):
                    raise _invalid()
                clean_function[field] = text
        result["function"] = clean_function
        clean.append(result)
    return clean


def _token_logprob(value: Any, *, alternatives: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("token"), str):
        raise _invalid()
    probability = value.get("logprob")
    if not isinstance(probability, int | float) or isinstance(probability, bool) or probability > 0:
        raise _invalid()
    try:
        finite = math.isfinite(probability)
    except OverflowError:
        raise _invalid() from None
    if not finite:
        raise _invalid()
    clean: dict[str, Any] = {"token": _text(value["token"]), "logprob": probability}
    if "bytes" in value:
        raw = value["bytes"]
        if raw is not None and (not isinstance(raw, list) or len(raw) > 4096):
            raise _invalid()
        clean["bytes"] = None if raw is None else [_count(byte, maximum=256) for byte in raw]
    if alternatives and "top_logprobs" in value:
        top = value["top_logprobs"]
        if not isinstance(top, list) or len(top) > 20:
            raise _invalid()
        clean["top_logprobs"] = [_token_logprob(item, alternatives=False) for item in top]
    return clean


def _logprobs(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _invalid()
    clean: dict[str, Any] = {}
    for field in ("content", "refusal"):
        tokens = value.get(field)
        if tokens is not None and not isinstance(tokens, list):
            raise _invalid()
        if field in value:
            clean[field] = None if tokens is None else [_token_logprob(item, alternatives=True) for item in tokens]
    return clean


def _message(value: Any, *, stream: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid()
    clean: dict[str, Any] = {}
    if value.get("role") is not None or not stream:
        if value.get("role") != "assistant":
            raise _invalid()
        clean["role"] = "assistant"
    for field in ("content", "refusal"):
        if field in value:
            text = value[field]
            clean[field] = None if text is None else _text(text)
    if value.get("tool_calls") is not None:
        clean["tool_calls"] = _tool_calls(value["tool_calls"], stream=stream)
    if not stream and not any(field in clean for field in ("content", "refusal", "tool_calls")):
        raise _invalid()
    return clean


class ChatStreamParser:
    """Normalize chat events; require each choice to finish and exact final usage."""

    def __init__(self, model: str, *, choices: int = 1) -> None:
        if isinstance(choices, bool) or not isinstance(choices, int) or not 1 <= choices <= _MAX_CHOICES:
            raise ValueError("choices must be between 1 and 128")
        self.model = model
        self.choices = choices
        self.done = False
        self._id = f"chatcmpl-{uuid.uuid4().hex}"
        self._created = int(time.time())
        self._finished: set[int] = set()
        self._usage: dict[str, Any] | None = None
        self._tools: dict[int, dict[int, dict[str, Any]]] = {}

    def _envelope(self, choices: list[dict[str, Any]], *, stream: bool) -> dict[str, Any]:
        return {
            "id": self._id,
            "object": "chat.completion.chunk" if stream else "chat.completion",
            "created": self._created,
            "model": self.model,
            "choices": choices,
        }

    def parse(self, data: bytes) -> dict[str, Any] | None:
        """Parse one complete SSE data event; None denotes a verified [DONE]."""
        if self.done:
            raise _invalid()
        if data == b"[DONE]":
            if len(self._finished) != self.choices or self._usage is None:
                raise RemoteUpstreamError("upstream chat stream ended without finished choices and exact usage")
            self.done = True
            return None
        payload = _json(data, max_bytes=_MAX_EVENT_BYTES)
        choices = self._parse_choices(payload.get("choices"), stream=True)
        clean = self._envelope(choices, stream=True)
        if payload.get("usage") is not None:
            if len(self._finished) != self.choices or self._usage is not None:
                raise _invalid()
            self._usage = _usage(payload["usage"])
            clean["usage"] = self._usage
        return clean

    def completion(self, data: bytes) -> dict[str, Any]:
        """Parse a buffered answer using the same choice, message and usage rules."""
        if self._finished or self._usage is not None or self.done:
            raise _invalid()
        payload = _json(data, max_bytes=_MAX_RESPONSE_BYTES)
        choices = self._parse_choices(payload.get("choices"), stream=False)
        if len(self._finished) != self.choices:
            raise _invalid()
        self._usage = _usage(payload.get("usage"))
        clean = self._envelope(choices, stream=False)
        clean["usage"] = self._usage
        self.done = True
        return clean

    def finish(self) -> None:
        """Refuse a disconnected or truncated stream even after a finish event."""
        if not self.done:
            raise RemoteUpstreamError("upstream chat stream ended before its terminal event")

    def _parse_choices(self, value: Any, *, stream: bool) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) > self.choices:
            raise _invalid()
        seen: set[int] = set()
        clean = []
        for choice in value:
            if not isinstance(choice, dict):
                raise _invalid()
            index = _count(choice.get("index"), maximum=self.choices)
            if index in seen or index in self._finished:
                raise _invalid()
            seen.add(index)
            reason = choice.get("finish_reason")
            if reason is not None and (not isinstance(reason, str) or reason not in _FINISH_REASONS):
                raise _invalid()
            if not stream and reason is None:
                raise _invalid()
            field = "delta" if stream else "message"
            raw_message = choice.get(field)
            message = _message(raw_message, stream=stream)
            tools = message.get("tool_calls", [])
            if stream:
                self._track_tools(index, tools, finished=reason is not None, require_tools=reason == "tool_calls")
            elif reason == "tool_calls" and not tools:
                raise _invalid()
            result = {"index": index, field: message, "finish_reason": reason}
            if "logprobs" in choice:
                # Reasoning is omitted from the normalized message. Its token
                # probabilities must be omitted with it, before losing that evidence.
                has_reasoning = any(
                    raw_message.get(key) not in (None, "") for key in ("reasoning_content", "reasoning")
                )
                result["logprobs"] = None if has_reasoning else _logprobs(choice["logprobs"])
            if reason is not None:
                self._finished.add(index)
            clean.append(result)
        return clean

    def _track_tools(
        self, choice: int, fragments: list[dict[str, Any]], *, finished: bool, require_tools: bool
    ) -> None:
        """Retain bounded headers and presence flags, never accumulate argument bodies."""
        tools = self._tools.setdefault(choice, {})
        for fragment in fragments:
            state = tools.setdefault(fragment["index"], {})
            function = fragment.get("function", {})
            for field, source in (("id", fragment), ("name", function)):
                if field in source:
                    text = state.get(field, "") + source[field]
                    if len(text) > 256:
                        raise _invalid()
                    state[field] = text
            if "type" in fragment:
                state["type"] = fragment["type"]
            if "arguments" in function:
                state["arguments"] = True
        if not finished:
            return
        if require_tools and not tools:
            raise _invalid()
        if any(not all(field in state for field in ("id", "name", "type", "arguments")) for state in tools.values()):
            raise _invalid()
