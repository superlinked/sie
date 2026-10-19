"""Anthropic Messages translation onto the chat completions wire.

The Python twin of the gateway's ``handlers::messages``: a Messages request
into a chat completions body, a chat completion into a Message, chat SSE
``data`` payloads into Messages events, and an error answer into the Anthropic
envelope. The shared vectors under ``conformance/anthropic_messages/`` pin both
implementations, so a request is accepted, refused and answered the same way
on ``sie-server`` and on the gateway.

Validation happens here, in Anthropic terms, so a refusal names the field the
caller sent. Unknown fields fail with ``<path>: Extra inputs are not
permitted``. The accepted-but-inert set is ``metadata``, ``cache_control`` in
every position, ``thinking.display: "summarized"`` and the tool flag
``eager_input_streaming``.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from fastapi import HTTPException

from sie_server.api.generate import _parse_native_grammar

SUPPORTED_ANTHROPIC_VERSION = "2023-06-01"
MAX_TOOLS = 64
THINKING_SIGNATURE = "sie-unsigned"
MAX_RETRY_AFTER_S = 60
PING_INTERVAL_S = 10.0
SYSTEM_TURN_REFUSAL = 'role "system" is not supported in messages; use the top-level system parameter'
CONTEXT_MANAGEMENT_REFUSAL = (
    "context_management: Extra inputs are not permitted (Claude Code: set CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1)"
)

_MAX_TOOL_NAME_LEN = 64
_MAX_USER_ID_CHARS = 512
_MIN_THINKING_BUDGET = 1024
_IMAGE_MEDIA_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_TOP_LEVEL_FIELDS = frozenset(
    {
        "model",
        "max_tokens",
        "messages",
        "system",
        "stop_sequences",
        "temperature",
        "top_p",
        "top_k",
        "stream",
        "tools",
        "tool_choice",
        "thinking",
        "metadata",
        "cache_control",
    }
)
_KNOWN_BLOCK_TYPES = frozenset(
    {
        "text",
        "image",
        "tool_result",
        "tool_use",
        "thinking",
        "redacted_thinking",
        "document",
        "search_result",
        "container_upload",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
        "mcp_tool_use",
        "mcp_tool_result",
    }
)
# Unicode White_Space, the set Rust's ``str::trim`` strips.
_WHITESPACE = frozenset(
    chr(code)
    for code in (
        *range(0x09, 0x0E),
        0x20,
        0x85,
        0xA0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
)
_LEADING_DIGITS = re.compile(r"[0-9]+")
_FORWARDED_HEADERS = frozenset(
    {
        "x-sie-request-id",
        "x-sie-served-by",
        "x-sie-upstream",
        "x-sie-fallback-reason",
        "x-sie-fallback-error",
        "x-sie-version",
        "x-sie-server-version",
        "x-sie-worker",
        "x-queue-publish-time",
        "x-queue-wait-time",
        "x-queue-time",
        "x-sie-model-revision",
        "x-sie-execution-identity-sha256",
        "x-sie-execution-binding-sha256",
        "x-inference-time",
        "x-tokenization-time",
        "x-postprocessing-time",
        "x-payload-fetch-time",
    }
)
_ERROR_HEADERS = _FORWARDED_HEADERS | {"x-sie-error-code", "x-sie-error-version", "retry-after", "www-authenticate"}
_ERROR_TYPES: dict[int, str] = {
    HTTPStatus.UNAUTHORIZED: "authentication_error",
    HTTPStatus.PAYMENT_REQUIRED: "billing_error",
    HTTPStatus.FORBIDDEN: "permission_error",
    HTTPStatus.NOT_FOUND: "not_found_error",
    HTTPStatus.REQUEST_TIMEOUT: "timeout_error",
    HTTPStatus.GATEWAY_TIMEOUT: "timeout_error",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "request_too_large",
    HTTPStatus.TOO_MANY_REQUESTS: "rate_limit_error",
    HTTPStatus.SERVICE_UNAVAILABLE: "overloaded_error",
    529: "overloaded_error",
}
_SIE_CODES: dict[int, str] = {
    HTTPStatus.UNAUTHORIZED: "UNAUTHORIZED",
    HTTPStatus.FORBIDDEN: "FORBIDDEN",
    HTTPStatus.NOT_FOUND: "NOT_FOUND",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "PAYLOAD_TOO_LARGE",
}


class MessagesError(Exception):
    """A refusal decided in Anthropic terms: the status and the client-facing message."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class ChatTranslation:
    """A translated request."""

    chat: dict[str, Any]
    model: str
    stream: bool
    first_image: str | None
    """Anthropic path of the first image block, named if the model refuses image input."""


@dataclass(frozen=True)
class ErrorContext:
    """What the Messages route knows about a request that the chat route's error cannot say."""

    first_image: str | None = None


def is_forwarded_header(name: str) -> bool:
    """Whether a chat answer header is copied onto the Messages answer."""
    return name.lower() in _FORWARDED_HEADERS


def json_text(value: Any) -> str:
    """Compact JSON, as the gateway writes it."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json(text: str | bytes) -> Any:
    """Parse strict JSON: no ``NaN`` or ``Infinity``, as the gateway parses it."""
    return json.loads(text, parse_constant=_reject_constant)


def _reject_constant(constant: str) -> Any:
    raise ValueError(f"{constant} is not valid JSON")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_u64(value: Any) -> int | None:
    return value if _is_int(value) and 0 <= value <= _U64_MAX else None


def _as_i64(value: Any) -> int | None:
    return value if _is_int(value) and _I64_MIN <= value <= _I64_MAX else None


def _as_f64(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return float(value)
    except OverflowError:
        return None


def _is_blank(text: str) -> bool:
    return all(char in _WHITESPACE for char in text)


def _invalid(message: str) -> MessagesError:
    return MessagesError(message)


def _extra(path: str) -> MessagesError:
    return _invalid(f"{path}: Extra inputs are not permitted")


def _required(path: str) -> MessagesError:
    return _invalid(f"{path}: Field required")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{path}: Input should be a valid dictionary")
    return value


def _only_keys(mapping: dict[str, Any], allowed: Iterable[str], path: str) -> None:
    allowed_keys = frozenset(allowed)
    for key in mapping:
        if key not in allowed_keys:
            raise _extra(f"{path}.{key}")


def _string(mapping: dict[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if value is None:
        raise _required(f"{path}.{key}")
    if not isinstance(value, str):
        raise _invalid(f"{path}.{key}: Input should be a valid string")
    return value


def _cache_control(mapping: dict[str, Any], path: str) -> None:
    value = mapping.get("cache_control")
    if value is not None and not isinstance(value, dict):
        raise _invalid(f"{path}.cache_control: Input should be a valid dictionary")


def _citations(mapping: dict[str, Any], path: str) -> None:
    value = mapping.get("citations")
    if value is None or value == []:
        return
    raise _invalid(f"{path}.citations: citations are not supported")


class _Parts:
    """Text accumulated in one turn, with image parts kept in order."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def text(self, text: str) -> None:
        if not text:
            return
        if self.items and self.items[-1][0] == "text":
            self.items[-1] = ("text", self.items[-1][1] + text)
        else:
            self.items.append(("text", text))

    def image(self, data_uri: str) -> None:
        self.items.append(("image", data_uri))

    def content(self) -> str | list[dict[str, Any]]:
        if all(kind == "text" for kind, _ in self.items):
            return "".join(value for _, value in self.items)
        return [
            {"type": "text", "text": value} if kind == "text" else {"type": "image_url", "image_url": {"url": value}}
            for kind, value in self.items
        ]


def messages_to_chat(body: Any, anthropic_version: str | None) -> ChatTranslation:
    """Translate an Anthropic Messages body into a chat completions body.

    ``anthropic_version`` is the ``anthropic-version`` header, absent or
    ``2023-06-01``.

    Raises:
        MessagesError: The request is refused, with the Anthropic wording.
    """
    if anthropic_version is not None and anthropic_version != SUPPORTED_ANTHROPIC_VERSION:
        raise _invalid(f"anthropic-version: only {SUPPORTED_ANTHROPIC_VERSION} is supported")
    if not isinstance(body, dict):
        raise _invalid("body: Input should be a valid dictionary")
    for key in body:
        if key in _TOP_LEVEL_FIELDS:
            continue
        if key == "output_config":
            _output_config(body[key])
        elif key == "context_management":
            raise _invalid(CONTEXT_MANAGEMENT_REFUSAL)
        else:
            raise _extra(key)

    model = body.get("model")
    if model is None:
        raise _required("model")
    if not isinstance(model, str):
        raise _invalid("model: Input should be a valid string")
    if not model:
        raise _invalid("model: String should have at least 1 character")
    max_tokens = _max_tokens(body.get("max_tokens"))
    stream = body.get("stream")
    if stream is None:
        stream = False
    elif not isinstance(stream, bool):
        raise _invalid("stream: Input should be a valid boolean")
    temperature = _bounded_number(body, "temperature", low_inclusive=True)
    top_p = _bounded_number(body, "top_p", low_inclusive=False)
    top_k = _top_k(body.get("top_k"))
    stop = _stop_sequences(body)
    _metadata(body)
    cache_control = body.get("cache_control")
    if cache_control is not None and not isinstance(cache_control, dict):
        raise _invalid("cache_control: Input should be a valid dictionary")
    include_reasoning = _thinking(body, max_tokens)
    tools, tool_names = _tools(body)
    tool_choice, parallel_tool_calls = _tool_choice(body, tool_names)

    first_image: list[str] = []
    chat_messages: list[dict[str, Any]] = []
    system = _system(body)
    if system is not None:
        chat_messages.append({"role": "system", "content": system})
    messages = body.get("messages")
    if messages is None:
        raise _required("messages")
    if not isinstance(messages, list):
        raise _invalid("messages: Input should be a valid list")
    if not messages:
        raise _invalid("messages: List should have at least 1 item")
    for index, message in enumerate(messages):
        _translate_message(message, index, chat_messages, first_image)
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "assistant":
        raise _invalid(
            "messages: assistant message prefill is not supported; the conversation must end with a user message"
        )

    chat: dict[str, Any] = {"model": model, "messages": chat_messages, "max_tokens": max_tokens}
    if stop is not None:
        chat["stop"] = stop
    if temperature is not None:
        chat["temperature"] = temperature
    if top_p is not None:
        chat["top_p"] = top_p
    if top_k is not None:
        chat["top_k"] = top_k
    chat["stream"] = stream
    if stream:
        chat["stream_options"] = {"include_usage": True}
    if tools is not None:
        chat["tools"] = tools
    if tool_choice is not None:
        chat["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        chat["parallel_tool_calls"] = parallel_tool_calls
    chat["include_reasoning"] = include_reasoning
    return ChatTranslation(chat=chat, model=model, stream=stream, first_image=first_image[0] if first_image else None)


def _output_config(value: Any) -> None:
    if value is None:
        return
    for key in _object(value, "output_config"):
        raise _extra(f"output_config.{key}")


def _max_tokens(value: Any) -> int:
    if value is None:
        raise _required("max_tokens")
    tokens = _as_u64(value)
    if tokens == 0:
        raise _invalid("max_tokens: Input should be greater than or equal to 1")
    if tokens is not None and tokens <= _U32_MAX:
        return tokens
    signed = _as_i64(value)
    if signed is not None and signed < 1:
        raise _invalid("max_tokens: Input should be greater than or equal to 1")
    raise _invalid("max_tokens: Input should be a valid integer")


def _bounded_number(body: dict[str, Any], key: str, *, low_inclusive: bool) -> Any:
    value = body.get(key)
    if value is None:
        return None
    number = _as_f64(value)
    if number is None or not math.isfinite(number):
        raise _invalid(f"{key}: Input should be a valid number")
    if low_inclusive and number < 0.0:
        raise _invalid(f"{key}: Input should be greater than or equal to 0")
    if not low_inclusive and number <= 0.0:
        raise _invalid(f"{key}: Input should be greater than 0")
    if number > 1.0:
        raise _invalid(f"{key}: Input should be less than or equal to 1")
    return value


def _top_k(value: Any) -> int | None:
    if value is None:
        return None
    k = _as_u64(value)
    if k is not None:
        if 1 <= k <= _U32_MAX:
            return k
        raise _invalid("top_k: Input should be greater than or equal to 1")
    if _as_i64(value) is not None:
        raise _invalid("top_k: Input should be greater than or equal to 1")
    raise _invalid("top_k: Input should be a valid integer")


def _stop_sequences(body: dict[str, Any]) -> list[str] | None:
    value = body.get("stop_sequences")
    if value is None:
        return None
    if not isinstance(value, list):
        raise _invalid("stop_sequences: Input should be a valid list")
    stop: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise _invalid(f"stop_sequences.{index}: Input should be a valid string")
        stop.append(item)
    return stop or None


def _metadata(body: dict[str, Any]) -> None:
    value = body.get("metadata")
    if value is None:
        return
    metadata = _object(value, "metadata")
    _only_keys(metadata, ("user_id",), "metadata")
    user_id = metadata.get("user_id")
    if user_id is None:
        return
    if not isinstance(user_id, str):
        raise _invalid("metadata.user_id: Input should be a valid string")
    if len(user_id) > _MAX_USER_ID_CHARS:
        raise _invalid(f"metadata.user_id: String should have at most {_MAX_USER_ID_CHARS} characters")


def _thinking(body: dict[str, Any], max_tokens: int) -> bool:
    """Whether the request opts into public reasoning."""
    value = body.get("thinking")
    if value is None:
        return False
    thinking = _object(value, "thinking")
    kind = thinking.get("type")
    if kind is None:
        raise _required("thinking.type")
    if not isinstance(kind, str):
        raise _invalid("thinking.type: Input should be a valid string")
    path = f"thinking.{kind}"
    if kind == "enabled":
        _only_keys(thinking, ("type", "budget_tokens", "display"), path)
        raw_budget = thinking.get("budget_tokens")
        if raw_budget is None:
            raise _required(f"{path}.budget_tokens")
        budget = _as_u64(raw_budget)
        if budget is None:
            raise _invalid(f"{path}.budget_tokens: Input should be a valid integer")
        if budget < _MIN_THINKING_BUDGET:
            raise _invalid(f"{path}.budget_tokens: Input should be greater than or equal to {_MIN_THINKING_BUDGET}")
        if budget >= max_tokens:
            raise _invalid(f"{path}.budget_tokens: must be less than max_tokens")
    elif kind == "adaptive":
        _only_keys(thinking, ("type", "display"), path)
    elif kind == "disabled":
        _only_keys(thinking, ("type",), path)
        return False
    elif kind == "between_tools":
        raise _invalid('"thinking.type.between_tools" is not supported for this model.')
    else:
        raise _invalid("thinking.type: Input should be 'enabled', 'adaptive' or 'disabled'")
    display = thinking.get("display")
    if display is None or display == "summarized":
        return True
    if display == "omitted":
        return False
    raise _invalid(f"{path}.display: Input should be 'summarized' or 'omitted'")


def _system(body: dict[str, Any]) -> str | None:
    value = body.get("system")
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        pieces: list[str] = []
        for index, raw in enumerate(value):
            path = f"system.{index}"
            block = _object(raw, path)
            kind = block.get("type")
            if not isinstance(kind, str):
                raise _required(f"{path}.type")
            if kind != "text":
                raise _invalid(f"{path}.type: Input should be 'text'")
            _only_keys(block, ("type", "text", "cache_control", "citations"), path)
            _cache_control(block, path)
            _citations(block, path)
            pieces.append(_string(block, "text", path))
        text = "".join(pieces)
    else:
        raise _invalid("system: Input should be a valid string or list")
    return text or None


def _tools(body: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, list[str]]:
    value = body.get("tools")
    if value is None:
        return None, []
    if not isinstance(value, list):
        raise _invalid("tools: Input should be a valid list")
    if len(value) > MAX_TOOLS:
        raise _invalid(f"tools: at most {MAX_TOOLS} tools are supported, got {len(value)}")
    tools: list[dict[str, Any]] = []
    names: list[str] = []
    for index, raw in enumerate(value):
        path = f"tools.{index}"
        tool = _object(raw, path)
        kind = tool.get("type")
        if kind is not None and kind != "custom":
            if not isinstance(kind, str):
                raise _invalid(f"{path}.type: Input should be a valid string")
            raise _invalid(
                f"{path}.type: tool type '{kind}' is not supported; only custom tools with an input_schema are"
            )
        _only_keys(
            tool, ("type", "name", "description", "input_schema", "cache_control", "eager_input_streaming"), path
        )
        _cache_control(tool, path)
        eager = tool.get("eager_input_streaming")
        if eager is not None and not isinstance(eager, bool):
            raise _invalid(f"{path}.eager_input_streaming: Input should be a valid boolean")
        name = _string(tool, "name", path)
        valid_chars = bool(name) and all(char.isascii() and (char.isalnum() or char in "_-") for char in name)
        if valid_chars and len(name) > _MAX_TOOL_NAME_LEN:
            raise _invalid(f"{path}.name: tool names longer than {_MAX_TOOL_NAME_LEN} characters are not supported")
        if not valid_chars:
            raise _invalid(f"{path}.name: String should match pattern '^[a-zA-Z0-9_-]{{1,{_MAX_TOOL_NAME_LEN}}}$'")
        function: dict[str, Any] = {"name": name}
        description = tool.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise _invalid(f"{path}.description: Input should be a valid string")
            function["description"] = description
        schema = tool.get("input_schema")
        if schema is None:
            raise _required(f"{path}.input_schema")
        if not isinstance(schema, dict):
            raise _invalid(f"{path}.input_schema: Input should be a valid dictionary")
        try:
            _parse_native_grammar({"json_schema": schema})
        except HTTPException:
            raise _invalid(
                f"{path}.input_schema: the schema uses an unsupported JSON Schema keyword (if, then, else, "
                "$dynamicRef, unevaluatedProperties, dependentSchemas) or exceeds the size or depth limits"
            ) from None
        function["parameters"] = schema
        tools.append({"type": "function", "function": function})
        names.append(name)
    return (tools or None), names


def _tool_choice(body: dict[str, Any], tool_names: list[str]) -> tuple[Any, bool | None]:
    value = body.get("tool_choice")
    if value is None:
        return None, None
    choice = _object(value, "tool_choice")
    kind = choice.get("type")
    if kind is None:
        raise _required("tool_choice.type")
    if not isinstance(kind, str):
        raise _invalid("tool_choice.type: Input should be a valid string")
    path = f"tool_choice.{kind}"
    translated: Any
    if kind in ("auto", "any", "none"):
        _only_keys(choice, ("type", "disable_parallel_tool_use"), path)
        translated = {"auto": "auto", "any": "required", "none": "none"}[kind]
    elif kind == "tool":
        _only_keys(choice, ("type", "name", "disable_parallel_tool_use"), path)
        name = _string(choice, "name", path)
        if name not in tool_names:
            raise _invalid(f"{path}.name: tool '{name}' is not declared in tools")
        translated = {"type": "function", "function": {"name": name}}
    else:
        raise _invalid("tool_choice.type: Input should be 'auto', 'any', 'tool' or 'none'")
    disable = choice.get("disable_parallel_tool_use")
    if disable is not None and not isinstance(disable, bool):
        raise _invalid(f"{path}.disable_parallel_tool_use: Input should be a valid boolean")
    parallel = False if disable is True else None
    if not tool_names:
        if kind in ("auto", "none"):
            return None, None
        raise _invalid(f"{path}: tool_choice requires tools")
    return translated, parallel


def _translate_message(message: Any, index: int, out: list[dict[str, Any]], first_image: list[str]) -> None:
    path = f"messages.{index}"
    message = _object(message, path)
    role = message.get("role")
    if role is None:
        raise _required(f"{path}.role")
    if not isinstance(role, str):
        raise _invalid(f"{path}.role: Input should be a valid string")
    if role == "system":
        raise _invalid(f"{path}.role: {SYSTEM_TURN_REFUSAL}")
    if role not in ("user", "assistant"):
        raise _invalid(f"{path}.role: Input should be 'user' or 'assistant'")
    _only_keys(message, ("role", "content"), path)
    content = message.get("content")
    if content is None:
        raise _required(f"{path}.content")
    if isinstance(content, str):
        out.append({"role": role, "content": content})
        return
    if not isinstance(content, list):
        raise _invalid(f"{path}.content: Input should be a valid string or list")
    if not content:
        raise _invalid(f"{path}.content: List should have at least 1 item")
    if role == "user":
        _user_blocks(content, path, out, first_image)
    else:
        _assistant_blocks(content, path, out)


def _block_type(block: dict[str, Any], path: str) -> str:
    kind = block.get("type")
    if kind is None:
        raise _required(f"{path}.type")
    if not isinstance(kind, str):
        raise _invalid(f"{path}.type: Input should be a valid string")
    return kind


def _unsupported_block(path: str, kind: str) -> MessagesError:
    if kind in _KNOWN_BLOCK_TYPES:
        return _invalid(f"{path}.{kind}: content blocks of type '{kind}' are not supported here")
    return _invalid(f"{path}.type: Input tag '{kind}' found using 'type' does not match any of the expected tags")


def _text_block(block: dict[str, Any], path: str) -> str:
    path = f"{path}.text"
    _only_keys(block, ("type", "text", "cache_control", "citations"), path)
    _cache_control(block, path)
    _citations(block, path)
    return _string(block, "text", path)


def _image_block(block: dict[str, Any], path: str) -> str:
    path = f"{path}.image"
    _only_keys(block, ("type", "source", "cache_control"), path)
    _cache_control(block, path)
    raw_source = block.get("source")
    if raw_source is None:
        raise _required(f"{path}.source")
    source = _object(raw_source, f"{path}.source")
    kind = source.get("type")
    if kind in ("url", "file"):
        raise _invalid(f"{path}.source: only base64 image sources are supported")
    if kind != "base64":
        raise _invalid(f"{path}.source.type: Input should be 'base64'")
    base64_path = f"{path}.source.base64"
    _only_keys(source, ("type", "media_type", "data"), base64_path)
    media_type = _string(source, "media_type", base64_path)
    if media_type not in _IMAGE_MEDIA_TYPES:
        raise _invalid(
            f"{base64_path}.media_type: Input should be 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'"
        )
    data = _string(source, "data", base64_path)
    return f"data:{media_type};base64,{data}"


def _user_blocks(blocks: list[Any], path: str, out: list[dict[str, Any]], first_image: list[str]) -> None:
    tool_messages: list[dict[str, Any]] = []
    parts = _Parts()
    tool_images = _Parts()
    for index, raw in enumerate(blocks):
        block_path = f"{path}.content.{index}"
        block = _object(raw, block_path)
        kind = _block_type(block, block_path)
        if kind == "text":
            parts.text(_text_block(block, block_path))
        elif kind == "image":
            parts.image(_image_block(block, block_path))
            if not first_image:
                first_image.append(f"{block_path}.image")
        elif kind == "tool_result":
            result_path = f"{block_path}.tool_result"
            _only_keys(block, ("type", "tool_use_id", "content", "is_error", "cache_control"), result_path)
            _cache_control(block, result_path)
            tool_use_id = _string(block, "tool_use_id", result_path)
            if not tool_use_id:
                raise _invalid(f"{result_path}.tool_use_id: String should have at least 1 character")
            is_error = block.get("is_error")
            if is_error is not None and not isinstance(is_error, bool):
                raise _invalid(f"{result_path}.is_error: Input should be a valid boolean")
            text = _tool_result_content(block.get("content"), result_path, tool_images, first_image)
            tool_messages.append({"role": "tool", "tool_call_id": tool_use_id, "content": text})
        else:
            raise _unsupported_block(block_path, kind)
    out.extend(tool_messages)
    combined = _Parts()
    combined.items = tool_images.items + parts.items
    if combined.items or not out or out[-1]["role"] != "tool":
        out.append({"role": "user", "content": combined.content()})


def _tool_result_content(content: Any, path: str, images: _Parts, first_image: list[str]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _invalid(f"{path}.content: Input should be a valid string or list")
    text: list[str] = []
    for index, raw in enumerate(content):
        block_path = f"{path}.content.{index}"
        block = _object(raw, block_path)
        kind = _block_type(block, block_path)
        if kind == "text":
            text.append(_text_block(block, block_path))
        elif kind == "image":
            images.image(_image_block(block, block_path))
            if not first_image:
                first_image.append(f"{block_path}.image")
        else:
            raise _unsupported_block(block_path, kind)
    return "".join(text)


def _assistant_blocks(blocks: list[Any], path: str, out: list[dict[str, Any]]) -> None:
    text: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for index, raw in enumerate(blocks):
        block_path = f"{path}.content.{index}"
        block = _object(raw, block_path)
        kind = _block_type(block, block_path)
        if kind == "text":
            text.append(_text_block(block, block_path))
        elif kind == "tool_use":
            use_path = f"{block_path}.tool_use"
            _only_keys(block, ("type", "id", "name", "input", "cache_control"), use_path)
            _cache_control(block, use_path)
            call_id = _string(block, "id", use_path)
            name = _string(block, "name", use_path)
            if not call_id or not name:
                field = "id" if not call_id else "name"
                raise _invalid(f"{use_path}.{field}: String should have at least 1 character")
            arguments = block.get("input")
            if arguments is None:
                raise _required(f"{use_path}.input")
            if not isinstance(arguments, dict):
                raise _invalid(f"{use_path}.input: Input should be a valid dictionary")
            tool_calls.append(
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": json_text(arguments)}}
            )
        elif kind == "thinking":
            _only_keys(block, ("type", "thinking", "signature"), f"{block_path}.thinking")
        elif kind == "redacted_thinking":
            _only_keys(block, ("type", "data"), f"{block_path}.redacted_thinking")
        else:
            raise _unsupported_block(block_path, kind)
    joined = "".join(text)
    if not tool_calls:
        out.append({"role": "assistant", "content": joined})
    else:
        out.append({"role": "assistant", "content": joined or None, "tool_calls": tool_calls})


def stop_reason(finish_reason: str | None, has_tool_use: bool) -> str:
    """The Anthropic ``stop_reason`` for a chat ``finish_reason``."""
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter":
        return "refusal"
    if finish_reason in ("tool_calls", "function_call") or has_tool_use:
        return "tool_use"
    return "end_turn"


def anthropic_usage(usage: Any) -> dict[str, int] | None:
    """Anthropic usage from a chat usage block.

    ``input_tokens`` excludes the prompt tokens served from the prefix cache,
    which are reported as ``cache_read_input_tokens``, so the two add up to the
    chat ``prompt_tokens``.
    """
    if not isinstance(usage, dict):
        return None
    prompt = _as_u64(usage.get("prompt_tokens"))
    completion = _as_u64(usage.get("completion_tokens"))
    if prompt is None or completion is None:
        return None
    details = usage.get("prompt_tokens_details")
    cached = _as_u64(details.get("cached_tokens")) if isinstance(details, dict) else None
    cached = min(cached or 0, prompt)
    return {
        "input_tokens": prompt - cached,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
        "output_tokens": completion,
    }


def _zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}


def tool_input(arguments: str) -> dict[str, Any]:
    """The tool input a chat tool call's ``arguments`` string carries.

    Raises:
        ValueError: The arguments are not a JSON object.
    """
    if _is_blank(arguments):
        return {}
    try:
        parsed = parse_json(arguments)
    except ValueError:
        raise ValueError("tool call arguments are not valid JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError("tool call arguments are not a JSON object")
    return parsed


def chat_completion_to_message(chat: Any, model: str, request_id: str) -> dict[str, Any]:
    """Translate a chat completion into a Message.

    ``model`` is the model id the caller sent; ``request_id`` names the answer.

    Raises:
        ValueError: The chat completion cannot be translated.
    """
    choices = chat.get("choices") if isinstance(chat, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat completion has no choices")
    choice = choices[0]
    if not isinstance(choice, dict) or "message" not in choice:
        raise ValueError("chat completion choice has no message")
    message = choice["message"] if isinstance(choice["message"], dict) else {}
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": THINKING_SIGNATURE})
    text = message.get("content")
    if isinstance(text, str) and not _is_blank(text):
        content.append({"type": "text", "text": text})
    has_tool_use = False
    tool_calls = message.get("tool_calls")
    for call in tool_calls if isinstance(tool_calls, list) else []:
        call_id = call.get("id") if isinstance(call, dict) else None
        if not isinstance(call_id, str):
            raise ValueError("tool call has no id")
        if "function" not in call:
            raise ValueError("tool call has no function")
        function = call["function"]
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str):
            raise ValueError("tool call has no name")
        arguments = function.get("arguments")
        content.append(
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": tool_input(arguments if isinstance(arguments, str) else ""),
            }
        )
        has_tool_use = True
    usage = anthropic_usage(chat.get("usage"))
    if usage is None:
        raise ValueError("chat completion has no usage")
    finish_reason = choice.get("finish_reason")
    return {
        "id": f"msg_{request_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(finish_reason if isinstance(finish_reason, str) else None, has_tool_use),
        "stop_sequence": None,
        "usage": usage,
    }


@dataclass(frozen=True)
class Event:
    """One Anthropic SSE event."""

    name: str
    data: dict[str, Any]

    def encode(self) -> bytes:
        return f"event: {self.name}\ndata: {json_text(self.data)}\n\n".encode()


PING = Event("ping", {"type": "ping"})


def _status_for_code(code: str) -> int:
    """The HTTP status a mid-stream chat error code stands for, which decides the Anthropic error type."""
    if code in ("invalid_request", "unsupported_field", "context_exceeded", "INPUT_TOO_LONG", "grammar_invalid"):
        return 400
    if code == "PAYLOAD_TOO_LARGE":
        return 413
    if code in ("rate_limit_exceeded", "COLD_START_RATE_LIMITED"):
        return 429
    if code in (
        "RESOURCE_EXHAUSTED",
        "QUEUE_FULL",
        "MODEL_LOADING",
        "LORA_LOADING",
        "transport_failure",
        "provisioning",
        "QUEUE_UNAVAILABLE",
    ):
        return 503
    if code in (
        "first_chunk_timeout",
        "inter_chunk_timeout",
        "overall_timeout",
        "GATEWAY_TIMEOUT",
        "cancelled",
        "generation_timeout",
    ):
        return 504
    return 500


class StreamMachine:
    """The chat-chunk to Anthropic-event state machine.

    One block is open at a time and block indices are contiguous from 0.
    """

    def __init__(self, message_id: str, model: str) -> None:
        self._message_id = message_id
        self._model = model
        self._next_index = 0
        self._open: tuple[str, int, int | None] | None = None
        self._held_whitespace = ""
        self._tool_indices: list[int] = []
        self._finish_reason: str | None = None
        self._usage: dict[str, int] | None = None
        self._started = False
        self.done = False
        """Whether ``message_stop`` or an ``error`` event has been produced."""

    def start(self) -> list[Event]:
        """The ``message_start`` event, sent as soon as the chat stream opens."""
        if self._started:
            return []
        self._started = True
        message = {
            "id": self._message_id,
            "type": "message",
            "role": "assistant",
            "model": self._model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": _zero_usage(),
        }
        return [Event("message_start", {"type": "message_start", "message": message})]

    def on_data(self, data: str) -> list[Event]:
        """Feed one chat SSE ``data`` payload."""
        events = self.start()
        if self.done:
            return events
        if data == "[DONE]":
            self._finish(events)
            return events
        try:
            chunk = parse_json(data)
        except ValueError:
            self._fail(events, "api_error", "the generation stream sent a malformed chunk")
            return events
        if not isinstance(chunk, dict):
            chunk = {}
        error = chunk.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            status = _status_for_code(code if isinstance(code, str) else "")
            error_type, message, _ = translate_error_body(status, {"error": error}, ErrorContext())
            self._fail(events, error_type, message)
            return events
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta")
            delta = delta if isinstance(delta, dict) else {}
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                self._thinking(reasoning, events)
            text = delta.get("content")
            if isinstance(text, str) and text:
                self._text(text, events)
            calls = delta.get("tool_calls")
            for call in calls if isinstance(calls, list) else []:
                problem = self._tool_call(call, events)
                if problem is not None:
                    self._fail(events, "api_error", problem)
                    return events
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self._close(events)
                self._held_whitespace = ""
                self._finish_reason = finish_reason
        usage = anthropic_usage(chunk.get("usage"))
        if usage is not None:
            self._usage = usage
        if self._finish_reason is not None and self._usage is not None:
            self._complete(events)
        return events

    def on_end(self) -> list[Event]:
        """The chat stream ended. A stream that reported a finish completes; any other ending is an ``error``."""
        events = self.start()
        if not self.done:
            self._finish(events)
        return events

    def on_transport_error(self) -> list[Event]:
        """The chat stream broke before it ended."""
        events = self.start()
        if not self.done:
            self._fail(events, "api_error", "the generation stream ended unexpectedly")
        return events

    def _finish(self, events: list[Event]) -> None:
        if self._finish_reason is not None:
            self._complete(events)
        else:
            self._fail(events, "api_error", "the generation stream ended without a finish reason")

    def _complete(self, events: list[Event]) -> None:
        self._close(events)
        delta = {"stop_reason": stop_reason(self._finish_reason, bool(self._tool_indices)), "stop_sequence": None}
        usage = self._usage if self._usage is not None else _zero_usage()
        events.append(Event("message_delta", {"type": "message_delta", "delta": delta, "usage": usage}))
        events.append(Event("message_stop", {"type": "message_stop"}))
        self.done = True

    def _fail(self, events: list[Event], error_type: str, message: str) -> None:
        events.append(Event("error", {"type": "error", "error": {"type": error_type, "message": message}}))
        self.done = True

    def _open_block(
        self, kind: str, chat_index: int | None, content_block: dict[str, Any], events: list[Event]
    ) -> None:
        index = self._next_index
        self._next_index += 1
        self._open = (kind, index, chat_index)
        events.append(
            Event(
                "content_block_start", {"type": "content_block_start", "index": index, "content_block": content_block}
            )
        )

    def _delta(self, delta: dict[str, Any], events: list[Event]) -> None:
        if self._open is not None:
            index = self._open[1]
            events.append(Event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}))

    def _close(self, events: list[Event]) -> None:
        if self._open is None:
            return
        kind, index, _ = self._open
        self._open = None
        if kind == "thinking":
            signature = {"type": "signature_delta", "signature": THINKING_SIGNATURE}
            events.append(
                Event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": signature})
            )
        events.append(Event("content_block_stop", {"type": "content_block_stop", "index": index}))

    def _thinking(self, text: str, events: list[Event]) -> None:
        if self._open is None or self._open[0] != "thinking":
            self._held_whitespace = ""
            self._close(events)
            self._open_block("thinking", None, {"type": "thinking", "thinking": "", "signature": ""}, events)
        self._delta({"type": "thinking_delta", "thinking": text}, events)

    def _text(self, text: str, events: list[Event]) -> None:
        if self._open is None or self._open[0] != "text":
            if _is_blank(text):
                self._held_whitespace += text
                return
            self._close(events)
            self._open_block("text", None, {"type": "text", "text": ""}, events)
            first = self._held_whitespace + text
            self._held_whitespace = ""
            self._delta({"type": "text_delta", "text": first}, events)
            return
        self._delta({"type": "text_delta", "text": text}, events)

    def _tool_call(self, call: Any, events: list[Event]) -> str | None:
        call = call if isinstance(call, dict) else {}
        chat_index = _as_u64(call.get("index"))
        chat_index = 0 if chat_index is None else chat_index
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        call_id = call.get("id")
        name = function.get("name")
        if chat_index not in self._tool_indices:
            if not isinstance(call_id, str) or not isinstance(name, str):
                return "a tool call arrived without its id and name"
            self._held_whitespace = ""
            self._close(events)
            self._open_block(
                "tool_use", chat_index, {"type": "tool_use", "id": call_id, "name": name, "input": {}}, events
            )
            self._tool_indices.append(chat_index)
        arguments = function.get("arguments")
        if not isinstance(arguments, str) or not arguments:
            return None
        if self._open is None or self._open[0] != "tool_use" or self._open[2] != chat_index:
            return "tool call arguments arrived after another content block started"
        self._delta({"type": "input_json_delta", "partial_json": arguments}, events)
        return None


class FrameParser:
    """Splits an SSE byte stream into the ``data`` payloads of its events."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def push(self, chunk: bytes) -> list[str]:
        self._buffer.extend(chunk.replace(b"\r", b""))
        payloads: list[str] = []
        while (end := self._buffer.find(b"\n\n")) >= 0:
            frame = bytes(self._buffer[: end + 2]).decode("utf-8", errors="replace")
            del self._buffer[: end + 2]
            data: str | None = None
            for line in frame.split("\n"):
                if not line.startswith("data:"):
                    continue
                rest = line[len("data:") :].removeprefix(" ")
                data = rest if data is None else f"{data}\n{rest}"
            if data is not None:
                payloads.append(data)
        return payloads


async def stream_events(
    chunks: AsyncIterator[bytes], machine: StreamMachine, ping_interval: float = PING_INTERVAL_S
) -> AsyncIterator[bytes]:
    """The Anthropic SSE body for a chat SSE body.

    A ``ping`` goes out whenever no event was sent for ``ping_interval``
    seconds. An ``error`` event ends the body and closes ``chunks``, so a
    stream the translation cannot follow stops generating, as a client that
    goes away does.
    """
    loop = asyncio.get_running_loop()
    parser = FrameParser()
    pending: asyncio.Future[bytes] | None = None
    try:
        for event in machine.start():
            yield event.encode()
        deadline = loop.time() + ping_interval
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(chunks))
            timeout = None if machine.done else max(deadline - loop.time(), 0.0)
            done, _ = await asyncio.wait({pending}, timeout=timeout)
            if not done:
                yield PING.encode()
                deadline = loop.time() + ping_interval
                continue
            arrived, pending = pending, None
            try:
                chunk = arrived.result()
            except StopAsyncIteration:
                for event in machine.on_end():
                    yield event.encode()
                return
            except Exception:  # noqa: BLE001 - any failure of the chat body ends the stream with an error event
                for event in machine.on_transport_error():
                    yield event.encode()
                return
            for payload in parser.push(chunk):
                events = machine.on_data(payload)
                for event in events:
                    yield event.encode()
                    deadline = loop.time() + ping_interval
                if any(event.name == "error" for event in events):
                    return
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.wait({pending})
            if not pending.cancelled():
                pending.exception()
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()


def error_type_for_status(status: int) -> str:
    """The Anthropic ``error.type`` for an HTTP status."""
    known = _ERROR_TYPES.get(status)
    if known is not None:
        return known
    if HTTPStatus.BAD_REQUEST <= status < HTTPStatus.INTERNAL_SERVER_ERROR:
        return "invalid_request_error"
    return "api_error"


def sie_code_for_status(status: int) -> str:
    """The ``X-SIE-Error-Code`` of a refusal this module decides."""
    known = _SIE_CODES.get(status)
    if known is not None:
        return known
    if HTTPStatus.BAD_REQUEST <= status < HTTPStatus.INTERNAL_SERVER_ERROR:
        return "INVALID_REQUEST"
    return "INTERNAL_SERVER_ERROR"


def _default_message(status: int) -> str:
    messages = {
        400: "invalid request",
        401: "invalid or missing credential",
        402: "payment required",
        403: "permission denied",
        404: "not found",
        405: "method not allowed",
        408: "request timed out",
        504: "request timed out",
        413: "request too large",
        429: "rate limited",
        503: "overloaded",
        529: "overloaded",
    }
    return messages.get(status, "internal server error")


def envelope(error_type: str, message: str, request_id: str | None) -> dict[str, Any]:
    """The Anthropic error body."""
    return {"type": "error", "error": {"type": error_type, "message": message}, "request_id": request_id}


def translate_error_body(status: int, body: Any, context: ErrorContext) -> tuple[str, str, str | None]:
    """The ``(type, message, SIE code)`` an inner error answer becomes.

    ``body`` is the inner answer as JSON: the OpenAI ``{error}`` envelope, the
    SIE ``{detail}`` envelope, or anything else. Messages are rewritten only
    where Anthropic clients match the wording: the output cap, a prompt that
    does not fit, and media the model cannot read.
    """
    error_type = error_type_for_status(status)
    body = body if isinstance(body, dict) else {}
    inner = body.get("error") if isinstance(body.get("error"), dict) else None
    if inner is None and isinstance(body.get("detail"), dict):
        inner = body["detail"]
    code = inner.get("code") if inner is not None and isinstance(inner.get("code"), str) else None
    inner_type = inner.get("type") if inner is not None else None
    param = inner.get("param") if inner is not None and isinstance(inner.get("param"), str) else None
    message: str | None = None
    if inner is not None and isinstance(inner.get("message"), str):
        message = inner["message"]
    elif isinstance(body.get("detail"), str):
        message = body["detail"]
    elif isinstance(body.get("message"), str):
        message = body["message"]
    if not message:
        message = _default_message(status)
    if status != HTTPStatus.BAD_REQUEST:
        return error_type, message, code
    cap = _output_cap(message)
    if cap is not None:
        requested, limit = cap
        return (
            error_type,
            f"max_tokens: {requested} > {limit}, which is the maximum allowed number of output tokens for this model",
            code,
        )
    if code in ("context_exceeded", "INPUT_TOO_LONG") or inner_type == "context_length_exceeded":
        size = _prompt_size(message)
        if size is None:
            return error_type, "prompt is too long", code
        return error_type, f"prompt is too long: {size[0]} tokens > {size[1]} maximum", code
    if "does not support image input" in message and context.first_image is not None:
        return error_type, f"{context.first_image}: {message}", code
    if "does not support tool calling" in message:
        return error_type, f"tools: {message}", code
    message = message.replace("max_completion_tokens", "max_tokens")
    if param is not None:
        path = _anthropic_param(param)
        if not message.startswith(f"{path}:"):
            return error_type, f"{path}: {message}", code
    return error_type, message, code


def _anthropic_param(param: str) -> str:
    if param == "max_completion_tokens":
        return "max_tokens"
    if param == "stop":
        return "stop_sequences"
    if param.startswith("tools["):
        index, separator, field = param[len("tools[") :].partition("].function.")
        if separator:
            return f"tools.{index}.{'input_schema' if field == 'parameters' else field}"
    return param


def _leading_number(text: str) -> tuple[int, str] | None:
    match = _LEADING_DIGITS.match(text)
    if match is None:
        return None
    value = int(match.group())
    if value > _U64_MAX:
        return None
    return value, text[match.end() :]


def _output_cap(message: str) -> tuple[int, int] | None:
    """``(requested, cap)`` from the chat route's or the worker's output-cap refusal."""
    for prefix in ("max_completion_tokens (", "max_tokens (", "max_new_tokens ("):
        if not message.startswith(prefix):
            continue
        requested = _leading_number(message[len(prefix) :])
        if requested is None:
            return None
        rest = requested[1]
        for middle in (") exceeds model max_output_tokens (", ") exceeds model cap ("):
            if rest.startswith(middle):
                rest = rest[len(middle) :]
                break
        else:
            return None
        cap = _leading_number(rest)
        if cap is None or cap[1] != ")":
            return None
        return requested[0], cap[0]
    return None


def _prompt_size(message: str) -> tuple[int, int] | None:
    """``(tokens, limit)`` from ``... = T exceeds context_length (C) ...`` or ``prompt_tokens (T) exceeds ...``."""
    before, separator, after = message.partition(" exceeds context_length (")
    if not separator:
        return None
    limit = _leading_number(after)
    if limit is None:
        return None
    _, equals, total = before.rpartition("= ")
    if equals:
        tokens = _leading_number(total)
    elif before.startswith("prompt_tokens ("):
        tokens = _leading_number(before[len("prompt_tokens (") :])
    else:
        return None
    if tokens is None:
        return None
    return tokens[0], limit[0]


def _is_anthropic_envelope(body: Any) -> bool:
    return (
        isinstance(body, dict)
        and body.get("type") == "error"
        and isinstance(body.get("error"), dict)
        and isinstance(body["error"].get("type"), str)
    )


def anthropic_error(
    status: int, headers: Iterable[tuple[str, str]], body: bytes, context: ErrorContext
) -> tuple[list[tuple[str, str]], bytes]:
    """Re-render a non-2xx answer in the Anthropic envelope.

    Returns the answer's headers and body; the status is kept. ``Retry-After``
    is kept and capped at :data:`MAX_RETRY_AFTER_S`, the longest wait Anthropic
    clients keep retrying through, and the SIE headers are kept.
    """
    try:
        parsed = parse_json(body) if body else None
    except (ValueError, UnicodeDecodeError):
        parsed = None
    kept = [(name.lower(), value) for name, value in headers if name.lower() in _ERROR_HEADERS]
    request_id = next((value for name, value in kept if name == "x-sie-request-id"), None) or None
    if _is_anthropic_envelope(parsed):
        rendered = body
    else:
        error_type, message, code = translate_error_body(status, parsed, context)
        if code is not None and not any(name == "x-sie-error-code" for name, _ in kept):
            kept.append(("x-sie-error-code", code))
        rendered = json_text(envelope(error_type, message, request_id)).encode()
    retry_after = next((value for name, value in kept if name == "retry-after"), None)
    kept = [(name, value) for name, value in kept if name != "retry-after"]
    seconds = _retry_after_seconds(retry_after)
    if seconds is not None:
        kept = [(name, value) for name, value in kept if name != "x-should-retry"]
        kept += [("retry-after", str(seconds)), ("x-should-retry", "true")]
    kept.append(("content-type", "application/json"))
    if request_id is not None:
        kept.append(("request-id", request_id))
    return kept, rendered


def _retry_after_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    digits = text.removeprefix("+")
    if not digits or not digits.isascii() or not digits.isdigit() or int(digits) > _U64_MAX:
        return None
    return min(max(int(digits), 1), MAX_RETRY_AFTER_S)


def is_messages_path(path: str) -> bool:
    """Whether a path is ``/v1/messages`` or below it."""
    return path == "/v1/messages" or path.startswith("/v1/messages/")
