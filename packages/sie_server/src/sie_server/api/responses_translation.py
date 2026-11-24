"""OpenAI Responses translation onto the chat completions wire.

A Responses request into a chat completions body, a chat completion into a
Response, chat SSE ``data`` payloads into Responses events, and a chat error
answer into the Responses error answer. The gateway serves the same
translation, and the shared vectors under ``conformance/openai_responses/`` pin
both, so a request is accepted, refused and answered the same way on
``sie-server`` and on the gateway.

Validation happens here, in Responses terms, so a refusal names the field the
caller sent. Unknown fields fail with ``<path>: Extra inputs are not
permitted``. The accepted-but-inert set is ``store: false``, ``background:
false``, ``truncation: "disabled"``, an ``include`` of only
``reasoning.encrypted_content``, ``client_metadata``, ``reasoning.summary``,
``reasoning.generate_summary`` and ``text.verbosity``.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from sie_server.api.anthropic_translation import json_text, parse_json

_U32_MAX = (1 << 32) - 1
_ENCRYPTED_REASONING = "reasoning.encrypted_content"
_STATELESS = "SIE does not store responses; send the full conversation in input"
_ACCEPTED_FIELDS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "stream",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "text",
        "metadata",
        "prompt_cache_key",
        "user",
        "safety_identifier",
        "service_tier",
        "client_metadata",
    }
)
_PASSTHROUGH_FIELDS = ("metadata", "prompt_cache_key", "user", "safety_identifier", "service_tier")
_ITEM_KEYS = {
    "message": ("type", "id", "status", "role", "content"),
    "function_call": ("type", "id", "status", "call_id", "name", "arguments"),
    "custom_tool_call": ("type", "id", "status", "call_id", "name", "input"),
    "function_call_output": ("type", "id", "status", "call_id", "output"),
    "custom_tool_call_output": ("type", "id", "status", "call_id", "output"),
    "reasoning": ("type", "id", "status", "summary", "content", "encrypted_content"),
}
_ROLES = {"user": "user", "assistant": "assistant", "system": "system", "developer": "system"}
_TEXT_PART_KEYS = {
    "input_text": ("type", "text"),
    "text": ("type", "text"),
    "output_text": ("type", "text", "annotations", "logprobs"),
}
_IMAGE_KEYS = ("type", "image_url", "file_id", "detail")
_FUNCTION_TOOL_KEYS = ("type", "name", "description", "parameters", "strict")
_CUSTOM_TOOL_KEYS = ("type", "name", "description", "format")
_FORMAT_KEYS = {
    "text": ("type",),
    "json_object": ("type",),
    "json_schema": ("type", "name", "schema", "strict", "description"),
}
_PARAM_RENAMES = {
    "max_completion_tokens": "max_output_tokens",
    "reasoning_effort": "reasoning.effort",
    "response_format": "text.format",
}
_SCHEMA_PARAM_PREFIX = "response_format.json_schema."
_TOOL_PARAM = re.compile(r"tools\[([0-9]+)\]\.function\.(.+)")


class ResponsesError(Exception):
    """A refusal decided in Responses terms, answered in the OpenAI error envelope."""

    def __init__(self, message: str, *, param: str | None, code: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code

    def body(self) -> dict[str, Any]:
        error_type = "invalid_request_error" if self.status < HTTPStatus.INTERNAL_SERVER_ERROR else "server_error"
        return {"error": {"message": self.message, "type": error_type, "param": self.param, "code": self.code}}


@dataclass(frozen=True)
class ResponseShape:
    """What a Response repeats from the request it answers."""

    model: str
    parallel_tool_calls: Any
    tool_choice: Any
    custom_tools: frozenset[str]
    """Names of the declared custom tools, whose calls come back as ``custom_tool_call`` items."""


@dataclass(frozen=True)
class ChatTranslation:
    """A translated request."""

    chat: dict[str, Any]
    stream: bool
    shape: ResponseShape


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unsupported(param: str, reason: str) -> ResponsesError:
    return ResponsesError(f"{param}: {reason}", param=param, code="unsupported_field")


def _invalid(param: str, reason: str) -> ResponsesError:
    return ResponsesError(f"{param}: {reason}", param=param, code="invalid_request")


def _extra(param: str) -> ResponsesError:
    return _unsupported(param, "Extra inputs are not permitted")


def _required(param: str) -> ResponsesError:
    return _invalid(param, "Field required")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(path, "Input should be a valid dictionary")
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
        raise _invalid(f"{path}.{key}", "Input should be a valid string")
    return value


def _type(mapping: dict[str, Any], path: str) -> str:
    kind = mapping.get("type")
    if kind is None:
        raise _required(f"{path}.type")
    if not isinstance(kind, str):
        raise _invalid(f"{path}.type", "Input should be a valid string")
    return kind


class _Parts:
    """The content of one chat message: adjacent texts joined, empty texts dropped, images kept in order."""

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


def responses_to_chat(body: Any) -> ChatTranslation:
    """Translate an OpenAI Responses body into a chat completions body.

    Raises:
        ResponsesError: The request is refused, with the Responses wording.
    """
    if not isinstance(body, dict):
        raise _invalid("body", "Input should be a valid dictionary")
    for key, value in body.items():
        _top_level_field(key, value)

    model = body.get("model")
    if model is None:
        raise _required("model")
    if not isinstance(model, str):
        raise _invalid("model", "Input should be a valid string")
    if not model:
        raise _invalid("model", "String should have at least 1 character")
    raw_input = body.get("input")
    if raw_input is None:
        raise _required("input")
    if not isinstance(raw_input, str | list):
        raise _invalid("input", "Input should be a valid string or list")
    if not raw_input and isinstance(raw_input, list):
        raise _invalid("input", "List should have at least 1 item")
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise _invalid("instructions", "Input should be a valid string")
    max_output_tokens = _max_output_tokens(body.get("max_output_tokens"))
    stream = body.get("stream")
    if stream is None:
        stream = False
    elif not isinstance(stream, bool):
        raise _invalid("stream", "Input should be a valid boolean")
    effort = _reasoning_effort(body.get("reasoning"))
    response_format = _response_format(body.get("text"))
    tools, names, custom_tools = _tools(body.get("tools"))
    tool_choice = _tool_choice(body.get("tool_choice"), names)
    items: list[Any] = [{"role": "user", "content": raw_input}] if isinstance(raw_input, str) else raw_input
    conversation = _Conversation(instructions)
    for index, item in enumerate(items):
        conversation.add(item, f"input[{index}]")

    chat: dict[str, Any] = {"model": model, "messages": conversation.messages()}
    if max_output_tokens is not None:
        chat["max_completion_tokens"] = max_output_tokens
    for key in ("temperature", "top_p", "seed"):
        if body.get(key) is not None:
            chat[key] = body[key]
    chat["stream"] = stream
    if stream:
        chat["stream_options"] = {"include_usage": True}
    if tools:
        chat["tools"] = tools
    if tool_choice is not None:
        chat["tool_choice"] = tool_choice
    parallel_tool_calls = body.get("parallel_tool_calls")
    if parallel_tool_calls is not None:
        chat["parallel_tool_calls"] = parallel_tool_calls
    if response_format is not None:
        chat["response_format"] = response_format
    for key in _PASSTHROUGH_FIELDS:
        if body.get(key) is not None:
            chat[key] = body[key]
    chat["include_reasoning"] = False
    if effort is not None:
        chat["reasoning_effort"] = effort
    requested_choice = body.get("tool_choice")
    shape = ResponseShape(
        model=model,
        parallel_tool_calls=True if parallel_tool_calls is None else parallel_tool_calls,
        tool_choice="auto" if requested_choice is None else requested_choice,
        custom_tools=frozenset(custom_tools),
    )
    return ChatTranslation(chat=chat, stream=stream, shape=shape)


def _top_level_field(key: str, value: Any) -> None:
    if key in _ACCEPTED_FIELDS:
        return
    if key in ("store", "background"):
        if value is None or value is False:
            return
        if value is not True:
            raise _invalid(key, "Input should be a valid boolean")
        if key == "store":
            raise _unsupported(key, _STATELESS)
        raise _unsupported(key, "background responses are not supported")
    if key in ("previous_response_id", "conversation"):
        if value is not None:
            raise _unsupported(key, _STATELESS)
        return
    if key == "truncation":
        if value is not None and value != "disabled":
            raise _unsupported(key, 'only "disabled" is supported')
        return
    if key == "include":
        _include(value)
        return
    raise _extra(key)


def _include(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise _invalid("include", "Input should be a valid list")
    for index, entry in enumerate(value):
        if entry != _ENCRYPTED_REASONING:
            shown = entry if isinstance(entry, str) else json_text(entry)
            raise _unsupported(f"include[{index}]", f"'{shown}' is not supported")


def _max_output_tokens(value: Any) -> int | None:
    if value is None:
        return None
    if _is_int(value) and 1 <= value <= _U32_MAX:
        return value
    if _is_int(value) and value < 1:
        raise _invalid("max_output_tokens", "Input should be greater than or equal to 1")
    raise _invalid("max_output_tokens", "Input should be a valid integer")


def _reasoning_effort(value: Any) -> Any:
    if value is None:
        return None
    reasoning = _object(value, "reasoning")
    _only_keys(reasoning, ("effort", "summary", "generate_summary"), "reasoning")
    return reasoning.get("effort")


def _response_format(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    text = _object(value, "text")
    _only_keys(text, ("format", "verbosity"), "text")
    if text.get("format") is None:
        return None
    fmt = _object(text["format"], "text.format")
    kind = _type(fmt, "text.format")
    if kind not in _FORMAT_KEYS:
        raise _unsupported("text.format.type", f"'{kind}' is not supported")
    _only_keys(fmt, _FORMAT_KEYS[kind], "text.format")
    if kind == "text":
        return None
    if kind == "json_object":
        return {"type": "json_object"}
    schema = {key: fmt[key] for key in ("name", "schema", "strict", "description") if fmt.get(key) is not None}
    return {"type": "json_schema", "json_schema": schema}


def _custom_tool_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
        "additionalProperties": False,
    }


def _tools(value: Any) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """The chat tools, the declared tool names, and the names of the custom tools."""
    if value is None:
        return [], [], []
    if not isinstance(value, list):
        raise _invalid("tools", "Input should be a valid list")
    tools: list[dict[str, Any]] = []
    names: list[str] = []
    custom: list[str] = []
    for index, raw in enumerate(value):
        path = f"tools[{index}]"
        tool = _object(raw, path)
        kind = _type(tool, path)
        if kind not in ("function", "custom"):
            raise _unsupported(f"{path}.type", f"'{kind}' tools are not supported; send function or custom tools")
        if kind == "function":
            _only_keys(tool, _FUNCTION_TOOL_KEYS, path)
            strict = tool.get("strict")
            if strict is True:
                raise _unsupported(f"{path}.strict", "strict function schemas are not supported")
            if strict is not None and strict is not False:
                raise _invalid(f"{path}.strict", "Input should be a valid boolean")
        else:
            _only_keys(tool, _CUSTOM_TOOL_KEYS, path)
        function = {key: tool[key] for key in ("name", "description") if tool.get(key) is not None}
        if kind == "custom":
            function["parameters"] = _custom_tool_parameters()
        elif tool.get("parameters") is not None:
            function["parameters"] = tool["parameters"]
        tools.append({"type": "function", "function": function})
        name = tool.get("name")
        if isinstance(name, str):
            names.append(name)
            if kind == "custom":
                custom.append(name)
    return tools, names, custom


def _tool_choice(value: Any, names: list[str]) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in ("auto", "none", "required"):
            raise _invalid("tool_choice", "Input should be 'auto', 'none' or 'required'")
        if not names and value in ("auto", "none"):
            return None
        return value
    if not isinstance(value, dict):
        raise _invalid("tool_choice", "Input should be a valid string or dictionary")
    kind = _type(value, "tool_choice")
    if kind not in ("function", "custom"):
        raise _unsupported("tool_choice.type", f"'{kind}' is not supported")
    _only_keys(value, ("type", "name"), "tool_choice")
    name = _string(value, "name", "tool_choice")
    if name not in names:
        raise _invalid("tool_choice.name", f"tool '{name}' is not declared in tools")
    return {"type": "function", "function": {"name": name}}


class _Conversation:
    """The chat messages of a Responses ``input``, built one item at a time.

    A run of assistant output (assistant messages and tool calls) becomes one
    chat assistant message. The images of a run of tool outputs become one chat
    user message after them, joined by the user message that follows the run.
    """

    def __init__(self, instructions: str | None) -> None:
        self._out: list[dict[str, Any]] = []
        self._system: list[str] = [instructions] if instructions else []
        self._leading = True
        self._assistant_text: list[str] | None = None
        self._tool_calls: list[dict[str, Any]] = []
        self._tool_images: list[str] | None = None

    def messages(self) -> list[dict[str, Any]]:
        self._flush_assistant()
        self._flush_tool_images()
        if self._system:
            return [{"role": "system", "content": "\n\n".join(self._system)}, *self._out]
        return list(self._out)

    def add(self, raw: Any, path: str) -> None:
        item = _object(raw, path)
        kind = item.get("type")
        if kind is None:
            if "role" not in item:
                raise _required(f"{path}.role")
            kind = "message"
        if not isinstance(kind, str):
            raise _invalid(f"{path}.type", "Input should be a valid string")
        if kind not in _ITEM_KEYS:
            raise _unsupported(f"{path}.type", f"items of type '{kind}' are not supported")
        _only_keys(item, _ITEM_KEYS[kind], path)
        if kind == "reasoning":
            return
        if kind == "message":
            self._message(item, path)
        elif kind in ("function_call", "custom_tool_call"):
            self._tool_call(item, kind, path)
        else:
            self._tool_output(item, path)

    def _message(self, item: dict[str, Any], path: str) -> None:
        role = item.get("role")
        if role is None:
            raise _required(f"{path}.role")
        if not isinstance(role, str) or role not in _ROLES:
            raise _invalid(f"{path}.role", "Input should be 'user', 'assistant', 'system' or 'developer'")
        parts = _content(item, role, path)
        if _ROLES[role] == "system":
            text = "".join(value for _, value in parts.items)
            if self._leading:
                self._system.append(text)
                return
            self._flush_assistant()
            self._flush_tool_images()
            self._out.append({"role": "system", "content": text})
            return
        self._leading = False
        if role == "assistant":
            self._flush_tool_images()
            if self._assistant_text is None:
                self._assistant_text = []
            self._assistant_text.append("".join(value for _, value in parts.items))
            return
        self._flush_assistant()
        if self._tool_images:
            merged = _Parts()
            for url in self._tool_images:
                merged.image(url)
            merged.items.extend(parts.items)
            parts = merged
        self._tool_images = None
        self._out.append({"role": "user", "content": parts.content()})

    def _tool_call(self, item: dict[str, Any], kind: str, path: str) -> None:
        self._leading = False
        self._flush_tool_images()
        call_id = _string(item, "call_id", path)
        name = _string(item, "name", path)
        if kind == "function_call":
            arguments = _string(item, "arguments", path)
        else:
            arguments = json_text({"input": _string(item, "input", path)})
        if self._assistant_text is None:
            self._assistant_text = []
        self._tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})

    def _tool_output(self, item: dict[str, Any], path: str) -> None:
        self._leading = False
        self._flush_assistant()
        call_id = _string(item, "call_id", path)
        text, images = _tool_output(item.get("output"), path)
        self._out.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if self._tool_images is None:
            self._tool_images = []
        self._tool_images.extend(images)

    def _flush_assistant(self) -> None:
        if self._assistant_text is None:
            return
        text = "".join(self._assistant_text)
        if self._tool_calls:
            self._out.append({"role": "assistant", "content": text or None, "tool_calls": self._tool_calls})
        else:
            self._out.append({"role": "assistant", "content": text})
        self._assistant_text = None
        self._tool_calls = []

    def _flush_tool_images(self) -> None:
        if self._tool_images:
            parts = _Parts()
            for url in self._tool_images:
                parts.image(url)
            self._out.append({"role": "user", "content": parts.content()})
        self._tool_images = None


def _content(item: dict[str, Any], role: str, path: str) -> _Parts:
    content = item.get("content")
    if content is None:
        raise _required(f"{path}.content")
    parts = _Parts()
    if isinstance(content, str):
        parts.text(content)
        return parts
    if not isinstance(content, list):
        raise _invalid(f"{path}.content", "Input should be a valid string or list")
    for index, raw in enumerate(content):
        part_path = f"{path}.content[{index}]"
        part = _object(raw, part_path)
        kind = _type(part, part_path)
        if kind in _TEXT_PART_KEYS:
            _only_keys(part, _TEXT_PART_KEYS[kind], part_path)
            parts.text(_string(part, "text", part_path))
        elif kind == "input_image":
            if role != "user":
                raise _unsupported(
                    f"{part_path}.type", "parts of type 'input_image' are only supported in user messages"
                )
            parts.image(_image(part, part_path))
        else:
            raise _unsupported(f"{part_path}.type", f"parts of type '{kind}' are not supported")
    return parts


def _image(part: dict[str, Any], path: str) -> str:
    _only_keys(part, _IMAGE_KEYS, path)
    if part.get("file_id") is not None:
        raise _unsupported(f"{path}.file_id", "only base64 data: URLs are supported")
    url = part.get("image_url")
    if url is None:
        raise _required(f"{path}.image_url")
    if not isinstance(url, str):
        raise _invalid(f"{path}.image_url", "Input should be a valid string")
    if not url.startswith("data:"):
        raise _unsupported(f"{path}.image_url", "only base64 data: URLs are supported")
    return url


def _tool_output(output: Any, path: str) -> tuple[str, list[str]]:
    if output is None:
        raise _required(f"{path}.output")
    if isinstance(output, str):
        return output, []
    if not isinstance(output, list):
        raise _invalid(f"{path}.output", "Input should be a valid string or list")
    texts: list[str] = []
    images: list[str] = []
    for index, raw in enumerate(output):
        part_path = f"{path}.output[{index}]"
        part = _object(raw, part_path)
        kind = _type(part, part_path)
        if kind in _TEXT_PART_KEYS:
            _only_keys(part, _TEXT_PART_KEYS[kind], part_path)
            texts.append(_string(part, "text", part_path))
        elif kind == "input_image":
            images.append(_image(part, part_path))
        else:
            raise _unsupported(f"{part_path}.type", f"parts of type '{kind}' are not supported")
    return "".join(texts), images


def custom_tool_input(arguments: str) -> str:
    """The input of a custom-tool call: ``arguments.input``, or the raw arguments when they carry none."""
    try:
        parsed = parse_json(arguments)
    except ValueError:
        return arguments
    if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
        return parsed["input"]
    return arguments


def responses_usage(usage: Any) -> dict[str, Any] | None:
    """Responses usage from a chat usage block."""
    if not isinstance(usage, dict):
        return None
    counts = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
    if not all(_is_int(count) for count in counts):
        return None
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    reasoning = completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None
    return {
        "input_tokens": counts[0],
        "input_tokens_details": {"cached_tokens": cached if _is_int(cached) else 0},
        "output_tokens": counts[1],
        "output_tokens_details": {"reasoning_tokens": reasoning if _is_int(reasoning) else 0},
        "total_tokens": counts[2],
    }


def _status(finish_reason: Any) -> tuple[str, dict[str, str] | None]:
    if finish_reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if finish_reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    return "completed", None


def response_object(
    shape: ResponseShape,
    request_id: str,
    created_at: int,
    *,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    incomplete_details: dict[str, str] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A Response object for the request ``shape`` describes."""
    return {
        "id": f"resp-{request_id}",
        "object": "response",
        "created_at": created_at,
        "model": shape.model,
        "status": status,
        "incomplete_details": incomplete_details,
        "error": error,
        "store": False,
        "parallel_tool_calls": shape.parallel_tool_calls,
        "tool_choice": shape.tool_choice,
        "output": output,
        "usage": usage,
    }


def _message_item(item_id: str, text: str | None) -> dict[str, Any]:
    """A message item, ``in_progress`` and empty while ``text`` is None."""
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "in_progress" if text is None else "completed",
        "content": [] if text is None else [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call_item(item_id: str, call_id: str, name: str, value: str, *, custom: bool, done: bool) -> dict[str, Any]:
    if custom:
        payload = {"type": "custom_tool_call", "id": item_id, "call_id": call_id, "name": name, "input": value}
    else:
        payload = {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": value}
    return {**payload, "status": "completed" if done else "in_progress"}


def _call_item_id(request_id: str, position: int, *, custom: bool) -> str:
    return f"{'ctc' if custom else 'fc'}-{request_id}-{position}"


def chat_completion_to_response(chat: Any, shape: ResponseShape, request_id: str, created_at: int) -> dict[str, Any]:
    """Translate a chat completion into a Response.

    ``request_id`` names the Response and its items; ``created_at`` is its
    creation time in Unix seconds.

    Raises:
        ValueError: The chat completion cannot be translated.
    """
    choices = chat.get("choices") if isinstance(chat, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat completion has no choices")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("chat completion choice has no message")
    message = choice["message"]
    output: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(_message_item(f"msg-{request_id}", content))
    tool_calls = message.get("tool_calls")
    for position, call in enumerate(tool_calls if isinstance(tool_calls, list) else []):
        function = call.get("function") if isinstance(call, dict) else None
        call_id = call.get("id") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise ValueError("tool call has no id or name")
        arguments = function.get("arguments") if isinstance(function, dict) else None
        arguments = arguments if isinstance(arguments, str) else ""
        custom = name in shape.custom_tools
        value = custom_tool_input(arguments) if custom else arguments
        item_id = _call_item_id(request_id, position, custom=custom)
        output.append(_call_item(item_id, call_id, name, value, custom=custom, done=True))
    status, incomplete = _status(choice.get("finish_reason"))
    return response_object(
        shape,
        request_id,
        created_at,
        status=status,
        incomplete_details=incomplete,
        output=output,
        usage=responses_usage(chat.get("usage")),
    )


@dataclass(frozen=True)
class Event:
    """One Responses SSE event."""

    name: str
    data: dict[str, Any]

    def encode(self) -> bytes:
        return f"event: {self.name}\ndata: {json_text(self.data)}\n\n".encode()


@dataclass
class _OpenText:
    output_index: int
    item_id: str
    text: str = ""


@dataclass
class _OpenCall:
    chat_index: int
    position: int
    output_index: int
    item_id: str
    call_id: str
    name: str
    custom: bool
    arguments: str = ""


class StreamMachine:
    """The chat-chunk to Responses-event state machine.

    One output item is open at a time and output indices are contiguous from 0.
    """

    def __init__(self, shape: ResponseShape, request_id: str, created_at: int) -> None:
        self._shape = shape
        self._request_id = request_id
        self._created_at = created_at
        self._sequence = 0
        self._next_index = 0
        self._output: list[dict[str, Any]] = []
        self._text: _OpenText | None = None
        self._call: _OpenCall | None = None
        self._calls = 0
        self._closed_calls: set[int] = set()
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None
        self._started = False
        self.done = False
        """Whether the terminal event has been produced."""

    def start(self) -> list[Event]:
        """``response.created`` and ``response.in_progress``, sent as soon as the chat stream opens."""
        if self._started:
            return []
        self._started = True
        return [
            self._event("response.created", response=self._response("in_progress")),
            self._event("response.in_progress", response=self._response("in_progress")),
        ]

    def on_data(self, data: str) -> list[Event]:
        """Feed one chat SSE ``data`` payload."""
        events = self.start()
        if self.done:
            return events
        if data == "[DONE]":
            self._terminal(events)
            return events
        try:
            chunk = parse_json(data)
        except ValueError:
            self._fail(events, "server_error", "the generation stream sent a malformed chunk")
            return events
        if not isinstance(chunk, dict):
            chunk = {}
        error = chunk.get("error")
        if isinstance(error, dict):
            self._fail(events, error.get("code"), error.get("message"))
            return events
        usage = responses_usage(chunk.get("usage"))
        if usage is not None:
            self._usage = usage
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return events
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        text = delta.get("content")
        if isinstance(text, str) and text:
            self._text_delta(text, events)
        calls = delta.get("tool_calls")
        for call in calls if isinstance(calls, list) else []:
            problem = self._call_delta(call, events)
            if problem is not None:
                self._fail(events, "server_error", problem)
                return events
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            self._finish_reason = finish_reason
            self._close_text(events)
            self._close_call(events)
        return events

    def on_end(self) -> list[Event]:
        """The chat stream ended. A stream that reported a finish completes; any other ending fails."""
        events = self.start()
        if not self.done:
            self._terminal(events)
        return events

    def on_transport_error(self) -> list[Event]:
        """The chat stream broke before it ended."""
        events = self.start()
        if not self.done:
            self._fail(events, "server_error", "the generation stream ended unexpectedly")
        return events

    def _event(self, name: str, **payload: Any) -> Event:
        event = Event(name, {"type": name, "sequence_number": self._sequence, **payload})
        self._sequence += 1
        return event

    def _response(self, status: str, **fields: Any) -> dict[str, Any]:
        output = list(self._output) if status != "in_progress" else []
        return response_object(self._shape, self._request_id, self._created_at, status=status, output=output, **fields)

    def _terminal(self, events: list[Event]) -> None:
        if self._finish_reason is None:
            self._fail(events, "server_error", "the generation stream ended without a finish reason")
            return
        self._close_text(events)
        self._close_call(events)
        status, incomplete = _status(self._finish_reason)
        name = "response.completed" if status == "completed" else "response.incomplete"
        response = self._response(status, incomplete_details=incomplete, usage=self._usage)
        events.append(self._event(name, response=response))
        self.done = True

    def _fail(self, events: list[Event], code: Any, message: Any) -> None:
        response = self._response("failed", error={"code": code, "message": message})
        events.append(self._event("response.failed", response=response))
        self.done = True

    def _text_delta(self, text: str, events: list[Event]) -> None:
        self._close_call(events)
        if self._text is None:
            index = self._next_index
            self._next_index += 1
            item_id = f"msg-{self._request_id}" if self._calls == 0 else f"msg-{self._request_id}-{index}"
            self._text = _OpenText(index, item_id)
            events.append(
                self._event("response.output_item.added", output_index=index, item=_message_item(item_id, None))
            )
            events.append(
                self._event(
                    "response.content_part.added",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    part={"type": "output_text", "text": "", "annotations": []},
                )
            )
        self._text.text += text
        events.append(
            self._event(
                "response.output_text.delta",
                item_id=self._text.item_id,
                output_index=self._text.output_index,
                content_index=0,
                delta=text,
            )
        )

    def _close_text(self, events: list[Event]) -> None:
        if self._text is None:
            return
        opened, self._text = self._text, None
        item_id, index, text = opened.item_id, opened.output_index, opened.text
        part = {"type": "output_text", "text": text, "annotations": []}
        item = _message_item(item_id, text)
        events.append(
            self._event("response.output_text.done", item_id=item_id, output_index=index, content_index=0, text=text)
        )
        events.append(
            self._event("response.content_part.done", item_id=item_id, output_index=index, content_index=0, part=part)
        )
        events.append(self._event("response.output_item.done", output_index=index, item=item))
        self._output.append(item)

    def _call_delta(self, call: Any, events: list[Event]) -> str | None:
        call = call if isinstance(call, dict) else {}
        index = call.get("index")
        chat_index = index if isinstance(index, int) and not isinstance(index, bool) else 0
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        if self._call is None or self._call.chat_index != chat_index:
            if chat_index in self._closed_calls:
                return "tool call arguments arrived after the call closed"
            call_id = call.get("id")
            name = function.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                return "a tool call arrived without its id and name"
            self._close_text(events)
            self._close_call(events)
            custom = name in self._shape.custom_tools
            item_id = _call_item_id(self._request_id, self._calls, custom=custom)
            self._call = _OpenCall(chat_index, self._calls, self._next_index, item_id, call_id, name, custom)
            self._calls += 1
            self._next_index += 1
            item = _call_item(item_id, call_id, name, "", custom=custom, done=False)
            events.append(self._event("response.output_item.added", output_index=self._call.output_index, item=item))
        fragment = function.get("arguments")
        if not isinstance(fragment, str) or not fragment:
            return None
        self._call.arguments += fragment
        if not self._call.custom:
            events.append(
                self._event(
                    "response.function_call_arguments.delta",
                    item_id=self._call.item_id,
                    output_index=self._call.output_index,
                    delta=fragment,
                )
            )
        return None

    def _close_call(self, events: list[Event]) -> None:
        if self._call is None:
            return
        opened, self._call = self._call, None
        item_id, index = opened.item_id, opened.output_index
        if opened.custom:
            value = custom_tool_input(opened.arguments)
            events.append(
                self._event("response.custom_tool_call_input.delta", item_id=item_id, output_index=index, delta=value)
            )
            events.append(
                self._event("response.custom_tool_call_input.done", item_id=item_id, output_index=index, input=value)
            )
        else:
            value = opened.arguments
            events.append(
                self._event(
                    "response.function_call_arguments.done", item_id=item_id, output_index=index, arguments=value
                )
            )
        item = _call_item(item_id, opened.call_id, opened.name, value, custom=opened.custom, done=True)
        events.append(self._event("response.output_item.done", output_index=index, item=item))
        self._output.append(item)
        self._closed_calls.add(opened.chat_index)


class FrameParser:
    """Splits an SSE byte stream into its events.

    Each event is ``("data", payload)``, or ``("comment", frame)`` for an event
    that carries only comments, such as a keep-alive.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def push(self, chunk: bytes) -> list[tuple[str, str]]:
        self._buffer.extend(chunk.replace(b"\r", b""))
        frames: list[tuple[str, str]] = []
        while (end := self._buffer.find(b"\n\n")) >= 0:
            text = bytes(self._buffer[:end]).decode("utf-8", errors="replace")
            del self._buffer[: end + 2]
            lines = text.split("\n")
            data = [line[len("data:") :].removeprefix(" ") for line in lines if line.startswith("data:")]
            if data:
                frames.append(("data", "\n".join(data)))
                continue
            comments = [line for line in lines if line.startswith(":")]
            if comments:
                frames.append(("comment", "\n".join(comments) + "\n\n"))
        return frames


async def stream_events(chunks: AsyncIterator[bytes], machine: StreamMachine) -> AsyncIterator[bytes]:
    """The Responses SSE body for a chat SSE body.

    The chat body's comments pass through. The terminal event ends the body
    and closes ``chunks``, so a stream that failed stops generating, as a
    client that goes away does.
    """
    parser = FrameParser()
    try:
        for event in machine.start():
            yield event.encode()
        while not machine.done:
            try:
                chunk = await anext(chunks)
            except StopAsyncIteration:
                for event in machine.on_end():
                    yield event.encode()
                return
            except Exception:  # noqa: BLE001 - any failure of the chat body fails the response
                for event in machine.on_transport_error():
                    yield event.encode()
                return
            for kind, frame in parser.push(chunk):
                if kind == "comment":
                    yield frame.encode()
                    continue
                for event in machine.on_data(frame):
                    yield event.encode()
                if machine.done:
                    return
    finally:
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()


def responses_param(param: str) -> str:
    """The Responses field a chat error ``param`` names."""
    renamed = _PARAM_RENAMES.get(param)
    if renamed is not None:
        return renamed
    if param.startswith(_SCHEMA_PARAM_PREFIX) and len(param) > len(_SCHEMA_PARAM_PREFIX):
        return f"text.format.{param[len(_SCHEMA_PARAM_PREFIX) :]}"
    match = _TOOL_PARAM.fullmatch(param)
    if match is not None:
        return f"tools[{match.group(1)}].{match.group(2)}"
    return param


def responses_error(headers: Iterable[tuple[str, str]], body: bytes) -> tuple[list[tuple[str, str]], bytes]:
    """A chat error answer as the Responses route returns it, at the same status.

    The headers are kept, ``Retry-After`` and ``x-sie-error-code`` among them,
    and so is the body, except that a ``param`` naming a chat field is renamed
    to the Responses field the caller sent.
    """
    kept = [(name.lower(), value) for name, value in headers if name.lower() != "content-length"]
    try:
        parsed = parse_json(body)
    except (ValueError, UnicodeDecodeError):
        return kept, body
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(error, dict) or not isinstance(error.get("param"), str):
        return kept, body
    renamed = responses_param(error["param"])
    if renamed == error["param"]:
        return kept, body
    return kept, json_text({**parsed, "error": {**error, "param": renamed}}).encode()
