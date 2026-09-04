"""Direct OpenAI-compatible surfaces: ``/v1/chat/completions`` + ``/v1/rerank``.

These complete the single-node OpenAI surface (alongside the existing
``/v1/embeddings``), so OpenAI/Cohere clients can use the worker HTTP server
directly in either the Apple-Silicon or CUDA image.

- ``/v1/chat/completions`` proxies the managed generation subprocess: MLX on
  Apple Silicon and SGLang on CUDA. Chat templating, streaming, tool parsing,
  and structured output remain owned by that already-loaded child. The proxy
  pins the request to the child's loopback URL and served model identity; a
  client cannot select another upstream.
- ``/v1/rerank`` wraps the in-process score adapter in the Cohere/OpenAI rerank
  shape (``{query, documents, top_n}`` -> ``{results: [{index, relevance_score}]}``).

These routes are mounted on the worker's own FastAPI app as a direct-container /
single-node convenience, mirroring the existing direct ``/v1/generate`` and
``/v1/embeddings`` routes. The Rust gateway remains the production cluster API
authority.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sie_sdk.queue_types import denormalize_model_id

from sie_server.adapters._generation_base import (
    ReasoningFormat,
    ThinkingBlockStripper,
    aclose_with_error_precedence,
    resolve_reasoning_format,
    thinking_blocks_must_be_hidden,
)
from sie_server.adapters.mlx.generation import MLXGenerationAdapter, normalize_mlx_seed
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.api.helpers import ModelStateChecker, ensure_finite_scores, openai_error_response
from sie_server.api.options import resolve_runtime_options
from sie_server.api.score import score_usage_from_output
from sie_server.api.validation import validate_machine_profile_header, validate_signed_i64
from sie_server.config.model import validate_chat_template_kwargs
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.score_cost import MAX_SCORE_ITEMS, build_score_prepared_items
from sie_server.core.timing import RequestTiming
from sie_server.observability.tracing import tracer
from sie_server.processors.streaming import _decode_data_uri_image
from sie_server.types.inputs import Item
from sie_server.types.responses import ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

# Inter-chunk read timeout — bounded so a wedged child cannot hang the direct
# ingress forever. Keep the established MLX override name for compatibility;
# it applies to either managed generation child on this route.
_PROXY_READ_TIMEOUT_S = float(os.environ.get("SIE_MLX_READ_TIMEOUT_S", "300"))
_PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=_PROXY_READ_TIMEOUT_S, write=10.0, pool=10.0)
# Cap the request body so a huge messages/documents array can't be buffered/forwarded
# unbounded (the gateway caps prod ingress; these local routes must cap themselves).
_MAX_BODY_BYTES = int(os.environ.get("SIE_CHAT_MAX_BODY_BYTES", str(8 * 1024 * 1024)))
# Bound even a misbehaving local child independently from its token limit. The
# token cap below remains authoritative for normal output; this is a final byte
# fence for malformed/non-token responses.
_MAX_CHAT_RESPONSE_BYTES = int(os.environ.get("SIE_CHAT_MAX_RESPONSE_BYTES", str(32 * 1024 * 1024)))
_MAX_CHAT_MESSAGES = 4096
_MAX_CHAT_CHOICES = 128
_MAX_U32 = (1 << 32) - 1
_ALLOWED_CHAT_ROLES = frozenset({"system", "user", "assistant", "tool", "developer"})
# Compatibility requests project onto the same native score bound.
_MAX_RERANK_DOCS = MAX_SCORE_ITEMS

# Mirror the gateway's public chat field boundary, minus controls that require
# gateway-owned admission/routing translation (best_of, LoRA selection, cache
# keys). Most importantly this excludes SGLang's backend/DP/custom-processor
# extensions, so the public body cannot steer the managed child or execute a
# client-supplied logit processor.
_CUDA_CHAT_FIELDS = frozenset(
    {
        "model",
        "messages",
        "max_completion_tokens",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "top_k",
        "repetition_penalty",
        "min_tokens",
        "chat_template_kwargs",
        "n",
        "user",
        "safety_identifier",
        "seed",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "response_format",
        "stream",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream_options",
    }
)

# mlx_lm's public chat request fields, excluding child process controls such as
# model adapters and speculative-decoding configuration. Validate this boundary
# before a potentially expensive MLX model download.
_MLX_CHAT_FIELDS = frozenset(
    {
        "model",
        "messages",
        "max_completion_tokens",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "repetition_context_size",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "seed",
        "chat_template_kwargs",
        "stop",
        "stream",
        "stream_options",
        "tools",
        "role_mapping",
    }
)

_DEFAULT_SAMPLING_TO_CHAT_FIELD = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "min_new_tokens": "min_tokens",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty",
    "seed": "seed",
    "sampling_seed": "seed",
}

# ``mlx_lm.server`` accepts this subset of model-config sampling defaults on
# its OpenAI chat route. Keep parity with the MLX adapter's native generation
# path and do not forward CUDA-only penalty knobs.
_MLX_DEFAULT_SAMPLING_FIELDS = frozenset(
    {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed", "sampling_seed"}
)


def _bad_request(message: str, *, param: str | None = None, code: str | None = None) -> HTTPException:
    detail: dict[str, Any] = {"code": code or ErrorCode.INVALID_INPUT.value, "message": message}
    if param is not None:
        detail["param"] = param
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _validate_chat_seed(value: Any) -> int | None:
    """Validate the public signed-i64 seed without changing its bit pattern."""
    try:
        return validate_signed_i64(value, param="seed")
    except ValueError as exc:
        raise _bad_request(str(exc), param="seed") from exc


def _validate_mlx_seed(value: Any) -> int | None:
    """Validate the public signed-i64 seed and convert it for MLX's uint64 API."""
    validated = _validate_chat_seed(value)
    return None if validated is None else normalize_mlx_seed(validated)


async def _read_json_body(http_request: Request) -> dict[str, Any]:
    """Size-cap + parse the request body as a JSON object.

    These routes are direct local ingress (no gateway in front), so they bound the
    buffered body themselves: reject early on Content-Length, then on the real read.
    """
    clen = http_request.headers.get("content-length")
    if clen is not None and clen.isdigit() and int(clen) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"code": ErrorCode.INPUT_TOO_LONG.value, "message": "request body too large"},
        )
    raw = bytearray()
    async for chunk in http_request.stream():
        if len(raw) + len(chunk) > _MAX_BODY_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={"code": ErrorCode.INPUT_TOO_LONG.value, "message": "request body too large"},
            )
        raw.extend(chunk)
    try:
        body = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise _bad_request("request body must be a JSON object") from exc
    if not isinstance(body, dict):
        raise _bad_request("request body must be a JSON object")
    return body


def _unsupported_chat(model: str, device: object) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "code": "unsupported_operation",
            "message": (
                f"/v1/chat/completions for '{model}' has no supported managed generation "
                f"child on device '{device}'. Use an MLX or SGLang generation profile."
            ),
        },
    )


def _upstream_error_response(
    message: str = "upstream generation request failed",
    *,
    response_status: int = status.HTTP_502_BAD_GATEWAY,
) -> JSONResponse:
    return JSONResponse(
        status_code=response_status,
        content={
            "error": {
                "message": message,
                "type": "upstream_error",
                "param": None,
                "code": "upstream_error",
            }
        },
    )


def _translated_upstream_status(upstream_status: int) -> int:
    """Preserve child request errors; collapse engine failures to 502."""
    if status.HTTP_400_BAD_REQUEST <= upstream_status < status.HTTP_500_INTERNAL_SERVER_ERROR:
        return upstream_status
    return status.HTTP_502_BAD_GATEWAY


def _upstream_error_event(message: str = "upstream stream error") -> bytes:
    payload = json.dumps(
        {
            "error": {
                "message": message,
                "type": "upstream_error",
                "param": None,
                "code": "upstream_error",
            }
        },
        separators=(",", ":"),
    )
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()


def _validate_optional_positive_int(body: dict[str, Any], field: str) -> int | None:
    value = body.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= _MAX_U32):
        raise _bad_request(f"'{field}' must be a positive integer", param=field)
    return value


def _validate_chat_messages(messages: list[Any]) -> None:
    """Reject malformed message structure before loading a model child."""
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        if not isinstance(message, dict):
            raise _bad_request(f"'{path}' must be an object", param=path)

        role = message.get("role")
        if not isinstance(role, str) or role not in _ALLOWED_CHAT_ROLES:
            raise _bad_request(
                f"'{path}.role' must be one of {sorted(_ALLOWED_CHAT_ROLES)!r}",
                param=f"{path}.role",
            )

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise _bad_request(f"'{path}.tool_calls' must be an array", param=f"{path}.tool_calls")
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    call_path = f"{path}.tool_calls[{call_index}]"
                    raise _bad_request(f"'{call_path}' must be an object", param=call_path)

        tool_call_id = message.get("tool_call_id")
        if role == "tool":
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise _bad_request(
                    f"'{path}.tool_call_id' is required and must be a non-empty string",
                    param=f"{path}.tool_call_id",
                )
        elif tool_call_id is not None:
            raise _bad_request(
                f"'{path}.tool_call_id' is only valid on role 'tool' messages",
                param=f"{path}.tool_call_id",
            )

        content = message.get("content")
        if content is None and role == "assistant" and tool_calls is not None:
            continue
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise _bad_request(
                f"'{path}.content' must be a string or content-part array",
                param=f"{path}.content",
            )
        for part_index, part in enumerate(content):
            part_path = f"{path}.content[{part_index}]"
            if not isinstance(part, dict):
                raise _bad_request(f"'{part_path}' must be an object", param=part_path)
            part_obj = cast("dict[str, Any]", part)
            part_type = part_obj.get("type")
            if part_type in {"text", "input_text"}:
                if not isinstance(part_obj.get("text"), str):
                    raise _bad_request(
                        f"'{part_path}.text' is required and must be a string",
                        param=f"{part_path}.text",
                    )
            elif part_type in {"image_url", "input_image"}:
                if "image_url" not in part_obj:
                    raise _bad_request(
                        f"'{part_path}.image_url' is required for image content parts",
                        param=f"{part_path}.image_url",
                    )
            else:
                raise _bad_request(
                    f"unsupported content part type {part_type!r}",
                    param=f"{part_path}.type",
                    code="unsupported_field",
                )


def _validate_cuda_chat_body(body: dict[str, Any]) -> None:
    unknown_fields = body.keys() - _CUDA_CHAT_FIELDS
    if unknown_fields:
        unknown = sorted(unknown_fields)[0]
        raise _bad_request(f"field '{unknown}' is not supported", param=unknown, code="unsupported_field")

    n = body.get("n")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= _MAX_CHAT_CHOICES)):
        raise _bad_request(f"'n' must be an integer in [1, {_MAX_CHAT_CHOICES}]", param="n")

    min_tokens = body.get("min_tokens")
    if min_tokens is not None and (
        isinstance(min_tokens, bool) or not isinstance(min_tokens, int) or not (0 <= min_tokens <= _MAX_U32)
    ):
        raise _bad_request("'min_tokens' must be an integer >= 0", param="min_tokens")

    for field, lower, upper, lower_inclusive in (
        ("temperature", 0.0, math.inf, True),
        ("top_p", 0.0, 1.0, False),
        ("frequency_penalty", -2.0, 2.0, True),
        ("presence_penalty", -2.0, 2.0, True),
        ("repetition_penalty", 0.0, 2.0, False),
    ):
        value = body.get(field)
        if value is None:
            continue
        numeric = isinstance(value, int | float) and not isinstance(value, bool)
        try:
            float_value = float(value) if numeric else math.nan
        except (OverflowError, ValueError):
            float_value = math.nan
        in_range = math.isfinite(float_value) and float_value <= upper
        in_range = in_range and (float_value >= lower if lower_inclusive else float_value > lower)
        if not in_range:
            left = "[" if lower_inclusive else "("
            raise _bad_request(f"'{field}' must be a finite number in {left}{lower}, {upper}]", param=field)

    top_k = body.get("top_k")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise _bad_request("'top_k' must be an integer >= 1", param="top_k")


def _validate_mlx_chat_body(body: dict[str, Any]) -> None:
    unknown_fields = body.keys() - _MLX_CHAT_FIELDS
    if unknown_fields:
        unknown = sorted(unknown_fields)[0]
        raise _bad_request(f"field '{unknown}' is not supported", param=unknown, code="unsupported_field")


def _validate_chat_message_media(messages: Any) -> None:
    """Reject child-side remote media fetches before proxying to a child."""

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                _walk(item, f"{path}[{index}]")
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key == "image_url":
                url = item.get("url") if isinstance(item, dict) else item
                if not isinstance(url, str) or not url:
                    raise _bad_request(
                        f"'{item_path}' must be a non-empty string or an object with a non-empty 'url'",
                        param=item_path,
                    )
                try:
                    _decode_data_uri_image(url)
                except ValueError as exc:
                    raise _bad_request(str(exc), param=item_path) from exc
                continue
            if key in {"audio_url", "video_url"}:
                raise _bad_request(
                    f"'{item_path}' is not supported; remote media fetching is disabled",
                    param=item_path,
                    code="unsupported_field",
                )
            _walk(item, item_path)

    _walk(messages, "messages")


def _validated_child_chat_url(server_url: object) -> str:
    """Build the only allowed upstream URL from an adapter-owned loopback URL."""
    if not isinstance(server_url, str):
        raise ValueError("generation child did not publish a server URL")
    parsed = urlsplit(server_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("generation child published an invalid server port") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("generation child URL must be an explicit loopback HTTP origin")
    return f"{server_url.rstrip('/')}/v1/chat/completions"


def _merge_stops(request_stop: Any, configured: Any) -> Any:
    defaults = [item for item in configured or [] if isinstance(item, str) and item]
    if not defaults:
        return request_stop
    if request_stop is None:
        return defaults
    if isinstance(request_stop, str):
        merged = [request_stop]
    elif isinstance(request_stop, list):
        merged = list(request_stop)
    else:
        # Let the child produce the contract error for a malformed client stop.
        return request_stop
    merged.extend(item for item in defaults if item not in merged)
    return merged


def _prepare_sglang_body(
    body: dict[str, Any],
    *,
    config: Any,
    adapter: SGLangGenerationAdapter,
    max_completion_tokens: int | None,
    max_tokens: int | None,
    seed: int | None,
) -> dict[str, Any]:
    proxied = dict(body)
    proxied["model"] = adapter.served_model_name
    # SIE owns these edge-only identifiers; do not send unknown metadata into
    # the engine process.
    proxied.pop("safety_identifier", None)

    cap = config.tasks.generate.max_output_tokens
    requested_cap = max_completion_tokens if max_completion_tokens is not None else max_tokens
    if requested_cap is not None and requested_cap > cap:
        field = "max_completion_tokens" if max_completion_tokens is not None else "max_tokens"
        raise _bad_request(
            f"{field} ({requested_cap}) exceeds model cap ({cap})",
            param=field,
        )
    if max_completion_tokens is not None:
        proxied["max_completion_tokens"] = max_completion_tokens
        # Preferred field wins. Do not forward an ignored legacy value that
        # could exceed the cap or trigger version-specific child validation.
        proxied.pop("max_tokens", None)
        effective_cap = max_completion_tokens
    else:
        proxied["max_tokens"] = max_tokens if max_tokens is not None else cap
        effective_cap = proxied["max_tokens"]
    if seed is not None:
        proxied["seed"] = seed

    profile = config.resolve_profile("default")
    runtime = dict(profile.runtime)
    default_sampling = runtime.get("default_sampling")
    if isinstance(default_sampling, dict):
        for config_field, chat_field in _DEFAULT_SAMPLING_TO_CHAT_FIELD.items():
            if config_field in default_sampling and proxied.get(chat_field) is None:
                proxied[chat_field] = default_sampling[config_field]

    min_tokens = proxied.get("min_tokens")
    if isinstance(min_tokens, int) and not isinstance(min_tokens, bool) and min_tokens > effective_cap:
        if body.get("min_tokens") is not None:
            raise _bad_request(
                f"min_tokens ({min_tokens}) must not exceed the output cap ({effective_cap})",
                param="min_tokens",
            )
        # Profile minimums are soft defaults, matching GenerationAdapter.
        proxied["min_tokens"] = effective_cap

    configured_stops = runtime.get("stop_tokens")
    merged_stop = _merge_stops(body.get("stop"), configured_stops)
    if merged_stop is not None:
        proxied["stop"] = merged_stop

    request_template_kwargs = body.get("chat_template_kwargs")
    merged_template_kwargs = dict(request_template_kwargs or {})
    operator_template_kwargs = config.tasks.generate.chat_template_kwargs
    if isinstance(operator_template_kwargs, dict):
        # Match queued generation: operator YAML is authoritative on conflict.
        merged_template_kwargs.update(operator_template_kwargs)
    if merged_template_kwargs:
        proxied["chat_template_kwargs"] = merged_template_kwargs

    # The public contract is final-answer-only. Ask the child parser to split
    # reasoning from visible content, then strip that private field again on
    # both response paths below. Raw-tag fallback covers models with no parser.
    proxied["separate_reasoning"] = True
    proxied["stream_reasoning"] = True
    return proxied


def _prepare_mlx_body(
    body: dict[str, Any],
    *,
    config: Any,
    adapter: MLXGenerationAdapter,
    cap: int,
    max_completion_tokens: int | None,
    max_tokens: int | None,
    seed: int | None,
) -> dict[str, Any]:
    proxied = dict(body)
    proxied["model"] = adapter.mlx_repo
    requested_cap = max_completion_tokens if max_completion_tokens is not None else max_tokens
    if requested_cap is not None and requested_cap > cap:
        field = "max_completion_tokens" if max_completion_tokens is not None else "max_tokens"
        raise _bad_request(
            f"{field} ({requested_cap}) exceeds model cap ({cap})",
            param=field,
        )
    proxied["max_tokens"] = requested_cap if requested_cap is not None else cap
    # mlx_lm versions used by the Mac bundle consume the legacy field.
    proxied.pop("max_completion_tokens", None)
    if seed is not None:
        proxied["seed"] = seed

    profile = config.resolve_profile("default")
    runtime = dict(profile.runtime)
    default_sampling = runtime.get("default_sampling")
    if isinstance(default_sampling, dict):
        for config_field, chat_field in _DEFAULT_SAMPLING_TO_CHAT_FIELD.items():
            if (
                config_field in _MLX_DEFAULT_SAMPLING_FIELDS
                and config_field in default_sampling
                and proxied.get(chat_field) is None
            ):
                proxied[chat_field] = default_sampling[config_field]
    if isinstance(proxied.get("seed"), int) and not isinstance(proxied["seed"], bool):
        proxied["seed"] = normalize_mlx_seed(proxied["seed"])

    configured_stops = runtime.get("stop_tokens")
    merged_stop = _merge_stops(body.get("stop"), configured_stops)
    if merged_stop is not None:
        proxied["stop"] = merged_stop

    request_template_kwargs = body.get("chat_template_kwargs")
    merged_template_kwargs = dict(request_template_kwargs or {})
    operator_template_kwargs = config.tasks.generate.chat_template_kwargs
    if isinstance(operator_template_kwargs, dict):
        # Match queued generation: operator YAML is authoritative on conflict.
        merged_template_kwargs.update(operator_template_kwargs)
    if merged_template_kwargs:
        proxied["chat_template_kwargs"] = merged_template_kwargs
    return proxied


def _sanitize_chat_payload(
    payload: Any,
    *,
    requested_model: str,
    hide_thinking_blocks: bool,
    initial_inside_thinking: bool,
    reasoning_format: ReasoningFormat,
    states: dict[int, ThinkingBlockStripper] | None = None,
    terminal: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("child chat response must be a JSON object")
    payload["model"] = requested_model
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload

    state_map = states if states is not None else {}
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        choice_obj = cast("dict[str, Any]", choice)
        raw_index = choice_obj.get("index", position)
        index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else position
        message = choice_obj.get("message")
        delta = choice_obj.get("delta")
        container = message if isinstance(message, dict) else delta if isinstance(delta, dict) else None
        if container is None:
            continue
        reasoning_content = container.pop("reasoning_content", None)
        reasoning = container.pop("reasoning", None)
        reasoning_was_present = (reasoning_content is not None and reasoning_content != "") or (
            reasoning is not None and reasoning != ""
        )

        if not hide_thinking_blocks:
            continue
        stripper = state_map.setdefault(
            index,
            ThinkingBlockStripper(
                inside=initial_inside_thinking,
                reasoning_format=reasoning_format,
            ),
        )
        raw_content = container.get("content")
        content = raw_content if isinstance(raw_content, str) else ""
        visible = stripper.feed(content) if content else ""
        is_terminal_choice = terminal or choice_obj.get("finish_reason") is not None
        if is_terminal_choice:
            visible += stripper.finish()
        if isinstance(raw_content, str) or visible:
            container["content"] = visible
        if (reasoning_was_present or visible != content) and "logprobs" in choice_obj:
            choice_obj["logprobs"] = None
    return payload


async def _read_upstream_body(response: httpx.Response) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > _MAX_CHAT_RESPONSE_BYTES:
            raise ValueError("upstream response exceeded the byte limit")
        body.extend(chunk)
    return bytes(body)


def _new_proxy_client() -> httpx.AsyncClient:
    # Ignore HTTP(S)_PROXY: even a loopback URL must connect directly to the
    # adapter-managed child rather than an operator-shell proxy.
    return httpx.AsyncClient(timeout=_PROXY_TIMEOUT, trust_env=False)


async def _sanitize_sse_stream(
    response: httpx.Response,
    client: httpx.AsyncClient,
    *,
    requested_model: str,
    hide_thinking_blocks: bool,
    initial_inside_thinking: bool,
    reasoning_format: ReasoningFormat,
    slot: asyncio.Semaphore | None,
) -> AsyncIterator[bytes]:
    states: dict[int, ThinkingBlockStripper] = {}
    buffered = bytearray()
    received = 0
    saw_done = False
    terminal_outcome_selected = False
    try:
        async for chunk in response.aiter_bytes():
            received += len(chunk)
            if received > _MAX_CHAT_RESPONSE_BYTES:
                logger.error("generation child SSE exceeded %d bytes", _MAX_CHAT_RESPONSE_BYTES)
                terminal_outcome_selected = True
                yield _upstream_error_event("upstream response exceeded the byte limit")
                return
            buffered.extend(chunk)
            while True:
                newline_at = buffered.find(b"\n")
                if newline_at < 0:
                    break
                line = bytes(buffered[:newline_at]).removesuffix(b"\r")
                del buffered[: newline_at + 1]
                if not line.startswith(b"data:"):
                    yield line + b"\n"
                    continue
                data = line[len(b"data:") :].lstrip()
                if data == b"[DONE]":
                    saw_done = True
                    terminal_outcome_selected = True
                    yield b"data: [DONE]\n"
                    continue
                if not data:
                    yield line + b"\n"
                    continue
                parsed = json.loads(data)
                sanitized = _sanitize_chat_payload(
                    parsed,
                    requested_model=requested_model,
                    hide_thinking_blocks=hide_thinking_blocks,
                    initial_inside_thinking=initial_inside_thinking,
                    reasoning_format=reasoning_format,
                    states=states,
                    terminal=False,
                )
                encoded = json.dumps(sanitized, separators=(",", ":"), ensure_ascii=False).encode()
                yield b"data: " + encoded + b"\n"

        if buffered:
            line = bytes(buffered).removesuffix(b"\r")
            if line.startswith(b"data:") and line[len(b"data:") :].lstrip() == b"[DONE]":
                saw_done = True
                terminal_outcome_selected = True
                yield b"data: [DONE]\n"
            elif line:
                raise ValueError("upstream SSE ended with an incomplete event")
        if not saw_done:
            logger.warning("generation child SSE ended without [DONE]")
            terminal_outcome_selected = True
            yield _upstream_error_event("upstream stream ended unexpectedly")
    except (httpx.HTTPError, UnicodeError, ValueError, json.JSONDecodeError):
        logger.warning("generation child chat proxy stream failed", exc_info=True)
        if not saw_done:
            terminal_outcome_selected = True
            yield _upstream_error_event()
    finally:
        try:
            await aclose_with_error_precedence(
                response,
                outcome_selected=terminal_outcome_selected,
                context="generation child SSE response",
            )
        finally:
            try:
                await aclose_with_error_precedence(
                    client,
                    outcome_selected=terminal_outcome_selected,
                    context="generation child SSE client",
                )
            finally:
                if slot is not None:
                    slot.release()


# -- /v1/chat/completions (proxy to the managed generation child) ------------


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> Response | StreamingResponse:
    """OpenAI-compatible chat completions through the loaded generation child.

    MLX remains the non-CUDA backend. CUDA profiles proxy to the exact SGLang
    subprocess the registry loaded for the requested model/profile. The child
    URL and served model name always come from that adapter, never the body.

    Errors are emitted as top-level OpenAI ``{"error": {...}}`` envelopes
    (never FastAPI's ``{"detail": ...}`` wrapper), matching ``/v1/completions``.
    """
    try:
        return await _chat_completions(http_request, x_machine_profile)
    except HTTPException as exc:
        return openai_error_response(exc)


async def _chat_completions(
    http_request: Request,
    x_machine_profile: str | None,
) -> Response | StreamingResponse:
    validate_machine_profile_header(x_machine_profile)

    body = await _read_json_body(http_request)

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _bad_request("'model' must be a non-empty string", param="model")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _bad_request("'messages' must be a non-empty array", param="messages")
    if len(messages) > _MAX_CHAT_MESSAGES:
        raise _bad_request(
            f"'messages' exceeds the maximum of {_MAX_CHAT_MESSAGES} per request",
            param="messages",
        )
    _validate_chat_messages(messages)
    _validate_chat_message_media(messages)
    chat_template_kwargs = body.get("chat_template_kwargs")
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        raise _bad_request("'chat_template_kwargs' must be an object", param="chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict):
        try:
            validate_chat_template_kwargs(chat_template_kwargs)
        except ValueError as exc:
            code = "unsupported_field" if str(exc).startswith("unsupported ") else "invalid_request"
            raise _bad_request(str(exc), param="chat_template_kwargs", code=code) from exc
    stream_opt = body.get("stream")
    if stream_opt is not None and not isinstance(stream_opt, bool):
        raise _bad_request("'stream' must be a boolean", param="stream")
    max_completion_tokens = _validate_optional_positive_int(body, "max_completion_tokens")
    max_tokens = _validate_optional_positive_int(body, "max_tokens")
    seed = _validate_chat_seed(body.get("seed"))

    registry = http_request.app.state.registry
    device = registry.device
    registry_key = denormalize_model_id(model)

    with tracer.start_as_current_span("chat_completions") as span:
        span.set_attribute("model", model)
        checker = ModelStateChecker(registry, registry_key, span)
        checker.check_exists()

        # Validate capability and all CUDA edge fields before a model download.
        config = registry.get_config(registry_key)
        gen_task = getattr(config.tasks, "generate", None)
        if gen_task is None:
            raise _bad_request(
                f"Model '{model}' does not support generation (no generate task). Use a generation model."
            )
        streaming_supported = getattr(getattr(gen_task, "capabilities", None), "streaming", True)
        if bool(stream_opt) and not streaming_supported:
            raise _bad_request(
                f"Model '{model}' does not support streaming generation",
                param="stream",
                code="unsupported_field",
            )
        if str(device).startswith("cuda"):
            _validate_cuda_chat_body(body)
        else:
            _validate_mlx_chat_body(body)

        checker.check_not_failed()
        checker.check_not_unloading()
        checker.check_not_loading()
        await checker.ensure_loaded(device)

        adapter = registry.get(registry_key)
        registry.touch_lru(registry_key)
        slot: asyncio.Semaphore | None = None
        reasoning_parser: str | None = None
        if isinstance(adapter, MLXGenerationAdapter) and not str(device).startswith("cuda"):
            if adapter.mlx_repo is None:
                raise _unsupported_chat(model, device)
            server_url = adapter.server_url
            proxied = _prepare_mlx_body(
                body,
                config=config,
                adapter=adapter,
                cap=gen_task.max_output_tokens,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                seed=seed,
            )
            slot = adapter.generation_slot()
        elif isinstance(adapter, SGLangGenerationAdapter) and str(device).startswith("cuda"):
            server_url = adapter.server_url
            reasoning_parser = adapter.reasoning_parser
            proxied = _prepare_sglang_body(
                body,
                config=config,
                adapter=adapter,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                seed=seed,
            )
        else:
            raise _unsupported_chat(model, device)

        try:
            url = _validated_child_chat_url(server_url)
        except ValueError:
            logger.exception("generation adapter exposed an unsafe or missing child URL")
            return _upstream_error_response("generation child is unavailable")

        stream = bool(stream_opt)  # validated above (bool or None)
        hide_thinking_blocks = thinking_blocks_must_be_hidden(config)
        reasoning_format = resolve_reasoning_format(config, adapter)
        configured_template_kwargs = gen_task.chat_template_kwargs
        thinking_enabled = (
            isinstance(configured_template_kwargs, dict) and configured_template_kwargs.get("enable_thinking") is True
        )
        # A parser returns final content separately from reasoning_content. A
        # parser-less Qwen template may seed <think> in the prompt, so its first
        # generated byte is already private reasoning and only </think> appears
        # in output.
        initial_inside_thinking = hide_thinking_blocks and thinking_enabled and reasoning_parser is None

        if stream:
            if slot is not None:
                await slot.acquire()
            client = _new_proxy_client()
            try:
                upstream_request = client.build_request("POST", url, json=proxied)
                upstream = await client.send(upstream_request, stream=True)
            except asyncio.CancelledError:
                await client.aclose()
                if slot is not None:
                    slot.release()
                raise
            except httpx.HTTPError:
                logger.warning("generation child chat stream connection failed", exc_info=True)
                await client.aclose()
                if slot is not None:
                    slot.release()
                return _upstream_error_response("generation child is unavailable")
            if upstream.status_code != status.HTTP_200_OK:
                try:
                    await _read_upstream_body(upstream)
                except (httpx.HTTPError, ValueError):
                    # The body is intentionally discarded; never log child
                    # validation details because they can include user input.
                    pass
                finally:
                    await upstream.aclose()
                    await client.aclose()
                    if slot is not None:
                        slot.release()
                logger.error("generation child chat error status=%d", upstream.status_code)
                return _upstream_error_response(
                    response_status=_translated_upstream_status(upstream.status_code),
                )
            content_type = upstream.headers.get("content-type", "").partition(";")[0].strip().lower()
            if content_type != "text/event-stream":
                try:
                    await _read_upstream_body(upstream)
                except (httpx.HTTPError, ValueError):
                    # Drain only for connection reuse and bounded cleanup; the
                    # unexpected response body is intentionally discarded.
                    pass
                finally:
                    await upstream.aclose()
                    await client.aclose()
                    if slot is not None:
                        slot.release()
                logger.error("generation child returned non-SSE content for a streaming request")
                return _upstream_error_response("generation child returned an invalid streaming response")
            return StreamingResponse(
                _sanitize_sse_stream(
                    upstream,
                    client,
                    requested_model=model,
                    hide_thinking_blocks=hide_thinking_blocks,
                    initial_inside_thinking=initial_inside_thinking,
                    reasoning_format=reasoning_format,
                    slot=slot,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        if slot is not None:
            await slot.acquire()
        try:
            async with _new_proxy_client() as client:
                try:
                    async with client.stream("POST", url, json=proxied) as upstream:
                        raw = await _read_upstream_body(upstream)
                        upstream_status = upstream.status_code
                except httpx.HTTPError:
                    logger.warning("generation child chat request failed", exc_info=True)
                    return _upstream_error_response("generation child is unavailable")
        except ValueError:
            logger.error("generation child chat response exceeded its byte limit")
            return _upstream_error_response("upstream response exceeded the byte limit")
        finally:
            if slot is not None:
                slot.release()

        if upstream_status != status.HTTP_200_OK:
            logger.error("generation child chat error status=%d", upstream_status)
            return _upstream_error_response(
                response_status=_translated_upstream_status(upstream_status),
            )
        try:
            payload = _sanitize_chat_payload(
                json.loads(raw),
                requested_model=model,
                hide_thinking_blocks=hide_thinking_blocks,
                initial_inside_thinking=initial_inside_thinking,
                reasoning_format=reasoning_format,
                terminal=True,
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.warning("generation child returned an invalid chat response", exc_info=True)
            return _upstream_error_response("generation child returned an invalid response")
        return JSONResponse(content=payload)


# -- /v1/rerank (Cohere/OpenAI shape over the score adapter) -----------------


@router.post("/rerank", response_model=None)
async def rerank(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> JSONResponse:
    """Cohere/OpenAI-style reranking backed by the in-process score adapter.

    Request: ``{model, query, documents: [str], top_n?, return_documents?}``.
    Response: ``{model, results: [{index, relevance_score, document?}], usage}``
    sorted by descending relevance.

    Errors are emitted as top-level OpenAI ``{"error": {...}}`` envelopes
    (never FastAPI's ``{"detail": ...}`` wrapper), matching ``/v1/completions``.
    """
    try:
        return await _rerank(http_request, x_machine_profile)
    except HTTPException as exc:
        return openai_error_response(exc)


async def _rerank(
    http_request: Request,
    x_machine_profile: str | None,
) -> JSONResponse:
    validate_machine_profile_header(x_machine_profile)

    body = await _read_json_body(http_request)
    allowed_fields = {"model", "query", "documents", "top_n", "return_documents", "options"}
    unknown_fields = body.keys() - allowed_fields
    if unknown_fields:
        unknown = sorted(unknown_fields)[0]
        raise _bad_request(f"field '{unknown}' is not supported", param=unknown)

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _bad_request("'model' must be a non-empty string", param="model")
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise _bad_request("'query' must be a non-empty string", param="query")
    documents = body.get("documents")
    if not isinstance(documents, list) or not documents or not all(isinstance(d, str) and d.strip() for d in documents):
        raise _bad_request("'documents' must be a non-empty array of strings", param="documents")
    if len(documents) > _MAX_RERANK_DOCS:
        raise _bad_request(f"'documents' exceeds the maximum of {_MAX_RERANK_DOCS} per request", param="documents")
    top_n = body.get("top_n")
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0):
        raise _bad_request("'top_n' must be a positive integer", param="top_n")
    return_documents = body.get("return_documents", False)
    if not isinstance(return_documents, bool):
        raise _bad_request("'return_documents' must be a boolean", param="return_documents")

    registry = http_request.app.state.registry
    device = registry.device
    registry_key = denormalize_model_id(model)

    with tracer.start_as_current_span("rerank") as span:
        span.set_attribute("model", model)
        span.set_attribute("batch_size", len(documents))
        checker = ModelStateChecker(registry, registry_key, span)
        checker.check_exists()

        config = registry.get_config(registry_key)
        if config.tasks.score is None:
            raise _bad_request(
                f"Model '{model}' does not support reranking (no score task). Use a reranker model.",
            )

        checker.check_not_failed()
        checker.check_not_unloading()
        checker.check_not_loading()
        await checker.ensure_loaded(device)

        query_item = Item(text=query)
        doc_items = [Item(id=str(i), text=str(doc)) for i, doc in enumerate(documents)]
        options_raw = body.get("options")
        if options_raw is not None and not isinstance(options_raw, dict):
            raise _bad_request("'options' must be an object", param="options")
        options = resolve_runtime_options(config, options_raw, span)
        instruction = options.get("instruction")

        timing = RequestTiming()
        timing.start_tokenization()
        prepared_items = build_score_prepared_items(query_item, doc_items)
        timing.end_tokenization()
        worker = await registry.start_worker(registry_key)
        future = await worker.submit_score(
            prepared_items=prepared_items,
            query=query_item,
            items=doc_items,
            instruction=instruction,
            options=options,
            timing=timing,
        )
        try:
            worker_result = await future
        except Exception as exc:
            logger.warning("rerank failed for %s", model, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "inference_error", "message": "internal error during reranking"},
            ) from exc

        score_output: ScoreOutput = worker_result.output
        scores = [float(score_output.scores[i]) for i in range(score_output.batch_size)]
        # Fail closed on non-finite (NaN/inf) model output before serialization;
        # the HTTPException is re-emitted as the OpenAI error envelope by rerank().
        ensure_finite_scores(scores, model)
        usage = score_usage_from_output(score_output)
        if usage is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "inference_error", "message": "score response missing authoritative usage"},
            )

    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    results: list[dict[str, Any]] = []
    for index, score in ranked:
        entry: dict[str, Any] = {"index": index, "relevance_score": score}
        if return_documents:
            entry["document"] = {"text": documents[index]}
        results.append(entry)

    return JSONResponse(
        content={
            "model": getattr(config, "name", None) or registry_key,
            "results": results,
            "usage": usage,
        }
    )
