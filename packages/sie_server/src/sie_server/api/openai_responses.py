"""Direct OpenAI-compatible ``/v1/responses`` generation surface.

This is the direct-server counterpart of the gateway's stateless, text-only
Responses MVP. Both raw string input and validated message input are reduced to
one rendered prompt and executed through ``GenerationAdapter.generate``.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, cast

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sie_sdk.queue_types import denormalize_model_id

from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationError,
    reasoning_starts_in_prompt,
    resolve_reasoning_format,
    suppress_thinking_blocks,
    thinking_blocks_must_be_hidden,
)
from sie_server.api.generate import _render_native_messages_prompt
from sie_server.api.helpers import ModelStateChecker, check_sdk_version
from sie_server.api.openai_completions import (
    _MAX_F32,
    _MAX_PROMPT_BYTES,
    _collect_completion,
    _CompletionError,
    _error_response,
    _from_generation_error,
    _from_http_exception,
    _number,
    _read_json_body,
)
from sie_server.api.validation import validate_machine_profile_header, validate_signed_i64
from sie_server.core.runtime_options import apply_generation_runtime_options
from sie_server.observability.tracing import tracer
from sie_server.types.openapi import OpenAIResponsesResponseModel

router = APIRouter(prefix="/v1", tags=["openai-compat"])
logger = logging.getLogger(__name__)

_MAX_U32 = (1 << 32) - 1
_SUPPORTED_FIELDS = frozenset(
    {
        "model",
        "input",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "stream",
    }
)
_KNOWN_UNSUPPORTED_FIELDS = frozenset(
    {
        "previous_response_id",
        "tools",
        "tool_choice",
        "reasoning",
        "background",
        "metadata",
        "instructions",
    }
)
_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "developer"})
_ALLOWED_CONTENT_TYPES = frozenset({"text", "input_text", "output_text"})


@dataclass(frozen=True, slots=True)
class _ResponsesParams:
    model: str
    prompt: str | None
    messages: list[dict[str, str]] | None
    max_output_tokens: int
    temperature: float | None
    top_p: float | None
    seed: int | None


def _unsupported(field: str) -> _CompletionError:
    return _CompletionError(
        f"'{field}' is not supported by this endpoint",
        param=field,
        code="unsupported_field",
    )


def _parse_message_content(value: Any, *, index: int) -> str:
    path = f"input[{index}].content"
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise _CompletionError(f"'{path}' must be a string or text parts", param=path)

    text: list[str] = []
    for part_index, part in enumerate(value):
        part_path = f"{path}[{part_index}]"
        if not isinstance(part, dict):
            raise _CompletionError(f"'{part_path}' must be an object", param=part_path)
        part_obj = cast("dict[str, Any]", part)
        unknown = set(part_obj) - {"type", "text"}
        if unknown:
            field = sorted(unknown)[0]
            raise _unsupported(f"{part_path}.{field}")
        part_type = part_obj.get("type")
        if not isinstance(part_type, str):
            raise _CompletionError(f"'{part_path}.type' is required and must be a string", param=f"{part_path}.type")
        if part_type not in _ALLOWED_CONTENT_TYPES:
            raise _CompletionError(
                f"unsupported content part type '{part_type}'",
                param=f"{part_path}.type",
                code="unsupported_field",
            )
        part_text = part_obj.get("text")
        if not isinstance(part_text, str):
            raise _CompletionError(
                f"'{part_path}.text' is required and must be a string",
                param=f"{part_path}.text",
            )
        text.append(part_text)
    return "".join(text)


def _parse_input(value: Any) -> tuple[str | None, list[dict[str, str]] | None]:
    if isinstance(value, str):
        return value, None
    if not isinstance(value, list) or not value:
        raise _CompletionError(
            'field "input" is required (a string or a non-empty list of messages)',
            param="input",
        )

    messages: list[dict[str, str]] = []
    for index, item in enumerate(value):
        path = f"input[{index}]"
        if not isinstance(item, dict):
            raise _CompletionError(f"'{path}' must be an object", param=path)
        item_obj = cast("dict[str, Any]", item)
        unknown = set(item_obj) - {"role", "content"}
        if unknown:
            field = sorted(unknown)[0]
            raise _unsupported(f"{path}.{field}")
        role = item_obj.get("role")
        if not isinstance(role, str):
            raise _CompletionError(f"'{path}.role' is required and must be a string", param=f"{path}.role")
        if role not in _ALLOWED_ROLES:
            raise _CompletionError(f"'{path}.role' is invalid", param=f"{path}.role")
        messages.append(
            {
                "role": "system" if role == "developer" else role,
                "content": _parse_message_content(item_obj.get("content"), index=index),
            }
        )
    return None, messages


def _parse_params(body: dict[str, Any]) -> _ResponsesParams:
    for field in _KNOWN_UNSUPPORTED_FIELDS:
        if field in body:
            raise _unsupported(field)
    unknown = set(body) - _SUPPORTED_FIELDS
    if unknown:
        raise _unsupported(sorted(unknown)[0])

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise _CompletionError('field "model" is required', param="model")

    prompt, messages = _parse_input(body.get("input"))

    stream = body.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise _CompletionError("'stream' must be a boolean", param="stream")
    if stream:
        raise _unsupported("stream")

    max_output_tokens = body.get("max_output_tokens", 16)
    if max_output_tokens is None:
        max_output_tokens = 16
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or not (1 <= max_output_tokens <= _MAX_U32)
    ):
        raise _CompletionError("'max_output_tokens' must be a positive integer", param="max_output_tokens")

    temperature = _number(body.get("temperature"), param="temperature", minimum=0, maximum=_MAX_F32)
    top_p = _number(body.get("top_p"), param="top_p", minimum=0, maximum=1)
    if top_p is not None and top_p == 0:
        raise _CompletionError("'top_p' must be a number in (0, 1]", param="top_p")
    try:
        seed = validate_signed_i64(body.get("seed"), param="seed")
    except ValueError as exc:
        raise _CompletionError(str(exc), param="seed") from exc

    return _ResponsesParams(
        model=model,
        prompt=prompt,
        messages=messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )


@router.post(
    "/responses",
    response_model=OpenAIResponsesResponseModel,
    responses={
        400: {"description": "Invalid or unsupported request"},
        404: {"description": "Model not found"},
        413: {"description": "Request body or rendered input is too large"},
        500: {"description": "Generation failed"},
        503: {"description": "Model loading or temporarily unavailable"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/OpenAIResponsesRequestModel"}}},
        }
    },
)
async def responses(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> JSONResponse:
    """Create one stateless, text-only OpenAI Responses result."""
    try:
        validate_machine_profile_header(x_machine_profile)
        check_sdk_version(http_request)
        params = _parse_params(await _read_json_body(http_request))

        registry = http_request.app.state.registry
        registry_key = denormalize_model_id(params.model)
        with tracer.start_as_current_span("openai_responses") as span:
            span.set_attribute("model", params.model)
            checker = ModelStateChecker(registry, registry_key, span)
            checker.check_exists()
            checker.check_not_failed()
            checker.check_not_unloading()
            checker.check_not_loading()

            config = registry.get_config(registry_key)
            generate_task = getattr(config.tasks, "generate", None)
            if generate_task is None:
                raise _CompletionError(
                    f"Model '{params.model}' does not support generation",
                    param="model",
                    code="model_not_found",
                )
            if params.max_output_tokens > generate_task.max_output_tokens:
                raise _CompletionError(
                    f"max_output_tokens ({params.max_output_tokens}) exceeds model cap "
                    f"({generate_task.max_output_tokens})",
                    param="max_output_tokens",
                    code="context_exceeded",
                )

            prompt = params.prompt
            if params.messages is not None:
                prompt = await _render_native_messages_prompt(config, params.messages)
            if prompt is None:  # pragma: no cover - parser invariant
                raise _CompletionError("input did not resolve to a prompt", param="input")
            prompt_bytes = len(prompt.encode("utf-8"))
            if prompt_bytes > _MAX_PROMPT_BYTES:
                raise _CompletionError(
                    f"rendered input is {prompt_bytes} bytes, exceeds the limit of {_MAX_PROMPT_BYTES} bytes",
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    param="input",
                    code="context_exceeded",
                )

            runtime_params: dict[str, Any] = {
                "prompt": prompt,
                "max_new_tokens": params.max_output_tokens,
                **{
                    key: value
                    for key, value in (
                        ("temperature", params.temperature),
                        ("top_p", params.top_p),
                        ("seed", params.seed),
                    )
                    if value is not None
                },
            }
            try:
                runtime_params = apply_generation_runtime_options(config, None, runtime_params)
            except ValueError as exc:
                raise _CompletionError(str(exc), code="invalid_request") from exc

            await checker.ensure_loaded(registry.device)
            adapter = registry.get(registry_key)
            registry.touch_lru(registry_key)
            if not isinstance(adapter, GenerationAdapter):
                raise _CompletionError(
                    f"Model '{params.model}' adapter does not support generation",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    param="model",
                    code="inference_error",
                )

            generation_parameters: dict[str, Any] = {
                "prompt": prompt,
                "max_new_tokens": params.max_output_tokens,
                "temperature": float(runtime_params.get("temperature", 1.0)),
                "top_p": float(runtime_params.get("top_p", 1.0)),
                "stop": runtime_params.get("stop"),
                "frequency_penalty": runtime_params.get("frequency_penalty"),
                "presence_penalty": runtime_params.get("presence_penalty"),
                "top_k": runtime_params.get("top_k"),
                "min_new_tokens": runtime_params.get("min_tokens"),
                "seed": runtime_params.get("seed"),
            }
            try:
                adapter.preflight_generate(generation_parameters, stream=False)
            except GenerationError as exc:
                raise _from_generation_error(exc, registry) from exc
            except Exception as exc:
                logger.warning("OpenAI Responses preflight failed for %s", params.model, exc_info=True)
                raise _CompletionError(
                    "internal error during generation",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    code="inference_error",
                ) from exc

            try:
                chunks = adapter.generate(**generation_parameters)
            except GenerationError as exc:
                raise _from_generation_error(exc, registry) from exc
            if thinking_blocks_must_be_hidden(config):
                reasoning_format = resolve_reasoning_format(config, adapter)
                chunks = suppress_thinking_blocks(
                    chunks,
                    start_inside=reasoning_starts_in_prompt(prompt, reasoning_format),
                    reasoning_format=reasoning_format,
                )
            try:
                text, terminal = await _collect_completion(chunks)
            except _CompletionError:
                raise
            except GenerationError as exc:
                raise _from_generation_error(exc, registry) from exc
            except Exception as exc:
                logger.warning("OpenAI Responses generation failed for %s", params.model, exc_info=True)
                raise _CompletionError(
                    "internal error during generation",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    code="inference_error",
                ) from exc

            canonical_model = getattr(config, "name", None) or registry_key
            response_id = f"resp-{uuid.uuid4().hex}"
            if terminal.prompt_tokens is None or terminal.completion_tokens is None:
                raise _CompletionError(
                    "generation terminal event omitted required usage",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    code="malformed_worker_response",
                )
            prompt_tokens = terminal.prompt_tokens
            completion_tokens = terminal.completion_tokens
            return JSONResponse(
                content={
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": canonical_model,
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "id": f"msg-{response_id.removeprefix('resp-')}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}],
                        }
                    ],
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
    except _CompletionError as exc:
        return _error_response(exc)
    except HTTPException as exc:
        return _error_response(_from_http_exception(exc))
