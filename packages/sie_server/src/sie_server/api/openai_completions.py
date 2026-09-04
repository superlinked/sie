"""Direct OpenAI-compatible ``/v1/completions`` generation surface.

The cluster gateway remains the production ingress. This route gives the
single-process ``sie-server serve`` topology the same raw-prompt compatibility
surface by calling the loaded :class:`GenerationAdapter` directly, just like
the SIE-native ``/v1/generate/{model}`` route. It deliberately does not proxy a
backend subprocess API: model caps, profile runtime defaults, output parsing,
and adapter fixes therefore remain authoritative in every deployment shape.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sie_sdk.queue_types import denormalize_model_id

from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationChunk,
    GenerationError,
    aclose_with_error_precedence,
    client_safe_generation_error_code,
    client_safe_generation_error_message,
    reasoning_starts_in_prompt,
    resolve_reasoning_format,
    suppress_thinking_blocks,
    thinking_blocks_must_be_hidden,
)
from sie_server.api.generate import _generation_http_exception
from sie_server.api.helpers import ModelStateChecker, check_sdk_version
from sie_server.api.validation import validate_machine_profile_header, validate_signed_i64
from sie_server.core.runtime_options import apply_generation_runtime_options
from sie_server.observability.tracing import tracer
from sie_server.types.openapi import OpenAICompletionResponseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

_MAX_COMPLETIONS_BODY_BYTES = int(os.environ.get("SIE_GENERATE_MAX_BODY_BYTES", str(24 * 1024 * 1024)))
_MAX_PROMPT_BYTES = int(os.environ.get("SIE_GENERATE_MAX_PROMPT_BYTES", str(4 * 1024 * 1024)))
_MAX_U32 = (1 << 32) - 1
_MAX_F32 = 3.4028234663852886e38
_MAX_RETRY_AFTER_S = 60
_SUPPORTED_FIELDS = frozenset(
    {
        "model",
        "prompt",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stream",
        "stream_options",
        "n",
    }
)
_KNOWN_UNSUPPORTED_FIELDS = frozenset({"echo", "suffix", "logprobs", "best_of"})


class _CompletionError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        param: str | None = None,
        code: str = "invalid_request",
        error_type: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.param = param
        self.code = code
        self.error_type = error_type or (
            "invalid_request_error" if status_code < status.HTTP_500_INTERNAL_SERVER_ERROR else "server_error"
        )
        self.headers = headers


@dataclass(frozen=True, slots=True)
class _CompletionParams:
    model: str
    prompt: str
    max_tokens: int
    temperature: float | None
    top_p: float | None
    stop: list[str] | None
    frequency_penalty: float | None
    presence_penalty: float | None
    seed: int | None
    stream: bool
    include_usage: bool


def _error_response(error: _CompletionError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {
                "message": error.message,
                "type": error.error_type,
                "param": error.param,
                "code": error.code,
            }
        },
        headers=error.headers,
    )


def _from_http_exception(exc: HTTPException) -> _CompletionError:
    detail = exc.detail
    if isinstance(detail, dict):
        nested = detail.get("error")
        if isinstance(nested, dict):
            detail = nested
        message = str(detail.get("message", "request failed"))
        code = str(detail.get("code", "invalid_request"))
        param = detail.get("param")
        param = param if isinstance(param, str) else None
    else:
        message = str(detail)
        code = "invalid_request"
        param = None
    return _CompletionError(
        message,
        status_code=exc.status_code,
        param=param,
        code=code,
        headers=dict(exc.headers) if exc.headers is not None else None,
    )


def _from_generation_error(error: GenerationError, registry: Any) -> _CompletionError:
    """Preserve the native generation error contract on compatibility routes."""
    return _from_http_exception(_generation_http_exception(error, registry))


def _validated_retry_after_s(error: _CompletionError) -> int | None:
    """Read the trusted direct-route retry authority for in-band SSE use."""
    if error.code != "RESOURCE_EXHAUSTED" or error.headers is None:
        return None
    raw = error.headers.get("Retry-After")
    if raw is None or not raw.isascii() or not raw.isdecimal():
        return None
    value = int(raw)
    return value if 1 <= value <= _MAX_RETRY_AFTER_S else None


async def _read_json_body(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = -1
        if declared_length > _MAX_COMPLETIONS_BODY_BYTES:
            raise _CompletionError(
                f"request body exceeds the limit of {_MAX_COMPLETIONS_BODY_BYTES} bytes",
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )

    raw = bytearray()
    async for chunk in request.stream():
        if len(chunk) > _MAX_COMPLETIONS_BODY_BYTES - len(raw):
            raise _CompletionError(
                f"request body exceeds the limit of {_MAX_COMPLETIONS_BODY_BYTES} bytes",
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
        raw.extend(chunk)
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise _CompletionError("request body must be a JSON object") from exc
    if not isinstance(body, dict):
        raise _CompletionError("request body must be a JSON object")
    return body


def _number(value: Any, *, param: str, minimum: float, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _CompletionError(f"'{param}' must be a number", param=param)
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise _CompletionError(f"'{param}' must be a finite number", param=param) from exc
    if not math.isfinite(parsed) or parsed < minimum or (maximum is not None and parsed > maximum):
        interval = f"[{minimum:g}, {maximum:g}]" if maximum is not None else f">= {minimum:g}"
        raise _CompletionError(f"'{param}' must be a finite number in {interval}", param=param)
    return parsed


def _parse_params(body: dict[str, Any]) -> _CompletionParams:
    for field in _KNOWN_UNSUPPORTED_FIELDS:
        if field in body:
            raise _CompletionError(
                f"'{field}' is not supported by this endpoint",
                param=field,
                code="unsupported_field",
            )
    unknown = set(body) - _SUPPORTED_FIELDS
    if unknown:
        field = sorted(unknown)[0]
        raise _CompletionError(
            f"'{field}' is not supported by this endpoint",
            param=field,
            code="unsupported_field",
        )

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise _CompletionError('field "model" is required', param="model")

    prompt_raw = body.get("prompt")
    if isinstance(prompt_raw, str):
        prompt = prompt_raw
    elif isinstance(prompt_raw, list) and len(prompt_raw) == 1 and isinstance(prompt_raw[0], str):
        prompt = prompt_raw[0]
    elif isinstance(prompt_raw, list):
        if len(prompt_raw) != 1:
            raise _CompletionError(
                "batched array prompts are not supported; send one prompt string",
                param="prompt",
                code="unsupported_field",
            )
        raise _CompletionError("'prompt' array entries must be strings", param="prompt")
    else:
        raise _CompletionError('field "prompt" is required and must be a string', param="prompt")
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > _MAX_PROMPT_BYTES:
        raise _CompletionError(
            f"'prompt' is {prompt_bytes} bytes, exceeds the limit of {_MAX_PROMPT_BYTES} bytes",
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            param="prompt",
            code="context_exceeded",
        )

    stream_raw = body.get("stream")
    if stream_raw is not None and not isinstance(stream_raw, bool):
        raise _CompletionError("'stream' must be a boolean", param="stream")
    stream = bool(stream_raw)

    include_usage = False
    stream_options = body.get("stream_options")
    if stream_options is not None:
        if not isinstance(stream_options, dict):
            raise _CompletionError("'stream_options' must be a JSON object", param="stream_options")
        unknown_options = set(stream_options) - {"include_usage"}
        if unknown_options:
            key = sorted(unknown_options)[0]
            raise _CompletionError(
                f"'stream_options.{key}' is not supported by this endpoint",
                param=f"stream_options.{key}",
                code="unsupported_field",
            )
        include_usage_raw = stream_options.get("include_usage")
        if include_usage_raw is not None and not isinstance(include_usage_raw, bool):
            raise _CompletionError(
                "'stream_options.include_usage' must be a boolean",
                param="stream_options.include_usage",
            )
        include_usage = bool(include_usage_raw)

    n = body.get("n")
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise _CompletionError("'n' must be an integer", param="n")
        if n <= 0:
            raise _CompletionError("'n' must be a positive integer", param="n")
        if n != 1:
            raise _CompletionError(
                "'n' > 1 is not yet supported on /v1/completions",
                param="n",
                code="unsupported_field",
            )

    max_tokens = body.get("max_tokens", 16)
    if max_tokens is None:
        max_tokens = 16
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not (1 <= max_tokens <= _MAX_U32):
        raise _CompletionError("'max_tokens' must be a positive integer", param="max_tokens")

    temperature = _number(body.get("temperature"), param="temperature", minimum=0, maximum=_MAX_F32)
    top_p = _number(body.get("top_p"), param="top_p", minimum=0)
    if top_p is not None and top_p == 0:
        raise _CompletionError("'top_p' must be a number in (0, 1]", param="top_p")
    if top_p is not None and top_p > 1:
        raise _CompletionError("'top_p' must be a number in (0, 1]", param="top_p")
    frequency_penalty = _number(
        body.get("frequency_penalty"),
        param="frequency_penalty",
        minimum=-2,
        maximum=2,
    )
    presence_penalty = _number(
        body.get("presence_penalty"),
        param="presence_penalty",
        minimum=-2,
        maximum=2,
    )

    try:
        seed = validate_signed_i64(body.get("seed"), param="seed")
    except ValueError as exc:
        raise _CompletionError(str(exc), param="seed") from exc

    stop_raw = body.get("stop")
    if stop_raw is None:
        stop = None
    elif isinstance(stop_raw, str):
        if not stop_raw:
            raise _CompletionError("'stop' must not contain empty strings", param="stop")
        stop = [stop_raw]
    elif isinstance(stop_raw, list) and all(isinstance(item, str) for item in stop_raw):
        if any(not item for item in stop_raw):
            raise _CompletionError("'stop' must not contain empty strings", param="stop")
        stop = [str(item) for item in stop_raw] or None
    else:
        raise _CompletionError("'stop' must be a string or array of strings", param="stop")

    return _CompletionParams(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        stream=stream,
        include_usage=include_usage,
    )


def _system_fingerprint(model: str) -> str:
    try:
        version = package_version("sie-server")
    except PackageNotFoundError:  # pragma: no cover - editable source always has metadata in tests
        version = "unknown"
    payload = f"{model}\0{version}".encode()
    return f"fp_{hashlib.blake2b(payload, digest_size=8).hexdigest()}"


def _finish_reason(value: str | None) -> str:
    if value in {"length", "tool_calls", "content_filter", "function_call"}:
        return value
    return "stop"


def _chunk_body(
    *,
    completion_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "system_fingerprint": _system_fingerprint(model),
        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
    }


async def _stream_completion(
    chunks: AsyncIterator[GenerationChunk],
    *,
    completion_id: str,
    created: int,
    model: str,
    include_usage: bool,
    registry: Any,
) -> AsyncIterator[str]:
    terminal: GenerationChunk | None = None
    terminal_outcome_selected = False
    cleanup_failed = False
    try:
        async for chunk in chunks:
            if chunk.done:
                terminal = chunk
            cancelled = chunk.done and chunk.finish_reason == "cancelled"
            errored = chunk.done and (
                chunk.finish_reason == "error" or chunk.error_code is not None or chunk.error_message is not None
            )
            body = _chunk_body(
                completion_id=completion_id,
                created=created,
                model=model,
                text=chunk.text_delta,
                finish_reason=(
                    None if cancelled or errored else (_finish_reason(chunk.finish_reason) if chunk.done else None)
                ),
            )
            if cancelled:
                body["error"] = {
                    "message": "generation was cancelled before completion",
                    "type": "server_error",
                    "param": None,
                    "code": "generation_cancelled",
                }
            elif errored:
                body["error"] = {
                    "message": client_safe_generation_error_message(chunk.error_code, chunk.error_message),
                    "type": "server_error",
                    "param": None,
                    "code": client_safe_generation_error_code(chunk.error_code),
                }
            yield f"data: {json.dumps(body)}\n\n"
            if chunk.done:
                break
    except Exception as exc:  # noqa: BLE001 - the stream must terminate in-band after headers are committed
        terminal_outcome_selected = True
        if isinstance(exc, GenerationError):
            error = _from_generation_error(exc, registry)
            logger.info("OpenAI completion received typed generation refusal mid-stream: %s", error.code)
        else:
            error = _CompletionError(
                "internal error during generation",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="inference_error",
            )
            logger.warning("OpenAI completion failed mid-stream", exc_info=True)
        body = _chunk_body(
            completion_id=completion_id,
            created=created,
            model=model,
            text="",
            finish_reason=None,
        )
        error_body: dict[str, Any] = {
            "message": error.message,
            "type": error.error_type,
            "param": error.param,
            "code": error.code,
        }
        if (retry_after_s := _validated_retry_after_s(error)) is not None:
            error_body["retry_after_s"] = retry_after_s
        body["error"] = error_body
        yield f"data: {json.dumps(body)}\n\n"
        yield "data: [DONE]\n\n"
        return
    finally:
        try:
            await aclose_with_error_precedence(
                chunks,
                outcome_selected=terminal_outcome_selected or terminal is not None,
                context="OpenAI completion effective iterator",
            )
        except Exception:  # noqa: BLE001 - establish the streaming terminal below
            cleanup_failed = True
            logger.warning("OpenAI completion iterator cleanup failed", exc_info=True)

    if cleanup_failed:
        body = _chunk_body(
            completion_id=completion_id,
            created=created,
            model=model,
            text="",
            finish_reason=None,
        )
        body["error"] = {
            "message": "internal error during generation cleanup",
            "type": "server_error",
            "param": None,
            "code": "inference_error",
        }
        yield f"data: {json.dumps(body)}\n\n"
    elif terminal is None:
        body = _chunk_body(
            completion_id=completion_id,
            created=created,
            model=model,
            text="",
            finish_reason=None,
        )
        body["error"] = {
            "message": "generation stream ended before a terminal event",
            "type": "server_error",
            "param": None,
            "code": "inference_error",
        }
        yield f"data: {json.dumps(body)}\n\n"
    elif (
        terminal.finish_reason not in {"cancelled", "error"}
        and not terminal.error_code
        and not terminal.error_message
        and include_usage
        and terminal.prompt_tokens is not None
        and terminal.completion_tokens is not None
    ):
        usage = {
            "prompt_tokens": terminal.prompt_tokens,
            "completion_tokens": terminal.completion_tokens,
            "total_tokens": terminal.prompt_tokens + terminal.completion_tokens,
        }
        yield f"data: {
            json.dumps(
                {
                    'id': completion_id,
                    'object': 'text_completion',
                    'created': created,
                    'model': model,
                    'system_fingerprint': _system_fingerprint(model),
                    'choices': [],
                    'usage': usage,
                }
            )
        }\n\n"
    yield "data: [DONE]\n\n"


async def _collect_completion(chunks: AsyncIterator[GenerationChunk]) -> tuple[str, GenerationChunk]:
    text_parts: list[str] = []
    terminal: GenerationChunk | None = None
    try:
        async for chunk in chunks:
            if chunk.text_delta:
                text_parts.append(chunk.text_delta)
            if chunk.done:
                terminal = chunk
                break
    finally:
        await aclose_with_error_precedence(
            chunks,
            outcome_selected=terminal is not None,
            context="buffered OpenAI completion iterator",
        )
    if terminal is None:
        raise _CompletionError(
            "generation stream ended before a terminal event",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="inference_error",
        )
    if terminal.finish_reason == "error" or terminal.error_code or terminal.error_message:
        raise _CompletionError(
            client_safe_generation_error_message(terminal.error_code, terminal.error_message),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=client_safe_generation_error_code(terminal.error_code),
        )
    if terminal.finish_reason == "cancelled":
        raise _CompletionError(
            "generation was cancelled before completion",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="generation_cancelled",
        )
    return "".join(text_parts), terminal


@router.post(
    "/completions",
    response_model=None,
    responses={
        200: {
            "description": "OpenAI text completion, or text_completion Server-Sent Events when stream is true",
            "model": OpenAICompletionResponseModel,
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string", "description": "OpenAI text_completion events ending in [DONE]"}
                }
            },
        },
        400: {"description": "Invalid or unsupported request"},
        404: {"description": "Model not found"},
        413: {"description": "Request body or prompt is too large"},
        500: {"description": "Generation failed"},
        503: {"description": "Model loading or temporarily unavailable"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/OpenAICompletionRequestModel"}}},
        }
    },
)
async def completions(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> JSONResponse | StreamingResponse:
    """Generate one raw-prompt OpenAI legacy completion."""
    try:
        validate_machine_profile_header(x_machine_profile)
        check_sdk_version(http_request)
        params = _parse_params(await _read_json_body(http_request))

        registry = http_request.app.state.registry
        registry_key = denormalize_model_id(params.model)
        with tracer.start_as_current_span("openai_completions") as span:
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
            if params.stream and not generate_task.capabilities.streaming:
                raise _CompletionError(
                    f"Model '{params.model}' does not support streaming generation",
                    param="stream",
                    code="unsupported_field",
                )
            if params.max_tokens > generate_task.max_output_tokens:
                raise _CompletionError(
                    f"max_tokens ({params.max_tokens}) exceeds model cap ({generate_task.max_output_tokens})",
                    param="max_tokens",
                    code="context_exceeded",
                )

            runtime_params: dict[str, Any] = {
                "prompt": params.prompt,
                "max_new_tokens": params.max_tokens,
                **{
                    key: value
                    for key, value in (
                        ("temperature", params.temperature),
                        ("top_p", params.top_p),
                        ("stop", params.stop),
                        ("frequency_penalty", params.frequency_penalty),
                        ("presence_penalty", params.presence_penalty),
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
                "prompt": params.prompt,
                "max_new_tokens": params.max_tokens,
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
                adapter.preflight_generate(generation_parameters, stream=params.stream)
            except GenerationError as exc:
                raise _from_generation_error(exc, registry) from exc
            except Exception as exc:
                logger.warning("OpenAI completion preflight failed for %s", params.model, exc_info=True)
                raise _CompletionError(
                    "internal error during generation",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    code="inference_error",
                ) from exc

            canonical_model = getattr(config, "name", None) or registry_key
            completion_id = f"cmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            try:
                chunks = adapter.generate(**generation_parameters)
            except GenerationError as exc:
                raise _from_generation_error(exc, registry) from exc
            if thinking_blocks_must_be_hidden(config):
                reasoning_format = resolve_reasoning_format(config, adapter)
                chunks = suppress_thinking_blocks(
                    chunks,
                    start_inside=reasoning_starts_in_prompt(params.prompt, reasoning_format),
                    reasoning_format=reasoning_format,
                )
            if params.stream:
                return StreamingResponse(
                    _stream_completion(
                        chunks,
                        completion_id=completion_id,
                        created=created,
                        model=canonical_model,
                        include_usage=params.include_usage,
                        registry=registry,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            try:
                text, terminal = await _collect_completion(chunks)
            except _CompletionError:
                raise
            except GenerationError as exc:
                raise _from_generation_error(exc, registry) from exc
            except Exception as exc:
                logger.warning("OpenAI completion failed for %s", params.model, exc_info=True)
                raise _CompletionError(
                    "internal error during generation",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    code="inference_error",
                ) from exc
            prompt_tokens = terminal.prompt_tokens or 0
            completion_tokens = terminal.completion_tokens or 0
            return JSONResponse(
                content={
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": canonical_model,
                    "choices": [
                        {
                            "text": text,
                            "index": 0,
                            "finish_reason": _finish_reason(terminal.finish_reason),
                        }
                    ],
                    "system_fingerprint": _system_fingerprint(canonical_model),
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
    except _CompletionError as exc:
        return _error_response(exc)
    except HTTPException as exc:
        return _error_response(_from_http_exception(exc))
