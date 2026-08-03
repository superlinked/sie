"""OpenAI-compatible audio transcription backed by native ``extract``."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from sie_sdk.queue_types import denormalize_model_id
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from sie_server.adapters.errors import InputTooLongError
from sie_server.api.extract import _extract_via_worker
from sie_server.api.helpers import (
    InferenceErrorHandler,
    ModelStateChecker,
    ResponseBuilder,
    check_sdk_version,
    oom_retry_after_from_registry,
)
from sie_server.api.options import resolve_runtime_options
from sie_server.api.validation import validate_machine_profile_header
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.worker import QueueFullError
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.observability.tracing import tracer
from sie_server.observability.worker_telemetry import worker_telemetry, worker_telemetry_enabled
from sie_server.types.inputs import AudioInput, Item
from sie_server.types.responses import ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

_MAX_AUDIO_FILE_BYTES = 24 * 1024 * 1024
_MAX_MULTIPART_BYTES = _MAX_AUDIO_FILE_BYTES + 1024 * 1024
_MAX_FORM_FIELDS = 16
_SERVER_ERROR_STATUS = status.HTTP_500_INTERNAL_SERVER_ERROR
_MAX_TEXT_FIELD_BYTES = 8 * 1024
_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")
_SUPPORTED_FORMATS = frozenset({"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"})
_SUPPORTED_RESPONSE_FORMATS = frozenset({"json", "text", "srt", "verbose_json", "vtt"})
_TIMESTAMP_FIELDS = frozenset({"timestamp_granularities", "timestamp_granularities[]"})
_SINGLE_TEXT_FIELDS = frozenset({"model", "language", "prompt", "response_format", "temperature", "stream"})
_ALLOWED_FIELDS = _SINGLE_TEXT_FIELDS | _TIMESTAMP_FIELDS | {"file"}
_TRANSCRIPTION_OPENAPI: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file", "model"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "model": {"type": "string"},
                        "language": {"type": "string"},
                        "prompt": {"type": "string"},
                        "response_format": {
                            "type": "string",
                            "enum": ["json", "text", "srt", "verbose_json", "vtt"],
                            "default": "json",
                        },
                        "temperature": {"type": "number", "minimum": 0, "maximum": 1},
                        "stream": {"type": "boolean", "enum": [False], "default": False},
                        "timestamp_granularities[]": {
                            "type": "array",
                            "maxItems": 2,
                            "items": {"type": "string", "enum": ["word", "segment"]},
                        },
                    },
                    "additionalProperties": False,
                }
            }
        },
    }
}


@dataclass(slots=True)
class _TranscriptionForm:
    audio: bytes
    audio_format: str
    model: str
    language: str | None
    prompt: str | None
    response_format: str
    temperature: float | None
    timestamp_granularities: list[str]


class _TranscriptionRequestError(Exception):
    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str = "invalid_request",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code
        self.status_code = status_code


def _openai_error(
    message: str,
    *,
    status_code: int,
    param: str | None = None,
    code: str = "invalid_request",
    error_type: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type
                or ("invalid_request_error" if status_code < _SERVER_ERROR_STATUS else "server_error"),
                "param": param,
                "code": code,
            }
        },
    )


def _request_error_response(error: _TranscriptionRequestError) -> JSONResponse:
    return _openai_error(
        error.message,
        status_code=error.status_code,
        param=error.param,
        code=error.code,
    )


def _http_error_response(error: HTTPException) -> JSONResponse:
    detail: dict[str, Any] = error.detail if isinstance(error.detail, dict) else {}
    nested_error = detail.get("error")
    inner: dict[str, Any] = nested_error if isinstance(nested_error, dict) else detail
    native_code = str(inner.get("code", ""))
    message = str(
        inner.get(
            "message", "internal server error" if error.status_code >= _SERVER_ERROR_STATUS else "invalid request"
        )
    )
    param = inner.get("param")
    if native_code == ErrorCode.MODEL_NOT_FOUND.value:
        code = "model_not_found"
        error_type = "invalid_request_error"
    elif error.status_code >= _SERVER_ERROR_STATUS:
        code = "transport_failure"
        error_type = "server_error"
    else:
        code = "invalid_request"
        error_type = "invalid_request_error"
    return _openai_error(
        message,
        status_code=error.status_code,
        param=param if isinstance(param, str) else None,
        code=code,
        error_type=error_type,
        headers=error.headers,
    )


async def _bounded_body_stream(request: Request) -> AsyncGenerator[bytes]:
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length.isdigit() and int(content_length) > _MAX_MULTIPART_BYTES:
        raise _TranscriptionRequestError(
            f"multipart body exceeds {_MAX_MULTIPART_BYTES} bytes",
            code="payload_too_large",
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_MULTIPART_BYTES:
            raise _TranscriptionRequestError(
                f"multipart body exceeds {_MAX_MULTIPART_BYTES} bytes",
                code="payload_too_large",
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
        yield chunk


async def _parse_transcription_form(request: Request) -> _TranscriptionForm:
    try:
        form = await MultiPartParser(
            request.headers,
            _bounded_body_stream(request),
            max_files=1,
            max_fields=_MAX_FORM_FIELDS - 1,
            max_part_size=_MAX_TEXT_FIELD_BYTES,
        ).parse()
    except _TranscriptionRequestError:
        raise
    except MultiPartException as exc:
        if str(exc).startswith("Part exceeded maximum size"):
            raise _TranscriptionRequestError(
                f"text field exceeds {_MAX_TEXT_FIELD_BYTES} bytes",
                code="payload_too_large",
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            ) from exc
        raise _TranscriptionRequestError("request must be valid multipart/form-data") from exc
    except (KeyError, ValueError) as exc:
        raise _TranscriptionRequestError("request must be valid multipart/form-data") from exc

    file: UploadFile | None = None
    values: dict[str, str] = {}
    granularities: list[str] = []
    try:
        for name, value in form.multi_items():
            if name not in _ALLOWED_FIELDS:
                raise _TranscriptionRequestError(
                    f"unsupported field '{name}'",
                    param=name,
                    code="unsupported_field",
                )
            if name == "file":
                if file is not None:
                    raise _TranscriptionRequestError("field 'file' must appear once", param="file")
                if not isinstance(value, UploadFile):
                    raise _TranscriptionRequestError("field 'file' must be a file upload", param="file")
                file = value
                continue
            if not isinstance(value, str):
                raise _TranscriptionRequestError(f"field '{name}' must be text", param=name)
            if name in _TIMESTAMP_FIELDS:
                granularities.append(value)
                continue
            if name in values:
                raise _TranscriptionRequestError(f"field '{name}' must appear once", param=name)
            values[name] = value

        if file is None:
            raise _TranscriptionRequestError("field 'file' is required", param="file")
        filename = file.filename or ""
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in _SUPPORTED_FORMATS:
            raise _TranscriptionRequestError(
                "unsupported audio file format; expected flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm",
                param="file",
            )
        audio = await file.read(_MAX_AUDIO_FILE_BYTES + 1)
        if not audio:
            raise _TranscriptionRequestError("field 'file' must not be empty", param="file")
        if len(audio) > _MAX_AUDIO_FILE_BYTES:
            raise _TranscriptionRequestError(
                f"audio file exceeds {_MAX_AUDIO_FILE_BYTES} bytes",
                param="file",
                code="payload_too_large",
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
    finally:
        await form.close()

    model = values.get("model", "").strip()
    if not model:
        raise _TranscriptionRequestError("field 'model' is required", param="model")
    if ".." in model or "\\" in model or not _MODEL_ID_PATTERN.fullmatch(model):
        raise _TranscriptionRequestError(
            "invalid model id for path",
            param="model",
        )

    language = values.get("language")
    if language is not None:
        language = language.strip()
        if not language:
            raise _TranscriptionRequestError("field 'language' must not be empty", param="language")

    prompt = values.get("prompt")
    response_format = values.get("response_format", "json")
    if response_format == "diarized_json":
        raise _TranscriptionRequestError(
            "response_format 'diarized_json' is not supported by this model",
            param="response_format",
            code="unsupported_field",
        )
    if response_format not in _SUPPORTED_RESPONSE_FORMATS:
        raise _TranscriptionRequestError(
            "response_format must be json, text, srt, verbose_json, or vtt",
            param="response_format",
        )

    stream = values.get("stream", "false")
    if stream not in {"true", "false"}:
        raise _TranscriptionRequestError("field 'stream' must be true or false", param="stream")
    if stream == "true":
        raise _TranscriptionRequestError(
            "streaming transcription is not supported by the native extract primitive",
            param="stream",
            code="unsupported_field",
        )

    temperature: float | None = None
    raw_temperature = values.get("temperature")
    if raw_temperature is not None:
        try:
            temperature = float(raw_temperature)
        except ValueError as exc:
            raise _TranscriptionRequestError(
                "temperature must be a number between 0 and 1", param="temperature"
            ) from exc
        if not math.isfinite(temperature) or not 0 <= temperature <= 1:
            raise _TranscriptionRequestError("temperature must be a number between 0 and 1", param="temperature")

    if any(value not in {"word", "segment"} for value in granularities):
        raise _TranscriptionRequestError(
            "timestamp_granularities must contain only 'word' or 'segment'",
            param="timestamp_granularities",
        )
    granularities = list(dict.fromkeys(granularities))
    if granularities and response_format != "verbose_json":
        raise _TranscriptionRequestError(
            "timestamp_granularities requires response_format='verbose_json'",
            param="timestamp_granularities",
        )
    if response_format in {"srt", "vtt"}:
        granularities = ["segment"]

    return _TranscriptionForm(
        audio=audio,
        audio_format=extension,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=granularities,
    )


def _duration_seconds(data: dict[str, Any]) -> float:
    duration_ms = data.get("duration_ms")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ValueError("extract response is missing a positive integer duration_ms")
    return float(duration_ms) / 1_000


def _timestamp(value: Any, *, separator: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("extract response contains an invalid timestamp")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("extract response contains an invalid timestamp")
    total_ms = round(float(value) * 1_000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def _subtitle(data: dict[str, Any], *, vtt: bool) -> str:
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("extract response is missing segment timestamps")
    separator = "." if vtt else ","
    blocks = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError("extract response contains an invalid segment")
        segment_data = cast("dict[str, Any]", segment)
        start = _timestamp(segment_data.get("start"), separator=separator)
        end = _timestamp(segment_data.get("end"), separator=separator)
        text = segment_data.get("text")
        if not isinstance(text, str):
            raise ValueError("extract response contains invalid segment text")
        text = text.strip()
        prefix = "" if vtt else f"{index}\n"
        blocks.append(f"{prefix}{start} --> {end}\n{text}")
    content = "\n\n".join(blocks) + "\n"
    return f"WEBVTT\n\n{content}" if vtt else content


def _transcription_response(
    data: dict[str, Any],
    form: _TranscriptionForm,
    headers: dict[str, str],
) -> Response:
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError("extract response is missing text")
    duration = _duration_seconds(data)
    usage = {"type": "duration", "seconds": duration}
    if form.response_format == "json":
        return JSONResponse(content={"text": text, "usage": usage}, headers=headers)
    if form.response_format == "text":
        return PlainTextResponse(text, headers=headers)
    if form.response_format == "srt":
        return PlainTextResponse(_subtitle(data, vtt=False), media_type="application/x-subrip", headers=headers)
    if form.response_format == "vtt":
        return PlainTextResponse(_subtitle(data, vtt=True), media_type="text/vtt", headers=headers)

    content: dict[str, Any] = {
        "task": "transcribe",
        "language": data.get("language") or form.language or "unknown",
        "duration": duration,
        "text": text,
        "usage": usage,
    }
    if "segment" in form.timestamp_granularities:
        content["segments"] = data.get("segments", [])
    if "word" in form.timestamp_granularities:
        content["words"] = data.get("words", [])
    return JSONResponse(content=content, headers=headers)


@router.post("/audio/transcriptions", response_model=None, openapi_extra=_TRANSCRIPTION_OPENAPI)
async def create_transcription(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> Response:
    """Transcribe a multipart audio upload through SIE's native extract path."""
    try:
        validate_machine_profile_header(x_machine_profile)
        check_sdk_version(http_request)
        form = await _parse_transcription_form(http_request)
    except _TranscriptionRequestError as error:
        return _request_error_response(error)
    except HTTPException as error:
        return _http_error_response(error)

    registry_key = denormalize_model_id(form.model)
    registry = http_request.app.state.registry
    device = registry.device
    try:
        with tracer.start_as_current_span("openai_audio_transcription") as span:
            span.set_attribute("model", registry_key)
            checker = ModelStateChecker(registry, registry_key, span)
            checker.check_exists()
            config = registry.get_config(registry_key)
            if config.tasks.extract is None or not config.inputs.audio:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": ErrorCode.INVALID_INPUT.value,
                        "message": f"Model '{form.model}' does not support audio extraction",
                        "param": "model",
                    },
                )
            checker.check_not_unloading()
            checker.check_not_loading()
            await checker.ensure_loaded(device)

            raw_options: dict[str, Any] = {"timestamp_granularities": form.timestamp_granularities}
            if form.language is not None:
                raw_options["language"] = form.language
            if form.temperature is not None:
                raw_options["temperature"] = form.temperature
            options = resolve_runtime_options(config, raw_options, span)
            item = Item(audio=AudioInput(data=form.audio, format=form.audio_format))
            error_handler = InferenceErrorHandler(
                registry_key,
                "extract",
                span,
                oom_retry_after_s=oom_retry_after_from_registry(registry),
            )
            try:
                worker_result = await _extract_via_worker(
                    registry,
                    registry_key,
                    [item],
                    instruction=form.prompt,
                    options=options,
                )
                extract_output = cast(
                    "ExtractOutput",
                    worker_result.output,
                )
                output = ExtractHandler.format_output(extract_output)
            except QueueFullError as error:
                raise error_handler.handle_queue_full(error) from error
            except InputTooLongError as error:
                raise error_handler.handle_input_too_long(error) from error
            except ValueError as error:
                raise error_handler.handle_value_error(error) from error
            except Exception as error:
                raise error_handler.handle_inference_error(error, "Transcription") from error

            if len(output) != 1 or not isinstance(output[0].get("data"), dict):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"code": ErrorCode.INFERENCE_ERROR.value, "message": "malformed extract response"},
                )
            headers = ResponseBuilder.build_headers(worker_result.timing)
            response = _transcription_response(output[0]["data"], form, headers)
            if worker_telemetry_enabled():
                worker_telemetry().item_completed(
                    operation="extract",
                    outcome="success",
                    model=registry_key,
                    profile="default",
                    duration_s=worker_result.timing.total_ms / 1_000.0,
                    item_count=1,
                    tokenization_s=worker_result.timing.tokenization_ms / 1_000.0,
                    inference_s=worker_result.timing.inference_ms / 1_000.0,
                    postprocessing_s=worker_result.timing.postprocessing_ms / 1_000.0,
                )
            return response
    except HTTPException as error:
        return _http_error_response(error)
    except ValueError:
        logger.exception("Failed to format transcription response for %s", registry_key)
        return _openai_error(
            "internal server error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_error",
        )
