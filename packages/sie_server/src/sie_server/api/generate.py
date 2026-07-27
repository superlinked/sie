"""Direct HTTP route for native generation (walking-skeleton local-dev path).

This is the **local-dev** counterpart of the gateway's ``proxy_generate`` —
it bypasses NATS/JetStream entirely and calls the
:class:`~sie_server.adapters._generation_base.GenerationAdapter` directly. The
same model config / adapter is exercised here as on the queue path; only the
transport differs.

Why ship a direct route at all? Two reasons:

1. End-to-end viability checking: a developer can run
   ``mise run serve -m Qwen/Qwen3-4B-Instruct -b sglang`` and immediately
   curl ``/v1/generate/...`` against the worker to confirm the
   adapter + registry + model config plumbing works against a real GPU,
   without needing to boot the Rust gateway and NATS first.

2. Integration tests under ``mise run test -- -i`` already speak to the
   Python server via the ``sie_client`` / ``sie_server`` fixtures; this
   route gives those tests a generation surface to validate before the
   streaming rollout lands the SDK :meth:`generate` method.

Request shape mirrors the gateway's walking-skeleton contract verbatim:

.. code-block:: json

   { "prompt": "...", "images": [{"data": "<base64>", "format": "png"}],
     "max_new_tokens": 64, "temperature": 0.7, "top_p": 0.9,
     "stop": ["</s>"] }

Response shape::

   {
       "model": "...",
       "text": "...",
       "finish_reason": "stop" | "length",
       "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
   }
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sie_sdk.queue_types import denormalize_model_id

from sie_server.adapters._generation_base import GenerationAdapter, collect_generation
from sie_server.api.helpers import ModelStateChecker
from sie_server.api.validation import validate_machine_profile_header, validate_signed_i64
from sie_server.core.runtime_options import apply_generation_runtime_options
from sie_server.core.tokenizer import image_first_chat_message, load_tokenizer
from sie_server.observability.tracing import tracer
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput
from sie_server.types.openapi import (
    GenerateInputTooLongErrorResponse,
    GenerateModelLoadFailedErrorResponse,
    GenerateResponseModel,
)
from sie_server.types.responses import ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["generate"])


# Field whitelist — matches the gateway's ``proxy_generate`` validation
# (see ``packages/sie_gateway/src/handlers/proxy.rs::generate_params_from_json``)
# and the OpenAPI schema published at
# ``packages/sie_gateway/openapi.json#/components/schemas/GenerateRequest``.
# Fields beyond the original walking-skeleton subset are accepted to keep
# the worker-local dev route from rejecting requests built against the
# published contract.
#
# Three tiers of handling:
#
# * Forwarded to the adapter and surfaced in the blocking response:
#   ``prompt`` / ``max_new_tokens`` / ``temperature`` / ``top_p`` /
#   ``stop``, plus ``seed`` / ``logit_bias``. The adapter's ``generate()``
#   accepts both and the production queue path forwards them too (see
#   ``processors/streaming.py``); their exact effect is backend-specific.
# * Forwarded only for streaming requests: ``logprobs`` /
#   ``top_logprobs``. Blocking requests reject them because the aggregate
#   ``GenerationResult`` has no logprob field.
# * Inert / accept-and-drop transport hints: ``routing_key`` /
#   ``prompt_cache_key`` / ``safety_identifier``.
_SUPPORTED_FIELDS = {
    "prompt",
    "images",
    "max_new_tokens",
    "temperature",
    "top_p",
    "stop",
    "stream",
    "frequency_penalty",
    "presence_penalty",
    "grammar",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "routing_key",
    "prompt_cache_key",
    "safety_identifier",
    "options",
}


# Maximum prompt size accepted by this direct route, in UTF-8 bytes. The
# gateway caps the whole body at 24 MiB so one 16 MiB decoded inline image
# fits after base64 expansion. This worker-local dev route never sits behind
# the gateway, so it independently caps both the prompt and request body.
# Override the prompt limit via ``SIE_GENERATE_MAX_PROMPT_BYTES``.
_MAX_PROMPT_BYTES = int(os.environ.get("SIE_GENERATE_MAX_PROMPT_BYTES", str(4 * 1024 * 1024)))
_MAX_GENERATE_BODY_BYTES = int(os.environ.get("SIE_GENERATE_MAX_BODY_BYTES", str(24 * 1024 * 1024)))
_MAX_GENERATE_IMAGES = 16
_MAX_GENERATE_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_GENERATE_IMAGE_BASE64_CHARS = 4 * ((_MAX_GENERATE_IMAGE_BYTES + 2) // 3)
_MAX_GENERATE_IMAGE_FORMAT_LENGTH = 32
_MAX_GRAMMAR_BYTES = 64 * 1024
_MAX_SCHEMA_DEPTH = 16
_MAX_SCHEMA_TRAVERSAL_DEPTH = 128
_MAX_SCHEMA_NODES = 16 * 1024
_MAX_REGEX_LENGTH = 4 * 1024
_MAX_EBNF_LENGTH = 8 * 1024
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$dynamicRef",
        "if",
        "then",
        "else",
        "unevaluatedProperties",
        "dependentSchemas",
    }
)
_NATIVE_TOKENIZER_CACHE_SIZE = 16
_NATIVE_TOKENIZER_LOAD_LOCK = threading.Lock()

# OpenAI penalty range (mirrors the gateway's ``proxy.rs::parse_penalty``):
# ``frequency_penalty`` / ``presence_penalty`` must be a finite number in
# ``[_PENALTY_MIN, _PENALTY_MAX]``.
_PENALTY_MIN = -2.0
_PENALTY_MAX = 2.0

# ``logit_bias`` map-size cap (mirrors the gateway's ``MAX_LOGIT_BIAS_KEYS``
# in ``proxy.rs``) so an oversized payload cannot DoS the worker's sampler.
_MAX_LOGIT_BIAS_KEYS = 1024
# Per-value range for ``logit_bias`` (gateway parity, ``proxy.rs``).
_LOGIT_BIAS_MIN = -100.0
_LOGIT_BIAS_MAX = 100.0
# ``top_logprobs`` upper bound (OpenAI spec / gateway ``proxy.rs``: [0, 20]).
_TOP_LOGPROBS_MAX = 20


def _bad_request(message: str, *, param: str | None = None, code: str | None = None) -> HTTPException:
    detail: dict[str, Any] = {
        "code": code or "INVALID_REQUEST",
        "message": message,
    }
    if param is not None:
        detail["param"] = param
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _validate_penalty(value: Any, *, param: str) -> float | None:
    """Validate ``frequency_penalty`` / ``presence_penalty`` (gateway parity).

    Mirrors ``proxy.rs::parse_penalty``: ``None`` is allowed (field absent →
    worker default); otherwise the value must be a finite JSON number in
    ``[-2.0, 2.0]``. Booleans are rejected explicitly (``isinstance(True,
    int)`` is True in Python) and so are strings / NaN / inf. The value is
    returned value is forwarded to the generation adapter.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _bad_request(f"'{param}' must be a number in [-2.0, 2.0]", param=param)
    f = float(value)
    if not math.isfinite(f) or not (_PENALTY_MIN <= f <= _PENALTY_MAX):
        raise _bad_request(f"'{param}' must be a number in [-2.0, 2.0]", param=param)
    return f


def _validate_seed(value: Any) -> int | None:
    """Validate ``seed`` (gateway parity) and return the parsed value.

    Mirrors ``proxy.rs``: ``None`` (absent) is allowed; otherwise the value
    must be a signed 64-bit integer and is returned unchanged. Booleans are
    rejected explicitly (``isinstance(True, int)`` is True in Python).
    """
    try:
        return validate_signed_i64(value, param="seed")
    except ValueError as exc:
        raise _bad_request(str(exc), param="seed") from exc


def _validate_logit_bias(value: Any) -> dict[str, float] | None:
    """Validate ``logit_bias`` (gateway parity) and return the parsed map.

    Mirrors ``proxy.rs``: ``None`` (absent) is allowed; otherwise the value
    must be an object mapping integer-token-id strings to finite numbers in
    ``[-100.0, 100.0]``, capped at ``_MAX_LOGIT_BIAS_KEYS`` entries. An empty
    map is treated as absent (``None``). The adapter forwards ``logit_bias``
    to SGLang so it is returned (not dropped).
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _bad_request("'logit_bias' must be an object", param="logit_bias")
    if len(value) > _MAX_LOGIT_BIAS_KEYS:
        raise _bad_request(
            f"'logit_bias' has too many entries (max {_MAX_LOGIT_BIAS_KEYS})",
            param="logit_bias",
        )
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            int(key)
        except (TypeError, ValueError) as exc:
            raise _bad_request(
                f"'logit_bias' keys must be token-id integers as strings (got {key!r})",
                param="logit_bias",
            ) from exc
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise _bad_request("'logit_bias' values must be finite numbers", param="logit_bias")
        f = float(raw)
        if not math.isfinite(f):
            raise _bad_request("'logit_bias' values must be finite numbers", param="logit_bias")
        if not (_LOGIT_BIAS_MIN <= f <= _LOGIT_BIAS_MAX):
            raise _bad_request("'logit_bias' values must be in [-100.0, 100.0]", param="logit_bias")
        out[key] = f
    return out or None


def _validate_logprobs(logprobs_value: Any, top_logprobs_value: Any) -> tuple[bool, int | None]:
    """Validate and return native streaming logprob controls.

    Mirrors ``proxy.rs``: ``logprobs`` must be a boolean (or absent);
    ``top_logprobs`` must be an integer in ``[0, 20]`` (or absent) and
    requires ``logprobs: true`` when ``> 0``. The blocking dev-route shape
    has no per-token logprob field, so the caller rejects them on the blocking
    path.
    """
    logprobs_enabled: bool | None
    if logprobs_value is None:
        logprobs_enabled = None
    elif isinstance(logprobs_value, bool):
        logprobs_enabled = logprobs_value
    else:
        raise _bad_request("'logprobs' must be a boolean", param="logprobs")

    if top_logprobs_value is None:
        return bool(logprobs_enabled), None
    if isinstance(top_logprobs_value, bool) or not isinstance(top_logprobs_value, int):
        raise _bad_request("'top_logprobs' must be an integer in [0, 20]", param="top_logprobs")
    if not (0 <= top_logprobs_value <= _TOP_LOGPROBS_MAX):
        raise _bad_request("'top_logprobs' must be an integer in [0, 20]", param="top_logprobs")
    if top_logprobs_value > 0 and logprobs_enabled is not True:
        raise _bad_request("'top_logprobs' requires 'logprobs: true'", param="top_logprobs")
    return bool(logprobs_enabled), top_logprobs_value


def _parse_native_images(value: Any) -> list[ImageInput] | None:
    """Validate and decode the native JSON image envelope."""
    if value is None:
        return None
    if not isinstance(value, list) or not (1 <= len(value) <= _MAX_GENERATE_IMAGES):
        raise _bad_request(
            f"'images' must contain between 1 and {_MAX_GENERATE_IMAGES} entries",
            param="images",
        )
    images: list[ImageInput] = []
    for index, entry in enumerate(value):
        owner = f"images[{index}]"
        if not isinstance(entry, dict):
            raise _bad_request(f"'{owner}' must be an object", param=owner)
        entry_dict = cast("dict[str, Any]", entry)
        unknown = set(entry_dict) - {"data", "format"}
        if unknown:
            param = f"{owner}.{sorted(unknown)[0]}"
            raise _bad_request(f"'{param}' is not supported", param=param, code="unsupported_field")
        encoded = entry_dict.get("data")
        data_owner = f"{owner}.data"
        if not isinstance(encoded, str) or not encoded:
            raise _bad_request(f"'{data_owner}' must be a non-empty base64 string", param=data_owner)
        if len(encoded) > _MAX_GENERATE_IMAGE_BASE64_CHARS:
            raise _bad_request(f"'{data_owner}' exceeds the 16 MiB decoded-image limit", param=data_owner)
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise _bad_request(f"'{data_owner}' must be valid standard base64", param=data_owner) from exc
        if base64.b64encode(data).decode("ascii") != encoded:
            raise _bad_request(f"'{data_owner}' must use canonical standard base64", param=data_owner)
        if not data or len(data) > _MAX_GENERATE_IMAGE_BYTES:
            raise _bad_request(
                f"'{data_owner}' must decode to between 1 byte and 16 MiB",
                param=data_owner,
            )
        format_value = entry_dict.get("format")
        if format_value is not None and (
            not isinstance(format_value, str)
            or not (1 <= len(format_value) <= _MAX_GENERATE_IMAGE_FORMAT_LENGTH)
            or not all(
                character.isascii() and (character.isalnum() or character in ".+-") for character in format_value
            )
        ):
            format_owner = f"{owner}.format"
            raise _bad_request(f"'{format_owner}' must be a short media-format token", param=format_owner)
        images.append({"data": data, "format": format_value.lower() if format_value else None})
    return images


def _schema_child_context(parent: str, key: str) -> str:
    if parent == "schema":
        if key in {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}:
            return "schema_map"
        if key in {"oneOf", "anyOf", "allOf", "prefixItems"}:
            return "schema_array"
        if key in {
            "items",
            "additionalProperties",
            "contains",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
        }:
            return "schema"
        return "other"
    if parent == "schema_map":
        return "schema"
    return "other"


def _json_pointer(root: Any, pointer: str) -> Any:
    current = root
    if not pointer:
        return current
    for raw_token in pointer.removeprefix("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isascii() and token.isdecimal():
            try:
                index = int(token)
            except ValueError as exc:
                raise KeyError(pointer) from exc
            if index >= len(current):
                raise KeyError(pointer)
            current = current[index]
        else:
            raise KeyError(pointer)
    return current


def _dereference_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    visited = 0
    stack: list[str] = []

    def resolve(value: Any, path: str, context: str, traversal_depth: int) -> Any:
        nonlocal visited
        visited += 1
        if visited > _MAX_SCHEMA_NODES:
            raise _bad_request(
                f"JSON Schema node count exceeds limit ({_MAX_SCHEMA_NODES})",
                param=path,
            )
        if traversal_depth > _MAX_SCHEMA_TRAVERSAL_DEPTH:
            raise _bad_request(
                f"JSON Schema traversal depth exceeds limit ({_MAX_SCHEMA_TRAVERSAL_DEPTH})",
                param=path,
            )
        if isinstance(value, dict):
            if context == "schema" and "$ref" in value:
                ref = value["$ref"]
                ref_path = f"{path}.$ref"
                if not isinstance(ref, str):
                    raise _bad_request("'$ref' must be a string", param=ref_path)
                if not ref.startswith("#"):
                    raise _bad_request("external '$ref' is not supported", param=ref_path, code="unsupported_field")
                pointer = ref[1:]
                if pointer and not pointer.startswith("/"):
                    raise _bad_request(
                        "only internal JSON-pointer '$ref' values are supported",
                        param=ref_path,
                        code="unsupported_field",
                    )
                if pointer in stack:
                    raise _bad_request(f"recursive '$ref' cycle detected at {ref!r}", param=ref_path)
                try:
                    target = _json_pointer(schema, pointer)
                except KeyError as exc:
                    raise _bad_request(f"unresolved internal '$ref' {ref!r}", param=ref_path) from exc
                stack.append(pointer)
                resolved = resolve(target, path, "schema", traversal_depth + 1)
                stack.pop()
                siblings = {
                    key: resolve(
                        child,
                        f"{path}.{key}",
                        _schema_child_context(context, key),
                        traversal_depth + 1,
                    )
                    for key, child in value.items()
                    if key not in {"$ref", "$defs", "definitions"}
                }
                return resolved if not siblings else {"allOf": [resolved, siblings]}

            return {
                key: resolve(
                    child,
                    f"{path}.{key}",
                    _schema_child_context(context, key),
                    traversal_depth + 1,
                )
                for key, child in value.items()
                if not (context == "schema" and key in {"$defs", "definitions"})
            }
        if isinstance(value, list):
            child_context = "schema" if context in {"schema", "schema_array"} else "other"
            return [
                resolve(child, f"{path}[{index}]", child_context, traversal_depth + 1)
                for index, child in enumerate(value)
            ]
        return value

    resolved = resolve(schema, "grammar.json_schema", "schema", 0)
    return cast("dict[str, Any]", resolved)


def _validate_schema_shape(schema: Any) -> None:
    visited = 0
    nesting_keys = {
        "properties",
        "patternProperties",
        "additionalProperties",
        "unevaluatedProperties",
        "items",
        "prefixItems",
        "contains",
        "propertyNames",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "definitions",
        "$defs",
        "dependentSchemas",
        "if",
        "then",
        "else",
    }

    def walk(value: Any, path: str, depth: int, context: str, traversal_depth: int) -> None:
        nonlocal visited
        visited += 1
        if visited > _MAX_SCHEMA_NODES:
            raise _bad_request(
                f"JSON Schema node count exceeds limit ({_MAX_SCHEMA_NODES})",
                param=path,
            )
        if traversal_depth > _MAX_SCHEMA_TRAVERSAL_DEPTH:
            raise _bad_request(
                f"JSON Schema traversal depth exceeds limit ({_MAX_SCHEMA_TRAVERSAL_DEPTH})",
                param=path,
            )
        if depth > _MAX_SCHEMA_DEPTH:
            raise _bad_request(f"JSON Schema depth exceeds limit ({_MAX_SCHEMA_DEPTH})", param=path)
        if isinstance(value, dict):
            unsupported = _UNSUPPORTED_SCHEMA_KEYWORDS.intersection(value) if context == "schema" else set()
            if unsupported:
                keyword = sorted(unsupported)[0]
                raise _bad_request(
                    f"JSON Schema keyword '{keyword}' is not supported",
                    param=f"{path}.{keyword}",
                    code="unsupported_field",
                )
            for key, child in value.items():
                child_context = _schema_child_context(context, key)
                child_depth = depth + 1 if context == "schema" and key in nesting_keys else depth
                walk(
                    child,
                    f"{path}.{key}",
                    child_depth,
                    child_context,
                    traversal_depth + 1,
                )
        elif isinstance(value, list):
            child_context = "schema" if context in {"schema", "schema_array"} else "other"
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", depth, child_context, traversal_depth + 1)

    walk(schema, "grammar.json_schema", 0, "schema", 0)


def _parse_native_grammar(value: Any) -> GrammarSpec | None:
    """Validate the public native grammar envelope and build the adapter type."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _bad_request("'grammar' must be a JSON object", param="grammar")
    encoded_size = len(json.dumps(value, separators=(",", ":")).encode("utf-8"))
    if encoded_size > _MAX_GRAMMAR_BYTES:
        raise _bad_request(
            f"grammar payload {encoded_size} bytes exceeds limit ({_MAX_GRAMMAR_BYTES} bytes)",
            param="grammar",
        )
    unknown = set(value) - {"json_schema", "regex", "ebnf", "label", "strict"}
    if unknown:
        field = sorted(unknown)[0]
        raise _bad_request(f"'grammar.{field}' is not supported", param=f"grammar.{field}", code="unsupported_field")
    kinds = [kind for kind in ("json_schema", "regex", "ebnf") if kind in value]
    if len(kinds) != 1:
        raise _bad_request(
            "'grammar' must contain exactly one of 'json_schema', 'regex' or 'ebnf'",
            param="grammar",
        )
    label = value.get("label")
    if label is not None and not isinstance(label, str):
        raise _bad_request("'grammar.label' must be a string", param="grammar.label")
    strict = value.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise _bad_request("'grammar.strict' must be a boolean", param="grammar.strict")

    kind = kinds[0]
    payload = value[kind]
    if kind == "json_schema":
        if not isinstance(payload, dict):
            raise _bad_request("'grammar.json_schema' must be an object", param="grammar.json_schema")
        payload = _dereference_schema_refs(cast("dict[str, Any]", payload))
        _validate_schema_shape(payload)
        resolved_size = len(
            json.dumps(
                {
                    "json_schema": payload,
                    **({"label": label} if label is not None else {}),
                    **({"strict": strict} if strict is not None else {}),
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if resolved_size > _MAX_GRAMMAR_BYTES:
            raise _bad_request(
                f"grammar payload {resolved_size} bytes exceeds limit ({_MAX_GRAMMAR_BYTES} bytes)",
                param="grammar",
            )
    else:
        limit = _MAX_REGEX_LENGTH if kind == "regex" else _MAX_EBNF_LENGTH
        if not isinstance(payload, str):
            raise _bad_request(f"'grammar.{kind}' must be a string", param=f"grammar.{kind}")
        if len(payload) > limit:
            raise _bad_request(f"{kind} length {len(payload)} exceeds limit ({limit})", param=f"grammar.{kind}")

    return GrammarSpec(kind=cast("Any", kind), value=payload, label=label, strict=strict)


async def _render_native_image_prompt(config: Any, prompt: str, image_count: int) -> str:
    """Render one image-aware user turn with the model's own chat template."""
    source = config.hf_id or config.weights_path
    if not isinstance(source, str | Path):
        raise _bad_request("model has no tokenizer source for image generation", param="images")
    revision = config.hf_revision if config.hf_id else None
    try:
        tokenizer = await asyncio.to_thread(
            _load_native_tokenizer_coalesced,
            str(source),
            revision,
        )
        message = image_first_chat_message(role="user", text=prompt, image_count=image_count)
        kwargs = dict(config.tasks.generate.chat_template_kwargs or {})
        apply_chat_template = cast("Any", tokenizer.apply_chat_template)
        rendered = await asyncio.to_thread(
            apply_chat_template,
            [message],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    except Exception as exc:
        logger.info("native image prompt render failed for %s: %s", config.name, exc)
        raise _bad_request("failed to render the model-native image prompt", param="images") from exc
    if not isinstance(rendered, str) or not rendered:
        raise _bad_request("model-native image prompt rendering returned no text", param="images")
    return rendered


@lru_cache(maxsize=_NATIVE_TOKENIZER_CACHE_SIZE)
def _load_native_tokenizer_cached(source: str, revision: str | None) -> Any:
    """Load one pinned tokenizer into the bounded direct-route cache."""
    return load_tokenizer(source, trust_remote_code=True, revision=revision)


def _load_native_tokenizer_coalesced(source: str, revision: str | None) -> Any:
    """Coalesce concurrent cache misses without blocking the event loop."""
    with _NATIVE_TOKENIZER_LOAD_LOCK:
        return _load_native_tokenizer_cached(source, revision)


def _payload_too_large(message: str, *, param: str | None = None) -> HTTPException:
    """413 Payload Too Large, OpenAI-shaped error detail."""
    detail: dict[str, Any] = {
        "code": ErrorCode.INPUT_TOO_LONG.value,
        "message": message,
    }
    if param is not None:
        detail["param"] = param
    return HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=detail)


async def _read_bounded_request_body(request: Request, limit: int) -> bytes:
    """Read an ASGI request without aggregating more than ``limit`` bytes."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = -1
        if declared_length > limit:
            raise _payload_too_large(f"request body exceeds the limit of {limit} bytes")

    body = bytearray()
    async for chunk in request.stream():
        if len(chunk) > limit - len(body):
            raise _payload_too_large(f"request body exceeds the limit of {limit} bytes")
        body.extend(chunk)
    return bytes(body)


async def _stream_generate_events(
    adapter: GenerationAdapter,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    top_k: int | None,
    min_new_tokens: int | None,
    grammar: GrammarSpec | None,
    seed: int | None,
    logit_bias: dict[str, float] | None,
    logprobs: bool,
    top_logprobs: int | None,
    images: list[ImageInput] | None = None,
) -> AsyncIterator[str]:
    """Yield SIE-native ``GenerateChunk`` SSE lines for ``SIEClient.stream_generate``.

    Wire shape mirrors the gateway's ``build_generate_chunk_event``
    (``sie_gateway/src/handlers/sse.rs``) and the ``GenerateChunk`` TypedDict in
    ``sie_sdk.types``: incremental ``text_delta`` chunks, a terminal ``done:
    true`` chunk carrying ``finish_reason`` / ``usage`` / ``ttft_ms``, then the
    literal ``[DONE]`` terminator the SDK's SSE reader honours. A mid-stream
    failure is surfaced as a terminal chunk with ``finish_reason: "error"`` +
    ``error`` so the SDK raises ``ServerError`` instead of truncating silently.
    """
    request_id = uuid.uuid4().hex
    seq = 0
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    finish_reason = "stop"
    prompt_tokens = 0
    completion_tokens = 0
    saw_terminal = False
    terminal_error: dict[str, str] | None = None
    optional_adapter_inputs: dict[str, Any] = {}
    if grammar is not None:
        optional_adapter_inputs["grammar"] = grammar
    if images is not None:
        optional_adapter_inputs["images"] = images
    try:
        async for chunk in adapter.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_k=top_k,
            min_new_tokens=min_new_tokens,
            seed=seed,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            **optional_adapter_inputs,
        ):
            if chunk.done:
                saw_terminal = True
                finish_reason = chunk.finish_reason or "stop"
                if chunk.error_code is not None or chunk.error_message is not None or finish_reason == "error":
                    finish_reason = "error"
                    terminal_error = {
                        "code": chunk.error_code or "inference_error",
                        "message": chunk.error_message or "generation terminated with an upstream error",
                    }
                if chunk.prompt_tokens is not None:
                    prompt_tokens = chunk.prompt_tokens
                if chunk.completion_tokens is not None:
                    completion_tokens = chunk.completion_tokens
                # The contract allows a terminal chunk to also carry final text; emit it as a
                # delta so it isn't dropped (MLX's terminal text is always empty, but SGLang
                # and future adapters may pack final tokens here).
                if chunk.text_delta or chunk.logprobs:
                    if chunk.text_delta and ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    event: dict[str, Any] = {
                        "request_id": request_id,
                        "seq": seq,
                        "text_delta": chunk.text_delta,
                        "done": False,
                    }
                    if chunk.logprobs:
                        event["logprobs"] = list(chunk.logprobs)
                    yield f"data: {json.dumps(event)}\n\n"
                    seq += 1
                break
            if chunk.text_delta or chunk.logprobs:
                if chunk.text_delta and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                event: dict[str, Any] = {
                    "request_id": request_id,
                    "seq": seq,
                    "text_delta": chunk.text_delta,
                    "done": False,
                }
                if chunk.logprobs:
                    event["logprobs"] = list(chunk.logprobs)
                seq += 1
                yield f"data: {json.dumps(event)}\n\n"
    except Exception:  # noqa: BLE001 — surface as a terminal error chunk, never 500 mid-stream
        logger.warning("stream_generate failed mid-stream", exc_info=True)
        err = {
            "request_id": request_id,
            "seq": seq,
            "text_delta": "",
            "done": True,
            "finish_reason": "error",
            # Generic client message — the exception detail is logged server-side
            # (above, with exc_info) and must not leak to the client (CodeQL).
            "error": {"code": "inference_error", "message": "internal error during generation"},
        }
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
        return

    if not saw_terminal:
        finish_reason = "error"
        terminal_error = {
            "code": "inference_error",
            "message": "generation stream ended before a terminal event",
        }

    terminal: dict[str, Any] = {
        "request_id": request_id,
        "seq": seq,
        "text_delta": "",
        "done": True,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if ttft_ms is not None:
        terminal["ttft_ms"] = ttft_ms
    if terminal_error is not None:
        terminal["error"] = terminal_error
    yield f"data: {json.dumps(terminal)}\n\n"
    yield "data: [DONE]\n\n"


@router.post(
    "/generate/{model:path}",
    response_model=None,
    responses={
        200: {
            "description": "Generated text, or a Server-Sent Event stream when stream is true",
            "model": GenerateResponseModel,
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": "SIE-native GenerateChunk events terminated by data: [DONE]",
                    },
                    "x-sie-event-schema": {"$ref": "#/components/schemas/GenerateChunk"},
                }
            },
        },
        400: {"description": "Invalid request"},
        404: {"description": "Model not found"},
        413: {
            "description": "Prompt exceeds the configured UTF-8 size limit (INPUT_TOO_LONG)",
            "model": GenerateInputTooLongErrorResponse,
        },
        502: {
            "description": (
                "Terminal model-load failure (MODEL_LOAD_FAILED). "
                "Carried in the detail envelope: {code, message, error_class, permanent, attempts}. "
                "No Retry-After header; clients must not auto-retry."
            ),
            "model": GenerateModelLoadFailedErrorResponse,
        },
        503: {"description": "Model loading or unavailable"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/GenerateRequestModel"},
                }
            },
        },
    },
)
async def generate(
    model: str,
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> JSONResponse | StreamingResponse:
    """Generate text from a prompt using the named model.

    The ``model`` path segment uses the **SIE-safe** id (double-underscore
    separator, e.g. ``Qwen__Qwen3-4B-Instruct``). HuggingFace-style slashes
    are rejected with 400 to keep parity with the gateway contract.
    """
    validate_machine_profile_header(x_machine_profile)

    # Reject HF-style slashes explicitly. FastAPI's ``{model:path}`` would
    # otherwise happily accept ``Qwen/Qwen3-4B-Instruct``; we require the
    # SIE-safe (``__``) form to keep parity with the gateway path contract.
    if "/" in model:
        sie_safe = model.replace("/", "__")
        raise _bad_request(
            f"model path '{model}' uses HuggingFace-style slashes; "
            f"use the SIE-safe id '{sie_safe}' (double-underscore separator)",
            param="model",
            code=ErrorCode.MODEL_NOT_FOUND.value,
        )

    # The registry keys on the canonical ``sie_id`` (slash form, e.g.
    # ``Qwen/Qwen3.5-4B``) — see ``ModelConfig.name``. The production
    # worker path reverses the NATS-subject normalization with
    # ``denormalize_model_id`` before every registry lookup; mirror that
    # here so the dev route resolves real models instead of 404ing.
    registry_key = denormalize_model_id(model)

    with tracer.start_as_current_span("generate") as span:
        span.set_attribute("model", model)
        if x_machine_profile:
            span.set_attribute("machine_profile", x_machine_profile)

        raw_body = await _read_bounded_request_body(http_request, _MAX_GENERATE_BODY_BYTES)
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise _bad_request("request body must be a JSON object") from exc
        if not isinstance(body, dict):
            raise _bad_request("request body must be a JSON object")

        unknown = set(body) - _SUPPORTED_FIELDS
        if unknown:
            param = sorted(unknown)[0]
            raise _bad_request(
                f"unsupported field(s): {sorted(unknown)}",
                param=param,
                code="unsupported_field",
            )

        for field in ("routing_key", "prompt_cache_key", "safety_identifier"):
            value = body.get(field)
            if value is not None and not isinstance(value, str):
                raise _bad_request(f"'{field}' must be a string", param=field)

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise _bad_request("'prompt' must be a non-empty string", param="prompt")

        # Worker-side prompt size cap. The gateway caps the whole request
        # body, but this direct dev route is reached without the gateway,
        # so it must enforce its own bound or an oversized prompt would be
        # tokenised and forwarded unbounded. 413 Payload Too Large,
        # OpenAI-shaped (mirrors the gateway's PAYLOAD_TOO_LARGE).
        prompt_bytes = len(prompt.encode("utf-8"))
        if prompt_bytes > _MAX_PROMPT_BYTES:
            raise _payload_too_large(
                f"'prompt' is {prompt_bytes} bytes, exceeds the limit of {_MAX_PROMPT_BYTES} bytes",
                param="prompt",
            )
        images = _parse_native_images(body.get("images"))
        grammar = _parse_native_grammar(body.get("grammar"))

        max_new_tokens = body.get("max_new_tokens")
        # ``isinstance(x, int)`` is True for ``bool`` in Python — reject
        # booleans explicitly so ``True`` doesn't sneak through as 1.
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise _bad_request("'max_new_tokens' must be a positive integer", param="max_new_tokens")

        registry = http_request.app.state.registry
        device = registry.device

        # Standard model-state gates: 404 if unknown, 503 if loading/unloading,
        # 502 if a terminal load failure is in cooldown.
        checker = ModelStateChecker(registry, registry_key, span)
        checker.check_exists()
        checker.check_not_failed()
        checker.check_not_unloading()
        checker.check_not_loading()

        config = registry.get_config(registry_key)
        # Enforce the gateway-side cap mirror: max_new_tokens ≤
        # tasks.generate.max_output_tokens. Worker-authoritative so the
        # local-dev route reports the same 400 the gateway would.
        gen_task = getattr(config.tasks, "generate", None)
        if gen_task is None:
            raise _bad_request(
                f"Model '{model}' does not declare a generate task",
                code=ErrorCode.MODEL_NOT_FOUND.value,
            )
        if images and not config.inputs.image:
            raise _bad_request(
                f"Model '{model}' does not support image input",
                param="images",
                code="unsupported_field",
            )
        if grammar is not None and grammar.kind not in gen_task.capabilities.grammar:
            raise _bad_request(
                f"Model '{model}' does not declare '{grammar.kind}' grammar support",
                param=f"grammar.{grammar.kind}",
                code="unsupported_field",
            )
        if max_new_tokens > gen_task.max_output_tokens:
            raise _bad_request(
                f"max_new_tokens ({max_new_tokens}) exceeds model cap ({gen_task.max_output_tokens})",
                param="max_new_tokens",
                code="context_exceeded",
            )

        try:
            body = apply_generation_runtime_options(config, body.get("options"), body)
        except ValueError as exc:
            message = str(exc)
            param = "options.profile" if "options.profile" in message else "options"
            raise _bad_request(message, param=param) from exc

        # Explicit top-level sampler fields win; omitted values now carry the
        # resolved profile/request runtime defaults applied above.
        temperature_raw = body.get("temperature")
        if temperature_raw is None:
            temperature_raw = 1.0
        if isinstance(temperature_raw, bool) or not isinstance(temperature_raw, int | float):
            raise _bad_request("temperature must be a number", param="temperature")
        temperature = float(temperature_raw)
        # Range-validate so NaN / inf / negative samplers don't reach the
        # engine (parity with the gateway-side numeric validation).
        if not math.isfinite(temperature) or temperature < 0.0:
            raise _bad_request("temperature must be a finite number >= 0", param="temperature")
        top_p_raw = body.get("top_p")
        if top_p_raw is None:
            top_p_raw = 1.0
        if isinstance(top_p_raw, bool) or not isinstance(top_p_raw, int | float):
            raise _bad_request("top_p must be a number", param="top_p")
        top_p = float(top_p_raw)
        if not math.isfinite(top_p) or not (0.0 < top_p <= 1.0):
            raise _bad_request("top_p must be in (0, 1]", param="top_p")
        stop_raw = body.get("stop")
        top_k_raw = body.get("top_k")
        if top_k_raw is not None and (isinstance(top_k_raw, bool) or not isinstance(top_k_raw, int) or top_k_raw < 1):
            raise _bad_request("top_k must be an integer >= 1", param="options.default_sampling.top_k")
        top_k = top_k_raw
        min_tokens_raw = body.get("min_tokens")
        if min_tokens_raw is not None and (
            isinstance(min_tokens_raw, bool) or not isinstance(min_tokens_raw, int) or min_tokens_raw < 0
        ):
            raise _bad_request(
                "min_new_tokens must be an integer >= 0", param="options.default_sampling.min_new_tokens"
            )
        min_new_tokens = min_tokens_raw
        if stop_raw is not None and (not isinstance(stop_raw, list) or not all(isinstance(s, str) for s in stop_raw)):
            raise _bad_request("'stop' must be a list of strings", param="stop")
        # Reject empty-string stop sequences. SGLang treats ``""`` as a
        # match after every token, so a single empty entry terminates
        # generation after one token — surprising and useless. The
        # gateway path silently drops these via Rust's filter_map; do
        # the same here.
        if stop_raw is not None and any(s == "" for s in stop_raw):
            raise _bad_request("'stop' must not contain empty strings", param="stop")
        stop = [str(s) for s in stop_raw] if stop_raw else None

        frequency_penalty = _validate_penalty(body.get("frequency_penalty"), param="frequency_penalty")
        presence_penalty = _validate_penalty(body.get("presence_penalty"), param="presence_penalty")

        # Sampler controls are validated and forwarded. Per-token logprobs
        # are native streaming output; the blocking response has no faithful
        # field for them and rejects the request instead of dropping data.
        seed = _validate_seed(body.get("seed"))
        logit_bias = _validate_logit_bias(body.get("logit_bias"))

        # Streaming path: emit SIE-native GenerateChunk SSE (drives
        # SIEClient.stream_generate). The blocking JSON path below is unchanged.
        stream_raw = body.get("stream", False)
        if stream_raw is not None and not isinstance(stream_raw, bool):
            raise _bad_request("'stream' must be a boolean", param="stream")
        for field in ("logprobs", "top_logprobs"):
            if not stream_raw and body.get(field) is not None:
                raise _bad_request(
                    f"'{field}' is supported only with 'stream: true' on the native endpoint",
                    param=field,
                    code="unsupported_field",
                )
        logprobs, top_logprobs = _validate_logprobs(body.get("logprobs"), body.get("top_logprobs"))

        generation_prompt = prompt
        if images:
            generation_prompt = await _render_native_image_prompt(config, prompt, len(images))

        # Do not start a potentially expensive model load until the complete
        # request has passed validation.
        await checker.ensure_loaded(device)
        adapter = registry.get(registry_key)
        registry.touch_lru(registry_key)
        if not isinstance(adapter, GenerationAdapter):
            raise _bad_request(
                f"Model '{model}' adapter does not support generate (not a GenerationAdapter)",
                code=ErrorCode.MODEL_NOT_FOUND.value,
            )

        if stream_raw:
            return StreamingResponse(
                _stream_generate_events(
                    adapter,
                    prompt=generation_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed,
                    logit_bias=logit_bias,
                    top_k=top_k,
                    min_new_tokens=min_new_tokens,
                    grammar=grammar,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    images=images,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            # ``adapter.generate`` is an async iterator. The
            # local-dev route keeps the walking-skeleton's blocking response shape
            # for backwards compatibility — drain the iterator into an
            # aggregate. SDK / gateway consume the iterator directly.
            optional_adapter_inputs: dict[str, Any] = {}
            if grammar is not None:
                optional_adapter_inputs["grammar"] = grammar
            if images is not None:
                optional_adapter_inputs["images"] = images
            chunks = adapter.generate(
                prompt=generation_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_k=top_k,
                min_new_tokens=min_new_tokens,
                seed=seed,
                logit_bias=logit_bias,
                **optional_adapter_inputs,
            )
            result = await collect_generation(chunks)
        except Exception as e:
            logger.warning("generate failed for %s", model, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "inference_error", "message": str(e)},
            ) from e

        # A stream that finished cleanly may still carry a *terminal* error /
        # cancellation status instead of raising — e.g. the adapter caught an
        # upstream SGLang 500 and surfaced it as a ``finish_reason: "error"``
        # chunk, or a cancel signal landed mid-stream
        # (``finish_reason: "cancelled"``). ``collect_generation`` returns
        # that partial text normally, so without this check the route would
        # answer HTTP 200 with truncated output. Map the failure terminators
        # to non-2xx, keeping the OpenAI-shaped error body the route uses
        # elsewhere. (``stop`` / ``length`` are the normal success
        # terminators and fall through to the 200 response.)
        if result.finish_reason == "error":
            logger.warning("generate produced terminal finish_reason=error for %s", model)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "inference_error",
                    "message": "generation terminated with an upstream error",
                },
            )
        if result.finish_reason == "cancelled":
            # 503 Service Unavailable: the generation was cancelled before it
            # could complete (worker observed a cancel signal mid-stream).
            # A retry may succeed, so this is a transient non-2xx rather than
            # a client error.
            logger.warning("generate produced terminal finish_reason=cancelled for %s", model)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "generation_cancelled",
                    "message": "generation was cancelled before completion",
                },
            )

        # Return the canonical model id (the registry/config ``name``, i.e.
        # the slash-form ``sie_id``) rather than the raw ``__``-form path
        # param. The SDK sends the canonical id and matches the response
        # ``model`` against it; echoing the path-encoded form broke that
        # round-trip. ``config.name`` == ``registry_key`` (denormalized
        # path param) — prefer the config value as the source of truth and
        # fall back to ``registry_key`` defensively.
        canonical_model = getattr(config, "name", None) or registry_key
        return JSONResponse(
            content={
                "model": canonical_model,
                "text": result.text,
                "finish_reason": result.finish_reason,
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            }
        )
