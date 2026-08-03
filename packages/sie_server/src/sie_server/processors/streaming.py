"""Streaming processor for generation work items.

The ``StreamingProcessor`` is the gen-only branch behind the sidecar IPC
``MessageProcessor`` path.

The streaming rollout (this revision) drives the adapter as an async iterator:

- pull one work item
- generate a worker-side ``attempt_id``
- iterate ``adapter.generate(...)``, **coalescing** yields by time/count
- publish each coalesced batch as a chunk envelope on
  ``_INBOX.{router_id}.{request_id}``
- on the iterator's terminal yield, publish a ``{done: true}`` chunk
  with ``usage`` / ``finish_reason`` / ``ttft_ms`` and ACK the JetStream
  work message
- on cancel (signaled by the worker-sidecar from the cluster-scope
  ``cancel.>`` subscription), close the iterator, flush a terminal
  ``{finish_reason: "cancelled", done: true}`` chunk, and ACK
- on sustained transport failure (>3 publish errors within 1s, or per-
  request queue cap >64 unsent), publish a final ``{error,
  code: "transport_failure", done: true}`` chunk and ACK

The chunk envelope uses a discriminated streaming wire shape:

    { kind: "chunk", request_id, attempt_id, seq, text_delta,
      is_first?, done, finish_reason?, usage?, ttft_ms?, error? }
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import collections
import json
import logging
import math
import os
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import msgpack
from opentelemetry import propagate
from opentelemetry import trace as _otel_trace
from sie_sdk.queue_types import WorkItem

from sie_server.adapters._generation_base import (
    FinishReason,
    GenerationAdapter,
    GenerationChunk,
    ToolCallDelta,
)
from sie_server.core.runtime_options import apply_generation_runtime_options
from sie_server.core.text_tokens import estimate_tokens_from_chars
from sie_server.core.tokenizer import image_first_chat_message, load_tokenizer
from sie_server.observability import worker_telemetry as _metrics
from sie_server.processors.grammar_cache import GrammarLRU
from sie_server.processors.grammar_compile import compile_outlines
from sie_server.processors.tool_call_grammar import (
    ToolChoiceError,
    build_tool_choice_grammar,
    normalize_tool_choice,
)
from sie_server.processors.tool_call_parser import ToolCallFormat, parse_tool_call_stream
from sie_server.types.grammar import GrammarSpec, GrammarValidationError, hash_grammar
from sie_server.types.inputs import ImageInput

# Module-level shim around :func:`asyncio.wait_for`. Tests monkey-patch
# this attribute (not the global ``asyncio.wait_for``) so the override
# is scoped to this module — without it a test could change every
# concurrent coroutine's timeout. The runtime cost is one extra
# indirection; the asyncio runtime caches the binding internally.
_wait_for = asyncio.wait_for

# Dedicated thread-pool for blocking grammar compiles. Bounded at 4
# threads so a misbehaving Outlines version that hangs cannot drain
# the default :func:`asyncio.to_thread` pool (which the rest of the
# worker shares — tokenizer loads, chat-template render, etc.). On
# timeout :func:`asyncio.wait_for` cancels the future but the thread
# keeps running until Outlines returns; bounding the pool keeps the
# blast radius local.
_GRAMMAR_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="grammar-compile")

# Ceiling a single-flight *follower* waits on the leader's shared compile
# future. The leader's own compile is capped at 5s plus a (cold) tokenizer
# load, so this is set comfortably above that — generous enough never to
# trip on a legitimately slow leader, but bounded so a leader that is
# cancelled/wedged without resolving the future cannot hang every follower
# until ack_wait (which would trigger a JetStream redelivery storm).
_GRAMMAR_FOLLOWER_TIMEOUT_S = 30.0

# SGLang owns request-time grammar compilation.
#
# By default the worker forwards the raw schema/regex/EBNF straight to
# SGLang's server-side grammar backend (the single grammar authority
# on the request hot path). The legacy worker-side Outlines preflight
# is preserved behind ``SIE_GRAMMAR_PREFLIGHT_DEBUG=1`` for diagnostic
# use — for debugging schema-rejection problems or slow compiles in a
# controlled environment. Not recommended for production traffic.
#
# See ``docs/adr/0002-sglang-owns-request-time-grammar.md``.
_GRAMMAR_PREFLIGHT_DEBUG_ENV = "SIE_GRAMMAR_PREFLIGHT_DEBUG"


def _grammar_preflight_debug_enabled() -> bool:
    """Return whether the legacy worker-side Outlines preflight is enabled.

    Reads ``SIE_GRAMMAR_PREFLIGHT_DEBUG`` on every call so tests
    (and operators) can toggle the behaviour without restarting the
    worker. Treats ``1``, ``true``, ``yes``, ``on`` (case-insensitive,
    whitespace-trimmed) as truthy; everything else (including unset)
    is falsy.
    """
    raw = os.environ.get(_GRAMMAR_PREFLIGHT_DEBUG_ENV, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Maximum entries retained in ``StreamingProcessor._released_requests``
# before FIFO eviction. Sized to comfortably cover the window in which
# a duplicate release could plausibly fire (paired finally blocks within
# a single request) without holding the worker's lifetime worth of ids.
_RELEASED_REQUESTS_MAX = 1024

# H9 — cancel-tombstone TTL (seconds).
#
# The first-chunk-fallback race: the gateway publishes the work item via
# direct-dispatch to a chosen worker, fires a cancel signal when the
# first-chunk deadline elapses, and republishes to the pool. If the cancel
# flies past before the direct-dispatch worker has even pulled the message
# off JetStream (model load, GC pause, scheduler hiccup), ``signal_cancel``
# has no in-flight attempt to find — the cancel is lost on the floor. The
# original then decodes in parallel with the pool-republished attempt.
#
# The tombstone records ``request_id -> deadline`` so a decode-start check
# can refuse to decode a request that was already cancelled while we were
# still picking up its work item. The TTL is generous enough to cover the
# largest plausible "pull lag" — JetStream ``ack_wait`` for generation
# pools is 5 minutes; we use 120s here because a request that takes longer
# than that to reach decode-start is overwhelmingly likely to have
# triggered the gateway's overall-timeout already (so the client has long
# since given up). 120s comfortably exceeds any first-chunk window we'd
# realistically configure, while keeping the per-tombstone memory cost
# small.
_CANCEL_TOMBSTONE_TTL_S = 120.0

# Soft cap on the tombstone map. Lazy cleanup sweeps expired entries on
# insertion when the map exceeds this size; if the sweep doesn't bring
# the size back down (e.g. all entries are still live), the oldest
# tombstone is evicted to bound memory. In practice request_ids are
# UUIDs so the only way to fill 1024 live tombstones is to be racing
# 1024 simultaneous cancel-before-register events — well outside the
# realistic operating envelope.
_CANCEL_TOMBSTONE_MAX = 1024

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    from sie_server.config.model import ModelConfig
    from sie_server.core.registry import ModelRegistry

    TokenizerLike = PreTrainedTokenizer | PreTrainedTokenizerFast


class NatsPublisher(Protocol):
    async def publish(self, reply_subject: str, payload: bytes) -> None:
        raise NotImplementedError


logger = logging.getLogger(__name__)

# Default delay when NAKing a generate item because the model is not yet
# loaded — matches the encode/extract path's behavior.
_NAK_DELAY_S = 5.0

# Coalescing knobs: flush either every ~50ms or every ~32 tokens, whichever
# fires first. Keeps NATS message rate at ~2-10 msgs/sec/request for
# typical decode rates (50-200 tok/s).
_FLUSH_INTERVAL_S = 0.05
_FLUSH_MAX_TOKENS = 32

# Per-request bounded queue of unsent published-chunk byte blobs. Cap
# exists so a slow gateway can't OOM the worker. Cap exceeded → publish
# transport_failure and abort.
_CHUNK_QUEUE_MAX = 64

# Bounded await timeout for enqueuing content chunks onto ``chunk_queue``.
# Brief publisher stalls (slow NATS publish) are absorbed cleanly; a
# sustained stall trips the no-silent-drop guarantee: we publish a
# ``transport_failure`` terminal and withhold the JetStream ACK so the
# message is redelivered. 100ms is well under the 50ms coalescing window
# upper bound × a factor that gives a healthy publisher slack while still
# tripping promptly when the gateway is genuinely wedged.
# Per H6 remediation (workstream C).
_CHUNK_PUT_TIMEOUT_S = 0.1

# Longer timeout for enqueuing the ``transport_failure`` terminal itself.
# Reached only after a content-chunk enqueue already gave up: the publisher
# is likely under brief sustained pressure (slow NATS, fan-out backlog).
# A longer window lets the failure terminal land cleanly on the wire when
# the publisher resumes — without it the gateway would see "no terminal,
# no ACK" (full JetStream redelivery, more cost) instead of "transport
# failure surfaced + redelivery" (client gets explicit error promptly).
# 5x the content-chunk timeout keeps the overall ceiling on a wedged
# publisher modest (~500ms by default).
_TRANSPORT_FAILURE_PUT_TIMEOUT_S = 0.5

# Sustained-failure detector: >N publish errors within window → abort.
_PUBLISH_FAIL_THRESHOLD = 3
_PUBLISH_FAIL_WINDOW_S = 1.0

# Heartbeat cadence for JetStream ``in_progress()`` calls. The pool
# consumer's ``ack_wait`` is 5 min for generation pools (per the streaming §4.4 spec);
# beat well below that so a slow decode doesn't lose the slot.
_INPROGRESS_INTERVAL_S = float(os.environ.get("SIE_GENERATION_INPROGRESS_INTERVAL_S", "10.0"))

# Input-size caps applied BEFORE the CPU-bound, input-sized tokenizer
# calls (``apply_chat_template`` / ``tok.encode``). These run on the
# bounded ``_GRAMMAR_EXECUTOR`` (see below), so a burst of pathologically
# large inputs can still saturate it; capping the input keeps each call's
# wall-time bounded. ``_MAX_CHAT_MESSAGES`` rejects absurd message lists;
# ``_MAX_PROMPT_CHARS`` rejects/truncates absurd single prompts before
# they reach the tokenizer.
_MAX_CHAT_MESSAGES = 4096
_MAX_PROMPT_CHARS = 4_000_000


# Per-message role validation. The gateway already normalizes OpenAI's
# ``developer`` role to ``system`` before publishing, so the wire normally
# carries only the core four. ``developer`` is included here defensively (and
# folded to ``system`` in :meth:`_render_chat_template`) so a direct worker
# caller is not rejected. Anything else surfaces as ``invalid_request`` with
# the offending ``messages[i].role`` path in ``param``.
_ALLOWED_CHAT_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool", "developer"})

# Hard cap on images per generate request. Bounds the worker-side decode +
# the engine's vision-encoder cost so one request can't pin a GPU. Vision
# requests over this are rejected with ``invalid_request`` rather than
# silently truncated. Mirrored at the gateway so oversized requests fail
# before decode/publish.
_MAX_IMAGES_PER_REQUEST = 16

# Coarse per-image token estimate for the context-length guard and the
# admission reservation. The chat template renders each image as a short
# placeholder, but the vision encoder expands it into many tokens (Qwen-VL:
# up to ~1280 at max_pixels). Exact counts need the image dimensions + the
# model's vision tokenizer, which we don't have at this layer — so we use a
# conservative constant. Over-reserving is safe; under-counting risks a
# context-window overflow (opaque SGLang error) and admission over-admit.
_VISION_TOKENS_PER_IMAGE_ESTIMATE = 1024

# Hard ceiling on a single decoded image's bytes. Bounds worker memory on the
# decode path even for the direct-queue caller that bypasses the gateway's
# request-body cap. Checked from the base64 length BEFORE decoding (so a huge
# payload never allocates) and again on the decoded bytes (padding/edge cases).
_MAX_DECODE_IMAGE_BYTES = 16 * 1024 * 1024

# OpenAI content-part ``type`` values that carry an image. ``image_url`` is the
# chat-completions shape; ``input_image`` is the Responses-API alias some
# clients send. Both are decoded from an inline ``data:`` URI.
_IMAGE_CONTENT_PART_TYPES: frozenset[str] = frozenset({"image_url", "input_image"})


def _decode_data_uri_image(url: str) -> tuple[bytes, str | None]:
    """Decode an inline ``data:`` image URI into ``(bytes, format)``.

    Only base64-encoded ``data:`` URIs are supported — the worker never
    fetches remote ``http(s)`` URLs (that would be an SSRF surface and a
    blocking network call on the hot path; clients inline images as data
    URIs, which the SDK/gateway already do). Returns the raw bytes plus the
    format hint parsed from the MIME subtype (e.g. ``"png"``), or ``None``
    when the MIME type is absent/uninformative.

    Raises:
        ValueError: on a non-``data:`` URI, a malformed URI, a non-base64
            payload, or undecodable base64.
    """
    if not url.startswith("data:"):
        raise ValueError("image content must be an inline base64 'data:' URI; remote URL fetching is not supported")
    header, sep, payload = url[len("data:") :].partition(",")
    if not sep:
        raise ValueError("malformed image data URI (missing ',')")
    params = header.split(";")
    if "base64" not in params[1:]:
        raise ValueError("image data URI must be base64-encoded")
    mime = params[0]
    # Mirror the gateway: only image media types (rejects data:text/... and
    # data:;base64,... for direct queue callers too).
    if not mime.startswith("image/"):
        raise ValueError("image data URI must have an image/* media type")
    fmt: str | None = None
    if "/" in mime:
        subtype = mime.split("/", 1)[1].strip().lower()
        fmt = subtype or None
    stripped = payload.strip()
    # Reject oversized images from the base64 length before allocating.
    if (len(stripped) * 3) // 4 > _MAX_DECODE_IMAGE_BYTES:
        raise ValueError(
            f"image too large: exceeds the {_MAX_DECODE_IMAGE_BYTES // (1024 * 1024)} MiB limit "
            "(_decode_data_uri_image)"
        )
    try:
        # ``.strip()`` mirrors the gateway's ``payload.trim()`` so both ends
        # accept the same data URIs (some clients pad base64 with whitespace).
        data = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 image data: {exc}") from exc
    if not data:
        raise ValueError("image data URI decoded to empty bytes")
    if len(data) > _MAX_DECODE_IMAGE_BYTES:
        raise ValueError(
            f"image too large: {len(data)} bytes exceeds the "
            f"{_MAX_DECODE_IMAGE_BYTES // (1024 * 1024)} MiB limit (_decode_data_uri_image)"
        )
    return data, fmt


def _parse_message_images_field(raw: object, idx: int) -> tuple[ImageInput, ...] | _ValidationError:
    """Parse a message-level ``images`` field of gateway-forwarded images.

    The gateway extracts OpenAI ``image_url`` data URIs and forwards them as
    ``messages[i].images = [{"data": <base64-str>, "format": <str?>}, ...]``.
    ``data`` is the **base64 payload string**, not raw bytes — that's what
    survives the sidecar's ``WorkItem.generate: serde_json::Value`` (msgpack
    ``bin`` does not; see the gateway ``ChatImage`` doc). We base64-decode here
    into the ``bytes`` the adapter expects. Missing/``None`` → no images.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        return _ValidationError(code="invalid_request", message=f"messages[{idx}].images must be an array")
    out: list[ImageInput] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return _ValidationError(
                code="invalid_request",
                message=f"messages[{idx}].images[{i}] must be an object",
            )
        entry_dict = cast("dict[str, Any]", entry)
        b64 = entry_dict.get("data")
        if not isinstance(b64, str) or not b64:
            return _ValidationError(
                code="invalid_request",
                message=f"messages[{idx}].images[{i}].data must be a non-empty base64 string",
            )
        stripped = b64.strip()
        if (len(stripped) * 3) // 4 > _MAX_DECODE_IMAGE_BYTES:
            return _ValidationError(
                code="invalid_request",
                message=(
                    f"messages[{idx}].images[{i}]: image too large; exceeds the "
                    f"{_MAX_DECODE_IMAGE_BYTES // (1024 * 1024)} MiB limit"
                ),
            )
        try:
            data = base64.b64decode(stripped, validate=True)
        except (binascii.Error, ValueError) as exc:
            return _ValidationError(
                code="invalid_request",
                message=f"messages[{idx}].images[{i}]: invalid base64 image data: {exc}",
            )
        if not data:
            return _ValidationError(
                code="invalid_request",
                message=f"messages[{idx}].images[{i}]: image data is empty",
            )
        if len(data) > _MAX_DECODE_IMAGE_BYTES:
            return _ValidationError(
                code="invalid_request",
                message=(
                    f"messages[{idx}].images[{i}]: image too large ({len(data)} bytes); exceeds the "
                    f"{_MAX_DECODE_IMAGE_BYTES // (1024 * 1024)} MiB limit"
                ),
            )
        fmt = entry_dict.get("format")
        out.append({"data": data, "format": fmt if isinstance(fmt, str) else None})
    return tuple(out)


def _parse_content_parts_field(raw: object, idx: int) -> tuple[dict[str, Any], ...] | None | _ValidationError:
    """Parse a message-level ``content_parts`` field of gateway-forwarded layout.

    The gateway records the original text↔image ordering (#1294) as
    ``messages[i].content_parts = [{"type":"text","text":…} | {"type":"image"}, …]``
    so the worker can place image placeholders in position. Returned in
    chat-template shape (ready for ``apply_chat_template``). ``None`` when the
    field is absent — the message then renders images-first from ``content`` +
    ``images``. The image-marker count is reconciled against the flat ``images``
    list by the caller.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        return _ValidationError(code="invalid_request", message=f"messages[{idx}].content_parts must be an array")
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return _ValidationError(
                code="invalid_request",
                message=f"messages[{idx}].content_parts[{i}] must be an object",
            )
        entry_dict = cast("dict[str, Any]", entry)
        ptype = entry_dict.get("type")
        if ptype == "text":
            text = entry_dict.get("text")
            if not isinstance(text, str):
                return _ValidationError(
                    code="invalid_request",
                    message=f"messages[{idx}].content_parts[{i}].text must be a string",
                )
            out.append({"type": "text", "text": text})
        elif ptype == "image":
            out.append({"type": "image"})
        else:
            return _ValidationError(
                code="invalid_request",
                message=f"messages[{idx}].content_parts[{i}].type must be 'text' or 'image', got {ptype!r}",
            )
    return tuple(out)


def _parse_chat_content(
    content: object, idx: int
) -> tuple[str, tuple[ImageInput, ...], tuple[dict[str, Any], ...] | None] | _ValidationError:
    """Parse an OpenAI message ``content`` into ``(text, images, content_parts)``.

    Accepts either the legacy string shape (returned verbatim with no
    images) or the multimodal list-of-parts shape, where each part is
    ``{"type": "text", "text": ...}`` or an image part
    (``{"type": "image_url", "image_url": {"url": "data:..."}}`` — the
    ``url`` may also be a bare string, and ``input_image`` is accepted as an
    alias). Text parts are concatenated in order; image parts are decoded to
    :class:`ImageInput` entries in order so they line up with the
    placeholders the chat template will render.

    ``content_parts`` is the ordered text↔image layout in chat-template shape
    (``{"type": "text", "text": …}`` / ``{"type": "image"}``), so
    :meth:`_render_chat_template` can interleave placeholders in place instead
    of emitting all images first (#1294). It is ``None`` for string content and
    for image-free part arrays — those keep the concatenated-``content`` render.

    Returns a :class:`_ValidationError` (not raised) on any malformed part,
    matching the surrounding validator's error-return contract.
    """
    if isinstance(content, str):
        return content, (), None
    if not isinstance(content, list):
        return _ValidationError(
            code="invalid_request",
            message=f"messages[{idx}].content must be a string or an array of content parts",
        )
    text_parts: list[str] = []
    images: list[ImageInput] = []
    layout: list[dict[str, Any]] = []
    for part_idx, part in enumerate(content):
        if not isinstance(part, dict):
            return _ValidationError(
                code="invalid_request",
                message=f"messages[{idx}].content[{part_idx}] must be an object",
            )
        part_dict = cast("dict[str, Any]", part)
        part_type = part_dict.get("type")
        if part_type in ("text", "input_text"):
            text = part_dict.get("text")
            if not isinstance(text, str):
                return _ValidationError(
                    code="invalid_request",
                    message=f"messages[{idx}].content[{part_idx}].text must be a string",
                )
            text_parts.append(text)
            layout.append({"type": "text", "text": text})
        elif part_type in _IMAGE_CONTENT_PART_TYPES:
            url_field = part_dict.get("image_url")
            url = url_field.get("url") if isinstance(url_field, dict) else url_field
            if not isinstance(url, str) or not url:
                return _ValidationError(
                    code="invalid_request",
                    message=f"messages[{idx}].content[{part_idx}].image_url.url must be a non-empty string",
                )
            try:
                data, fmt = _decode_data_uri_image(url)
            except ValueError as exc:
                return _ValidationError(
                    code="invalid_request",
                    message=f"messages[{idx}].content[{part_idx}]: {exc}",
                )
            images.append({"data": data, "format": fmt})
            layout.append({"type": "image"})
        else:
            return _ValidationError(
                code="invalid_request",
                message=(
                    f"messages[{idx}].content[{part_idx}].type must be 'text'/'input_text' or an image part "
                    f"({sorted(_IMAGE_CONTENT_PART_TYPES)!r}), got {part_type!r}"
                ),
            )
    # Only carry the ordered layout when there's an image to place — text-only
    # arrays render fine from the concatenated ``content`` (no wire/format bloat).
    return "".join(text_parts), tuple(images), (tuple(layout) if images else None)


# Input-token estimate for the admission controller. Delegates
# to :func:`sie_server.core.text_tokens.estimate_tokens_from_chars` so the
# preprocessor's cost-proxy and the admission gate share a single
# ``chars_per_token`` constant. The exact tokenizer-based
# ``_check_context_length`` upstream stays authoritative for the
# context-length guard; this helper is the cheap, fast pre-flight
# (no per-request tokenizer call) the admission controller needs.


@dataclass(frozen=True)
class _ChatMessage:
    role: str
    content: str
    # Multi-turn tool use: assistant messages may carry tool_calls
    # (OpenAI shape), and tool-result messages carry tool_call_id. Both
    # are replayed into the chat template so the model can produce its
    # final answer after a tool returns.
    tool_calls: tuple[dict[str, Any], ...] | None = None
    tool_call_id: str | None = None
    # Vision input: images attached to this message (decoded from OpenAI
    # ``image_url`` content parts). ``None`` for text-only messages. The
    # chat template renders one image placeholder per entry, in order, and
    # the bytes are forwarded to the engine as ``image_data``.
    images: tuple[ImageInput, ...] | None = None
    # Ordered text↔image layout in chat-template shape
    # (``{"type":"text","text":…}`` / ``{"type":"image"}``) preserving the
    # original content-part order so :meth:`_render_chat_template` interleaves
    # placeholders in place (#1294). ``None`` → fall back to images-first
    # rendering from ``content`` + ``images``. Image markers here line up 1:1
    # with ``images`` (same order); only placeholder POSITIONS differ.
    content_parts: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class _PromptInput:
    prompt: str


@dataclass(frozen=True)
class _MessagesInput:
    messages: tuple[_ChatMessage, ...]


@dataclass(frozen=True)
class _ValidationError:
    """Validation outcome for a generate work item.

    Carries a wire-stable terminal-chunk ``code`` and the human-readable
    message that the caller propagates to the gateway via
    :meth:`_publish_terminal_error`.
    """

    code: str
    message: str


@dataclass
class _GenerationCompletion:
    """One attempt's shared worker-completion observation state.

    Terminal publication is the authoritative seam: a generation has not
    completed from the worker's perspective until its success/error/cancelled
    chunk reaches the reply transport. ``completed`` is a defensive exactly-once
    latch should a future path attempt a second terminal publication.
    """

    model_id: str
    profile_id: str
    started_at: float
    completed: bool = False


@dataclass(frozen=True)
class _GenerateRequestParams:
    """Worker-side decoded view of ``WorkItem.generate``.

    ``input`` is either a raw prompt (prompt wire shape) or a list of
    chat messages (chat-completions shape). The remaining fields mirror the sampling
    knobs already on :class:`publisher.GenerateParams`.

    ``grammar`` is the structured-output spec; ``None`` when
    the request omitted ``grammar`` / ``response_format`` upstream. The
    gateway has already validated shape, depth, size, and the model's
    declared capability — the worker only re-runs the Outlines compile
    (with the model's tokenizer) and surfaces a
    ``grammar_compile_failed`` chunk on failure.
    """

    input: _PromptInput | _MessagesInput
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] | None = None
    # OpenAI penalty knobs (range ``[-2.0, 2.0]``), gateway-validated.
    # ``None`` → adapter uses its own default (typically 0.0). Forwarded
    # to SGLang via the same sampling-params dict as ``temperature``.
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    # Non-OpenAI sampling knobs (Together / Fireworks / vLLM): ``top_k``
    # (int >= 1) and ``repetition_penalty`` (float in (0, 2]). Gateway-
    # validated; ``None`` → model/sampler default.
    top_k: int | None = None
    repetition_penalty: float | None = None
    # SGLang ``sampling_params["min_new_tokens"]`` (int >= 0). Anti-
    # stop-first knob; ``None`` → no minimum.
    min_tokens: int | None = None
    # Per-request kwargs for ``apply_chat_template`` (e.g.
    # ``{"enable_thinking": False}``). Merged on top of the model YAML's
    # ``chat_template_kwargs`` at render time. ``None`` → only YAML
    # defaults apply.
    chat_template_kwargs: dict[str, Any] | None = None
    grammar: GrammarSpec | None = None
    # OpenAI tools / tool_choice. The gateway has already shape-validated
    # the schemas; the worker uses ``tools is not None`` as the flag that
    # turns on :func:`parse_tool_call_stream` between the adapter and the
    # chunk encoder. ``tool_choice`` is currently informational — Qwen3's
    # chat template emits ``<tool_call>...</tool_call>`` blocks driven by
    # the model itself; we don't force a specific call at the sampler
    # layer yet.
    tools: tuple[dict[str, Any], ...] | None = None
    tool_choice: dict[str, Any] | str | None = None
    parallel_tool_calls: bool = True
    # OpenAI ``seed`` — signed 64-bit per-request sampling seed, validated by
    # the gateway and forwarded unchanged. Reproducibility semantics are
    # owned by the active generation backend.
    seed: int | None = None
    # OpenAI ``logit_bias`` — ``{token_id_str: bias_float}``, gateway
    # range-validates ``[-100.0, 100.0]`` and caps the map size.
    # Forwarded to SGLang ``sampling_params["logit_bias"]``.
    logit_bias: dict[str, float] | None = None
    # OpenAI ``logprobs`` — when ``True``, the worker asks SGLang for
    # per-token log-probabilities and propagates them on each chunk.
    logprobs: bool = False
    # OpenAI ``top_logprobs`` — how many alternates per position
    # (``[0, 20]``); only used when ``logprobs`` is True.
    top_logprobs: int | None = None
    # OpenAI ``n`` — number of candidates. ``None``/``1`` is the
    # single-candidate stream; ``> 1`` triggers the adapter's server-side
    # multi-candidate fan-out. Supported on both paths: non-streaming
    # collects candidates into a terminal ``candidates[]`` aggregate;
    # streaming fans candidates out as per-``choice_index`` delta chunks
    # with per-choice ``finish_reason`` / ``logprobs`` and one global
    # ``done=True`` terminal closing the stream.
    n: int | None = None
    # OpenAI ``best_of`` — generate this many candidates, return the top ``n``
    # by cumulative logprob (gateway-validated ``best_of >= n``; non-streaming).
    best_of: int | None = None
    # Whether the client requested SSE streaming. Only consulted for ``n>1``:
    # streaming fans candidates out as per-``choice_index`` delta chunks; non-
    # streaming uses the single terminal ``candidates[]`` aggregate.
    stream: bool = False
    # Multi-LoRA — served-name of the adapter to apply (passed to SGLang as
    # ``sampling_params.lora_path``). ``None`` → base model.
    lora_adapter: str | None = None


class StreamingProcessor:
    """Process a single generate work item end-to-end (streaming)."""

    def __init__(
        self,
        nc: NatsPublisher,
        registry: ModelRegistry,
        worker_id: str,
        *,
        grammar_cache: GrammarLRU | None = None,
        kv_budget_tokens: int | None = None,
        admission_enabled: bool = False,
        admission_resolver: Callable[[str], tuple[int | None, bool | None]] | None = None,
        record_worker_completion: bool = True,
    ) -> None:
        self._nc = nc
        self._registry = registry
        self._worker_id = worker_id
        # The NATS/sidecar path owns completion when the terminal reaches its
        # reply publisher. A transport wrapper that synthesizes client-facing
        # terminals (the dedicated Modal generation lane) can take ownership
        # instead, so one logical attempt is never counted twice.
        self._record_worker_completion = record_worker_completion
        # request_id → {attempt_id → asyncio.Event} for in-flight generations.
        # The worker-sidecar's cancel-subscription listener calls
        # ``signal_cancel(request_id)`` when a cancel.* message arrives, which
        # sets EVERY attempt's event for that request_id. The cluster-scope
        # subscription is on ``cancel.>``; this map is the actual filter.
        #
        # Why nested-by-attempt (BUG 2/7): redelivery after ``ack_wait`` can
        # dispatch a SECOND ``process()`` with the same ``request_id`` while
        # the first is still draining. A flat ``dict[request_id, Event]``
        # let the second registration clobber the first, and the first
        # attempt's ``finally`` then popped the survivor's handle — so a
        # later cancel was a silent no-op. Keying the inner map on the
        # per-pickup ``attempt_id`` keeps each live attempt's handle
        # independent; cleanup removes only its own.
        self._in_flight_cancels: dict[str, dict[str, asyncio.Event]] = {}
        # Per-model lazy tokenizer cache. Populated on first
        # ``Messages`` request — or any request that needs context-length
        # validation. ``apply_chat_template`` and ``AutoTokenizer.from_pretrained``
        # both block, so cache hits avoid round-tripping through
        # ``asyncio.to_thread`` more than once per model. The lock guards
        # against concurrent first-request races; subsequent reads are
        # plain dict lookups (no contention).
        self._tokenizers: dict[str, TokenizerLike] = {}
        self._tokenizer_lock = asyncio.Lock()
        # Per-process LRU of validated grammar compiles. The
        # default cap (64) is plenty for realistic workloads — schemas
        # tend to concentrate on a handful of canonical shapes. Inject
        # a smaller cache from tests to exercise eviction.
        self._grammar_cache = grammar_cache if grammar_cache is not None else GrammarLRU(maxsize=64)
        # Single-flight table: concurrent first-requests for the same
        # ``(tokenizer, schema, backend)`` triple all wait on the same
        # :class:`asyncio.Future` instead of each running their own
        # compile. The dict is keyed by the cache key and cleared
        # after the compile resolves (success or failure). Mutation
        # is guarded by ``_grammar_inflight_lock`` because the same
        # key can be looked up from multiple coroutines.
        self._grammar_inflight: dict[tuple[str, str, str], asyncio.Future[Any]] = {}
        self._grammar_inflight_lock = asyncio.Lock()
        # KV-cache admission control.
        #
        # The constructor seeds the budget from the boot-time profile
        # resolution (``_kv_budget_tokens`` parameter), but :meth:`process`
        # re-resolves it from the per-request model's profile so that
        # later-loaded models (model_filter excluded at boot, hot-reload,
        # ``rescan_configs`` discovery) still drive the gate. This also
        # means a single worker can host multiple generation profiles
        # with distinct budgets (the speculative side-cell case) — the
        # ``_reserved`` accumulator still bounds total in-flight reserve.
        #
        # ``_reserved`` tracks the total ``input_tokens_estimate +
        # max_new_tokens`` summed across in-flight generations; the lock
        # guards reserve/release against concurrent ``process()`` calls.
        # Even when admission is False the gauges still update on every
        # reserve/release — the routing saturation gate reads
        # ``reserved / budget`` regardless of the admission flag.
        self._kv_budget_tokens = kv_budget_tokens
        self._admission_enabled = admission_enabled
        self._admission_resolver = admission_resolver
        if admission_enabled and kv_budget_tokens is None:
            logger.warning(
                "StreamingProcessor admission_enabled=True but kv_budget_tokens is None — "
                "admission gate will be inert until a generation profile with "
                "kv_budget_tokens is resolved per-request",
            )
        self._reserved: int = 0
        self._generation_inflight: int = 0
        self._admission_lock = asyncio.Lock()
        # Per-released-request sentinel: guards against double-release in
        # the (defensive) case that two finally blocks both try to drop
        # the reservation. Bounded by ``_RELEASED_REQUESTS_MAX`` with
        # FIFO eviction so the set doesn't grow without bound across
        # the worker's lifetime; the eviction window is wide enough
        # (1024 ids) that the realistic window in which a double-
        # release could plausibly fire is fully covered.
        self._released_requests: collections.OrderedDict[str, None] = collections.OrderedDict()
        # H9 — cancel tombstones. Maps ``request_id -> expiry_monotonic`` so a
        # cancel that arrives before any decode attempt registers can still
        # block a later decode-start for the same request_id. Keyed on
        # ``request_id`` so the (gateway-driven) cancel signal — which knows
        # only ``request_id``, not the per-pickup ``attempt_id`` — can write
        # an entry. Order is *roughly* insertion (Python 3.7+ dict guarantee);
        # cleanup is lazy on insert.
        self._cancel_tombstones: dict[str, float] = {}
        self._completion_by_attempt: dict[str, _GenerationCompletion] = {}

    # -- Grammar prewarm -------------------------------------------

    async def prewarm_grammars_for_model(self, model_id: str) -> None:
        """Compile and cache ``tasks.generate.prewarm_grammars`` for ``model_id``.

        Called once per generation model at worker boot through
        ``IpcServer`` after the registry knows the config but before the
        sidecar dispatches request-shape work.
        The cache key shape mirrors :meth:`_ensure_grammar_ready` exactly
        so a request hitting one of the prewarmed grammars takes the
        cache-hit branch.

        Failure model: a single entry's failure (missing Outlines,
        invalid schema, tokenizer load error) increments the
        ``outcome="failed"`` counter, logs an ERROR, and the loop
        continues to the next entry. The model still loads. Operators
        see the failure via metrics + logs rather than as a startup crash.

        Non-generation models are silent no-ops (the
        :func:`~sie_server.core.pool_isolation.is_generation_model` gate
        runs at the call site; this method also short-circuits on
        missing ``tasks.generate`` for defence-in-depth).
        """
        try:
            config = self._registry.get_config(model_id)
        except KeyError:
            # Race: registry mutated between the call-site filter and this
            # call (hot-reload, model removal). Silent no-op rather than
            # crashing the start-up sequence.
            logger.debug("prewarm: model %s not in registry; skipping", model_id)
            return

        gen = config.tasks.generate
        if gen is None or not gen.prewarm_grammars:
            return

        grammar_backend = self._grammar_backend_for_model_config(config)
        if grammar_backend != "outlines":
            logger.info(
                "prewarm: skipping %d grammar(s) for model %s because grammar_backend=%s",
                len(gen.prewarm_grammars),
                model_id,
                grammar_backend or "default",
            )
            return

        # Tokenizer hash derivation mirrors :meth:`_ensure_grammar_ready`
        # so the cache key collides correctly with later request-time
        # lookups. Keep both call sites in sync.
        tokenizer_hash = str(config.hf_id or config.weights_path or model_id)

        logger.info(
            "prewarming %d grammar(s) for model %s",
            len(gen.prewarm_grammars),
            model_id,
        )

        for entry in gen.prewarm_grammars:
            await self._prewarm_one(
                model_id=model_id,
                tokenizer_hash=tokenizer_hash,
                name=entry.name,
                kind=entry.kind,
                value=entry.value,
            )

    async def _prewarm_one(
        self,
        *,
        model_id: str,
        tokenizer_hash: str,
        name: str,
        kind: str,
        value: dict[str, Any] | str,
    ) -> None:
        """Compile a single prewarm entry and populate the cache.

        Isolated so the per-entry try/except in
        :meth:`prewarm_grammars_for_model` stays tight; a failure here
        records the metric and returns rather than propagating.
        """
        grammar = GrammarSpec(kind=cast("Any", kind), value=value, label=name)
        key: tuple[str, str, str] = (tokenizer_hash, hash_grammar(grammar), "outlines")

        # Idempotency: if the same ``(name, value)`` was already
        # prewarmed (e.g. duplicate entry, or a second
        # ``prewarm_grammars_for_model`` call after hot-reload) skip the
        # compile rather than double-counting.
        if self._grammar_cache.get(key) is not None:
            _metrics.worker_telemetry().grammar_cache_lookup(
                model=model_id,
                backend="outlines",
                grammar=kind,
                phase="prewarm",
                result="hit",
            )
            logger.debug("prewarm: cache already populated for %s/%s; skipping", model_id, name)
            return

        _metrics.worker_telemetry().grammar_cache_lookup(
            model=model_id,
            backend="outlines",
            grammar=kind,
            phase="prewarm",
            result="miss",
        )
        t0 = time.monotonic()
        try:
            tok = await self._get_tokenizer(model_id)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            _metrics.worker_telemetry().grammar_compile_completed(
                model=model_id,
                backend="outlines",
                grammar=kind,
                phase="prewarm",
                outcome="error",
                duration_s=elapsed,
            )
            logger.error(
                "prewarm: tokenizer load failed for %s/%s (%.3fs): %s",
                model_id,
                name,
                elapsed,
                exc,
            )
            return

        try:
            # Use the dedicated bounded grammar executor (same as the
            # request-path compile) so a hung Outlines compile cannot
            # exhaust the default ``asyncio.to_thread`` pool that the
            # rest of the worker shares (tokenizer loads, chat-template
            # renders, etc.).
            loop = asyncio.get_running_loop()
            compiled = await loop.run_in_executor(_GRAMMAR_EXECUTOR, compile_outlines, tok, grammar)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            _metrics.worker_telemetry().grammar_compile_completed(
                model=model_id,
                backend="outlines",
                grammar=kind,
                phase="prewarm",
                outcome="error",
                duration_s=elapsed,
            )
            # ERROR not WARNING: a misconfigured prewarm entry is operator-
            # actionable and should not be lost in info-level noise.
            logger.error(
                "prewarm: compile failed for %s/%s (%.3fs): %s",
                model_id,
                name,
                elapsed,
                exc,
            )
            return

        elapsed = time.monotonic() - t0
        self._grammar_cache.put(key, compiled)
        _metrics.worker_telemetry().grammar_compile_completed(
            model=model_id,
            backend="outlines",
            grammar=kind,
            phase="prewarm",
            outcome="success",
            duration_s=elapsed,
        )
        logger.info(
            "prewarm: %s/%s (%s) compiled in %.3fs",
            model_id,
            name,
            kind,
            elapsed,
        )

    # -- KV-cache admission control ------------------------------------------

    @property
    def kv_budget_tokens(self) -> int | None:
        """Per-worker KV-cache budget; ``None`` if no budget plumbed in."""
        return self._kv_budget_tokens

    @property
    def admission_enabled(self) -> bool:
        """Whether admission control will actively reject overload."""
        return self._admission_enabled

    def kv_reserved_tokens(self) -> int:
        """Currently-reserved KV tokens across all in-flight generations."""
        return self._reserved

    def _resolve_budget_for_model(self, model_id: str) -> int | None:
        """Look up the per-model ``kv_budget_tokens`` from the registry.

        Returns ``None`` when the model has no config, no generation
        task, or fails to resolve its ``default`` profile. The caller
        then falls back to the worker's boot-time ``_kv_budget_tokens``.

        Defensive: returns ``None`` for any value that isn't a positive
        integer so a misconfigured registry / test MagicMock can't
        propagate junk into the admission decision.
        """
        try:
            cfg = self._registry.get_config(model_id)
        except (KeyError, AttributeError):
            return None
        try:
            if cfg.tasks.generate is None:
                return None
            resolved = cfg.resolve_profile("default")
        except Exception:  # noqa: BLE001 — defensive: malformed profiles fall back to boot-time budget
            logger.debug(
                "Failed to resolve 'default' profile for admission lookup on %s",
                model_id,
                exc_info=True,
            )
            return None
        budget = resolved.kv_budget_tokens
        if not isinstance(budget, int) or budget <= 0:
            return None
        return budget

    def _resolve_admission_for_model(self, model_id: str) -> tuple[int | None, bool | None]:
        if self._admission_resolver is None:
            return self._resolve_budget_for_model(model_id), None
        try:
            budget, enabled = self._admission_resolver(model_id)
        except Exception:  # noqa: BLE001
            # Log at WARNING (not debug) so a buggy resolver doesn't
            # silently disable admission for the affected model — the
            # gauges keep ticking, but operators get no signal at debug
            # level in production logs. The metric counter lets a
            # dashboard surface the rate of resolver failures.
            logger.warning("Failed to resolve generation admission for %s", model_id, exc_info=True)
            _metrics.worker_telemetry().admission_decided(
                model=model_id,
                outcome="error",
                reason="resolver_error",
            )
            return None, None
        if not isinstance(budget, int) or budget <= 0:
            budget = None
        if not isinstance(enabled, bool):
            enabled = None
        return budget, enabled

    @staticmethod
    def _adapter_uses_outlines_grammar(adapter: GenerationAdapter) -> bool:
        """Return whether worker-side Outlines preflight should run."""
        return getattr(adapter, "_grammar_backend", "outlines") == "outlines"

    @staticmethod
    def _resolve_grammar_backend_label(adapter: GenerationAdapter) -> str:
        """Bounded-cardinality backend label for structured-output metrics.

        Reads the adapter's ``_grammar_backend`` (the same field used to
        drive the SGLang ``--grammar-backend`` flag). Anything that
        isn't one of the known SGLang backends collapses to
        ``"other"`` to keep the telemetry label cardinality bounded.
        """
        backend = getattr(adapter, "_grammar_backend", None)
        if backend in ("outlines", "xgrammar", "llguidance"):
            return str(backend)
        return "other"

    def _record_grammar_observed(
        self,
        adapter: GenerationAdapter,
        grammar: GrammarSpec,
        model_id: str,
    ) -> None:
        """Record one bounded structured-output request event.

        Runs on every structured-output request, regardless of whether
        diagnostic preflight is enabled. No schema hash, name, or payload
        becomes an attribute.
        """
        backend = self._resolve_grammar_backend_label(adapter)
        _metrics.worker_telemetry().grammar_requested(
            model=model_id,
            backend=backend,
            grammar=grammar.kind,
        )

    def _resolve_tool_call_format(self, model_id: str) -> ToolCallFormat:
        """Resolve the on-wire tool-call format from the model config.

        Drives format selection from a single source of truth — the
        adapter's ``tool_call_parser`` (the same value passed to SGLang's
        ``--tool-call-parser`` launch flag) — instead of guessing per
        block from the model output. ``qwen3_coder`` (and any other
        ``qwen*`` parser) → ``qwen_xml``; ``hermes`` → ``hermes_json``;
        anything unknown / missing → ``auto`` (keep the runtime
        heuristic so out-of-tree models still work).
        """
        try:
            config = self._registry.get_config(model_id)
            resolved = config.resolve_profile("default")
        except Exception:  # noqa: BLE001 - missing config falls back to auto-detect
            return "auto"
        loadtime = getattr(resolved, "loadtime", {})
        if not hasattr(loadtime, "get"):
            return "auto"
        parser = loadtime.get("tool_call_parser")
        if not isinstance(parser, str):
            return "auto"
        parser_l = parser.lower()
        if parser_l.startswith("qwen"):
            return "qwen_xml"
        if "hermes" in parser_l:
            return "hermes_json"
        return "auto"

    @staticmethod
    def _grammar_backend_for_model_config(config: Any) -> str | None:
        """Resolve configured SGLang grammar backend for prewarm.

        Missing config preserves the historical Outlines prewarm path;
        profiles that explicitly opt into XGrammar bypass Outlines.
        """
        try:
            resolved = config.resolve_profile("default")
        except Exception:  # noqa: BLE001
            logger.debug("Failed to resolve default profile for grammar prewarm", exc_info=True)
            return "outlines"
        loadtime = getattr(resolved, "loadtime", {})
        if not hasattr(loadtime, "get"):
            return "outlines"
        backend = loadtime.get("grammar_backend", "outlines")
        return str(backend) if backend is not None else None

    @staticmethod
    def _release_dedup_key(request_id: str | None, attempt_id: str | None) -> str | None:
        """Dedup key for reserve/release idempotency.

        Prefers ``attempt_id`` (BUG 2/7): two legitimate attempts for the
        SAME ``request_id`` (redelivery after ``ack_wait``) each make their
        own reservation, so deduping the release on ``request_id`` would
        wrongly drop the second attempt's release and leak budget. Falls
        back to ``request_id`` for legacy callers / unit tests that don't
        thread an ``attempt_id`` through.
        """
        if attempt_id is not None:
            return attempt_id
        return request_id

    async def _try_reserve(
        self,
        model_id: str,
        reserve_tokens: int,
        *,
        budget_override: int | None = None,
        admission_enabled_override: bool | None = None,
        request_id: str | None = None,
        attempt_id: str | None = None,
    ) -> bool:
        """Reserve KV budget for an in-flight generation.

        ``budget_override`` — when set, replaces the worker-wide
        ``_kv_budget_tokens`` for this admission decision. Used to
        honour per-profile budgets resolved lazily in :meth:`process`
        so model_filter / rescan-discovered generation profiles still
        drive the gate.

        When admission is **on** and the reservation would exceed the
        budget, returns ``False`` without mutating state — the caller
        publishes a ``kind:"nak"`` envelope with ``reason="kv_budget"``.

        When admission is **off** (or no budget is configured), the
        reservation always succeeds; the lock-protected counters still
        update so the gauges and the routing saturation signal stay live.
        """
        budget = budget_override if budget_override is not None else self._kv_budget_tokens
        admission_enabled = (
            admission_enabled_override if admission_enabled_override is not None else self._admission_enabled
        )
        async with self._admission_lock:
            if admission_enabled and budget is None:
                # Misconfigured: admission was resolved as enabled but no
                # budget reached this call. The boot-time warning fires
                # once on construction; this fires once per request and
                # is the diagnostic operators will hit when a config-load
                # bug lets a generation profile through without a budget.
                logger.warning(
                    "Admission enabled for model %s but no kv_budget_tokens "
                    "resolved — admission gate is inert; check config-load.",
                    model_id,
                )
                _metrics.worker_telemetry().admission_decided(
                    model=model_id,
                    outcome="error",
                    reason="missing_budget",
                )
            if admission_enabled and budget is not None and self._reserved + reserve_tokens > budget:
                return False
            self._reserved += reserve_tokens
            self._generation_inflight += 1
            _metrics.worker_telemetry().state_changed(
                model=model_id,
                reserved_tokens=self._reserved,
                inflight=self._generation_inflight,
                budget_tokens=budget,
            )
            _metrics.worker_telemetry().admission_decided(
                model=model_id,
                outcome="admitted",
                reason="none",
            )
            dedup_key = self._release_dedup_key(request_id, attempt_id)
            if dedup_key is not None:
                # Allow ``_release_reservation`` to act exactly once for
                # this attempt (keyed on attempt_id so concurrent attempts
                # for the same request_id don't dedup each other — BUG 2/7).
                self._released_requests.pop(dedup_key, None)
        return True

    async def _release_reservation(
        self,
        model_id: str,
        reserve_tokens: int,
        *,
        request_id: str | None = None,
        attempt_id: str | None = None,
        budget_tokens: int | None = None,
    ) -> None:
        """Release a previously-acquired KV reservation.

        Idempotent per-attempt: a duplicate call for the same
        ``attempt_id`` is a no-op. The dedup key is the ``attempt_id``
        when supplied (BUG 2/7 — two legitimate attempts for the same
        ``request_id`` after redelivery must each release their own
        reservation), falling back to ``request_id`` for legacy callers /
        unit tests. Without either, the gauge ``dec()`` is still guarded by
        clamping ``_reserved`` at zero, but the in-flight gauge could in
        principle drift — production callers always supply both.
        """
        dedup_key = self._release_dedup_key(request_id, attempt_id)
        async with self._admission_lock:
            if dedup_key is not None:
                if dedup_key in self._released_requests:
                    return
                self._released_requests[dedup_key] = None
                # FIFO eviction so the set doesn't grow without bound.
                while len(self._released_requests) > _RELEASED_REQUESTS_MAX:
                    self._released_requests.popitem(last=False)
            self._reserved = max(0, self._reserved - reserve_tokens)
            self._generation_inflight = max(0, self._generation_inflight - 1)
            _metrics.worker_telemetry().state_changed(
                model=model_id,
                reserved_tokens=self._reserved,
                inflight=self._generation_inflight,
                budget_tokens=budget_tokens,
            )

    # -- Cancel registration (called by the sidecar cancel subscriber) --------

    def _register_cancel(self, request_id: str, attempt_id: str, event: asyncio.Event) -> None:
        """Register a per-attempt cancel handle.

        Concurrent attempts for the same ``request_id`` (redelivery after
        ``ack_wait``) each register under their own ``attempt_id`` so they
        don't clobber each other (BUG 2/7).
        """
        self._in_flight_cancels.setdefault(request_id, {})[attempt_id] = event

    def _unregister_cancel(self, request_id: str, attempt_id: str) -> None:
        """Remove ONLY this attempt's cancel handle.

        Deletes the ``request_id`` entry only once its last attempt is gone,
        so an attempt finishing never tears down a sibling attempt's handle.
        """
        attempts = self._in_flight_cancels.get(request_id)
        if attempts is None:
            return
        attempts.pop(attempt_id, None)
        if not attempts:
            self._in_flight_cancels.pop(request_id, None)

    def signal_cancel(self, request_id: str) -> bool:
        """Mark in-flight generation(s) as cancelled. Returns True if matched.

        Sets the cancel event for EVERY live attempt of ``request_id`` so a
        single cancel reaches all concurrent attempts (BUG 2/7). The
        ``request_id``-only signature is the sidecar IPC cancel contract.

        H9 — when no attempt is registered yet (the cancel arrived before
        the work-item pull / decode-start), write a TTL'd tombstone so the
        decode-start check refuses to decode if the work item shows up
        later. Returns True if any live attempt was signalled OR a tombstone
        was written; the boolean is informational only.
        """
        attempts = self._in_flight_cancels.get(request_id)
        if attempts:
            for event in attempts.values():
                event.set()
            return True
        # No live attempt: stash a tombstone so a later decode-start for
        # this request_id refuses to run.
        self._add_tombstone(request_id)
        return False

    def _add_tombstone(self, request_id: str) -> None:
        """Insert a cancel tombstone with lazy cleanup.

        Sweeps expired entries when the map exceeds the soft cap; if the
        sweep didn't free space (every tombstone still live), evict the
        oldest. Insertion order is preserved by Python's dict ordering
        (3.7+), so the first ``next(iter(...))`` is the eldest.
        """
        now = time.monotonic()
        deadline = now + _CANCEL_TOMBSTONE_TTL_S
        if len(self._cancel_tombstones) >= _CANCEL_TOMBSTONE_MAX:
            # Sweep expired entries first.
            expired = [rid for rid, exp in self._cancel_tombstones.items() if exp <= now]
            for rid in expired:
                self._cancel_tombstones.pop(rid, None)
            # If still at cap, evict the oldest entry.
            while len(self._cancel_tombstones) >= _CANCEL_TOMBSTONE_MAX:
                try:
                    oldest = next(iter(self._cancel_tombstones))
                except StopIteration:
                    break
                self._cancel_tombstones.pop(oldest, None)
        # Refresh the entry (re-insert moves to the tail in dict order so
        # FIFO eviction targets the genuinely oldest tombstone).
        self._cancel_tombstones.pop(request_id, None)
        self._cancel_tombstones[request_id] = deadline

    def _check_and_consume_tombstone(self, request_id: str) -> bool:
        """Return True iff ``request_id`` has a live (unexpired) tombstone.

        Consumes the tombstone in both branches (live or expired) so a
        legitimate later request reusing the same id is not blocked. The
        caller increments the duplicate-execution metric on a live hit.
        """
        deadline = self._cancel_tombstones.pop(request_id, None)
        if deadline is None:
            return False
        return time.monotonic() < deadline

    def in_flight_request_ids(self) -> set[str]:
        return set(self._in_flight_cancels.keys())

    def in_flight_count(self) -> int:
        """Number of generate requests currently being processed by this worker.

        Drives the saturation gate's ``in_flight / capacity`` fraction.
        Counts *requests*, not fetched batches; per-batch accounting
        dramatically understates concurrency under fan-in load.
        """
        return len(self._in_flight_cancels)

    # -- Main entry point ----------------------------------------------------

    async def process(self, msg: Any, model_id: str) -> None:
        try:
            wi: WorkItem = msgpack.unpackb(msg.data, raw=False)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to deserialize generate work item", exc_info=True)
            try:
                await msg.nak()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to NAK undecodable generate msg")
            return

        reply_subject = wi.get("reply_subject", "")
        request_id = wi.get("request_id", "")
        if not reply_subject:
            logger.warning("Generate work item missing reply_subject; ACKing without reply")
            await _safe_ack(msg)
            return

        # Every pickup gets a fresh attempt_id. Redelivery after a
        # worker crash produces a new attempt; the gateway latches the first
        # attempt_id it observes and drops chunks from any other.
        attempt_id = uuid.uuid4().hex
        if self._record_worker_completion and _metrics.worker_telemetry_enabled():
            self._completion_by_attempt[attempt_id] = _GenerationCompletion(
                model_id=model_id,
                profile_id=str(wi.get("profile_id") or "default"),
                started_at=time.perf_counter(),
            )

        # W3C Trace Context propagation. If the gateway populated
        # the ``traceparent`` (and optionally ``tracestate``) fields on
        # the envelope, extract a parent ``opentelemetry.Context`` and
        # open the worker-side processing span as its child. The span
        # stays attached for the rest of this handler so the
        # adapter's own ``tracing`` calls (and any sub-spans the
        # SGLang adapter opens) inherit it automatically.
        #
        # When no traceparent is present we open a root span — the
        # worker still emits a span, it just isn't correlated with a
        # client-side trace. ``opentelemetry.propagate.extract`` is a
        # no-op on an empty carrier so this is safe either way.
        carrier: dict[str, str] = {}
        if (tp := wi.get("traceparent")) is not None:
            carrier["traceparent"] = tp
        if (ts := wi.get("tracestate")) is not None:
            carrier["tracestate"] = ts
        parent_ctx = propagate.extract(carrier)
        tracer = _otel_trace.get_tracer("sie_server.processors.streaming")
        try:
            with tracer.start_as_current_span(
                "worker.streaming_processor",
                context=parent_ctx,
                attributes={
                    "sie.request_id": request_id,
                    "sie.attempt_id": attempt_id,
                    "sie.model": model_id,
                    "sie.adapter": "streaming",
                },
            ):
                await self._process_inner(msg, model_id, wi, reply_subject, request_id, attempt_id)
        finally:
            self._completion_by_attempt.pop(attempt_id, None)

    async def _process_inner(
        self,
        msg: Any,
        model_id: str,
        wi: WorkItem,
        reply_subject: str,
        request_id: str,
        attempt_id: str,
    ) -> None:
        """Process a generate work item with all trace context already attached.

        Extracted out of :meth:`process` to keep the trace-context
        boundary tight: the parent span is opened in ``process``,
        every code path below sees it as the current context, and
        the span auto-closes on exit. Any future restructuring of
        the generate pipeline should land here, not in ``process``.
        """
        # Register the cancel handle as EARLY as possible — right after the
        # request_id is known, BEFORE the (potentially multi-second) model
        # load, grammar compile, and KV-admission reserve. Previously the
        # handle was registered only after admission, so a cancel arriving
        # during that window had nothing to set and was silently lost; the
        # generation then ran to completion despite the client having gone
        # away. ``signal_cancel`` sets this event; ``_stream_generate``
        # selects on it to tear the iterator down. The handle is removed in
        # the ``finally`` at the bottom on every exit path (early returns,
        # exceptions, normal completion).
        cancel_event = asyncio.Event()
        # Register per-attempt so a concurrent redelivery (same request_id,
        # different attempt_id) doesn't clobber this handle, and so this
        # attempt's cleanup removes only its own (BUG 2/7).
        self._register_cancel(request_id, attempt_id, cancel_event)
        try:
            await self._process_inner_guarded(
                msg=msg,
                model_id=model_id,
                wi=wi,
                reply_subject=reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                cancel_event=cancel_event,
            )
        finally:
            self._unregister_cancel(request_id, attempt_id)

    async def _process_inner_guarded(
        self,
        *,
        msg: Any,
        model_id: str,
        wi: WorkItem,
        reply_subject: str,
        request_id: str,
        attempt_id: str,
        cancel_event: asyncio.Event,
    ) -> None:
        """Body of :meth:`_process_inner`, run with the cancel handle live.

        Split out so the cancel handle is registered/cleaned up by the
        caller around the whole body — including the model load, grammar
        compile, and KV-admission reserve — so a cancel arriving in that
        early window is honoured.
        """
        # H9 — cancel tombstone check.
        #
        # If the gateway's first-chunk fallback cancelled this request_id
        # BEFORE this worker process registered the in-flight handle,
        # ``signal_cancel`` left a tombstone behind. The pool-republished
        # attempt is already in flight (or has completed) on another worker;
        # decoding here would burn GPU/KV/billing for a request the client
        # has long since transitioned to a different attempt. Refuse to
        # decode: emit a transport_failure terminal so the gateway closes
        # this attempt's collector slot cleanly (the stale-attempt filter
        # in queue/streaming.rs will treat any chunks from this attempt_id
        # as stale even if the terminal races), then ACK so JetStream does
        # not redeliver indefinitely.
        if self._check_and_consume_tombstone(request_id):
            pool_label = str(wi.get("pool_name") or "default")
            _metrics.worker_telemetry().duplicate_prevented(
                model=model_id,
                path="first_chunk_fallback",
            )
            logger.warning(
                "cancel-tombstone hit: refusing to decode request_id=%s model=%s pool=%s "
                "(cancel arrived before this worker registered; pool-republished attempt is authoritative)",
                request_id,
                model_id,
                pool_label,
            )
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code="transport_failure",
                message=(
                    "request cancelled before this worker registered (first-chunk fallback race); "
                    "another attempt is authoritative"
                ),
                msg=msg,
            )
            return

        # Lazy-load the model on first request. NAK for redelivery if
        # load fails or the model isn't registered for this worker.
        try:
            adapter = await self._ensure_loaded(model_id)
        except KeyError:
            logger.info(
                "Model %s not registered for generate; emitting inbox NAK + ACKing",
                model_id,
            )
            # Surface an inbox-side NAK envelope so the
            # gateway can republish to the pool subject (likely
            # reaching a worker that has the model loaded), then ACK
            # the JetStream message so it is *not* redelivered to
            # this worker. Pre-fix we also called ``_safe_nak`` which
            # caused JetStream to redeliver the same message up to
            # ``_MAX_DELIVER`` times — each redelivery emitted another
            # inbox NAK that the gateway's idempotency guard had to
            # drop, wasting ~100s of effort per request. Acking
            # avoids the storm; the gateway's republish is the
            # authoritative retry path.
            nak_ok = await self._publish_nak_envelope(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                reason="model_not_loaded",
            )
            # Only ACK if the NAK reached the gateway — otherwise let
            # JetStream redeliver so the request isn't orphaned.
            if nak_ok:
                await _safe_ack(msg)
            return
        except Exception:  # noqa: BLE001
            # A load failure (vs. a registration miss) may be
            # transient (memory pressure, hot reload races). Keep the
            # JetStream NAK here so redelivery still gives the
            # worker a chance to retry; do not emit the inbox NAK
            # because the gateway shouldn't route around a worker
            # that's still owning the model.
            logger.warning("Failed to load model %s for generate", model_id, exc_info=True)
            await _safe_nak(msg)
            return

        if not isinstance(adapter, GenerationAdapter):
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code="unsupported_field",
                message=f"Model '{model_id}' adapter does not support generate (not a GenerationAdapter)",
                msg=msg,
            )
            return

        # Cancel may have arrived during the (potentially multi-second)
        # cold load above. Honour it before any further preflight work.
        if cancel_event.is_set():
            await self._publish_cancelled_then_settle(
                reply_subject, request_id=request_id, attempt_id=attempt_id, msg=msg
            )
            return

        # Routing-key plumbing: inert at this layer. Read so the values land
        # in a single place, but do not act on them.
        _ = wi.get("routing_key")  # routing-affinity hint
        _ = wi.get("prompt_cache_key")  # prompt-cache hint

        try:
            config = self._registry.get_config(model_id)
        except (KeyError, AttributeError):
            config = None
        validation = self._validate_generate_params(wi, config)
        if isinstance(validation, _ValidationError):
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code=validation.code,
                message=validation.message,
                msg=msg,
            )
            return
        params = validation

        # Resolve ``tool_choice`` mode up front. When the caller set
        # ``tool_choice == "none"`` the OpenAI contract is that the model
        # must not call any tool — enforce this at the *prompt* layer by
        # never rendering tool definitions into the chat template. With
        # the model literally unaware of the tools it cannot emit
        # ``<tool_call>`` syntax in the first place, so we no longer have
        # to scrub the output stream. ``params.tool_choice`` is kept on
        # the work params (for telemetry / audit); only the effective
        # view passed downstream is cleared.
        tool_choice_mode, _tool_choice_name = normalize_tool_choice(params.tool_choice)
        effective_tools: tuple[dict[str, Any], ...] | None = None if tool_choice_mode == "none" else params.tools

        # Chat-template rendering. For ``Messages`` shape we
        # tokenize-via-template before handing the rendered string to
        # the underlying adapter. ``Prompt`` shape goes straight through.
        prompt_str: str
        # Flat, in-order image list for vision requests — lines up with the
        # placeholders the chat template renders. ``None`` for the prompt
        # shape and for text-only message lists.
        request_images: list[ImageInput] | None = None
        if isinstance(params.input, _MessagesInput):
            collected = [img for m in params.input.messages for img in (m.images or ())]
            request_images = collected or None
            # Worker-side vision capability gate (defense-in-depth for the
            # queue-bypass path). Run it BEFORE chat-template rendering so a
            # text-only model handed images fails fast with a clear error
            # rather than doing — or erroring inside — the CPU-bound render.
            # Skip only when the config is unavailable (mirrors the
            # context-length check's defensive skip).
            if request_images:
                try:
                    vision_config = self._registry.get_config(model_id)
                except KeyError:
                    vision_config = None
                if vision_config is not None and not vision_config.inputs.image:
                    await self._terminal_error_then_settle(
                        reply_subject,
                        request_id=request_id,
                        attempt_id=attempt_id,
                        seq=0,
                        code="invalid_request",
                        message=f"model '{model_id}' does not support image input",
                        msg=msg,
                    )
                    return
            rendered = await self._render_chat_template(
                model_id,
                params.input.messages,
                effective_tools,
                request_kwargs=params.chat_template_kwargs,
            )
            if isinstance(rendered, _ValidationError):
                await self._terminal_error_then_settle(
                    reply_subject,
                    request_id=request_id,
                    attempt_id=attempt_id,
                    seq=0,
                    code=rendered.code,
                    message=rendered.message,
                    msg=msg,
                )
                return
            prompt_str = rendered
        else:
            prompt_str = params.input.prompt

        # Cancel may have arrived during chat-template rendering (CPU-bound
        # on a large message list).
        if cancel_event.is_set():
            await self._publish_cancelled_then_settle(
                reply_subject, request_id=request_id, attempt_id=attempt_id, msg=msg
            )
            return

        # §4.3: worker-side context-length validation. Applies
        # to both Prompt and Messages paths. The rendered string is
        # already in memory; tokenizing is fast on a cached fast
        # tokenizer (<1ms for typical prompts). Vision requests add a coarse
        # per-image token estimate so placeholder expansion doesn't overflow
        # the context window unaccounted.
        ctx_error = await self._check_context_length(
            model_id,
            prompt_str,
            params.max_new_tokens,
            num_images=len(request_images) if request_images else 0,
        )
        if ctx_error is not None:
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code=ctx_error.code,
                message=ctx_error.message,
                msg=msg,
            )
            return

        # Cancel may have arrived during context-length tokenization.
        if cancel_event.is_set():
            await self._publish_cancelled_then_settle(
                reply_subject, request_id=request_id, attempt_id=attempt_id, msg=msg
            )
            return

        # Resolve the on-wire tool-call format once (config-driven, not a
        # per-block heuristic) and, for an enforced tool_choice
        # ("required" / named function), build a constrained-decoding
        # grammar that forces a well-formed tool call. "auto"/"none"/absent
        # yield no forcing grammar. ``effective_tools`` is used here
        # rather than ``params.tools`` so the ``tool_choice == "none"``
        # branch (where tools have already been hidden from the prompt)
        # also skips forcing-grammar construction and parser wrapping.
        tool_call_format: ToolCallFormat = self._resolve_tool_call_format(model_id) if effective_tools else "auto"
        forcing_grammar: GrammarSpec | None = None
        if effective_tools:
            try:
                forcing_grammar = build_tool_choice_grammar(effective_tools, params.tool_choice, tool_call_format)
            except ToolChoiceError as exc:
                await self._terminal_error_then_settle(
                    reply_subject,
                    request_id=request_id,
                    attempt_id=attempt_id,
                    seq=0,
                    code="invalid_request",
                    message=str(exc),
                    msg=msg,
                )
                return

        # tool_choice == "none" has already been enforced upstream by
        # passing ``tools=None`` to the chat template; the model literally
        # cannot emit ``<tool_call>`` syntax, so wrapping the parser would
        # be a no-op. Skip it.
        enable_tool_parser = bool(effective_tools)

        # A forced tool_choice grammar takes precedence over a user
        # response_format grammar; the two are mutually exclusive (the
        # gateway rejects the combo before the request reaches here).
        effective_grammar = forcing_grammar if forcing_grammar is not None else params.grammar

        # SGLang owns request-time grammar compilation.
        #
        # The default path forwards the raw schema/regex/EBNF straight to
        # SGLang's server-side grammar backend (the single grammar
        # authority on the request hot path). The worker-side Outlines
        # preflight is preserved behind ``SIE_GRAMMAR_PREFLIGHT_DEBUG=1``
        # for diagnostic use only — see ``_grammar_preflight_debug_enabled``
        # and ``docs/adr/0002-sglang-owns-request-time-grammar.md``.
        #
        # One bounded request event always runs so operators see structured-
        # output traffic regardless of the debug flag.
        if effective_grammar is not None:
            self._record_grammar_observed(adapter, effective_grammar, model_id)

            if self._adapter_uses_outlines_grammar(adapter) and _grammar_preflight_debug_enabled():
                ready = await self._ensure_grammar_ready(
                    effective_grammar,
                    model_id=model_id,
                    reply_subject=reply_subject,
                    request_id=request_id,
                    attempt_id=attempt_id,
                    msg=msg,
                )
                if not ready:
                    return

                # Cancel may have arrived during the (up to 5s) grammar
                # compile. Honour it before reserving KV budget / decoding.
                if cancel_event.is_set():
                    await self._publish_cancelled_then_settle(
                        reply_subject, request_id=request_id, attempt_id=attempt_id, msg=msg
                    )
                    return

        # KV-cache admission control. The reservation covers
        # both the (cheap, char-based) input-token estimate and the
        # client-requested ``max_new_tokens``. When admission is on and
        # the budget is exhausted, publish a NAK envelope with
        # ``reason="kv_budget"`` so the gateway can re-publish to a
        # different worker (via the routing plumbing), then ACK the JetStream
        # msg to avoid a redelivery storm (mirrors the model-not-loaded
        # pattern documented above). When admission is off the reserve
        # still happens — gauges and the routing saturation signal depend
        # on a live ``_reserved`` counter regardless of the flag.
        #
        # Budget is resolved lazily from this model's resolved profile so
        # late-loaded models (model_filter exclusions, hot reload,
        # rescan_configs discovery) still drive the gate. Falls back to
        # the worker's boot-time default when the lookup fails.
        budget_override, admission_enabled_override = self._resolve_admission_for_model(model_id)
        # Include the coarse per-image token estimate so vision requests
        # reserve KV budget proportional to their real (placeholder-expanded)
        # footprint instead of just the short placeholder text.
        image_reserve = (len(request_images) if request_images else 0) * _VISION_TOKENS_PER_IMAGE_ESTIMATE
        reserve_tokens = estimate_tokens_from_chars(prompt_str) + params.max_new_tokens + image_reserve
        effective_budget = budget_override if budget_override is not None else self._kv_budget_tokens
        admitted = await self._try_reserve(
            model_id,
            reserve_tokens,
            budget_override=budget_override,
            admission_enabled_override=admission_enabled_override,
            request_id=request_id,
            attempt_id=attempt_id,
        )
        if not admitted:
            logger.info(
                "Admission reject for %s/%s: reserved=%d + req=%d > budget=%s",
                request_id,
                attempt_id,
                self._reserved,
                reserve_tokens,
                effective_budget,
            )
            _metrics.worker_telemetry().admission_decided(
                model=model_id,
                outcome="rejected",
                reason="kv_budget",
            )
            nak_ok = await self._publish_nak_envelope(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                reason="kv_budget",
            )
            # Only ACK if the gateway got the NAK — otherwise let
            # JetStream redeliver so the request isn't orphaned.
            if nak_ok:
                await _safe_ack(msg)
            return

        # The cancel handle is already registered by ``_process_inner`` (so
        # cancels during load/compile/admission are honoured); just pass it
        # through to the streaming core. Cleanup of ``_in_flight_cancels``
        # happens in ``_process_inner``'s finally.
        try:
            await self._stream_generate(
                msg=msg,
                adapter=adapter,
                reply_subject=reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                prompt=prompt_str,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                stop=params.stop,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                top_k=params.top_k,
                repetition_penalty=params.repetition_penalty,
                min_tokens=params.min_tokens,
                chat_template_kwargs=params.chat_template_kwargs,
                grammar=effective_grammar,
                tools=effective_tools,
                enable_tool_parser=enable_tool_parser,
                tool_call_format=tool_call_format,
                parallel_tool_calls=params.parallel_tool_calls,
                seed=params.seed,
                logit_bias=params.logit_bias,
                logprobs=params.logprobs,
                top_logprobs=params.top_logprobs,
                n=params.n,
                best_of=params.best_of,
                stream=params.stream,
                lora_adapter=params.lora_adapter,
                images=request_images,
                cancel_event=cancel_event,
            )
        finally:
            # Release the reservation on every exit path —
            # normal stream end, cancel, transport_failure, and
            # inference error. ``_stream_generate``'s own try/finally
            # already cleaned up the publisher loop and adapter
            # iterator; here we restore the budget counter. Pass
            # ``attempt_id`` so the release dedup is per-attempt and a
            # concurrent same-request_id attempt's release isn't swallowed
            # (BUG 2/7).
            await self._release_reservation(
                model_id,
                reserve_tokens,
                request_id=request_id,
                attempt_id=attempt_id,
                budget_tokens=effective_budget,
            )

    # -- Streaming core ------------------------------------------------------

    async def _stream_generate(
        self,
        *,
        msg: Any,
        adapter: GenerationAdapter,
        reply_subject: str,
        request_id: str,
        attempt_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_tokens: int | None = None,
        # ``chat_template_kwargs`` is consumed upstream in
        # :meth:`_render_chat_template`; accepted here only so the
        # caller can forward the whole ``params`` block without sniffing
        # which fields belong to which stage.
        chat_template_kwargs: dict[str, Any] | None = None,
        grammar: GrammarSpec | None = None,
        tools: tuple[dict[str, Any], ...] | None = None,
        enable_tool_parser: bool | None = None,
        tool_call_format: ToolCallFormat = "auto",
        parallel_tool_calls: bool = True,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        n: int | None = None,
        best_of: int | None = None,
        stream: bool = False,
        lora_adapter: str | None = None,
        images: list[ImageInput] | None = None,
        cancel_event: asyncio.Event,
    ) -> None:
        # ``adapter.generate`` is typed as returning ``AsyncIterator`` on
        # the protocol, but concrete adapters are async generators (use
        # ``yield``) — narrow here so we can call ``aclose()`` for
        # cancellation cleanup.
        #
        # ``grammar`` is passed via ``**kwargs`` so existing
        # GenerationAdapter implementations that don't accept the kwarg
        # still work for non-grammar requests (``grammar is None`` →
        # not forwarded). The SGLang adapter accepts the kwarg and
        # forwards the raw schema/regex to its ``/generate`` endpoint.
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        # Only forward penalties when set. Older adapters that don't
        # accept the kwarg still work for the (overwhelmingly common)
        # no-penalty path; the SGLang adapter accepts both.
        if frequency_penalty is not None:
            gen_kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            gen_kwargs["presence_penalty"] = presence_penalty
        # Non-OpenAI sampler knobs — forward only when set so older
        # adapters that don't accept the kwargs still work for the common
        # path; the SGLang adapter accepts both.
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if min_tokens is not None:
            # SGLang names the knob ``min_new_tokens`` server-side; the
            # OpenAI-compat surface (mirroring vLLM / Together) takes it
            # as ``min_tokens``. Translate at this edge.
            gen_kwargs["min_new_tokens"] = min_tokens
        # ``chat_template_kwargs`` is consumed in _render_chat_template
        # upstream; nothing to forward from here.
        _ = chat_template_kwargs
        if grammar is not None:
            gen_kwargs["grammar"] = grammar
        # Optional sampler controls: seed / logit_bias / logprobs / top_logprobs. Only
        # forward when set so older adapters that don't accept the
        # kwargs still work for the (common) no-extra-sampler path.
        if seed is not None:
            gen_kwargs["seed"] = seed
        if logit_bias:
            gen_kwargs["logit_bias"] = logit_bias
        if logprobs:
            gen_kwargs["logprobs"] = logprobs
            if top_logprobs is not None and top_logprobs > 0:
                gen_kwargs["top_logprobs"] = top_logprobs
        if n is not None and n > 1:
            gen_kwargs["n"] = n
            # Streaming n>1 fans candidates out as per-choice_index deltas;
            # non-streaming collects them into the terminal candidates[].
            gen_kwargs["stream"] = stream
        if best_of is not None and best_of > 1:
            gen_kwargs["best_of"] = best_of
        if lora_adapter:
            # SGLang selects the adapter by its registered served-name via
            # sampling_params.lora_path; no path resolution needed here.
            gen_kwargs["lora_path"] = lora_adapter
        # Vision: forward decoded images only when present so text-only
        # requests (and adapters that don't accept the kwarg) are unaffected.
        if images:
            gen_kwargs["images"] = images
        chunks_iter = cast(
            "AsyncGenerator[GenerationChunk, None]",
            adapter.generate(**gen_kwargs),
        )
        # OpenAI tools: when enabled, wrap the adapter iterator with the
        # tool-call parser so ``<tool_call>{...}</tool_call>`` blocks
        # emitted by the chat template surface as ``ToolCallDelta``s on
        # the chunk envelope. ``enable_tool_parser`` lets the caller turn
        # this off for ``tool_choice: "none"`` (tools visible, but no
        # call surfaced); when unset we default to "wrap iff tools" for
        # backward compatibility. The format is config-driven and
        # ``parallel_tool_calls`` is honoured by the parser.
        wrap_parser = enable_tool_parser if enable_tool_parser is not None else bool(tools)
        if wrap_parser:
            chunks_iter = cast(
                "AsyncGenerator[GenerationChunk, None]",
                parse_tool_call_stream(
                    chunks_iter,
                    tool_call_format=tool_call_format,
                    parallel_tool_calls=parallel_tool_calls,
                ),
            )

        # Bounded chunk queue + a single publisher task drains it. If the
        # queue fills up, the iterator-driver triggers transport_failure.
        chunk_queue: asyncio.Queue[tuple[bytes, bool] | None] = asyncio.Queue(maxsize=_CHUNK_QUEUE_MAX)
        publisher_task = asyncio.create_task(self._publisher_loop(chunk_queue, reply_subject))

        # JetStream ``in_progress()`` heartbeat — refreshes ack_wait
        # while the adapter is decoding. Bounded by the work message's
        # natural lifetime; cancelled in the ``finally`` block below.
        heartbeat_stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(msg, heartbeat_stop))

        seq = 0
        # Streaming multi-candidate (n>1 && stream): each adapter chunk is a
        # per-candidate delta tagged with choice_index, emitted individually
        # (no cross-candidate text coalescing).
        multi_candidate_stream = bool(stream) and n is not None and n > 1
        pending_text: list[str] = []
        # OpenAI per-token logprob entries accumulated alongside
        # ``pending_text`` and flushed in the same coalesced chunk. The
        # adapter aligns each entry with the tokens in ``text_delta``.
        pending_logprobs: list[dict[str, Any]] = []
        pending_count = 0
        last_flush_ts = time.monotonic()
        first_text_at: float | None = None
        publish_at = time.monotonic()
        first_yield_done = False
        terminal_sent = False
        # H6: separate flag for "the terminal we published was
        # ``transport_failure``" — even when the terminal lands on the
        # wire successfully, we MUST withhold the JetStream ACK so the
        # message is redelivered (per ADR-0001 supported-primitive
        # contract). ``terminal_sent`` stays True so the publisher loop
        # drains cleanly; the ACK gate at the bottom of run_one() reads
        # both flags.
        transport_failure_published = False
        publish_failures: list[float] = []

        def _record_failure() -> bool:
            """Append a failure; return True when threshold breached."""
            now = time.monotonic()
            publish_failures.append(now)
            cutoff = now - _PUBLISH_FAIL_WINDOW_S
            while publish_failures and publish_failures[0] < cutoff:
                publish_failures.pop(0)
            return len(publish_failures) >= _PUBLISH_FAIL_THRESHOLD

        async def _flush_pending() -> bool:
            """Enqueue any pending coalesced text. Returns False on overflow.

            Uses a bounded await (``_CHUNK_PUT_TIMEOUT_S``) so a brief
            publisher stall is absorbed cleanly; a sustained stall returns
            False and the caller must publish ``transport_failure`` +
            withhold the JetStream ACK. ``seq`` is advanced ONLY after the
            put completes successfully — a failed enqueue must not leave
            a gap in the wire sequence (the gateway rejects gaps as a
            stream error).
            """
            nonlocal seq, pending_count, last_flush_ts
            if not pending_text:
                return True
            payload = _encode_chunk(
                kind="chunk",
                request_id=request_id,
                attempt_id=attempt_id,
                seq=seq,
                text_delta="".join(pending_text),
                done=False,
                is_first=(seq == 0),
                logprobs=pending_logprobs or None,
            )
            try:
                await _wait_for(chunk_queue.put((payload, False)), timeout=_CHUNK_PUT_TIMEOUT_S)
            except (asyncio.QueueFull, TimeoutError):
                return False
            seq += 1
            pending_text.clear()
            pending_logprobs.clear()
            pending_count = 0
            last_flush_ts = time.monotonic()
            return True

        try:
            # Drive the iterator and the cancel-event together. When the
            # cancel event fires, we close the adapter iterator which
            # triggers its cleanup (e.g. SGLang /abort_request).
            chunk_aiter = chunks_iter.__aiter__()

            async def _next_chunk() -> GenerationChunk:
                # Wrap ``__anext__`` in a coroutine so the type-checker
                # accepts it as ``CoroutineLike`` for ``create_task``.
                return await chunk_aiter.__anext__()

            while True:
                next_task = asyncio.create_task(_next_chunk())
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, _pending = await asyncio.wait(
                    {next_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # Cancel wins ties: if the cancel event fired in the same
                # tick as the next chunk resolved, treat it as a cancel so
                # the caller's intent is honoured promptly.
                if cancel_event.is_set():
                    if next_task in done:
                        # Drain the result (or exception) to avoid warnings.
                        with _suppress():
                            next_task.result()
                    else:
                        next_task.cancel()
                        with _suppress():
                            await next_task
                    # Bound the time we spend in adapter teardown. The
                    # SGLang adapter's aclose handler spawns /abort_request
                    # as an independent background task (bounded separately)
                    # rather than awaiting it here, so aclose returns
                    # promptly; the cap is defence-in-depth against a
                    # misbehaving adapter blocking the publisher loop and
                    # stalling the JetStream heartbeat for the inflight
                    # window. ``TimeoutError`` is a subclass of ``Exception``
                    # so a single ``except Exception`` covers the cap.
                    try:
                        await asyncio.wait_for(chunks_iter.aclose(), timeout=2.0)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "adapter aclose timed out / errored on cancel for %s; continuing teardown",
                            request_id,
                        )
                    # Flush whatever we have, then emit a cancelled terminal.
                    # If the pending flush fails, do NOT enqueue the cancel
                    # terminal — putting the terminal ahead of unflushed text
                    # would reorder the stream the gateway sees.
                    flushed_ok = await _flush_pending()
                    if not flushed_ok:
                        _record_failure()
                        break
                    cancel_payload = _encode_chunk(
                        kind="chunk",
                        request_id=request_id,
                        attempt_id=attempt_id,
                        seq=seq,
                        text_delta="",
                        done=True,
                        finish_reason="cancelled",
                        ttft_ms=_compute_ttft_ms(publish_at, first_text_at),
                    )
                    try:
                        await _wait_for(
                            chunk_queue.put((cancel_payload, True)),
                            timeout=_CHUNK_PUT_TIMEOUT_S,
                        )
                        seq += 1
                        terminal_sent = True
                    except (asyncio.QueueFull, TimeoutError):
                        # Terminal not enqueued — leave terminal_sent False so the
                        # ACK gate at the bottom of run_one() blocks ACK and
                        # JetStream redelivers. _record_failure() bumps the
                        # transport-failure counter; we don't try to enqueue
                        # a fallback because the queue is already full.
                        _record_failure()
                    break

                cancel_task.cancel()
                with _suppress():
                    await cancel_task

                try:
                    chunk: GenerationChunk = next_task.result()
                except StopAsyncIteration:
                    # Iterator ended without a terminal chunk — synthesize one.
                    # If the pending-text flush fails, do NOT enqueue a
                    # fallback terminal on top of dropped content: that would
                    # let the gateway see a successful "stop" while the worker
                    # silently lost the trailing text. Publish a
                    # ``transport_failure`` terminal and let the ACK gate
                    # withhold the ACK (JetStream redelivers).
                    flushed_ok = await _flush_pending()
                    if not flushed_ok:
                        transport_failure_published = True
                        terminal_sent = await self._enqueue_transport_failure(chunk_queue, request_id, attempt_id, seq)
                        break
                    fallback = _encode_chunk(
                        kind="chunk",
                        request_id=request_id,
                        attempt_id=attempt_id,
                        seq=seq,
                        text_delta="",
                        done=True,
                        finish_reason="stop",
                        ttft_ms=_compute_ttft_ms(publish_at, first_text_at),
                    )
                    try:
                        await _wait_for(
                            chunk_queue.put((fallback, True)),
                            timeout=_CHUNK_PUT_TIMEOUT_S,
                        )
                        terminal_sent = True
                    except (asyncio.QueueFull, TimeoutError):
                        _record_failure()
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Generate iterator raised for %s/%s", request_id, attempt_id, exc_info=True)
                    err_payload = _encode_chunk(
                        kind="chunk",
                        request_id=request_id,
                        attempt_id=attempt_id,
                        seq=seq,
                        text_delta="",
                        done=True,
                        finish_reason="error",
                        error_code="inference_error",
                        error_message=str(exc),
                    )
                    try:
                        await _wait_for(
                            chunk_queue.put((err_payload, True)),
                            timeout=_CHUNK_PUT_TIMEOUT_S,
                        )
                        terminal_sent = True
                    except (asyncio.QueueFull, TimeoutError):
                        _record_failure()
                    break

                # Record first-text timing for TTFT.
                if chunk.text_delta and first_text_at is None:
                    first_text_at = time.monotonic()
                if chunk.text_delta and not first_yield_done:
                    first_yield_done = True

                # Streaming multi-candidate: emit each per-candidate delta as its
                # own wire chunk tagged with choice_index (no cross-candidate
                # coalescing). Per-choice ``finish_reason`` / ``logprobs`` /
                # ``tool_call_delta`` all surface on these chunks; the final
                # ``done=True`` terminal flows through the done path below and
                # carries aggregate usage. Each choice is independent — the
                # gateway tracks per-``choice_index`` role-emitted / finish
                # state when it forwards these.
                if multi_candidate_stream and not chunk.done:
                    payload = _encode_chunk(
                        kind="chunk",
                        request_id=request_id,
                        attempt_id=attempt_id,
                        seq=seq,
                        text_delta=chunk.text_delta,
                        done=False,
                        finish_reason=chunk.finish_reason,
                        choice_index=chunk.choice_index,
                        logprobs=list(chunk.logprobs) if chunk.logprobs else None,
                        tool_call_delta=chunk.tool_call_delta,
                    )
                    # H6: seq advances ONLY after the put succeeds. Per-choice
                    # deltas are required content chunks: a dropped delta
                    # would shorten the candidate's output without surfacing
                    # an error. On enqueue failure we publish a
                    # ``transport_failure`` terminal at the *current* seq
                    # (contiguous with the last accepted chunk) and withhold
                    # the ACK so JetStream redelivers.
                    try:
                        await _wait_for(
                            chunk_queue.put((payload, False)),
                            timeout=_CHUNK_PUT_TIMEOUT_S,
                        )
                    except (asyncio.QueueFull, TimeoutError):
                        _record_failure()
                        transport_failure_published = True
                        terminal_sent = await self._enqueue_transport_failure(chunk_queue, request_id, attempt_id, seq)
                        break
                    seq += 1
                    continue

                if chunk.done:
                    # Flush any coalesced text before the terminal chunk.
                    if chunk.text_delta:
                        pending_text.append(chunk.text_delta)
                        pending_count += max(1, len(chunk.text_delta))
                    if chunk.logprobs:
                        pending_logprobs.extend(chunk.logprobs)
                    # H6: if the pending-text flush failed, we MUST NOT
                    # build a normal ``stop`` terminal on top of the
                    # dropped content — that would surface a successful
                    # completion to the gateway while pending text was
                    # silently lost. Publish a ``transport_failure``
                    # terminal at the unchanged seq and withhold the ACK.
                    flushed_ok = await _flush_pending()
                    if not flushed_ok:
                        _record_failure()
                        transport_failure_published = True
                        terminal_sent = await self._enqueue_transport_failure(chunk_queue, request_id, attempt_id, seq)
                        break
                    # Tool-call parser may surface a terminal ``error``
                    # chunk with ``error_code`` / ``error_message``
                    # populated (e.g. malformed ``<tool_call>`` JSON);
                    # propagate those to the wire envelope so the
                    # gateway can surface the parse failure as a normal
                    # error chunk instead of swallowing it under the
                    # default ``stop`` reason.
                    terminal = _encode_chunk(
                        kind="chunk",
                        request_id=request_id,
                        attempt_id=attempt_id,
                        seq=seq,
                        text_delta="",
                        done=True,
                        finish_reason=chunk.finish_reason or "stop",
                        prompt_tokens=chunk.prompt_tokens,
                        completion_tokens=chunk.completion_tokens,
                        ttft_ms=_compute_ttft_ms(publish_at, first_text_at),
                        error_code=chunk.error_code,
                        error_message=chunk.error_message,
                        candidates=list(chunk.candidates) if chunk.candidates else None,
                    )
                    try:
                        await _wait_for(
                            chunk_queue.put((terminal, True)),
                            timeout=_CHUNK_PUT_TIMEOUT_S,
                        )
                        terminal_sent = True
                    except (asyncio.QueueFull, TimeoutError):
                        _record_failure()
                        transport_failure_published = True
                        terminal_sent = await self._enqueue_transport_failure(chunk_queue, request_id, attempt_id, seq)
                    break

                # Tool-call delta: flush any pending text first, then
                # publish a dedicated chunk carrying the delta. Mixing
                # ``text_delta`` and ``tool_calls`` in a single envelope
                # is legal in the OpenAI streaming format but the SSE
                # encoder is simpler when each chunk has exactly one of
                # them populated.
                if chunk.tool_call_delta is not None:
                    flushed_ok = await _flush_pending()
                    if not flushed_ok:
                        # Pending text dropped — do NOT enqueue the
                        # tool-call delta on top of the gap. Publish
                        # ``transport_failure`` and withhold ACK.
                        _record_failure()
                        transport_failure_published = True
                        terminal_sent = await self._enqueue_transport_failure(chunk_queue, request_id, attempt_id, seq)
                        break
                    tc_payload = _encode_chunk(
                        kind="chunk",
                        request_id=request_id,
                        attempt_id=attempt_id,
                        seq=seq,
                        text_delta="",
                        done=False,
                        is_first=(seq == 0),
                        tool_call_delta=chunk.tool_call_delta,
                    )
                    try:
                        await _wait_for(
                            chunk_queue.put((tc_payload, False)),
                            timeout=_CHUNK_PUT_TIMEOUT_S,
                        )
                        seq += 1
                    except (asyncio.QueueFull, TimeoutError):
                        _record_failure()
                        transport_failure_published = True
                        terminal_sent = await self._enqueue_transport_failure(chunk_queue, request_id, attempt_id, seq)
                        break
                    continue

                # Non-terminal delta: accumulate, flush on time/count.
                if chunk.text_delta:
                    pending_text.append(chunk.text_delta)
                    pending_count += max(1, len(chunk.text_delta))
                if chunk.logprobs:
                    pending_logprobs.extend(chunk.logprobs)

                now = time.monotonic()
                if pending_count >= _FLUSH_MAX_TOKENS or (pending_text and (now - last_flush_ts) >= _FLUSH_INTERVAL_S):
                    ok = await _flush_pending()
                    if not ok:
                        # Flush failed after a bounded await on a non-terminal
                        # chunk — content was dropped. Publish
                        # ``transport_failure`` immediately rather than
                        # waiting for the failure-threshold to trip; with
                        # the bounded-await semantics, a single
                        # ``flush_pending`` False already represents a
                        # ~100ms publisher stall and is sufficient to
                        # invoke the no-silent-drop guarantee.
                        _record_failure()
                        transport_failure_published = True
                        terminal_sent = await self._enqueue_transport_failure(chunk_queue, request_id, attempt_id, seq)
                        break
        finally:
            # Stop the heartbeat before draining the publisher so a
            # late in_progress() call doesn't race the final ACK.
            heartbeat_stop.set()
            with _suppress():
                await asyncio.wait_for(heartbeat_task, timeout=1.0)
            if not heartbeat_task.done():
                heartbeat_task.cancel()
                with _suppress():
                    await heartbeat_task

            # Signal the publisher loop to drain and exit.
            try:
                chunk_queue.put_nowait(None)
            except asyncio.QueueFull:
                # Queue is full. NEVER drop a queued payload to make room —
                # discarding the terminal chunk loses the ACK (full-generation
                # redelivery) and discarding a mid-stream chunk opens a silent
                # ``seq`` gap that still ACKs (corrupted output). Instead wait,
                # bounded, for the concurrently-draining publisher to free a
                # slot. Only give up if the publisher already exited (then
                # nothing will drain and the sentinel is moot anyway).
                if not publisher_task.done():
                    with _suppress():
                        await asyncio.wait_for(chunk_queue.put(None), timeout=5.0)
            try:
                publisher_ok = await asyncio.wait_for(publisher_task, timeout=5.0)
            except TimeoutError:
                publisher_task.cancel()
                with _suppress():
                    await publisher_task
                publisher_ok = False

        # ACK after terminal publish (durable guarantee per §4.4 ACK timing).
        # H6: withhold ACK when the terminal we published was a
        # ``transport_failure`` — required content was dropped on the
        # backpressure path and JetStream redelivery is the recovery
        # mechanism (per ADR-0001 supported-primitive contract). The
        # gateway forwards the transport_failure terminal to the client
        # so the surface error is still visible; redelivery gives the
        # request a second chance on a healthy worker.
        if terminal_sent and publisher_ok and not transport_failure_published:
            await _safe_ack(msg)
        elif transport_failure_published:
            logger.warning(
                "Generate stream for %s/%s ended with transport_failure terminal — not ACKing (JetStream will redeliver)",
                request_id,
                attempt_id,
            )
        else:
            # No terminal was published — let JetStream redeliver.
            logger.warning(
                "Generate stream for %s/%s ended without confirmed terminal publish — not ACKing",
                request_id,
                attempt_id,
            )

    async def _heartbeat_loop(self, msg: Any, stop: asyncio.Event) -> None:
        """Refresh JetStream ``ack_wait`` while a generation streams.

        The pool consumer's ``ack_wait`` is 5 min for generation pools
        (§4.4). For a genuinely slow decode (long prompt, low TPS,
        grammar compile) we issue ``msg.in_progress()`` well before
        that window elapses so the message is not redelivered. The
        method is a no-op if the underlying NATS msg doesn't support
        ``in_progress`` (e.g. test mocks).
        """
        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=_INPROGRESS_INTERVAL_S)
                    return
                except TimeoutError:
                    pass
                fn = getattr(msg, "in_progress", None)
                if fn is None:
                    return
                try:
                    result = fn()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: BLE001
                    logger.debug("in_progress() raised", exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("heartbeat loop error", exc_info=True)

    async def _publisher_loop(self, chunk_queue: asyncio.Queue[tuple[bytes, bool] | None], reply_subject: str) -> bool:
        """Drain ``chunk_queue`` into ``self._nc.publish``.

        Returns ``True`` only if all queued chunks, including the terminal
        chunk, were successfully published. Any publish failure returns
        ``False`` so the caller does **not** ACK the JetStream work item;
        redelivery is safer than losing a completed generation result.
        """
        saw_terminal = False
        while True:
            item = await chunk_queue.get()
            if item is None:
                return saw_terminal
            payload, is_terminal = item
            try:
                if is_terminal:
                    await self._publish_terminal_envelope(reply_subject, payload)
                else:
                    # Nonterminal chunks are the high-volume path. Keep their
                    # disabled-mode cost identical to the pre-telemetry path.
                    await self._nc.publish(reply_subject, payload)
                saw_terminal = saw_terminal or is_terminal
            except Exception:  # noqa: BLE001
                logger.warning("Failed to publish generation chunk on %s", reply_subject, exc_info=True)
                return False

    async def _publish_terminal_envelope(self, reply_subject: str, payload: bytes) -> None:
        """Publish a terminal and observe completion only after transport success."""
        await self._nc.publish(reply_subject, payload)
        # Empty in disabled mode, so no msgpack decode or attribute work lands
        # on the default serving path.
        if self._completion_by_attempt:
            self._record_published_terminal(payload)

    def _record_published_terminal(self, payload: bytes) -> None:
        """Emit exactly one shared worker completion for a published terminal.

        Generation-specific TTFT/TPOT/token observations remain owned by the
        adapter stream timer. This is the common engine-completion seam for
        ``sie.worker.requests`` and ``sie.worker.request.duration``: no metric is
        emitted for an unconfirmed terminal publish or a NAK that another worker
        may retry.
        """
        try:
            envelope = msgpack.unpackb(payload, raw=False)
        except (ValueError, msgpack.exceptions.UnpackException):
            logger.warning("could not decode a published generation terminal for telemetry")
            return
        if not isinstance(envelope, dict) or envelope.get("kind") != "chunk" or envelope.get("done") is not True:
            return
        attempt_id = str(envelope.get("attempt_id") or "")
        completion = self._completion_by_attempt.get(attempt_id)
        if completion is None or completion.completed:
            return
        completion.completed = True
        finish_reason = str(envelope.get("finish_reason") or "stop")
        if finish_reason == "cancelled":
            outcome = "cancelled"
        elif finish_reason == "error" or envelope.get("error") is not None:
            outcome = "error"
        else:
            outcome = "success"
        _metrics.worker_telemetry().item_completed(
            operation="generate",
            outcome=outcome,
            model=completion.model_id,
            profile=completion.profile_id,
            duration_s=time.perf_counter() - completion.started_at,
        )

    async def _enqueue_transport_failure(
        self,
        chunk_queue: asyncio.Queue[tuple[bytes, bool] | None],
        request_id: str,
        attempt_id: str,
        seq: int,
    ) -> bool:
        """Publish a ``transport_failure`` terminal at ``seq``.

        Returns True iff the terminal was successfully enqueued for the
        publisher; the caller MUST treat False as "no terminal published"
        and withhold the JetStream ACK so the message is redelivered.

        Uses a bounded await so a brief publisher stall after the queue
        filled does not lock out the failure terminal forever — but the
        timeout is intentionally short: this path is reached only after
        an earlier enqueue already gave up at the same window, so the
        publisher is almost certainly stalled. Either it drained a slot
        in the window (terminal published, no ACK because the dropped
        content was the prior chunk) or the queue is still wedged
        (terminal_sent stays False, no ACK).
        """
        payload = _encode_chunk(
            kind="chunk",
            request_id=request_id,
            attempt_id=attempt_id,
            seq=seq,
            text_delta="",
            done=True,
            finish_reason="error",
            error_code="transport_failure",
            error_message="sustained chunk publish failure or queue overflow",
        )
        try:
            await _wait_for(
                chunk_queue.put((payload, True)),
                timeout=_TRANSPORT_FAILURE_PUT_TIMEOUT_S,
            )
        except (asyncio.QueueFull, TimeoutError):
            return False
        return True

    # -- Helpers -------------------------------------------------------------

    async def _ensure_loaded(self, model_id: str) -> Any:
        if self._registry.is_loaded(model_id):
            self._registry.touch_lru(model_id)
            return self._registry.get(model_id)
        device = self._registry.device
        return await self._registry.load_async(model_id, device)

    @staticmethod
    def _extract_generate_params(wi: WorkItem) -> dict[str, Any] | None:
        params = wi.get("generate")  # type: ignore[call-overload]
        if isinstance(params, dict):
            return params
        options = wi.get("options")
        if isinstance(options, dict) and ("prompt" in options or "messages" in options):
            return options
        return None

    @classmethod
    def _validate_generate_params(
        cls,
        wi: WorkItem,
        config: ModelConfig | None = None,
    ) -> _GenerateRequestParams | _ValidationError:
        """Decode and validate the ``generate`` payload.

        Accepts two mutually-exclusive input shapes:

        - ``{prompt: str, ...}`` — prompt wire shape.
        - ``{messages: [{role, content}, ...], ...}`` — chat wire shape.

        Returns either a validated :class:`_GenerateRequestParams` or a
        :class:`_ValidationError` that the caller turns into a terminal
        ``code: "invalid_request"`` chunk.
        """
        params = cls._extract_generate_params(wi)
        if params is None:
            return _ValidationError(
                code="invalid_request",
                message=(
                    "Generate work item missing 'generate' params "
                    "(prompt + max_new_tokens or messages + max_new_tokens required)"
                ),
            )

        options = wi.get("options")
        if options is not None and not isinstance(options, dict):
            return _ValidationError(code="invalid_request", message="'options' must be an object")
        if config is not None:
            try:
                params = apply_generation_runtime_options(config, options, params)
            except ValueError as exc:
                return _ValidationError(code="invalid_request", message=str(exc))

        max_new_tokens = params.get("max_new_tokens")
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            return _ValidationError(
                code="invalid_request",
                message="'max_new_tokens' must be a positive integer",
            )

        has_prompt = "prompt" in params
        has_messages = "messages" in params
        if has_prompt and has_messages:
            return _ValidationError(
                code="invalid_request",
                message="'prompt' and 'messages' are mutually exclusive on a generate work item",
            )

        input_value: _PromptInput | _MessagesInput
        if has_messages:
            messages_raw = params.get("messages")
            if not isinstance(messages_raw, list) or not messages_raw:
                return _ValidationError(
                    code="invalid_request",
                    message="'messages' must be a non-empty array",
                )
            decoded: list[_ChatMessage] = []
            total_images = 0
            for idx, item in enumerate(messages_raw):
                if not isinstance(item, dict):
                    return _ValidationError(
                        code="invalid_request",
                        message=f"messages[{idx}] must be an object",
                    )
                item_dict = cast("dict[str, Any]", item)
                role = item_dict.get("role")
                content = item_dict.get("content")
                tool_calls_raw = item_dict.get("tool_calls")
                tool_call_id = item_dict.get("tool_call_id")
                if not isinstance(role, str) or role not in _ALLOWED_CHAT_ROLES:
                    return _ValidationError(
                        code="invalid_request",
                        message=(f"messages[{idx}].role must be one of {sorted(_ALLOWED_CHAT_ROLES)!r}, got {role!r}"),
                    )
                tool_calls = (
                    tuple(tool_calls_raw) if isinstance(tool_calls_raw, (list, tuple)) and tool_calls_raw else None
                )
                # Content is required except on an assistant message that
                # only carries tool_calls (OpenAI sends content:null there).
                if content is None and role == "assistant" and tool_calls is not None:
                    content = ""
                # Parse the string-or-parts content. Returns concatenated text
                # plus any inline-decoded images (empty for the common string
                # shape and for the gateway path, which sends text here and the
                # images in the message-level ``images`` field below).
                parsed = _parse_chat_content(content, idx)
                if isinstance(parsed, _ValidationError):
                    return parsed
                content_str, content_images, content_layout = parsed
                # Gateway-decoded images ride a message-level ``images`` field
                # (bytes already extracted from the data URI at the edge). Merge
                # with any inline content-part images; in practice exactly one
                # source is populated (gateway path vs direct OpenAI caller).
                field_images = _parse_message_images_field(item_dict.get("images"), idx)
                if isinstance(field_images, _ValidationError):
                    return field_images
                message_images = (*content_images, *field_images)
                total_images += len(message_images)
                if total_images > _MAX_IMAGES_PER_REQUEST:
                    return _ValidationError(
                        code="invalid_request",
                        message=(f"too many images ({total_images}); maximum is {_MAX_IMAGES_PER_REQUEST} per request"),
                    )
                # Ordered layout (#1294): gateway forwards it as a message-level
                # ``content_parts`` field; the direct OpenAI caller's layout comes
                # back from ``_parse_chat_content``. Exactly one is populated.
                field_parts = _parse_content_parts_field(item_dict.get("content_parts"), idx)
                if isinstance(field_parts, _ValidationError):
                    return field_parts
                # Prefer the gateway-forwarded layout only when it actually
                # carries an image marker — an empty or image-free ``content_parts``
                # field must not shadow a direct-path ``content_layout``. Falling
                # back to ``content_layout`` (which is None unless it has images)
                # also keeps an empty tuple off the interleaved render path, so the
                # render guard (``if m.content_parts``) and the reconcile guard
                # below agree.
                field_has_image = field_parts is not None and any(p.get("type") == "image" for p in field_parts)
                normalized_field_parts = field_parts if field_has_image else None
                # Exactly one image-bearing layout source is allowed. An image-bearing
                # ``content_parts`` field alongside an image-bearing ``content`` array
                # (``content_layout``) would let the field's layout override the layout
                # that ``content`` — and thus the validated ``content_str`` — was parsed
                # from, so the checked text and the rendered layout could diverge. Reject
                # rather than silently honoring one over the other.
                if normalized_field_parts is not None and content_layout is not None:
                    return _ValidationError(
                        code="invalid_request",
                        message=(
                            f"messages[{idx}] has both an image-bearing 'content' array and an "
                            "image-bearing 'content_parts' field; exactly one layout source is allowed"
                        ),
                    )
                message_content_parts = normalized_field_parts or content_layout
                # Reconcile placeholder count with the flat image list — a
                # mismatch would desync placeholders from ``image_data`` at the
                # engine, so reject it here rather than mis-render downstream.
                if message_content_parts is not None:
                    image_markers = sum(1 for p in message_content_parts if p.get("type") == "image")
                    if image_markers != len(message_images):
                        return _ValidationError(
                            code="invalid_request",
                            message=(
                                f"messages[{idx}].content_parts has {image_markers} image placeholder(s) "
                                f"but {len(message_images)} image(s) were provided"
                            ),
                        )
                decoded.append(
                    _ChatMessage(
                        role=role,
                        content=content_str,
                        tool_calls=tool_calls,
                        tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
                        images=message_images or None,
                        content_parts=message_content_parts,
                    )
                )
            input_value = _MessagesInput(messages=tuple(decoded))
        else:
            prompt = params.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                return _ValidationError(
                    code="invalid_request",
                    message="'prompt' must be a non-empty string",
                )
            input_value = _PromptInput(prompt=prompt)

        # Repeat strict gateway checks for queue-bypass callers and profile defaults.
        # Omitted values have already resolved from model runtime or fall back to 1.0.
        temperature_raw = params.get("temperature", 1.0)
        top_p_raw = params.get("top_p", 1.0)
        if isinstance(temperature_raw, bool) or not isinstance(temperature_raw, int | float):
            return _ValidationError(
                code="invalid_request",
                message="'temperature' must be a finite number >= 0",
            )
        temperature = float(temperature_raw)
        if not math.isfinite(temperature) or temperature < 0:
            return _ValidationError(
                code="invalid_request",
                message="'temperature' must be a finite number >= 0",
            )
        if isinstance(top_p_raw, bool) or not isinstance(top_p_raw, int | float):
            return _ValidationError(code="invalid_request", message="'top_p' must be a number in (0, 1]")
        top_p = float(top_p_raw)
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            return _ValidationError(code="invalid_request", message="'top_p' must be a number in (0, 1]")
        stop_raw = params.get("stop")
        stop: list[str] | None
        if isinstance(stop_raw, list) and all(isinstance(s, str) for s in stop_raw):
            stop = [s for s in stop_raw if isinstance(s, str)] or None
        elif stop_raw is not None:
            return _ValidationError(code="invalid_request", message="'stop' must be an array of strings")
        else:
            stop = None

        # Optional ``grammar`` field. The gateway is the
        # authority for shape/safety; the worker only deserialises and
        # surfaces a malformed-payload error for anything that doesn't
        # match the wire contract (caller bug, not a user error).
        grammar_raw = params.get("grammar")
        grammar: GrammarSpec | None
        if grammar_raw is None:
            grammar = None
        elif isinstance(grammar_raw, dict):
            kind_raw = grammar_raw.get("kind")
            value = grammar_raw.get("value")
            if kind_raw not in ("json_schema", "regex", "ebnf"):
                return _ValidationError(
                    code="invalid_request",
                    message=(f"'grammar.kind' must be 'json_schema', 'regex' or 'ebnf', got {kind_raw!r}"),
                )
            if kind_raw == "json_schema" and not isinstance(value, dict):
                return _ValidationError(
                    code="invalid_request",
                    message="'grammar.value' must be an object when kind=='json_schema'",
                )
            if kind_raw == "regex" and not isinstance(value, str):
                return _ValidationError(
                    code="invalid_request",
                    message="'grammar.value' must be a string when kind=='regex'",
                )
            if kind_raw == "ebnf" and not isinstance(value, str):
                return _ValidationError(
                    code="invalid_request",
                    message="'grammar.value' must be a string when kind=='ebnf'",
                )
            label_raw = grammar_raw.get("label")
            strict_raw = grammar_raw.get("strict")
            grammar = GrammarSpec(
                kind=kind_raw,
                value=value,
                label=label_raw if isinstance(label_raw, str) else None,
                strict=strict_raw if isinstance(strict_raw, bool) else None,
            )
        else:
            return _ValidationError(
                code="invalid_request",
                message="'grammar' must be an object",
            )

        # OpenAI penalty knobs. The gateway already validated
        # ``[-2.0, 2.0]``; the worker only coerces and clamps
        # defensively so a malformed mocked request in a test cannot
        # poison the sampler. Missing / null → ``None`` (adapter
        # default).
        def _coerce_penalty(name: str) -> float | None:
            raw = params.get(name)
            if raw is None:
                return None
            try:
                v = float(raw)
            except (TypeError, ValueError):
                logger.debug(
                    "generate request: non-numeric %s %r — dropping",
                    name,
                    raw,
                )
                return None
            if math.isnan(v) or v < -2.0 or v > 2.0:  # noqa: PLR2004 — OpenAI penalty range is the spec value
                logger.debug(
                    "generate request: %s %r out of [-2.0, 2.0] — dropping",
                    name,
                    raw,
                )
                return None
            return v

        frequency_penalty = _coerce_penalty("frequency_penalty")
        presence_penalty = _coerce_penalty("presence_penalty")

        # Non-OpenAI sampling knobs. Gateway is the authority for shape /
        # range (top_k int >= 1; repetition_penalty float in (0, 2]); the
        # worker only coerces defensively so a malformed mocked request
        # cannot poison the sampler. Bad / missing → ``None`` (default).
        top_k_raw = params.get("top_k")
        top_k: int | None
        if top_k_raw is None or isinstance(top_k_raw, bool):
            top_k = None
        else:
            try:
                top_k_val = int(top_k_raw)
            except (TypeError, ValueError):
                logger.debug("generate request: non-integer top_k %r — dropping", top_k_raw)
                top_k = None
            else:
                top_k = top_k_val if top_k_val >= 1 else None

        rep_pen_raw = params.get("repetition_penalty")
        repetition_penalty: float | None
        if rep_pen_raw is None:
            repetition_penalty = None
        else:
            try:
                rep_pen_val = float(rep_pen_raw)
            except (TypeError, ValueError):
                logger.debug(
                    "generate request: non-numeric repetition_penalty %r — dropping",
                    rep_pen_raw,
                )
                repetition_penalty = None
            else:
                # Range (0, 2] is the documented gateway contract;
                # NaN is rejected via ``math.isnan``.
                repetition_penalty = (
                    rep_pen_val if (not math.isnan(rep_pen_val) and 0.0 < rep_pen_val <= 2.0) else None  # noqa: PLR2004
                )

        # SGLang ``sampling_params["min_new_tokens"]``. Gateway validates
        # as ``>= 0``; defensive coerce here.
        min_tokens_raw = params.get("min_tokens")
        min_tokens: int | None
        if min_tokens_raw is None or isinstance(min_tokens_raw, bool):
            min_tokens = None
        else:
            try:
                min_tokens_val = int(min_tokens_raw)
            except (TypeError, ValueError):
                logger.debug("generate request: non-integer min_tokens %r — dropping", min_tokens_raw)
                min_tokens = None
            else:
                min_tokens = min_tokens_val if min_tokens_val >= 0 else None

        # Per-request chat-template kwargs. Gateway has already validated
        # this is an object (or absent); store as-is and merge with the
        # model YAML defaults in ``_render_chat_template`` below.
        chat_template_kwargs_raw = params.get("chat_template_kwargs")
        chat_template_kwargs: dict[str, Any] | None
        if isinstance(chat_template_kwargs_raw, dict):
            chat_template_kwargs = dict(chat_template_kwargs_raw)
        else:
            chat_template_kwargs = None

        # OpenAI ``seed`` / ``logit_bias`` / ``logprobs`` / ``top_logprobs``
        # round-trips. Gateway is the authority for shape validation
        # (signed i64 / range / cap); the worker only coerces defensively.
        seed_raw = params.get("seed")
        seed: int | None
        if seed_raw is None:
            seed = None
        elif isinstance(seed_raw, int) and not isinstance(seed_raw, bool):
            if -(1 << 63) <= seed_raw <= (1 << 63) - 1:
                seed = seed_raw
            else:
                logger.debug("generate request: invalid signed i64 seed %r — dropping", seed_raw)
                seed = None
        else:
            logger.debug("generate request: invalid signed i64 seed %r — dropping", seed_raw)
            seed = None

        logit_bias_raw = params.get("logit_bias")
        logit_bias: dict[str, float] | None
        if logit_bias_raw is None:
            logit_bias = None
        elif isinstance(logit_bias_raw, dict):
            cleaned: dict[str, float] = {}
            for k, v in logit_bias_raw.items():
                if not isinstance(k, str):
                    continue
                try:
                    cleaned[k] = float(v)
                except (TypeError, ValueError):
                    continue
            logit_bias = cleaned or None
        else:
            logit_bias = None

        logprobs_raw = params.get("logprobs")
        logprobs = bool(logprobs_raw) if logprobs_raw is not None else False
        top_logprobs_raw = params.get("top_logprobs")
        top_logprobs: int | None
        if top_logprobs_raw is None:
            top_logprobs = None
        else:
            try:
                top_logprobs = int(top_logprobs_raw)
            except (TypeError, ValueError):
                top_logprobs = None

        # OpenAI ``n`` — multi-candidate count. The gateway has already
        # validated the range (and the ``best_of && stream`` reject); the
        # worker only coerces and forwards (``> 1`` triggers the adapter
        # fan-out, streaming or non-streaming).
        n_raw = params.get("n")
        n: int | None
        if n_raw is None:
            n = None
        else:
            try:
                n = int(n_raw)
            except (TypeError, ValueError):
                n = None

        # OpenAI ``best_of`` — gateway-validated (best_of >= n, non-streaming);
        # the worker coerces and forwards (> 1 triggers over-generate + rank).
        best_of_raw = params.get("best_of")
        best_of: int | None
        if best_of_raw is None:
            best_of = None
        else:
            try:
                best_of = int(best_of_raw)
            except (TypeError, ValueError):
                best_of = None

        # Streaming flag — only consulted for n>1 (per-candidate fan-out).
        stream = bool(params.get("stream", False))

        # Multi-LoRA — served-name of the adapter (gateway-validated shape).
        lora_adapter_raw = params.get("lora_adapter")
        lora_adapter = lora_adapter_raw if isinstance(lora_adapter_raw, str) and lora_adapter_raw else None

        # OpenAI ``tools`` / ``tool_choice`` / ``parallel_tool_calls``.
        # The gateway is the authority for shape validation (JSON-schema
        # safety caps on each ``function.parameters``, structural shape
        # of ``tool_choice``); the worker only checks that the wire-form
        # is the expected dict-of-strings tree before forwarding to
        # :func:`parse_tool_call_stream`.
        tools_raw = params.get("tools")
        tools: tuple[dict[str, Any], ...] | None
        if tools_raw is None:
            tools = None
        elif isinstance(tools_raw, list) and tools_raw and all(isinstance(t, dict) for t in tools_raw):
            tools = tuple(cast("list[dict[str, Any]]", tools_raw))
        else:
            return _ValidationError(
                code="invalid_request",
                message="'tools' must be a non-empty array of objects",
            )
        tool_choice_raw = params.get("tool_choice")
        tool_choice: dict[str, Any] | str | None
        if tool_choice_raw is None:
            tool_choice = None
        elif isinstance(tool_choice_raw, str | dict):
            tool_choice = tool_choice_raw  # type: ignore[assignment]
        else:
            return _ValidationError(
                code="invalid_request",
                message="'tool_choice' must be a string or object",
            )
        parallel_raw = params.get("parallel_tool_calls", True)
        parallel_tool_calls = bool(parallel_raw) if parallel_raw is not None else True

        return _GenerateRequestParams(
            input=input_value,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            min_tokens=min_tokens,
            chat_template_kwargs=chat_template_kwargs,
            grammar=grammar,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            seed=seed,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            n=n,
            best_of=best_of,
            stream=stream,
            lora_adapter=lora_adapter,
        )

    async def _get_tokenizer(self, model_id: str) -> TokenizerLike:
        """Load (or return cached) tokenizer for ``model_id``.

        Both ``AutoTokenizer.from_pretrained`` and ``apply_chat_template``
        block, so we run the actual load in a worker thread. The lock is
        ``asyncio.Lock`` rather than ``threading.Lock`` because all
        callers are coroutines; concurrent first-request races on the
        same model coalesce to a single load.
        """
        cached = self._tokenizers.get(model_id)
        if cached is not None:
            return cached
        async with self._tokenizer_lock:
            cached = self._tokenizers.get(model_id)
            if cached is not None:
                return cached
            config = self._registry.get_config(model_id)
            source = config.hf_id or config.weights_path
            # ``hf_id`` is typed ``str | None`` and ``weights_path`` is
            # ``Path | None``; require a str-or-Path to defend against
            # test fixtures that return a MagicMock for unconfigured
            # attributes.
            if not isinstance(source, str | Path):
                msg = f"model '{model_id}' has no hf_id or weights_path; cannot load tokenizer"
                raise RuntimeError(msg)
            # Pin the tokenizer load to the model's immutable ``hf_revision``
            # when loading from an ``hf_id`` — mirrors how the weights loader
            # pins the revision (``core/loader.py``). Under ``HF_HUB_OFFLINE=1``
            # the offline HF cache resolves the exact ``snapshots/<sha>`` dir
            # directly and never touches the ``refs/main`` alias (absent for
            # repos staged by pinned SHA), which otherwise fails the
            # chat-template render (#1801). A local ``weights_path`` has no HF
            # revision, so it stays unpinned.
            revision = config.hf_revision if config.hf_id else None
            tok = await asyncio.to_thread(
                load_tokenizer,
                source,
                trust_remote_code=True,
                revision=revision,
            )
            self._tokenizers[model_id] = tok
            return tok

    @staticmethod
    def _normalise_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
        """Coerce an OpenAI tool_call into the shape Qwen's chat template wants.

        OpenAI sends ``function.arguments`` as a JSON *string*; the Qwen
        template iterates it as a mapping (``.items()``) and errors with
        "Can only get item pairs from a mapping" on a string. Parse the
        arguments to a dict here, leaving everything else untouched.
        """
        out = dict(tc)
        fn = out.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args) if args.strip() else {}
                except (json.JSONDecodeError, ValueError):
                    fn["arguments"] = {}
            out["function"] = fn
        return out

    async def _render_chat_template(
        self,
        model_id: str,
        messages: tuple[_ChatMessage, ...],
        tools: tuple[dict[str, Any], ...] | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> str | _ValidationError:
        """Render a chat-message list into a prompt string via the tokenizer's
        chat template. Returns the rendered string or a validation error
        that the caller surfaces as a terminal chunk.

        When ``tools`` is provided it is passed to ``apply_chat_template`` so
        the tool definitions are rendered into the prompt — without this the
        model never sees the available tools and never emits ``<tool_call>``
        blocks, so native tool calling silently degrades to prose.

        ``request_kwargs`` are per-request overrides for the tokenizer's
        template kwargs (sourced from the client's
        ``chat_template_kwargs`` body field). Merged on top of the model
        YAML's ``chat_template_kwargs`` so the YAML stays the default
        while clients can flip individual switches per request (e.g.
        ``{"enable_thinking": False}`` to suppress Qwen3's ``<think>``
        reasoning on a one-off basis). The YAML wins on conflict —
        operator config is authoritative; per-request overrides only
        fill in keys the YAML didn't set.
        """
        try:
            tok = await self._get_tokenizer(model_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load tokenizer for chat-template rendering on %s",
                model_id,
                exc_info=True,
            )
            return _ValidationError(
                code="invalid_request",
                message=f"failed to load tokenizer for chat-template rendering: {exc}",
            )

        # Tokenizer-extension kwargs (e.g. Qwen3's ``enable_thinking``).
        # Start from any per-request overrides, then layer the model
        # YAML on top — YAML defaults win on conflict so an operator
        # can fence off a kwarg the client shouldn't be flipping just
        # by setting it in the YAML. Empty/missing client kwargs are
        # a no-op.
        kwargs: dict[str, Any] = {}
        if request_kwargs:
            kwargs.update(request_kwargs)
        try:
            config = self._registry.get_config(model_id)
        except KeyError:
            config = None  # type: ignore[assignment]
        if config is not None and config.tasks.generate is not None:
            raw_kwargs = config.tasks.generate.chat_template_kwargs
            if isinstance(raw_kwargs, dict):
                kwargs.update(raw_kwargs)

        # Reject pathologically large message lists before the CPU-bound
        # render: ``apply_chat_template`` is input-sized and runs on the
        # bounded ``_GRAMMAR_EXECUTOR`` (see below), so an unbounded list
        # could pin a pool thread and starve concurrent requests.
        if len(messages) > _MAX_CHAT_MESSAGES:
            return _ValidationError(
                code="invalid_request",
                message=(f"too many messages ({len(messages)}); maximum is {_MAX_CHAT_MESSAGES}"),
            )

        # Build the message dicts, carrying tool_calls / tool_call_id so a
        # multi-turn tool exchange (assistant requests a tool → tool result
        # → final answer) renders correctly. Qwen's template reads
        # ``tool_calls`` on assistant messages and ``role:"tool"`` results.
        message_dicts: list[dict[str, Any]] = []
        for m in messages:
            # Fold ``developer`` → ``system`` (the gateway normally does this;
            # defensive for direct worker callers). Qwen's template has no
            # ``developer`` slot.
            role = "system" if m.role == "developer" else m.role
            d: dict[str, Any]
            if m.content_parts:
                # #1294: render text↔image parts in their ORIGINAL order so each
                # placeholder sits where the client placed it. Already in
                # chat-template shape; copy each part so ``apply_chat_template``
                # can't mutate the stored tuple. The bytes ride ``image_data`` in
                # the same order as the ``"image"`` markers here.
                d = {"role": role, "content": [dict(p) for p in m.content_parts]}
            elif m.images:
                # Legacy / no-layout path: hand the template a content-parts list
                # so it inserts one image placeholder per image (Qwen-VL templates
                # key off the ``"image"`` part type; the bytes travel separately
                # as ``image_data``). Images lead, then the text — the flat image
                # order the adapter receives.
                d = image_first_chat_message(role=role, text=m.content, image_count=len(m.images))
            else:
                d = {"role": role, "content": m.content}
            if m.tool_calls:
                d["tool_calls"] = [self._normalise_tool_call(tc) for tc in m.tool_calls]
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            message_dicts.append(d)
        # Pass the OpenAI-shaped tools through to the chat template. Qwen3's
        # template renders the function schemas into a system preamble and
        # instructs the model to emit ``<tool_call>{...}</tool_call>`` — which
        # ``parse_tool_call_stream`` then converts to OpenAI tool_calls.
        if tools:
            kwargs["tools"] = list(tools)
        try:
            # Route through the bounded ``_GRAMMAR_EXECUTOR`` rather than
            # the shared default ``asyncio.to_thread`` pool: this call is
            # CPU-bound and input-sized, so a burst on the default pool
            # could starve the rest of the worker (tokenizer loads, etc.).
            # ``run_in_executor`` takes no kwargs, so bind them in a
            # closure (matching the ``lambda`` style in
            # ``_check_context_length``).
            loop = asyncio.get_running_loop()
            rendered = await loop.run_in_executor(
                _GRAMMAR_EXECUTOR,
                lambda: tok.apply_chat_template(
                    message_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "chat template render failed for %s: %s",
                model_id,
                exc,
            )
            return _ValidationError(
                code="invalid_request",
                message=f"chat template render failed: {exc}",
            )
        if not isinstance(rendered, str):
            return _ValidationError(
                code="invalid_request",
                message="chat template did not return a string (tokenize=False)",
            )
        return rendered

    async def _check_context_length(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        num_images: int = 0,
    ) -> _ValidationError | None:
        """Worker-side ``prompt_tokens + max_new_tokens > context_length``
        guard. Emits a terminal ``code: "context_exceeded"`` chunk when
        the rendered prompt already overflows the budget the model
        config declares.

        Returns ``None`` if the budget is OK or if the model has no
        declared ``context_length`` (defensive — config validation
        normally requires it).
        """
        try:
            config = self._registry.get_config(model_id)
        except KeyError:
            return None
        gen = config.tasks.generate
        if gen is None:
            return None
        context_length = gen.context_length
        # Defensive: ``context_length`` is typed as ``int`` by
        # :class:`GenerateTask`, but test fixtures and out-of-tree
        # configs may not enforce that. Skip the check if it's not a
        # positive int rather than crashing the work loop.
        if not isinstance(context_length, int) or context_length <= 0:
            return None
        try:
            tok = await self._get_tokenizer(model_id)
        except Exception:  # noqa: BLE001
            # If we cannot load a tokenizer, skip the check rather than
            # fail-closed — the SGLang subprocess will still enforce its
            # own limits. The deferred grammar/admission path adds a hard gate.
            logger.debug(
                "context-length check skipped for %s: tokenizer unavailable",
                model_id,
                exc_info=True,
            )
            return None
        # Cap the prompt before the CPU-bound encode so a pathologically
        # large prompt can't pin a pool thread. Truncating only the chars
        # fed to the tokenizer is safe for the guard: a prompt this large
        # already overflows any realistic ``context_length`` (the check
        # below still fires on the truncated prefix), and the SGLang
        # subprocess enforces its own hard limit besides.
        encode_prompt = prompt
        if len(prompt) > _MAX_PROMPT_CHARS:
            logger.debug(
                "context-length check truncating oversized prompt for %s (%d > %d chars)",
                model_id,
                len(prompt),
                _MAX_PROMPT_CHARS,
            )
            encode_prompt = prompt[:_MAX_PROMPT_CHARS]
        try:
            # Route through the bounded ``_GRAMMAR_EXECUTOR`` rather than
            # the shared default ``asyncio.to_thread`` pool — this encode
            # is CPU-bound and input-sized.
            loop = asyncio.get_running_loop()
            prompt_tokens = await loop.run_in_executor(
                _GRAMMAR_EXECUTOR,
                lambda: len(tok.encode(encode_prompt, add_special_tokens=False)),
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "context-length tokenization failed for %s",
                model_id,
                exc_info=True,
            )
            return None
        # Account for vision-encoder token expansion: each image placeholder
        # in ``prompt`` tokenizes to a handful of text tokens but expands to
        # many tokens at inference. Add a coarse per-image estimate so the
        # guard fires before SGLang overflows its context window.
        image_tokens = num_images * _VISION_TOKENS_PER_IMAGE_ESTIMATE
        total = prompt_tokens + image_tokens + max_new_tokens
        if total > context_length:
            image_note = f" + ~image_tokens ({image_tokens})" if image_tokens else ""
            return _ValidationError(
                code="context_exceeded",
                message=(
                    f"prompt_tokens ({prompt_tokens}){image_note} + max_new_tokens "
                    f"({max_new_tokens}) = {total} exceeds context_length "
                    f"({context_length}) for model '{model_id}'"
                ),
            )
        return None

    async def _ensure_grammar_ready(
        self,
        grammar: GrammarSpec,
        *,
        model_id: str,
        reply_subject: str,
        request_id: str,
        attempt_id: str,
        msg: Any,
    ) -> bool:
        """Validate (and cache) the Outlines compile for ``grammar``.

        On miss runs :func:`compile_outlines` inside
        :func:`asyncio.to_thread` with a 5-second wall-clock timeout.
        On hit increments the cache-hits counter and returns
        immediately.

        On failure (timeout, validation error, missing Outlines) publishes
        a terminal-error chunk on ``reply_subject`` and ACKs the work
        message (failure is delivered cleanly via the error chunk —
        redelivery cannot help fix the schema). Returns ``False`` so
        the caller skips :meth:`_stream_generate`. Returns ``True`` on
        success.

        The compile runs only for cache misses so a steady-state
        request mix (a few schemas, many requests) pays the compile
        cost once per (tokenizer, schema) pair. The 5-second cap is
        the contractual limit from §4.6 of the parent plan.
        """
        # Resolve a stable tokenizer hash. The model wrapper does not
        # advertise one yet, so we synthesise from the model config.
        # Cache invariant: this hash must be stable for the lifetime
        # of a (worker process, model) pair and must differ across
        # models. ``hf_id`` is the primary signal — it's the
        # canonical model identifier and never changes for a loaded
        # model. ``weights_path`` is the local-checkpoint fallback for
        # tests / offline deployments. ``model_id`` is a last-resort
        # fallback so the cache still functions when the registry has
        # no config (e.g. ad-hoc test fixtures); a future
        # ``model.tokenizer_hash`` property would replace this chain.
        try:
            config = self._registry.get_config(model_id)
            tokenizer_hash = str(config.hf_id or config.weights_path or model_id)
        except (KeyError, AttributeError):
            tokenizer_hash = model_id

        # Compute the cache key inside the guarded block. ``hash_grammar``
        # can raise ``TypeError`` on a malformed ``grammar.value`` (e.g. a
        # JSON schema that isn't a dict) — previously this was OUTSIDE any
        # try/except, so the TypeError propagated unhandled all the way up,
        # leaving the request with no terminal chunk and no ACK → JetStream
        # redelivered it forever (a hang/redeliver loop). Surface it as a
        # clean ``grammar_invalid`` terminal error and settle the message.
        try:
            key: tuple[str, str, str] = (tokenizer_hash, hash_grammar(grammar), "outlines")
        except (TypeError, ValueError) as exc:
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code="grammar_invalid",
                message=f"invalid grammar: {exc}",
                msg=msg,
            )
            return False
        cached = self._grammar_cache.get(key)
        if cached is not None:
            _metrics.worker_telemetry().grammar_cache_lookup(
                model=model_id,
                backend="outlines",
                grammar=grammar.kind,
                phase="request",
                result="hit",
            )
            return True

        # Single-flight: if another coroutine is already compiling this
        # exact ``(tokenizer, schema)`` pair, wait on its future
        # instead of running our own. This collapses thundering-herd
        # cold-start traffic (N concurrent first-requests for the
        # same schema → 1 compile + N-1 cache hits).
        async with self._grammar_inflight_lock:
            existing = self._grammar_inflight.get(key)
            if existing is not None:
                # We're a follower; share the leader's future. Bump
                # the hit counter because, semantically, we did not
                # pay the compile cost — the leader did.
                future_to_await = existing
                is_leader = False
            else:
                future_to_await = asyncio.get_running_loop().create_future()
                self._grammar_inflight[key] = future_to_await
                is_leader = True

        if not is_leader:
            try:
                # Bounded wait: never block forever on the leader. The
                # leader's compile is itself capped at 5s plus tokenizer
                # load, so a generous ceiling here still guarantees the
                # follower settles its own work item rather than hanging
                # until ack_wait and triggering redelivery.
                await _wait_for(future_to_await, timeout=_GRAMMAR_FOLLOWER_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                # Leader's compile failed, was cancelled, or did not
                # resolve in time; surface the same terminal-error path.
                await self._terminal_error_then_settle(
                    reply_subject,
                    request_id=request_id,
                    attempt_id=attempt_id,
                    seq=0,
                    code="grammar_compile_failed",
                    message=f"shared grammar compile failed: {exc}",
                    msg=msg,
                )
                return False
            _metrics.worker_telemetry().grammar_cache_lookup(
                model=model_id,
                backend="outlines",
                grammar=grammar.kind,
                phase="request",
                result="hit",
            )
            return True

        _metrics.worker_telemetry().grammar_cache_lookup(
            model=model_id,
            backend="outlines",
            grammar=grammar.kind,
            phase="request",
            result="miss",
        )

        # Leader path: load tokenizer + run the bounded-pool compile.
        # The whole leader body is wrapped so that even on
        # ``CancelledError`` (which the per-except clauses below do NOT
        # catch) the shared future is always resolved and the inflight
        # key always popped. Otherwise a cancel between creating the
        # future and resolving it would leave every follower awaiting it
        # forever — the process never ACKs and JetStream redelivers in a
        # storm.
        try:
            return await self._run_leader_compile(
                key,
                grammar,
                future_to_await,
                model_id=model_id,
                reply_subject=reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                msg=msg,
            )
        finally:
            if not future_to_await.done():
                # Reached only when the leader was cancelled (or raised
                # ``BaseException``) before resolving the future. Resolve it
                # with a *regular* exception — NOT ``CancelledError`` — so
                # followers' ``except Exception`` catches it and surfaces a
                # clean terminal error instead of having their own task
                # cancelled. The leader's own ``CancelledError`` still
                # propagates out of this ``finally`` unchanged.
                future_to_await.set_exception(RuntimeError("grammar compile leader cancelled before resolving"))
            async with self._grammar_inflight_lock:
                self._grammar_inflight.pop(key, None)

    async def _run_leader_compile(
        self,
        key: tuple[str, str, str],
        grammar: GrammarSpec,
        future_to_await: asyncio.Future[Any],
        *,
        model_id: str,
        reply_subject: str,
        request_id: str,
        attempt_id: str,
        msg: Any,
    ) -> bool:
        """Run the single-flight leader compile for ``key``.

        The shared future and ``_grammar_inflight`` cleanup are owned by
        the caller's ``finally`` so this body can return/raise freely
        without leaking a never-resolved future to followers.
        """
        started_at = time.monotonic()

        def record_compile(outcome: str) -> None:
            _metrics.worker_telemetry().grammar_compile_completed(
                model=model_id,
                backend="outlines",
                grammar=grammar.kind,
                phase="request",
                outcome=outcome,
                duration_s=time.monotonic() - started_at,
            )

        try:
            tok = await self._get_tokenizer(model_id)
        except Exception as exc:  # noqa: BLE001
            record_compile("error")
            future_to_await.set_exception(exc)
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code="grammar_compile_failed",
                message=f"failed to load tokenizer for grammar compile: {exc}",
                msg=msg,
            )
            return False

        try:
            # Submit the blocking compile to the dedicated executor
            # (bounded at 4 threads). ``run_in_executor`` returns a
            # concurrent future wrapped as an asyncio future, so
            # :func:`_wait_for` cancels cleanly even though the
            # underlying thread continues until Outlines returns.
            loop = asyncio.get_running_loop()
            compiled = await _wait_for(
                loop.run_in_executor(_GRAMMAR_EXECUTOR, compile_outlines, tok, grammar),
                timeout=5.0,
            )
        except TimeoutError as exc:
            record_compile("timeout")
            future_to_await.set_exception(exc)
            logger.warning(
                "grammar compile for %s exceeded 5s — thread continues until Outlines returns",
                model_id,
            )
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code="grammar_compile_failed",
                message="grammar compile exceeded 5s",
                msg=msg,
            )
            return False
        except GrammarValidationError as exc:
            record_compile("error")
            future_to_await.set_exception(exc)
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code=exc.code,
                message=str(exc),
                msg=msg,
            )
            return False
        except Exception as exc:  # noqa: BLE001
            # Defensive: any Outlines-internal exception not already
            # wrapped by :func:`compile_outlines`. Surface as
            # ``grammar_compile_failed`` so the client sees a stable
            # code rather than ``inference_error``.
            record_compile("error")
            future_to_await.set_exception(exc)
            await self._terminal_error_then_settle(
                reply_subject,
                request_id=request_id,
                attempt_id=attempt_id,
                seq=0,
                code="grammar_compile_failed",
                message=f"grammar compile raised: {exc}",
                msg=msg,
            )
            return False

        record_compile("success")
        self._grammar_cache.put(key, compiled)
        future_to_await.set_result(compiled)
        return True

    async def _publish_nak_envelope(
        self,
        reply_subject: str,
        *,
        request_id: str,
        attempt_id: str,
        reason: str,
    ) -> bool:
        """Emit a ``kind:"nak"`` envelope to the inbox.

        The gateway's inbox subscriber routes this to
        ``republish_to_pool`` instead of treating it as a terminal
        chunk. Use sparingly — only when the gateway is expected to
        retry on another worker (model not loaded, KV budget
        rejection, worker shutting down). For all other failures
        prefer ``_publish_terminal_error`` which surfaces directly
        to the client.

        Returns ``True`` if the NAK was successfully published to core
        NATS; ``False`` otherwise. Callers MUST check this before
        ACKing the JetStream work item — ACKing after a failed NAK
        publish leaves the request orphaned (no redelivery, no client
        notification). On ``False`` the caller should skip the ACK
        and let JetStream redeliver.
        """
        payload = msgpack.packb(
            {
                "kind": "nak",
                "request_id": request_id,
                "attempt_id": attempt_id,
                "reason": reason,
            },
            use_bin_type=True,
        )
        try:
            # A NAK is a retry handoff, not a completed generation. Do not
            # route it through the terminal-completion seam; the redelivered
            # attempt will own the eventual success/error/cancelled outcome.
            await self._nc.publish(reply_subject, payload)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to publish NAK envelope for %s/%s reason=%s — skipping ACK so JetStream redelivers",
                request_id,
                attempt_id,
                reason,
                exc_info=True,
            )
            return False
        return True

    async def _terminal_error_then_settle(
        self,
        reply_subject: str,
        *,
        request_id: str,
        attempt_id: str,
        seq: int,
        code: str,
        message: str,
        msg: Any,
    ) -> None:
        """Publish a terminal error chunk, then ACK on success / NAK on failure.

        Single settle point for the pre-stream error paths
        (validation, chat-template, context-length, tool-choice, grammar
        compile). ACKing unconditionally after a failed terminal publish
        orphans the request — no redelivery, no client notification — so
        we only ACK when the chunk reached the gateway; otherwise we NAK
        so JetStream redelivers (mirrors the NAK-envelope pattern).
        """
        published = await self._publish_terminal_error(
            reply_subject,
            request_id=request_id,
            attempt_id=attempt_id,
            seq=seq,
            code=code,
            message=message,
        )
        if published:
            await _safe_ack(msg)
        else:
            await _safe_nak(msg)

    async def _publish_cancelled_then_settle(
        self,
        reply_subject: str,
        *,
        request_id: str,
        attempt_id: str,
        msg: Any,
    ) -> None:
        """Emit a terminal ``finish_reason: "cancelled"`` chunk, then settle.

        Used by the preflight cancel short-circuits so a cancel arriving
        during a multi-second cold load / chat-template render / grammar
        compile is honoured before we ever start decoding — mirroring the
        ``cancelled`` terminal the streaming loop emits. ACK only when the
        chunk reached the gateway, else NAK for redelivery.
        """
        payload = _encode_chunk(
            kind="chunk",
            request_id=request_id,
            attempt_id=attempt_id,
            seq=0,
            text_delta="",
            done=True,
            finish_reason="cancelled",
        )
        try:
            await self._publish_terminal_envelope(reply_subject, payload)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to publish cancelled terminal chunk for %s/%s — skipping ACK so JetStream redelivers",
                request_id,
                attempt_id,
                exc_info=True,
            )
            await _safe_nak(msg)
            return
        await _safe_ack(msg)

    async def _publish_terminal_error(
        self,
        reply_subject: str,
        *,
        request_id: str,
        attempt_id: str,
        seq: int,
        code: str,
        message: str,
    ) -> bool:
        """Publish a terminal error chunk to the inbox.

        Returns ``True`` if the chunk reached core NATS, ``False``
        otherwise. Callers MUST check this before ACKing the JetStream
        work item — ACKing after a failed terminal publish swallows the
        failure and orphans the request (no redelivery, no client
        notification). On ``False`` the caller should NAK so JetStream
        redelivers (mirrors :meth:`_publish_nak_envelope`).
        """
        payload = _encode_chunk(
            kind="chunk",
            request_id=request_id,
            attempt_id=attempt_id,
            seq=seq,
            text_delta="",
            done=True,
            finish_reason="error",
            error_code=code,
            error_message=message,
        )
        try:
            await self._publish_terminal_envelope(reply_subject, payload)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to publish terminal error chunk for %s/%s (code=%s) — skipping ACK so JetStream redelivers",
                request_id,
                attempt_id,
                code,
                exc_info=True,
            )
            return False
        return True


# -- Module-level helpers ----------------------------------------------------


class _Suppress:
    """Context manager that swallows any exception. Local replacement for
    ``contextlib.suppress`` without the all-exceptions-bare-except smell.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return True


# Lowercase alias for the historical call sites that read like
# ``with _suppress(): ...``. New code should prefer :class:`_Suppress`.
_suppress = _Suppress


def _compute_ttft_ms(publish_at: float, first_text_at: float | None) -> float | None:
    if first_text_at is None:
        return None
    return max(0.0, (first_text_at - publish_at) * 1000.0)


def _encode_chunk(
    *,
    kind: str,
    request_id: str,
    attempt_id: str,
    seq: int,
    text_delta: str,
    done: bool,
    is_first: bool = False,
    finish_reason: FinishReason | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    ttft_ms: float | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    tool_call_delta: ToolCallDelta | None = None,
    logprobs: list[dict[str, Any]] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    choice_index: int = 0,
) -> bytes:
    """Encode one chunk envelope as msgpack bytes.

    When ``tool_call_delta`` is set, the envelope carries
    ``tool_calls: [<wire-shape>]`` — a single-element list that matches
    OpenAI's streaming ``delta.tool_calls`` shape so the gateway can
    forward it byte-identical to the SSE client. The wire shape is the
    flat OpenAI dict (``index``, optional ``id``, ``type``,
    ``function: {name?, arguments}``); see :class:`ToolCallDelta` for
    the source of truth.
    """
    payload: dict[str, Any] = {
        "kind": kind,
        "request_id": request_id,
        "attempt_id": attempt_id,
        "seq": seq,
        "text_delta": text_delta,
        "done": done,
    }
    if is_first:
        payload["is_first"] = True
    if finish_reason is not None:
        payload["finish_reason"] = finish_reason
    if prompt_tokens is not None or completion_tokens is not None:
        payload["usage"] = {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int((prompt_tokens or 0) + (completion_tokens or 0)),
        }
    if ttft_ms is not None:
        payload["ttft_ms"] = ttft_ms
    if error_code is not None or error_message is not None:
        payload["error"] = {
            "code": error_code or "error",
            "message": error_message or "",
        }
    if logprobs:
        # OpenAI ``ChatCompletionTokenLogprob`` entries, produced by the
        # adapter for the tokens in ``text_delta``. Forwarded verbatim;
        # the gateway wraps them in ``{content, refusal}`` per surface.
        payload["logprobs"] = logprobs
    if candidates:
        # Multi-candidate (`n > 1`) results on the terminal chunk. Each entry
        # is ``{text, finish_reason, logprobs?}``; the gateway turns them into
        # the OpenAI multi-entry ``choices`` array.
        payload["candidates"] = candidates
    if choice_index:
        # Streaming multi-candidate: the candidate ordinal this delta belongs to.
        payload["choice_index"] = choice_index
    if tool_call_delta is not None:
        function_block: dict[str, Any] = {}
        if tool_call_delta.function_name is not None:
            function_block["name"] = tool_call_delta.function_name
        # The arguments string is always present in the wire shape —
        # OpenAI clients concatenate ``arguments`` deltas, so an empty
        # string on the announcement chunk is correct (and required).
        function_block["arguments"] = tool_call_delta.arguments_delta
        wire: dict[str, Any] = {
            "index": tool_call_delta.index,
            "type": tool_call_delta.type,
            "function": function_block,
        }
        if tool_call_delta.id is not None:
            wire["id"] = tool_call_delta.id
        payload["tool_calls"] = [wire]
    return msgpack.packb(payload, use_bin_type=True)


async def _safe_ack(msg: Any) -> None:
    try:
        await msg.ack()
    except Exception:  # noqa: BLE001
        logger.debug("Failed to ACK generate msg", exc_info=True)


async def _safe_nak(msg: Any) -> None:
    try:
        await msg.nak(delay=_NAK_DELAY_S)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to NAK generate msg", exc_info=True)
