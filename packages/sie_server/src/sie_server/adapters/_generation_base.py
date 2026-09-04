"""Abstract base for generation (LLM autoregressive decode) adapters.

Sibling to :class:`~sie_server.adapters._base_adapter.BaseAdapter`. Generation
is categorically different from the embedding/score/extract triad: lifecycle,
cancellation, and partial-state semantics are not a method bolt-on. The
``GenerationAdapter`` ABC declares the streaming contract:

- async-iterator ``generate(prompt, ...)`` yielding :class:`GenerationChunk`
- worker dispatch on ``isinstance(adapter, GenerationAdapter)``

Concrete adapters yield chunks as the upstream engine produces them, with the
terminal chunk carrying ``finish_reason`` and ``usage``.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import sys
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, cast

from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput

logger = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_GEMMA_CHANNEL_OPEN = "<" + "|channel" + ">"
_GEMMA_CHANNEL_CLOSE = "<" + "channel|" + ">"

ReasoningFormat = Literal["qwen3", "gemma4"]
_REASONING_BOUNDARIES: dict[ReasoningFormat, tuple[str, str]] = {
    "qwen3": (_THINK_OPEN, _THINK_CLOSE),
    "gemma4": (_GEMMA_CHANNEL_OPEN + "thought\n", _GEMMA_CHANNEL_CLOSE),
}


# Finish reason values surfaced to gateway / client. ``cancelled`` lands when
# the worker observed a cancel signal mid-stream (§4.4.2). ``error`` lands
# when the upstream engine raised; concrete adapters may also produce
# ``length`` (max_new_tokens reached) and ``stop`` (natural EOS / stop string).
# ``tool_calls`` is the OpenAI-compatible terminator emitted by the
# tool-call parser when one or more ``<tool_call>...</tool_call>`` blocks
# were consumed before the underlying model stopped.
FinishReason = Literal["stop", "length", "cancelled", "error", "tool_calls"]


class GenerationError(RuntimeError):
    """Base for adapter-declared generation failures with stable semantics."""

    code: ClassVar[str] = "inference_error"
    param: str | None = None


class GenerationUnsupportedFieldError(GenerationError):
    """A request parameter is not supported by the active generation backend."""

    code = "unsupported_field"

    def __init__(self, param: str, message: str | None = None) -> None:
        self.param = param
        super().__init__(message or f"'{param}' is not supported by this generation backend")


class GenerationInvalidRequestError(GenerationError):
    """An adapter-specific request relationship is invalid."""

    code = "invalid_request"

    def __init__(self, param: str, message: str) -> None:
        self.param = param
        super().__init__(message)


class GenerationInputTooLongError(GenerationError):
    """The rendered input exceeds an adapter's qualified token ceiling."""

    code = "INPUT_TOO_LONG"

    def __init__(self, message: str, *, param: str = "prompt") -> None:
        self.param = param
        super().__init__(message)


class GenerationCapacityError(GenerationError):
    """The generation backend is temporarily at bounded capacity."""

    code = "RESOURCE_EXHAUSTED"


class GenerationDrainingError(GenerationCapacityError):
    """The generation backend is draining and cannot accept new work."""

    code = "MODEL_LOADING"


_CLIENT_SAFE_GENERATION_ERROR_MESSAGES = {
    "inference_error": "internal error during generation",
    "grammar_compile_failed": "internal error compiling grammar",
}
_CLIENT_SAFE_GENERATION_ERROR_CODES = frozenset(
    {
        "inference_error",
        "grammar_compile_failed",
        "INPUT_TOO_LONG",
        "MODEL_LOADING",
        "MODEL_OUTPUT_PARSE_ERROR",
        "RESOURCE_EXHAUSTED",
        "COLD_START_RATE_LIMITED",
        "LORA_LOADING",
        "PAYLOAD_TOO_LARGE",
        "cancelled",
        "context_exceeded",
        "empty_model_output",
        "grammar_invalid",
        "invalid_request",
        "parallel_tool_calls_violated",
        "rate_limit_exceeded",
        "tool_call_parse_error",
        "transport_failure",
        "unsupported_field",
    }
)
_CLIENT_PASSTHROUGH_GENERATION_ERROR_CODES = frozenset(
    {
        "INPUT_TOO_LONG",
        "MODEL_LOADING",
        "MODEL_OUTPUT_PARSE_ERROR",
        "RESOURCE_EXHAUSTED",
        "empty_model_output",
        "grammar_invalid",
        "invalid_request",
        "parallel_tool_calls_violated",
        "tool_call_parse_error",
        "unsupported_field",
    }
)
_GENERIC_UPSTREAM_GENERATION_ERROR_MESSAGE = "generation terminated with an upstream error"
_CLIENT_SAFE_GENERATION_PARAMS = frozenset(
    {
        "prompt",
        "max_new_tokens",
        "n",
        "best_of",
        "lora_adapter",
        "temperature",
        "top_p",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "top_k",
        "repetition_penalty",
        "min_new_tokens",
        "grammar",
        "seed",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "images",
        "stream",
    }
)
_CLIENT_REFUSAL_ERROR_TYPES = (
    GenerationUnsupportedFieldError,
    GenerationInvalidRequestError,
    GenerationInputTooLongError,
)
_INTERNAL_TO_PUBLIC_GENERATION_PARAMS = {"lora_path": "lora_adapter"}


def client_safe_generation_error_code(error_code: str | None) -> str:
    """Return the closed public discriminator for an adapter terminal error."""
    code = error_code or "inference_error"
    return code if code in _CLIENT_SAFE_GENERATION_ERROR_CODES else "inference_error"


def client_safe_generation_error_message(error_code: str | None, error_message: str | None) -> str:
    """Return the public message for an adapter-yielded terminal error.

    Adapter terminal chunks are an untrusted backend boundary just like raised
    exceptions. Generic runtime/grammar-internal codes therefore use stable
    public text, while only explicitly approved client/model codes retain
    their actionable message. Unknown future adapter codes fail closed rather
    than making their backend-controlled message public. A terminal error
    without a code is the generic ``inference_error`` fallback used by every
    public generation surface.
    """
    code = error_code or "inference_error"
    if safe_message := _CLIENT_SAFE_GENERATION_ERROR_MESSAGES.get(code):
        return safe_message
    if code in _CLIENT_PASSTHROUGH_GENERATION_ERROR_CODES and error_message:
        return error_message
    return _GENERIC_UPSTREAM_GENERATION_ERROR_MESSAGE


def client_safe_generation_error_param(error: GenerationError) -> str | None:
    """Return a bounded public parameter for an adapter-raised refusal.

    A generic/future ``GenerationError`` may set ``param`` arbitrarily, so the
    parameter is public only for concrete client-refusal types and the finite
    generation parameter contract. New types or fields require an explicit
    contract update and regression before they can cross this boundary.
    """
    if type(error) not in _CLIENT_REFUSAL_ERROR_TYPES:
        return None
    raw_param = error.param
    if raw_param is None:
        return None
    param = _INTERNAL_TO_PUBLIC_GENERATION_PARAMS.get(raw_param, raw_param)
    return param if param in _CLIENT_SAFE_GENERATION_PARAMS else None


def _freeze_preflight_value(value: Any) -> Any:
    """Return a deeply immutable comparison snapshot for generation kwargs."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze_preflight_value(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_preflight_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_preflight_value(item) for item in value)
    return value


class GenerationPreflightResult:
    """Opaque, request-scoped adapter preflight evidence.

    The result is carried by the caller into :meth:`generate_with_preflight`
    and may be consumed exactly once by the adapter that produced it, for an
    exact deeply-immutable parameter snapshot. It is never stored on the
    adapter or keyed by prompt/model text, so concurrent requests cannot share
    evidence accidentally.
    """

    __slots__ = ("_adapter", "_consumed", "_parameters", "_payload")

    def __init__(self, adapter: object, parameters: Mapping[str, Any], payload: Any) -> None:
        self._adapter: object | None = adapter
        self._parameters: Any | None = _freeze_preflight_value(parameters)
        self._payload: Any | None = payload
        self._consumed = False

    def _consume(self, adapter: object, parameters: Mapping[str, Any]) -> Any | None:
        matches = (
            not self._consumed and self._adapter is adapter and self._parameters == _freeze_preflight_value(parameters)
        )
        payload = self._payload if matches else None
        self.discard()
        return payload

    def discard(self) -> None:
        """Erase request-owned references and make the result unusable."""
        self._adapter = None
        self._parameters = None
        self._payload = None
        self._consumed = True


_ACTIVE_GENERATION_PREFLIGHT: ContextVar[GenerationPreflightResult | None] = ContextVar(
    "active_generation_preflight",
    default=None,
)


def consume_generation_preflight(adapter: object, parameters: Mapping[str, Any]) -> Any | None:
    """Consume exact preflight evidence bound by ``generate_with_preflight``."""
    result = _ACTIVE_GENERATION_PREFLIGHT.get()
    return None if result is None else result._consume(adapter, parameters)


class _PreflightBoundGenerationIterator(AsyncIterator["GenerationChunk"]):
    """Own one preflight result from binding through iterator teardown.

    A plain nested async generator cannot run its ``finally`` block when it is
    closed before its first ``__anext__``. This explicit owner discards the
    evidence from ``aclose()`` even in that unstarted state. The state lock
    also serializes first-start and close, while the queue remains free to run
    successive ``__anext__`` calls in distinct child tasks.
    """

    def __init__(
        self,
        adapter: Any,
        parameters: Mapping[str, Any],
        preflight: GenerationPreflightResult | None,
    ) -> None:
        self._adapter: Any | None = adapter
        self._parameters: dict[str, Any] | None = dict(parameters)
        self._preflight = preflight
        self._chunks: AsyncIterator[GenerationChunk] | None = None
        self._terminal_outcome_selected = False
        self._closed = False
        self._exhausted = False
        self._state_lock = asyncio.Lock()

    def __aiter__(self) -> _PreflightBoundGenerationIterator:
        return self

    async def __anext__(self) -> GenerationChunk:
        async with self._state_lock:
            if self._closed or self._exhausted:
                raise StopAsyncIteration

            if self._chunks is None:
                adapter = self._adapter
                parameters = self._parameters
                if adapter is None or parameters is None:
                    raise StopAsyncIteration
                try:
                    self._chunks = adapter.generate(**dict(parameters))
                except BaseException:
                    await self._close_locked()
                    raise

            token = _ACTIVE_GENERATION_PREFLIGHT.set(self._preflight)
            exhausted = False
            try:
                try:
                    chunk = await self._chunks.__anext__()
                except StopAsyncIteration:
                    exhausted = True
                except BaseException:
                    await self._close_locked()
                    raise
            finally:
                _ACTIVE_GENERATION_PREFLIGHT.reset(token)

            if exhausted:
                # The effective iterator's caller owns final closure. Defer
                # upstream ``aclose`` until that outer finally block so a
                # cleanup failure remains authoritative after clean
                # exhaustion instead of being misclassified as an inference
                # error from ``__anext__``.
                self._exhausted = True
                self._discard_owned_references()
                raise StopAsyncIteration
            if chunk.done:
                self._terminal_outcome_selected = True
            return chunk

    async def aclose(self) -> None:
        async with self._state_lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        chunks = self._chunks
        self._chunks = None
        outcome_selected = self._terminal_outcome_selected
        self._discard_owned_references()
        if chunks is not None:
            await aclose_with_error_precedence(
                chunks,
                outcome_selected=outcome_selected,
                context="preflight-bound adapter iterator",
            )

    def _discard_owned_references(self) -> None:
        preflight = self._preflight
        self._preflight = None
        self._adapter = None
        self._parameters = None
        if preflight is not None:
            preflight.discard()


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """One streaming-shape OpenAI tool-call delta.

    OpenAI's chat-completion streaming format carries tool calls as a
    list of deltas: each delta has an ``index`` (which call within the
    response), an ``id`` set on the first delta of each call only, a
    ``function.name`` set on the first delta only, and an
    ``function.arguments`` string that accumulates JSON across deltas.

    The worker emits these as **two** delta chunks per parsed
    ``<tool_call>{...}</tool_call>`` block: one with
    ``id`` + ``function_name`` + empty ``arguments_delta``, then one
    with the full JSON-encoded arguments under ``arguments_delta`` (no
    ``id`` / ``function_name``). The gateway forwards each as one
    ``delta.tool_calls`` SSE event.

    Multiple parallel tool calls map to multiple ``index`` values; the
    parser increments ``index`` per ``<tool_call>`` block observed.
    """

    index: int
    id: str | None = None
    type: Literal["function"] = "function"
    function_name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True, slots=True)
class GenerationChunk:
    """One chunk yielded by a streaming :meth:`GenerationAdapter.generate`.

    The adapter contract is: yield zero or more *delta* chunks
    (``done=False``, ``text_delta`` populated), followed by exactly one
    *terminal* chunk (``done=True``, optional ``text_delta``, mandatory
    ``finish_reason``, optional ``prompt_tokens`` / ``completion_tokens``).

    ``is_first`` marks the first chunk that carries non-empty text — the
    worker uses it to record TTFT (§4.11).

    ``tool_call_delta`` carries a single OpenAI-compatible tool-call
    delta when the tool-call parser is active and emitted one. Each
    parsed ``<tool_call>{...}</tool_call>`` block yields exactly two
    chunks: one with ``id`` + ``function_name`` set (announcement) and
    one with ``arguments_delta`` set to the JSON-encoded arguments
    (body). The wire envelope serialises each chunk's delta as a
    single-element ``tool_calls`` list — using a list at the envelope
    boundary matches OpenAI's streaming shape exactly. ``error_code``
    / ``error_message`` carry a parser-detected terminal error (e.g.
    malformed tool-call JSON) so the worker can surface a
    ``finish_reason: "error"`` chunk without inventing the wire shape
    here.
    """

    text_delta: str
    done: bool = False
    is_first: bool = False
    finish_reason: FinishReason | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_call_delta: ToolCallDelta | None = None
    error_code: str | None = None
    error_message: str | None = None
    # OpenAI-shape per-token log-probabilities for the tokens that
    # produced ``text_delta``. ``None`` (the default) when the request
    # did not ask for logprobs. Each entry is the OpenAI
    # ``ChatCompletionTokenLogprob`` shape: ``{token: str, logprob: float,
    # bytes: list[int] | None, top_logprobs: list[{token, logprob, bytes}]}``.
    # The adapter translates from SGLang's
    # ``meta_info.output_token_logprobs`` / ``output_top_logprobs`` into
    # this shape so neither the worker chunk-encoder nor the gateway
    # has to know SGLang's specific layout.
    logprobs: tuple[dict[str, Any], ...] | None = None
    # Multi-candidate (`n > 1`) results, set ONLY on the terminal chunk when
    # the request asked for more than one candidate. Each entry is the wire
    # shape the gateway turns into one OpenAI ``choices[]`` entry:
    # ``{text: str, finish_reason: str | None, logprobs: list | None}``. For
    # single-candidate requests (the default) this stays ``None`` and the
    # ordinary ``text_delta`` stream path is used.
    candidates: tuple[dict[str, Any], ...] | None = None
    # Streaming multi-candidate (`n>1 && stream`): the candidate ordinal this
    # delta belongs to (`[0, n)`). Default 0 — the single-candidate stream. The
    # worker forwards it on the wire chunk; the gateway maps it to
    # ``choices[0].index``.
    choice_index: int = 0


# Backwards-compatibility alias: walking-skeleton callers (the local-dev
# /v1/generate route and a couple of tests) consume a single
# :class:`GenerationResult`. The streaming contract keeps the type so those callers can
# drain the iterator and build the same shape without changing wire-visible
# response fields. Marked for removal once the chat-completions surface lands streaming SDKs.
@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Aggregated, walking-skeleton-shape result of a streaming generation.

    Used by callers that don't yet consume the chunk iterator — currently
    the local-dev ``/v1/generate`` HTTP route and unit tests for the
    blocking adapter shape. The async iterator is the canonical contract;
    this aggregate is built from it. ``error_code`` / ``error_message``
    mirror the terminal chunk's typed error fields (e.g.
    ``empty_model_output``) so buffered consumers can fail closed with the
    same semantics as the streaming path (#3104/#3136).
    """

    text: str
    finish_reason: Literal["stop", "length", "error", "cancelled"]
    prompt_tokens: int
    completion_tokens: int
    error_code: str | None = None
    error_message: str | None = None


class ThinkingBlockStripper:
    """Incrementally remove family-specific model reasoning blocks from text.

    SGLang's native ``/generate`` endpoint returns raw decoded text, so its
    launch-time ``--reasoning-parser`` does not protect SIE's generation
    stream. Delimiters may be split across arbitrary engine chunks; retain
    only the longest possible delimiter prefix between calls so neither a
    partial tag nor its body can escape before the closing tag arrives.
    """

    __slots__ = ("_buffer", "_close", "_inside", "_open")

    def __init__(self, inside: bool = False, *, reasoning_format: ReasoningFormat = "qwen3") -> None:
        self._buffer = ""
        self._inside = inside
        self._open, self._close = _REASONING_BOUNDARIES[reasoning_format]

    @staticmethod
    def _delimiter_prefix_length(text: str, delimiters: tuple[str, ...]) -> int:
        max_len = min(max(map(len, delimiters)) - 1, len(text))
        for length in range(max_len, 0, -1):
            suffix = text[-length:]
            if any(delimiter.startswith(suffix) for delimiter in delimiters):
                return length
        return 0

    def feed(self, text: str) -> str:
        self._buffer += text
        visible: list[str] = []

        while self._buffer:
            if self._inside:
                close_at = self._buffer.find(self._close)
                if close_at >= 0:
                    self._buffer = self._buffer[close_at + len(self._close) :]
                    self._inside = False
                    continue

                keep = self._delimiter_prefix_length(self._buffer, (self._close,))
                self._buffer = self._buffer[-keep:] if keep else ""
                break

            open_at = self._buffer.find(self._open)
            close_at = self._buffer.find(self._close)
            matches = [
                (index, delimiter)
                for index, delimiter in ((open_at, self._open), (close_at, self._close))
                if index >= 0
            ]
            if matches:
                marker_at, marker = min(matches, key=lambda match: match[0])
                visible.append(self._buffer[:marker_at])
                self._buffer = self._buffer[marker_at + len(marker) :]
                if marker == self._open:
                    self._inside = True
                # A stray closing marker is still hidden, but does not cause
                # surrounding ordinary answer text to be discarded.
                continue

            keep = self._delimiter_prefix_length(self._buffer, (self._open, self._close))
            emit_upto = len(self._buffer) - keep
            visible.append(self._buffer[:emit_upto])
            self._buffer = self._buffer[emit_upto:]
            break

        return "".join(visible)

    @property
    def inside(self) -> bool:
        return self._inside

    def finish(self) -> str:
        """Flush ordinary buffered text, but never truncated private reasoning."""
        visible = "" if self._inside else self._buffer
        self._buffer = ""
        self._inside = False
        return visible


def _strip_complete_thinking_blocks(
    text: str,
    *,
    start_inside: bool = False,
    reasoning_format: ReasoningFormat = "qwen3",
) -> str:
    stripper = ThinkingBlockStripper(inside=start_inside, reasoning_format=reasoning_format)
    return stripper.feed(text) + stripper.finish()


def _candidates_have_visible_text(candidates: tuple[dict[str, Any], ...] | None) -> bool:
    """Return whether any ``n > 1`` candidate aggregate carries visible text."""
    if not candidates:
        return False
    return any(isinstance(text := candidate.get("text"), str) and text.strip() for candidate in candidates)


async def aclose_with_error_precedence(
    resource: Any,
    *,
    outcome_selected: bool,
    context: str,
    timeout_s: float | None = None,
) -> None:
    """Close an async resource without replacing a primary outcome.

    ``finally: await resource.aclose()`` has a subtle failure mode: a cleanup
    exception replaces an exception already propagating through the
    ``finally`` block. That is especially harmful for ``CancelledError`` and
    ``GeneratorExit`` because an ordinary cleanup ``RuntimeError`` can then be
    translated into a successful/in-band response. Preserve the in-flight
    exception (or an already-selected public terminal outcome) and log the
    secondary close failure instead. With no primary exception or selected
    outcome, the close failure remains authoritative and is re-raised.
    """
    aclose = getattr(resource, "aclose", None)
    if aclose is None:
        return

    primary = sys.exception()
    try:
        close_awaitable = aclose()
        if timeout_s is None:
            await close_awaitable
        else:
            await asyncio.wait_for(close_awaitable, timeout=timeout_s)
    except BaseException as cleanup_error:
        if isinstance(cleanup_error, asyncio.CancelledError):
            raise
        if primary is not None or outcome_selected:
            logger.warning(
                "%s cleanup failed after a primary exception or selected outcome; preserving the primary",
                context,
                exc_info=True,
            )
            return
        raise


async def suppress_thinking_blocks(
    chunks: AsyncIterator[GenerationChunk],
    *,
    start_inside: bool = False,
    reasoning_format: ReasoningFormat = "qwen3",
) -> AsyncIterator[GenerationChunk]:
    """Hide model reasoning blocks while preserving generation metadata.

    Callers decide whether the resolved model declares the hidden-thinking
    contract. Both non-thinking and explicit thinking profiles use this wrapper:
    the profile controls tokenizer/model behavior, never public visibility of
    the private reasoning channel. Per-choice state keeps interleaved ``n > 1``
    streams independent, and terminal ``candidates`` are normalized through the
    same rule. When text is rewritten its token logprobs are omitted because the
    engine's token list can no longer be faithfully aligned with the visible
    delta. ``start_inside`` covers chat templates that place the opening
    reasoning marker in the prompt, so generated text begins with private
    reasoning and only emits the closing marker. ``reasoning_format`` selects
    the boundary pair declared by the model's generation backend.

    Closing the wrapper eagerly closes the upstream adapter iterator so client
    cancellation still reaches SGLang's ``/abort_request`` cleanup.

    A stream that terminates without ever producing meaningful visible output
    (no non-whitespace text on any choice or candidate, no tool calls) is
    stamped with ``error_code="empty_model_output"`` on the terminal chunk —
    e.g. when private reasoning consumed the entire generation budget. Without
    the stamp such a request settles as a nominally successful but unusable
    response that clients cannot distinguish from a real answer (#3104/#3136).
    Cancelled and already-errored terminals are left untouched.
    """
    states: dict[int, ThinkingBlockStripper] = {}
    emitted_visible_text = False
    emitted_meaningful_text = False
    emitted_tool_call = False
    hid_reasoning = False
    terminal_outcome_selected = False
    try:
        async for chunk in chunks:
            rewritten_candidates = chunk.candidates
            if chunk.candidates is not None:
                candidate_items: list[dict[str, Any]] = []
                for candidate in chunk.candidates:
                    item = dict(candidate)
                    text = item.get("text")
                    if isinstance(text, str):
                        visible = _strip_complete_thinking_blocks(
                            text,
                            start_inside=start_inside,
                            reasoning_format=reasoning_format,
                        )
                        if visible != text:
                            item["text"] = visible
                            item["logprobs"] = None
                            hid_reasoning = True
                    candidate_items.append(item)
                rewritten_candidates = tuple(candidate_items)

            state = states.setdefault(
                chunk.choice_index,
                ThinkingBlockStripper(inside=start_inside, reasoning_format=reasoning_format),
            )
            was_inside = state.inside
            visible_delta = state.feed(chunk.text_delta)
            text_changed = visible_delta != chunk.text_delta
            if chunk.done or chunk.finish_reason is not None:
                visible_delta += state.finish()

            is_first = bool(visible_delta) and not emitted_visible_text
            emitted_visible_text = emitted_visible_text or bool(visible_delta)
            emitted_meaningful_text = emitted_meaningful_text or bool(visible_delta.strip())
            emitted_tool_call = emitted_tool_call or chunk.tool_call_delta is not None
            hid_reasoning = hid_reasoning or text_changed or was_inside
            rewritten = replace(
                chunk,
                text_delta=visible_delta,
                is_first=is_first,
                logprobs=None if text_changed or was_inside else chunk.logprobs,
                candidates=rewritten_candidates,
            )

            if (
                rewritten.done
                and not emitted_meaningful_text
                and not emitted_tool_call
                and rewritten.error_code is None
                and rewritten.finish_reason in ("stop", "length")
                and not _candidates_have_visible_text(rewritten.candidates)
            ):
                reason = " (private reasoning consumed the generation budget)" if hid_reasoning else ""
                rewritten = replace(
                    rewritten,
                    error_code="empty_model_output",
                    error_message=f"model produced no visible output text{reason}",
                )

            # Reasoning-only engine deltas have no wire-visible information.
            # Keep terminals, per-choice finish markers, tool/error chunks, and
            # non-streaming candidate aggregates intact.
            if (
                not rewritten.text_delta
                and not rewritten.done
                and rewritten.finish_reason is None
                and rewritten.tool_call_delta is None
                and rewritten.error_code is None
                and rewritten.error_message is None
                and rewritten.logprobs is None
                and rewritten.candidates is None
            ):
                continue

            if rewritten.done:
                terminal_outcome_selected = True
            yield rewritten
    finally:
        await aclose_with_error_precedence(
            chunks,
            outcome_selected=terminal_outcome_selected,
            context="thinking-suppression upstream iterator",
        )


def thinking_blocks_must_be_hidden(config: Any) -> bool:
    """Return whether the resolved config declares hidden thinking semantics.

    An explicit boolean is materialized for every reasoning-capable Qwen and
    Gemma 4 variant. ``False`` keeps model reasoning disabled; ``True`` enables
    it internally. In both cases raw ``<think>`` blocks are never public output.
    """
    tasks = getattr(config, "tasks", None)
    generate = getattr(tasks, "generate", None)
    kwargs = getattr(generate, "chat_template_kwargs", None)
    return isinstance(kwargs, dict) and isinstance(kwargs.get("enable_thinking"), bool)


def _reasoning_format(value: object) -> ReasoningFormat | None:
    if value == "gemma4":
        return "gemma4"
    if value == "qwen3":
        return "qwen3"
    return None


def resolve_reasoning_format(config: Any, adapter: Any | None = None) -> ReasoningFormat:
    """Resolve the model's reasoning boundaries, preferring the loaded adapter."""
    adapter_format = _reasoning_format(getattr(adapter, "reasoning_parser", None))
    if adapter_format is not None:
        return adapter_format

    resolver = getattr(config, "resolve_profile", None)
    if callable(resolver):
        try:
            loadtime = getattr(resolver("default"), "loadtime", None)
        except (KeyError, ValueError):
            loadtime = None
        if isinstance(loadtime, Mapping):
            configured_format = _reasoning_format(loadtime.get("reasoning_parser"))
            if configured_format is not None:
                return configured_format
    return "qwen3"


def reasoning_starts_in_prompt(prompt: str, reasoning_format: ReasoningFormat) -> bool:
    """Return whether the rendered prompt ends inside a reasoning boundary."""
    opening, closing = _REASONING_BOUNDARIES[reasoning_format]
    opening_at = prompt.rfind(opening)
    if reasoning_format == "gemma4" and prompt.endswith(_GEMMA_CHANNEL_OPEN):
        # Gemma's template can seed only the channel control token in the
        # prompt; the model then generates the ``thought\n`` self-label. Treat
        # that split boundary as open so the reasoning body stays private.
        opening_at = len(prompt) - len(_GEMMA_CHANNEL_OPEN)
    return opening_at > prompt.rfind(closing)


def thinking_mode_is_enabled(config: Any) -> bool:
    """Return whether the resolved tokenizer profile enables private reasoning."""
    tasks = getattr(config, "tasks", None)
    generate = getattr(tasks, "generate", None)
    kwargs = getattr(generate, "chat_template_kwargs", None)
    return isinstance(kwargs, dict) and kwargs.get("enable_thinking") is True


async def collect_generation(
    chunks: AsyncIterator[GenerationChunk],
) -> GenerationResult:
    """Drain an async generation iterator into a :class:`GenerationResult`.

    Convenience for the local-dev ``/v1/generate`` route and unit-test code
    paths that historically consumed the blocking shape. The terminal
    chunk's ``finish_reason`` / token counts are propagated; missing
    counts default to 0. Terminal ``error_code`` / ``error_message``
    (e.g. ``empty_model_output``, which keeps a ``stop``/``length``
    finish_reason) are carried verbatim so buffered consumers can fail
    closed instead of settling as a nominal success (#3104/#3136).
    """
    parts: list[str] = []
    finish_reason: FinishReason = "stop"
    prompt_tokens = 0
    completion_tokens = 0
    error_code: str | None = None
    error_message: str | None = None
    result_selected = False
    try:
        async for chunk in chunks:
            if chunk.text_delta:
                parts.append(chunk.text_delta)
            if chunk.done:
                finish_reason = chunk.finish_reason or "stop"
                if chunk.prompt_tokens is not None:
                    prompt_tokens = chunk.prompt_tokens
                if chunk.completion_tokens is not None:
                    completion_tokens = chunk.completion_tokens
                error_code = chunk.error_code
                error_message = chunk.error_message
                break
        result = GenerationResult(
            text="".join(parts),
            finish_reason=cast("Any", finish_reason),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_code=error_code,
            error_message=error_message,
        )
        result_selected = True
        return result
    finally:
        await aclose_with_error_precedence(
            chunks,
            outcome_selected=result_selected,
            context="buffered generation input iterator",
        )


class GenerationAdapter(ModelAdapter):
    """Abstract base class for generation (text decode) adapters.

    Concrete subclasses must declare a ``spec`` with
    ``outputs=("tokens",)`` and implement :meth:`generate` as an
    ``async def`` generator (uses ``yield``) returning
    :class:`AsyncIterator[GenerationChunk]`. The default ``unload()`` is
    driven by ``spec.unload_fields``.
    """

    spec: ClassVar[AdapterSpec]
    # Decoder-only adapters share one prompt-plus-output context envelope.
    # Encoder-decoder adapters with separately bounded axes must opt in.
    context_length_accounting: ClassVar[Literal["shared", "independent"]] = "shared"
    # The worker-side context guard tokenizes the exact prompt string before
    # dispatch. The default preserves the currently qualified worker behavior;
    # adapters whose engine tokenizer adds boundary tokens must opt in so
    # context validation and terminal usage share one rule.
    prompt_tokenization_add_special_tokens: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only validate classes that declare their own spec.
        if "spec" not in cls.__dict__:
            return
        spec = cls.spec
        if not isinstance(spec, AdapterSpec):
            msg = f"{cls.__name__}.spec must be an AdapterSpec instance"
            raise TypeError(msg)
        if "tokens" not in spec.outputs:
            msg = f"{cls.__name__} (GenerationAdapter) must declare 'tokens' in spec.outputs"
            raise TypeError(msg)
        if cls.generate is GenerationAdapter.generate:
            msg = f"{cls.__name__} declares 'tokens' in outputs but does not implement generate()"
            raise TypeError(msg)

    # -- Properties derived from spec ----------------------------------------

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            inputs=cast("Any", list(self.spec.inputs)),
            outputs=cast("Any", list(self.spec.outputs)),
        )

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    # -- Lifecycle -----------------------------------------------------------

    async def drain_generation(self) -> None:
        """Stop admission and drain adapter-owned generation work.

        The registry awaits this hook before synchronous :meth:`unload`.
        In-process adapters without their own scheduler have nothing to drain.
        Implementations must be safe to call more than once.
        """

    def unload(self) -> None:
        """Unload model state. Iterates ``spec.unload_fields`` and clears each."""
        for attr in self.spec.unload_fields:
            if hasattr(self, attr):
                setattr(self, attr, None)
        self._device = None
        gc.collect()

    # -- Contract ------------------------------------------------------------

    def preflight_generate(
        self,
        parameters: Mapping[str, Any],
        *,
        stream: bool,
    ) -> GenerationPreflightResult | None:
        """Validate one resolved request before admission or engine dispatch.

        ``parameters`` is the exact keyword mapping that will be passed to
        :meth:`generate`; ``stream`` is the public response mode even when the
        backend does not need it as a generation kwarg. Adapters should raise
        one of the typed generation exceptions above for expected refusals.
        """

    def generate_with_preflight(
        self,
        parameters: Mapping[str, Any],
        preflight: GenerationPreflightResult | None,
    ) -> AsyncIterator[GenerationChunk]:
        """Drive one request with its opaque, consume-once preflight result."""
        return _PreflightBoundGenerationIterator(self, parameters, preflight)

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int | None = None,
        grammar: GrammarSpec | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        images: list[ImageInput] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream generation chunks from a prompt.

        Implementations are ``async def`` generators that ``yield``
        :class:`GenerationChunk` objects. The terminal chunk carries
        ``done=True`` and a ``finish_reason``; if the caller drops the
        iterator (``aclose()``) the implementation must propagate the
        cancel to the upstream engine.

        Args:
            prompt: Raw prompt string (chat template applied upstream).
            max_new_tokens: Hard cap on output tokens.
            temperature: Sampling temperature (1.0 = neutral).
            top_p: Nucleus sampling cutoff.
            stop: Optional list of stop strings.
            frequency_penalty: Optional OpenAI-style frequency penalty
                in ``[-2.0, 2.0]``. ``None`` means use the adapter's
                default (typically 0.0). Gateway-validated upstream.
            min_new_tokens: Optional minimum generated-token floor. Adapters
                that cannot enforce it must reject rather than silently ignore it.
            presence_penalty: Optional OpenAI-style presence penalty
                in ``[-2.0, 2.0]``. Same semantics as
                ``frequency_penalty``.
            top_k: Optional non-OpenAI top-k cutoff (integer ``>= 1``).
                ``None`` → top-k disabled (model default).
            repetition_penalty: Optional non-OpenAI multiplicative
                penalty in ``(0.0, 2.0]`` (``1.0`` = no penalty).
                ``None`` → sampler default.
            grammar: Optional structured-output grammar.
            seed: Optional per-request sampling seed. Reproducibility
                semantics are backend-specific.
            logit_bias: Optional ``{token_id_str: bias_float}`` map.
            logprobs: When True, populate ``GenerationChunk.logprobs``
                with per-token log-probabilities.
            top_logprobs: How many alternates per position; only
                consulted when ``logprobs`` is True.
            images: Optional list of wire-format :class:`ImageInput`
                entries for vision-language models. The ``prompt`` is
                expected to already carry the model's image placeholder
                tokens (rendered by the chat template upstream); the
                adapter forwards the image bytes to the engine. ``None``
                or empty for text-only generation. Text-only adapters may
                ignore this argument.

        Yields:
            :class:`GenerationChunk` instances. At least one terminal
            chunk (``done=True``) is yielded for every successful
            generation; the iterator may also raise on transport failure.
        """
        # Declared as a regular ``def`` returning an async iterator (rather
        # than ``async def`` with ``yield``) so ``__init_subclass__`` can
        # detect non-overriding subclasses via ``cls.generate is
        # GenerationAdapter.generate``. Subclasses provide an ``async def``
        # body that ``yield``s.
        raise NotImplementedError
