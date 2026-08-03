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

import gc
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput

logger = logging.getLogger(__name__)


# Finish reason values surfaced to gateway / client. ``cancelled`` lands when
# the worker observed a cancel signal mid-stream (§4.4.2). ``error`` lands
# when the upstream engine raised; concrete adapters may also produce
# ``length`` (max_new_tokens reached) and ``stop`` (natural EOS / stop string).
# ``tool_calls`` is the OpenAI-compatible terminator emitted by the
# tool-call parser when one or more ``<tool_call>...</tool_call>`` blocks
# were consumed before the underlying model stopped.
FinishReason = Literal["stop", "length", "cancelled", "error", "tool_calls"]


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
    this aggregate is built from it.
    """

    text: str
    finish_reason: Literal["stop", "length", "error", "cancelled"]
    prompt_tokens: int
    completion_tokens: int


async def collect_generation(
    chunks: AsyncIterator[GenerationChunk],
) -> GenerationResult:
    """Drain an async generation iterator into a :class:`GenerationResult`.

    Convenience for the local-dev ``/v1/generate`` route and unit-test code
    paths that historically consumed the blocking shape. The terminal
    chunk's ``finish_reason`` / token counts are propagated; missing
    counts default to 0.
    """
    parts: list[str] = []
    finish_reason: FinishReason = "stop"
    prompt_tokens = 0
    completion_tokens = 0
    async for chunk in chunks:
        if chunk.text_delta:
            parts.append(chunk.text_delta)
        if chunk.done:
            finish_reason = chunk.finish_reason or "stop"
            if chunk.prompt_tokens is not None:
                prompt_tokens = chunk.prompt_tokens
            if chunk.completion_tokens is not None:
                completion_tokens = chunk.completion_tokens
            break
    return GenerationResult(
        text="".join(parts),
        finish_reason=cast("Any", finish_reason),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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

    def unload(self) -> None:
        """Unload model state. Iterates ``spec.unload_fields`` and clears each."""
        for attr in self.spec.unload_fields:
            if hasattr(self, attr):
                setattr(self, attr, None)
        self._device = None
        gc.collect()

    # -- Contract ------------------------------------------------------------

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
