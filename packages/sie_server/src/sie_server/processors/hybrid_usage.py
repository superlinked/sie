"""Count an upstream's generation for a model that also has local weights.

Such a model is counted as if it had run locally. The worker renders the
model's chat template and counts the prompt with the model's tokenizer. This
wrapper counts every generated text the upstream returned, private reasoning
included, with the same tokenizer. The terminal chunk then reports that count,
and the upstream's own counts ride beside it in ``upstream_usage``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from sie_server.adapters._generation_base import GenerationChunk, UpstreamTokenUsage, aclose_with_error_precedence


@dataclass(frozen=True, slots=True)
class HybridCount:
    """The worker's own count of one request that an upstream serves.

    ``count_tokens`` returns how many tokens the model's tokenizer makes of a
    text. ``completion_limit`` is the most completion tokens the request may
    report, over all of its choices.
    """

    prompt_tokens: int
    completion_limit: int
    count_tokens: Callable[[str], Awaitable[int]]


def _only_reasoning(chunk: GenerationChunk) -> bool:
    return (
        not chunk.text_delta
        and not chunk.done
        and chunk.finish_reason is None
        and chunk.tool_call_delta is None
        and chunk.logprobs is None
        and chunk.candidates is None
        and chunk.error_code is None
    )


def _candidate_texts(candidate: dict[str, Any]) -> list[str]:
    texts = [candidate.get("text") or ""]
    for call in candidate.get("tool_calls") or ():
        function = call.get("function") or {}
        texts.extend((function.get("name") or "", function.get("arguments") or ""))
    return texts


async def count_hybrid_usage(
    chunks: AsyncIterator[GenerationChunk], count: HybridCount
) -> AsyncIterator[GenerationChunk]:
    """Report the worker's count on the terminal chunk and drop private reasoning.

    A terminal that carries the upstream's counts is reported with
    ``count.prompt_tokens``, the counted completion, no cached tokens, and the
    upstream's counts in ``upstream_usage``. A completion the upstream counted
    but that returned no text counts as one token, and no completion counts
    above ``count.completion_limit``. Any other terminal passes through.
    """
    texts: dict[int, list[str]] = {}
    outcome_selected = False
    try:
        async for chunk in chunks:
            choice = texts.setdefault(chunk.choice_index, [])
            choice.extend((chunk.reasoning_delta, chunk.text_delta))
            if chunk.tool_call_delta is not None:
                choice.extend((chunk.tool_call_delta.function_name or "", chunk.tool_call_delta.arguments_delta))
            for index, candidate in enumerate(chunk.candidates or ()):
                texts.setdefault(index, []).extend(_candidate_texts(candidate))
            if chunk.reasoning_delta:
                if _only_reasoning(chunk):
                    continue
                chunk = replace(chunk, reasoning_delta="")
            if chunk.done and chunk.prompt_tokens is not None and chunk.completion_tokens is not None:
                completion = 0
                for parts in texts.values():
                    completion += await count.count_tokens("".join(parts))
                if chunk.completion_tokens > 0:
                    completion = max(completion, 1)
                chunk = replace(
                    chunk,
                    prompt_tokens=count.prompt_tokens,
                    completion_tokens=min(completion, count.completion_limit),
                    cached_tokens=None,
                    upstream_usage=UpstreamTokenUsage(
                        prompt_tokens=chunk.prompt_tokens,
                        completion_tokens=chunk.completion_tokens,
                        cached_tokens=chunk.cached_tokens,
                    ),
                )
            outcome_selected = outcome_selected or chunk.done
            yield chunk
    finally:
        await aclose_with_error_precedence(chunks, outcome_selected=outcome_selected, context="hybrid usage count")
