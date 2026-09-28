"""Terminal output validation for strict structured-output grammars.

The engine's grammar backend constrains decoding. A request that also sets
``strict: true`` requires the completed output to satisfy the grammar: a
``json_schema`` output must parse as JSON and validate against the schema, and
a ``regex`` output must match the pattern in full. No verifier exists for the
backend's EBNF dialects, so ingress rejects ``strict: true`` on ``ebnf``
grammars instead of accepting a guarantee it cannot check.

Validation runs on the visible text of every choice that stopped naturally
without a tool call. A violation replaces the terminal with a typed
``MODEL_OUTPUT_PARSE_ERROR`` terminal, which every generation ingress already
surfaces as an error. Choices that finished with ``length`` or a tool call, and
``cancelled`` or ``error`` terminals, are not verified.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache
from typing import Any

import regex
from jsonschema import Draft202012Validator, ValidationError, validators
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource

from sie_server.adapters._generation_base import GenerationChunk, aclose_with_error_precedence
from sie_server.types.grammar import GrammarSpec

logger = logging.getLogger(__name__)

MODEL_OUTPUT_PARSE_ERROR = "MODEL_OUTPUT_PARSE_ERROR"
VERIFIABLE_GRAMMAR_KINDS = frozenset({"json_schema", "regex"})

# Upper bound on the wall time spent evaluating caller-supplied patterns for
# one output. Patterns run on the backtracking ``regex`` engine, whose timeout
# keeps a pathological pattern from pinning the worker.
_PATTERN_BUDGET_S = 2.0
_MAX_PATH_CHARS = 200
_UNVERIFIABLE_MESSAGE = "generated output could not be verified against the requested grammar"
# A shared ``tool_calls`` terminal can still close choices that stopped
# without a tool call; per-choice eligibility is decided in ``_terminal_outputs``.
_VERIFIED_TERMINAL_REASONS = frozenset({"stop", "tool_calls"})

_pattern_deadline: ContextVar[float] = ContextVar("strict_grammar_pattern_deadline")


def _refuse_retrieval(uri: str) -> Resource:
    raise NoSuchResource(ref=uri)


# Schemas are caller-supplied: only references inside the schema resolve, and
# the verifier never fetches a remote document.
_LOCAL_REFERENCES = Registry(retrieve=_refuse_retrieval)


def requires_output_validation(grammar: GrammarSpec | None) -> bool:
    """Return whether ``grammar`` asks for a verified output."""
    return grammar is not None and grammar.strict is True and grammar.kind in VERIFIABLE_GRAMMAR_KINDS


def output_violation(grammar: GrammarSpec, text: str) -> str | None:
    """Return a client-safe violation message, or ``None`` when ``text`` conforms."""
    token = _pattern_deadline.set(time.monotonic() + _PATTERN_BUDGET_S)
    try:
        if grammar.kind == "json_schema" and isinstance(grammar.value, dict):
            return _json_schema_violation(grammar.value, text)
        if grammar.kind == "regex" and isinstance(grammar.value, str):
            return _regex_violation(grammar.value, text)
        return _UNVERIFIABLE_MESSAGE
    except Exception:  # noqa: BLE001 - a grammar the verifier cannot evaluate fails closed
        logger.warning("strict grammar verification failed for kind=%s", grammar.kind, exc_info=True)
        return _UNVERIFIABLE_MESSAGE
    finally:
        _pattern_deadline.reset(token)


def first_output_violation(grammar: GrammarSpec, outputs: Iterable[str]) -> str | None:
    """Return the first violation across ``outputs`` (one per choice)."""
    for text in outputs:
        violation = output_violation(grammar, text)
        if violation is not None:
            return violation
    return None


def enforce_strict_grammar(
    chunks: AsyncIterator[GenerationChunk],
    grammar: GrammarSpec | None,
) -> AsyncIterator[GenerationChunk]:
    """Return ``chunks`` with strict-grammar terminal validation applied.

    Apply this to the visible stream, after reasoning suppression and
    tool-call parsing. Non-strict grammars return ``chunks`` unchanged.
    """
    if grammar is None or not requires_output_validation(grammar):
        return chunks
    return _validated_chunks(chunks, grammar)


async def _validated_chunks(
    chunks: AsyncIterator[GenerationChunk],
    grammar: GrammarSpec,
) -> AsyncIterator[GenerationChunk]:
    texts: dict[int, list[str]] = {}
    choice_finish_reasons: dict[int, str] = {}
    tool_call_choices: set[int] = set()
    terminal_outcome_selected = False
    try:
        async for chunk in chunks:
            if chunk.text_delta:
                texts.setdefault(chunk.choice_index, []).append(chunk.text_delta)
            if chunk.tool_call_delta is not None:
                tool_call_choices.add(chunk.choice_index)
            if not chunk.done and chunk.finish_reason is not None:
                choice_finish_reasons[chunk.choice_index] = chunk.finish_reason
            if chunk.done:
                if chunk.error_code is None and (chunk.finish_reason or "stop") in _VERIFIED_TERMINAL_REASONS:
                    outputs = _terminal_outputs(chunk, texts, choice_finish_reasons, tool_call_choices)
                    violation = await asyncio.to_thread(first_output_violation, grammar, outputs)
                    if violation is not None:
                        chunk = replace(
                            chunk,
                            finish_reason="error",
                            error_code=MODEL_OUTPUT_PARSE_ERROR,
                            error_message=violation,
                        )
                terminal_outcome_selected = True
            yield chunk
        if not terminal_outcome_selected:
            # Consumers settle an iterator that ends without a terminal as a
            # natural stop, so the accumulated output is verified here.
            implicit_stop = GenerationChunk(text_delta="", done=True)
            outputs = _terminal_outputs(implicit_stop, texts, choice_finish_reasons, tool_call_choices)
            violation = await asyncio.to_thread(first_output_violation, grammar, outputs)
            if violation is not None:
                terminal_outcome_selected = True
                yield replace(
                    implicit_stop,
                    finish_reason="error",
                    error_code=MODEL_OUTPUT_PARSE_ERROR,
                    error_message=violation,
                )
    finally:
        await aclose_with_error_precedence(
            chunks,
            outcome_selected=terminal_outcome_selected,
            context="strict grammar upstream iterator",
        )


def _terminal_outputs(
    chunk: GenerationChunk,
    texts: Mapping[int, Sequence[str]],
    choice_finish_reasons: Mapping[int, str],
    tool_call_choices: set[int],
) -> list[str]:
    """Return the text of every choice that stopped naturally without a tool call.

    Non-streamed candidates carry their own finish reasons on the terminal.
    Streamed choices of an ``n > 1`` request finish on earlier per-choice
    chunks, before the shared terminal.
    """
    if chunk.candidates:
        return [
            text if isinstance(text := candidate.get("text"), str) else ""
            for candidate in chunk.candidates
            if candidate.get("finish_reason") in (None, "stop") and not candidate.get("tool_calls")
        ]
    choices = sorted(set(texts) | set(choice_finish_reasons) | tool_call_choices) or [chunk.choice_index]
    return [
        "".join(texts.get(index, ()))
        for index in choices
        if index not in tool_call_choices and choice_finish_reasons.get(index, "stop") == "stop"
    ]


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"{value} is not valid JSON")


def _json_schema_violation(schema: dict[str, Any], text: str) -> str | None:
    try:
        instance = json.loads(text, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError):
        return "generated output is not valid JSON"
    validator_class = _bounded_validator_class(validators.validator_for(schema, default=Draft202012Validator))
    error = next(validator_class(schema, registry=_LOCAL_REFERENCES).iter_errors(instance), None)
    if error is None:
        return None
    path = error.json_path
    if len(path) > _MAX_PATH_CHARS:
        path = path[: _MAX_PATH_CHARS - 3] + "..."
    return f"generated output does not match the requested JSON schema at {path} ('{error.validator}' keyword)"


def _regex_violation(pattern: str, text: str) -> str | None:
    if regex.fullmatch(pattern, text, timeout=_remaining_pattern_budget()) is None:
        return "generated output does not match the requested regex"
    return None


def _remaining_pattern_budget() -> float:
    remaining = _pattern_deadline.get() - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("strict grammar pattern budget exhausted")
    return remaining


def _pattern_search(pattern: str, text: str) -> bool:
    return regex.search(pattern, text, timeout=_remaining_pattern_budget()) is not None


def _pattern(validator: Any, pattern: str, instance: Any, schema: Mapping[str, Any]) -> Iterator[ValidationError]:
    if validator.is_type(instance, "string") and not _pattern_search(pattern, instance):
        yield ValidationError("string does not match pattern")


def _pattern_properties(
    validator: Any,
    pattern_properties: Mapping[str, Any],
    instance: Any,
    schema: Mapping[str, Any],
) -> Iterator[ValidationError]:
    if not validator.is_type(instance, "object"):
        return
    for pattern, subschema in pattern_properties.items():
        for key, value in instance.items():
            if _pattern_search(pattern, key):
                yield from validator.descend(value, subschema, path=key, schema_path=pattern)


def _additional_properties(
    validator: Any,
    additional: Any,
    instance: Any,
    schema: Mapping[str, Any],
) -> Iterator[ValidationError]:
    if not validator.is_type(instance, "object"):
        return
    properties = schema.get("properties", {})
    patterns = tuple(schema.get("patternProperties", {}))
    extras = [
        key
        for key in instance
        if key not in properties and not any(_pattern_search(pattern, key) for pattern in patterns)
    ]
    if validator.is_type(additional, "object"):
        for extra in extras:
            yield from validator.descend(instance[extra], additional, path=extra)
    elif not additional and extras:
        yield ValidationError("additional properties are not allowed")


@lru_cache(maxsize=8)
def _bounded_validator_class(base: type) -> Any:
    return validators.extend(
        base,
        {
            "pattern": _pattern,
            "patternProperties": _pattern_properties,
            "additionalProperties": _additional_properties,
        },
    )
