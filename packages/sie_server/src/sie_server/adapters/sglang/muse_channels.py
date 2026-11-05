"""Translate Muse Glimmer's channel framing into SIE's reasoning and tool-call forms.

SGLang's native ``/generate`` returns the model's raw wire format. A turn is a
sequence of channels, each addressed to a recipient::

     to=self<|message|>REASONING<|eom|><|start|>assistant to=user<|message|>ANSWER

Tool calls travel on a channel addressed to the tool::

    <|start|>assistant to=NAME<|message|><atem:function_calls>
    <atem:invoke name="NAME">
    <atem:parameter name="KEY">VALUE</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

Every generation surface already handles ``<think>`` and ``<tool_call>`` blocks,
so rewriting the stream once at the adapter boundary keeps the format out of the
rest of the stack: the ``to=self`` channel becomes a ``<think>`` block, tool
channels become ``<tool_call>`` blocks in the Qwen XML form, and the ``to=user``
channel becomes plain text.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from sie_server.adapters._generation_base import GenerationChunk, aclose_with_error_precedence

_ASSISTANT = "<|start|>assistant"
_MESSAGE = "<|message|>"
_CHANNEL_ENDS = ("<|eom|>", "<|eot|>")
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_HEADER_RE = re.compile(r"(?:<\|start\|>assistant)?\s*(?:to=([^\s<]+))?\s*<\|message\|>")
_INVOKE_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)"[^>]*?>(.*?)</atem:invoke>', re.DOTALL)
_PARAMETER_RE = re.compile(r'<atem:parameter\b[^>]*?\bname="([^"]+)"[^>]*?>(.*?)</atem:parameter>', re.DOTALL)


def _may_become_header(text: str) -> bool:
    """Return whether buffered text could still grow into a channel header."""
    if text.startswith(_ASSISTANT):
        text = text[len(_ASSISTANT) :].lstrip()
    elif _ASSISTANT.startswith(text):
        return True
    if text.startswith("to="):
        recipient, angle, marker = text[3:].partition("<")
        if any(char.isspace() for char in recipient.rstrip()):
            return False
        return not angle or _MESSAGE.startswith("<" + marker)
    return "to=".startswith(text) or _MESSAGE.startswith(text)


def _held_back(text: str, markers: tuple[str, ...]) -> int:
    """Return the length of the longest suffix of ``text`` that could become a marker."""
    for length in range(min(len(text), max(map(len, markers)) - 1), 0, -1):
        suffix = text[-length:]
        if any(marker.startswith(suffix) for marker in markers):
            return length
    return 0


def _tool_name(emitted: str) -> str:
    """Undo the doubled namespace (``name.name``) the chat template can provoke."""
    head, _, tail = emitted.partition(".")
    return head if head == tail else emitted


def _tool_call_blocks(body: str) -> str:
    """Rewrite each complete ``atem:invoke`` as a ``<tool_call>`` block."""
    blocks: list[str] = []
    for name, invoke in _INVOKE_RE.findall(body):
        parameters = "".join(
            f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in _PARAMETER_RE.findall(invoke)
        )
        blocks.append(f"<tool_call>\n<function={_tool_name(name)}>\n{parameters}</function>\n</tool_call>")
    return "".join(blocks)


class MuseChannelTranslator:
    """Rewrite one candidate's generated text incrementally."""

    def __init__(self) -> None:
        self._buffer = ""
        # ``self``, ``user`` or ``tool``; ``None`` while between channels.
        self._channel: str | None = None

    def feed(self, text: str) -> str:
        self._buffer += text
        output: list[str] = []
        progressed = True
        while self._buffer and progressed:
            progressed = self._drain_channel(output) if self._channel else self._open_channel(output)
        return "".join(output)

    def finish(self) -> str:
        """Flush at end of stream. Truncated control text never becomes output."""
        buffered, self._buffer = self._buffer, ""
        channel, self._channel = self._channel, None
        if channel == "tool":
            return _tool_call_blocks(buffered)
        if channel == "self":
            return buffered.replace(_THINK_CLOSE, "") + _THINK_CLOSE
        if channel == "user":
            return buffered
        stripped = buffered.lstrip()
        return "" if not stripped or "=" in stripped or stripped.startswith("<") else buffered

    def _open_channel(self, output: list[str]) -> bool:
        stripped = self._buffer.lstrip()
        header = _HEADER_RE.match(stripped)
        if header is not None:
            recipient = header[1] or "user"
            self._channel = recipient if recipient in ("self", "user") else "tool"
            self._buffer = stripped[header.end() :]
            if self._channel == "self":
                output.append(_THINK_OPEN)
            return True
        if _may_become_header(stripped):
            return False
        # A JSON schema or regex response format constrains the output from its
        # first byte, so it arrives without a header.
        self._channel = "user"
        return True

    def _drain_channel(self, output: list[str]) -> bool:
        ends = [(at, len(end)) for end in _CHANNEL_ENDS if (at := self._buffer.find(end)) >= 0]
        if not ends:
            if self._channel == "tool":
                return False  # A call is rewritten whole, once it has closed.
            markers = (*_CHANNEL_ENDS, _THINK_CLOSE) if self._channel == "self" else _CHANNEL_ENDS
            cut = len(self._buffer) - _held_back(self._buffer, markers)
            output.append(self._visible(self._buffer[:cut]))
            self._buffer = self._buffer[cut:]
            return False
        end_at, end_length = min(ends)
        body = self._buffer[:end_at]
        output.append(_tool_call_blocks(body) if self._channel == "tool" else self._visible(body))
        if self._channel == "self":
            output.append(_THINK_CLOSE)
        self._buffer = self._buffer[end_at + end_length :]
        self._channel = None
        return True

    def _visible(self, text: str) -> str:
        # Reasoning that quotes the closing tag must not end the private block early.
        return text.replace(_THINK_CLOSE, "") if self._channel == "self" else text


def translate_muse_text(text: str) -> str:
    """Rewrite a complete generation."""
    translator = MuseChannelTranslator()
    return translator.feed(text) + translator.finish()


def _translate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    text = candidate.get("text")
    if not isinstance(text, str):
        return candidate
    translated = translate_muse_text(text)
    return candidate if translated == text else {**candidate, "text": translated, "logprobs": None}


async def translate_muse_channels(chunks: AsyncIterator[GenerationChunk]) -> AsyncIterator[GenerationChunk]:
    """Rewrite an adapter stream, keeping per-choice state for ``n > 1``."""
    translators: dict[int, MuseChannelTranslator] = {}
    emitted_text = False
    terminal_outcome_selected = False
    try:
        async for chunk in chunks:
            translator = translators.setdefault(chunk.choice_index, MuseChannelTranslator())
            text = translator.feed(chunk.text_delta)
            ending = chunk.done or chunk.finish_reason is not None
            if ending:
                text += translator.finish()
            elif not text:
                continue
            candidates = chunk.candidates
            if candidates is not None:
                candidates = tuple(_translate_candidate(candidate) for candidate in candidates)
            is_first = bool(text) and not emitted_text
            emitted_text = emitted_text or bool(text)
            terminal_outcome_selected = terminal_outcome_selected or chunk.done
            yield replace(
                chunk,
                text_delta=text,
                is_first=is_first,
                logprobs=chunk.logprobs if text == chunk.text_delta else None,
                candidates=candidates,
            )
    finally:
        await aclose_with_error_precedence(
            chunks,
            outcome_selected=terminal_outcome_selected,
            context="muse-channel upstream iterator",
        )
