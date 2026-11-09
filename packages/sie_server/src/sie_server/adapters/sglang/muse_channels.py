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

Only the channel framing produces those tags. The model's own text is data, and
the tags a downstream parser reads are removed from it. Downstream, the reasoning
and tool-call blocks are taken out of the answer, so the answer text of a whole
turn is cleaned as one run, and so is its reasoning. Text on either side of a
block cannot join into a tag. A tool call that the Qwen XML form cannot carry
becomes an empty ``<tool_call>`` block, which the parser treats as malformed.
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
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TEXT_TAGS = (_THINK_OPEN, _THINK_CLOSE, _TOOL_CALL_OPEN)
_ARGUMENT_TAGS = (_THINK_OPEN, _THINK_CLOSE, _TOOL_CALL_CLOSE, "</parameter>")

_HEADER_RE = re.compile(r"(?:<\|start\|>assistant)?\s*(?:to=([^\s<]+))?\s*<\|message\|>")
_INVOKE_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)"[^>]*?>(.*?)</atem:invoke>', re.DOTALL)
_PARAMETER_RE = re.compile(r'<atem:parameter\b[^>]*?\bname="([^"]+)"[^>]*?>(.*?)</atem:parameter>', re.DOTALL)
_MAX_HELD = 1024


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


class _TagRemover:
    """Remove tags from streamed text, including a tag that an earlier removal joins together.

    Every tag starts with its only ``<``, so text since the last ``<`` is the one
    place a tag can still form. Each held run is a proper prefix of a tag, and
    removing a later run exposes the one before it, so the whole held tail stays
    back until it can no longer become a tag. Only text made of nothing but tag
    prefixes keeps it growing; past ``_MAX_HELD`` characters it is released
    without its ``<``, so no later text can complete a tag in it.
    """

    def __init__(self, tags: tuple[str, ...]) -> None:
        self._tags = tags
        self._held: list[str] = []
        self._held_size = 0

    def feed(self, text: str) -> str:
        """Return the text that can no longer become part of a tag."""
        output: list[str] = []
        at = 0
        while at < len(text):
            if not self._held:
                start = text.find("<", at)
                if start < 0:
                    output.append(text[at:])
                    break
                output.append(text[at:start])
                self._held.append("<")
                self._held_size = 1
                at = start + 1
                continue
            char = text[at]
            at += 1
            if char == "<":
                self._held.append(char)
            else:
                run = self._held[-1] + char
                if run in self._tags:
                    self._held_size -= len(self._held.pop())
                    continue
                if not any(tag.startswith(run) for tag in self._tags):
                    output.append(self.flush() + char)
                    continue
                self._held[-1] = run
            self._held_size += 1
            if self._held_size > _MAX_HELD:
                output.append(self.flush().replace("<", ""))
        return "".join(output)

    def flush(self) -> str:
        """Return the held tail; nothing that follows can complete it any more."""
        held = "".join(self._held)
        self._held.clear()
        self._held_size = 0
        return held


def _tool_name(emitted: str) -> str:
    """Undo the doubled namespace (``name.name``) the chat template can provoke."""
    head, _, tail = emitted.partition(".")
    return head if head == tail else emitted


def _function_name_in_place(name: str) -> bool:
    """Return whether the parser reads ``<function=name>`` here: no ``<``, ``>`` or whitespace before trailing space."""
    stem = name.rstrip()
    return bool(stem) and not any(char in "<>" or char.isspace() for char in stem)


def _parameter_name_in_place(key: str) -> bool:
    """Return whether the parser reads ``<parameter=key>`` here: no ``<`` or ``>``, and not only whitespace."""
    return bool(key.strip()) and "<" not in key and ">" not in key


def _tool_call_blocks(body: str) -> str:
    """Rewrite each complete ``atem:invoke`` as a ``<tool_call>`` block.

    A call the Qwen XML form cannot carry becomes an empty block, which the parser
    treats as malformed: the parser would not find its name or a key in place, or
    a value holds a tag that a parser reads there.
    """
    blocks: list[str] = []
    for emitted, invoke in _INVOKE_RE.findall(body):
        name, parameters = _tool_name(emitted), _PARAMETER_RE.findall(invoke)
        carried = _function_name_in_place(name) and all(
            _parameter_name_in_place(key) and not any(tag in value for tag in _ARGUMENT_TAGS)
            for key, value in parameters
        )
        if not carried:
            blocks.append(f"{_TOOL_CALL_OPEN}\n{_TOOL_CALL_CLOSE}")
            continue
        arguments = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in parameters)
        blocks.append(f"{_TOOL_CALL_OPEN}\n<function={name}>\n{arguments}</function>\n{_TOOL_CALL_CLOSE}")
    return "".join(blocks)


class MuseChannelTranslator:
    """Rewrite one candidate's generated text incrementally."""

    def __init__(self) -> None:
        self._buffer = ""
        # ``self``, ``user`` or ``tool``; ``None`` while between channels.
        self._channel: str | None = None
        self._answer = _TagRemover(_TEXT_TAGS)
        self._reasoning = _TagRemover(_TEXT_TAGS)

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
            output = _tool_call_blocks(buffered)
        elif channel == "self":
            output = self._reasoning.feed(buffered)
        elif channel == "user":
            output = self._answer.feed(buffered)
        else:
            stripped = buffered.lstrip()
            ignored = not stripped or "=" in stripped or stripped.startswith("<")
            output = "" if ignored else self._answer.feed(buffered)
        reasoning = self._reasoning.flush()
        if channel == "self":
            output += reasoning + _THINK_CLOSE
        elif reasoning:
            output += _THINK_OPEN + reasoning + _THINK_CLOSE
        return output + self._answer.flush()

    def _text(self) -> _TagRemover:
        """Return the remover for the open channel: the turn's reasoning or its answer."""
        return self._reasoning if self._channel == "self" else self._answer

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
            cut = len(self._buffer) - _held_back(self._buffer, _CHANNEL_ENDS)
            output.append(self._text().feed(self._buffer[:cut]))
            self._buffer = self._buffer[cut:]
            return False
        end_at, end_length = min(ends)
        body = self._buffer[:end_at]
        output.append(_tool_call_blocks(body) if self._channel == "tool" else self._text().feed(body))
        if self._channel == "self":
            output.append(_THINK_CLOSE)
        self._buffer = self._buffer[end_at + end_length :]
        self._channel = None
        return True


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
