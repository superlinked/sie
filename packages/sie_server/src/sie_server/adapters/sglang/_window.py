"""Recognise the SGLang engine refusing a request that does not fit the model's window.

The engine reports this as a plain message, in an event, a finish reason or a
400 body depending on where it noticed. A retry of such a request can never
succeed, so it must reach the caller as a client error rather than as an engine
failure. Only the exact texts below qualify: the engine's other messages can
carry request or engine details and stay generic, and the caller only ever sees
the fixed text built here.
"""

from __future__ import annotations

import re

from sie_server.adapters.errors import InputTooLongError
from sie_server.types.inputs import InvalidInputError

# Identical in the SGLang 0.5.10 and 0.5.20 the bundles pin: the first two come from
# its tokenizer process, the last two from its scheduler, which also counts image
# tokens and keeps a small margin.
_WINDOW_REFUSALS = tuple(
    re.compile(pattern)
    for pattern in (
        r"The input \((?P<tokens>\d+) tokens\) is longer than the model's context length \((?P<limit>\d+) tokens\)\.",
        r"Requested token count exceeds the model's maximum context length of (?P<limit>\d+) tokens\. "
        r"You requested a total of (?P<tokens>\d+) tokens: \d+ tokens from the input messages and \d+ tokens "
        r"for the completion\. Please reduce the number of tokens in the input messages or the completion "
        r"to fit within the limit\.",
        r"Input length \((?P<tokens>\d+) tokens\) exceeds the maximum allowed length \((?P<limit>\d+) tokens\)\. "
        r"Use a shorter input or enable --allow-auto-truncate\.",
        r"Multimodal prompt is too long after expanding multimodal tokens\. "
        r"After expanding len\(req\.origin_input_ids_unpadded\)=\d+ => (?P<tokens>\d+) >= (?P<limit>\d+)\.",
    )
)


class SGLangInputTooLongError(InputTooLongError, InvalidInputError):
    """``INPUT_TOO_LONG`` that still lets a shared batch isolate the request at fault."""


def window_refusal_message(message: object) -> str | None:
    """The text to show the caller when ``message`` is a window refusal, else ``None``."""
    if not isinstance(message, str):
        return None
    for pattern in _WINDOW_REFUSALS:
        if (match := pattern.fullmatch(message)) is not None:
            return f"The request needs {match['tokens']} tokens, at or over this model's limit of {match['limit']}"
    return None
