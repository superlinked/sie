r"""A linear-time equivalent of gliner2's whitespace word splitter.

gliner2 splits a document into words with one regular expression
(``WhitespaceTokenSplitter``), matched case-insensitively::

    https?://[^\s]+ | www\.[^\s]+          a URL
    [a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}    an e-mail address
    @[a-z0-9_]+                              a mention
    \w+(?:[-_]\w+)*                          a word
    \S                                      any other character

The e-mail alternative is tried at every word. Its local part runs over every
e-mail character ahead of the word before failing on the missing ``@``, so a run
of n such characters (``"...."``, ``"a.a.a."``) costs O(n^2): splitting 64 KiB
of ``"."`` takes about 17 seconds, on the thread that serves every request.

``word_spans`` yields the same matches in linear time. It tries the alternatives
in the same order and decides the e-mail alternative without rescanning: a local
part can only end at the first non-e-mail character after the word start, which
is the end of the maximal run of e-mail characters holding it, so each such run
is scanned once, and the domain after an ``@`` is matched once.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

# gliner2's pattern (WhitespaceTokenSplitter._PATTERN), with its flags.
PACKAGE_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |\w+(?:[-_]\w+)*
        |\S""",
    re.VERBOSE | re.IGNORECASE,
)

_START = re.compile(r"\S")
_URL = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_LOCAL = re.compile(r"[a-z0-9._%+-]+", re.IGNORECASE)
_DOMAIN = re.compile(r"[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)
_OTHER = re.compile(r"@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S", re.IGNORECASE)

WordSplitter = Callable[..., Iterator[tuple[str, int, int]]]


def word_spans(text: str) -> Iterator[tuple[int, int]]:
    """``(start, end)`` of each match of ``PACKAGE_PATTERN.finditer(text)``, in linear time."""
    size = len(text)
    position = 0
    local_end = 0  # end of the run of e-mail characters holding the current word start
    email_end: int | None = None  # where an e-mail address starting in that run ends, if one does
    while (start := _START.search(text, position)) is not None:
        begin = start.start()
        if (url := _URL.match(text, begin)) is not None:
            end = url.end()
        else:
            if begin >= local_end:
                local = _LOCAL.match(text, begin)
                local_end = begin if local is None else local.end()
                email_end = None
                if local_end > begin and local_end < size and text[local_end] == "@":
                    domain = _DOMAIN.match(text, local_end + 1)
                    email_end = None if domain is None else domain.end()
            if begin < local_end and email_end is not None:
                end = email_end
            else:
                other = _OTHER.match(text, begin)
                end = other.end() if other is not None else begin + 1
        yield begin, end
        position = end


class LinearWordSplitter:
    """Drop-in for gliner2's ``WhitespaceTokenSplitter``: the same words, in linear time.

    gliner2 1.x lowercases the text and then splits it; gliner2 2.x splits the
    text as given and lowercases each word (offsets then always index the text).
    ``lower_text_first`` selects the 1.x behavior.
    """

    __slots__ = ("lower_text_first",)

    def __init__(self, *, lower_text_first: bool) -> None:
        self.lower_text_first = lower_text_first

    def __call__(self, text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
        if lower and self.lower_text_first:
            text = text.lower()
            for begin, end in word_spans(text):
                yield text[begin:end], begin, end
            return
        for begin, end in word_spans(text):
            word = text[begin:end]
            yield (word.lower() if lower else word), begin, end


# Texts on which the two lowercasing behaviors (and a different pattern) disagree.
_PROBES = (
    "\u0130stanbul and \u03a3\u038a\u03a3\u03a5\u03a6\u039f\u03a3 met at https://Example.com/x?y=1, "
    "mail A.B@Example.CO.uk @Team_1 x-y_z 3.14 ...@ a@b",
    "Stra\u00dfe \u01c5emal \ufb01ne KELVIN\u212a na\u00efve",
)


def linear_equivalent(splitter: Any) -> LinearWordSplitter | None:
    """A ``LinearWordSplitter`` that yields exactly what gliner2's ``splitter`` yields, or None.

    None unless ``splitter`` is gliner2's ``WhitespaceTokenSplitter`` with the
    pattern above, and one of the two lowercasing behaviors reproduces it on
    probe texts.
    """
    pattern = getattr(type(splitter), "_PATTERN", None)
    if (
        type(splitter).__name__ != "WhitespaceTokenSplitter"
        or not isinstance(pattern, re.Pattern)
        or _normalized(pattern.pattern) != _normalized(PACKAGE_PATTERN.pattern)
        or pattern.flags != PACKAGE_PATTERN.flags
    ):
        return None
    for lower_text_first in (False, True):
        candidate = LinearWordSplitter(lower_text_first=lower_text_first)
        if all(
            list(candidate(probe, lower)) == list(splitter(probe, lower))
            for probe in _PROBES
            for lower in (True, False)
        ):
            return candidate
    return None


def _normalized(pattern: str) -> str:
    """A verbose pattern without its layout whitespace (this pattern has no escaped spaces)."""
    return "".join(pattern.split())
