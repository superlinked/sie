"""The linear-time word splitter yields exactly what gliner2's ``WhitespaceTokenSplitter`` yields.

The reference splitters below are gliner2's own code (1.3.2 ``gliner2.processor``
and 2.0.0 ``gliner2.processing.word_splitter``), copied verbatim; the installed
package's splitter is checked too.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Iterator

import pytest
from sie_server.adapters.gliner2.words import PACKAGE_PATTERN, LinearWordSplitter, linear_equivalent, word_spans


class WhitespaceTokenSplitter:
    """gliner2 2.0.0's splitter: match the text, lowercase each word."""

    __slots__ = ()
    _PATTERN = re.compile(
        r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |\w+(?:[-_]\w+)*
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    )

    def __call__(self, text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
        for m in self._PATTERN.finditer(text):
            token = m.group()
            yield (token.lower() if lower else token), m.start(), m.end()


class Gliner2V1Splitter(WhitespaceTokenSplitter):
    """gliner2 1.3.2's splitter: lowercase the text, then match it."""

    __slots__ = ()

    def __call__(self, text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
        if lower:
            text = text.lower()
        for m in self._PATTERN.finditer(text):
            yield m.group(), m.start(), m.end()


Gliner2V1Splitter.__name__ = "WhitespaceTokenSplitter"

CORPUS = [
    "",
    "   ",
    "My subscription renewed on April 15 for 5,400 yen after the service was already down. Can I get a refund?",
    "Guest in room 1408 says the AC has been out since yesterday; they want to move tonight or leave!",
    ("Das Paket kam besch\u00e4digt an, bitte schicken Sie Ersatz. Ich brauche es bis Freitag, sonst storniere ich!"),
    (
        "\u8bf7\u5e2e\u6211\u53d6\u6d88\u8ba2\u5355\uff0c\u6211\u4e0d\u60f3\u8981\u4e86\u3002\u9000\u6b3e\u4ec0\u4e48\u65f6\u5019\u5230\u8d26\uff1f"
    ),
    ("Visit https://example.com/a?b=c or www.Example.org/x, mail Jane.Doe+tag@Mail.Example.co.uk, ping @support_team!"),
    "user@host a@b.c a@b.co x.y@z.com. ....@ ...@x.io @@@ a@@b.cc @a_b-c foo@bar",
    "state-of-the-art under_score e-mail co-op x--y a_-b -lead trail- __init__ 3.14 1,000,000 v2.0.1",
    "Tabs\tand\nnewlines\r\nand\u00a0no-break\u2003em spaces\u200bzero-width",
    (
        "\u0130stanbul \u03a3\u038a\u03a3\u03a5\u03a6\u039f\u03a3 \u039f\u0394\u039f\u03a3'\u0391 Stra\u00dfe "
        "\u01c5emal \ufb01ne KELVIN\u212a na\u00efve \u017fun"
    ),
    (
        "emoji \U0001f600\U0001f44d\U0001f3fd and flags \U0001f1eb\U0001f1f7, arrows \u2192, math \u2211\u222b,"
        " CJK \u6f22\u5b57\u304b\u306a\u30ab\u30ca\ud55c\uad6d\uc5b4"
    ),
    '{"subject": "Login broken", "body": "Error 500 since the update.", "tags": ["urgent", "web"]}',
    (
        "a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a.a."
        " ..................................................................................................."
        ". %+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_%+-_@xxxxxxxxxx"
    ),
    "HTTP://UPPER.CASE/Path WWW.UPPER.ORG http:// https:/ www. wwwx.org",
]


def _spans(splitter: object, text: str, lower: bool) -> list[tuple[str, int, int]]:
    return list(splitter(text, lower))  # ty:ignore[call-non-callable]


@pytest.mark.parametrize("text", CORPUS)
def test_word_spans_match_the_package_pattern(text: str) -> None:
    assert list(word_spans(text)) == [(m.start(), m.end()) for m in PACKAGE_PATTERN.finditer(text)]


@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize("lower", [True, False])
def test_splitters_match_both_gliner2_versions(text: str, lower: bool) -> None:
    v2 = linear_equivalent(WhitespaceTokenSplitter())
    v1 = linear_equivalent(Gliner2V1Splitter())
    assert v2 is not None
    assert v1 is not None
    assert not v2.lower_text_first
    assert v1.lower_text_first
    assert _spans(v2, text, lower) == _spans(WhitespaceTokenSplitter(), text, lower)
    assert _spans(v1, text, lower) == _spans(Gliner2V1Splitter(), text, lower)


def test_fuzzed_texts_split_identically() -> None:
    alphabet = [*"aZ9._%+-@:/ \t\nwhtps_", "http://", "https://", "www.", ".com", "@x.io", "\u0130", "\u017f"]
    alphabet += ["\u212a", "\u00e9", "\u4e2d", "\uff0c", "--", "\u00a0", "\u200b", "\U0001f600", "\u01c5"]
    alphabet += ["\u00df", "\u03a3", "'"]
    rng = random.Random(0)  # noqa: S311 -- deterministic test data
    for _ in range(20_000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert list(word_spans(text)) == [(m.start(), m.end()) for m in PACKAGE_PATTERN.finditer(text)], text


def test_the_installed_gliner2_splitter_has_a_linear_equivalent() -> None:
    gliner2 = pytest.importorskip("gliner2")
    major = int(gliner2.__version__.split(".")[0])
    if major >= 2:
        from gliner2.processing.word_splitter import (
            WhitespaceTokenSplitter as Installed,  # ty:ignore[unresolved-import]
        )
    else:
        from gliner2.processor import WhitespaceTokenSplitter as Installed  # ty:ignore[unresolved-import]
    installed = Installed()
    linear = linear_equivalent(installed)
    assert linear is not None
    assert linear.lower_text_first == (major < 2)
    for text in CORPUS:
        for lower in (True, False):
            assert _spans(linear, text, lower) == _spans(installed, text, lower)


def test_other_splitters_have_no_linear_equivalent() -> None:
    class CharLevelSplitter:
        _PATTERN = re.compile(r"[A-Za-z0-9@._\-+]+|\S")

        def __call__(self, text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
            yield from ()

    class WhitespaceTokenSplitterLookalike(WhitespaceTokenSplitter):
        _PATTERN = re.compile(r"\S+", re.VERBOSE | re.IGNORECASE)

    WhitespaceTokenSplitterLookalike.__name__ = "WhitespaceTokenSplitter"
    assert linear_equivalent(CharLevelSplitter()) is None
    assert linear_equivalent(WhitespaceTokenSplitterLookalike()) is None
    assert linear_equivalent(lambda text, lower=True: iter(())) is None


@pytest.mark.parametrize(
    "text",
    [
        "." * (2 * 1024 * 1024),
        "a." * (1024 * 1024),
        "%" * (2 * 1024 * 1024 - 2) + "@x",
        "a@" * (1024 * 1024),
        ". " * (1024 * 1024),
    ],
    ids=["dots", "a-dots", "local-run-then-at", "at-chain", "spaced-dots"],
)
def test_splitting_is_linear(text: str) -> None:
    started = time.perf_counter()
    first = [span for _, span in zip(range(2048), word_spans(text), strict=False)]
    assert len(first) == 2048
    assert time.perf_counter() - started < 1.0  # the package's regex takes minutes to hours on these

    started = time.perf_counter()
    count = sum(1 for _ in word_spans(text[: 256 * 1024]))
    assert count > 0
    assert time.perf_counter() - started < 2.0


def test_lower_text_first_offsets_index_the_lowered_text() -> None:
    text = "İstanbul x"
    assert list(LinearWordSplitter(lower_text_first=True)(text)) == list(Gliner2V1Splitter()(text))
    assert list(LinearWordSplitter(lower_text_first=False)(text)) == list(WhitespaceTokenSplitter()(text))
