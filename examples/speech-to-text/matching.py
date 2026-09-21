"""The key-term matching rule, in one place.

`run.py` scores with it and `derive_display.py` places the highlights with it,
so the two can never drift: a term counted as found is always a term the page
can point at in the transcript.

A term counts as found when the transcript spells its content. Content is the
words and the digits. These are not content, and the rule drops them from both
sides before comparing:

- the decimal point inside a number, so "twelve fifty" (normalized 1250) can
  meet "€12.50";
- a currency symbol, so "eleven ninety nine" (normalized 1199) can meet
  "€11.99";
- a leading article, so "a hundred percent" (normalized "a 100%") can meet
  "100%".

The amount itself is never dropped, so "twelve fifty Euros" still fails against
"€1250": that is a different number, and `scoring_amendment.json` keeps it a
miss.

Standard library only.
"""

from __future__ import annotations

import re

ARTICLES = frozenset({"a", "an", "the"})
CURRENCY_SYMBOLS = "€£$¢"
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_CURRENCY = {ord(symbol): None for symbol in CURRENCY_SYMBOLS}
# A term never spans more words than this; keeps the span search bounded.
MAX_SPAN_WORDS = 16


def joined(tokens: list[str]) -> str:
    """The content of a normalized token run, as one comparable string."""
    words = list(tokens)
    while words and words[0] in ARTICLES:
        words.pop(0)
    text = _DECIMAL_POINT.sub("", "".join(words))
    return text.translate(_CURRENCY)


def find_term(term_tokens: list[str], tokens: list[str]) -> list[int] | None:
    """[start, end) span of `tokens` whose content equals the term's, else None."""
    target = joined(term_tokens)
    if not target:
        return None
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + MAX_SPAN_WORDS) + 1):
            candidate = joined(tokens[start:end])
            if candidate == target:
                return [start, end]
            if len(candidate) > len(target):
                break
    return None
