"""The key-term matching rule, in one place.

`score.py` is the only thing in this example that uses it: it imports
`find_term` to decide whether a transcript spells a registered key term. The
rule lives in a module of its own because in superlinked/sie-web the same file
also places the highlights the page draws over each transcript, so a term
counted as found is always a term the page can point at. That second consumer
does not ship here.

**This file does not normalize anything.** The tokens it compares have already
been through OpenAI's Whisper English text normalizer, the standard-library
port in `whisper_normalizer.py`, loaded with the spelling map the run recorded.
That is where "twelve fifty" becomes "1250" and "a hundred percent" becomes
"a 100%"; nothing below spells a number, folds case or drops a filler word.
Both published figures, the 56 of 61 key terms and the pooled 8.1% word error
rate, are counted after that normalization, so reproducing either one means
using the same normalizer and the same map. A different normalizer gives
different figures from the same transcripts.

Against already-normalized tokens, a term counts as found when the transcript
spells its content. Content is the words and the digits. These are not content,
and the rule drops them from both sides before comparing:

- the decimal point inside a number, so "twelve fifty" (normalized 1250) can
  meet "€12.50";
- a currency symbol, so "eleven ninety nine" (normalized 1199) can meet
  "€11.99";
- a leading article, so "a hundred percent" (normalized "a 100%") can meet
  "100%".

What this rule cannot do is separate a spoken "twelve fifty Euros" from a
written "€1250". The normalizer writes the first as "€1250" as well, so both
join to "1250" and `find_term` returns a match for them. No string rule can
tell them apart, which is exactly why `scoring_amendment.json` registers the
amount that was actually spoken and `score.py` rejects the hit by comparing
it. That miss is the amendment's doing, not this file's.

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
