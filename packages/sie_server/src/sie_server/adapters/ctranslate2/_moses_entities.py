"""Undo the Moses tokenizer's entity escaping in translation output.

The Moses tokenizer escapes ``& < > ' " [ ] |`` as entities and splits them
off the surrounding words with spaces. MADLAD-400 returns text in that form
for some target languages, so a French translation can come back as
``d &apos; Android`` instead of ``d'Android``.

The repair is deliberately conservative:

* An entity is only unescaped when the input text does not contain it
  literally. A request that really translates ``&amp;`` keeps it.
* Spacing is only repaired where the form is unambiguous: French elision,
  when the prompt targets French (``<2fr>``, ``<2fr_CA>``), so
  ``l &apos; offre`` becomes ``l'offre``; and balanced, space-delimited
  ``&quot;`` pairs (``&quot; Bonjour &quot;`` becomes ``"Bonjour"``).
  Everything else is unescaped in place and keeps its spaces, so
  ``R &amp; D`` becomes ``R & D``.
"""

from __future__ import annotations

import re

_ENTITIES = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&apos;": "'",
    "&quot;": '"',
    "&#91;": "[",
    "&#93;": "]",
    "&#124;": "|",
}
# One pass over the text, so an escaped entity such as ``&amp;lt;`` is
# unescaped once, to ``&lt;``, and not again to ``<``.
_ENTITY = re.compile("|".join(re.escape(entity) for entity in _ENTITIES))

_APOS = "&apos;"
_QUOT = "&quot;"

# French elided forms followed by an escaped apostrophe and the next word:
# c', d', j', l', m', n', s', t', the qu' family (qu', jusqu', lorsqu',
# quelqu'...) and aujourd'hui. Only the elided word and the word after it
# decide the join, so a single-quoted span such as ``dit &apos; oui &apos;``
# is left spaced, while a quoted single letter that is itself an elided form
# (``&apos; l &apos; est``) is joined to the word after it. The join only runs
# when the prompt targets French.
_ELISION = re.compile(
    r"(?<!\w)([cdjlmnst]|[^\W\d_]*qu|aujourd) &apos; (?=[^\W\d_])",
    re.IGNORECASE,
)
# MADLAD-400's target tag for French and its regional forms.
_FRENCH_TARGET = re.compile(r"\s*<2fr(?:_[A-Za-z]+)?>")


def unescape_moses_entities(text: str, source: str) -> str:
    """Return ``text`` with Moses entities unescaped and their spacing repaired.

    Args:
        text: Decoded model output.
        source: The input text the output was generated from, target tag
            included. Entities that occur literally in it are kept as they
            are, and French elision is only repaired for a French target tag.
    """
    escaped = frozenset(entity for entity in _ENTITIES if entity in text and entity not in source)
    if not escaped:
        return text
    if _APOS in escaped and _FRENCH_TARGET.match(source):
        text = _ELISION.sub(r"\1'", text)
    if _QUOT in escaped:
        text = _attach_quote_pairs(text)
    return _ENTITY.sub(lambda match: _ENTITIES[match[0]] if match[0] in escaped else match[0], text)


def _attach_quote_pairs(text: str) -> str:
    """Attach balanced ``&quot;`` pairs to the words they enclose.

    The tokenizer turns ``"word"`` into ``&quot; word &quot;``. The output is
    only respaced when every escaped quote can be paired in order, each pair
    encloses text with exactly one space on either side, an opening quote
    starts the text or follows whitespace and a closing quote does not precede
    a word character. A literal ``"`` elsewhere in the output, an odd count or any other
    shape leaves the spacing as it is.
    """
    if '"' in text:
        return text
    parts = text.split(_QUOT)
    if len(parts) % 2 == 0:
        return text
    for index in range(1, len(parts), 2):
        before, inner, after = parts[index - 1], parts[index], parts[index + 1]
        if (before or index > 1) and not before[-1:].isspace():
            return text
        if len(inner) < 3 or inner[0] != " " or inner[-1] != " " or inner[1].isspace() or inner[-2].isspace():
            return text
        if after and (after[0].isalnum() or after[0] == "_"):
            return text
    pieces = [parts[0]]
    for index in range(1, len(parts), 2):
        pieces.extend(('"', parts[index][1:-1], '"', parts[index + 1]))
    return "".join(pieces)
