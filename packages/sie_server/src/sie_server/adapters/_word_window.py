"""Bound the words a GLiNER-family model reads by subword tokens as well as by count.

GLiNER, GLiNER2 and GLiREL split a document into words, keep the first
``max_len`` of them, and give the encoder every subword token of each kept
word. ``max_len`` counts words, so it does not bound the encoder input: one
unbroken run of characters (a long number, hash or base64 string, a URL, or
text in a script the tokenizer splits into single characters) is one word of
about as many subwords as it has characters, and the encoder's time and memory
grow with the square of its input.

``read_window`` reads a document's words lazily and returns the part a model
reads:

* A word longer than ``MAX_WORD_CHARS`` characters is read as consecutive
  pieces of at most that many characters. Each piece is a word with the offsets
  of its own characters, so a span found in it maps back to the text.
* Reading stops after ``max_words`` words, or before the first word that would
  take the document past ``max_subwords`` subword tokens. The first word is
  always read, so no document is left empty.

``subword_budget`` allows a number of subwords per word of a model's word
window: ``SUBWORDS_PER_WORD`` by default, which holds ordinary text in the
languages these tokenizers read. Prose, code, CSV and JSON logs run at 1 to 2.4
subwords per word, other European languages and Korean at up to 3.2, and
Chinese or Japanese at up to 5.6 for the multilingual checkpoints. The budget
is capped at ``MAX_DOCUMENT_SUBWORDS`` (``DEBERTA_MAX_DOCUMENT_SUBWORDS`` for
DeBERTa encoders, whose attention materializes score matrices of the square of
the row length), and within the position table of encoders with absolute
positions.

``plan_forwards`` splits a batch into forward passes whose rows times the
square of their longest row stay within ``ATTENTION_BUDGET``, so that the
attention memory of a pass stays bounded for a batch of long rows too.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

# A longer word is read in pieces of this many characters.
MAX_WORD_CHARS = 256
# Subword tokens a document may take per word of the model's word window.
SUBWORDS_PER_WORD = 6
# Most document subwords a model reads...
MAX_DOCUMENT_SUBWORDS = 8192
# ...and for DeBERTa encoders: one such row takes about 2 GiB of attention scores at float16.
DEBERTA_MAX_DOCUMENT_SUBWORDS = 4096
# Rows times the square of the longest row's tokens in one forward pass of a DeBERTa encoder.
ATTENTION_BUDGET = DEBERTA_MAX_DOCUMENT_SUBWORDS * DEBERTA_MAX_DOCUMENT_SUBWORDS
# Tokens an encoder with absolute positions adds around a document ([CLS], [SEP], position offsets).
_SPECIAL_POSITIONS = 4
# Words are counted in blocks, so reading stops at most one block past the window.
_READ_BLOCK = 64
_COUNT_CACHE_SIZE = 16384
_QUADRATIC_ATTENTION_TYPES = frozenset({"deberta", "deberta-v2"})

Word = tuple[str, int, int]
CountSubwords = Callable[[Sequence[str]], Sequence[int]]


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def quadratic_attention(encoder_config: Any) -> bool:
    """Whether the encoder materializes full attention score matrices (DeBERTa)."""
    return _config_value(encoder_config, "model_type") in _QUADRATIC_ATTENTION_TYPES


def absolute_positions(encoder_config: Any) -> int | None:
    """The size of the encoder's absolute position table, or None when positions are relative or rotary."""
    positions = _config_value(encoder_config, "max_position_embeddings")
    if _config_value(encoder_config, "position_embedding_type") != "absolute" or not isinstance(positions, int):
        return None
    return positions


def subword_budget(max_words: int, encoder_config: Any = None, *, per_word: int = SUBWORDS_PER_WORD) -> int:
    """Subword tokens a document may take in a model reading at most ``max_words`` words."""
    cap = DEBERTA_MAX_DOCUMENT_SUBWORDS if quadratic_attention(encoder_config) else MAX_DOCUMENT_SUBWORDS
    positions = absolute_positions(encoder_config)
    if positions is not None:
        cap = min(cap, positions - _SPECIAL_POSITIONS)
    return max(1, min(per_word * max_words, cap))


@dataclass(frozen=True, slots=True)
class Window:
    """The words of a document a model reads."""

    words: list[Word]
    subwords: int
    # ``text[:cut]`` gives the same window: the end of the word holding the
    # first piece not read (or, when the word count ran out, the last piece
    # read). None when every word was read.
    cut: int | None


def pieces(words: Iterable[Word], text: str, max_chars: int = MAX_WORD_CHARS) -> Iterator[tuple[str, int, int, int]]:
    """``(word, start, end, word_end)``: each word, a longer word as pieces of at most ``max_chars`` characters.

    ``word_end`` is the end of the word a piece comes from. A splitter that
    changes a word's length (gliner2 2.x lowercases each word, and ``"\u0130"``
    lowercases to two characters) is cut on the text's characters instead.
    """
    for word, start, end in words:
        if end - start <= max_chars:
            yield word, start, end, end
            continue
        source, lower = word, False
        if len(word) != end - start:
            source = text[start:end]
            lower = word == source.lower()
        for offset in range(0, end - start, max_chars):
            piece = source[offset : offset + max_chars]
            yield (piece.lower() if lower else piece), start + offset, start + offset + len(piece), end


def read_window(
    text: str,
    words: Iterable[Word],
    *,
    max_words: int,
    max_subwords: int,
    count_subwords: CountSubwords,
) -> Window:
    """The words of ``text`` a model reads: see the module docstring.

    ``words`` are ``text``'s words as ``(word, start, end)``, read lazily, and
    ``count_subwords`` gives the subword tokens of each of a list of words.
    """
    source = pieces(words, text)
    kept: list[Word] = []
    subwords = 0
    last_word_end = 0
    while len(kept) < max_words:
        block = list(islice(source, min(_READ_BLOCK, max_words - len(kept))))
        if not block:
            return Window(kept, subwords, None)
        counts = count_subwords([word for word, _, _, _ in block])
        if len(counts) != len(block):
            raise RuntimeError("subword counts do not match the words counted")
        for (word, start, end, word_end), count in zip(block, counts, strict=True):
            if kept and subwords + count > max_subwords:
                return Window(kept, subwords, word_end)
            kept.append((word, start, end))
            subwords += count
            last_word_end = word_end
    if next(source, None) is None:
        return Window(kept, subwords, None)
    return Window(kept, subwords, last_word_end)


class WindowedSplitter:
    """A word splitter that yields only the window of each text a model reads.

    Wraps ``splitter`` (called as ``splitter(text, *args, **kwargs)`` and
    yielding ``(word, start, end)``), so a library that splits and truncates
    a document itself reads the bounded window.
    """

    __slots__ = ("count_subwords", "max_subwords", "max_words", "splitter")

    def __init__(self, splitter: Any, *, max_words: int, max_subwords: int, count_subwords: CountSubwords) -> None:
        self.splitter = splitter
        self.max_words = max_words
        self.max_subwords = max_subwords
        self.count_subwords = count_subwords

    def __call__(self, text: str, *args: Any, **kwargs: Any) -> Iterator[Word]:
        return iter(self.window(text, *args, **kwargs).words)

    def window(self, text: str, *args: Any, **kwargs: Any) -> Window:
        return read_window(
            text,
            self.splitter(text, *args, **kwargs),
            max_words=self.max_words,
            max_subwords=self.max_subwords,
            count_subwords=self.count_subwords,
        )


class SubwordCounter:
    """Subword counts of words, from ``count`` (a list of words at a time), with the recent ones kept."""

    __slots__ = ("_cache", "_count", "_size")

    def __init__(self, count: CountSubwords, size: int = _COUNT_CACHE_SIZE) -> None:
        self._count = count
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._size = size

    def __call__(self, words: Sequence[str]) -> list[int]:
        missing = list(dict.fromkeys(word for word in words if word not in self._cache))
        if missing:
            counts = self._count(missing)
            if len(counts) != len(missing):
                raise RuntimeError("subword counts do not match the words counted")
            for word, count in zip(missing, counts, strict=True):
                self._cache[word] = int(count)
        result = []
        for word in words:
            self._cache.move_to_end(word)
            result.append(self._cache[word])
        while len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return result


def split_word_counter(tokenizer: Any) -> CountSubwords:
    """Subwords per word as ``tokenizer`` encodes a list of words (``is_split_into_words``)."""

    def count(words: Sequence[str]) -> list[int]:
        if not words:
            return []
        try:
            encoding = tokenizer(list(words), is_split_into_words=True, add_special_tokens=False)
            word_ids = encoding.word_ids()
        except (AttributeError, ValueError, TypeError):  # a tokenizer without word alignment
            return [len(tokenizer.tokenize(word)) for word in words]
        per_word = Counter(index for index in word_ids if index is not None)
        return [per_word.get(index, 0) for index in range(len(words))]

    return count


def plan_forwards(
    row_tokens: Sequence[int],
    *,
    rows_per_pass: int,
    budget: int = ATTENTION_BUDGET,
) -> list[list[int]] | None:
    """Row indices to run as separate forward passes, or None when the batch fits as it is.

    The batch fits when every run of ``rows_per_pass`` consecutive rows (the
    passes a library forms) keeps its row count times the square of its
    longest row within ``budget``. Otherwise rows are grouped by length: a
    group adds rows, shortest first, while its count times the square of its
    longest row stays within ``budget``, it holds at most ``rows_per_pass``,
    and no row is more than twice as long as its shortest (so short rows are
    not padded to long ones). A row too long to share a pass runs alone.
    """
    size = max(1, rows_per_pass)
    chunks = [row_tokens[start : start + size] for start in range(0, len(row_tokens), size)]
    if all(len(chunk) * max(chunk) ** 2 <= budget for chunk in chunks):
        return None
    groups: list[list[int]] = []
    current: list[int] = []
    for index in sorted(range(len(row_tokens)), key=lambda row: row_tokens[row]):
        longest = row_tokens[index]
        if current and (
            len(current) >= size
            or (len(current) + 1) * longest * longest > budget
            or longest > 2 * row_tokens[current[0]]
        ):
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups


def bound_gliner_words(model: Any) -> bool:
    """Make a loaded ``gliner`` model read the bounded window of each document.

    ``gliner`` splits a document with ``data_processor.words_splitter`` (in
    ``prepare_inputs``, which inference and the adapters' metering share),
    keeps ``config.max_len`` words, and encodes every subword of each. The
    splitter is wrapped in a ``WindowedSplitter`` counting subwords as the
    model's tokenizer encodes split words. An encoder with absolute positions
    also gets its tokenizer capped at its position table, so that a long label
    prompt is truncated rather than run past the table.

    Returns whether the encoder's attention memory grows with the square of a
    row (DeBERTa), for ``plan_forwards``.
    """
    processor = model.data_processor
    encoder_config = getattr(model.config, "encoder_config", None)
    max_words = int(model.config.max_len)
    tokenizer = processor.transformer_tokenizer
    processor.words_splitter = WindowedSplitter(
        processor.words_splitter,
        max_words=max_words,
        max_subwords=subword_budget(max_words, encoder_config),
        count_subwords=SubwordCounter(split_word_counter(tokenizer)),
    )
    positions = absolute_positions(encoder_config)
    if positions is not None and tokenizer.model_max_length > positions:
        tokenizer.model_max_length = positions
    return quadratic_attention(encoder_config)
