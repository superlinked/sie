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
Chinese or Japanese at about 4 to 6 for the multilingual checkpoints. The budget
is capped at ``MAX_DOCUMENT_SUBWORDS`` (``DEBERTA_MAX_DOCUMENT_SUBWORDS`` for
DeBERTa encoders, whose attention materializes score matrices of the square of
the row length), and within the position table of encoders with absolute
positions.

``document_windows`` reads a document longer than one window as overlapping
windows, each of which a model reads whole, and ``merge_window_spans`` maps
the spans found in them back to the document and merges them (see those
functions).

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
SUBWORDS_PER_WORD = 8
# Most document subwords a model reads...
MAX_DOCUMENT_SUBWORDS = 8192
# ...and for DeBERTa encoders: one such row takes about 2 GiB of attention scores at float16.
DEBERTA_MAX_DOCUMENT_SUBWORDS = 4096
# Rows times the square of the longest row's tokens in one forward pass of a DeBERTa encoder.
ATTENTION_BUDGET = DEBERTA_MAX_DOCUMENT_SUBWORDS * DEBERTA_MAX_DOCUMENT_SUBWORDS
# Tokens an encoder with absolute positions adds around a document ([CLS], [SEP], position offsets).
_SPECIAL_POSITIONS = 4
# Words a window shares with the one before it. GLiNER models find spans of at
# most 12 words, so an entity cut by one window's edge is whole in its neighbour.
WINDOW_OVERLAP_WORDS = 64
# Most windows read of one document (about 40,000 words at 384 words a window).
# A longer document fails with ``INPUT_TOO_LONG`` rather than being read in part.
MAX_DOCUMENT_WINDOWS = 128
# Characters of a document read to find one window, doubled until the window
# ends this far inside them (or the document ends), so that a window of a long
# document is found without copying the rest of the document each time.
_WINDOW_SLICE_CHARS = 16384
_WINDOW_SLICE_MARGIN = MAX_WORD_CHARS
# Words are counted in blocks, so reading stops at most one block past the window.
_READ_BLOCK = 64
_COUNT_CACHE_SIZE = 16384
_QUADRATIC_ATTENTION_TYPES = frozenset({"deberta", "deberta-v2"})
# Encoders with a learned table of absolute positions. transformers 5 no longer
# writes ``position_embedding_type`` for them, so the model type decides.
_ABSOLUTE_POSITION_TYPES = frozenset(
    {"albert", "bert", "camembert", "distilbert", "electra", "mpnet", "roberta", "xlm-roberta"}
)

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
    if not isinstance(positions, int) or isinstance(positions, bool):
        return None
    kind = _config_value(encoder_config, "position_embedding_type")
    if kind == "absolute" or (kind is None and _config_value(encoder_config, "model_type") in _ABSOLUTE_POSITION_TYPES):
        return positions
    return None


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


@dataclass(frozen=True, slots=True)
class DocumentWindow:
    """One window of a document: ``text[start:end]`` is read whole."""

    start: int
    end: int
    # The end of the last word read.
    read_end: int
    # Words shared with the window before, or None for the first window.
    overlap: int | None


def document_windows(
    text: str,
    splitter: WindowedSplitter,
    *,
    overlap_words: int = WINDOW_OVERLAP_WORDS,
    max_windows: int = MAX_DOCUMENT_WINDOWS,
) -> list[DocumentWindow] | None:
    """The windows ``splitter`` reads ``text`` in, or None when it takes more than ``max_windows``.

    A text read whole is one window of all of it, so a model is given exactly
    the text it was given before. A longer text is read as successive windows,
    each ending where ``splitter`` stops reading (``Window.cut``) and each
    after the first starting at the start of the last ``overlap_words`` words
    of the one before it (at most half of that window's words, so that every
    window reads new words).
    """
    windows: list[DocumentWindow] = []
    start = 0
    overlap: int | None = None
    while True:
        window, rest, end = _window_from(text, start, splitter)
        words = window.words
        if window.cut is None or not words:
            read_end = start + (words[-1][2] if words else len(rest))
            windows.append(DocumentWindow(start, len(text), read_end, overlap))
            return windows
        windows.append(DocumentWindow(start, start + end, start + words[-1][2], overlap))
        if len(windows) >= max_windows:
            return None
        shared = min(overlap_words, len(words) // 2)
        if shared:
            start += words[len(words) - shared][1]
        else:
            # One word read: the next window starts at the first piece not read.
            following = _second_piece_start(splitter, rest) or _second_piece_start(splitter, text[start:])
            if not following:
                raise RuntimeError("a window stopped before the end of a document, but no word follows it")
            start += following
        overlap = shared


def gliner_windows(model: Any, texts: Sequence[str]) -> list[list[DocumentWindow] | None]:
    """The windows a loaded ``gliner`` model reads each text in; None for a text needing too many.

    See ``document_windows``. A model without the bounded word splitter of
    ``bound_gliner_words`` reads each text as one window.
    """
    splitter = getattr(getattr(model, "data_processor", None), "words_splitter", None)
    if not isinstance(splitter, WindowedSplitter):
        return [[DocumentWindow(0, len(text), len(text), None)] for text in texts]
    return [document_windows(text, splitter) for text in texts]


def window_rows(
    texts: Sequence[str], plans: Sequence[list[DocumentWindow] | None]
) -> tuple[list[str], list[int], list[int | None]]:
    """The text of each window read, the item it belongs to, and its overlap with the window before."""
    rows: list[str] = []
    owners: list[int] = []
    overlaps: list[int | None] = []
    for index, (text, windows) in enumerate(zip(texts, plans, strict=True)):
        for window in windows or []:
            rows.append(text[window.start : window.end])
            owners.append(index)
            overlaps.append(window.overlap)
    return rows, owners, overlaps


def window_item_counts(row_counts: Sequence[int] | None, owners: Sequence[int], items: int) -> list[int] | None:
    """Each item's input tokens: the sum over its windows (0 for an item not read)."""
    if row_counts is None:
        return None
    counts = [0] * items
    for owner, count in zip(owners, row_counts, strict=True):
        counts[owner] += count
    return counts


def _second_piece_start(splitter: WindowedSplitter, text: str) -> int | None:
    second = next(islice(pieces(splitter.splitter(text), text), 1, 2), None)
    return None if second is None else second[1]


def _window_from(text: str, start: int, splitter: WindowedSplitter) -> tuple[Window, str, int]:
    """The window ``splitter`` reads of ``text[start:]``, the text it was read from, and where the window's text ends.

    Reads a prefix of ``text[start:]`` that grows until the first piece the
    window does not read ends at least ``_WINDOW_SLICE_MARGIN`` characters
    before the prefix does: a word splitter matching words by pattern then
    reads the same pieces up to it as in the whole text, and so the same
    window. The window's text is ``rest[:cut]`` when that ends inside the
    prefix, or else the prefix itself (a word too long for the prefix is read
    in pieces of which the window reads only the first).
    """
    size = _WINDOW_SLICE_CHARS
    while True:
        rest = text[start : start + size]
        window = splitter.window(rest)
        if start + size >= len(text):
            return window, rest, len(rest) if window.cut is None else window.cut
        if window.cut is not None and window.words:
            if window.cut + _WINDOW_SLICE_MARGIN <= len(rest):
                return window, rest, window.cut
            following = _piece_end_after(splitter, rest, window.words[-1][2])
            if following is not None and following + _WINDOW_SLICE_MARGIN <= len(rest):
                return window, rest, len(rest)
        size *= 2


def _piece_end_after(splitter: WindowedSplitter, text: str, offset: int) -> int | None:
    """The end of the first piece of ``text`` after ``offset`` (the end of a piece), or None when there is none."""
    tail = text[offset:]
    first = next(pieces(splitter.splitter(tail), tail), None)
    return None if first is None else offset + first[2]


def merge_window_spans(
    windows: Sequence[DocumentWindow],
    spans: Sequence[Sequence[dict[str, Any]]],
    text: str,
    *,
    flat_ner: bool,
    multi_label: bool,
) -> list[dict[str, Any]]:
    """The spans of a document from the spans found in each of its windows.

    ``spans[i]`` are the spans (``start``, ``end``, ``label``, ``score`` and
    ``text``) found in ``windows[i]``, with offsets into that window's text.
    One window's spans are returned as they are. Otherwise offsets are moved
    to the document; a span reaching a window's edge that the window next to
    it reads past is dropped when that window found an overlapping span of
    the same label (the span may have been cut short at the edge, and the
    other window reads it whole); the same span found twice keeps its highest
    score; and overlapping spans are resolved as gliner resolves them in one
    window: highest score first, keeping a span that overlaps none kept
    (``flat_ner``) or that overlaps none kept without one holding the other,
    where ``multi_label`` allows one span several labels. Spans are returned
    in document order.
    """
    if len(windows) == 1:
        return list(spans[0])
    moved = [
        [{**span, "start": span["start"] + window.start, "end": span["end"] + window.start} for span in found]
        for window, found in zip(windows, spans, strict=True)
    ]
    best: dict[tuple[int, int, str], dict[str, Any]] = {}
    for index, (window, found) in enumerate(zip(windows, moved, strict=True)):
        for span in found:
            if _cut_at_edge(span, index, window, windows, moved):
                continue
            key = (span["start"], span["end"], span["label"])
            if key not in best or span["score"] > best[key]["score"]:
                best[key] = {**span, "text": text[span["start"] : span["end"]]}
    kept: list[dict[str, Any]] = []
    for span in sorted(best.values(), key=lambda one: -one["score"]):
        if not any(_conflict(span, other, flat_ner=flat_ner, multi_label=multi_label) for other in kept):
            kept.append(span)
    kept.sort(key=lambda one: (one["start"], one["end"]))
    return kept


def _cut_at_edge(
    span: dict[str, Any],
    index: int,
    window: DocumentWindow,
    windows: Sequence[DocumentWindow],
    moved: Sequence[Sequence[dict[str, Any]]],
) -> bool:
    """Whether a neighbouring window reads ``span`` whole and found an overlapping span of its label there."""
    neighbours = []
    if index > 0 and span["start"] <= window.start:
        neighbours.append(index - 1)
    if index + 1 < len(windows) and span["end"] >= window.read_end:
        neighbours.append(index + 1)
    for other in neighbours:
        reader = windows[other]
        if not reader.start <= span["start"] < span["end"] <= reader.read_end:
            continue
        if any(
            found["label"] == span["label"] and found["start"] < span["end"] and span["start"] < found["end"]
            for found in moved[other]
        ):
            return True
    return False


def _conflict(span: dict[str, Any], other: dict[str, Any], *, flat_ner: bool, multi_label: bool) -> bool:
    """Whether two spans conflict under gliner's overlap rule (``greedy_search``), on character offsets."""
    if (span["start"], span["end"]) == (other["start"], other["end"]):
        return not multi_label
    if span["start"] >= other["end"] or other["start"] >= span["end"]:
        return False
    if flat_ner:
        return True
    nested = (span["start"] <= other["start"] and other["end"] <= span["end"]) or (
        other["start"] <= span["start"] and span["end"] <= other["end"]
    )
    return not nested


class SubwordCounter:
    """Subword counts of words, from ``count`` (a list of words at a time), with the recent ones kept.

    Only words of at most ``MAX_WORD_CHARS`` characters (every piece a window
    reads) are kept, so the cache holds at most ``size`` such words.
    """

    __slots__ = ("_cache", "_count", "_size")

    def __init__(self, count: CountSubwords, size: int = _COUNT_CACHE_SIZE) -> None:
        self._count = count
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._size = size

    def __call__(self, words: Sequence[str]) -> list[int]:
        missing = list(dict.fromkeys(word for word in words if word not in self._cache))
        counted: dict[str, int] = {}
        if missing:
            counts = self._count(missing)
            if len(counts) != len(missing):
                raise RuntimeError("subword counts do not match the words counted")
            counted = {word: int(count) for word, count in zip(missing, counts, strict=True)}
        result = []
        for word in words:
            if word in counted:
                count = counted[word]
                if len(word) <= MAX_WORD_CHARS:
                    self._cache[word] = count
            else:
                count = self._cache[word]
                self._cache.move_to_end(word)
            result.append(count)
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
