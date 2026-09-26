"""The bounded word window GLiNER-family adapters read documents through."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from sie_server.adapters._word_window import (
    ATTENTION_BUDGET,
    DEBERTA_MAX_DOCUMENT_SUBWORDS,
    MAX_DOCUMENT_SUBWORDS,
    MAX_WORD_CHARS,
    SUBWORDS_PER_WORD,
    SubwordCounter,
    WindowedSplitter,
    pieces,
    plan_forwards,
    read_window,
    split_word_counter,
    subword_budget,
)

_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")


def words(text: str) -> Iterator[tuple[str, int, int]]:
    for match in _WORD.finditer(text):
        yield match.group(), match.start(), match.end()


def per_char(batch: Sequence[str]) -> list[int]:
    """One subword per character, as DeBERTa tokenizers split a run of accented letters."""
    return [len(word) for word in batch]


def window(text: str, *, max_words: int = 100, max_subwords: int = 1000) -> Any:
    return read_window(text, words(text), max_words=max_words, max_subwords=max_subwords, count_subwords=per_char)


def test_short_words_pass_through() -> None:
    result = window("Priya Raman works at Novartis.")

    assert [word for word, _, _ in result.words] == ["Priya", "Raman", "works", "at", "Novartis", "."]
    assert result.subwords == 26
    assert result.cut is None


def test_a_long_word_is_read_in_pieces_with_their_own_offsets() -> None:
    text = "id " + "x" * (2 * MAX_WORD_CHARS + 10) + " end"

    result = window(text, max_subwords=10_000)

    spans = [(start, end) for _, start, end in result.words]
    assert spans == [(0, 2), (3, 259), (259, 515), (515, 525), (526, 529)]
    assert all(word == text[start:end] for word, start, end in result.words)
    assert result.cut is None


def test_reading_stops_before_the_word_that_passes_the_budget() -> None:
    text = "aaaa bbbb cccc dddd"

    result = window(text, max_subwords=10)

    assert [word for word, _, _ in result.words] == ["aaaa", "bbbb"]
    assert result.subwords == 8
    assert result.cut == 14  # the end of "cccc": text[:14] reads the same window


def test_reading_stops_inside_a_long_word_at_the_budget() -> None:
    text = "go " + "y" * 1000 + " tail"

    result = window(text, max_subwords=600)

    assert [(start, end) for _, start, end in result.words] == [(0, 2), (3, 259), (259, 515)]
    assert result.subwords == 514
    assert result.cut == 1003  # the end of the long word
    assert window(text[: result.cut], max_subwords=600) == result


def test_the_first_word_is_always_read() -> None:
    result = window("x" * 1000, max_subwords=5)

    assert [(start, end) for _, start, end in result.words] == [(0, MAX_WORD_CHARS)]
    assert result.subwords == MAX_WORD_CHARS


def test_reading_stops_after_max_words() -> None:
    text = "a b c d e"

    assert [word for word, _, _ in window(text, max_words=3).words] == ["a", "b", "c"]
    assert window(text, max_words=3).cut == 5
    assert window("a b c", max_words=3).cut is None
    assert window("a b c  ", max_words=3).cut is None


def test_pieces_of_a_word_whose_lowercase_changes_length_follow_the_text() -> None:
    # gliner2 2.x lowercases each word; "\u0130" lowercases to two characters.
    text = "\u0130" * 300
    lowered = [(text.lower(), 0, 300)]

    result = list(pieces(lowered, text))

    assert [(start, end) for _, start, end, _ in result] == [(0, 256), (256, 300)]
    assert [word for word, _, _, _ in result] == [text[0:256].lower(), text[256:300].lower()]
    assert all(word_end == 300 for _, _, _, word_end in result)


def test_the_windowed_splitter_passes_arguments_through() -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def splitter(text: str, *args: Any, **kwargs: Any) -> Iterator[tuple[str, int, int]]:
        calls.append((text, args, kwargs))
        return words(text)

    windowed = WindowedSplitter(splitter, max_words=2, max_subwords=100, count_subwords=per_char)

    assert list(windowed("one two three", lower=True)) == [("one", 0, 3), ("two", 4, 7)]
    assert calls == [("one two three", (), {"lower": True})]


def test_the_counter_counts_each_word_once_and_forgets_the_oldest() -> None:
    counted: list[list[str]] = []

    def count(batch: Sequence[str]) -> list[int]:
        counted.append(list(batch))
        return [len(word) for word in batch]

    counter = SubwordCounter(count, size=2)

    assert counter(["ab", "c", "ab"]) == [2, 1, 2]
    assert counter(["c", "ddd"]) == [1, 3]
    assert counter(["ab"]) == [2]
    assert counted == [["ab", "c"], ["ddd"], ["ab"]]


class SplitWordTokenizer:
    """Two subwords per word, reported through ``word_ids`` as fast tokenizers do."""

    def __call__(self, words: list[str], *, is_split_into_words: bool, add_special_tokens: bool) -> Any:
        assert is_split_into_words
        assert not add_special_tokens
        ids = [index for index, word in enumerate(words) for _ in range(2 if word else 0)]
        return type("Encoding", (), {"word_ids": lambda self: ids})()


def test_the_counter_keeps_no_word_longer_than_a_piece() -> None:
    counter = SubwordCounter(per_char)
    long_word = "q" * (MAX_WORD_CHARS + 1)

    assert counter([long_word, "ab", long_word]) == [MAX_WORD_CHARS + 1, 2, MAX_WORD_CHARS + 1]
    assert counter._cache == {"ab": 2}


def test_split_words_are_counted_as_the_tokenizer_encodes_them() -> None:
    assert split_word_counter(SplitWordTokenizer())(["a", "", "bc"]) == [2, 0, 2]


def test_a_slow_tokenizer_is_counted_word_by_word() -> None:
    class Slow:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("word_ids() is not available when using non-fast tokenizers")

        def tokenize(self, word: str) -> list[str]:
            return list(word)

    assert split_word_counter(Slow())(["ab", "cde"]) == [2, 3]


def test_the_budget_allows_subwords_per_word_up_to_the_encoder_cap() -> None:
    assert subword_budget(512) == SUBWORDS_PER_WORD * 512
    assert subword_budget(512, per_word=4) == 2048
    assert subword_budget(2048) == MAX_DOCUMENT_SUBWORDS
    assert subword_budget(2048, {"model_type": "deberta-v2"}) == DEBERTA_MAX_DOCUMENT_SUBWORDS
    assert subword_budget(384, {"model_type": "deberta-v2"}) == SUBWORDS_PER_WORD * 384
    bert = {"model_type": "bert", "position_embedding_type": "absolute", "max_position_embeddings": 512}
    assert subword_budget(386, bert) == 508
    relative = {"model_type": "deberta-v2", "max_position_embeddings": 512}
    assert subword_budget(384, relative) == SUBWORDS_PER_WORD * 384
    # transformers 5 writes no position_embedding_type for BERT-style encoders.
    assert subword_budget(386, {"model_type": "bert", "max_position_embeddings": 512}) == 508
    assert subword_budget(386, {"model_type": "xlm-roberta", "max_position_embeddings": 514}) == 510
    bert_relative = {"model_type": "bert", "position_embedding_type": "relative_key", "max_position_embeddings": 512}
    assert subword_budget(386, bert_relative) == SUBWORDS_PER_WORD * 386
    rotary = {"model_type": "modernbert", "max_position_embeddings": 8192}
    assert subword_budget(2048, rotary) == MAX_DOCUMENT_SUBWORDS


def test_a_batch_that_fits_runs_as_it_is() -> None:
    assert plan_forwards([600] * 12, rows_per_pass=12) is None
    assert plan_forwards([3000] * 12, rows_per_pass=1) is None
    assert plan_forwards([], rows_per_pass=8) is None


@pytest.mark.parametrize("rows_per_pass", [8, 12])
def test_long_rows_run_in_passes_within_the_budget(rows_per_pass: int) -> None:
    rows = [3100, 40, 2000, 40, 1500, 3100, 40, 40, 900, 40, 40, 4096]

    groups = plan_forwards(rows, rows_per_pass=rows_per_pass)

    assert groups is not None
    assert sorted(index for group in groups for index in group) == list(range(len(rows)))
    for group in groups:
        longest = max(rows[index] for index in group)
        assert len(group) <= rows_per_pass
        assert len(group) == 1 or len(group) * longest**2 <= ATTENTION_BUDGET
    for group in groups:
        lengths = [rows[index] for index in group]
        assert max(lengths) <= 2 * min(lengths)  # short rows are not padded to long ones


def test_a_long_row_does_not_pad_the_short_rows_it_would_fit_with() -> None:
    rows = [2000] + [50] * 11

    groups = plan_forwards(rows, rows_per_pass=8)

    assert groups == [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11], [0]]
