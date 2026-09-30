"""GLiNER and GLiNER bi-encoder adapters read a long document whole, in overlapping windows.

gliner reads at most ``max_len`` words of a text. A longer document is read as
overlapping windows (``_word_window.document_windows``), and the spans found in
each are mapped back to the document and merged. These tests use gliner's real
processor and word splitter with the one-subword-per-character tokenizer of
``test_gliner_long_words`` and a model window of 64 words; the stand-in model
finds names in the words it reads.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

import pytest
from sie_server.adapters._word_window import MAX_DOCUMENT_WINDOWS, WINDOW_OVERLAP_WORDS, gliner_windows
from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.adapters.gliner_bi import GLiNERBiAdapter
from sie_server.types.inputs import Item
from sie_server.types.responses import ErrorCode

from .test_gliner_long_words import DEBERTA, MAX_LEN, FakeGLiNER, load

NAMES = ("Priya Raman", "Tomas Novak", "Ines Duarte")


class NameFinder(FakeGLiNER):
    """Finds each name of ``NAMES`` as a person in the words gliner reads of a text.

    A name read whole scores 0.9 in the first half of the words read and 0.6
    in the second. A name whose first word is the last word read is found cut
    short, as a model may find it at a window's edge, with a score of 0.99.
    """

    def inference(self, texts: list[str], labels: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(list(texts))
        return [self._find(text) for text in texts]

    def batch_predict_with_embeds(self, texts: list[str], *args: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        return self.inference(texts, [])

    def _find(self, text: str) -> list[dict[str, Any]]:
        words = list(self.data_processor.words_splitter(text))
        read_end = words[-1][2]
        found = []
        for name in NAMES:
            for match in re.finditer(re.escape(name), text):
                if match.end() <= read_end:
                    score = 0.9 if match.start() < read_end / 2 else 0.6
                    found.append(span(text, match.start(), match.end(), score))
            first = name.split()[0]
            if text[words[-1][1] : read_end] == first and text[words[-1][1] :].startswith(name):
                found.append(span(text, words[-1][1], read_end, 0.99))
        return sorted(found, key=lambda one: one["start"])


def span(text: str, start: int, end: int, score: float, label: str = "person") -> dict[str, Any]:
    return {"text": text[start:end], "label": label, "score": score, "start": start, "end": end}


def filler(count: int, first: int = 0) -> str:
    return " ".join(f"w{index}" for index in range(first, first + count))


def tokens(text: str) -> int:
    """Tokens of a document read whole: one per character but whitespace, with [CLS] and [SEP]."""
    return sum(not char.isspace() for char in text) + 2


def found(output: Any, item: int = 0) -> list[tuple[str, int, int]]:
    return [(entity["text"], entity["start"], entity["end"]) for entity in output.entities[item]]


@pytest.fixture(params=[GLiNERAdapter, GLiNERBiAdapter], ids=["gliner", "gliner-bi"])
def adapter_and_model(request: pytest.FixtureRequest) -> tuple[Any, NameFinder]:
    model = NameFinder(DEBERTA, bi_encoder=request.param is GLiNERBiAdapter)
    return load(request.param, model), model


def test_a_short_document_is_read_as_one_window_as_before(adapter_and_model: tuple[Any, NameFinder]) -> None:
    adapter, model = adapter_and_model
    text = f"{filler(20)} Priya Raman {filler(20, 20)}"

    output = adapter.extract([Item(text=text)], labels=["person"])

    assert model.calls == [[text]]
    assert found(output) == [("Priya Raman", text.index("Priya"), text.index("Priya") + 11)]
    assert output.errors is None


def test_a_name_past_the_model_window_is_found_at_its_offsets(adapter_and_model: tuple[Any, NameFinder]) -> None:
    adapter, model = adapter_and_model
    text = f"{filler(5 * MAX_LEN)} Priya Raman wrote in."
    start = text.index("Priya")

    output = adapter.extract([Item(text=text)], labels=["person"])

    assert len(model.calls[0]) > 1
    assert found(output) == [("Priya Raman", start, start + len("Priya Raman"))]


def test_names_at_the_start_middle_and_end_are_each_found_once(adapter_and_model: tuple[Any, NameFinder]) -> None:
    adapter, _ = adapter_and_model
    text = f"Priya Raman {filler(200)} Tomas Novak {filler(200, 200)} Ines Duarte"

    output = adapter.extract([Item(text=text)], labels=["person"])

    assert found(output) == [(name, text.index(name), text.index(name) + len(name)) for name in NAMES]
    assert all(entity["text"] == text[entity["start"] : entity["end"]] for entity in output.entities[0])


def test_a_name_straddling_a_window_edge_is_found_whole_once(adapter_and_model: tuple[Any, NameFinder]) -> None:
    adapter, model = adapter_and_model
    # "Priya" is the last word of the first window and "Raman" the first word past it.
    text = f"{filler(MAX_LEN - 1)} Priya Raman {filler(3 * MAX_LEN, MAX_LEN)}"
    start = text.index("Priya")
    windows = gliner_windows(model, [text])[0]
    assert windows is not None
    assert windows[0].read_end == start + len("Priya")

    output = adapter.extract([Item(text=text)], labels=["person"])

    assert found(output) == [("Priya Raman", start, start + len("Priya Raman"))]


def test_a_name_read_by_two_windows_keeps_its_highest_score(adapter_and_model: tuple[Any, NameFinder]) -> None:
    adapter, model = adapter_and_model
    # In the second half of the first window and the first half of the second.
    text = f"{filler(MAX_LEN - 10)} Priya Raman {filler(3 * MAX_LEN, MAX_LEN)}"
    windows = gliner_windows(model, [text])[0]
    assert windows is not None
    assert windows[1].start < text.index("Priya") < windows[0].read_end

    output = adapter.extract([Item(text=text)], labels=["person"])

    assert found(output) == [("Priya Raman", text.index("Priya"), text.index("Priya") + len("Priya Raman"))]
    assert output.entities[0][0]["score"] == 0.9


def test_windows_overlap_and_cover_the_whole_document(adapter_and_model: tuple[Any, NameFinder]) -> None:
    _, model = adapter_and_model
    text = filler(10 * MAX_LEN)

    windows = gliner_windows(model, [text])[0]

    assert windows is not None
    assert windows[0].start == 0
    assert windows[0].overlap is None
    assert windows[-1].read_end == len(text)
    for before, after in pairwise(windows):
        assert after.overlap == min(WINDOW_OVERLAP_WORDS, MAX_LEN // 2)
        assert before.start < after.start < before.read_end < after.read_end
        # The words shared are the last ``overlap`` words the window before reads.
        shared = text[after.start : before.read_end].split()
        assert len(shared) == after.overlap


def test_each_token_of_a_long_document_is_counted_once(adapter_and_model: tuple[Any, NameFinder]) -> None:
    adapter, model = adapter_and_model
    text = f"Priya Raman {filler(400)} Ines Duarte"
    short = "Tomas Novak wrote in."

    output = adapter.extract([Item(text=text), Item(text=short)], labels=["person"])

    assert len(model.calls[0]) > 2
    # One subword per character, with [CLS] and [SEP] once per document.
    assert output.input_token_counts == [tokens(text), tokens(short)]
    assert adapter._doc_input_token_counts([text, short], ["person"]) == output.input_token_counts


def test_a_document_needing_too_many_windows_fails_rather_than_being_read_in_part(
    adapter_and_model: tuple[Any, NameFinder],
) -> None:
    adapter, model = adapter_and_model
    too_long = filler(MAX_LEN + MAX_DOCUMENT_WINDOWS * (MAX_LEN // 2))
    short = "Tomas Novak wrote in."

    output = adapter.extract([Item(text=too_long), Item(text=short)], labels=["person"])

    assert gliner_windows(model, [too_long]) == [None]
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[0].code == ErrorCode.INPUT_TOO_LONG.value
    assert str(MAX_DOCUMENT_WINDOWS) in output.errors[0].message
    assert output.errors[1] is None
    assert output.entities[0] == []
    assert found(output, 1) == [("Tomas Novak", 0, 11)]
    assert output.input_token_counts == [0, tokens(short)]
    assert model.calls == [[short]]


def test_a_document_just_within_the_window_limit_is_read_whole(adapter_and_model: tuple[Any, NameFinder]) -> None:
    adapter, model = adapter_and_model
    # The first window reads 64 words, and each after it 32 more.
    text = f"{filler(MAX_LEN + (MAX_DOCUMENT_WINDOWS - 1) * (MAX_LEN // 2) - 2)} Priya Raman"

    output = adapter.extract([Item(text=text)], labels=["person"])

    windows = gliner_windows(model, [text])[0]
    assert windows is not None
    assert len(windows) == MAX_DOCUMENT_WINDOWS
    assert output.errors is None
    assert found(output) == [("Priya Raman", text.index("Priya"), len(text))]


def test_adjacent_tokens_merge_across_a_window_edge() -> None:
    model = NameFinder(DEBERTA)
    adapter = load(GLiNERAdapter, model)
    text = f"{filler(MAX_LEN - 1)} Priya Raman {filler(3 * MAX_LEN, MAX_LEN)}"

    output = adapter.extract([Item(text=text)], labels=["person"], options={"merge_adjacent_entities": True})

    start = text.index("Priya")
    assert found(output) == [("Priya Raman", start, start + len("Priya Raman"))]
