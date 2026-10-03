from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import Any

import pytest
from sie_mcp import offload

# GLiNER's default word splitter (punctuation marks are words too), and how many
# of those words the default GLiNER models read from each text before silently
# dropping the rest.
_GLINER_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")
_GLINER_MAX_WORDS = 384
_FILLER = "The committee reviewed the figures, then adjourned. "  # 9 GLiNER words


class _FakeOffloadClient:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.extract_calls: list[dict[str, Any]] = []

    async def generate(self, model: str, prompt: str, **kwargs: Any) -> dict[str, str]:
        self.generate_calls.append({"model": model, "prompt": prompt, **kwargs})
        return {"text": f"summary {len(self.generate_calls)}"}

    async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
        self.extract_calls.append({"model": model, "items": items, **kwargs})
        # Like the SDK: one result for a single item, a list for a list of items.
        if isinstance(items, list):
            return [self._extract_one(model, item["text"]) for item in items]
        return self._extract_one(model, items["text"])

    def _extract_one(self, model: str, text: str) -> dict[str, Any]:
        if model == offload.PII_MODEL:
            return {
                "entities": [
                    {
                        "text": "Alice Example",
                        "label": "person",
                        "start": text.index("Alice Example"),
                        "end": text.index("Alice Example") + len("Alice Example"),
                        "score": 0.97,
                    },
                    {
                        "text": "alice@example.com",
                        "label": "email",
                        "start": text.index("alice@example.com"),
                        "end": text.index("alice@example.com") + len("alice@example.com"),
                        "score": 0.98,
                    },
                ]
            }
        return {
            "entities": [
                {"text": "Alice Example", "label": "person", "score": 0.93},
                {"text": "Superlinked", "label": "organization", "score": 0.88},
            ]
        }


async def test_summarize_document_uses_map_reduce_for_large_content() -> None:
    client = _FakeOffloadClient()
    text = ("alpha " * 1200) + "\n\n" + ("bravo " * 1200)

    result = await offload.summarize_document(
        client,
        content=text,
        model="gen",
        gpu="l4",
        max_output_tokens=128,
    )

    assert result["summary"] == "summary 3"
    assert result["metadata"]["chunks"] == 2
    assert result["metadata"]["token_savings_estimate"] > 0
    assert [call["model"] for call in client.generate_calls] == ["gen", "gen", "gen"]
    assert all(call["gpu"] == "l4" for call in client.generate_calls)


async def test_summarize_document_reduces_many_chunks_hierarchically() -> None:
    client = _FakeOffloadClient()
    chunk_count = offload.SUMMARY_REDUCE_BATCH_SIZE + 1
    text = "\n\n".join("x" * offload.SUMMARY_CHUNK_CHARS for _ in range(chunk_count))

    result = await offload.summarize_document(
        client,
        content=text,
        model="gen",
        max_output_tokens=128,
    )

    reduce_prompts = [call["prompt"] for call in client.generate_calls if "SECTION SUMMARIES:" in call["prompt"]]
    assert result["metadata"]["chunks"] == chunk_count
    assert result["metadata"]["reduce_calls"] == 2
    assert result["metadata"]["reduction_rounds"] == 2
    assert result["summary"] == f"summary {chunk_count + 2}"
    assert len(client.generate_calls) == chunk_count + 2
    assert all(prompt.count("## Section") <= offload.SUMMARY_REDUCE_BATCH_SIZE for prompt in reduce_prompts)


async def test_extract_entities_returns_entities_and_markdown_table() -> None:
    client = _FakeOffloadClient()

    result = await offload.extract_entities(
        client,
        content="Alice Example works at Superlinked.",
        labels=["person", "organization"],
        model="ner",
    )

    assert result["metadata"]["entity_count"] == 2
    assert result["entities"][0]["text"] == "Alice Example"
    assert "| Alice Example | person | 0.93 |" in result["markdown_table"]
    assert client.extract_calls[0]["labels"] == ["person", "organization"]
    assert client.extract_calls[0]["options"] is None  # the model's own threshold


async def test_redact_pii_returns_redacted_text_without_original_map() -> None:
    client = _FakeOffloadClient()

    result = await offload.redact_pii(
        client,
        content="Alice Example can be reached at alice@example.com.",
    )

    assert result["redacted_text"] == "[PERSON_1] can be reached at [EMAIL_1]."
    assert result["metadata"]["span_count"] == 2
    assert result["metadata"]["label_counts"] == {"person": 1, "email": 1}
    assert result["metadata"]["pii_map_returned"] is False
    assert "Alice Example" not in result["redacted_text"]


def test_redact_text_drops_overlapping_lower_score_spans() -> None:
    redacted, placeholder_map, count = offload.redact_text(
        "Alice Example",
        [
            {"label": "person", "start": 0, "end": 5, "score": 0.5},
            {"label": "person", "start": 0, "end": 13, "score": 0.9},
        ],
    )

    assert redacted == "[PERSON_1]"
    assert placeholder_map == {"[PERSON_1]": "Alice Example"}
    assert count == 1


@pytest.mark.parametrize(
    ("content", "labels"),
    [
        ("", ["person"]),
        ("content", []),
    ],
)
async def test_extract_entities_rejects_bad_input(content: str, labels: list[str]) -> None:
    with pytest.raises(offload.OffloadError):
        await offload.extract_entities(_FakeOffloadClient(), content=content, labels=labels)


class _WindowAwareClient:
    """Mimics GLiNER on every text it is sent: returns each known phrase found in the
    first ``_GLINER_MAX_WORDS`` words, with start/end relative to that text. Like the
    real model, it silently ignores everything after that.
    """

    def __init__(self, phrases: list[tuple[str, str]]) -> None:
        self.phrases = phrases
        self.calls: list[dict[str, Any]] = []
        self.windows_seen: list[str] = []

    async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
        self.calls.append({"model": model, "items": items, **kwargs})
        if isinstance(items, list):
            return [self._extract_one(item["text"]) for item in items]
        return self._extract_one(items["text"])

    def _extract_one(self, text: str) -> dict[str, Any]:
        self.windows_seen.append(text)
        words = list(_GLINER_WORD.finditer(text))
        read = text[: words[_GLINER_MAX_WORDS - 1].end()] if len(words) > _GLINER_MAX_WORDS else text
        entities: list[dict[str, Any]] = []
        for phrase, label in self.phrases:
            idx = read.find(phrase)
            while idx != -1:
                entities.append({"text": phrase, "label": label, "start": idx, "end": idx + len(phrase), "score": 0.9})
                idx = read.find(phrase, idx + 1)
        return {"entities": entities}


async def test_redact_pii_redacts_pii_past_the_model_word_limit() -> None:
    # Each paragraph is 40 filler sentences (360 words) and one email address, so all
    # but the first address sit well past the first 384 words GLiNER reads per text.
    emails = [f"member{index}@example.com" for index in range(10)]
    content = "".join(_FILLER * 40 + f"Write to {email} today. " for email in emails)
    client = _WindowAwareClient([(email, "email") for email in emails])

    result = await offload.redact_pii(client, content=content, labels=["email"], model="pii")

    assert [email for email in emails if email in result["redacted_text"]] == []
    assert result["metadata"]["span_count"] == len(emails)


async def test_extract_entities_finds_entities_past_the_model_word_limit() -> None:
    phrase = "Acme Corporation"
    # About 450 words before the phrase: a short text, but longer than GLiNER reads at once.
    content = _FILLER * 50 + f"They signed with {phrase} in May. " + _FILLER * 5
    abs_start = content.index(phrase)
    client = _WindowAwareClient([(phrase, "organization")])

    result = await offload.extract_entities(client, content=content, labels=["organization"], model="gliner")

    spans = [(entity["start"], entity["end"]) for entity in result["entities"] if entity["text"] == phrase]
    assert spans == [(abs_start, abs_start + len(phrase))]


async def test_extract_entities_chunks_large_content_and_shifts_offsets() -> None:
    phrase = "Acme Corporation"
    # Past the first window, so it is only found in a later chunk.
    content = ("lorem " * (offload.EXTRACT_WINDOW_WORDS + 30)) + phrase + (" ipsum" * 200)
    abs_start = content.index(phrase)

    client = _WindowAwareClient([(phrase, "organization")])
    result = await offload.extract_entities(client, content=content, labels=["organization"], model="gliner")

    assert len(client.windows_seen) > 1  # actually chunked
    assert phrase not in client.windows_seen[0]
    matches = [e for e in result["entities"] if e["text"] == phrase]
    assert len(matches) == 1
    assert matches[0]["start"] == abs_start
    assert matches[0]["end"] == abs_start + len(phrase)


async def test_extract_entities_dedupes_phrase_in_window_overlap() -> None:
    phrase = "Globex Inc"
    # Inside the overlap region, so it is seen in two adjacent windows.
    pos_words = offload.EXTRACT_WINDOW_WORDS - offload.EXTRACT_OVERLAP_WORDS + 20
    content = ("lorem " * pos_words) + phrase + (" ipsum" * 300)
    client = _WindowAwareClient([(phrase, "organization")])

    result = await offload.extract_entities(client, content=content, labels=["organization"], model="gliner")

    assert sum(phrase in window for window in client.windows_seen) == 2
    matches = [e for e in result["entities"] if e["text"] == phrase]
    assert len(matches) == 1
    assert matches[0]["start"] == content.index(phrase)


async def test_redact_pii_chunks_large_content_and_redacts_full_span() -> None:
    ssn = "123-45-6789"
    content = ("lorem " * (offload.EXTRACT_WINDOW_WORDS + 200)) + "SSN: " + ssn + " end"
    client = _WindowAwareClient([(ssn, "social security number")])

    result = await offload.redact_pii(client, content=content, labels=["social security number"], model="pii")

    assert len(client.windows_seen) > 1
    assert ssn not in result["redacted_text"]
    assert "[SOCIAL_SECURITY_NUMBER_1]" in result["redacted_text"]
    assert result["metadata"]["span_count"] == 1


def test_extract_windows_fit_the_model_word_limit_and_cover_the_text() -> None:
    content = _FILLER * 200
    windows = offload._extract_windows(content, offload.EXTRACT_WINDOW_WORDS, offload.EXTRACT_OVERLAP_WORDS)

    assert len(windows) > 1
    assert offload.EXTRACT_WINDOW_WORDS < _GLINER_MAX_WORDS
    for base, window in windows:
        assert content[base : base + len(window)] == window
        assert len(_GLINER_WORD.findall(window)) <= offload.EXTRACT_WINDOW_WORDS
    # Neighbors overlap, and together the windows run from the first word to the last.
    assert windows[0][0] == 0
    assert windows[-1][0] + len(windows[-1][1]) == len(content.rstrip())
    for (base, window), (next_base, _) in pairwise(windows):
        assert next_base < base + len(window)


def test_extract_windows_stay_bounded_for_text_without_word_breaks() -> None:
    content = "x" * 100_000
    windows = offload._extract_windows(content, offload.EXTRACT_WINDOW_WORDS, offload.EXTRACT_OVERLAP_WORDS)

    assert max(len(window) for _, window in windows) <= offload.EXTRACT_WINDOW_WORDS * offload.EXTRACT_MAX_WORD_CHARS
    assert windows[-1][0] + len(windows[-1][1]) == len(content)


async def test_redact_pii_batches_windows_and_sends_its_score_threshold() -> None:
    content = _FILLER * 300
    windows = offload._extract_windows(content, offload.EXTRACT_WINDOW_WORDS, offload.EXTRACT_OVERLAP_WORDS)
    assert len(windows) > offload.EXTRACT_BATCH_WINDOWS
    client = _WindowAwareClient([])

    await offload.redact_pii(client, content=content, labels=["person"], model="pii")

    # Every window is sent once, in order, at most EXTRACT_BATCH_WINDOWS to a request.
    assert [item["text"] for call in client.calls for item in call["items"]] == [window for _, window in windows]
    assert len(client.calls) == math.ceil(len(windows) / offload.EXTRACT_BATCH_WINDOWS)
    assert all(call["options"] == {"threshold": offload.MIN_PII_SCORE} for call in client.calls)


@pytest.mark.parametrize("tool", ["extract", "redact"])
async def test_extraction_rejects_content_over_cap(tool: str) -> None:
    client = _FakeOffloadClient()
    oversized = "z" * (offload.EXTRACT_MAX_CHARS + 1)
    with pytest.raises(offload.OffloadError, match="extraction limit"):
        if tool == "extract":
            await offload.extract_entities(client, content=oversized, labels=["person"], model="g")
        else:
            await offload.redact_pii(client, content=oversized, labels=["person"], model="g")
