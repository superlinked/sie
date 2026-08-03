from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch
from sie_server.adapters.qwen3_vl_reranker.adapter import (
    Qwen3VLRerankerAdapter,
    _build_reranker_conversation,
)
from sie_server.types.inputs import InvalidInputError, InvalidMediaError, Item

_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _render_user_image_tokens(messages: list[dict[str, Any]]) -> str:
    """Render the image-token behavior relevant to Qwen3-VL-Reranker's template."""
    tokens = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                tokens.append(_IMAGE_PLACEHOLDER)
    return "".join(tokens)


def test_document_image_is_in_user_message_for_vision_token_rendering() -> None:
    """Document images render in user content so Qwen emits image placeholders."""
    doc_image: Any = object()

    conversation = _build_reranker_conversation(
        query_text="solid propellant rocket nozzle cross-section",
        doc_image=doc_image,
        instruction="Retrieve images or text relevant to the user's query.",
    )

    assert [message["role"] for message in conversation] == ["system", "user"]
    assert _render_user_image_tokens(conversation).count(_IMAGE_PLACEHOLDER) == 1
    assert [
        part["image"]
        for message in conversation
        if message["role"] == "user"
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image"
    ] == [doc_image]


def test_non_empty_query_with_image_document_places_image_after_document_marker() -> None:
    """Document images stay under the document section, not the query section."""
    doc_image: Any = object()

    conversation = _build_reranker_conversation(
        query_text="solid propellant rocket nozzle cross-section",
        doc_image=doc_image,
        instruction="Retrieve images or text relevant to the user's query.",
    )

    user_content = conversation[1]["content"]
    text_parts = [part["text"] for part in user_content if isinstance(part, dict) and part.get("type") == "text"]
    document_marker_index = next(
        idx
        for idx, part in enumerate(user_content)
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text") == "\n<Document>:"
    )
    image_index = next(
        idx
        for idx, part in enumerate(user_content)
        if isinstance(part, dict) and part.get("type") == "image" and part.get("image") is doc_image
    )

    assert text_parts == [
        "<Instruct>: Retrieve images or text relevant to the user's query.",
        "<Query>:",
        "solid propellant rocket nozzle cross-section",
        "\n<Document>:",
    ]
    assert image_index > document_marker_index
    assert _render_user_image_tokens(conversation).count(_IMAGE_PLACEHOLDER) == 1


def test_empty_query_or_document_side_uses_null_placeholder() -> None:
    """Empty query or document sides use the upstream reranker NULL sentinel."""
    conversation = _build_reranker_conversation(
        doc_text="Cross-section drawing of a solid propellant rocket motor nozzle.",
        instruction="Retrieve relevant documents for the query.",
    )

    user_content = conversation[1]["content"]
    rendered_text = [part["text"] for part in user_content if isinstance(part, dict) and part.get("type") == "text"]

    assert rendered_text == [
        "<Instruct>: Retrieve relevant documents for the query.",
        "<Query>:",
        "NULL",
        "\n<Document>:",
        "Cross-section drawing of a solid propellant rocket motor nozzle.",
    ]


def test_vl_reranker_rejects_invalid_public_items_with_typed_errors() -> None:
    with pytest.raises(InvalidMediaError, match="at most one image"):
        Qwen3VLRerankerAdapter._validate_item(Item(images=[{"data": b"first"}, {"data": b"second"}]))

    with pytest.raises(InvalidMediaError, match="only text and image"):
        Qwen3VLRerankerAdapter._validate_item(Item(audio={"data": b"audio"}))

    for item in (Item(), Item(text=""), Item(text="   ")):
        with pytest.raises(InvalidInputError, match="nonblank text or one image"):
            Qwen3VLRerankerAdapter._validate_item(item)


def test_vl_reranker_allows_image_only_and_text_plus_image_items() -> None:
    Qwen3VLRerankerAdapter._validate_item(Item(images=[{"data": b"image"}]))
    Qwen3VLRerankerAdapter._validate_item(Item(text="caption", images=[{"data": b"image"}]))


def test_vl_reranker_rejects_undecodable_image_with_typed_error() -> None:
    adapter = Qwen3VLRerankerAdapter("unused")

    with pytest.raises(InvalidMediaError, match="valid decodable image"):
        adapter._load_first_image(Item(images=[{"data": b"not-an-image"}]))


def test_vl_runtime_max_length_is_strict_and_clamped() -> None:
    adapter = Qwen3VLRerankerAdapter("unused", max_seq_length=1024)

    assert adapter._runtime_max_length(None) == 1024
    assert adapter._runtime_max_length(2048) == 1024
    assert adapter._runtime_max_length(512) == 512
    for value in (True, 0, -1, "512"):
        with pytest.raises(InvalidInputError, match="positive integer"):
            adapter._runtime_max_length(value)


def test_vl_document_truncation_preserves_prefix_and_scoring_suffix() -> None:
    input_ids = list(range(12))
    offsets = [(index, index + 1) for index in range(12)]

    trimmed = Qwen3VLRerankerAdapter._trim_document_tokens(
        input_ids,
        offsets,
        (4, 9),
        9,
        has_image=False,
        pair_index=0,
    )

    assert trimmed == [0, 1, 2, 3, 4, 5, 9, 10, 11]


def test_vl_document_truncation_rejects_window_smaller_than_protected_image_context() -> None:
    with pytest.raises(InvalidInputError, match="complete image tokens"):
        Qwen3VLRerankerAdapter._trim_document_tokens(
            [0, 1, 2, 3, 4, 5],
            [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (0, 0)],
            (1, 4),
            2,
            has_image=True,
            pair_index=1,
        )


class _FakeTokenizer:
    all_special_tokens: ClassVar[list[str]] = ["<|image_pad|>", "<|im_start|>", "<|im_end|>"]

    @staticmethod
    def pad(
        encoded: dict[str, list[list[int]]],
        *,
        padding: bool,
        return_attention_mask: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert return_attention_mask is True
        assert return_tensors == "pt"
        rows = encoded["input_ids"]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeProcessor:
    image_token = "<|image_pad|>"  # noqa: S105
    video_token = "<|video_pad|>"  # noqa: S105
    vision_start_token = "<|vision_start|>"  # noqa: S105
    vision_end_token = "<|vision_end|>"  # noqa: S105

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.template_calls = 0
        self.tokenizer = _FakeTokenizer()
        self.image_processor = SimpleNamespace(merge_size=1)

    def apply_chat_template(self, conversation: list[dict[str, Any]], **_kwargs: Any) -> str:
        self.template_calls += 1
        rendered: list[str] = []
        for message in conversation:
            for part in message["content"]:
                rendered.append(part["text"] if part["type"] == "text" else self.image_token)
        return "".join(rendered) + "|SUFFIX"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        prompts = kwargs["text"]
        output: dict[str, Any] = {
            "input_ids": [[ord(character) % 127 for character in prompt] for prompt in prompts],
            "attention_mask": [[1] * len(prompt) for prompt in prompts],
            "offset_mapping": [[(index, index + 1) for index in range(len(prompt))] for prompt in prompts],
        }
        if kwargs.get("images"):
            image_count = len(kwargs["images"])
            output["pixel_values"] = torch.ones((image_count, 1))
            output["image_grid_thw"] = torch.ones((image_count, 3), dtype=torch.long)
        return output


class _RaceDetectingProcessor(_FakeProcessor):
    """Emulate shared fast-tokenizer mutation across processor methods."""

    def __init__(self) -> None:
        super().__init__()
        self._active = 0
        self._guard = threading.Lock()
        self.peak_concurrency = 0
        tokenizer = self.tokenizer
        processor = self

        class RaceDetectingTokenizer:
            all_special_tokens = tokenizer.all_special_tokens

            def pad(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
                processor._enter()
                try:
                    return tokenizer.pad(*args, **kwargs)
                finally:
                    processor._exit()

        self.tokenizer = RaceDetectingTokenizer()

    def _enter(self) -> None:
        with self._guard:
            self._active += 1
            self.peak_concurrency = max(self.peak_concurrency, self._active)
            concurrent = self._active > 1
        if concurrent:
            with self._guard:
                self._active -= 1
            raise RuntimeError("Already borrowed")
        time.sleep(0.002)

    def _exit(self) -> None:
        with self._guard:
            self._active -= 1

    def apply_chat_template(self, conversation: list[dict[str, Any]], **kwargs: Any) -> str:
        self._enter()
        try:
            return super().apply_chat_template(conversation, **kwargs)
        finally:
            self._exit()

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self._enter()
        try:
            return super().__call__(**kwargs)
        finally:
            self._exit()


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.attention_implementations: list[str] = []
        self.last_input_ids: torch.Tensor | None = None

    def set_attn_implementation(self, attention: str) -> None:
        self.attention_implementations.append(attention)

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        self.last_input_ids = kwargs["input_ids"]
        batch_size, sequence_length = kwargs["input_ids"].shape
        logits = torch.zeros((batch_size, sequence_length, 4))
        for index in range(batch_size):
            last_index = int(torch.where(kwargs["attention_mask"][index].bool())[0][-1])
            if index == 0:
                logits[index, last_index, 1] = 2.0
            else:
                logits[index, last_index, 2] = 2.0
        return SimpleNamespace(logits=logits)


def test_vl_score_pairs_batches_forward_and_surfaces_measured_units(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = Qwen3VLRerankerAdapter("unused", max_seq_length=1024)
    model = _FakeModel()
    processor = _FakeProcessor()
    adapter._model = model  # ty: ignore[invalid-assignment]
    adapter._processor = processor  # ty: ignore[invalid-assignment]
    adapter._device = "cpu"
    adapter._yes_token_id = 1
    adapter._no_token_id = 2
    decoded_image = object()
    monkeypatch.setattr(adapter, "_load_first_image", lambda _item: decoded_image)

    output = adapter.score_pairs(
        [Item(text="query one"), Item(text="query two")],
        [Item(text="document"), Item(images=[{"data": b"doc-image"}])],
        options={"max_seq_length": 512},
    )

    assert model.calls == 1
    assert processor.template_calls == 4
    assert len(processor.calls) == 1
    processed_prompts = processor.calls[0]["text"]
    assert processor.calls[0]["images"] == [decoded_image]
    assert processor.calls[0]["truncation"] is False
    assert processor.calls[0]["padding"] is False
    assert processor.calls[0]["return_tensors"] is None
    assert processor.calls[0]["return_offsets_mapping"] is True
    assert output.scores.tolist() == pytest.approx([0.880797, 0.119203])
    assert output.input_token_counts == [len(prompt) for prompt in processed_prompts]
    assert output.input_image_counts == [0, 1]


def test_vl_processor_tokenization_is_thread_safe() -> None:
    adapter = Qwen3VLRerankerAdapter("unused")
    model = _FakeModel()
    processor = _RaceDetectingProcessor()
    adapter._model = model  # ty: ignore[invalid-assignment]
    adapter._processor = processor  # ty: ignore[invalid-assignment]
    adapter._device = "cpu"
    adapter._yes_token_id = 1
    adapter._no_token_id = 2

    def score(_: int) -> float:
        return adapter.score_pairs([Item(text="query")], [Item(text="document")]).scores[0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        scores = list(executor.map(score, range(32)))

    assert len(scores) == 32
    assert processor.peak_concurrency == 1


def test_vl_score_pairs_keeps_cuda_allocator_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = Qwen3VLRerankerAdapter("unused")
    adapter._model = object()
    adapter._device = "cuda"
    empty_cache_calls: list[None] = []
    monkeypatch.setattr(
        adapter,
        "_score_pair_batch",
        lambda *_args, **_kwargs: (torch.tensor([0.5]).numpy(), [2], [0]),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None))

    adapter.score_pairs([Item(text="query")], [Item(text="document")])

    assert empty_cache_calls == []


def test_vl_text_singleton_uses_sdpa_and_restores_flash_for_images(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = Qwen3VLRerankerAdapter("unused")
    model = _FakeModel()
    processor = _FakeProcessor()
    adapter._model = model  # ty: ignore[invalid-assignment]
    adapter._processor = processor  # ty: ignore[invalid-assignment]
    adapter._device = "cpu"
    adapter._yes_token_id = 1
    adapter._no_token_id = 2
    adapter._attn_implementation = "flash_attention_2"
    monkeypatch.setattr(adapter, "_load_first_image", lambda _item: object())

    adapter.score_pairs(
        [Item(text="query")],
        [Item(text="document")],
    )
    adapter.score_pairs(
        [Item(text="query")],
        [Item(images=[{"data": b"doc-image"}])],
    )

    assert model.calls == 2
    assert model.attention_implementations == ["sdpa", "flash_attention_2"]


def test_pair_image_counts_match_consumed_images() -> None:
    adapter = Qwen3VLRerankerAdapter("stub/model")
    query = Item(text="query", images=[{"data": b"q1"}])
    docs = [
        Item(images=[{"data": b"d1"}]),
        Item(text="text only"),
        Item(text="mixed", images=[{"data": b"d3"}]),
    ]

    assert adapter.count_pair_input_images(query, docs) == [2, 1, 2]
