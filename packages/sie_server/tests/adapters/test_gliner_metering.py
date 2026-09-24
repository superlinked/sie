from importlib.metadata import version
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.types.inputs import InvalidInputError, Item


class FakeEncoding:
    def __init__(
        self,
        word_ids: list[list[int | None]],
        words_masks: list[list[int]],
        attention_masks: list[list[int]],
    ) -> None:
        self._word_ids = word_ids
        self._attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_masks]
        self._words_mask = [torch.tensor(mask, dtype=torch.long) for mask in words_masks]

    def word_ids(self, batch_index: int) -> list[int | None]:
        return self._word_ids[batch_index]

    def __getitem__(self, key: str) -> list[torch.Tensor]:
        if key == "attention_mask":
            return self._attention_mask
        if key == "words_mask":
            return self._words_mask
        raise KeyError(key)


class FakeProcessor:
    def __init__(self, *, tokenizer_word_limit: int | None = None) -> None:
        self.tokenizer_word_limit = tokenizer_word_limit
        self.raw_batches: list[list[dict[str, Any]]] = []

    def collate_raw_batch(
        self,
        batch: list[dict[str, Any]],
        *,
        entity_types: list[str],
    ) -> dict[str, Any]:
        self.raw_batches.append(batch)
        labels = list(dict.fromkeys(entity_types))
        return {
            "tokens": [item["tokenized_text"][:3] for item in batch],
            "classes_to_id": {label: index for index, label in enumerate(labels, start=1)},
        }

    def prepare_inputs(
        self,
        texts: list[list[str]],
        entity_mappings: dict[str, int],
    ) -> tuple[list[list[str]], list[int]]:
        prompt_length = len(entity_mappings) * 2 + 1
        return texts, [prompt_length] * len(texts)

    def tokenize_inputs(
        self,
        texts: list[list[str]],
        entity_mappings: dict[str, int],
    ) -> FakeEncoding:
        prompt_length = len(entity_mappings) * 2 + 1
        batches: list[list[int | None]] = []
        words_masks: list[list[int]] = []
        for words in texts:
            ids: list[int | None] = [None]
            word_mask = [0]
            ids.extend(range(prompt_length))
            word_mask.extend([0] * prompt_length)
            for word_index, word in enumerate(words, start=prompt_length):
                ids.extend([word_index] * len(word))
                word_mask.extend([word_index - prompt_length + 1] + [0] * (len(word) - 1))
            ids.append(None)
            word_mask.append(0)
            if self.tokenizer_word_limit is not None:
                ids = ids[: self.tokenizer_word_limit]
                word_mask = word_mask[: self.tokenizer_word_limit]
            batches.append(ids)
            words_masks.append(word_mask)
        max_length = max(map(len, batches))
        attention_masks = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in batches]
        for ids, word_mask in zip(batches, words_masks):
            padding = max_length - len(ids)
            ids.extend([None] * padding)
            word_mask.extend([0] * padding)
        return FakeEncoding(batches, words_masks, attention_masks)


def adapter_with_processor(processor: FakeProcessor) -> GLiNERAdapter:
    adapter = GLiNERAdapter("test-model")
    model = MagicMock()
    model.data_processor = processor
    model.prepare_inputs.side_effect = lambda texts: (
        [text.split() for text in texts],
        [[] for _ in texts],
        [[] for _ in texts],
    )
    model.prepare_base_input.side_effect = lambda texts: [{"tokenized_text": text, "ner": None} for text in texts]
    model.inference.return_value = [[]]
    adapter._model = model
    adapter._device = "cpu"
    return adapter


def test_meter_counts_only_processor_retained_document_subwords() -> None:
    adapter = adapter_with_processor(FakeProcessor())

    counts = adapter._doc_input_token_counts(
        ["a bb ccc discarded tail"],
        ["party", "date", "party"],
    )
    deduplicated_counts = adapter._doc_input_token_counts(
        ["a bb ccc discarded tail"],
        ["party", "date"],
    )

    assert counts == deduplicated_counts == [8]


def test_meter_batches_like_inference_and_preserves_alignment() -> None:
    processor = FakeProcessor()
    adapter = adapter_with_processor(processor)
    texts: list[str] = [f"{'x' * (index + 1)} tail" for index in range(10)]

    counts = adapter._doc_input_token_counts(texts, ["entity"])

    assert counts == [index + 1 + 6 for index in range(10)]
    assert [len(batch) for batch in processor.raw_batches] == [8, 2]


def test_extract_rejects_blank_document_before_inference() -> None:
    adapter = adapter_with_processor(FakeProcessor())

    with pytest.raises(ValueError, match="non-blank"):
        adapter.extract([Item(text=" \n\t")], labels=["entity"])

    adapter._model.inference.assert_not_called()


def test_extract_rejects_prompt_that_leaves_no_document_subword() -> None:
    adapter = adapter_with_processor(FakeProcessor(tokenizer_word_limit=4))

    with pytest.raises(InvalidInputError, match="leaves no document tokens"):
        adapter.extract([Item(text="represented")], labels=["entity"])

    adapter._model.inference.assert_not_called()


def test_meter_observes_finite_tokenizer_tail_truncation() -> None:
    adapter = adapter_with_processor(FakeProcessor(tokenizer_word_limit=6))

    counts = adapter._doc_input_token_counts(["represented"], ["entity"])

    # The finite tokenizer represents only two document subwords after CLS and
    # the three-word prompt. The truncated SEP is not counted.
    assert counts == [3]


def test_appending_words_beyond_processor_window_does_not_change_meter() -> None:
    adapter = adapter_with_processor(FakeProcessor())

    prefix = "a bb ccc"
    counts = adapter._doc_input_token_counts(
        [prefix, f"{prefix} {'suffix ' * 500}"],
        ["entity"],
    )

    assert counts == [8, 8]


@pytest.mark.model
@pytest.mark.gpu_hw
@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("urchade/gliner_multi-v2.1", "443d26d654e0324125a96bebd8e796c14ff2efe6"),
        ("numind/NuNER_Zero", "c90187673f464518dca09f41689184ed6976242c"),
    ],
)
def test_pinned_classic_processors_meter_only_the_executed_word_window(
    model_id: str,
    revision: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("real classic GLiNER metering regression requires a GPU")
    assert version("gliner") == "0.2.26"

    adapter = GLiNERAdapter(model_id, revision=revision)
    adapter.load("cuda:0")
    labels = ["party", "date", "term", "law", "money", "obligation", "notice"]
    prefix = " ".join(f"contractword{index}" for index in range(384))
    suffix = " ".join(f"discarded{index}" for index in range(120))
    try:
        prefix_count = adapter._doc_input_token_counts([prefix], labels)
        extended_count = adapter._doc_input_token_counts([f"{prefix} {suffix}"], labels)
        output = adapter.extract([Item(text=f"{prefix} {suffix}")], labels=labels)
    finally:
        adapter.unload()

    assert prefix_count is not None
    assert extended_count == prefix_count
    assert output.input_token_counts == prefix_count
    assert prefix_count[0] > 384
