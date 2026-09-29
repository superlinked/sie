"""Unit tests for TopkEmbedAdapter with a stand-in backbone and tokenizer.

The real backbone needs transformers >= 5.2 (the transformers5 bundle), so these
tests exercise everything around it: templates, truncation, scoring masks,
modality routing, batching, the head (truncate, then normalize), image
preprocessing and positions, scoring and metering. Parity with TopK's reference
pipeline on the real checkpoints lives in ``test_topk_embed_parity.py``.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from sie_server.adapters.topk_embed.adapter import TopkEmbedAdapter, _ImageRow, smart_resize
from sie_server.types.inputs import ImageInput, InvalidInputError, Item

REV = "d09d8a7a8cdd6c287f792b3c4d7b41233d46e66a"
HIDDEN = 8
HEAD_WIDTH = 6
PAD_ID = 0
IMAGE_TOKEN_ID = 99
# Punctuation ids of the stand-in tokenizer, all in the scoring skip list.
PUNCT = {":": 25, ",": 11, ".": 13, "!": 0}
PREFIX = [90, 91, 92]
SUFFIX = [93, 94]


class FakeTokenizer:
    """Whitespace words plus single-character punctuation, deterministic ids."""

    eos_token = "<|im_end|>"  # noqa: S105 -- a special token, not a secret
    pad_token_id = PAD_ID

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode_one(self, text: str) -> list[int]:
        ids: list[int] = []
        word = ""
        for char in text:
            if char in PUNCT or char.isspace():
                if word:
                    ids.append(self._word_id(word))
                    word = ""
                if char in PUNCT:
                    ids.append(PUNCT[char])
            else:
                word += char
        if word:
            ids.append(self._word_id(word))
        return ids

    @staticmethod
    def _word_id(word: str) -> int:
        return 100 + sum(ord(c) for c in word) % 900

    def __call__(self, texts: Any, truncation: bool = False, max_length: int | None = None) -> dict[str, Any]:
        self.calls.append({"texts": texts, "truncation": truncation, "max_length": max_length})
        if isinstance(texts, str):
            return {"input_ids": self.encode_one(texts)}
        rows = [self.encode_one(text) for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        return {"input_ids": rows}


class FakeLanguageModel:
    """Identity text tower that records what it was called with."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, inputs_embeds, attention_mask, position_ids, use_cache):
        self.calls.append(
            {"shape": tuple(inputs_embeds.shape), "attention_mask": attention_mask, "position_ids": position_ids}
        )
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class FakeBackbone:
    def __init__(self) -> None:
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(1000, HIDDEN)
        self.language_model = FakeLanguageModel()

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed


def _png(color: str = "white", size: tuple[int, int] = (64, 64)) -> ImageInput:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return ImageInput(data=buf.getvalue(), format="png")


@pytest.fixture
def adapter() -> TopkEmbedAdapter:
    return make_adapter()


def make_adapter(**kwargs: Any) -> TopkEmbedAdapter:
    """An adapter in the state ``load()`` leaves it, around stand-in modules."""
    adapter = TopkEmbedAdapter("topk-io/topk-embed-v1-xsmall", revision=REV, max_seq_length=64, **kwargs)
    torch.manual_seed(1)
    adapter._model = FakeBackbone()
    adapter._head = torch.nn.Linear(HIDDEN, HEAD_WIDTH, bias=False)
    adapter._tokenizer = FakeTokenizer()
    adapter._device = "cpu"
    adapter._dtype = torch.float32
    adapter._multivector_dim = HEAD_WIDTH
    adapter._query_max_length = 16
    adapter._doc_max_length = 64
    adapter._image_token_id = IMAGE_TOKEN_ID
    adapter._skip_ids = torch.tensor(sorted(PUNCT.values()))
    adapter._image_prefix = torch.tensor(PREFIX)
    adapter._image_suffix = torch.tensor(SUFFIX)
    adapter._patch_size = 4
    adapter._merge_size = 2
    adapter._temporal_patch_size = 2
    adapter._min_pixels = 16 * 16
    adapter._max_pixels = 16 * (4 * 2) ** 2
    adapter._pixel_scale = 1 / 127.5
    adapter._pixel_bias = -1.0
    adapter._num_grid_per_side = 8
    return adapter


def _image_tokens(row: _ImageRow) -> int:
    return int((row.input_ids == IMAGE_TOKEN_ID).sum())


def _fake_vision(adapter: TopkEmbedAdapter):
    def vision(images: list[_ImageRow]) -> torch.Tensor:
        count = sum(_image_tokens(row) for row in images)
        return torch.arange(count * HIDDEN, dtype=torch.float32).reshape(count, HIDDEN) / 100.0

    return patch.object(adapter, "_vision_forward", side_effect=vision)


class TestSpec:
    def test_capabilities(self) -> None:
        adapter = TopkEmbedAdapter("topk-io/topk-embed-v1-xsmall")
        assert adapter.capabilities.inputs == ["text", "image"]
        assert adapter.capabilities.outputs == ["multivector", "score"]

    def test_dims_before_load(self) -> None:
        assert TopkEmbedAdapter("m").dims.multivector is None
        assert TopkEmbedAdapter("m", token_dim=128).dims.multivector == 128

    def test_encode_before_load_raises(self) -> None:
        with pytest.raises(RuntimeError):
            TopkEmbedAdapter("m").encode([Item(text="x")], ["multivector"])

    def test_unload_clears_modules(self, adapter: TopkEmbedAdapter) -> None:
        adapter.unload()
        assert adapter._model is None
        assert adapter._head is None
        assert adapter._tokenizer is None


class TestText:
    def test_query_template_strips_and_keeps_every_token(self, adapter: TopkEmbedAdapter) -> None:
        out = adapter.encode([Item(text="  what is up?  ")], ["multivector"], is_query=True)
        texts = adapter._tokenizer.calls[-1]["texts"]
        assert texts == ["Query: what is up?"]
        expected = adapter._tokenizer.encode_one("Query: what is up?")
        assert out.multivector is not None
        assert out.multivector[0].shape == (len(expected), HEAD_WIDTH)
        assert out.extra["input_token_counts"] == [len(expected)]

    def test_document_template_drops_skiplist_tokens(self, adapter: TopkEmbedAdapter) -> None:
        out = adapter.encode([Item(text="Hello, world.")], ["multivector"])
        assert adapter._tokenizer.calls[-1]["texts"] == ["Document: Hello, world."]
        ids = adapter._tokenizer.encode_one("Document: Hello, world.")
        kept = [i for i in ids if i not in PUNCT.values()]
        assert out.multivector is not None
        assert out.multivector[0].shape == (len(kept), HEAD_WIDTH)
        # Billing counts every token the model ran, punctuation included.
        assert out.extra["input_token_counts"] == [len(ids)]

    def test_empty_document_keeps_the_prompt_word(self, adapter: TopkEmbedAdapter) -> None:
        out = adapter.encode([Item(text=""), Item(text="   ")], ["multivector"])
        assert adapter._tokenizer.calls[-1]["texts"] == ["Document:", "Document:"]
        assert out.multivector is not None
        assert [v.shape[0] for v in out.multivector] == [1, 1]

    def test_caps_are_passed_to_the_tokenizer(self, adapter: TopkEmbedAdapter) -> None:
        adapter.encode([Item(text="a b c")], ["multivector"], is_query=True)
        assert adapter._tokenizer.calls[-1]["max_length"] == 16
        adapter.encode([Item(text="a b c")], ["multivector"])
        assert adapter._tokenizer.calls[-1]["max_length"] == 64

    def test_long_query_is_truncated(self, adapter: TopkEmbedAdapter) -> None:
        out = adapter.encode([Item(text=" ".join(["word"] * 40))], ["multivector"], is_query=True)
        assert out.multivector is not None
        assert out.multivector[0].shape[0] == 16

    def test_rows_are_right_padded_without_leaking_pads(self, adapter: TopkEmbedAdapter) -> None:
        short, long = "one", "one two three four five six"
        batched = adapter.encode([Item(text=short), Item(text=long)], ["multivector"], is_query=True)
        alone = adapter.encode([Item(text=short)], ["multivector"], is_query=True)
        assert batched.multivector is not None
        assert alone.multivector is not None
        np.testing.assert_allclose(batched.multivector[0], alone.multivector[0], rtol=0, atol=1e-6)
        mask = adapter._model.language_model.calls[-2]["attention_mask"]
        assert mask.tolist()[0][:3] == [1, 1, 1]
        assert mask[0].sum() < mask[1].sum()
        assert mask[0, -1] == 0

    def test_text_rows_get_default_positions(self, adapter: TopkEmbedAdapter) -> None:
        adapter.encode([Item(text="abc")], ["multivector"], is_query=True)
        assert adapter._model.language_model.calls[-1]["position_ids"] is None

    def test_vectors_are_truncated_then_normalized(self, adapter: TopkEmbedAdapter) -> None:
        adapter._multivector_dim = 4
        out = adapter.encode([Item(text="alpha beta")], ["multivector"], is_query=True)
        assert out.multivector is not None
        vectors = out.multivector[0]
        assert vectors.shape[1] == 4
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        ids = torch.tensor(adapter._tokenizer.encode_one("Query: alpha beta"))
        with torch.no_grad():
            raw = adapter._head(adapter._model.embed(ids))[:, :4]
        np.testing.assert_allclose(vectors, torch.nn.functional.normalize(raw, dim=-1).numpy(), atol=1e-5)

    def test_normalize_can_be_turned_off(self, adapter: TopkEmbedAdapter) -> None:
        out = adapter.encode([Item(text="alpha")], ["multivector"], is_query=True, options={"normalize": False})
        assert out.multivector is not None
        assert not np.allclose(np.linalg.norm(out.multivector[0], axis=1), 1.0)

    def test_batches_respect_the_padded_token_budget(self, adapter: TopkEmbedAdapter) -> None:
        adapter._text_batch_tokens = 10
        batches = adapter._plan_text_batches([2, 9, 3, 5, 1])
        assert sorted(i for batch in batches for i in batch) == [0, 1, 2, 3, 4]
        lengths = [2, 9, 3, 5, 1]
        for batch in batches:
            assert len(batch) == 1 or len(batch) * max(lengths[i] for i in batch) <= 10


class TestInputs:
    def test_rejects_unsupported_output_types(self, adapter: TopkEmbedAdapter) -> None:
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode([Item(text="x")], ["dense"])

    def test_rejects_image_queries(self, adapter: TopkEmbedAdapter) -> None:
        with pytest.raises(InvalidInputError, match="queries must be text"):
            adapter.encode([Item(images=[_png()])], ["multivector"], is_query=True)

    def test_rejects_instructions(self, adapter: TopkEmbedAdapter) -> None:
        with pytest.raises(InvalidInputError, match="instructions are not supported"):
            adapter.encode([Item(text="x")], ["multivector"], instruction="Represent this")

    def test_rejects_items_without_input(self, adapter: TopkEmbedAdapter) -> None:
        with pytest.raises(InvalidInputError, match="requires text or image"):
            adapter.encode([Item()], ["multivector"])


class TestImages:
    def test_mixed_batch_routes_by_modality_and_keeps_order(self, adapter: TopkEmbedAdapter) -> None:
        items = [Item(text="first text"), Item(images=[_png()]), Item(text="second")]
        with _fake_vision(adapter):
            out = adapter.encode(items, ["multivector"])
        assert out.multivector is not None
        assert len(out.multivector) == 3
        row = adapter._image_row(Image.new("RGB", (64, 64), "white"))
        assert out.multivector[1].shape == (_image_tokens(row), HEAD_WIDTH)
        ids = adapter._tokenizer.encode_one("Document: first text")
        assert out.multivector[0].shape[0] == len([i for i in ids if i not in PUNCT.values()])
        counts = out.extra["input_token_counts"]
        assert counts[1] == 0
        assert counts[0] == len(ids)
        # One text forward and one image forward.
        assert len(adapter._model.language_model.calls) == 2

    def test_image_rows_get_3d_positions(self, adapter: TopkEmbedAdapter) -> None:
        with _fake_vision(adapter):
            adapter.encode([Item(images=[_png()])], ["multivector"])
        positions = adapter._model.language_model.calls[-1]["position_ids"]
        assert positions is not None
        assert positions.shape[0] == 3

    def test_multi_image_item_concatenates(self, adapter: TopkEmbedAdapter) -> None:
        with _fake_vision(adapter):
            one = adapter.encode([Item(images=[_png()])], ["multivector"])
            two = adapter.encode([Item(images=[_png(), _png("black")])], ["multivector"])
        assert one.multivector is not None
        assert two.multivector is not None
        assert two.multivector[0].shape[0] == 2 * one.multivector[0].shape[0]

    def test_image_row_layout(self, adapter: TopkEmbedAdapter) -> None:
        row = adapter._image_row(Image.new("RGB", (64, 48), "white"))
        _, grid_h, grid_w = row.grid_thw
        count = grid_h * grid_w // 4
        assert row.input_ids.tolist() == PREFIX + [IMAGE_TOKEN_ID] * count + SUFFIX
        assert row.pixel_values.shape == (grid_h * grid_w, 3 * 2 * 4 * 4)
        # White pixels normalize to 255 / 127.5 - 1 == 1.
        torch.testing.assert_close(row.pixel_values, torch.ones_like(row.pixel_values))

    def test_image_token_budget_bounds_the_patch_count(self, adapter: TopkEmbedAdapter) -> None:
        row = adapter._image_row(Image.new("RGB", (400, 300), "white"))
        assert _image_tokens(row) <= 16

    def test_position_ids_follow_the_merged_grid(self, adapter: TopkEmbedAdapter) -> None:
        # grid (1, 4, 6) merges 2x2 -> a 2 x 3 image; prefix of 3 tokens, suffix of 2.
        ids = torch.tensor(PREFIX + [IMAGE_TOKEN_ID] * 6 + SUFFIX)
        row = _ImageRow(input_ids=ids, pixel_values=torch.zeros(24, 96), grid_thw=(1, 4, 6))
        positions = adapter._image_position_ids([row], len(ids) + 1)[:, 0, :]
        assert positions[:, :3].tolist() == [[0, 1, 2]] * 3
        assert positions[0, 3:9].tolist() == [3] * 6
        assert positions[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]
        assert positions[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]
        # Text after the image resumes at prefix + max(2, 3).
        assert positions[:, 9:11].tolist() == [[6, 7]] * 3
        assert positions[:, 11].tolist() == [0, 0, 0]

    def test_grid_inputs_are_cached_and_consistent(self, adapter: TopkEmbedAdapter) -> None:
        index, weight, rot = adapter._grid_inputs((1, 4, 6))
        assert index.shape == (4, 24)
        assert weight.shape == (4, 24)
        assert rot.shape == (24, 2)
        torch.testing.assert_close(weight.sum(0), torch.ones(24))
        assert adapter._grid_inputs((1, 4, 6))[0] is index


class TestSmartResize:
    def test_page_fits_the_default_budget(self) -> None:
        # A 1600x1200 page: 30 x 41 merged patches, the 1230 tokens the reference produces.
        assert smart_resize(1200, 1600, factor=32, min_pixels=65536, max_pixels=1280 * 32 * 32) == (960, 1312)

    def test_rounds_to_the_factor(self) -> None:
        assert smart_resize(500, 700, factor=32, min_pixels=65536, max_pixels=1280 * 32 * 32) == (512, 704)

    def test_scales_down_to_the_budget(self) -> None:
        height, width = smart_resize(4000, 3000, factor=32, min_pixels=65536, max_pixels=1280 * 32 * 32)
        assert height * width <= 1280 * 32 * 32
        assert height % 32 == 0
        assert width % 32 == 0

    def test_scales_up_small_images(self) -> None:
        height, width = smart_resize(100, 100, factor=32, min_pixels=65536, max_pixels=1280 * 32 * 32)
        assert height * width >= 65536

    def test_rejects_extreme_aspect_ratios(self) -> None:
        with pytest.raises(InvalidInputError, match="aspect ratio"):
            smart_resize(10, 3000, factor=32, min_pixels=1, max_pixels=10**7)


class TestScoring:
    def test_score_is_maxsim_over_encode(self, adapter: TopkEmbedAdapter) -> None:
        query, docs = Item(text="alpha beta"), [Item(text="alpha"), Item(text="gamma delta")]
        scores = adapter.score(query, docs)
        q = adapter.encode([query], ["multivector"], is_query=True).multivector
        d = adapter.encode(docs, ["multivector"]).multivector
        assert q is not None
        assert d is not None
        expected = [float((q[0] @ doc.T).max(axis=1).sum()) for doc in d]
        np.testing.assert_allclose(scores, expected, rtol=1e-5)

    def test_score_pairs_forwards_options(self, adapter: TopkEmbedAdapter) -> None:
        queries = [Item(text="alpha"), Item(text="alpha")]
        docs = [Item(text="alpha"), Item(text="beta")]
        normalized = adapter.score_pairs(queries, docs)
        raw = adapter.score_pairs(queries, docs, options={"normalize": False})
        assert normalized.scores.shape == (2,)
        assert not np.allclose(normalized.scores, raw.scores)


class TestMetering:
    def test_count_input_tokens_scatters_over_mixed_items(self, adapter: TopkEmbedAdapter) -> None:
        items = [Item(text="one two"), Item(images=[_png()], text="ignored caption"), Item(text="three")]
        assert adapter.count_input_tokens(items) == [2, 0, 1]

    def test_postprocessor_uses_the_token_dim(self, adapter: TopkEmbedAdapter) -> None:
        postprocessors = adapter.get_postprocessors()
        assert set(postprocessors) == {"muvera"}
        assert postprocessors["muvera"].token_dim == HEAD_WIDTH
