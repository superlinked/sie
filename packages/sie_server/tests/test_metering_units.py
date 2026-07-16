"""Unit-meter token-count generalization (§7.3).

Covers the shared metering seam that lets EVERY encode/score adapter surface
authoritative per-item / per-pair input-token counts without per-adapter code:

* ``BaseAdapter.count_input_tokens`` — encode ground truth (bert_flash/e5,
  ColBERT, and every flash text encoder inherit it).
* ``BaseAdapter.count_pair_input_tokens`` — reranker ground truth (flash
  cross-encoders inherit it).
* ``EncodePipeline.run_encode`` fallback wiring the encode seam.
* ``QueueExecutor.process_score_batch`` backfilling the score seam.

Ground-truth assertions mirror G2: the emitted count must equal the adapter
tokenizer's own ``len(input_ids)`` for the item/pair. Regression assertions
prove the fallbacks never override counts an adapter already produced (so
bge-m3 / GLiNER / cross_encoder keep their exact values).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.bert_flash import BertFlashAdapter
from sie_server.adapters.clip import CLIPAdapter
from sie_server.adapters.colbert import ColBERTAdapter
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
from sie_server.adapters.jina_flash_cross_encoder import JinaFlashCrossEncoderAdapter
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.siglip import SiglipAdapter
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.inference_output import EncodeOutput, ExtractOutput, ScoreOutput
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.types import WorkerResult
from sie_server.ipc_types import (
    EncodeBatchItem,
    ExtractBatchItem,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    ScoreBatchItem,
)
from sie_server.queue_executor import QueueExecutor
from sie_server.types.inputs import Item


class _FakeTokenizer:
    """Deterministic HF-shaped tokenizer for ground-truth assertions.

    A single text encodes to ``len(text.split()) + 2`` ids (words + CLS/SEP);
    a joint ``(query, doc)`` pair encodes to ``words(q) + words(d) + 3`` ids
    (the shared separators of a cross-encoder). Honors ``truncation`` /
    ``max_length`` so the truncation cap is exercised. Returns ``input_ids`` as
    a list of lists, exactly like a real fast tokenizer called without
    ``return_tensors``.
    """

    def __init__(self, model_max_length: int = 512) -> None:
        self.model_max_length = model_max_length

    def __call__(
        self,
        text: list[str],
        text_pair: list[str] | None = None,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, list[list[int]]]:
        if text_pair is not None:
            lengths = [len(a.split()) + len(b.split()) + 3 for a, b in zip(text, text_pair, strict=True)]
        else:
            lengths = [len(t.split()) + 2 for t in text]
        if truncation and max_length is not None:
            lengths = [min(n, max_length) for n in lengths]
        return {"input_ids": [[0] * n for n in lengths]}


# ---------------------------------------------------------------------------
# count_input_tokens — encode ground truth
# ---------------------------------------------------------------------------


class TestCountInputTokens:
    def test_bert_flash_matches_tokenizer_ground_truth(self) -> None:
        adapter = BertFlashAdapter(model_name_or_path="stub/model")
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        items = [Item(text="alpha beta"), Item(text="one two three four")]
        # Ground truth mirrors the tokenizer: words + 2 special tokens.
        assert adapter.count_input_tokens(items) == [4, 6]

    def test_colbert_matches_tokenizer_ground_truth(self) -> None:
        adapter = ColBERTAdapter(model_name_or_path="stub/model")
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        items = [Item(text="a b c"), Item(text="single")]
        assert adapter.count_input_tokens(items) == [5, 3]

    def test_truncation_cap_applied(self) -> None:
        adapter = BertFlashAdapter(model_name_or_path="stub/model", max_seq_length=4)
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        # "one two three four five" -> 5 words + 2 = 7, capped at 4.
        assert adapter.count_input_tokens([Item(text="one two three four five")]) == [4]

    def test_no_tokenizer_returns_none(self) -> None:
        # Server-backed / image adapters have no in-process tokenizer -> reserve
        # fallback rather than an approximation billed as a count.
        adapter = BertFlashAdapter(model_name_or_path="stub/model")
        adapter._tokenizer = None
        assert adapter.count_input_tokens([Item(text="hello")]) is None

    def test_non_text_item_returns_none(self) -> None:
        adapter = BertFlashAdapter(model_name_or_path="stub/model")
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        assert adapter.count_input_tokens([Item(images=[{"data": b"fake"}])]) is None

    def test_empty_items_returns_empty(self) -> None:
        adapter = BertFlashAdapter(model_name_or_path="stub/model")
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        assert adapter.count_input_tokens([]) == []


# ---------------------------------------------------------------------------
# count_pair_input_tokens — reranker ground truth
# ---------------------------------------------------------------------------


class TestCountPairInputTokens:
    def _adapter(self, *, max_seq_length: int = 512) -> JinaFlashCrossEncoderAdapter:
        adapter = JinaFlashCrossEncoderAdapter(model_name_or_path="stub/model", max_seq_length=max_seq_length)
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        return adapter

    def test_flash_cross_encoder_matches_joint_ground_truth(self) -> None:
        adapter = self._adapter()
        counts = adapter.count_pair_input_tokens(
            Item(text="query terms"),
            [Item(text="doc one"), Item(text="a longer document body")],
        )
        # Joint: words(q) + words(d) + 3 separators.
        assert counts == [2 + 2 + 3, 2 + 4 + 3]

    def test_instruction_is_counted_on_the_query(self) -> None:
        adapter = self._adapter()
        base = adapter.count_pair_input_tokens(Item(text="q"), [Item(text="d")])
        with_instr = adapter.count_pair_input_tokens(Item(text="q"), [Item(text="d")], instruction="please rank")
        assert base == [1 + 1 + 3]
        # "please rank q" -> 3 query words instead of 1 -> +2 tokens.
        assert with_instr == [3 + 1 + 3]

    def test_no_tokenizer_returns_none(self) -> None:
        adapter = self._adapter()
        adapter._tokenizer = None
        assert adapter.count_pair_input_tokens(Item(text="q"), [Item(text="d")]) is None


# ---------------------------------------------------------------------------
# EncodePipeline.run_encode — encode seam wiring
# ---------------------------------------------------------------------------


class _FakeEncodeAdapter(BaseAdapter):
    """Minimal dense encoder that (optionally) owns its token counts."""

    spec = AdapterSpec(inputs=("text",), outputs=("dense",), unload_fields=("_model",))

    def __init__(self, *, tokenizer: Any = None, stamp_extra: bool = False) -> None:
        self._model = object()
        self._tokenizer = tokenizer
        self._max_seq_length = 512
        self._stamp_extra = stamp_extra
        self._device = "cpu"

    def load(self, device: str) -> None:  # pragma: no cover - not exercised
        _ = device

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: Any = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        _ = (output_types, instruction, is_query, prepared_items, options)
        out = EncodeOutput(dense=np.zeros((len(items), 4), dtype=np.float32), batch_size=len(items))
        if self._stamp_extra:
            out.extra["input_token_counts"] = [999 for _ in items]
        return out


def _registry_for(adapter: _FakeEncodeAdapter) -> MagicMock:
    reg = MagicMock()
    # No text/image preprocessor -> _prepare_batch returns None -> direct path.
    reg.preprocessor_registry.has_preprocessor.return_value = False
    reg.postprocessor_registry.transform_sync.return_value = 0.0
    reg.get.return_value = adapter
    return reg


class TestEncodePipelineFallback:
    @pytest.mark.asyncio
    async def test_fallback_populates_units_from_tokenizer(self) -> None:
        adapter = _FakeEncodeAdapter(tokenizer=_FakeTokenizer())
        reg = _registry_for(adapter)
        _formatted, timing = await EncodePipeline.run_encode(
            registry=reg,
            model="m",
            items=[Item(text="alpha beta"), Item(text="one two three four")],
            output_types=["dense"],
            instruction=None,
            config=MagicMock(),
            is_query=False,
            options={},
        )
        assert timing.input_token_counts == [4, 6]

    @pytest.mark.asyncio
    async def test_extra_counts_win_over_fallback(self) -> None:
        # Adapter that pre-stamps extra (like bge-m3) must keep its own counts;
        # the fallback must not re-tokenize over them.
        adapter = _FakeEncodeAdapter(tokenizer=_FakeTokenizer(), stamp_extra=True)
        reg = _registry_for(adapter)
        _formatted, timing = await EncodePipeline.run_encode(
            registry=reg,
            model="m",
            items=[Item(text="alpha beta"), Item(text="x")],
            output_types=["dense"],
            instruction=None,
            config=MagicMock(),
            is_query=False,
            options={},
        )
        assert timing.input_token_counts == [999, 999]

    @pytest.mark.asyncio
    async def test_no_tokenizer_leaves_counts_unset(self) -> None:
        adapter = _FakeEncodeAdapter(tokenizer=None)
        reg = _registry_for(adapter)
        _formatted, timing = await EncodePipeline.run_encode(
            registry=reg,
            model="m",
            items=[Item(text="alpha beta")],
            output_types=["dense"],
            instruction=None,
            config=MagicMock(),
            is_query=False,
            options={},
        )
        assert timing.input_token_counts is None


# ---------------------------------------------------------------------------
# QueueExecutor.process_score_batch — score seam backfill
# ---------------------------------------------------------------------------


def _score_registry() -> MagicMock:
    reg = MagicMock()
    reg.device = "cpu"
    reg.get_config.return_value = MagicMock()
    return reg


def _score_worker(score_output: ScoreOutput) -> AsyncMock:
    worker = AsyncMock()
    fut: asyncio.Future[WorkerResult] = asyncio.Future()
    fut.set_result(WorkerResult(output=score_output, timing=RequestTiming()))
    worker.submit_score_preformed_batch = AsyncMock(return_value=[fut])
    return worker


def _score_request() -> ProcessScoreBatchRequest:
    return ProcessScoreBatchRequest(
        model_id="test/model",
        items=[
            ScoreBatchItem(
                work_item_id="req-1.0",
                request_id="req-1",
                item_index=0,
                total_items=1,
                timestamp=time.time(),
                query_item={"text": "q"},
                score_items=[{"text": "a", "id": "doc-a"}, {"text": "b", "id": "doc-b"}],
            )
        ],
    )


class TestScoreBackfill:
    @pytest.mark.asyncio
    async def test_backfills_units_for_flash_cross_encoder(self) -> None:
        adapter = JinaFlashCrossEncoderAdapter(model_name_or_path="stub/model", max_seq_length=512)
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        # Reranker did not surface counts (flash cross-encoder gap).
        score_output = ScoreOutput(scores=np.array([0.9, 0.1], dtype=np.float32))
        reg = _score_registry()
        reg.get.return_value = adapter
        reg.start_worker = AsyncMock(return_value=_score_worker(score_output))

        ex = QueueExecutor(reg)
        outcome = await ex.process_score_batch(_score_request())

        o = outcome.outcomes[0]
        assert o.units is not None
        # Two pairs, joint count 1 + 1 + 3 = 5 each -> summed billable = 10.
        assert o.units.input_tokens == 10

    @pytest.mark.asyncio
    async def test_existing_counts_are_not_overwritten(self) -> None:
        # An adapter that already surfaced counts (cross_encoder / bge-m3) keeps
        # them; the backfill is a pure fallback.
        adapter = JinaFlashCrossEncoderAdapter(model_name_or_path="stub/model", max_seq_length=512)
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        score_output = ScoreOutput(
            scores=np.array([0.9, 0.1], dtype=np.float32),
            input_token_counts=[7, 7],
        )
        reg = _score_registry()
        reg.get.return_value = adapter
        reg.start_worker = AsyncMock(return_value=_score_worker(score_output))

        ex = QueueExecutor(reg)
        outcome = await ex.process_score_batch(_score_request())

        o = outcome.outcomes[0]
        assert o.units is not None
        assert o.units.input_tokens == 14  # 7 + 7, not the fallback 10


# ---------------------------------------------------------------------------
# Per-image metering (§7 "$ per image")
# ---------------------------------------------------------------------------
#
# The vision analogue of the per-token seam above: any vision adapter inherits
# authoritative per-image counts from the base ``count_input_images`` hook, the
# encode/extract result seam stamps ``UnitCounts.images``, and CLIP/SigLIP TEXT
# (whose tokenizer lives inside the processor, not ``_tokenizer``) now surfaces
# real token counts through the enhanced ``_metering_tokenizer`` fallback.


class _FakeProcessor:
    """Minimal CLIP/SigLIP-style processor exposing a ``.tokenizer`` (the base
    ``_metering_tokenizer`` fallback reads it for text-token metering).
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer


class _FakeVisionEncodeAdapter(BaseAdapter):
    """Dual (text+image) dense encoder: inherits ``count_input_images`` and,
    when given a processor tokenizer, ``count_input_tokens`` for text.
    """

    spec = AdapterSpec(inputs=("text", "image"), outputs=("dense",), unload_fields=("_model",))

    def __init__(self, *, tokenizer: Any = None) -> None:
        self._model = object()
        self._processor = _FakeProcessor(tokenizer) if tokenizer is not None else None
        self._device = "cpu"

    def load(self, device: str) -> None:  # pragma: no cover - not exercised
        _ = device

    def encode(self, items: list[Item], output_types: list[str], **_: Any) -> EncodeOutput:
        return EncodeOutput(dense=np.zeros((len(items), 4), dtype=np.float32), batch_size=len(items))


class _FakeVisionExtractAdapter(BaseAdapter):
    """Image-input extractor (Florence-2 shape): inherits ``count_input_images``
    and surfaces no token counts.
    """

    spec = AdapterSpec(inputs=("image",), outputs=("json",), unload_fields=("_model",))

    def __init__(self) -> None:
        self._model = object()
        self._device = "cpu"

    def load(self, device: str) -> None:  # pragma: no cover - not exercised
        _ = device

    def extract(self, items: list[Item], **_: Any) -> ExtractOutput:  # pragma: no cover - worker mocked
        return ExtractOutput(entities=[[] for _ in items])


class _FakeTextExtractAdapter(BaseAdapter):
    """Text-input extractor (GLiNER shape): no images."""

    spec = AdapterSpec(inputs=("text",), outputs=("json",), unload_fields=("_model",))

    def __init__(self) -> None:
        self._model = object()
        self._device = "cpu"

    def load(self, device: str) -> None:  # pragma: no cover - not exercised
        _ = device

    def extract(self, items: list[Item], **_: Any) -> ExtractOutput:  # pragma: no cover - worker mocked
        return ExtractOutput(entities=[[] for _ in items])


def _img(fmt: str = "png") -> dict[str, Any]:
    return {"data": b"fake-image-bytes", "format": fmt}


# ---------------------------------------------------------------------------
# count_input_images — vision ground truth (base hook)
# ---------------------------------------------------------------------------


class TestCountInputImages:
    def test_counts_images_per_item(self) -> None:
        adapter = _FakeVisionEncodeAdapter()
        items = [
            Item(images=[{"data": b"a"}]),
            Item(images=[{"data": b"b"}, {"data": b"c"}]),
        ]
        assert adapter.count_input_images(items) == [1, 2]

    def test_text_only_items_count_zero(self) -> None:
        adapter = _FakeVisionEncodeAdapter()
        assert adapter.count_input_images([Item(text="alpha"), Item(text="beta")]) == [0, 0]

    def test_mixed_batch(self) -> None:
        adapter = _FakeVisionEncodeAdapter()
        assert adapter.count_input_images([Item(text="a caption"), Item(images=[{"data": b"x"}])]) == [0, 1]

    def test_empty_items(self) -> None:
        assert _FakeVisionEncodeAdapter().count_input_images([]) == []


# ---------------------------------------------------------------------------
# CLIP/SigLIP TEXT — processor-tokenizer metering fallback
# ---------------------------------------------------------------------------


class TestProcessorTokenizerMetering:
    def test_clip_text_counts_via_processor_tokenizer(self) -> None:
        # CLIP keeps its tokenizer inside _processor, not _tokenizer, so the
        # base hook must reach it — otherwise CLIP TEXT bills nothing.
        adapter = CLIPAdapter(model_name_or_path="stub/clip")
        adapter._processor = _FakeProcessor(_FakeTokenizer())  # type: ignore[assignment]
        assert adapter.count_input_tokens([Item(text="alpha beta"), Item(text="one two three")]) == [4, 5]

    def test_siglip_text_counts_via_processor_tokenizer(self) -> None:
        adapter = SiglipAdapter(model_name_or_path="stub/siglip")
        adapter._processor = _FakeProcessor(_FakeTokenizer())  # type: ignore[assignment]
        assert adapter.count_input_tokens([Item(text="a b c")]) == [5]

    def test_processor_model_max_length_caps_the_count(self) -> None:
        # A long text truncates at the tokenizer's model_max_length (the CLIP
        # 77 / SigLIP 64 context window) rather than over-billing.
        adapter = CLIPAdapter(model_name_or_path="stub/clip")
        adapter._processor = _FakeProcessor(_FakeTokenizer(model_max_length=4))  # type: ignore[assignment]
        # "one two three four five" -> 5 words + 2 specials = 7, capped at 4.
        assert adapter.count_input_tokens([Item(text="one two three four five")]) == [4]

    def test_image_only_item_returns_none(self) -> None:
        # No text -> no token count (the image dimension meters it instead).
        adapter = CLIPAdapter(model_name_or_path="stub/clip")
        adapter._processor = _FakeProcessor(_FakeTokenizer())  # type: ignore[assignment]
        assert adapter.count_input_tokens([Item(images=[{"data": b"x"}])]) is None

    def test_no_processor_returns_none(self) -> None:
        adapter = CLIPAdapter(model_name_or_path="stub/clip")
        adapter._processor = None
        assert adapter.count_input_tokens([Item(text="hello")]) is None


# ---------------------------------------------------------------------------
# SGLangEmbeddingAdapter — server-backed self-metering (§7.3)
# ---------------------------------------------------------------------------
#
# SGLang runs the model in a subprocess, so the base ``count_input_tokens``
# seam has no in-process tokenizer and ``units.input_tokens`` would stay 0 (the
# meter's reserve fallback) for the promoted dense-SMARTEST tier. The adapter
# now stamps exact per-item counts onto ``EncodeOutput.extra`` from a lazy,
# weights-free metering tokenizer, counting the EXACT (template/EOS-formatted,
# truncated) strings it POSTs to sglang. These tests inject the deterministic
# ``_FakeTokenizer`` and stub the HTTP POST so no server / weights are needed.


class TestSGLangEmbeddingMetering:
    def _adapter(self, **kwargs: Any) -> SGLangEmbeddingAdapter:
        adapter = SGLangEmbeddingAdapter(model_name_or_path="stub/qwen3-emb", **kwargs)
        adapter._server_url = "http://localhost:0"  # satisfy _check_loaded
        adapter._dense_dim = 4
        adapter._configured_dense_dim = 4
        # Inject the deterministic metering tokenizer (words + 2 specials),
        # bypassing the lazy HF load.
        adapter._metering_tokenizer_obj = _FakeTokenizer()  # type: ignore[assignment]
        adapter._metering_tokenizer_loaded = True
        return adapter

    @staticmethod
    def _stub_embed(adapter: SGLangEmbeddingAdapter, dim: int = 4) -> None:
        # Replace the sglang HTTP POST with a deterministic embedding of the
        # right shape so encode() runs without a live server.
        def fake_embed(texts: list[str], model_name: str) -> np.ndarray:
            _ = model_name
            return np.ones((len(texts), dim), dtype=np.float32)

        adapter._embed_texts = fake_embed  # type: ignore[method-assign]

    def test_doc_side_counts_raw_text(self) -> None:
        # No doc_template -> the posted text == the raw item text -> words + 2.
        adapter = self._adapter()
        self._stub_embed(adapter)
        items = [Item(text="alpha beta"), Item(text="one two three")]
        out = adapter.encode(items, ["dense"], is_query=False)
        assert out.extra["input_token_counts"] == [4, 5]

    def test_query_side_counts_post_template(self) -> None:
        # Qwen3-Embedding-4B applies an Instruct/Query template to queries; the
        # count must reflect the FORMATTED string that is actually sent, not the
        # raw text (that is exactly the metering gap this fixes).
        adapter = self._adapter(
            query_template="Instruct: {instruction}\nQuery: {text}",
            default_instruction="find it",
        )
        self._stub_embed(adapter)
        items = [Item(text="alpha beta")]
        formatted = adapter._format_texts(items, None, is_query=True)
        out = adapter.encode(items, ["dense"], is_query=True)
        assert out.extra["input_token_counts"] == [len(formatted[0].split()) + 2]
        # Strictly MORE than the raw-text count — the template adds tokens.
        assert out.extra["input_token_counts"][0] > len(["alpha", "beta"]) + 2

    def test_empty_items_bill_zero_and_scatter(self) -> None:
        # Whitespace-only items take the zero-vector fallback (not posted) and
        # must bill 0 while the sent items keep their exact counts, aligned 1:1.
        adapter = self._adapter()
        self._stub_embed(adapter)
        items = [Item(text="alpha beta"), Item(text="   "), Item(text="x y z")]
        out = adapter.encode(items, ["dense"], is_query=False)
        assert out.extra["input_token_counts"] == [4, 0, 5]

    def test_all_empty_bills_zero(self) -> None:
        # All-empty batch short-circuits to zero vectors with no POST -> all 0.
        adapter = self._adapter()
        items = [Item(text=""), Item(text="  ")]
        out = adapter.encode(items, ["dense"], is_query=False)
        assert out.extra["input_token_counts"] == [0, 0]

    def test_truncation_cap_applied(self) -> None:
        # A text longer than max_seq_length counts at the cap (sglang truncates
        # at --context-length), not the untruncated length.
        adapter = self._adapter(max_seq_length=4)
        self._stub_embed(adapter)
        out = adapter.encode([Item(text="one two three four five")], ["dense"], is_query=False)
        # 5 words + 2 specials = 7, capped at 4.
        assert out.extra["input_token_counts"] == [4]

    def test_no_tokenizer_leaves_counts_unstamped(self) -> None:
        # A tokenizer load failure degrades to the meter's reserve estimate
        # (no counts) rather than billing an approximation or raising.
        adapter = self._adapter()
        adapter._metering_tokenizer_obj = None  # simulate load failure
        self._stub_embed(adapter)
        out = adapter.encode([Item(text="alpha beta")], ["dense"], is_query=False)
        assert "input_token_counts" not in out.extra


# ---------------------------------------------------------------------------
# QueueExecutor.process_encode_batch — encode seam stamps images
# ---------------------------------------------------------------------------


def _encode_registry(adapter: BaseAdapter) -> MagicMock:
    reg = MagicMock()
    reg.device = "cpu"
    reg.get_config.return_value = MagicMock()
    reg.get.return_value = adapter
    return reg


def _encode_request(items: list[dict[str, Any]]) -> ProcessEncodeBatchRequest:
    return ProcessEncodeBatchRequest(
        model_id="test/model",
        items=[
            EncodeBatchItem(
                work_item_id=f"req-1.{i}",
                request_id="req-1",
                item_index=i,
                total_items=len(items),
                timestamp=time.time(),
                item=item,
            )
            for i, item in enumerate(items)
        ],
    )


class TestEncodeSeamImages:
    @pytest.mark.asyncio
    async def test_image_encode_stamps_images(self) -> None:
        adapter = _FakeVisionEncodeAdapter()
        reg = _encode_registry(adapter)
        ex = QueueExecutor(reg)

        # Image path: the pipeline records no token counts (image input).
        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            return [{"dense": [0.0]} for _ in kwargs["items"]], RequestTiming()

        with patch.object(EncodePipeline, "run_encode", new=AsyncMock(side_effect=fake_run_encode)):
            outcome = await ex.process_encode_batch(_encode_request([{"images": [_img(), _img()]}]))

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.units is not None
        assert o.units.images == 2  # two images in the item
        assert o.units.input_tokens is None  # image path bills no tokens

    @pytest.mark.asyncio
    async def test_text_encode_stamps_tokens_not_images(self) -> None:
        adapter = _FakeVisionEncodeAdapter(tokenizer=_FakeTokenizer())
        reg = _encode_registry(adapter)
        ex = QueueExecutor(reg)

        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            timing = RequestTiming()
            timing.input_token_counts = [4]  # pipeline recorded a real token count
            return [{"dense": [0.0]} for _ in kwargs["items"]], timing

        with patch.object(EncodePipeline, "run_encode", new=AsyncMock(side_effect=fake_run_encode)):
            outcome = await ex.process_encode_batch(_encode_request([{"text": "alpha beta"}]))

        o = outcome.outcomes[0]
        assert o.units is not None
        assert o.units.input_tokens == 4
        assert o.units.images is None  # text-only item never emits images=0

    @pytest.mark.asyncio
    async def test_no_units_when_neither_dimension_present(self) -> None:
        # A vision adapter with no tokenizer on a text item (no token count) and
        # no images leaves units unset -> the meter falls back to the reserve.
        adapter = _FakeVisionEncodeAdapter()
        reg = _encode_registry(adapter)
        ex = QueueExecutor(reg)

        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            return [{"dense": [0.0]} for _ in kwargs["items"]], RequestTiming()

        with patch.object(EncodePipeline, "run_encode", new=AsyncMock(side_effect=fake_run_encode)):
            outcome = await ex.process_encode_batch(_encode_request([{"text": "alpha beta"}]))

        assert outcome.outcomes[0].units is None


# ---------------------------------------------------------------------------
# QueueExecutor.process_extract_batch — extract seam stamps images
# ---------------------------------------------------------------------------


def _extract_worker(extract_output: ExtractOutput) -> AsyncMock:
    worker = AsyncMock()
    fut: asyncio.Future[WorkerResult] = asyncio.Future()
    fut.set_result(WorkerResult(output=extract_output, timing=RequestTiming()))
    worker.submit_extract_preformed_batch = AsyncMock(return_value=[fut])
    return worker


def _extract_registry(adapter: BaseAdapter, worker: AsyncMock) -> MagicMock:
    reg = MagicMock()
    reg.device = "cpu"
    reg.get_config.return_value = MagicMock()
    reg.get.return_value = adapter
    reg.start_worker = AsyncMock(return_value=worker)
    return reg


def _extract_request(item: dict[str, Any]) -> ProcessExtractBatchRequest:
    return ProcessExtractBatchRequest(
        model_id="test/model",
        items=[
            ExtractBatchItem(
                work_item_id="req-1.0",
                request_id="req-1",
                item_index=0,
                total_items=1,
                timestamp=time.time(),
                item=item,
            )
        ],
    )


class TestExtractSeamImages:
    @pytest.mark.asyncio
    async def test_florence2_image_extract_stamps_images(self) -> None:
        adapter = _FakeVisionExtractAdapter()
        # Florence-2 surfaces no input_token_counts on its ExtractOutput.
        extract_output = ExtractOutput(entities=[[{"text": "a red circle", "label": "caption", "score": 1.0}]])
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)

        outcome = await ex.process_extract_batch(_extract_request({"images": [_img()]}))

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.units is not None
        assert o.units.images == 1
        assert o.units.input_tokens is None  # no tokenizer count on the VLM path

    @pytest.mark.asyncio
    async def test_text_extract_keeps_tokens_no_images(self) -> None:
        # Regression: GLiNER-style text extract still bills per token and emits
        # no images (the image fold is a pure no-op for text-only docs).
        adapter = _FakeTextExtractAdapter()
        extract_output = ExtractOutput(
            entities=[[{"text": "Alice", "label": "person", "score": 0.99, "start": 0, "end": 5}]],
            input_token_counts=[6],
        )
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)

        outcome = await ex.process_extract_batch(_extract_request({"text": "Alice works at Acme."}))

        o = outcome.outcomes[0]
        assert o.units is not None
        assert o.units.input_tokens == 6
        assert o.units.images is None


# ---------------------------------------------------------------------------
# QueueExecutor.process_extract_batch — parse/OCR page dimension (§7)
# ---------------------------------------------------------------------------
#
# The canonical parse/OCR billing unit is PAGES ("$ per 1k pages", design §7).
# Document-model parsers (docling) surface the real page count on
# ``ExtractOutput.pages``; the extract result seam folds it into
# ``UnitCounts.pages`` — the third independent §7 dimension alongside tokens
# and images.


class _FakeParseExtractAdapter(BaseAdapter):
    """Document parser (docling shape): surfaces per-item page counts and no
    token counts. Input is a document, so it consumes no images.
    """

    spec = AdapterSpec(inputs=("document", "image"), outputs=("json",), unload_fields=("_model",))

    def __init__(self) -> None:
        self._model = object()
        self._device = "cpu"

    def load(self, device: str) -> None:  # pragma: no cover - not exercised
        _ = device

    def extract(self, items: list[Item], **_: Any) -> ExtractOutput:  # pragma: no cover - worker mocked
        return ExtractOutput(entities=[[] for _ in items])


class TestExtractSeamPages:
    @pytest.mark.asyncio
    async def test_docling_parse_extract_stamps_pages(self) -> None:
        adapter = _FakeParseExtractAdapter()
        # Docling surfaces the real page count on ExtractOutput.pages and no
        # token counts (package-backed parser).
        extract_output = ExtractOutput(
            entities=[[]],
            data=[{"text": "parsed", "markdown": "# parsed", "document": {}}],
            pages=[7],
        )
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)

        outcome = await ex.process_extract_batch(
            _extract_request({"document": {"data": b"%PDF-1.4 ...", "format": "pdf"}})
        )

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.units is not None
        assert o.units.pages == 7  # the §7 parse dimension
        assert o.units.input_tokens is None  # package-backed parse bills no tokens
        assert o.units.images is None  # document input consumes no images

    @pytest.mark.asyncio
    async def test_parse_extract_without_pages_leaves_units_unset(self) -> None:
        # An all-error parse (0 pages) surfaces no pages → the meter falls back
        # to its reserve estimate rather than billing zero.
        adapter = _FakeParseExtractAdapter()
        extract_output = ExtractOutput(entities=[[]], data=[{"error": "bad pdf"}], pages=None)
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)

        outcome = await ex.process_extract_batch(_extract_request({"document": {"data": b"garbage", "format": "pdf"}}))

        assert outcome.outcomes[0].units is None


# ---------------------------------------------------------------------------
# CLIP / SigLIP TEXT tower — exact per-item token-count stamp (§7.3)
# ---------------------------------------------------------------------------
#
# The image towers meter per image (``count_input_images``); the TEXT towers
# tokenize in-process, so they stamp the exact per-item token counts they
# encoded onto ``EncodeOutput.extra`` (scattered to item positions, 0 for image
# items, unstamped for a pure-image batch so it stays on the image dimension).


class TestVisionTextTokenStamp:
    def _clip(self) -> CLIPAdapter:
        adapter = CLIPAdapter(model_name_or_path="stub/clip")
        adapter._model = object()  # type: ignore[assignment]
        adapter._processor = _FakeProcessor(_FakeTokenizer())  # type: ignore[assignment]
        adapter._dense_dim = 4
        adapter._device = "cpu"
        # Stub the tower forwards so no real weights are needed. The text tower
        # still derives its per-text counts from the shared base counter over
        # the processor tokenizer (the real metering path), returning matching
        # zero vectors; the image tower returns zero vectors.
        adapter._encode_image_items = lambda items: np.zeros((len(items), 4), dtype=np.float32)  # type: ignore[method-assign]

        def fake_encode_texts(texts: list[str]) -> tuple[Any, list[int] | None]:
            counts = adapter._token_counts_or_none(adapter._processor.tokenizer, list(texts), expected_len=len(texts))
            return np.zeros((len(texts), 4), dtype=np.float32), counts

        adapter._encode_texts = fake_encode_texts  # type: ignore[method-assign]
        return adapter

    def test_clip_text_encode_stamps_exact_counts(self) -> None:
        adapter = self._clip()
        out = adapter.encode([Item(text="alpha beta"), Item(text="one two three")], ["dense"])
        # _FakeTokenizer: words + 2 specials.
        assert out.extra["input_token_counts"] == [4, 5]

    def test_clip_mixed_batch_scatters_zero_for_image_items(self) -> None:
        adapter = self._clip()
        items = [Item(text="alpha beta"), Item(images=[{"data": b"x"}]), Item(text="one two three")]
        out = adapter.encode(items, ["dense"])
        # Text items keep their real counts; the image item contributes 0 text
        # tokens (it is metered per image instead).
        assert out.extra["input_token_counts"] == [4, 0, 5]

    def test_clip_pure_image_batch_leaves_extra_unstamped(self) -> None:
        adapter = self._clip()
        out = adapter.encode([Item(images=[{"data": b"x"}])], ["dense"])
        assert "input_token_counts" not in out.extra

    def test_siglip_text_encode_stamps_exact_counts(self) -> None:
        adapter = SiglipAdapter(model_name_or_path="stub/siglip")
        adapter._model = object()  # type: ignore[assignment]
        adapter._processor = _FakeProcessor(_FakeTokenizer())  # type: ignore[assignment]
        adapter._dense_dim = 4
        adapter._device = "cpu"
        adapter._backend = "transformers"  # type: ignore[assignment]

        # Stub the text forward: real counts come from the shared base counter
        # over the processor tokenizer; return matching zero vectors.
        def fake_encode_texts(texts: list[str]) -> tuple[Any, list[int] | None]:
            counts = adapter._token_counts_or_none(adapter._processor.tokenizer, list(texts), expected_len=len(texts))
            return np.zeros((len(texts), 4), dtype=np.float32), counts

        adapter._encode_texts = fake_encode_texts  # type: ignore[method-assign]
        out = adapter.encode([Item(text="a b c")], ["dense"])
        assert out.extra["input_token_counts"] == [5]


# ---------------------------------------------------------------------------
# cross_encoder predict/metering concurrency guard (#1800 class-fix, #1782)
# ---------------------------------------------------------------------------


class _ReentrancyDetectingTokenizer:
    """Emulates a HuggingFace fast tokenizer's non-re-entrancy.

    A real fast tokenizer wraps a Rust object behind a ``RefCell``; a second
    call that begins while a first is still in flight raises
    ``RuntimeError("Already borrowed")`` (#1800). This fake reproduces that
    exact failure mode: it raises if two calls are inside a tokenizer call at
    the same time, so a correct serialization guard makes it pass and a missing
    guard makes it fail. It matches the shape the base metering counter expects
    (list-of-lists ``input_ids``, joint ``(text, text_pair)`` support).
    """

    def __init__(self, model_max_length: int = 512) -> None:
        self.model_max_length = model_max_length
        self._active = 0
        self._active_lock = threading.Lock()

    def _enter(self) -> None:
        with self._active_lock:
            if self._active != 0:
                raise RuntimeError("Already borrowed")
            self._active += 1

    def _exit(self) -> None:
        with self._active_lock:
            self._active -= 1

    def __call__(
        self,
        text: list[str] | str,
        text_pair: list[str] | str | None = None,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, list[list[int]]]:
        self._enter()
        try:
            # Simulate real tokenizer latency so overlapping threads collide.
            time.sleep(0.002)
            texts = [text] if isinstance(text, str) else list(text)
            if text_pair is not None:
                pairs = [text_pair] if isinstance(text_pair, str) else list(text_pair)
                lengths = [len(a.split()) + len(b.split()) + 3 for a, b in zip(texts, pairs, strict=True)]
            else:
                lengths = [len(t.split()) + 2 for t in texts]
            if truncation and max_length is not None:
                lengths = [min(n, max_length) for n in lengths]
            return {"input_ids": [[0] * n for n in lengths]}
        finally:
            self._exit()


class TestCrossEncoderTokenizerConcurrencyGuard:
    """Regression for the #1800 class-fix on the sentence-transformers
    ``CrossEncoderAdapter``.

    ``score_pairs`` runs on the single inference-executor thread and does two
    things against the SAME ``self._model.tokenizer``: ``CrossEncoder.predict``
    tokenizes internally (fused with the GPU forward), then the inline
    ``_pair_input_token_counts`` re-tokenizes the pairs for §7.3 metering. The
    shared ``count_pair_input_tokens`` metering fallback re-tokenizes the very
    same tokenizer on a separate thread-pool thread for another concurrent
    request. A bare HF fast tokenizer raises ``Already borrowed`` when two
    tokenize calls overlap; ``_tokenizer_guard()`` on the encode/score-side
    re-tokenize (the sibling of the guard the metering entry point already
    holds) must serialise them.
    """

    def _make_adapter(self, tokenizer: _ReentrancyDetectingTokenizer) -> CrossEncoderAdapter:
        adapter = CrossEncoderAdapter("stub/reranker")
        adapter._device = "cpu"

        def _fake_predict(pairs: list[tuple[str, str]]) -> np.ndarray:
            # Mirror CrossEncoder.predict: tokenize the batch (fused with the
            # forward) on the same tokenizer the metering path re-tokenizes.
            tokenizer([q for q, _ in pairs], [d for _, d in pairs], truncation=True, max_length=512)
            return np.zeros(len(pairs), dtype=np.float32)

        model = MagicMock()
        model.predict.side_effect = _fake_predict
        model.tokenizer = tokenizer
        model.max_length = 512
        adapter._model = model  # ty: ignore[invalid-assignment]
        return adapter

    def test_score_and_metering_do_not_collide_under_concurrency(self) -> None:
        """Batched scoring on one thread and metering-tokenizing on another must
        not raise ``Already borrowed`` — the guard serialises tokenizer access.
        """
        tokenizer = _ReentrancyDetectingTokenizer()
        adapter = self._make_adapter(tokenizer)
        query = Item(text="the query terms")
        docs = [Item(text="doc one"), Item(text="a considerably longer document body")]
        queries = [query] * len(docs)

        errors: list[BaseException] = []
        stop = threading.Event()

        def score_loop() -> None:
            try:
                for _ in range(40):
                    if stop.is_set():
                        return
                    out = adapter.score_pairs(queries, docs)
                    assert out.input_token_counts is not None
                    assert len(out.input_token_counts) == len(docs)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        def metering_loop() -> None:
            try:
                for _ in range(40):
                    if stop.is_set():
                        return
                    counts = adapter.count_pair_input_tokens(query, docs)
                    assert counts is not None
                    assert len(counts) == len(docs)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=score_loop) for _ in range(2)]
        threads += [threading.Thread(target=metering_loop) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        stop.set()

        assert not errors, f"tokenizer collision under concurrency: {errors!r}"

    def test_reentrancy_detector_fires_without_guard(self) -> None:
        """Sanity check the harness: bare concurrent tokenizer calls (no guard)
        DO raise ``Already borrowed`` — otherwise the guard test is vacuous.
        """
        tokenizer = _ReentrancyDetectingTokenizer()
        errors: list[BaseException] = []

        def call_loop() -> None:
            try:
                for _ in range(40):
                    tokenizer(["a", "b c"], ["x", "y z"], truncation=True, max_length=512)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=call_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert any(isinstance(e, RuntimeError) and "Already borrowed" in str(e) for e in errors)
