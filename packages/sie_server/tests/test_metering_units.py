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
import pathlib
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
import torch
from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.bert_flash import BertFlashAdapter
from sie_server.adapters.clip import CLIPAdapter
from sie_server.adapters.colbert import ColBERTAdapter
from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
from sie_server.adapters.jina_flash_cross_encoder import JinaFlashCrossEncoderAdapter
from sie_server.adapters.qwen3_vl_reranker.adapter import Qwen3VLRerankerAdapter
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.siglip.adapter import SiglipAdapter
from sie_server.adapters.splade_flash.adapter import SPLADEFlashAdapter
from sie_server.core.encode_pipeline import EncodePipeline, _wholly_skipped_text_tower_zeros
from sie_server.core.inference_output import EncodeOutput, ExtractItemError, ExtractOutput, ScoreOutput
from sie_server.core.score_cost import MAX_SCORE_ITEMS, build_score_prepared_items
from sie_server.core.timing import RequestTiming
from sie_server.core.video_frames import VideoDecodeError
from sie_server.core.worker.handlers.encode import EncodeHandler
from sie_server.core.worker.handlers.score import ScoreHandler
from sie_server.core.worker.types import WorkerResult
from sie_server.ipc_types import (
    EncodeBatchItem,
    ExtractBatchItem,
    PreparedAudioPcm16,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    ScoreBatchItem,
    UnitCounts,
)
from sie_server.queue_executor import (
    _MAX_ENCODE_ISOLATION_PASSES,
    QueueExecutor,
    _encode_units,
    _per_pair_image_counts,
    _with_audio_ms,
)
from sie_server.types.inputs import Item
from sie_server.types.responses import ErrorCode


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
# Retrieval-family adapter-emitted counts
# ---------------------------------------------------------------------------


class TestRetrievalAdapterCounts:
    def test_splade_idf_path_counts_transformed_truncated_input(self) -> None:
        adapter = SPLADEFlashAdapter("stub/model", max_seq_length=4, query_template="query: {instruction} {text}")
        adapter._model = object()
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        adapter._idf = torch.ones(8, dtype=torch.float32)

        output = adapter.encode(
            [Item(text="alpha beta gamma")],
            ["sparse"],
            instruction="retrieve",
            is_query=True,
        )

        assert output.extra["input_token_counts"] == [4]

    def test_gte_score_counts_separate_query_and_document_caps(self) -> None:
        adapter = ColBERTModernBERTFlashAdapter(
            model_name_or_path="stub/model",
            query_max_length=4,
            max_seq_length=6,
            query_prefix="q: ",
            doc_prefix="d: ",
        )
        adapter._tokenizer = _FakeTokenizer()  # type: ignore[assignment]
        adapter.score = MagicMock(side_effect=lambda _query, items, **_kwargs: [0.5] * len(items))  # type: ignore[method-assign]
        queries = [Item(text="short"), Item(text="another query")]
        docs = [
            Item(text="two words"),
            Item(text="one two three four five six seven"),
        ]

        output = adapter.score_pairs(
            queries,
            docs,
            instruction="please rank",
        )

        assert output.scores.tolist() == [0.5, 0.5]
        assert output.input_token_counts == [9, 10]


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
    @pytest.mark.parametrize("counts", [[-1], [True]])
    def test_score_output_rejects_invalid_token_counts(self, counts: list[int]) -> None:
        with pytest.raises(ValueError, match="non-negative integers"):
            ScoreOutput(scores=np.array([0.5], dtype=np.float32), input_token_counts=counts)

    @pytest.mark.parametrize("counts", [[-1], [True]])
    def test_score_output_rejects_invalid_image_counts(self, counts: list[int]) -> None:
        with pytest.raises(ValueError, match="non-negative integers"):
            ScoreOutput(scores=np.array([0.5], dtype=np.float32), input_image_counts=counts)

    def test_native_score_candidate_bounds(self) -> None:
        query = Item(text="query")
        with pytest.raises(ValueError, match="at least one"):
            build_score_prepared_items(query, [])
        with pytest.raises(ValueError, match="at most 1000"):
            build_score_prepared_items(query, [Item(text="candidate")] * (MAX_SCORE_ITEMS + 1))

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
            input_image_counts=[1, 2],
        )
        reg = _score_registry()
        reg.get.return_value = adapter
        reg.start_worker = AsyncMock(return_value=_score_worker(score_output))

        ex = QueueExecutor(reg)
        outcome = await ex.process_score_batch(_score_request())

        o = outcome.outcomes[0]
        assert o.units is not None
        assert o.units.input_tokens == 14  # 7 + 7, not the fallback 10
        assert o.units.images == 3
        assert o.units.pairs == 2

    def test_score_handler_preserves_image_counts_across_oom_slicing(self) -> None:
        handler = ScoreHandler()
        output = ScoreOutput(
            scores=np.array([0.9, 0.1], dtype=np.float32),
            input_token_counts=[7, 8],
            input_image_counts=[1, 2],
        )

        partials = {index: handler.slice_output(output, index) for index in range(2)}
        assembled = handler.assemble_output(partials, batch_size=2)

        assert assembled.input_token_counts == [7, 8]
        assert assembled.input_image_counts == [1, 2]

    @pytest.mark.asyncio
    async def test_qwen3_vl_settles_only_images_consumed_per_pair(self) -> None:
        adapter = Qwen3VLRerankerAdapter(model_name_or_path="stub/model")
        score_output = ScoreOutput(scores=np.array([0.9, 0.1], dtype=np.float32))
        reg = _score_registry()
        reg.get.return_value = adapter
        reg.start_worker = AsyncMock(return_value=_score_worker(score_output))
        request = ProcessScoreBatchRequest(
            model_id="test/model",
            items=[
                ScoreBatchItem(
                    work_item_id="req-vision.0",
                    request_id="req-vision",
                    item_index=0,
                    total_items=1,
                    timestamp=time.time(),
                    query_item={"images": [_img(), _img()]},
                    score_items=[
                        {"images": [_img(), _img()], "id": "doc-image"},
                        {"text": "text only", "id": "doc-text"},
                    ],
                )
            ],
        )

        outcome = await QueueExecutor(reg).process_score_batch(request)

        units = outcome.outcomes[0].units
        assert units is not None
        assert units.pairs == 2
        # The image-query launch path cannot surface exact processor prompt
        # tokens yet, so the catalog intentionally declares only pairs/images.
        assert units.input_tokens is None
        # Query first image is consumed once per pair; doc first image once.
        assert units.images == 3

    @pytest.mark.parametrize("invalid", [[1], [1, -1], [1, True], "two"])
    def test_pair_image_evidence_fails_closed_when_malformed(self, invalid: object) -> None:
        adapter = MagicMock()
        adapter.count_pair_input_images.return_value = invalid

        assert (
            _per_pair_image_counts(
                adapter,
                Item(text="query"),
                [Item(text="a"), Item(text="b")],
                2,
            )
            is None
        )

    def test_pair_image_evidence_is_optional_for_other_adapters(self) -> None:
        assert (
            _per_pair_image_counts(
                object(),
                Item(text="query"),
                [Item(text="doc")],
                1,
            )
            is None
        )


# ---------------------------------------------------------------------------
# Accepted-audio duration metering
# ---------------------------------------------------------------------------


class TestAudioDurationUnits:
    def test_exact_milliseconds_are_preserved(self) -> None:
        units = _with_audio_ms(None, 12_345)
        assert units is not None
        assert units.audio_ms == 12_345

    def test_audio_fold_preserves_other_dimensions(self) -> None:
        units = _with_audio_ms(UnitCounts(input_tokens=7, pages=2, images=1), 8)
        assert units == UnitCounts(input_tokens=7, pages=2, images=1, audio_ms=8)

    @pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "1000", 1 << 64])
    def test_invalid_audio_duration_fails_closed(self, invalid: object) -> None:
        with pytest.raises(ValueError, match="positive u64"):
            _with_audio_ms(None, invalid)  # type: ignore[arg-type]

    def test_none_means_no_audio_dimension(self) -> None:
        assert _with_audio_ms(None, None) is None


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
    config = MagicMock()
    config.sie_id = "test/model"
    config.outputs = ["dense"]
    config.resolve_profile.return_value.runtime = {}
    reg.get_config.return_value = config
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
# Sampled video frames bill as images (§7, issue #2433)
# ---------------------------------------------------------------------------
#
# A video item's wire payload is opaque compressed bytes, so the wire-derived
# ``count_input_images`` hook cannot know its billable image count. The
# video-capable adapter therefore stamps the frames it ACTUALLY sampled on
# ``EncodeOutput.extra["input_image_counts"]``; the pipeline lifts them onto
# ``RequestTiming`` and the encode seam prefers them over the hook. Nothing
# processes a frame without settling it, and nothing settles a frame it did not
# process.


class _FakeVideoEncodeAdapter(BaseAdapter):
    """Video-capable encoder (qwen3_vl_embedding shape): stamps the frames it
    processed as the item's authoritative image count.
    """

    spec = AdapterSpec(inputs=("text", "image", "video"), outputs=("dense",), unload_fields=("_model",))

    def __init__(self, *, frames_per_video: int = 0, counts: Any = None) -> None:
        self._model = object()
        self._device = "cpu"
        self._frames_per_video = frames_per_video
        self._counts = counts

    def load(self, device: str) -> None:  # pragma: no cover - not exercised
        _ = device

    def encode(self, items: list[Item], output_types: list[str], **_: Any) -> EncodeOutput:
        _ = output_types
        counts = self._counts
        if counts is None:
            counts = [
                len(item.images or []) + (self._frames_per_video if item.video is not None else 0) for item in items
            ]
        return EncodeOutput(
            dense=np.zeros((len(items), 4), dtype=np.float32),
            batch_size=len(items),
            extra={"input_image_counts": counts},
        )


def _vid() -> dict[str, Any]:
    return {"data": b"fake-mp4-bytes", "format": "mp4"}


class TestEncodePipelineVideoFrames:
    async def _run(self, adapter: BaseAdapter, items: list[Item]) -> RequestTiming:
        reg = _registry_for(adapter)
        _formatted, timing = await EncodePipeline.run_encode(
            registry=reg,
            model="m",
            items=items,
            output_types=["dense"],
            instruction=None,
            config=MagicMock(),
            is_query=False,
            options={},
        )
        return timing

    @pytest.mark.asyncio
    async def test_pipeline_lifts_processed_frame_counts(self) -> None:
        adapter = _FakeVideoEncodeAdapter(frames_per_video=6)
        timing = await self._run(adapter, [Item(video=_vid()), Item(text="text only")])
        assert timing.input_image_counts == [6, 0]

    @pytest.mark.asyncio
    async def test_misaligned_counts_are_dropped_not_mis_attributed(self) -> None:
        adapter = _FakeVideoEncodeAdapter(counts=[3])
        timing = await self._run(adapter, [Item(video=_vid()), Item(video=_vid())])
        assert timing.input_image_counts is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [[-1], [True], ["4"], [1.5], "not-a-list"])
    async def test_malformed_counts_are_dropped(self, bad: Any) -> None:
        timing = await self._run(_FakeVideoEncodeAdapter(counts=bad), [Item(video=_vid())])
        assert timing.input_image_counts is None

    @pytest.mark.asyncio
    async def test_adapters_that_stamp_nothing_leave_counts_unset(self) -> None:
        # Every non-video adapter is unchanged: no ``extra`` stamp, so the
        # result path stays on the wire-derived hook.
        timing = await self._run(_FakeEncodeAdapter(), [Item(text="alpha")])
        assert timing.input_image_counts is None


class _FusingWorker:
    """Fake ModelWorker that fans out through the REAL ``EncodeHandler``.

    Reproduces the batched worker path exactly: the adapter is called once for
    the whole GPU batch, ``slice_output`` splits that output per item and
    ``assemble_output`` rebuilds this request's own output.

    ``sibling_items`` are another request's items fused into the same batch, so
    this request's slices start at a non-zero offset — the cross-request shape
    the sidecar produces whenever two tenants' encodes share a sub-group.
    """

    def __init__(self, adapter: BaseAdapter, sibling_items: list[Item] | None = None) -> None:
        self._adapter = adapter
        self._sibling_items = sibling_items or []
        self.fused_batch_size = 0

    async def submit_preformed(self, **kwargs: Any) -> asyncio.Future[WorkerResult]:
        items = kwargs["items"]
        handler = EncodeHandler()
        fused = [*self._sibling_items, *items]
        self.fused_batch_size = len(fused)
        output = handler.encode(
            adapter=self._adapter,
            items=fused,
            output_types=kwargs["output_types"],
            is_query=kwargs["is_query"],
            options=kwargs["options"],
        )
        offset = len(self._sibling_items)
        partials = {index: handler.slice_output(output, offset + index) for index in range(len(items))}
        future: asyncio.Future[WorkerResult] = asyncio.Future()
        future.set_result(WorkerResult(output=handler.assemble_output(partials, len(items)), timing=kwargs["timing"]))
        return future

    submit = submit_preformed


def _batched_registry(adapter: BaseAdapter, worker: _FusingWorker) -> MagicMock:
    """Registry whose model takes the BATCHED worker path.

    ``config.inputs.image`` is declared and no image preprocessor is registered
    — exactly ``models/Qwen__Qwen3-VL-Embedding-2B.yaml`` — so
    ``_prepare_batch`` returns the passthrough ``PreparedBatch`` and
    ``run_encode`` submits to the worker instead of calling the adapter direct.
    """
    reg = MagicMock()
    reg.preprocessor_registry.has_preprocessor.return_value = False
    reg.postprocessor_registry.transform_sync.return_value = 0.0
    reg.get.return_value = adapter
    reg.start_worker = AsyncMock(return_value=worker)
    return reg


class TestBatchedWorkerPathCarriesFrameCounts:
    """The settlement basis must survive the batched worker path.

    ``EncodeHandler.slice_output`` / ``assemble_output`` rebuild
    ``EncodeOutput.extra`` from scratch, and used to copy only
    ``input_token_counts``. Since ANY item carrying images routes the whole
    sub-batch through the worker, a co-batched video encode arrived at the
    meter with no frame count at all: a silent 32x under-bill when the item
    also carried images, and a hard ``MeteringFault`` (which discards the
    computed embeddings) when it did not.
    """

    @staticmethod
    async def _run(
        adapter: BaseAdapter,
        items: list[Item],
        sibling_items: list[Item] | None = None,
    ) -> tuple[RequestTiming, _FusingWorker]:
        worker = _FusingWorker(adapter, sibling_items)
        config = MagicMock()
        config.inputs.image = True
        _formatted, timing = await EncodePipeline.run_encode(
            registry=_batched_registry(adapter, worker),
            model="m",
            items=items,
            output_types=["dense"],
            instruction=None,
            config=config,
            is_query=False,
            options={},
            preformed_batch=True,
        )
        return timing, worker

    @pytest.mark.asyncio
    async def test_video_co_batched_with_another_requests_images(self) -> None:
        # Tenant A encodes a video; tenant B concurrently encodes a plain image
        # with the same options, so the sidecar fuses them into one sub-group —
        # and B's images are what force the worker path onto A.
        adapter = _FakeVideoEncodeAdapter(frames_per_video=32)
        timing, worker = await self._run(
            adapter,
            [Item(video=_vid()), Item(images=[_img()])],
            sibling_items=[Item(images=[_img(), _img()])],
        )
        assert worker.fused_batch_size == 3  # the batch really was fused
        assert timing.input_image_counts == [32, 1]  # each item keeps its own count

    @pytest.mark.asyncio
    async def test_a_lone_video_item_keeps_its_count_on_the_batched_path(self) -> None:
        # One request, one item carrying an image AND a video: 1 + 32 = 33
        # processed images. ``assemble_output`` takes its batch_size==1 fast
        # path over an already-sliced partial, so the count was dropped here
        # too — 33 images settling as the 1 the wire hook can see.
        adapter = _FakeVideoEncodeAdapter(frames_per_video=32)
        timing, _worker = await self._run(adapter, [Item(images=[_img()], video=_vid())])
        assert timing.input_image_counts == [33]

    def test_slice_and_assemble_preserve_both_unit_dimensions(self) -> None:
        # The dimensions are independent: a video batch carries image counts
        # with no token counts at all, so the all-or-nothing rule is per key.
        handler = EncodeHandler()
        output = EncodeOutput(
            dense=np.zeros((2, 4), dtype=np.float32),
            batch_size=2,
            extra={"input_token_counts": [7, 9], "input_image_counts": [1, 33]},
        )
        partials = {index: handler.slice_output(output, index) for index in range(2)}
        assembled = handler.assemble_output(partials, batch_size=2)
        assert assembled.extra["input_token_counts"] == [7, 9]
        assert assembled.extra["input_image_counts"] == [1, 33]

    def test_each_request_assembles_only_its_own_slices(self) -> None:
        # One fused GPU batch, two requests: item 0 is one tenant's image,
        # items 1-2 are another's videos.
        handler = EncodeHandler()
        output = EncodeOutput(
            dense=np.zeros((3, 4), dtype=np.float32),
            batch_size=3,
            extra={"input_image_counts": [1, 32, 7]},
        )
        first = handler.assemble_output({0: handler.slice_output(output, 0)}, batch_size=1)
        second = handler.assemble_output(
            {0: handler.slice_output(output, 1), 1: handler.slice_output(output, 2)},
            batch_size=2,
        )
        assert first.extra["input_image_counts"] == [1]
        assert second.extra["input_image_counts"] == [32, 7]

    def test_a_partial_without_a_count_drops_that_dimension(self) -> None:
        # All-or-nothing per key: one partial missing its single-element list
        # means the request cannot be attributed exactly, so the dimension is
        # dropped rather than assembled from a subset. Assembling the partial
        # evidence instead would UNDER-bill (the missing item silently counts
        # as nothing); dropping it leaves the meter on its reserve estimate.
        handler = EncodeHandler()
        good = EncodeOutput(
            dense=np.zeros((1, 4), dtype=np.float32),
            batch_size=1,
            extra={"input_image_counts": [5], "input_token_counts": [7]},
        )
        # Same shape, but the image count never made it onto this partial.
        bare = EncodeOutput(
            dense=np.zeros((1, 4), dtype=np.float32),
            batch_size=1,
            extra={"input_token_counts": [9]},
        )
        assembled = handler.assemble_output({0: good, 1: bare}, batch_size=2)

        assert "input_image_counts" not in assembled.extra
        # The keys are independent: the intact dimension still assembles.
        assert assembled.extra["input_token_counts"] == [7, 9]


class TestVideoWithoutAnAuthoritativeCountFailsClosed:
    """A video item never falls back to the wire-derived hook.

    ``count_input_images`` sees opaque compressed bytes and scores them zero,
    so using it for a video item would settle the sampled frames as free.
    Leaving the count unset instead makes settlement refuse the item for want
    of evidence — visible, not a silent revenue leak.
    """

    @staticmethod
    def _run_encode_without_a_stamp() -> AsyncMock:
        """``run_encode`` that records no authoritative image counts at all."""

        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            return [{"dense": [0.0]} for _ in kwargs["items"]], RequestTiming()

        return AsyncMock(side_effect=fake_run_encode)

    @pytest.mark.asyncio
    async def test_video_item_settles_no_images_when_the_stamp_is_missing(self) -> None:
        ex = QueueExecutor(_encode_registry(_FakeVisionEncodeAdapter()))
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode_without_a_stamp()):
            outcome = await ex.process_encode_batch(
                _encode_request([{"images": [_img()], "video": _vid()}, {"images": [_img(), _img()]}])
            )

        # The video item declines the hook's under-count of 1 …
        assert outcome.outcomes[0].units is None
        # … while its plain-image sibling keeps its exact wire-derived count.
        assert outcome.outcomes[1].units is not None
        assert outcome.outcomes[1].units.images == 2


class TestMixedBatchDimensionsAgreeWithThePlan:
    """A mixed text+video batch must report BOTH dimensions.

    The gateway plans ``input_tokens`` whenever any item carries tokenizable
    text and ``images`` whenever any item carries video, then settles only if
    the terminal reports exactly the planned dimensions: a planned dimension
    the terminal omits, or a priced one it reports unplanned, faults the whole
    dispatch after the GPU is already spent.

    Since #2538 a visual-only item reports its token zero rather than hiding
    it, because that zero now carries its own witness — the item's positive
    image count — and ``meter::zero_is_authoritative`` admits exactly that
    shape. Credits are unchanged either way (a zero rates to nothing); what the
    zero buys is a chunk in which EVERY item took the image tower, where the
    dimension would otherwise be absent and settlement would fault it as
    reserved-but-missing.
    """

    @staticmethod
    def _run_encode(token_counts: list[int] | None, image_counts: list[int] | None) -> AsyncMock:
        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            timing = RequestTiming()
            timing.input_token_counts = token_counts
            timing.input_image_counts = image_counts
            return [{"dense": [0.0]} for _ in kwargs["items"]], timing

        return AsyncMock(side_effect=fake_run_encode)

    @pytest.mark.asyncio
    async def test_text_and_video_each_settle_their_own_dimension(self) -> None:
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter(frames_per_video=32)))
        # The adapter scatters 0 text tokens to the video item (siglip shape).
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode([4, 0], [0, 32])):
            outcome = await ex.process_encode_batch(_encode_request([{"text": "alpha beta"}, {"video": _vid()}]))

        text_units = outcome.outcomes[0].units
        video_units = outcome.outcomes[1].units
        assert text_units is not None
        assert text_units.input_tokens == 4
        assert text_units.images is None
        assert video_units is not None
        assert video_units.images == 32
        # The zero is REPORTED (#2538): its own image count witnesses it, and
        # the gateway admits exactly that shape. It rates to nothing, so the
        # request's debit is identical to what it was when the zero was hidden.
        assert video_units.input_tokens == 0

    def test_an_image_witnessed_zero_token_count_is_reported(self) -> None:
        units = _encode_units(0, 32)
        assert units is not None
        assert units.input_tokens == 0
        assert units.images == 32

    def test_an_unwitnessed_zero_token_count_is_still_dropped(self) -> None:
        # THE rail. Without a positive image count a zero explains nothing —
        # it is a tokenizer that counted nothing on a text item — so it stays
        # off the wire and the gateway keeps failing it closed. Byte for byte
        # the rule `meter::zero_is_authoritative` applies on the other side.
        units = _encode_units(0, None)
        assert units is None
        assert _encode_units(0, 0) is None
        # And a positive count is unaffected in either direction.
        positive = _encode_units(7, None)
        assert positive is not None
        assert positive.input_tokens == 7

    def test_an_item_with_only_zeros_yields_no_units(self) -> None:
        assert _encode_units(0, 0) is None


class TestMalformedVideoIsolation:
    """One caller's undecodable video must not fail its co-batched siblings.

    ``process_encode_batch`` fuses items from DIFFERENT API requests into one
    sub-group, and the direct-adapter path had no ``_isolate_invalid_input``
    equivalent — so a single ``VideoDecodeError`` out of ``run_encode`` was
    applied to every item in the group, 400-ing tenants whose input was fine.
    """

    @staticmethod
    def _request(specs: list[tuple[str, dict[str, Any]]]) -> ProcessEncodeBatchRequest:
        """Build one work batch from ``(request_id, item)`` pairs."""
        totals: dict[str, int] = {}
        for request_id, _item in specs:
            totals[request_id] = totals.get(request_id, 0) + 1
        seen: dict[str, int] = {}
        items = []
        for request_id, item in specs:
            index = seen.get(request_id, 0)
            seen[request_id] = index + 1
            items.append(
                EncodeBatchItem(
                    work_item_id=f"{request_id}.{index}",
                    request_id=request_id,
                    item_index=index,
                    total_items=totals[request_id],
                    timestamp=time.time(),
                    item=item,
                )
            )
        return ProcessEncodeBatchRequest(model_id="test/model", items=items)

    @staticmethod
    def _run_encode_failing_on_video() -> AsyncMock:
        """``run_encode`` that raises exactly where ``extract_frames`` does."""

        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            items = kwargs["items"]
            if any(item.video is not None for item in items):
                raise VideoDecodeError("video input could not be opened by the decoder")
            timing = RequestTiming()
            timing.input_token_counts = [4] * len(items)
            return [{"dense": [0.0]} for _ in items], timing

        return AsyncMock(side_effect=fake_run_encode)

    @pytest.mark.asyncio
    async def test_only_the_offending_request_fails(self) -> None:
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter()))
        request = self._request(
            [
                ("req-a", {"text": "alpha"}),
                ("req-bad", {"video": _vid()}),
                ("req-c", {"text": "gamma"}),
                ("req-d", {"text": "delta"}),
            ]
        )
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode_failing_on_video()):
            outcome = await ex.process_encode_batch(request)

        by_id = {o.work_item_id: o for o in outcome.outcomes}
        assert by_id["req-bad.0"].disposition == "publish_error_and_ack"
        assert by_id["req-bad.0"].error_code == ErrorCode.INVALID_INPUT.value
        assert by_id["req-bad.0"].units is None
        for good in ("req-a.0", "req-c.0", "req-d.0"):
            assert by_id[good].disposition == "publish_and_ack", good
            assert by_id[good].units is not None, good
            assert by_id[good].units.input_tokens == 4, good

    @pytest.mark.asyncio
    async def test_the_offending_request_stays_atomic(self) -> None:
        # Matches the worker-batched ``_isolate_invalid_input`` semantics: the
        # split is by request identity, so a multi-item request fails whole
        # rather than publishing a half-successful response.
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter()))
        request = self._request(
            [
                ("req-bad", {"text": "caption"}),
                ("req-ok", {"text": "alpha"}),
                ("req-bad", {"video": _vid()}),
            ]
        )
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode_failing_on_video()):
            outcome = await ex.process_encode_batch(request)

        by_id = {o.work_item_id: o for o in outcome.outcomes}
        assert by_id["req-bad.0"].disposition == "publish_error_and_ack"
        assert by_id["req-bad.1"].disposition == "publish_error_and_ack"
        assert by_id["req-ok.0"].disposition == "publish_and_ack"

    @pytest.mark.asyncio
    async def test_a_lone_malformed_request_still_reports_invalid_input(self) -> None:
        # The terminal case of the recursion, unchanged from before the split.
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter()))
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode_failing_on_video()):
            outcome = await ex.process_encode_batch(self._request([("req-solo", {"video": _vid()})]))

        assert outcome.outcomes[0].error_code == ErrorCode.INVALID_INPUT.value

    @pytest.mark.asyncio
    async def test_isolating_one_bad_request_is_a_binary_search(self) -> None:
        # The case isolation exists for: only the half holding the bad request
        # fails, so the recursion is a true binary search — ~2*log2(R) passes,
        # nowhere near the budget.
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter()))
        specs: list[tuple[str, dict[str, Any]]] = [(f"req-{i}", {"text": f"doc {i}"}) for i in range(15)]
        specs.insert(9, ("req-bad", {"video": _vid()}))
        run_encode = self._run_encode_failing_on_video()
        with patch.object(EncodePipeline, "run_encode", new=run_encode):
            outcome = await ex.process_encode_batch(self._request(specs))

        by_id = {o.work_item_id: o for o in outcome.outcomes}
        assert by_id["req-bad.0"].error_code == ErrorCode.INVALID_INPUT.value
        assert all(by_id[f"req-{i}.0"].disposition == "publish_and_ack" for i in range(15))
        # 16 requests -> at most ~2*log2(16) + 1 passes, well under the budget.
        assert run_encode.await_count <= 12, run_encode.await_count

    @pytest.mark.asyncio
    async def test_an_environmental_failure_cannot_fan_out_unbounded(self) -> None:
        # The broken-OpenCV-wheel shape: EVERY video request fails, so both
        # halves of every split fail and unbounded bisection would degenerate
        # to ~2R re-encode passes on the single-threaded inference executor.
        # The batch's isolation budget caps that; every request still gets the
        # typed INVALID_INPUT it was always going to get.
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter()))
        specs: list[tuple[str, dict[str, Any]]] = [(f"req-{i}", {"video": _vid()}) for i in range(32)]
        run_encode = self._run_encode_failing_on_video()
        with patch.object(EncodePipeline, "run_encode", new=run_encode):
            outcome = await ex.process_encode_batch(self._request(specs))

        assert len(outcome.outcomes) == 32
        assert all(o.error_code == ErrorCode.INVALID_INPUT.value for o in outcome.outcomes)
        assert all(o.units is None for o in outcome.outcomes)
        # Unbounded bisection would be ~2*32 = 64 passes; the budget holds it
        # to the cap plus the one initial pass per split branch.
        assert run_encode.await_count <= _MAX_ENCODE_ISOLATION_PASSES + 2, run_encode.await_count


class TestEncodeSeamVideoFrames:
    @staticmethod
    def _run_encode(image_counts: list[int] | None, token_counts: list[int] | None = None) -> Any:
        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            timing = RequestTiming()
            timing.input_image_counts = image_counts
            timing.input_token_counts = token_counts
            return [{"dense": [0.0]} for _ in kwargs["items"]], timing

        return AsyncMock(side_effect=fake_run_encode)

    @pytest.mark.asyncio
    async def test_video_only_item_settles_processed_frames_as_images(self) -> None:
        # Previously this item faulted ("all failed to load") and settled
        # nothing; now it settles the exact frames the model saw.
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter(frames_per_video=9)))
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode([9])):
            outcome = await ex.process_encode_batch(_encode_request([{"video": _vid()}]))

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.units is not None
        assert o.units.images == 9
        assert o.units.input_tokens is None

    @pytest.mark.asyncio
    async def test_mixed_text_and_video_settles_both_dimensions(self) -> None:
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter(frames_per_video=5)))
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode([5], [12])):
            outcome = await ex.process_encode_batch(_encode_request([{"text": "a caption", "video": _vid()}]))

        o = outcome.outcomes[0]
        assert o.units is not None
        assert o.units.input_tokens == 12
        assert o.units.images == 5

    @pytest.mark.asyncio
    async def test_processed_count_wins_over_the_wire_derived_hook(self) -> None:
        # The wire item carries ONE image and one opaque video; the hook would
        # bill 1. The adapter processed 1 image + 4 frames -> 5 billable images.
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter(frames_per_video=4)))
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode([5])):
            outcome = await ex.process_encode_batch(_encode_request([{"images": [_img()], "video": _vid()}]))

        assert outcome.outcomes[0].units is not None
        assert outcome.outcomes[0].units.images == 5

    @pytest.mark.asyncio
    async def test_per_item_counts_stay_aligned_across_a_batch(self) -> None:
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter(frames_per_video=3)))
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode([0, 3])):
            outcome = await ex.process_encode_batch(_encode_request([{"text": "alpha"}, {"video": _vid()}]))

        assert outcome.outcomes[0].units is None  # text-only, no token count either
        assert outcome.outcomes[1].units is not None
        assert outcome.outcomes[1].units.images == 3

    @pytest.mark.asyncio
    async def test_undecodable_video_settles_nothing_as_typed_invalid_input(self) -> None:
        # No path settles frames it did not process: a decode failure is a
        # typed 4xx with no units, not a billed success missing its video.
        ex = QueueExecutor(_encode_registry(_FakeVideoEncodeAdapter()))
        failing = AsyncMock(side_effect=VideoDecodeError("video input could not be opened by the decoder"))
        with patch.object(EncodePipeline, "run_encode", new=failing):
            outcome = await ex.process_encode_batch(_encode_request([{"video": _vid()}]))

        o = outcome.outcomes[0]
        assert o.disposition == "publish_error_and_ack"
        assert o.error_code == ErrorCode.INVALID_INPUT.value
        assert o.units is None

    @pytest.mark.asyncio
    async def test_absent_processed_counts_fall_back_to_the_wire_hook(self) -> None:
        # Regression guard for every existing image adapter: with no ``extra``
        # stamp the seam still bills submitted images off the shared hook.
        ex = QueueExecutor(_encode_registry(_FakeVisionEncodeAdapter()))
        with patch.object(EncodePipeline, "run_encode", new=self._run_encode(None)):
            outcome = await ex.process_encode_batch(_encode_request([{"images": [_img(), _img()]}]))

        assert outcome.outcomes[0].units is not None
        assert outcome.outcomes[0].units.images == 2


# ---------------------------------------------------------------------------
# Measured modality routing — the authoritative-zero basis (#2538)
# ---------------------------------------------------------------------------
#
# SigLIP/CLIP route ANY item carrying images to the image tower, even one that
# also carries text, so an item's text-token count can be a genuine zero. Only
# the adapter's own partition knows that; nothing downstream can re-derive it
# from the wire item, because "has text" does not imply "was tokenized". The
# adapters publish the partition on ``extra["text_tower_skipped"]`` and the
# pipeline uses it as the LAST-resort count basis — after every real tokenizer
# path has failed — so batches that count today keep counting exactly as before.


class TestMeasuredModalityRouting:
    def test_clip_publishes_its_partition(self) -> None:
        adapter = TestVisionTextTokenStamp()._clip()
        items = [Item(text="alpha beta"), Item(images=[{"data": b"x"}]), Item(text="one two three")]
        out = adapter.encode(items, ["dense"])
        assert out.extra["text_tower_skipped"] == [False, True, False]

    def test_an_item_carrying_both_text_and_images_is_reported_as_skipped(self) -> None:
        # The routing fact that finding 2 turns on: `has_images` wins, so this
        # item never reaches the text tower and its text is never billed.
        adapter = TestVisionTextTokenStamp()._clip()
        out = adapter.encode([Item(text="a photo of a cat", images=[{"data": b"x"}])], ["dense"])
        assert out.extra["text_tower_skipped"] == [True]
        assert "input_token_counts" not in out.extra

    def test_the_partition_is_per_item_sliceable(self) -> None:
        # A fused GPU batch is split back into per-request outputs; a per-item
        # extra that is not registered for slicing arrives misaligned, which
        # would mis-attribute the zero to the wrong caller.
        from sie_server.core.worker.handlers.encode import PER_ITEM_EXTRA_KEYS

        assert "text_tower_skipped" in PER_ITEM_EXTRA_KEYS


class TestWhollyImageTowerBatchCounts:
    """The pipeline's last-resort basis: no tokenizer could count, but the
    adapter measured that every item skipped the text tower.
    """

    @staticmethod
    def _adapter(skipped: list[bool] | None, counts: list[int] | None) -> Any:
        adapter = MagicMock()
        adapter.count_input_tokens.return_value = counts
        adapter._skipped = skipped
        return adapter

    @staticmethod
    def _mixed(n: int = 2) -> list[Item]:
        """Text+image items — the SigLIP shape where `has_images` wins, so the
        item carries text the gateway reserved for and still skips the tower.
        """
        return [Item(text="a cat", images=[_img()]) for _ in range(n)]

    def test_a_wholly_skipped_text_bearing_partition_is_an_authoritative_zero(self) -> None:
        assert _wholly_skipped_text_tower_zeros([True, True], self._mixed(2)) == [0, 0]
        assert _wholly_skipped_text_tower_zeros([True], self._mixed(1)) == [0]

    def test_any_unskipped_item_keeps_the_dimension_absent(self) -> None:
        # THE rail that keeps today's billing intact. A batch with even one
        # text item had text a tokenizer should have counted; zeros here would
        # convert "bill text the model read" into "bill nothing", which is a
        # pricing decision and not a bug fix. `None` sends the meter back to
        # its reserve estimate, exactly as before #2538.
        assert _wholly_skipped_text_tower_zeros([True, False], self._mixed(2)) is None
        assert _wholly_skipped_text_tower_zeros([False, False], self._mixed(2)) is None

    def test_a_pure_image_batch_stays_off_the_token_dimension(self) -> None:
        # `text_tower_skipped` is True for a pure-image item exactly as it is
        # for a text+image one, so without the text condition this fallback
        # would put `input_tokens = 0` on a request that never carried text and
        # never reserved the dimension. The SigLIP/CLIP adapters already refuse
        # to stamp `input_token_counts` for such a batch on purpose; #2538 must
        # not reverse that from one layer up.
        image_only = [Item(images=[_img()]), Item(images=[_img()])]
        assert _wholly_skipped_text_tower_zeros([True, True], image_only) is None
        # An empty string is not text either.
        assert _wholly_skipped_text_tower_zeros([True], [Item(text="", images=[_img()])]) is None

    def test_a_malformed_or_misaligned_partition_is_dropped(self) -> None:
        # Same contract as every other metering basis (`_validated_counts`):
        # a partition that cannot be attributed exactly is not a measurement.
        # `1` is truthy and would pass a bare `all()`, so the bool check is
        # load-bearing rather than decorative.
        assert _wholly_skipped_text_tower_zeros([True, True], self._mixed(3)) is None
        assert _wholly_skipped_text_tower_zeros("TT", self._mixed(2)) is None
        assert _wholly_skipped_text_tower_zeros(None, self._mixed(2)) is None
        assert _wholly_skipped_text_tower_zeros([1, 1], self._mixed(2)) is None
        # An empty partition satisfies `all()` vacuously and must NOT mint a
        # bare unwitnessed zero.
        assert _wholly_skipped_text_tower_zeros([], []) is None

    @pytest.mark.asyncio
    async def test_all_skipped_settles_with_images_end_to_end(self) -> None:
        ex = QueueExecutor(_encode_registry(_FakeVisionEncodeAdapter()))

        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            timing = RequestTiming()
            timing.input_image_counts = [1, 1]
            # What the pipeline's last-resort basis produces for this batch.
            timing.input_token_counts = _wholly_skipped_text_tower_zeros([True, True], kwargs["items"])
            return [{"dense": [0.0]} for _ in kwargs["items"]], timing

        with patch.object(EncodePipeline, "run_encode", new=AsyncMock(side_effect=fake_run_encode)):
            outcome = await ex.process_encode_batch(
                _encode_request([{"text": "a cat", "images": [_img()]}, {"text": "a dog", "images": [_img()]}])
            )
        # Both dimensions reach the wire: the images that bill, and the token
        # zero whose image witness lets settlement release that dimension
        # instead of faulting the dispatch as reserved-but-missing.
        for item in outcome.outcomes:
            assert item.disposition == "publish_and_ack"
            assert item.units is not None
            assert item.units.images == 1
            assert item.units.input_tokens == 0

    @pytest.mark.asyncio
    async def test_an_image_only_request_never_gains_a_token_dimension(self) -> None:
        # The end-to-end half of the pure-image rail: an image-only request
        # reserved no `input_tokens`, so its terminal must not carry one. This
        # is the shape the last-resort basis would have captured if it keyed on
        # `text_tower_skipped` alone.
        ex = QueueExecutor(_encode_registry(_FakeVisionEncodeAdapter()))

        async def fake_run_encode(**kwargs: Any) -> tuple[list[dict[str, Any]], RequestTiming]:
            timing = RequestTiming()
            timing.input_image_counts = [1, 1]
            timing.input_token_counts = _wholly_skipped_text_tower_zeros([True, True], kwargs["items"])
            return [{"dense": [0.0]} for _ in kwargs["items"]], timing

        with patch.object(EncodePipeline, "run_encode", new=AsyncMock(side_effect=fake_run_encode)):
            outcome = await ex.process_encode_batch(_encode_request([{"images": [_img()]}, {"images": [_img()]}]))
        for item in outcome.outcomes:
            assert item.disposition == "publish_and_ack"
            assert item.units is not None
            assert item.units.images == 1
            assert item.units.input_tokens is None

    def test_pipeline_only_uses_the_partition_when_nothing_else_counted(self) -> None:
        # Ordering, which no unit test of the helper can observe: the branch
        # sits AFTER the `count_input_tokens` fallback, so any batch a real
        # tokenizer could count keeps those numbers.
        source = pathlib.Path(EncodePipeline.__module__.replace(".", "/") + ".py")
        text = (pathlib.Path(__file__).parents[3] / "packages/sie_server/src" / source).read_text()
        fallback = text.index("adapter.count_input_tokens")
        # The CALL site, not the helper's definition (which sits above both).
        last_resort = text.index("timing.input_token_counts = _wholly_skipped_text_tower_zeros(")
        assert fallback < last_resort, "the last-resort basis must not pre-empt a real tokenizer"


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

    def count_input_images(self, items: list[Item]) -> None:
        del items

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
    async def test_ocr_image_stamps_pages_without_image_double_count(self) -> None:
        adapter = _FakeParseExtractAdapter()
        extract_output = ExtractOutput(
            entities=[[{"text": "# page", "label": "markdown", "score": 1.0}]],
            pages=[1],
        )
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)

        outcome = await ex.process_extract_batch(_extract_request({"images": [{"data": b"png", "format": "png"}]}))

        units = outcome.outcomes[0].units
        assert units is not None
        assert units.pages == 1
        assert units.images is None

    @pytest.mark.asyncio
    async def test_parse_extract_error_retains_authoritative_pages(self) -> None:
        adapter = _FakeParseExtractAdapter()
        extract_output = ExtractOutput(
            entities=[[]],
            data=[{"partial": True}],
            errors=[ExtractItemError(code="INFERENCE_ERROR", message="Document export failed")],
            pages=[3],
        )
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)

        outcome = await ex.process_extract_batch(
            _extract_request({"document": {"data": b"%PDF-1.4 ...", "format": "pdf"}})
        )

        item = outcome.outcomes[0]
        assert item.disposition == "publish_and_ack"
        assert item.result_msgpack is not None
        result = msgpack.unpackb(item.result_msgpack, raw=False)
        assert result["data"] == {"partial": True}
        assert result["error"] == {
            "code": "INFERENCE_ERROR",
            "message": "Document export failed",
        }
        assert item.units is not None
        assert item.units.pages == 3

    @pytest.mark.asyncio
    async def test_parse_extract_zero_pages_remains_authoritative(self) -> None:
        adapter = _FakeParseExtractAdapter()
        extract_output = ExtractOutput(
            entities=[[]],
            data=[{}],
            errors=[ExtractItemError(code="INFERENCE_ERROR", message="Document conversion failed")],
            pages=[0],
        )
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)

        outcome = await ex.process_extract_batch(_extract_request({"document": {"data": b"garbage", "format": "pdf"}}))

        item = outcome.outcomes[0]
        assert item.disposition == "publish_and_ack"
        assert item.units is not None
        assert item.units.pages == 0

    @pytest.mark.asyncio
    async def test_parse_extract_without_pages_leaves_units_unset(self) -> None:
        # A parser that cannot surface a page count leaves the dimension missing,
        # so the meter retains its compatibility fallback.
        adapter = _FakeParseExtractAdapter()
        extract_output = ExtractOutput(
            entities=[[]],
            data=[{}],
            errors=[ExtractItemError(code="INFERENCE_ERROR", message="Document conversion failed")],
            pages=None,
        )
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)

        outcome = await ex.process_extract_batch(_extract_request({"document": {"data": b"garbage", "format": "pdf"}}))

        assert outcome.outcomes[0].disposition == "publish_and_ack"
        assert outcome.outcomes[0].result_msgpack is not None
        result = msgpack.unpackb(outcome.outcomes[0].result_msgpack, raw=False)
        assert result["error"]["code"] == "INFERENCE_ERROR"
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

        # Stub the text forward with SigLIP's fixed padded-work billing count.
        def fake_encode_texts(texts: list[str]) -> tuple[Any, list[int] | None]:
            return np.zeros((len(texts), 4), dtype=np.float32), [adapter._max_seq_length] * len(texts)

        adapter._encode_texts = fake_encode_texts  # type: ignore[method-assign]
        out = adapter.encode([Item(text="a b c")], ["dense"])
        assert out.extra["input_token_counts"] == [64]


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


class TestExtractSeamAudio:
    @pytest.mark.asyncio
    async def test_whisper_extract_stamps_exact_audio_ms(self) -> None:
        adapter = _FakeTextExtractAdapter()
        extract_output = ExtractOutput(
            entities=[[]],
            data=[{"text": "hello", "duration_ms": 1_001}],
        )
        reg = _extract_registry(adapter, _extract_worker(extract_output))
        ex = QueueExecutor(reg)
        prepared_audio = PreparedAudioPcm16(
            pcm_s16le=b"\x00\x00" * 16_016,
            sample_rate=16_000,
            sample_count=16_016,
            duration_ms=1_001,
            source_sample_rate=16_000,
            source_sample_count=16_016,
            source_channels=1,
            container="wav",
        )
        request = ProcessExtractBatchRequest(
            model_id="test/model",
            items=[
                ExtractBatchItem(
                    work_item_id="req-1.0",
                    request_id="req-1",
                    item_index=0,
                    total_items=1,
                    timestamp=time.time(),
                    item={},
                    prepared_audio=prepared_audio,
                )
            ],
        )

        outcome = await ex.process_extract_batch(request)

        result = outcome.outcomes[0]
        assert result.disposition == "publish_and_ack"
        assert result.units is not None
        assert result.units.audio_ms == 1_001
        assert result.units.input_tokens is None
        assert result.units.images is None
        assert result.units.pages is None

    @pytest.mark.asyncio
    async def test_partial_audio_batch_emits_units_only_for_success(self) -> None:
        adapter = _FakeTextExtractAdapter()
        extract_output = ExtractOutput(
            entities=[[]],
            data=[{"text": "hello", "duration_ms": 1_001}],
        )
        ex = QueueExecutor(_extract_registry(adapter, _extract_worker(extract_output)))
        valid_audio = PreparedAudioPcm16(
            pcm_s16le=b"\x00\x00" * 16_016,
            sample_rate=16_000,
            sample_count=16_016,
            duration_ms=1_001,
            source_sample_rate=16_000,
            source_sample_count=16_016,
            source_channels=1,
            container="wav",
        )
        invalid_audio = PreparedAudioPcm16(
            pcm_s16le=b"\x00\x00",
            sample_rate=16_000,
            sample_count=1,
            duration_ms=1,
            source_sample_rate=0,
            source_sample_count=1,
            source_channels=1,
            container="wav",
        )
        request = ProcessExtractBatchRequest(
            model_id="test/model",
            items=[
                ExtractBatchItem(
                    work_item_id="req-1.0",
                    request_id="req-1",
                    item_index=0,
                    total_items=2,
                    timestamp=time.time(),
                    item={},
                    prepared_audio=valid_audio,
                ),
                ExtractBatchItem(
                    work_item_id="req-1.1",
                    request_id="req-1",
                    item_index=1,
                    total_items=2,
                    timestamp=time.time(),
                    item={},
                    prepared_audio=invalid_audio,
                ),
            ],
        )

        outcome = await ex.process_extract_batch(request)

        success, failure = outcome.outcomes
        assert success.disposition == "publish_and_ack"
        assert success.units is not None
        assert success.units.audio_ms == 1_001
        assert failure.disposition == "publish_error_and_ack"
        assert failure.units is None
