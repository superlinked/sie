"""#2874: text requests to preprocessor-less multimodal encoders must batch.

The negative-return admission diagnosis (issue #2874, Layer A) showed a family
of encode paths where batching never engaged: models without a registered
``"text"`` preprocessor bypassed ``ModelWorker`` entirely and ran one
``asyncio.to_thread`` adapter call per request, serialized across N threads on
the adapters' ``_forward_lock``/``_tokenizer_lock`` — worse than serial under
concurrency (the c=1-peak sweep cells). Affected: the colpali/colqwen/
qwen3-vl-embedding family (no preprocessor at all) and SigLIP/CLIP text
requests (image-only preprocessor registered).

These tests drive the REAL adapter ``get_preprocessor()`` through the same
modality registration ``ModelLoader._finish_load`` performs, then push N
concurrent requests through ``EncodePipeline.run_encode`` with a REAL
``ModelWorker``. They fail if the adapters stop registering a text
preprocessor (reverting to the unbatched fallback makes every adapter call a
batch-of-1, so the fused-batch assertions below trip) — that is the mutation
this suite exists to kill.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from sie_server.adapters.clip import CLIPAdapter
from sie_server.adapters.colpali import ColPaliAdapter
from sie_server.adapters.colqwen2 import ColQwen2Adapter
from sie_server.adapters.colqwen3 import ColQwen3Adapter
from sie_server.adapters.colsmol import ColSmolAdapter
from sie_server.adapters.qwen3_vl_embedding import Qwen3VLEmbeddingAdapter
from sie_server.adapters.siglip.adapter import SiglipAdapter
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.preprocessor.text import CharCountPreprocessor
from sie_server.core.preprocessor_registry import PreprocessorRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.types.inputs import Item

MODEL = "stub/model-2874"


def _register_by_modality(preproc_registry: PreprocessorRegistry, name: str, adapter: Any) -> None:
    """Mirror ``ModelLoader._finish_load``'s modality registration loop.

    Registration is driven by the REAL ``adapter.get_preprocessor()`` so these
    tests fail when an adapter stops advertising a text preprocessor.
    """
    preprocessors = adapter.get_preprocessor()
    if not isinstance(preprocessors, list):
        preprocessors = [preprocessors]
    for preprocessor in preprocessors:
        if preprocessor is None:
            continue
        modality = getattr(preprocessor, "modality", None)
        if modality in ("text", "audio"):
            preproc_registry.register(name, preprocessor)
        elif modality == "image":
            preproc_registry.register_image(name, preprocessor)


def _pipeline_registry(adapter: Any, worker: ModelWorker, preproc_registry: PreprocessorRegistry) -> MagicMock:
    """Registry facade wiring a real preprocessor registry + real worker."""
    reg = MagicMock()
    reg.preprocessor_registry = preproc_registry
    reg.postprocessor_registry.transform_sync.return_value = 0.0
    reg.get.return_value = adapter
    reg.start_worker = AsyncMock(return_value=worker)
    return reg


def _worker_for(adapter: Any) -> ModelWorker:
    """Worker with windows generous enough to fuse a gathered burst."""
    return ModelWorker(
        adapter,
        WorkerConfig(
            max_batch_tokens=100_000,
            max_batch_requests=64,
            max_batch_wait_ms=200.0,
            coalesce_ms=100.0,
            coalesce_ratio=0.5,
            idle_coalesce_ms=50.0,
        ),
        model_name=MODEL,
    )


def _record_encode_calls(adapter: Any) -> list[int]:
    """Wrap the real ``encode`` to record per-call fused batch sizes."""
    calls: list[int] = []
    real_encode = adapter.encode

    def recording_encode(items: list[Item], *args: Any, **kwargs: Any) -> Any:
        calls.append(len(items))
        return real_encode(items, *args, **kwargs)

    adapter.encode = recording_encode
    return calls


async def _run_concurrent(
    adapter: Any,
    *,
    n_requests: int,
    output_types: list[str],
    is_query: bool,
) -> tuple[list[int], ModelWorker]:
    preproc_registry = PreprocessorRegistry(max_workers=2)
    _register_by_modality(preproc_registry, MODEL, adapter)
    worker = _worker_for(adapter)
    await worker.start()
    calls = _record_encode_calls(adapter)
    reg = _pipeline_registry(adapter, worker, preproc_registry)
    config = MagicMock()
    config.inputs.image = True

    async def _one(i: int) -> Any:
        return await EncodePipeline.run_encode(
            registry=reg,
            model=MODEL,
            items=[Item(text=f"query number {i}")],
            output_types=output_types,
            instruction=None,
            config=config,
            is_query=is_query,
            options={},
        )

    try:
        results = await asyncio.gather(*[_one(i) for i in range(n_requests)])
    finally:
        await worker.stop()

    assert len(results) == n_requests
    for formatted, _timing in results:
        assert len(formatted) == 1
    return calls, worker


class TestColFamilyTextQueriesBatchThroughWorker:
    """N concurrent text queries fuse into worker batches, not N direct calls."""

    @pytest.mark.asyncio
    async def test_colqwen2_text_queries_fuse(self) -> None:
        adapter = ColQwen2Adapter("stub/colqwen2")
        adapter._model = object()
        adapter._processor = object()
        adapter._device = "cpu"
        adapter._encode_text = lambda text: np.zeros((3, 4), dtype=np.float32)  # type: ignore[method-assign]

        calls, worker = await _run_concurrent(adapter, n_requests=6, output_types=["multivector"], is_query=True)

        # The unbatched fallback makes exactly 6 direct batch-of-1 calls;
        # the worker path fuses concurrent arrivals into fewer, larger calls.
        assert sum(calls) == 6
        assert len(calls) < 6, f"expected fused worker batches, got per-request calls: {calls}"
        assert max(calls) >= 2, f"expected at least one fused batch, got: {calls}"
        assert worker.stats.batches_processed >= 1

    @pytest.mark.asyncio
    async def test_colpali_text_queries_fuse(self) -> None:
        adapter = ColPaliAdapter("stub/colpali")
        adapter._model = object()
        adapter._processor = object()
        adapter._device = "cpu"
        adapter._encode_text = lambda text: np.zeros((3, 4), dtype=np.float32)  # type: ignore[method-assign]

        calls, worker = await _run_concurrent(adapter, n_requests=6, output_types=["multivector"], is_query=True)

        assert sum(calls) == 6
        assert len(calls) < 6
        assert max(calls) >= 2
        assert worker.stats.batches_processed >= 1


class TestTwinTowerTextPathBatches:
    """SigLIP/CLIP text-only requests fuse AND run as one text-tower forward."""

    @staticmethod
    def _patch_text_tower(adapter: Any) -> list[list[str]]:
        tower_calls: list[list[str]] = []

        def fake_encode_texts(texts: list[str]) -> tuple[list[np.ndarray], list[int]]:
            tower_calls.append(list(texts))
            return [np.zeros(4, dtype=np.float32) for _ in texts], [len(t.split()) for t in texts]

        adapter._encode_texts = fake_encode_texts
        return tower_calls

    @pytest.mark.asyncio
    async def test_siglip_text_requests_fuse_into_one_tower_forward(self) -> None:
        adapter = SiglipAdapter("stub/siglip")
        adapter._model = object()
        adapter._processor = MagicMock()
        tower_calls = self._patch_text_tower(adapter)

        calls, worker = await _run_concurrent(adapter, n_requests=6, output_types=["dense"], is_query=False)

        assert sum(calls) == 6
        assert len(calls) < 6
        assert max(calls) >= 2
        # The fused sub-batch is ONE stacked text-tower forward, not a
        # per-item loop: some tower call carries >= 2 texts.
        assert max(len(texts) for texts in tower_calls) >= 2
        assert worker.stats.batches_processed >= 1

    @pytest.mark.asyncio
    async def test_clip_text_requests_fuse_into_one_tower_forward(self) -> None:
        adapter = CLIPAdapter("stub/clip")
        adapter._model = object()
        adapter._processor = MagicMock()
        tower_calls = self._patch_text_tower(adapter)

        calls, worker = await _run_concurrent(adapter, n_requests=6, output_types=["dense"], is_query=False)

        assert sum(calls) == 6
        assert len(calls) < 6
        assert max(calls) >= 2
        assert max(len(texts) for texts in tower_calls) >= 2
        assert worker.stats.batches_processed >= 1


class TestAffectedAdaptersRegisterTextPreprocessor:
    """Every #2874 Layer-A adapter must advertise a text-modality preprocessor.

    ``_prepare_batch`` routes text-only requests through the worker IFF a
    ``"text"`` preprocessor is registered; without one the request takes the
    unbatched direct path. ``CharCountPreprocessor`` is cost-only, so the
    adapters keep owning their tokenization and the unit meter's bases are
    unchanged (it is skipped by ``_record_input_token_counts``).
    """

    @pytest.mark.parametrize(
        "adapter_factory",
        [
            lambda: ColPaliAdapter("stub/colpali"),
            lambda: ColQwen2Adapter("stub/colqwen2"),
            lambda: ColQwen3Adapter("stub/colqwen3"),
            lambda: ColSmolAdapter("stub/colsmol"),
            lambda: Qwen3VLEmbeddingAdapter("stub/qwen3-vl-embedding"),
            lambda: SiglipAdapter("stub/siglip"),
            lambda: CLIPAdapter("stub/clip"),
        ],
        ids=["colpali", "colqwen2", "colqwen3", "colsmol", "qwen3_vl_embedding", "siglip", "clip"],
    )
    def test_text_preprocessor_registered(self, adapter_factory: Any) -> None:
        adapter = adapter_factory()
        registry = PreprocessorRegistry(max_workers=1)
        _register_by_modality(registry, MODEL, adapter)
        assert registry.has_preprocessor(MODEL, "text"), (
            f"{type(adapter).__name__} must register a text preprocessor so text "
            "requests take the batched ModelWorker path (#2874)"
        )
        text_preprocessor = registry.get_preprocessor(MODEL, "text")
        assert isinstance(text_preprocessor, CharCountPreprocessor)

    @pytest.mark.parametrize(
        "adapter_factory",
        [
            lambda: _loaded(ColPaliAdapter("stub/colpali")),
            lambda: _loaded(SiglipAdapter("stub/siglip")),
            lambda: _loaded(CLIPAdapter("stub/clip")),
        ],
        ids=["colpali", "siglip", "clip"],
    )
    def test_image_preprocessor_still_registered(self, adapter_factory: Any) -> None:
        """Adding the text entry must not drop the existing image entry."""
        adapter = adapter_factory()
        registry = PreprocessorRegistry(max_workers=1)
        _register_by_modality(registry, MODEL, adapter)
        assert registry.has_preprocessor(MODEL, "image")

    @pytest.mark.asyncio
    async def test_prepare_batch_routes_text_to_worker_path(self) -> None:
        """With the text preprocessor registered, ``_prepare_batch`` returns a
        PreparedBatch (worker path) instead of None (direct fallback).
        """
        adapter = ColQwen2Adapter("stub/colqwen2")
        preproc_registry = PreprocessorRegistry(max_workers=1)
        _register_by_modality(preproc_registry, MODEL, adapter)
        reg = MagicMock()
        reg.preprocessor_registry = preproc_registry
        config = MagicMock()
        config.inputs.image = True

        prepared = await EncodePipeline._prepare_batch(
            reg, MODEL, [Item(text="a query")], config, True, RequestTiming()
        )
        assert prepared is not None
        assert len(prepared.items) == 1


def _loaded(adapter: Any) -> Any:
    """Stub just enough load state for image-preprocessor construction."""
    adapter._processor = MagicMock()
    return adapter
