"""Tests for visual document retrieval adapters (ColPali, ColQwen2, NemoColEmbed).

These adapters encode document images into multi-vector representations
for late interaction retrieval using MaxSim scoring.
"""

import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from sie_server.adapters.colpali import ColPaliAdapter
from sie_server.adapters.colqwen2 import ColQwen2Adapter
from sie_server.adapters.colqwen3 import ColQwen3Adapter
from sie_server.adapters.colsmol import ColSmolAdapter
from sie_server.adapters.nemo_colembed import NemoColEmbedAdapter
from sie_server.core.preprocessor import ImagePreprocessor
from sie_server.types.inputs import Item


class _BorrowGuard:
    """Fake HF processor that flags any overlapping (concurrent) entry.

    Stands in for the non-thread-safe Rust fast tokenizer: a second call while
    one is still in flight is the ``RuntimeError: Already borrowed`` that #2098
    serialises away. Returns both ``input_ids`` (text-encode paths) and
    ``pixel_values`` (the ImagePreprocessor path) so it fits either call site.
    """

    def __init__(self, sleep_s: float = 0.02) -> None:
        self._guard = threading.Lock()
        self._sleep_s = sleep_s
        self.active = 0
        self.max_concurrent = 0
        self.borrow_error = False

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        with self._guard:
            if self.active > 0:
                self.borrow_error = True
                raise RuntimeError("Already borrowed")
            self.active += 1
            self.max_concurrent = max(self.max_concurrent, self.active)
        time.sleep(self._sleep_s)
        with self._guard:
            self.active -= 1
        return {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 4, 4)),
        }


class TestColPaliAdapter:
    """Tests for ColPaliAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColPaliAdapter:
        """Create an adapter instance."""
        return ColPaliAdapter(
            "vidore/colpali-v1.3-hf",
            normalize=True,
            compute_precision="float32",
        )

    def test_capabilities(self, adapter: ColPaliAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColPaliAdapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # ColPali default

    def test_encode_before_load_raises(self, adapter: ColPaliAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColPaliAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColPaliAdapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    @staticmethod
    def _wire_fake_model(adapter: ColPaliAdapter, forward: Any) -> MagicMock:
        """Install a fake model/processor pair that satisfies all forward paths."""
        import torch

        model = MagicMock(side_effect=forward)
        adapter._model = model
        adapter._processor = MagicMock(return_value={"input_ids": torch.zeros(1, 4, dtype=torch.long)})
        adapter._device = "cpu"
        # Prepared-batch path uses load()-cached placeholder tokens.
        adapter._cached_input_ids = torch.zeros(1, 4, dtype=torch.long)
        adapter._cached_attention_mask = torch.ones(1, 4, dtype=torch.long)
        return model

    @staticmethod
    def _call_each_forward_path(adapter: ColPaliAdapter, path: str) -> None:
        """Invoke one of the three model-forward entry points."""
        import torch

        if path == "text":
            adapter._encode_text("q")
        elif path == "images":
            adapter._encode_images([MagicMock()])
        else:
            prepared = SimpleNamespace(payload=SimpleNamespace(pixel_values=torch.zeros(3, 4, 4)))
            adapter._encode_prepared_batch([Item(text="d")], [prepared], is_query=False)

    def test_forward_passes_use_cache_false(self, adapter: ColPaliAdapter) -> None:
        """All three forward entry points must suppress the KV cache (#2144).

        PaliGemma's inner Gemma config defaults to use_cache=True and the
        load-time config edit cannot reach it, so the per-forward kwarg is
        the only reliable suppression.
        """
        import torch

        def forward(**kwargs: Any) -> Any:
            return SimpleNamespace(embeddings=torch.zeros(1, 4, 128))

        model = self._wire_fake_model(adapter, forward)

        for path in ("text", "images", "prepared"):
            model.call_args = None
            self._call_each_forward_path(adapter, path)
            assert model.call_args.kwargs["use_cache"] is False, path

    def test_concurrent_forwards_serialize(self, adapter: ColPaliAdapter) -> None:
        """Model forwards never overlap across threads, on any path (#2144).

        transformers' output recorder monkey-patches each decoder layer's
        forward per call with no locking; overlapping forwards race the
        patch/restore and leak every later layer output. The adapter's
        forward lock must serialize all three entry points against each
        other, not just same-path calls.
        """
        import torch

        counter_guard = threading.Lock()
        active = 0
        max_active = 0

        def forward(**kwargs: Any) -> Any:
            nonlocal active, max_active
            with counter_guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with counter_guard:
                active -= 1
            return SimpleNamespace(embeddings=torch.zeros(1, 4, 128))

        self._wire_fake_model(adapter, forward)

        paths = ["text", "images", "prepared"] * 3
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda p: self._call_each_forward_path(adapter, p), paths))

        assert max_active == 1


class TestColQwen2Adapter:
    """Tests for ColQwen2Adapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColQwen2Adapter:
        """Create an adapter instance."""
        return ColQwen2Adapter(
            "vidore/colqwen2.5-v0.2",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: ColQwen2Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColQwen2Adapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # ColQwen2 default

    def test_encode_before_load_raises(self, adapter: ColQwen2Adapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColQwen2Adapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColQwen2Adapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    def test_concurrent_forwards_serialize(self, adapter: ColQwen2Adapter) -> None:
        """The forward lock serializes model forwards across threads (#2144/#2204).

        ColQwen2_5.forward requests output_hidden_states=True and this adapter
        has no preprocessor, so every request runs on the unserialized
        to_thread path; overlapping forwards race transformers' recorder
        patch/restore and leak layer outputs on GPU. The tokenizer lock is
        released before the forward, so only ``_forward_lock`` can serialize
        text, image, and batched-image model calls against one another.
        """
        counter_guard = threading.Lock()
        active = 0
        max_active = 0

        def processor(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            count = len(kwargs.get("text") or [""])
            batch = {
                "input_ids": torch.zeros((count, 4), dtype=torch.long),
                "attention_mask": torch.ones((count, 4), dtype=torch.long),
            }
            if kwargs.get("images") is not None:
                batch["image_grid_thw"] = torch.ones((count, 3), dtype=torch.long)
                batch["pixel_values"] = torch.zeros((count, 1))
            return batch

        def forward(**batch: Any) -> torch.Tensor:
            nonlocal active, max_active
            with counter_guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with counter_guard:
                active -= 1
            return torch.zeros((batch["input_ids"].shape[0], 4, 128))

        adapter._processor = processor
        adapter._model = forward
        adapter._device = "cpu"

        def call(path: str) -> None:
            if path == "text":
                adapter._encode_text("query")
            elif path == "image":
                adapter._encode_images([MagicMock()])
            else:
                adapter._encode_images_batched([MagicMock(), MagicMock()])

        paths = ["text", "image", "batched"] * 3
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(call, paths))

        assert max_active == 1


class TestColQwen3Adapter:
    """Tests for ColQwen3Adapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColQwen3Adapter:
        """Create an adapter instance."""
        return ColQwen3Adapter(
            "TomoroAI/tomoro-colqwen3-embed-4b",
            normalize=True,
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: ColQwen3Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColQwen3Adapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 320  # ColQwen3 default

    def test_encode_before_load_raises(self, adapter: ColQwen3Adapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColQwen3Adapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColQwen3Adapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    def test_encode_mixed_batch_preserves_order(self, adapter: ColQwen3Adapter) -> None:
        """Mixed text/image items round-trip in input order with one mv per item."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._multivector_dim = 4

        # Stamp each output with a unique value so we can assert which item it came from.
        def fake_load_images(item: Any) -> list[Any]:
            return [f"img-{item.text}"] * (1 if item.images else 0)

        def fake_encode_images(images: list[Any]) -> list[np.ndarray]:
            return [np.full((2, 4), float(i + 100), dtype=np.float32) for i, _ in enumerate(images)]

        def fake_encode_text(text: str) -> np.ndarray:
            return np.full((1, 4), float(hash(text) % 1000), dtype=np.float32)

        adapter._load_images = fake_load_images  # type: ignore[method-assign]
        adapter._encode_images = fake_encode_images  # type: ignore[method-assign]
        adapter._encode_text = fake_encode_text  # type: ignore[method-assign]

        items = [
            Item(text="a", images=[{"data": b"x", "format": "png"}]),
            Item(text="b"),
            Item(text="c", images=[{"data": b"y", "format": "png"}]),
            Item(text="d"),
        ]
        out = adapter.encode(items, output_types=["multivector"])
        assert out.batch_size == len(items)
        assert out.multivector is not None
        assert len(out.multivector) == len(items)

        # Image items got the per-image stamp (100, 101) in input order; text items got hash stamps.
        assert out.multivector[0][0, 0] == 100.0
        assert out.multivector[1][0, 0] == float(hash("b") % 1000)
        assert out.multivector[2][0, 0] == 101.0
        assert out.multivector[3][0, 0] == float(hash("d") % 1000)

    def test_encode_multi_image_item_concatenates_seq_dim(self, adapter: ColQwen3Adapter) -> None:
        """A single item with N images yields one mv with seq = sum of per-image seqs."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._multivector_dim = 4

        adapter._load_images = lambda item: ["img-1", "img-2", "img-3"]  # type: ignore[method-assign]
        per_image = [
            np.full((3, 4), 1.0, dtype=np.float32),
            np.full((5, 4), 2.0, dtype=np.float32),
            np.full((7, 4), 3.0, dtype=np.float32),
        ]
        adapter._encode_images = lambda images: per_image  # type: ignore[method-assign]

        items = [Item(images=[{"data": b"x", "format": "png"}] * 3)]
        out = adapter.encode(items, output_types=["multivector"])
        assert out.multivector is not None
        assert len(out.multivector) == 1
        mv = out.multivector[0]
        assert mv.shape == (3 + 5 + 7, 4)
        # Concatenation order is per-image order: 3 rows of 1.0, 5 rows of 2.0, 7 rows of 3.0.
        assert mv[0, 0] == 1.0
        assert mv[3, 0] == 2.0
        assert mv[8, 0] == 3.0

    def test_colqwen3_processor_call_holds_tokenizer_lock(self, adapter: ColQwen3Adapter) -> None:
        """The processor call runs under ``_tokenizer_lock`` (#2098).

        HF fast tokenizers are not thread-safe under per-call padding, so the
        adapter must hold the lock across the ``self._processor(...)`` call.
        The fake processor samples ``_tokenizer_lock.locked()`` at call time.
        """
        held: dict[str, bool] = {}

        def fake_processor(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            held["value"] = adapter._tokenizer_lock.locked()
            return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

        adapter._processor = fake_processor
        adapter._model = lambda **kwargs: SimpleNamespace(embeddings=torch.zeros((1, 3, 4)))
        adapter._device = "cpu"

        adapter._encode_text("q")

        assert held.get("value") is True

    def test_colqwen3_concurrent_encode_text_serialized(self, adapter: ColQwen3Adapter) -> None:
        """Concurrent text encodes are serialized, so the shared processor is
        never entered re-entrantly (proxy for the Rust ``Already borrowed``).
        """
        guard = _BorrowGuard()
        adapter._processor = guard
        adapter._model = lambda **kwargs: SimpleNamespace(embeddings=torch.zeros((1, 3, 4)))
        adapter._device = "cpu"

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def worker() -> None:
            barrier.wait()
            try:
                adapter._encode_text("q")
            except BaseException as exc:  # noqa: BLE001 - record for assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert guard.borrow_error is False
        assert guard.max_concurrent == 1

    def test_colqwen3_concurrent_forwards_serialize(self, adapter: ColQwen3Adapter) -> None:
        """Text and image model calls share one forward lock."""
        counter_guard = threading.Lock()
        active = 0
        max_active = 0

        def processor(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            count = len(kwargs.get("text") or kwargs.get("images") or [""])
            return {"input_ids": torch.zeros((count, 4), dtype=torch.long)}

        def forward(**batch: Any) -> Any:
            nonlocal active, max_active
            with counter_guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with counter_guard:
                active -= 1
            count = batch["input_ids"].shape[0]
            return SimpleNamespace(embeddings=torch.zeros((count, 4, 320)))

        adapter._processor = processor
        adapter._model = forward
        adapter._device = "cpu"

        def call(path: str) -> None:
            if path == "text":
                adapter._encode_text("query")
            else:
                adapter._encode_images([MagicMock()])

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(call, ["text", "image"] * 4))

        assert max_active == 1


class TestColSmolAdapter:
    """Tests for ColSmolAdapter with mocked model (small permissive visual-MV)."""

    @pytest.fixture
    def adapter(self) -> ColSmolAdapter:
        """Create an adapter instance."""
        return ColSmolAdapter(
            "vidore/colSmol-256M",
            normalize=True,
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: ColSmolAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColSmolAdapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # ColSmol default

    def test_encode_before_load_raises(self, adapter: ColSmolAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColSmolAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColSmolAdapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    def test_encode_mixed_batch_preserves_order(self, adapter: ColSmolAdapter) -> None:
        """Mixed text/image items round-trip in input order with one mv per item."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._multivector_dim = 4

        adapter._load_images = lambda item: ["img"] * (1 if item.images else 0)  # type: ignore[method-assign]

        def fake_encode_images(images: list[Any]) -> list[np.ndarray]:
            return [np.full((2, 4), float(i + 100), dtype=np.float32) for i, _ in enumerate(images)]

        def fake_encode_text(text: str) -> np.ndarray:
            return np.full((1, 4), float(hash(text) % 1000), dtype=np.float32)

        adapter._encode_images = fake_encode_images  # type: ignore[method-assign]
        adapter._encode_text = fake_encode_text  # type: ignore[method-assign]

        items = [
            Item(text="a", images=[{"data": b"x", "format": "png"}]),
            Item(text="b"),
            Item(text="c", images=[{"data": b"y", "format": "png"}]),
            Item(text="d"),
        ]
        out = adapter.encode(items, output_types=["multivector"])
        assert out.batch_size == len(items)
        assert out.multivector is not None
        assert len(out.multivector) == len(items)
        assert out.multivector[0][0, 0] == 100.0
        assert out.multivector[1][0, 0] == float(hash("b") % 1000)
        assert out.multivector[2][0, 0] == 101.0
        assert out.multivector[3][0, 0] == float(hash("d") % 1000)


class TestNemoColEmbedV2Config:
    """Tests for NemoColEmbedAdapter v2 configuration (Qwen3-VL backbone, token_dim=2560)."""

    def test_v2_token_dim_constructor(self) -> None:
        """V2 adapter accepts token_dim=2560 and stores it on _multivector_dim."""
        adapter = NemoColEmbedAdapter(
            "nvidia/nemotron-colembed-vl-4b-v2",
            token_dim=2560,
            normalize=True,
        )
        assert adapter._multivector_dim == 2560
        assert adapter.dims.multivector == 2560

    def test_v2_default_compute_precision(self) -> None:
        """V2 adapter inherits the bf16 default."""
        adapter = NemoColEmbedAdapter(
            "nvidia/nemotron-colembed-vl-4b-v2",
            token_dim=2560,
        )
        assert adapter._compute_precision == "bfloat16"


class TestNemoColEmbedAdapter:
    """Tests for NemoColEmbedAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> NemoColEmbedAdapter:
        """Create an adapter instance."""
        return NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: NemoColEmbedAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: NemoColEmbedAdapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 3072

    def test_discovers_v1_nested_llama_hidden_width(self) -> None:
        """v1 emits the nested Llama hidden state, not the stale 128-d catalog width."""
        config = SimpleNamespace(
            embedding_dim=128,
            hidden_size=None,
            llm_config=SimpleNamespace(hidden_size=3072),
        )

        assert NemoColEmbedAdapter._discover_multivector_dim(config) == 3072

    def test_discovers_v2_nested_text_hidden_width(self) -> None:
        """v2 emits the nested Qwen3-VL text width before legacy fallbacks."""
        config = SimpleNamespace(
            embedding_dim=128,
            hidden_size=None,
            llm_config=None,
            text_config=SimpleNamespace(hidden_size=2560),
        )

        assert NemoColEmbedAdapter._discover_multivector_dim(config) == 2560

    def test_encode_before_load_raises(self, adapter: NemoColEmbedAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: NemoColEmbedAdapter) -> None:
        """Only multivector output type is supported."""
        # Mock model as loaded
        adapter._model = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    def test_concurrent_forwards_serialize(
        self,
        adapter: NemoColEmbedAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All v1 model-forward entry points share one lock (#2144/#2204).

        The remote v1 model requests hidden states, so overlapping native
        query/image and prepared-image forwards could race transformers'
        output-recorder patch/restore and leak layer outputs on GPU.
        """
        counter_guard = threading.Lock()
        active = 0
        max_active = 0

        def enter(result: Any) -> Any:
            nonlocal active, max_active
            with counter_guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with counter_guard:
                active -= 1
            return result

        class ForwardProbe:
            def forward_queries(self, texts: list[str], *, batch_size: int) -> list[torch.Tensor]:
                return enter([torch.zeros((2, 128)) for _ in texts])

            def forward_passages(self, images: list[Any], *, batch_size: int) -> list[torch.Tensor]:
                return enter([torch.zeros((2, 128)) for _ in images])

            def __call__(self, **kwargs: Any) -> Any:
                batch_size = kwargs["input_ids"].shape[0]
                return enter(SimpleNamespace(hidden_states=[torch.zeros((batch_size, 2, 128))]))

        processor = MagicMock()
        processor.collate.return_value = {
            "pixel_values": torch.zeros((1, 1)),
            "input_ids": torch.zeros((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        adapter._model = ForwardProbe()
        adapter._processor = processor
        adapter._device = "cpu"

        monkeypatch.setattr(PILImage, "open", lambda *args, **kwargs: MagicMock(mode="RGB"))
        monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())

        text_item = Item(text="query")
        image_item = Item(images=[{"data": b"image", "format": "png"}])
        prepared_item = SimpleNamespace(payload=SimpleNamespace())

        def call(path: str) -> None:
            if path == "text":
                adapter._encode_texts([text_item], is_query=True)
            elif path == "image":
                adapter._encode_images([image_item], is_query=False)
            else:
                adapter._encode_images_preprocessed([image_item], [prepared_item], is_query=False)

        paths = ["text", "image", "prepared"] * 3
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(call, paths))

        assert max_active == 1

    def test_concurrent_v1_loads_serialize_dynamic_class_resolution(
        self,
        adapter: NemoColEmbedAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dynamic resolution cannot observe another load's to_dict patch."""

        def original_to_dict(self: Any) -> dict[str, Any]:
            return {"model_type": "nemo"}

        class Config:
            to_dict = original_to_dict

        class ProbeLock:
            """Expose when a second loader reaches the adapter lock."""

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._entry_guard = threading.Lock()
                self._entry_count = 0
                self.second_entered = threading.Event()

            def __enter__(self) -> Self:
                with self._entry_guard:
                    self._entry_count += 1
                    if self._entry_count == 2:
                        self.second_entered.set()
                self._lock.acquire()
                return self

            def __exit__(self, *args: object) -> None:
                self._lock.release()

        probe_lock = ProbeLock()
        load_started = threading.Event()
        load_count_lock = threading.Lock()
        load_count = 0

        class DynamicModel:
            config_class = Config

            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> object:
                nonlocal load_count
                assert Config.to_dict is not original_to_dict
                with load_count_lock:
                    load_count += 1
                    current_load = load_count
                if current_load == 1:
                    load_started.set()
                    assert probe_lock.second_entered.wait(timeout=5)
                return object()

        resolution_saw_original: list[bool] = []

        def get_dynamic_class(*args: Any, **kwargs: Any) -> type[DynamicModel]:
            resolution_saw_original.append(Config.to_dict is original_to_dict)
            return DynamicModel

        monkeypatch.setitem(sys.modules, "flash_attn", MagicMock())
        monkeypatch.setattr("sie_server.adapters.nemo_colembed._V1_TO_DICT_PATCH_LOCK", probe_lock)
        monkeypatch.setattr(
            "transformers.dynamic_module_utils.get_class_from_dynamic_module",
            get_dynamic_class,
        )
        other = NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
            compute_precision="bfloat16",
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(adapter._load_v1_dynamic, "cpu", torch.float32)
            assert load_started.wait(timeout=5)
            second = pool.submit(other._load_v1_dynamic, "cpu", torch.float32)
            first.result()
            second.result()

        assert resolution_saw_original == [True, True]
        assert Config.to_dict is original_to_dict


class TestNemoColEmbedPreprocessor:
    """Tests for NemoColEmbedPreprocessor infrastructure."""

    def test_preprocessor_class_exists(self) -> None:
        """NemoColEmbedPreprocessor class is defined."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        assert NemoColEmbedPreprocessor is not None

    def test_preprocessor_payload_class_exists(self) -> None:
        """NemoColEmbedPayload dataclass is defined."""
        from sie_server.core.prepared import NemoColEmbedPayload

        assert NemoColEmbedPayload is not None

    def test_adapter_has_preprocessor_method(self) -> None:
        """NemoColEmbedAdapter has get_preprocessor method."""
        adapter = NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
        )
        # get_preprocessor returns CharCountPreprocessor for cost estimation
        preprocessor = adapter.get_preprocessor()
        assert preprocessor is not None

    def test_adapter_processor_created_on_load(self) -> None:
        """NemoColEmbedAdapter creates _processor after load().

        Note: Full _create_processor() test requires loaded model with
        tokenizer/config attributes. This test just verifies the interface.
        """
        adapter = NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
        )
        # Before load, _processor should be None
        assert adapter._processor is None


class TestVLMCudaCacheClearing:
    """Tests that VLM adapters contain torch.cuda.empty_cache() after inference.

    This prevents GPU memory accumulation (OOM) on L4 22GB GPUs when
    encoding many document images in sequence.
    """

    def test_colqwen2_encode_images_has_empty_cache(self) -> None:
        """ColQwen2 _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColQwen2Adapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "ColQwen2._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colqwen3_encode_images_has_empty_cache(self) -> None:
        """ColQwen3 _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColQwen3Adapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "ColQwen3._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colpali_encode_prepared_batch_has_empty_cache(self) -> None:
        """ColPali _encode_prepared_batch source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColPaliAdapter._encode_prepared_batch)
        assert "torch.cuda.empty_cache()" in source, (
            "ColPali._encode_prepared_batch must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colpali_encode_images_has_empty_cache(self) -> None:
        """ColPali _encode_images (single-item fallback) source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColPaliAdapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "ColPali._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colqwen2_encode_images_batched_has_empty_cache(self) -> None:
        """ColQwen2 _encode_images_batched (document-path) source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColQwen2Adapter._encode_images_batched)
        assert "torch.cuda.empty_cache()" in source, (
            "ColQwen2._encode_images_batched must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colsmol_encode_images_has_empty_cache(self) -> None:
        """ColSmol _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColSmolAdapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "ColSmol._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_nemo_colembed_encode_images_has_empty_cache(self) -> None:
        """NemoColEmbed _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(NemoColEmbedAdapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "NemoColEmbed._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_nemo_colembed_preprocessed_has_empty_cache(self) -> None:
        """NemoColEmbed _encode_images_preprocessed source contains empty_cache call."""
        import inspect

        source = inspect.getsource(NemoColEmbedAdapter._encode_images_preprocessed)
        assert "torch.cuda.empty_cache()" in source, (
            "NemoColEmbed._encode_images_preprocessed must call torch.cuda.empty_cache() to prevent OOM"
        )


class TestColFamilyTokenizerLock:
    """#2098: every Col-family adapter must hold a per-call tokenizer lock around
    the shared HF processor on the direct (get_preprocessor -> None) encode path,
    so concurrent text queries cannot race the Rust fast tokenizer.
    """

    @staticmethod
    def _obj_model(**kwargs: Any) -> Any:
        """Fake model returning a ModelOutput-like object (colqwen3/colpali)."""
        return SimpleNamespace(embeddings=torch.zeros((1, 3, 4)))

    @staticmethod
    def _tensor_model(**kwargs: Any) -> torch.Tensor:
        """Fake model returning the embeddings tensor directly (colqwen2/colsmol)."""
        return torch.zeros((1, 3, 4))

    @pytest.mark.parametrize(
        ("adapter_factory", "model_kind"),
        [
            (lambda: ColQwen3Adapter("stub/colqwen3"), "obj"),
            (lambda: ColQwen2Adapter("stub/colqwen2"), "tensor"),
            (lambda: ColSmolAdapter("stub/colsmol"), "tensor"),
            (lambda: ColPaliAdapter("stub/colpali"), "obj"),
        ],
    )
    def test_encode_text_holds_tokenizer_lock(self, adapter_factory: Any, model_kind: str) -> None:
        adapter = adapter_factory()
        held: dict[str, bool] = {}

        def fake_processor(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            held["value"] = adapter._tokenizer_lock.locked()
            return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

        adapter._processor = fake_processor
        adapter._model = self._obj_model if model_kind == "obj" else self._tensor_model
        adapter._device = "cpu"

        adapter._encode_text("q")

        assert held.get("value") is True


class _PoolProbeState:
    """Shared state behind a per-thread processor factory (#2098).

    Each ``ImagePreprocessor.prepare`` thread gets its own ``_PoolProbeProcessor``
    (one factory call), but they all report into this single state object. It
    counts how many processors the factory built and — via a shared atomic
    counter — the peak number of threads simultaneously inside *any* processor
    call. With per-thread instances and a real sleep in each call, two concurrent
    prepares overlap, so ``max_cross_instance`` reaches 2: proof the pool is NOT
    serialised. A pool-wide lock would cap it at 1.
    """

    def __init__(self, sleep_s: float = 0.02) -> None:
        self.lock = threading.Lock()
        self._sleep_s = sleep_s
        self.instances = 0  # processors the factory built (== threads that touched it)
        self.active = 0  # threads currently inside some processor call
        self.max_cross_instance = 0  # peak of `active`
        self.reentered = False  # a single instance was entered concurrently (must not happen)

    def make_processor(self) -> "_PoolProbeProcessor":
        with self.lock:
            self.instances += 1
        return _PoolProbeProcessor(self, self._sleep_s)


class _PoolProbeProcessor:
    """A single per-thread fake processor tied to a shared ``_PoolProbeState``."""

    def __init__(self, state: _PoolProbeState, sleep_s: float) -> None:
        self._state = state
        self._sleep_s = sleep_s
        self._own = threading.Lock()
        self._inside = 0

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        state = self._state
        with self._own:
            if self._inside > 0:  # per-thread instance entered re-entrantly: a bug
                state.reentered = True
            self._inside += 1
        with state.lock:
            state.active += 1
            state.max_cross_instance = max(state.max_cross_instance, state.active)
        time.sleep(self._sleep_s)
        with state.lock:
            state.active -= 1
        with self._own:
            self._inside -= 1
        return {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 4, 4)),
        }


class TestImagePreprocessorThreadLocalProcessor:
    """#2098 image path: per-thread processors instead of a pool-wide lock.

    ColPali hands its raw shared processor to ``ImagePreprocessor``, which runs
    on the 8-worker preprocessor executor. The ColPali processor invokes its Rust
    fast tokenizer even for image-only calls, so one shared instance races with
    "Already borrowed". The fix is a per-thread processor (``processor_factory``):
    no shared mutable state, no lock, full pool parallelism. (A pool-wide lock
    removed the race but serialised image transforms and collapsed colpali eval
    throughput → server 503s.)
    """

    @staticmethod
    def _png_item() -> Item:
        """An Item carrying real PNG bytes (ImagePreprocessor decodes them)."""
        buf = io.BytesIO()
        PILImage.new("RGB", (8, 8), color="white").save(buf, format="PNG")
        return Item(images=[{"data": buf.getvalue(), "format": "png"}])

    @staticmethod
    def _run_two_prepares(pre: ImagePreprocessor, item: Item) -> list[BaseException]:
        """Drive two concurrent ``prepare`` calls through ``pre``, gated on a barrier."""
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def worker() -> None:
            barrier.wait()
            try:
                pre.prepare([item], config=MagicMock())
            except BaseException as exc:  # noqa: BLE001 - record for assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return errors

    def test_prepare_uses_thread_local_processor(self) -> None:
        """Two concurrent prepares each get their own processor and run in
        parallel — no "Already borrowed", one factory call per thread, and the
        shared processor is never touched.

        Anti-serialisation proof: ``max_cross_instance`` reaches 2. Under the old
        pool-wide lock this would cap at 1, so this assertion is the regression
        guard for the throughput collapse.
        """
        state = _PoolProbeState()

        def _shared_must_not_be_called(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            raise AssertionError("shared processor must not be used when a factory is set")

        pre = ImagePreprocessor(
            _shared_must_not_be_called,
            "stub/colpali",
            processor_factory=state.make_processor,
        )

        errors = self._run_two_prepares(pre, self._png_item())

        assert errors == []  # no RuntimeError: Already borrowed
        assert state.reentered is False  # each per-thread instance stayed single-entry
        assert state.instances == 2  # one factory call per pool thread → distinct instances
        assert state.max_cross_instance >= 2  # calls overlapped: the pool is NOT serialised

    def test_prepare_without_factory_shares_processor(self) -> None:
        """Default path (no factory): the shared processor is used directly,
        byte-identical to the pre-#2098 behaviour (CLIP/SigLIP/registry).
        """
        calls: list[Any] = []

        def shared_processor(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            calls.append(kwargs.get("images"))
            return {"pixel_values": torch.zeros((1, 3, 4, 4))}

        pre = ImagePreprocessor(shared_processor, "stub/clip")
        assert pre._tls is None  # no thread-local state when no factory

        batch = pre.prepare([self._png_item()], config=MagicMock())

        assert len(calls) == 1  # the shared processor did the work
        assert len(batch.items) == 1

    def test_shared_processor_without_factory_can_race(self) -> None:
        """Control: a single shared instance with no factory CAN be entered
        concurrently — this is exactly the "Already borrowed" hazard the
        per-thread factory exists to remove. Not wired to any production path;
        colpali always supplies a factory now.
        """
        guard = _BorrowGuard()
        pre = ImagePreprocessor(guard, "stub/colpali")  # no factory: shared instance

        self._run_two_prepares(pre, self._png_item())

        assert guard.borrow_error is True
