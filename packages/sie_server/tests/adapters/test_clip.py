from __future__ import annotations

import io
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from sie_server.adapters.clip import CLIPAdapter
from sie_server.types.inputs import Item
from torch.nn import functional

# Create a random generator for tests
_RNG = np.random.default_rng(42)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    """Encode a solid-color 8x8 RGB PNG (a real image the decode path can open)."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _img_item(item_id: str, colors: list[tuple[int, int, int]]) -> Item:
    return Item(id=item_id, images=[{"data": _png_bytes(c), "format": "png"} for c in colors])


class _SimpleTokenizer:
    """Minimal HF-tokenizer stand-in for the §7.3 text-metering seam.

    ``_token_counts_or_none`` calls ``tokenizer(texts, truncation=..., max_length=...)``
    and reads ``["input_ids"]`` as a list of per-text id lists.
    """

    def __call__(self, texts: list[str], truncation: bool = False, max_length: int | None = None, **_kw: Any):
        return {"input_ids": [list(range(len(t))) for t in texts]}


class _FakeVisionProcessor:
    """Deterministic, pure-function stand-in for CLIP/SigLIP processors.

    Maps each image to its mean RGB (a ``[3]`` feature) and each text to a
    ``[3]`` id vector, so per-item preprocessing is independent and the model
    below can be a pure per-row linear map — making a stacked forward exactly
    equal to per-item serial forwards.
    """

    def __init__(self) -> None:
        self.image_calls = 0
        self.text_calls = 0
        # The §7.3 text-metering seam recounts tokens off ``processor.tokenizer``.
        self.tokenizer = _SimpleTokenizer()

    def __call__(
        self, images: Any = None, text: Any = None, return_tensors: Any = None, **_kw: Any
    ) -> dict[str, torch.Tensor]:
        if images is not None:
            self.image_calls += 1
            imgs = images if isinstance(images, list) else [images]
            rows = [np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0 for im in imgs]
            return {"pixel_values": torch.tensor(np.stack(rows), dtype=torch.float32)}
        self.text_calls += 1
        txts = text if isinstance(text, list) else [text]
        rows = [[float(len(t)), float(sum(ord(c) for c in t) % 89), 1.0] for t in txts]
        return {"input_ids": torch.tensor(rows, dtype=torch.float32)}


class _FakeVisionModel:
    """Pure per-row linear model; counts stacked forwards to prove batching."""

    def __init__(self, dim: int) -> None:
        gen = torch.Generator().manual_seed(7)
        self.img_w = torch.randn(3, dim, generator=gen)
        self.txt_w = torch.randn(3, dim, generator=gen)
        self.image_forward_calls = 0
        self.text_forward_calls = 0

    def get_image_features(self, pixel_values: torch.Tensor = None, **_kw: Any) -> torch.Tensor:  # type: ignore[assignment]
        self.image_forward_calls += 1
        return pixel_values @ self.img_w

    def get_text_features(self, input_ids: torch.Tensor = None, **_kw: Any) -> torch.Tensor:  # type: ignore[assignment]
        self.text_forward_calls += 1
        return input_ids @ self.txt_w


class _RaceDetectingTokenizer:
    """Stand-in for an HF fast tokenizer that is NOT thread-safe.

    Mimics the ``RuntimeError: Already borrowed`` failure: it raises if a
    second call enters while another is in flight, and records the peak
    observed concurrency so a serialising lock can be asserted (peak == 1).
    """

    def __init__(self) -> None:
        self._active = 0
        self._guard = threading.Lock()
        self.peak_concurrency = 0
        # The §7.3 metering seam reads ``processor.tokenizer``; a stateless stub
        # keeps the race detection focused on the processor's text call.
        self.tokenizer = _SimpleTokenizer()

    def __call__(self, *args: object, **kwargs: object) -> dict[str, MagicMock]:
        with self._guard:
            self._active += 1
            self.peak_concurrency = max(self.peak_concurrency, self._active)
            concurrent = self._active > 1
        try:
            if concurrent:
                raise RuntimeError("Already borrowed")
            # Hold the "borrow" briefly so an unsynchronised racer would overlap.
            time.sleep(0.002)
            return {"input_ids": MagicMock()}
        finally:
            with self._guard:
                self._active -= 1


class TestCLIPAdapter:
    """Tests for CLIPAdapter with mocked model."""

    @pytest.fixture
    def mock_clip_model(self) -> MagicMock:
        """Create a mock CLIPModel."""
        mock = MagicMock()
        # Mock config with projection_dim
        mock.config.projection_dim = 512
        # Mock get_text_features and get_image_features
        mock.get_text_features.return_value = MagicMock(
            __getitem__=lambda self, idx: MagicMock(
                float=lambda: MagicMock(
                    cpu=lambda: MagicMock(numpy=lambda: _RNG.standard_normal(512).astype(np.float32))
                )
            )
        )
        mock.get_image_features.return_value = MagicMock(
            mean=lambda dim, keepdim: MagicMock(
                __getitem__=lambda self, idx: MagicMock(
                    float=lambda: MagicMock(
                        cpu=lambda: MagicMock(numpy=lambda: _RNG.standard_normal(512).astype(np.float32))
                    )
                )
            )
        )
        return mock

    @pytest.fixture
    def mock_clip_processor(self) -> MagicMock:
        """Create a mock CLIPProcessor."""
        mock = MagicMock()
        # Return dict-like object for processor outputs
        mock.return_value = {"pixel_values": MagicMock(), "input_ids": MagicMock()}
        return mock

    @pytest.fixture
    def adapter(self) -> CLIPAdapter:
        """Create an adapter instance."""
        return CLIPAdapter(
            "openai/clip-vit-base-patch32",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: CLIPAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["dense"]

    def test_dims_before_load(self, adapter: CLIPAdapter) -> None:
        """Dims returns None dense before load (BaseAdapter reads from spec)."""
        dims = adapter.dims
        assert dims.dense is None

    def test_accepts_dense_dim_kwarg(self) -> None:
        """Construction with dense_dim= must not raise (loader contract)."""
        adapter = CLIPAdapter(
            "openai/clip-vit-base-patch32",
            dense_dim=512,
        )
        assert adapter._configured_dense_dim == 512
        assert adapter.dims.dense == 512

    def test_encode_before_load_raises(self, adapter: CLIPAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_without_input_raises(self, adapter: CLIPAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["dense"])

    def test_validate_output_types(self, adapter: CLIPAdapter) -> None:
        """Only dense output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["multivector"])

    @patch("transformers.CLIPModel.from_pretrained")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_load(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
        adapter: CLIPAdapter,
        mock_clip_model: MagicMock,
        mock_clip_processor: MagicMock,
    ) -> None:
        """Load initializes the model."""
        mock_model_from_pretrained.return_value = mock_clip_model
        mock_processor_from_pretrained.return_value = mock_clip_processor

        adapter.load("cpu")

        mock_model_from_pretrained.assert_called_once()
        mock_processor_from_pretrained.assert_called_once()
        assert adapter.dims.dense == 512

    @patch("transformers.CLIPModel.from_pretrained")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_load_accepts_matching_dense_dim(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
        mock_clip_model: MagicMock,
        mock_clip_processor: MagicMock,
    ) -> None:
        """Load accepts catalog dense_dim when it matches projection_dim."""
        adapter = CLIPAdapter("openai/clip-vit-base-patch32", dense_dim=512)
        mock_model_from_pretrained.return_value = mock_clip_model
        mock_processor_from_pretrained.return_value = mock_clip_processor

        adapter.load("cpu")

        assert adapter.dims.dense == 512

    @patch("transformers.CLIPModel.from_pretrained")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_load_rejects_dense_dim_mismatch(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
        mock_clip_model: MagicMock,
        mock_clip_processor: MagicMock,
    ) -> None:
        """Load rejects catalog dense_dim when it differs from projection_dim."""
        adapter = CLIPAdapter("openai/clip-vit-base-patch32", dense_dim=768)
        mock_model_from_pretrained.return_value = mock_clip_model
        mock_processor_from_pretrained.return_value = mock_clip_processor

        with pytest.raises(ValueError, match="configured dense_dim=768, model projection_dim=512"):
            adapter.load("cpu")

    def test_encode_text_tokenizer_thread_safe(self) -> None:
        """Regression (MP.4 34026b6cd): concurrent text encodes must not race
        on the shared HF fast tokenizer (``RuntimeError: Already borrowed``).

        Drives ``_encode_text`` from several threads through a tokenizer that
        raises on concurrent entry; the adapter's ``_tokenizer_lock`` must
        serialise the call so no overlap and no error occurs. Not a strict
        serialisability proof — a threaded smoke test that the lock holds.
        """
        adapter = CLIPAdapter("openai/clip-vit-base-patch32", normalize=False)
        adapter._model = MagicMock()
        # get_text_features must yield a real tensor (the adapter unwraps the
        # feature tensor from the model output before returning).
        adapter._model.get_text_features.return_value = torch.zeros((1, 8))
        adapter._device = "cpu"
        tokenizer = _RaceDetectingTokenizer()
        adapter._processor = tokenizer  # ty: ignore[invalid-assignment]

        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(20):
                    adapter._encode_text("a photo of a cat")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"tokenizer race under concurrent encode: {errors!r}"
        assert tokenizer.peak_concurrency == 1

    @patch("sie_server.adapters.clip.torch")
    def test_unload(self, mock_torch: MagicMock, adapter: CLIPAdapter) -> None:
        """Unload clears the model."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._dense_dim = 512

        adapter.unload()

        assert adapter._model is None
        assert adapter._processor is None
        assert adapter._dense_dim is None
        assert adapter.dims.dense is None

    def test_unload_shuts_down_preprocess_pool(self, adapter: CLIPAdapter) -> None:
        """Unload shuts down and clears the lazily-created preprocessing pool."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        pool = adapter._get_preprocess_pool()
        assert adapter._preprocess_pool is pool

        adapter.unload()

        assert adapter._preprocess_pool is None
        assert pool._shutdown

    @pytest.mark.parametrize("normalize", [True, False])
    def test_encode_batched_matches_serial(self, normalize: bool) -> None:
        """EQUIVALENCE GATE: one stacked forward per modality must reproduce the
        per-item serial semantics exactly — output order, per-image normalize,
        multi-image mean-pool, and mixed image/text batches.
        """
        dim = 16
        adapter = CLIPAdapter("openai/clip-vit-base-patch32", normalize=normalize)
        model = _FakeVisionModel(dim)
        proc = _FakeVisionProcessor()
        adapter._model = model  # ty: ignore[invalid-assignment]
        adapter._processor = proc  # ty: ignore[invalid-assignment]
        adapter._device = "cpu"
        adapter._dense_dim = dim

        items = [
            _img_item("a", [(200, 10, 10)]),  # single image
            _img_item("bc", [(10, 200, 10), (10, 10, 200)]),  # multi-image -> mean-pool
            Item(id="t1", text="a red circle"),  # text
            _img_item("d", [(90, 90, 90)]),  # single image
            Item(id="t2", text="a blue rectangle here and there"),  # text
        ]

        out = adapter.encode(items, ["dense"])
        assert out.dense is not None
        assert out.dense.shape == (5, dim)
        # Batching proof: exactly one stacked forward per tower for the whole call.
        assert model.image_forward_calls == 1
        assert model.text_forward_calls == 1

        # Serial reference — replicate the pre-batch per-item algorithm exactly.
        def ref_image(item: Item) -> np.ndarray:
            pils = [Image.open(io.BytesIO(im["data"])).convert("RGB") for im in item.images]  # type: ignore[index]
            feats = model.get_image_features(**proc(images=pils))
            if normalize:
                feats = functional.normalize(feats, p=2, dim=-1)
            if len(pils) > 1:
                feats = feats.mean(dim=0, keepdim=True)
            return feats[0].numpy()

        def ref_text(item: Item) -> np.ndarray:
            feats = model.get_text_features(**proc(text=[item.text]))
            if normalize:
                feats = functional.normalize(feats, p=2, dim=-1)
            return feats[0].numpy()

        for i, item in enumerate(items):
            expected = ref_image(item) if item.images else ref_text(item)
            got = out.dense[i]
            denom = float(np.linalg.norm(got) * np.linalg.norm(expected)) or 1.0
            cos = float(np.dot(got, expected) / denom)
            assert cos >= 0.9999, f"item {i} cosine {cos}"
            assert np.allclose(got, expected, atol=1e-5), f"item {i} value mismatch"


def test_preprocess_one_rejects_undecodable_bytes_with_typed_error() -> None:
    """Non-image bytes are a typed InvalidMediaError (-> 400), not a PIL OSError 500."""
    from sie_server.types.inputs import InvalidMediaError

    adapter = CLIPAdapter("openai/clip-vit-base-patch32")
    adapter._processor = _FakeVisionProcessor()

    with pytest.raises(InvalidMediaError, match="image data is not a decodable image"):
        adapter._preprocess_one({"data": b"valid base64, but not an image"})


def test_undecodable_bytes_name_the_original_item_index() -> None:
    """The flattened preprocessing jobs keep each image's request-local indices.

    encode() partitions to image items (here original positions 0 and 2); the
    decode failure must name the request-local index 2, not subset slot 1.
    """
    from sie_server.types.inputs import InvalidMediaError

    adapter = CLIPAdapter("openai/clip-vit-base-patch32")
    adapter._model = MagicMock()
    adapter._processor = _FakeVisionProcessor()
    good = Item(images=[{"data": _png_bytes((1, 2, 3)), "format": "png"}])
    bad = Item(images=[{"data": b"valid base64, but not an image"}])

    with pytest.raises(InvalidMediaError, match=r"at `\$\.items\[2\]\.images\[0\]\.data`"):
        adapter._encode_image_items([good, bad], [0, 2])
