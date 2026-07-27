from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from sie_server.adapters.siglip.adapter import SiglipAdapter
from sie_server.core.prepared import ImagePayload, PreparedItem
from sie_server.types.inputs import Item


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

    model_max_length = 64

    def __call__(self, texts: list[str], truncation: bool = False, max_length: int | None = None, **_kw: Any):
        return {"input_ids": [list(range(len(t))) for t in texts]}


class _FakeVisionProcessor:
    """Deterministic, pure-function stand-in for the SigLIP processor.

    Maps each image to its mean RGB (a ``[3]`` feature) and each text to a
    ``[3]`` id vector, so per-item preprocessing is independent and the model
    below can be a pure per-row linear map — making a stacked forward exactly
    equal to per-item serial forwards.
    """

    def __init__(self) -> None:
        self.image_calls = 0
        self.text_calls = 0
        self.text_call_kwargs: list[dict[str, Any]] = []
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
        self.text_call_kwargs.append(dict(_kw))
        txts = text if isinstance(text, list) else [text]
        rows = [[float(len(t)), float(sum(ord(c) for c in t) % 89), 1.0] for t in txts]
        return {"input_ids": torch.tensor(rows, dtype=torch.float32)}


class _FakeVisionModel:
    """Pure per-row linear model; counts stacked forwards to prove batching."""

    def __init__(self, dim: int) -> None:
        gen = torch.Generator().manual_seed(11)
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


class TestSiglipAdapter:
    """Tests for SiglipAdapter with mocked model."""

    @pytest.fixture
    def mock_siglip_model(self) -> MagicMock:
        """Create a mock SiglipModel."""
        mock = MagicMock()
        # Mock config with vision_config.hidden_size (SigLIP uses hidden_size, not projection_dim)
        mock.config.vision_config.hidden_size = 1152
        return mock

    @pytest.fixture
    def mock_siglip_processor(self) -> MagicMock:
        """Create a mock SiglipProcessor."""
        mock = MagicMock()
        mock.return_value = {"pixel_values": MagicMock(), "input_ids": MagicMock()}
        return mock

    @pytest.fixture
    def adapter(self) -> SiglipAdapter:
        """Create an adapter instance."""
        return SiglipAdapter(
            "google/siglip-so400m-patch14-384",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: SiglipAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["dense"]

    def test_dims_before_load(self, adapter: SiglipAdapter) -> None:
        """Dims returns None dense before load (BaseAdapter reads from spec)."""
        dims = adapter.dims
        assert dims.dense is None

    def test_encode_before_load_raises(self, adapter: SiglipAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_without_input_raises(self, adapter: SiglipAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["dense"])

    def test_validate_output_types(self, adapter: SiglipAdapter) -> None:
        """Only dense output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    @patch("transformers.SiglipModel.from_pretrained")
    @patch("transformers.SiglipProcessor.from_pretrained")
    def test_load(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
        adapter: SiglipAdapter,
        mock_siglip_model: MagicMock,
        mock_siglip_processor: MagicMock,
    ) -> None:
        """Load initializes the model."""
        mock_model_from_pretrained.return_value = mock_siglip_model
        mock_processor_from_pretrained.return_value = mock_siglip_processor

        adapter.load("cpu")

        mock_model_from_pretrained.assert_called_once()
        assert mock_model_from_pretrained.call_args.kwargs["dtype"] is torch.float32
        assert "torch_dtype" not in mock_model_from_pretrained.call_args.kwargs
        mock_processor_from_pretrained.assert_called_once_with(
            "google/siglip-so400m-patch14-384",
            trust_remote_code=False,
            use_fast=False,
        )
        assert adapter.dims.dense == 1152

    @pytest.mark.parametrize(
        ("tokenizer_max", "position_max"),
        [(16, 64), (int(1e30), 16)],
    )
    @patch("transformers.SiglipModel.from_pretrained")
    @patch("transformers.SiglipProcessor.from_pretrained")
    def test_load_clamps_configured_length_to_text_tower_capacity(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
        tokenizer_max: int,
        position_max: int,
    ) -> None:
        processor = MagicMock()
        processor.tokenizer.model_max_length = tokenizer_max
        model = MagicMock()
        model.config.vision_config.hidden_size = 1152
        model.config.text_config.max_position_embeddings = position_max
        mock_processor_from_pretrained.return_value = processor
        mock_model_from_pretrained.return_value = model
        adapter = SiglipAdapter("google/siglip-so400m-patch14-224", max_seq_length=64)

        adapter.load("cpu")

        assert adapter._max_seq_length == 16

    @pytest.mark.parametrize("revision", [None, "main", "C56244CC94F92419E8369FA71EFDAF403B124CE8"])
    def test_open_clip_hub_requires_full_lowercase_commit_sha(self, revision: str | None) -> None:
        with pytest.raises(ValueError, match="immutable 40-character lowercase commit SHA"):
            SiglipAdapter(
                "Marqo/marqo-fashionSigLIP",
                backend="open_clip",
                open_clip_model_id="hf-hub:Marqo/marqo-fashionSigLIP",
                dense_dim=768,
                revision=revision,
            )

    def test_open_clip_hub_loads_checkpoint_tokenizer_and_preprocess_from_pinned_snapshot(
        self,
        tmp_path: Path,
    ) -> None:
        revision = "c56244cc94f92419e8369fa71efdaf403b124ce8"
        config = {
            "model_cfg": {
                "embed_dim": 768,
                "vision_cfg": {"image_size": 224},
                "text_cfg": {"hf_tokenizer_name": "mutable/tokenizer"},
            },
            "preprocess_cfg": {
                "mean": [0.1, 0.2, 0.3],
                "std": [0.4, 0.5, 0.6],
                "interpolation": "bicubic",
                "resize_mode": "squash",
            },
        }
        (tmp_path / "open_clip_config.json").write_text(json.dumps(config), encoding="utf-8")
        weights_path = tmp_path / "open_clip_model.safetensors"
        weights_path.touch()
        model = MagicMock()
        val_preprocess = MagicMock()
        tokenizer = MagicMock()
        registered_config: dict[str, Any] = {}

        def capture_config(path: Path) -> None:
            registered_config.update(json.loads(path.read_text(encoding="utf-8")))

        with (
            patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as snapshot_download,
            patch("open_clip.add_model_config", side_effect=capture_config) as add_model_config,
            patch(
                "open_clip.create_model_and_transforms",
                return_value=(model, MagicMock(), val_preprocess),
            ) as create_model,
            patch("open_clip.get_tokenizer", return_value=tokenizer) as get_tokenizer,
        ):
            adapter = SiglipAdapter(
                "Marqo/marqo-fashionSigLIP",
                backend="open_clip",
                open_clip_model_id="hf-hub:Marqo/marqo-fashionSigLIP",
                dense_dim=768,
                revision=revision,
            )
            adapter.load("cpu")

        snapshot_download.assert_called_once()
        assert snapshot_download.call_args.kwargs["repo_id"] == "Marqo/marqo-fashionSigLIP"
        assert snapshot_download.call_args.kwargs["revision"] == revision
        assert "open_clip_model.safetensors" in snapshot_download.call_args.kwargs["allow_patterns"]
        config_name = f"sie-Marqo--marqo-fashionSigLIP-{revision}"
        add_model_config.assert_called_once()
        create_model.assert_called_once_with(
            config_name,
            pretrained=str(weights_path),
            image_mean=(0.1, 0.2, 0.3),
            image_std=(0.4, 0.5, 0.6),
            image_interpolation="bicubic",
            image_resize_mode="squash",
        )
        get_tokenizer.assert_called_once_with(config_name)
        assert registered_config["text_cfg"]["hf_tokenizer_name"] == str(tmp_path)
        model.to.assert_called_once_with(device="cpu", dtype=torch.float32)
        model.eval.assert_called_once_with()
        assert adapter._open_clip_preprocess is val_preprocess
        assert adapter._open_clip_tokenizer is tokenizer
        assert adapter.dims.dense == 768

    @pytest.mark.parametrize(
        ("filename", "revision"),
        [
            ("Marqo__marqo-fashionSigLIP.yaml", "c56244cc94f92419e8369fa71efdaf403b124ce8"),
            ("Marqo__marqo-ecommerce-embeddings-B.yaml", "6854090aa39b2a9d70f390d83ae9e70cf9d9004e"),
        ],
    )
    def test_open_clip_catalog_models_pin_reviewed_snapshots(self, filename: str, revision: str) -> None:
        config_path = Path(__file__).resolve().parents[2] / "models" / filename
        data = yaml.safe_load(config_path.read_text())
        assert data["hf_revision"] == revision

    @patch("transformers.SiglipModel.from_pretrained")
    @patch("transformers.SiglipProcessor.from_pretrained")
    def test_encode_uses_loaded_text_tower_capacity(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
    ) -> None:
        processor = _FakeVisionProcessor()
        processor.tokenizer.model_max_length = 16
        model = MagicMock()
        model.config.vision_config.hidden_size = 1152
        model.config.text_config.max_position_embeddings = 16
        model.get_text_features.return_value = torch.zeros((1, 1152))
        mock_processor_from_pretrained.return_value = processor
        mock_model_from_pretrained.return_value = model
        adapter = SiglipAdapter("google/siglip-so400m-patch14-224", max_seq_length=64)

        adapter.load("cpu")
        adapter._encode_texts(["a caption that must use the loaded context"])

        assert processor.text_call_kwargs == [
            {
                "padding": "max_length",
                "truncation": True,
                "max_length": 16,
            }
        ]

    def test_encode_text_tokenizer_thread_safe(self) -> None:
        """Regression (MP.4 34026b6cd): concurrent text encodes must not race
        on the shared HF fast tokenizer (``RuntimeError: Already borrowed``).

        Drives the transformers-backend ``_encode_text`` from several threads
        through a tokenizer that raises on concurrent entry; the adapter's
        ``_tokenizer_lock`` must serialise the call so no overlap and no error
        occurs. Not a strict serialisability proof — a threaded smoke test.
        """
        adapter = SiglipAdapter("google/siglip-so400m-patch14-384", normalize=False)
        adapter._model = MagicMock()
        # get_text_features must yield a real tensor (the adapter unwraps the
        # feature tensor from the model output before returning).
        adapter._model.get_text_features.return_value = torch.zeros((1, 8))
        adapter._device = "cpu"
        tokenizer = _RaceDetectingTokenizer()
        adapter._processor = tokenizer

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

    def test_siglip_224_bundled_context_matches_text_tower_capacity(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "models" / "google__siglip-so400m-patch14-224.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert data["max_sequence_length"] == 16

    @pytest.mark.parametrize(("configured", "expected"), [(None, 64), (32, 32)])
    def test_encode_text_passes_explicit_max_length(self, configured: int | None, expected: int) -> None:
        adapter = SiglipAdapter(
            "google/siglip2-base-patch16-224",
            normalize=False,
            max_seq_length=configured,
        )
        model = _FakeVisionModel(dim=8)
        processor = _FakeVisionProcessor()
        adapter._model = model
        adapter._processor = processor
        adapter._device = "cpu"

        adapter._encode_texts(["short caption", "a caption with a different length"])

        assert processor.text_call_kwargs == [
            {
                "padding": "max_length",
                "truncation": True,
                "max_length": expected,
            }
        ]

    def test_encode_text_counts_from_primary_attention_mask_when_pad_equals_eos(self) -> None:
        adapter = SiglipAdapter("google/siglip2-base-patch16-224", normalize=False)
        model = MagicMock()
        model.get_text_features.return_value = torch.zeros((2, 8))
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 1
        tokenizer.side_effect = AssertionError("padding-free recount must not run")
        processor = MagicMock()
        processor.tokenizer = tokenizer
        processor.return_value = {
            "input_ids": torch.tensor(
                [
                    [12, 13, 1, 1],
                    [42, 1, 1, 1],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 0],
                    [1, 1, 0, 0],
                ]
            ),
        }
        adapter._model = model
        adapter._processor = processor
        adapter._device = "cpu"

        _embeddings, counts = adapter._encode_texts(["first", "second"])

        assert counts == [3, 2]
        tokenizer.assert_not_called()

    @pytest.mark.parametrize(
        "attention_mask",
        [
            torch.tensor([[1, 1, 1], [1, 1, 0]]),
            torch.tensor([[1, 1, 2, 0], [1, 1, 0, 0]]),
        ],
        ids=["wrong-shape", "non-binary"],
    )
    def test_encode_text_recounts_when_primary_attention_mask_is_malformed(self, attention_mask: torch.Tensor) -> None:
        adapter = SiglipAdapter("google/siglip2-base-patch16-224", normalize=False)
        model = MagicMock()
        model.get_text_features.return_value = torch.zeros((2, 8))
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 1
        tokenizer.return_value = {"input_ids": [[12, 13, 1], [42, 1]]}
        processor = MagicMock()
        processor.tokenizer = tokenizer
        processor.return_value = {
            "input_ids": torch.tensor(
                [
                    [12, 13, 1, 1],
                    [42, 1, 1, 1],
                ]
            ),
            "attention_mask": attention_mask,
        }
        adapter._model = model
        adapter._processor = processor
        adapter._device = "cpu"

        _embeddings, counts = adapter._encode_texts(["first", "second"])

        assert counts == [3, 2]
        tokenizer.assert_called_once_with(
            ["first", "second"],
            truncation=True,
            max_length=64,
        )

    def test_transformers_backend_reuses_positionally_aligned_prepared_images(self) -> None:
        adapter = SiglipAdapter("google/siglip2-base-patch16-224", normalize=False)
        model = MagicMock()
        model.get_image_features.side_effect = lambda pixel_values: pixel_values[:, :, 0, 0]
        adapter._model = model
        adapter._processor = MagicMock(side_effect=AssertionError("prepared fast path expected"))
        adapter._device = "cpu"
        adapter._dense_dim = 3
        prepared = [
            PreparedItem(
                payload=ImagePayload(pixel_values=torch.full((3, 1, 1), value), original_size=(1, 1)),
                cost=1,
                original_index=0,
            )
            for value in (1.0, 2.0)
        ]

        output = adapter.encode(
            [_img_item("first", [(1, 1, 1)]), _img_item("second", [(2, 2, 2)])],
            ["dense"],
            prepared_items=prepared,
        )

        assert output.dense is not None
        np.testing.assert_array_equal(output.dense, np.array([[1.0] * 3, [2.0] * 3]))

    @pytest.mark.parametrize("invalid", [0, -1])
    def test_rejects_non_positive_max_length(self, invalid: int) -> None:
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            SiglipAdapter("google/siglip2-base-patch16-224", max_seq_length=invalid)

    @patch("sie_server.adapters.siglip.adapter.torch")
    def test_unload(self, mock_torch: MagicMock, adapter: SiglipAdapter) -> None:
        """Unload clears the model."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._dense_dim = 1152

        adapter.unload()

        assert adapter._model is None
        assert adapter._processor is None
        assert adapter._dense_dim is None
        assert adapter.dims.dense is None

    def test_unload_shuts_down_preprocess_pool(self, adapter: SiglipAdapter) -> None:
        """Unload shuts down and clears the lazily-created preprocessing pool."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        pool = adapter._get_preprocess_pool()
        assert adapter._preprocess_pool is pool

        with patch.object(pool, "shutdown", wraps=pool.shutdown) as shutdown:
            adapter.unload()

        assert adapter._preprocess_pool is None
        assert pool._shutdown
        shutdown.assert_called_once_with(wait=True)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_encode_batched_matches_serial(self, normalize: bool) -> None:
        """EQUIVALENCE GATE: one stacked forward per modality must reproduce the
        per-item serial semantics exactly — output order, per-image normalize,
        multi-image mean-pool, and mixed image/text batches (transformers backend).
        """
        from torch.nn import functional

        dim = 16
        adapter = SiglipAdapter("google/siglip-so400m-patch14-384", normalize=normalize)
        model = _FakeVisionModel(dim)
        proc = _FakeVisionProcessor()
        adapter._model = model
        adapter._processor = proc
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

    def test_fused_requests_use_positionally_aligned_prepared_images(self) -> None:
        """Duplicate request-local indices must not cross-wire fused images."""
        adapter = SiglipAdapter(
            "Marqo/marqo-ecommerce-embeddings-B",
            backend="open_clip",
            open_clip_model_id="local-model",
            dense_dim=3,
            normalize=False,
        )
        model = MagicMock()
        model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        model.encode_image.side_effect = lambda pixel_values, normalize: pixel_values[:, :, 0, 0]
        adapter._model = model
        adapter._open_clip_preprocess = MagicMock(side_effect=AssertionError("prepared fast path expected"))
        adapter._device = "cpu"
        adapter._dense_dim = 3

        # Two two-item requests fused into one inference call. Each request's
        # preprocessor numbers its items 0, 1, so original_index repeats.
        prepared = [
            PreparedItem(
                payload=ImagePayload(pixel_values=torch.full((3, 1, 1), value), original_size=(1, 1)),
                cost=1,
                original_index=request_index,
            )
            for value, request_index in [(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 1)]
        ]
        items = [_img_item(str(index), [(index, index, index)]) for index in range(4)]

        output = adapter.encode(items, ["dense"], prepared_items=prepared)

        assert output.dense is not None
        np.testing.assert_array_equal(
            output.dense,
            np.array([[1.0] * 3, [2.0] * 3, [3.0] * 3, [4.0] * 3]),
        )
