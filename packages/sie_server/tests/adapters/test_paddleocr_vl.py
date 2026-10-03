from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import transformers
from sie_server.types.inputs import ImageInput, Item

# Force eager resolution of lazy-imported transformers classes so
# unittest.mock.patch("transformers.AutoProcessor", ...) can intercept them.
# Without this, `from transformers import AutoProcessor` inside the adapter's
# load() bypasses the patched attribute and hits the real HuggingFace Hub.
_ = transformers.AutoProcessor
_ = transformers.AutoModelForCausalLM
_ = transformers.masking_utils.create_causal_mask

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "PaddlePaddle__PaddleOCR-VL-1.5.yaml"


class TestPaddleOCRVLAdapter:
    """Tests for PaddleOCRVLAdapter with mocked model + processor."""

    @pytest.fixture
    def adapter(self) -> PaddleOCRVLAdapter:
        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        return PaddleOCRVLAdapter(
            "PaddlePaddle/PaddleOCR-VL-1.5",
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: PaddleOCRVLAdapter) -> None:
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_dims_empty(self, adapter: PaddleOCRVLAdapter) -> None:
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises(self, adapter: PaddleOCRVLAdapter) -> None:
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode([Item(text="hello")], output_types=["dense"])

    def test_extract_before_load_raises(self, adapter: PaddleOCRVLAdapter) -> None:
        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    def test_build_messages_default(self, adapter: PaddleOCRVLAdapter) -> None:
        messages = adapter._build_messages(task="ocr", instruction=None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0] == {"type": "image"}
        assert content[1] == {"type": "text", "text": "OCR:"}

    def test_build_messages_each_task(self, adapter: PaddleOCRVLAdapter) -> None:
        expected = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        }
        for task, prompt in expected.items():
            messages = adapter._build_messages(task=task, instruction=None)
            assert messages[0]["content"][1]["text"] == prompt, f"task={task}"

    def test_build_messages_instruction_overrides_task(self, adapter: PaddleOCRVLAdapter) -> None:
        messages = adapter._build_messages(task="ocr", instruction="Read all text in Tibetan.")
        assert messages[0]["content"][1]["text"] == "Read all text in Tibetan."

    def test_default_task_validated(self) -> None:
        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        with pytest.raises(ValueError, match="default_task"):
            PaddleOCRVLAdapter("PaddlePaddle/PaddleOCR-VL-1.5", default_task="nonsense")

    def test_fp16_on_cuda_rejected(self, adapter: PaddleOCRVLAdapter) -> None:
        adapter._compute_precision = "float16"
        with pytest.raises(ValueError, match="float16"):
            adapter._resolve_dtype_for("cuda:0")

    def test_fp16_on_cpu_falls_back_to_fp32(self, adapter: PaddleOCRVLAdapter) -> None:
        import torch

        adapter._compute_precision = "float16"
        # CPU path ignores compute_precision and uses fp32 — no crash.
        assert adapter._resolve_dtype_for("cpu") == torch.float32

    def test_convert_output_markdown_tasks(self, adapter: PaddleOCRVLAdapter) -> None:
        for task in ("ocr", "table", "formula", "chart"):
            entities = adapter._convert_output("# Title\n\nbody", task=task)
            assert len(entities) == 1
            assert entities[0]["text"] == "# Title\n\nbody"
            assert entities[0]["label"] == "markdown"
            assert entities[0]["score"] == 1.0

    def test_convert_output_spotting_and_seal(self, adapter: PaddleOCRVLAdapter) -> None:
        entities = adapter._convert_output('[{"bbox":[1,2,3,4],"text":"HELLO"}]', task="spotting")
        assert entities[0]["label"] == "spotting"
        entities = adapter._convert_output('{"seal": "A"}', task="seal")
        assert entities[0]["label"] == "seal"

    def test_convert_output_strips_whitespace(self, adapter: PaddleOCRVLAdapter) -> None:
        entities = adapter._convert_output("  \n  body  \n  ", task="ocr")
        assert entities[0]["text"] == "body"

    def test_extract_invalid_task_rejected(self, adapter: PaddleOCRVLAdapter) -> None:
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(ValueError, match="must be one of"):
            adapter.extract(items, options={"task": "bogus"})

    def test_load_resolves_revision_snapshot_and_passes_trust_remote_code(self) -> None:
        """load() resolves the pinned revision to a local dir, then loads from it.

        The revision must reach ``snapshot_download`` (not from_pretrained): on
        transformers 4.57.x ``AutoProcessor.from_pretrained`` drops ``revision``
        during its offline config probe, so the adapter hands it a pre-resolved
        local directory instead. ``trust_remote_code`` still threads to both
        from_pretrained calls.
        """
        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        adapter = PaddleOCRVLAdapter(
            "PaddlePaddle/PaddleOCR-VL-1.5",
            revision="abc123",
            trust_remote_code=True,
        )

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("sie_server.adapters.paddleocr_vl.snapshot_download") as mock_snap,
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.AutoModelForCausalLM") as mock_am_cls,
        ):
            mock_snap.return_value = "/models/paddleocr-vl-snapshot"
            mock_ap_cls.from_pretrained.return_value = mock_processor
            mock_am_cls.from_pretrained.return_value = mock_model

            adapter.load("cpu")

            assert mock_snap.call_args.kwargs["revision"] == "abc123"
            ap_args, ap_kwargs = mock_ap_cls.from_pretrained.call_args
            am_args, am_kwargs = mock_am_cls.from_pretrained.call_args
            assert ap_args[0] == "/models/paddleocr-vl-snapshot"
            assert am_args[0] == "/models/paddleocr-vl-snapshot"
            assert ap_kwargs["trust_remote_code"] is True
            assert am_kwargs["trust_remote_code"] is True
            # revision must NOT reach from_pretrained (it is dropped offline).
            assert "revision" not in ap_kwargs
            assert "revision" not in am_kwargs

    def test_load_without_revision_omits_snapshot_kwarg(self) -> None:
        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        adapter = PaddleOCRVLAdapter(
            "PaddlePaddle/PaddleOCR-VL-1.5",
            revision=None,
            trust_remote_code=True,
        )

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("sie_server.adapters.paddleocr_vl.snapshot_download") as mock_snap,
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.AutoModelForCausalLM") as mock_am_cls,
        ):
            mock_snap.return_value = "/models/paddleocr-vl-snapshot"
            mock_ap_cls.from_pretrained.return_value = MagicMock()
            mock_am_cls.from_pretrained.return_value = mock_model

            adapter.load("cpu")

            assert "revision" not in mock_snap.call_args.kwargs

    def test_load_uses_local_dir_without_download(self, tmp_path: object) -> None:
        """A model_name_or_path that is already a local dir skips snapshot_download."""
        from pathlib import Path

        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        local_dir = Path(str(tmp_path)) / "paddleocr-vl"
        local_dir.mkdir()
        adapter = PaddleOCRVLAdapter(str(local_dir), revision="abc123")

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("sie_server.adapters.paddleocr_vl.snapshot_download") as mock_snap,
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.AutoModelForCausalLM") as mock_am_cls,
        ):
            mock_ap_cls.from_pretrained.return_value = MagicMock()
            mock_am_cls.from_pretrained.return_value = mock_model

            adapter.load("cpu")

            mock_snap.assert_not_called()
            assert mock_ap_cls.from_pretrained.call_args.args[0] == str(local_dir)

    def test_preprocessor_built_after_load(self) -> None:
        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        adapter = PaddleOCRVLAdapter("PaddlePaddle/PaddleOCR-VL-1.5")

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("sie_server.adapters.paddleocr_vl.snapshot_download") as mock_snap,
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.AutoModelForCausalLM") as mock_am_cls,
        ):
            mock_snap.return_value = "/models/paddleocr-vl-snapshot"
            mock_ap_cls.from_pretrained.return_value = MagicMock()
            mock_am_cls.from_pretrained.return_value = mock_model

            assert adapter.get_preprocessor() is None
            adapter.load("cpu")

            prep = adapter.get_preprocessor()
            assert prep is not None
            assert prep.modality == "image"

    def test_compat_shim_aliases_inputs_embeds(self) -> None:
        """The compat shim translates ``inputs_embeds`` to ``input_embeds``.

        PaddleOCR-VL's modeling code (at the pinned revision) calls
        ``create_causal_mask(inputs_embeds=...)`` but transformers 4.57.x
        exposes ``input_embeds`` (singular). Without the shim, generate()
        crashes with a TypeError.
        """
        from sie_server.adapters.paddleocr_vl import _apply_transformers_compat_shim
        from transformers import masking_utils

        original = masking_utils.create_causal_mask

        captured: dict[str, object] = {}

        def fake_original(*args: object, **kwargs: object) -> str:
            captured.update(kwargs)
            return "MASK"

        masking_utils.create_causal_mask = fake_original
        try:
            _apply_transformers_compat_shim()
            wrapped = masking_utils.create_causal_mask
            # Reach through the wrapper and verify it aliases inputs_embeds.
            result = wrapped(inputs_embeds="embeds", attention_mask="am")
            assert result == "MASK"
            assert captured.get("input_embeds") == "embeds"
            assert "inputs_embeds" not in captured
        finally:
            masking_utils.create_causal_mask = original

    def test_compat_shim_idempotent(self) -> None:
        """Calling the shim twice does not double-wrap."""
        from sie_server.adapters.paddleocr_vl import _apply_transformers_compat_shim
        from transformers import masking_utils

        original = masking_utils.create_causal_mask
        try:
            _apply_transformers_compat_shim()
            after_first = masking_utils.create_causal_mask
            _apply_transformers_compat_shim()
            after_second = masking_utils.create_causal_mask
            assert after_first is after_second
        finally:
            masking_utils.create_causal_mask = original

    def test_extract_single_passes_use_cache_true(self) -> None:
        """``_extract_single`` must pass ``use_cache=True`` to ``model.generate``.

        Upstream's generation_config.json ships ``use_cache=false``; without an
        explicit override KV-cache stays off and decode degrades to O(N^2).
        """
        import io

        import torch
        from PIL import Image as PILImage
        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        adapter = PaddleOCRVLAdapter("PaddlePaddle/PaddleOCR-VL-1.5")
        adapter._device = "cpu"

        mock_model = MagicMock()
        mock_model.dtype = torch.float32
        mock_model.generate.return_value = torch.zeros((1, 4), dtype=torch.long)
        adapter._model = mock_model

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = ""
        mock_processor.decode.return_value = ""
        mock_processor.return_value = {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "attention_mask": torch.zeros((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros((1, 1), dtype=torch.long),
            "image_grid_thw": torch.zeros((1, 1), dtype=torch.long),
        }
        adapter._processor = mock_processor

        buf = io.BytesIO()
        PILImage.new("RGB", (4, 4)).save(buf, format="JPEG")
        item = Item(images=[ImageInput(data=buf.getvalue(), format="jpeg")])

        adapter._extract_single(
            item,
            task="ocr",
            instruction=None,
            max_new_tokens=8,
            num_beams=1,
        )

        assert mock_model.generate.call_args.kwargs["use_cache"] is True

    def test_yaml_config_loads(self) -> None:
        """The shipped model YAML parses and resolves an adapter path."""
        import yaml
        from sie_server.config.model import ModelConfig

        with _MODEL_PATH.open() as f:
            data = yaml.safe_load(f)
        config = ModelConfig(**data)
        assert config.sie_id == "PaddlePaddle/PaddleOCR-VL-1.5"
        assert config.hf_id == "PaddlePaddle/PaddleOCR-VL-1.5"
        assert config.hf_revision == "6819afc8509ac9afa50e91b34627a7cf8f7900bb"
        assert config.inputs.image is True
        resolved = config.resolve_profile("default")
        assert resolved.adapter_path == "sie_server.adapters.sglang_vision_extract.adapter:SGLangVisionExtractAdapter"
        assert resolved.compute_precision == "bfloat16"
        fallback = config.resolve_profile("transformers")
        assert fallback.adapter_path == "sie_server.adapters.paddleocr_vl:PaddleOCRVLAdapter"
        assert fallback.compute_precision == "bfloat16"
