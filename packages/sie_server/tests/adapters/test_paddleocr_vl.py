from __future__ import annotations

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

    def test_load_passes_revision_and_trust_remote_code(self) -> None:
        """load() threads revision + trust_remote_code to both from_pretrained calls."""
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
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.AutoModelForCausalLM") as mock_am_cls,
        ):
            mock_ap_cls.from_pretrained.return_value = mock_processor
            mock_am_cls.from_pretrained.return_value = mock_model

            adapter.load("cpu")

            ap_kwargs = mock_ap_cls.from_pretrained.call_args.kwargs
            am_kwargs = mock_am_cls.from_pretrained.call_args.kwargs
            assert ap_kwargs["revision"] == "abc123"
            assert ap_kwargs["trust_remote_code"] is True
            assert am_kwargs["revision"] == "abc123"
            assert am_kwargs["trust_remote_code"] is True

    def test_load_without_revision_omits_kwarg(self) -> None:
        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        adapter = PaddleOCRVLAdapter(
            "PaddlePaddle/PaddleOCR-VL-1.5",
            revision=None,
            trust_remote_code=True,
        )

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.AutoModelForCausalLM") as mock_am_cls,
        ):
            mock_ap_cls.from_pretrained.return_value = MagicMock()
            mock_am_cls.from_pretrained.return_value = mock_model

            adapter.load("cpu")

            assert "revision" not in mock_ap_cls.from_pretrained.call_args.kwargs
            assert "revision" not in mock_am_cls.from_pretrained.call_args.kwargs

    def test_preprocessor_built_after_load(self) -> None:
        from sie_server.adapters.paddleocr_vl import PaddleOCRVLAdapter

        adapter = PaddleOCRVLAdapter("PaddlePaddle/PaddleOCR-VL-1.5")

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.AutoModelForCausalLM") as mock_am_cls,
        ):
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

    def test_yaml_config_loads(self) -> None:
        """The shipped model YAML parses and resolves an adapter path."""
        import yaml
        from sie_server.config.model import ModelConfig

        path = "packages/sie_server/models/PaddlePaddle__PaddleOCR-VL-1.5.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        config = ModelConfig(**data)
        assert config.sie_id == "PaddlePaddle/PaddleOCR-VL-1.5"
        assert config.hf_id == "PaddlePaddle/PaddleOCR-VL-1.5"
        assert config.hf_revision == "6819afc8509ac9afa50e91b34627a7cf8f7900bb"
        assert config.inputs.image is True
        resolved = config.resolve_profile("default")
        assert resolved.adapter_path == "sie_server.adapters.paddleocr_vl:PaddleOCRVLAdapter"
        assert resolved.compute_precision == "bfloat16"
