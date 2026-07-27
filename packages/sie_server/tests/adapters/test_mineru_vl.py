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
_ = transformers.Qwen2VLForConditionalGeneration

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "opendatalab__MinerU2.5-Pro-2604-1.2B.yaml"


class TestMinerUVLAdapter:
    """Tests for MinerUVLAdapter with mocked model + processor."""

    @pytest.fixture
    def adapter(self) -> MinerUVLAdapter:
        from sie_server.adapters.mineru_vl import MinerUVLAdapter

        return MinerUVLAdapter(
            "opendatalab/MinerU2.5-Pro-2604-1.2B",
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: MinerUVLAdapter) -> None:
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_dims_empty(self, adapter: MinerUVLAdapter) -> None:
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises(self, adapter: MinerUVLAdapter) -> None:
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode([Item(text="hello")], output_types=["dense"])

    def test_extract_before_load_raises(self, adapter: MinerUVLAdapter) -> None:
        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    def test_build_messages_default(self, adapter: MinerUVLAdapter) -> None:
        messages = adapter._build_messages(task="[default]", instruction=None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0] == {"type": "image"}
        assert content[1] == {"type": "text", "text": "\nText Recognition:"}

    def test_build_messages_each_task(self, adapter: MinerUVLAdapter) -> None:
        expected = {
            "[default]": "\nText Recognition:",
            "table": "\nTable Recognition:",
            "equation": "\nFormula Recognition:",
            "image": "\nImage Analysis:",
            "chart": "\nImage Analysis:",
            "[layout]": "\nLayout Detection:",
        }
        for task, prompt in expected.items():
            messages = adapter._build_messages(task=task, instruction=None)
            assert messages[0]["content"][1]["text"] == prompt, f"task={task}"

    def test_build_messages_instruction_overrides_task(self, adapter: MinerUVLAdapter) -> None:
        messages = adapter._build_messages(task="[default]", instruction="Read all text in Tibetan.")
        assert messages[0]["content"][1]["text"] == "Read all text in Tibetan."

    def test_default_task_validated(self) -> None:
        from sie_server.adapters.mineru_vl import MinerUVLAdapter

        with pytest.raises(ValueError, match="default_task"):
            MinerUVLAdapter("opendatalab/MinerU2.5-Pro-2604-1.2B", default_task="nonsense")

    def test_fp16_on_cuda_rejected(self, adapter: MinerUVLAdapter) -> None:
        adapter._compute_precision = "float16"
        with pytest.raises(ValueError, match="float16"):
            adapter._resolve_dtype_for("cuda:0")

    def test_fp16_on_cpu_falls_back_to_fp32(self, adapter: MinerUVLAdapter) -> None:
        import torch

        adapter._compute_precision = "float16"
        # CPU path ignores compute_precision and uses fp32 — no crash.
        assert adapter._resolve_dtype_for("cpu") == torch.float32

    def test_convert_output_label_mapping(self, adapter: MinerUVLAdapter) -> None:
        expected = {
            "[default]": "mineru_text",
            "table": "mineru_table",
            "equation": "mineru_equation",
            "image": "mineru_image",
            "chart": "mineru_chart",
            "[layout]": "mineru_layout",
        }
        for task, label in expected.items():
            entities = adapter._convert_output("hello", task=task)
            assert len(entities) == 1
            assert entities[0]["text"] == "hello"
            assert entities[0]["label"] == label, f"task={task}"
            assert entities[0]["score"] == 1.0

    def test_convert_output_strips_whitespace(self, adapter: MinerUVLAdapter) -> None:
        entities = adapter._convert_output("  \n  body  \n  ", task="[default]")
        assert entities[0]["text"] == "body"

    def test_extract_invalid_task_rejected(self, adapter: MinerUVLAdapter) -> None:
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(ValueError, match="must be one of"):
            adapter.extract(items, options={"task": "bogus"})

    def test_load_passes_revision(self) -> None:
        """load() threads revision into both from_pretrained calls and never sets trust_remote_code."""
        from sie_server.adapters.mineru_vl import MinerUVLAdapter

        adapter = MinerUVLAdapter(
            "opendatalab/MinerU2.5-Pro-2604-1.2B",
            revision="abc123",
        )

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.Qwen2VLForConditionalGeneration") as mock_qm_cls,
        ):
            mock_ap_cls.from_pretrained.return_value = mock_processor
            mock_qm_cls.from_pretrained.return_value = mock_model

            adapter.load("cpu")

            ap_kwargs = mock_ap_cls.from_pretrained.call_args.kwargs
            qm_kwargs = mock_qm_cls.from_pretrained.call_args.kwargs
            assert ap_kwargs["revision"] == "abc123"
            assert ap_kwargs["use_fast"] is True
            assert qm_kwargs["revision"] == "abc123"
            assert "trust_remote_code" not in ap_kwargs
            assert "trust_remote_code" not in qm_kwargs

    def test_load_without_revision_omits_kwarg(self) -> None:
        from sie_server.adapters.mineru_vl import MinerUVLAdapter

        adapter = MinerUVLAdapter(
            "opendatalab/MinerU2.5-Pro-2604-1.2B",
            revision=None,
        )

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.Qwen2VLForConditionalGeneration") as mock_qm_cls,
        ):
            mock_ap_cls.from_pretrained.return_value = MagicMock()
            mock_qm_cls.from_pretrained.return_value = mock_model

            adapter.load("cpu")

            assert "revision" not in mock_ap_cls.from_pretrained.call_args.kwargs
            assert "revision" not in mock_qm_cls.from_pretrained.call_args.kwargs

    def test_preprocessor_built_after_load(self) -> None:
        from sie_server.adapters.mineru_vl import MinerUVLAdapter

        adapter = MinerUVLAdapter("opendatalab/MinerU2.5-Pro-2604-1.2B")

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoProcessor") as mock_ap_cls,
            patch("transformers.Qwen2VLForConditionalGeneration") as mock_qm_cls,
        ):
            mock_ap_cls.from_pretrained.return_value = MagicMock()
            mock_qm_cls.from_pretrained.return_value = mock_model

            assert adapter.get_preprocessor() is None
            adapter.load("cpu")

            prep = adapter.get_preprocessor()
            assert prep is not None
            assert prep.modality == "image"

    def test_extract_single_passes_use_cache_true_and_mineru_kwargs(self) -> None:
        """``_extract_single`` must pass ``use_cache=True`` plus MinerU's HF recipe.

        MinerU's transformers backend wires ``repetition_penalty=1.0`` and
        ``do_sample=False`` into HF generate; ``use_cache=True`` keeps the
        KV-cache enabled for O(N) decode. The ``no_repeat_ngram_size=100``
        guard is applied for greedy decoding via the fast incremental
        ``logits_processor`` (not the slow built-in kwarg).
        """
        import io

        import torch
        from PIL import Image as PILImage
        from sie_server.adapters.mineru_vl import (
            MinerUVLAdapter,
            _IncrementalNoRepeatNGramLogitsProcessor,
        )

        adapter = MinerUVLAdapter("opendatalab/MinerU2.5-Pro-2604-1.2B")
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
            task="[default]",
            instruction=None,
            max_new_tokens=8,
            num_beams=1,
        )

        kwargs = mock_model.generate.call_args.kwargs
        assert kwargs["use_cache"] is True
        assert kwargs["do_sample"] is False
        assert kwargs["repetition_penalty"] == 1.0
        # Greedy (num_beams=1): ngram guard rides the fast incremental processor,
        # NOT the slow built-in kwarg.
        assert "no_repeat_ngram_size" not in kwargs
        procs = kwargs["logits_processor"]
        assert len(procs) == 1
        assert isinstance(procs[0], _IncrementalNoRepeatNGramLogitsProcessor)
        assert procs[0]._n == 100

    def test_extract_single_beam_search_uses_builtin_ngram_kwarg(self) -> None:
        """Beam search (num_beams>1) keeps HF's stateless ngram kwarg.

        The incremental processor caches per-sequence n-gram state, which beam
        reordering would invalidate, so the adapter must fall back to the
        built-in ``no_repeat_ngram_size`` kwarg and use no custom processor.
        """
        import io

        import torch
        from PIL import Image as PILImage
        from sie_server.adapters.mineru_vl import MinerUVLAdapter

        adapter = MinerUVLAdapter("opendatalab/MinerU2.5-Pro-2604-1.2B")
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

        adapter._extract_single(item, task="[default]", instruction=None, max_new_tokens=8, num_beams=4)

        kwargs = mock_model.generate.call_args.kwargs
        assert kwargs["no_repeat_ngram_size"] == 100
        assert "logits_processor" not in kwargs

    def test_yaml_config_loads(self) -> None:
        """The shipped model YAML parses and resolves an adapter path."""
        import yaml
        from sie_server.config.model import ModelConfig

        with _MODEL_PATH.open() as f:
            data = yaml.safe_load(f)
        config = ModelConfig(**data)
        assert config.sie_id == "opendatalab/MinerU2.5-Pro-2604-1.2B"
        assert config.hf_id == "opendatalab/MinerU2.5-Pro-2604-1.2B"
        assert config.hf_revision == "d3f5e08d073c21466bbabe21c71bb1e9c2e595da"
        assert config.inputs.image is True
        resolved = config.resolve_profile("default")
        assert resolved.adapter_path == "sie_server.adapters.mineru_vl:MinerUVLAdapter"
        assert resolved.compute_precision == "bfloat16"


class TestIncrementalNoRepeatNGram:
    """The fast incremental processor must be bit-identical to HF's stock one.

    Each test drives a real greedy loop, calling both processors with the same
    growing ``input_ids`` every step and asserting identical ban masks and
    identical finite scores — so an identical next token, keeping the sequences
    in lockstep.
    """

    @staticmethod
    def _assert_same(fast_scores, hf_scores, step: int) -> None:
        import torch

        fast_banned = torch.isneginf(fast_scores)
        hf_banned = torch.isneginf(hf_scores)
        assert torch.equal(fast_banned, hf_banned), f"ban-mask mismatch at step {step}"
        finite = ~hf_banned
        assert torch.equal(fast_scores[finite], hf_scores[finite]), f"finite-score mismatch at step {step}"

    @pytest.mark.parametrize("ngram_size", [3, 4, 8, 100])
    def test_matches_hf_random_greedy_loop(self, ngram_size: int) -> None:
        import torch
        from sie_server.adapters.mineru_vl import _IncrementalNoRepeatNGramLogitsProcessor
        from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor

        torch.manual_seed(0)
        vocab = 40  # small vocab + repeats so small-n guards actually fire
        input_ids = torch.randint(0, vocab, (1, ngram_size + 5))
        fast = _IncrementalNoRepeatNGramLogitsProcessor(ngram_size)
        hf = NoRepeatNGramLogitsProcessor(ngram_size)
        bans = 0
        for step in range(250):
            logits = torch.randn(1, vocab)
            fast_scores = fast(input_ids, logits.clone())
            hf_scores = hf(input_ids, logits.clone())
            self._assert_same(fast_scores, hf_scores, step)
            bans += int(torch.isneginf(hf_scores).sum())
            nxt = torch.argmax(fast_scores, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, nxt], dim=-1)
        # Small n reliably repeats in random text; larger-n ban firing is
        # covered deterministically by test_matches_hf_repetitive_prompt_forces_bans.
        if ngram_size <= 4:
            assert bans > 0, "test never exercised a ban; not a meaningful comparison"

    @pytest.mark.parametrize("ngram_size", [3, 100])
    def test_matches_hf_repetitive_prompt_forces_bans(self, ngram_size: int) -> None:
        """A constant-token prompt forces the guard to fire even at n=100."""
        import torch
        from sie_server.adapters.mineru_vl import _IncrementalNoRepeatNGramLogitsProcessor
        from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor

        torch.manual_seed(1)
        vocab = 20
        input_ids = torch.full((1, ngram_size + 30), 5, dtype=torch.long)
        fast = _IncrementalNoRepeatNGramLogitsProcessor(ngram_size)
        hf = NoRepeatNGramLogitsProcessor(ngram_size)
        fired = False
        for step in range(25):
            logits = torch.randn(1, vocab)
            fast_scores = fast(input_ids, logits.clone())
            hf_scores = hf(input_ids, logits.clone())
            self._assert_same(fast_scores, hf_scores, step)
            fired = fired or bool(torch.isneginf(hf_scores).any())
            nxt = torch.argmax(fast_scores, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, nxt], dim=-1)
        assert fired, "repetitive prompt should have triggered the n-gram guard"
